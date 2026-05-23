from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from isaac_bench.mapping.vertical_profile import VerticalProfileMap


@dataclass
class FloorCeilingEstimate:
    floor_z: float
    ceiling_z: float
    room_height_m: float
    inner_low_z: float
    inner_high_z: float
    confidence: float


@dataclass
class HeightProfileState:
    """Per-z-bin evidence for the v6.2 strict wall/door classifier."""

    z_edges_m: np.ndarray
    z_centers_m: np.ndarray
    free_count: np.ndarray
    endpoint_count: np.ndarray
    observed_count: np.ndarray
    unknown_count: np.ndarray
    free_supported: np.ndarray
    endpoint_supported: np.ndarray
    occupied_supported: np.ndarray
    unknown_supported: np.ndarray
    floor_ceiling: FloorCeilingEstimate
    debug: dict[str, object] = field(default_factory=dict)


@dataclass
class ColumnProfileClassification:
    """Column-level v6.2 wall/door/free/unknown classification."""

    wall_mask_raw: np.ndarray
    door_mask_raw: np.ndarray
    free_mask_raw: np.ndarray
    unknown_mask: np.ndarray
    wall_endpoint_ratio: np.ndarray
    best_door_lower_free_ratio: np.ndarray
    best_door_upper_occupied_ratio: np.ndarray
    best_door_transition_z: np.ndarray
    wall_mask: np.ndarray
    door_mask: np.ndarray
    wall_line_support_mask: np.ndarray
    room_boundary_mask: np.ndarray
    debug: dict[str, object] = field(default_factory=dict)


def build_height_profile_v6_2(
    *,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    vertical_profile: VerticalProfileMap | None = None,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    config: Mapping[str, object] | None = None,
    resolution_m: float = 0.05,
) -> HeightProfileState:
    """Build a dynamic z-bin profile from the available online evidence.

    The Isaac online mapper currently exposes a coarse ``VerticalProfileMap``
    instead of raw rays at this stage.  v6.2 still uses strict z-bin ratios: a
    bin copies evidence from the coarse band covering that height.  Terminal
    endpoint splats remain raw debug evidence only; they are not promoted into
    occupied bins unless a future caller supplies real per-height endpoint data.
    """

    _ = float(resolution_m)
    cfg = dict(config or {})
    shape = np.asarray(observed_free_mask, dtype=bool).shape
    floor_ceiling = estimate_floor_ceiling_v6_2(vertical_profile=vertical_profile, config=cfg)
    z_edges = _z_edges(floor_ceiling.inner_low_z, floor_ceiling.inner_high_z, _z_bin_m(cfg))
    z_centers = (z_edges[:-1] + z_edges[1:]) * 0.5
    z_count = int(z_centers.size)
    free_count = np.zeros((z_count, *shape), dtype=np.uint16)
    endpoint_count = np.zeros_like(free_count)
    observed_count = np.zeros_like(free_count)

    source = "synthetic_2d_masks"
    if vertical_profile is not None and tuple(vertical_profile.shape) == tuple(shape):
        source = "vertical_profile_band_projection"
        band_ranges = tuple((float(lo), float(hi)) for lo, hi in vertical_profile.band_ranges_m)
        for z_idx, z_center in enumerate(z_centers):
            band_idx = _band_index(float(z_center), band_ranges)
            if band_idx is None:
                continue
            free_count[z_idx] = np.asarray(vertical_profile.free_ray_count[band_idx], dtype=np.uint16)
            endpoint_count[z_idx] = np.asarray(vertical_profile.occupied_count[band_idx], dtype=np.uint16)
            observed_count[z_idx] = np.asarray(vertical_profile.observed_count[band_idx], dtype=np.uint16)
    else:
        free_2d = np.asarray(observed_free_mask, dtype=bool)
        obstacle_2d = np.asarray(obstacle_mask, dtype=bool) & ~free_2d
        observed_2d = (~np.asarray(unknown_mask, dtype=bool)) | free_2d | obstacle_2d
        free_count[:, free_2d] = 1
        endpoint_count[:, obstacle_2d] = 1
        observed_count[:, observed_2d] = 1

    terminal_height_cells = _add_terminal_height_endpoint_bins(
        endpoint_count=endpoint_count,
        observed_count=observed_count,
        z_centers=z_centers,
        roomseg_ray_evidence=roomseg_ray_evidence,
        shape=shape,
    )

    free_supported, endpoint_supported, unknown_supported, support_debug = _support_masks(
        free_count=free_count,
        endpoint_count=endpoint_count,
        config=cfg,
    )
    occupied_supported = endpoint_supported.copy()
    unknown_count = unknown_supported.astype(np.uint16)
    raw_endpoint_evidence = _raw_endpoint_evidence_2d(roomseg_ray_evidence, shape)
    debug = {
        "source": "height_profile_v6_2",
        "profile_source": source,
        "z_bin_count": int(z_count),
        "z_bin_m": float(_z_bin_m(cfg)),
        "floor_z": float(floor_ceiling.floor_z),
        "ceiling_z": float(floor_ceiling.ceiling_z),
        "inner_low_z": float(floor_ceiling.inner_low_z),
        "inner_high_z": float(floor_ceiling.inner_high_z),
        "floor_ceiling_confidence": float(floor_ceiling.confidence),
        "raw_endpoint_evidence_2d_cells": int(np.count_nonzero(raw_endpoint_evidence)),
        "terminal_height_profile_endpoint_cells": int(terminal_height_cells),
        "terminal_endpoint_promoted_to_wall_cells": 0,
        **support_debug,
    }
    return HeightProfileState(
        z_edges_m=z_edges.astype(np.float32),
        z_centers_m=z_centers.astype(np.float32),
        free_count=free_count,
        endpoint_count=endpoint_count,
        observed_count=observed_count,
        unknown_count=unknown_count,
        free_supported=free_supported.astype(bool),
        endpoint_supported=endpoint_supported.astype(bool),
        occupied_supported=occupied_supported.astype(bool),
        unknown_supported=unknown_supported.astype(bool),
        floor_ceiling=floor_ceiling,
        debug=debug,
    )


def estimate_floor_ceiling_v6_2(
    *,
    vertical_profile: VerticalProfileMap | None = None,
    config: Mapping[str, object] | None = None,
) -> FloorCeilingEstimate:
    cfg = dict(config or {})
    fc_cfg = dict(cfg.get("floor_ceiling", {}) or {})
    range_cfg = dict(cfg.get("vertical_range", {}) or {})
    floor_z = float(fc_cfg.get("fallback_floor_z", 0.0))
    ceiling_z = float(fc_cfg.get("fallback_ceiling_z", 2.8))
    confidence = 0.35

    inner_low_percent = float(range_cfg.get("inner_low_percent", 0.05))
    inner_high_percent = float(range_cfg.get("inner_high_percent", 0.95))
    if vertical_profile is not None and getattr(vertical_profile, "band_ranges_m", None):
        # The coarse online profile only covers measured bands up to its upper
        # band edge.  Keep the v6.2 denominator inside that covered evidence
        # range so a full-height wall in the profile can actually reach 95%.
        band_low = min(float(lo) for lo, _hi in vertical_profile.band_ranges_m)
        band_high = max(float(hi) for _lo, hi in vertical_profile.band_ranges_m)
        floor_z = min(floor_z, band_low - inner_low_percent * max(0.1, band_high - band_low))
        ceiling_from_band_high = floor_z + (band_high - floor_z) / max(1e-6, inner_high_percent)
        ceiling_z = min(ceiling_z, ceiling_from_band_high)
        confidence = 0.65

    min_height = float(fc_cfg.get("min_height_m", 2.0))
    max_height = float(fc_cfg.get("max_height_m", 4.0))
    room_height = float(ceiling_z - floor_z)
    if room_height < min_height or room_height > max_height:
        floor_z = float(fc_cfg.get("fallback_floor_z", 0.0))
        ceiling_z = float(fc_cfg.get("fallback_ceiling_z", 2.8))
        room_height = float(ceiling_z - floor_z)
        confidence = 0.25
    inner_low_z = floor_z + inner_low_percent * room_height
    inner_high_z = floor_z + inner_high_percent * room_height
    if inner_high_z <= inner_low_z:
        inner_low_z = floor_z
        inner_high_z = ceiling_z
    return FloorCeilingEstimate(
        floor_z=float(floor_z),
        ceiling_z=float(ceiling_z),
        room_height_m=float(room_height),
        inner_low_z=float(inner_low_z),
        inner_high_z=float(inner_high_z),
        confidence=float(confidence),
    )


def _support_masks(
    *,
    free_count: np.ndarray,
    endpoint_count: np.ndarray,
    config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    support_cfg = dict(config.get("bin_support", {}) or {})
    min_free_hits = int(support_cfg.get("min_free_hits_per_bin", 1))
    min_endpoint_hits = int(support_cfg.get("min_endpoint_hits_per_bin", 1))
    free_dominance = float(support_cfg.get("free_dominance_ratio", 0.70))
    endpoint_dominance = float(support_cfg.get("endpoint_dominance_ratio", 0.70))
    ambiguous_unknown = bool(support_cfg.get("ambiguous_bin_counts_as_unknown", True))
    free = np.asarray(free_count, dtype=np.float32)
    endpoint = np.asarray(endpoint_count, dtype=np.float32)
    total = free + endpoint
    safe_total = np.maximum(total, 1.0)
    free_conf = free / safe_total
    endpoint_conf = endpoint / safe_total
    free_supported = (free >= float(min_free_hits)) & (free_conf >= free_dominance)
    endpoint_supported = (endpoint >= float(min_endpoint_hits)) & (endpoint_conf >= endpoint_dominance)
    if ambiguous_unknown:
        ambiguous = ~(free_supported | endpoint_supported)
    else:
        ambiguous = total <= 0
    return (
        free_supported.astype(bool),
        endpoint_supported.astype(bool),
        ambiguous.astype(bool),
        {
            "free_supported_bins": int(np.count_nonzero(free_supported)),
            "endpoint_supported_bins": int(np.count_nonzero(endpoint_supported)),
            "unknown_or_ambiguous_bins": int(np.count_nonzero(ambiguous)),
            "min_free_hits_per_bin": int(min_free_hits),
            "min_endpoint_hits_per_bin": int(min_endpoint_hits),
            "free_dominance_ratio": float(free_dominance),
            "endpoint_dominance_ratio": float(endpoint_dominance),
        },
    )


def _z_bin_m(config: Mapping[str, object]) -> float:
    range_cfg = dict(config.get("vertical_range", {}) or {})
    return max(1e-3, float(range_cfg.get("z_bin_m", 0.05)))


def _z_edges(inner_low_z: float, inner_high_z: float, z_bin_m: float) -> np.ndarray:
    span = max(float(z_bin_m), float(inner_high_z) - float(inner_low_z))
    count = max(1, int(np.ceil(span / float(z_bin_m))))
    edges = float(inner_low_z) + np.arange(count + 1, dtype=np.float32) * float(z_bin_m)
    edges[-1] = float(inner_high_z)
    return edges


def _band_index(z_m: float, band_ranges: tuple[tuple[float, float], ...]) -> int | None:
    for idx, (lo, hi) in enumerate(band_ranges):
        if float(lo) <= float(z_m) < float(hi):
            return int(idx)
    if band_ranges and np.isclose(float(z_m), float(band_ranges[-1][1])):
        return int(len(band_ranges) - 1)
    return None


def _raw_endpoint_evidence_2d(evidence: Mapping[str, np.ndarray] | None, shape: tuple[int, int]) -> np.ndarray:
    raw = dict(evidence or {})
    out = np.zeros(shape, dtype=bool)
    for name in ("terminal_wall_count", "roomseg_terminal_wall_count", "terminal_wall_splat", "roomseg_terminal_wall_splat"):
        value = raw.get(name)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape == shape:
            out |= arr.astype(bool)
    return out


def _add_terminal_height_endpoint_bins(
    *,
    endpoint_count: np.ndarray,
    observed_count: np.ndarray,
    z_centers: np.ndarray,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None,
    shape: tuple[int, int],
) -> int:
    evidence = dict(roomseg_ray_evidence or {})
    count = _array_like(evidence, shape, "terminal_wall_count", "roomseg_terminal_wall_count", dtype=np.uint16)
    z_min = _array_like(evidence, shape, "terminal_wall_height_min", "roomseg_terminal_wall_height_min", dtype=np.float32, fill=np.inf)
    z_max = _array_like(evidence, shape, "terminal_wall_height_max", "roomseg_terminal_wall_height_max", dtype=np.float32, fill=-np.inf)
    valid = (np.asarray(count, dtype=np.uint16) > 0) & np.isfinite(z_min) & np.isfinite(z_max) & (z_max >= z_min)
    if not np.any(valid):
        return 0
    rows, cols = np.nonzero(valid)
    for r, c in zip(rows.tolist(), cols.tolist()):
        z_mask = (np.asarray(z_centers, dtype=np.float32) >= float(z_min[r, c])) & (np.asarray(z_centers, dtype=np.float32) <= float(z_max[r, c]))
        if not np.any(z_mask):
            nearest = int(np.argmin(np.abs(np.asarray(z_centers, dtype=np.float32) - float(z_min[r, c]))))
            z_mask[nearest] = True
        endpoint_count[z_mask, int(r), int(c)] = np.maximum(endpoint_count[z_mask, int(r), int(c)], count[int(r), int(c)])
        observed_count[z_mask, int(r), int(c)] = np.maximum(observed_count[z_mask, int(r), int(c)], count[int(r), int(c)])
    return int(rows.size)


def _array_like(
    evidence: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    *names: str,
    dtype: type,
    fill: float | int = 0,
) -> np.ndarray:
    for name in names:
        value = evidence.get(name)
        if value is None:
            continue
        arr = np.asarray(value, dtype=dtype)
        if arr.shape == shape:
            return arr
    return np.full(shape, fill, dtype=dtype)
