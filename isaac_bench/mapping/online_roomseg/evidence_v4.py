from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.roomseg_evidence_v3 import RoomSegEvidenceV3, build_roomseg_evidence_v3
from isaac_bench.mapping.vertical_profile import VerticalProfileMap

from .utils import conn, disk, label_components, relabel_compact


ONLINE_LINE_EXTEND_ROOMSEG_V4_BACKEND = "online_line_extend_roomseg_v4"


@dataclass
class RoomSegEvidenceV4:
    vertical_free_raw: np.ndarray
    vertical_occupied_raw: np.ndarray
    vertical_observed_raw: np.ndarray
    vertical_unknown_raw: np.ndarray
    ray_covered_count: np.ndarray
    terminal_wall_count: np.ndarray
    terminal_wall_structural_candidate: np.ndarray
    terminal_wall_structural_clean: np.ndarray
    roomseg_free_raw: np.ndarray
    roomseg_free_stable: np.ndarray
    roomseg_free_clean: np.ndarray
    roomseg_unknown_clean: np.ndarray
    structural_wall_candidate: np.ndarray
    structural_wall_clean: np.ndarray
    structural_wall_confidence: np.ndarray
    nav_reachable_free: np.ndarray
    free_noise_rejected: np.ndarray
    ray_fan_spur_rejected: np.ndarray
    isolated_free_rejected: np.ndarray
    debug: dict = field(default_factory=dict)
    v3_evidence: RoomSegEvidenceV3 | None = None

    @property
    def free_clean(self) -> np.ndarray:
        return self.roomseg_free_clean

    @property
    def unknown_clean(self) -> np.ndarray:
        return self.roomseg_unknown_clean

    @property
    def wall_candidate_clean(self) -> np.ndarray:
        return self.structural_wall_clean


def build_roomseg_evidence_v4(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    vertical_profile: VerticalProfileMap | None,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None,
    traversible: np.ndarray | None,
    agent_grid: tuple[int, int] | None,
    map_info: MapInfo | None,
    config: Mapping[str, object] | object | None,
) -> RoomSegEvidenceV4:
    root_cfg = _root_config(config)
    cfg = _section(root_cfg, "roomseg_evidence_v4")
    shape = np.asarray(observed_free_mask, dtype=bool).shape
    observed = ~np.asarray(unknown_mask, dtype=bool)
    vertical_profile = vertical_profile or _synthetic_vertical_profile_from_masks(
        observed_free_mask,
        obstacle_mask,
        observed,
    )
    ray_evidence = dict(roomseg_ray_evidence or {})
    resolution_m = float(
        getattr(map_info, "resolution_m", None)
        or root_cfg.get("resolution_m", 0.05)
        or 0.05
    )
    v3 = build_roomseg_evidence_v3(
        vertical_profile=vertical_profile,
        grid_free=np.asarray(observed_free_mask, dtype=bool),
        grid_occupied=np.asarray(obstacle_mask, dtype=bool),
        grid_observed=observed,
        roomseg_ray_covered_count=_ray(ray_evidence, "ray_covered_count", "roomseg_ray_covered_count"),
        roomseg_terminal_wall_count=_ray(ray_evidence, "terminal_wall_count", "roomseg_terminal_wall_count"),
        roomseg_terminal_wall_splat=_ray(ray_evidence, "terminal_wall_splat", "roomseg_terminal_wall_splat"),
        roomseg_terminal_wall_height_min=_ray(ray_evidence, "terminal_wall_height_min", "roomseg_terminal_wall_height_min"),
        roomseg_terminal_wall_height_max=_ray(ray_evidence, "terminal_wall_height_max", "roomseg_terminal_wall_height_max"),
        roomseg_terminal_wall_depth_min=_ray(ray_evidence, "terminal_wall_depth_min", "roomseg_terminal_wall_depth_min"),
        robot_rc=agent_grid,
        resolution_m=float(resolution_m),
        config=root_cfg,
    )

    free_raw = np.asarray(v3.roomseg_free_raw, dtype=bool)
    ray_free_count = np.asarray(v3.ray_free_count, dtype=np.uint32)
    observed_count = np.sum(np.asarray(v3.observed_count_by_band, dtype=np.uint32), axis=0, dtype=np.uint32)
    min_free_rays = max(1, int(cfg.get("min_free_rays_for_stable_free", 2)))
    min_observed = max(1, int(cfg.get("min_observed_rays_for_stable_cell", 2)))
    strong_stable = (ray_free_count >= int(min_free_rays)) & (observed_count >= int(min_observed))

    nav_reachable = _reachable_free(
        traversible=traversible,
        observed_free=observed_free_mask,
        obstacle=obstacle_mask,
        agent_grid=agent_grid,
        resolution_m=float(resolution_m),
        dilation_m=float(cfg.get("nav_reachable_dilation_m", 0.10)),
        shape=shape,
    )
    if bool(cfg.get("allow_single_ray_free_if_nav_reachable", True)):
        single_ray_reachable = (ray_free_count >= 1) & nav_reachable
    else:
        single_ray_reachable = np.zeros(shape, dtype=bool)
    roomseg_free_stable = free_raw & (strong_stable | single_ray_reachable)
    if bool(cfg.get("require_nav_reachable_for_roomseg_free", True)):
        strong_without_nav = strong_stable & (ray_free_count >= max(int(min_free_rays) + 1, 3))
        roomseg_free_stable &= nav_reachable | strong_without_nav

    structural_wall_clean = np.asarray(v3.structural_wall_clean, dtype=bool).copy()
    terminal_count = _optional_uint(_ray(ray_evidence, "terminal_wall_count", "roomseg_terminal_wall_count"), shape)
    terminal_view_count = _optional_uint(_ray(ray_evidence, "terminal_wall_view_count", "roomseg_terminal_wall_view_count"), shape)
    min_terminal_count = max(1, int(cfg.get("terminal_wall_min_count", 2)))
    min_terminal_views = max(1, int(cfg.get("terminal_wall_min_views", 1)))
    terminal_wall_candidate = np.asarray(v3.terminal_wall_candidate, dtype=bool) | (
        (terminal_count >= int(min_terminal_count))
        & ((terminal_view_count <= 0) | (terminal_view_count >= int(min_terminal_views)))
        & ~free_raw
    )
    terminal_wall_clean = terminal_wall_candidate & structural_wall_clean
    structural_wall_candidate = np.asarray(v3.structural_wall_candidate, dtype=bool) | terminal_wall_candidate
    structural_wall_clean |= terminal_wall_clean

    before_morph = roomseg_free_stable & ~structural_wall_clean
    roomseg_free_clean, morph_debug = _clean_stable_free(
        before_morph,
        nav_reachable=nav_reachable,
        resolution_m=float(resolution_m),
        cfg=cfg,
    )
    structural_wall_clean &= ~roomseg_free_clean
    structural_wall_candidate &= ~roomseg_free_clean
    terminal_wall_clean &= ~roomseg_free_clean

    ray_fan_input = _optional_bool(_ray(ray_evidence, "ray_fan_candidate", "roomseg_ray_fan_candidate"), shape)
    weak_ray_fan = free_raw & ~strong_stable & ~nav_reachable
    ray_fan_spur_rejected = (ray_fan_input | weak_ray_fan) & free_raw & ~roomseg_free_clean
    free_noise_rejected = free_raw & ~roomseg_free_clean
    isolated_free_rejected = np.asarray(morph_debug.get("_isolated_free_rejected", np.zeros(shape, dtype=bool)), dtype=bool)

    ray_covered = _optional_uint(_ray(ray_evidence, "ray_covered_count", "roomseg_ray_covered_count"), shape)
    observed_clean = roomseg_free_clean | structural_wall_clean | (ray_covered > 0) | terminal_wall_candidate
    roomseg_unknown_clean = ~observed_clean
    roomseg_unknown_clean &= ~roomseg_free_clean
    roomseg_unknown_clean &= ~structural_wall_clean

    structural_conf = np.zeros(shape, dtype=np.float32)
    structural_conf[structural_wall_candidate] = 0.60
    structural_conf[structural_wall_clean] = 0.95
    structural_conf[terminal_wall_clean] = 1.00

    raw_labels, raw_count = label_components(roomseg_free_clean, 4)
    raw_labels = relabel_compact(raw_labels)
    largest_free_cells = _largest_label_cells(raw_labels)
    debug = {
        "algorithm": ONLINE_LINE_EXTEND_ROOMSEG_V4_BACKEND,
        "source": "roomseg_evidence_v4",
        "v3_source_algorithm": str(v3.debug.get("algorithm", "")),
        "free_raw_cells": int(np.count_nonzero(free_raw)),
        "free_stable_cells": int(np.count_nonzero(roomseg_free_stable)),
        "free_clean_cells": int(np.count_nonzero(roomseg_free_clean)),
        "free_noise_rejected_cells": int(np.count_nonzero(free_noise_rejected)),
        "ray_fan_spur_rejected_cells": int(np.count_nonzero(ray_fan_spur_rejected)),
        "isolated_free_rejected_cells": int(np.count_nonzero(isolated_free_rejected)),
        "unknown_clean_cells": int(np.count_nonzero(roomseg_unknown_clean)),
        "nav_reachable_free_cells": int(np.count_nonzero(nav_reachable)),
        "terminal_wall_structural_candidate_cells": int(np.count_nonzero(terminal_wall_candidate)),
        "terminal_wall_structural_clean_cells": int(np.count_nonzero(terminal_wall_clean)),
        "structural_wall_candidate_cells": int(np.count_nonzero(structural_wall_candidate)),
        "structural_wall_clean_cells": int(np.count_nonzero(structural_wall_clean)),
        "ray_covered_cells": int(np.count_nonzero(ray_covered)),
        "ray_covered_count_sum": int(np.sum(ray_covered, dtype=np.uint64)),
        "terminal_wall_count_sum": int(np.sum(terminal_count, dtype=np.uint64)),
        "roomseg_free_component_count": int(raw_count),
        "largest_free_area_m2": float(largest_free_cells) * float(resolution_m) ** 2,
        "stable_filter": {
            "min_free_rays_for_stable_free": int(min_free_rays),
            "min_observed_rays_for_stable_cell": int(min_observed),
            "allow_single_ray_free_if_nav_reachable": bool(cfg.get("allow_single_ray_free_if_nav_reachable", True)),
            "require_nav_reachable_for_roomseg_free": bool(cfg.get("require_nav_reachable_for_roomseg_free", True)),
            "strong_stable_cells": int(np.count_nonzero(strong_stable)),
            "single_ray_reachable_cells": int(np.count_nonzero(single_ray_reachable)),
        },
        "morphology": {key: value for key, value in morph_debug.items() if not str(key).startswith("_")},
        "invariants": {
            "free_unknown_overlap_cells": int(np.count_nonzero(roomseg_free_clean & roomseg_unknown_clean)),
            "wall_free_overlap_cells": int(np.count_nonzero(structural_wall_clean & roomseg_free_clean)),
        },
    }
    return RoomSegEvidenceV4(
        vertical_free_raw=free_raw.astype(bool),
        vertical_occupied_raw=np.asarray(v3.raw_endpoint_occupied, dtype=bool),
        vertical_observed_raw=(~np.asarray(v3.roomseg_unknown_raw, dtype=bool)).astype(bool),
        vertical_unknown_raw=np.asarray(v3.roomseg_unknown_raw, dtype=bool),
        ray_covered_count=ray_covered.astype(np.uint16),
        terminal_wall_count=terminal_count.astype(np.uint16),
        terminal_wall_structural_candidate=terminal_wall_candidate.astype(bool),
        terminal_wall_structural_clean=terminal_wall_clean.astype(bool),
        roomseg_free_raw=free_raw.astype(bool),
        roomseg_free_stable=roomseg_free_stable.astype(bool),
        roomseg_free_clean=roomseg_free_clean.astype(bool),
        roomseg_unknown_clean=roomseg_unknown_clean.astype(bool),
        structural_wall_candidate=structural_wall_candidate.astype(bool),
        structural_wall_clean=structural_wall_clean.astype(bool),
        structural_wall_confidence=structural_conf.astype(np.float32),
        nav_reachable_free=nav_reachable.astype(bool),
        free_noise_rejected=free_noise_rejected.astype(bool),
        ray_fan_spur_rejected=ray_fan_spur_rejected.astype(bool),
        isolated_free_rejected=isolated_free_rejected.astype(bool),
        debug=debug,
        v3_evidence=v3,
    )


def _clean_stable_free(
    free: np.ndarray,
    *,
    nav_reachable: np.ndarray,
    resolution_m: float,
    cfg: Mapping[str, object],
) -> tuple[np.ndarray, dict]:
    out = np.asarray(free, dtype=bool).copy()
    before = out.copy()
    close_cells = _meters_to_cells(float(cfg.get("free_close_radius_m", 0.05)), float(resolution_m))
    open_cells = _meters_to_cells(float(cfg.get("free_open_radius_m", 0.00)), float(resolution_m))
    if close_cells > 0 and np.any(out):
        out = ndimage.binary_closing(out, structure=disk(close_cells)).astype(bool)
    if open_cells > 0 and np.any(out):
        out = ndimage.binary_opening(out, structure=disk(open_cells)).astype(bool)
    min_area_cells = max(1, int(round(float(cfg.get("min_free_component_area_m2", 0.20)) / max(float(resolution_m) ** 2, 1e-9))))
    labels, count = label_components(out, 4)
    keep = np.zeros_like(out, dtype=bool)
    isolated = np.zeros_like(out, dtype=bool)
    for label in range(1, int(count) + 1):
        comp = labels == label
        touches_nav = bool(np.any(comp & nav_reachable))
        area = int(np.count_nonzero(comp))
        if area >= int(min_area_cells) and (touches_nav or not bool(cfg.get("require_nav_reachable_for_roomseg_free", True))):
            keep |= comp
        else:
            isolated |= comp
    out = keep
    spur_cells = _meters_to_cells(float(cfg.get("spur_prune_radius_m", 0.10)), float(resolution_m))
    if spur_cells > 0 and np.any(out):
        opened = ndimage.binary_opening(out, structure=conn(4), iterations=max(1, spur_cells)).astype(bool)
        if np.count_nonzero(opened) >= max(1, int(0.75 * np.count_nonzero(out))):
            isolated |= out & ~opened
            out = opened
    return out.astype(bool), {
        "free_cells_before_morphology": int(np.count_nonzero(before)),
        "free_cells_after_morphology": int(np.count_nonzero(out)),
        "free_close_radius_cells": int(close_cells),
        "free_open_radius_cells": int(open_cells),
        "spur_prune_radius_cells": int(spur_cells),
        "min_free_component_area_cells": int(min_area_cells),
        "isolated_component_rejected_cells": int(np.count_nonzero(isolated)),
        "_isolated_free_rejected": isolated.astype(bool),
    }


def _reachable_free(
    *,
    traversible: np.ndarray | None,
    observed_free: np.ndarray,
    obstacle: np.ndarray,
    agent_grid: tuple[int, int] | None,
    resolution_m: float,
    dilation_m: float,
    shape: tuple[int, int],
) -> np.ndarray:
    if traversible is None:
        passable = np.asarray(observed_free, dtype=bool) & ~np.asarray(obstacle, dtype=bool)
    else:
        passable = np.asarray(traversible, dtype=bool)
        if passable.shape != shape:
            passable = np.asarray(observed_free, dtype=bool) & ~np.asarray(obstacle, dtype=bool)
    seed = _snap_seed(passable, agent_grid, max_radius=max(1, _meters_to_cells(0.50, resolution_m)))
    if seed is None:
        return np.zeros(shape, dtype=bool)
    reached = np.zeros(shape, dtype=bool)
    reached[seed] = True
    q: deque[tuple[int, int]] = deque([seed])
    h, w = shape
    while q:
        row, col = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if reached[rr, cc] or not passable[rr, cc]:
                    continue
                reached[rr, cc] = True
                q.append((rr, cc))
    radius = _meters_to_cells(float(dilation_m), float(resolution_m))
    if radius > 0:
        reached = ndimage.binary_dilation(reached, structure=disk(radius)).astype(bool)
    return reached.astype(bool)


def _snap_seed(mask: np.ndarray, seed: tuple[int, int] | None, *, max_radius: int) -> tuple[int, int] | None:
    src = np.asarray(mask, dtype=bool)
    if seed is None:
        rows, cols = np.nonzero(src)
        if rows.size == 0:
            return None
        idx = int(rows.size // 2)
        return int(rows[idx]), int(cols[idx])
    row, col = int(seed[0]), int(seed[1])
    if 0 <= row < src.shape[0] and 0 <= col < src.shape[1] and bool(src[row, col]):
        return row, col
    best: tuple[int, int] | None = None
    best_d2 = None
    radius = max(0, int(max_radius))
    for rr in range(max(0, row - radius), min(src.shape[0], row + radius + 1)):
        for cc in range(max(0, col - radius), min(src.shape[1], col + radius + 1)):
            if not bool(src[rr, cc]):
                continue
            d2 = (rr - row) * (rr - row) + (cc - col) * (cc - col)
            if best_d2 is None or d2 < best_d2:
                best = (int(rr), int(cc))
                best_d2 = int(d2)
    return best


def _synthetic_vertical_profile_from_masks(free_mask: np.ndarray, occupied_mask: np.ndarray, observed_mask: np.ndarray) -> VerticalProfileMap:
    free = np.asarray(free_mask, dtype=bool)
    occupied = np.asarray(occupied_mask, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool) | free | occupied
    bands = 4
    free_count = np.zeros((bands, *free.shape), dtype=np.uint16)
    occupied_count = np.zeros_like(free_count)
    observed_count = np.zeros_like(free_count)
    unknown_count = np.ones_like(free_count)
    free_count[:, free] = 1
    occupied_count[:, occupied & ~free] = 1
    observed_count[:, observed] = 1
    unknown_count[observed_count > 0] = 0
    return VerticalProfileMap.from_counts(
        occupied_count=occupied_count,
        free_ray_count=free_count,
        observed_count=observed_count,
        unknown_count=unknown_count,
    )


def _ray(evidence: Mapping[str, np.ndarray], *names: str) -> np.ndarray | None:
    for name in names:
        if name in evidence and evidence[name] is not None:
            return np.asarray(evidence[name])
    return None


def _optional_uint(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.uint16)
    arr = np.asarray(value, dtype=np.uint32)
    if arr.shape != shape:
        return np.zeros(shape, dtype=np.uint16)
    return np.minimum(arr, np.iinfo(np.uint16).max).astype(np.uint16)


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        return np.zeros(shape, dtype=bool)
    return arr.astype(bool)


def _meters_to_cells(value_m: float, resolution_m: float) -> int:
    return max(0, int(round(float(value_m) / max(float(resolution_m), 1e-6))))


def _largest_label_cells(labels: np.ndarray) -> int:
    best = 0
    for label in np.unique(np.asarray(labels, dtype=np.int32)):
        if int(label) <= 0:
            continue
        best = max(best, int(np.count_nonzero(labels == int(label))))
    return int(best)


def _root_config(config: Mapping[str, object] | object | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    out = {}
    for name in (
        "roomseg_evidence_v3",
        "roomseg_evidence_v4",
        "navigation_consistency",
        "structural_wall_v3",
        "resolution_m",
    ):
        if hasattr(config, name):
            out[name] = getattr(config, name)
    return out


def _section(config: Mapping[str, object], name: str) -> dict:
    raw = config.get(name, {})
    return dict(raw or {}) if isinstance(raw, Mapping) else {}
