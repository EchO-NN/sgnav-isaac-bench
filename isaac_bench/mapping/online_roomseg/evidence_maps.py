from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.vertical_profile import VerticalProfileMap, ensure_vertical_profile

from .utils import component_metrics, fill_small_holes, label_components, remove_small_components


@dataclass
class FreeCleanConfig:
    enabled: bool = True
    hole_fill_max_area_cells: int = 16
    hole_fill_max_radius_cells: int = 3
    island_remove_max_area_cells: int = 8
    do_not_fill_if_occupied_ratio_gt: float = 0.15

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "FreeCleanConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WallCandidateConfig:
    enabled: bool = True
    mode: str = "jitter_permissive"
    jitter_tolerance_cells: int = 1
    isolated_remove_max_area_cells: int = 1
    shape_gate_enabled: bool = False
    min_component_area_cells: int = 6
    min_component_length_m: float = 0.25
    max_component_thickness_m: float = 0.25
    min_component_elongation: float = 2.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WallCandidateConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class RoomSegEvidence:
    vertical_free_raw: np.ndarray
    vertical_occupied_raw: np.ndarray
    vertical_observed_raw: np.ndarray
    vertical_unknown_raw: np.ndarray
    free_clean: np.ndarray
    wall_candidate_clean: np.ndarray
    unknown_clean: np.ndarray
    resolution_m: float
    debug: dict = field(default_factory=dict)


def build_evidence_maps(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    resolution_m: float,
    vertical_profile: VerticalProfileMap | None = None,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    free_clean_config: FreeCleanConfig | Mapping[str, object] | None = None,
    wall_candidate_config: WallCandidateConfig | Mapping[str, object] | None = None,
    z_min_m: float = 0.20,
    z_max_m: float = 2.00,
    min_free_rays: int = 1,
    min_observed_rays: int = 1,
) -> RoomSegEvidence:
    shape = np.asarray(observed_free_mask, dtype=bool).shape
    free_cfg = free_clean_config if isinstance(free_clean_config, FreeCleanConfig) else FreeCleanConfig.from_mapping(free_clean_config)
    wall_cfg = wall_candidate_config if isinstance(wall_candidate_config, WallCandidateConfig) else WallCandidateConfig.from_mapping(wall_candidate_config)
    vertical_free, vertical_occupied, vertical_observed, source_debug = _vertical_maps(
        shape=shape,
        occupancy_map=occupancy_map,
        observed_free_mask=observed_free_mask,
        obstacle_mask=obstacle_mask,
        unknown_mask=unknown_mask,
        vertical_profile=vertical_profile,
        z_min_m=float(z_min_m),
        z_max_m=float(z_max_m),
        min_free_rays=int(min_free_rays),
        min_observed_rays=int(min_observed_rays),
    )
    vertical_occupied, vertical_observed, endpoint_debug = _merge_terminal_wall_endpoints(
        vertical_free=vertical_free,
        vertical_occupied=vertical_occupied,
        vertical_observed=vertical_observed,
        roomseg_ray_evidence=roomseg_ray_evidence,
    )
    vertical_unknown = ~vertical_observed
    free_clean, free_debug = clean_free_map(
        vertical_free,
        vertical_occupied,
        vertical_unknown,
        free_cfg,
    )
    wall_clean, wall_debug = clean_wall_candidate_map(
        vertical_free_raw=vertical_free,
        vertical_occupied_raw=vertical_occupied,
        vertical_observed_raw=vertical_observed,
        free_clean=free_clean,
        resolution_m=float(resolution_m),
        config=wall_cfg,
    )
    unknown_clean = ~(free_clean | wall_clean)
    debug = {
        "source": "online_roomseg_evidence_maps",
        **source_debug,
        "vertical_free_raw_cells": int(np.count_nonzero(vertical_free)),
        "vertical_occupied_raw_cells": int(np.count_nonzero(vertical_occupied)),
        "vertical_observed_raw_cells": int(np.count_nonzero(vertical_observed)),
        "vertical_unknown_raw_cells": int(np.count_nonzero(vertical_unknown)),
        **endpoint_debug,
        "free_clean_cells": int(np.count_nonzero(free_clean)),
        "wall_candidate_clean_cells": int(np.count_nonzero(wall_clean)),
        "unknown_clean_cells": int(np.count_nonzero(unknown_clean)),
        "free_clean": free_debug,
        "wall_candidate_clean": wall_debug,
    }
    return RoomSegEvidence(
        vertical_free_raw=vertical_free.astype(bool),
        vertical_occupied_raw=vertical_occupied.astype(bool),
        vertical_observed_raw=vertical_observed.astype(bool),
        vertical_unknown_raw=vertical_unknown.astype(bool),
        free_clean=free_clean.astype(bool),
        wall_candidate_clean=wall_clean.astype(bool),
        unknown_clean=unknown_clean.astype(bool),
        resolution_m=float(resolution_m),
        debug=debug,
    )


def _merge_terminal_wall_endpoints(
    *,
    vertical_free: np.ndarray,
    vertical_occupied: np.ndarray,
    vertical_observed: np.ndarray,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    free = np.asarray(vertical_free, dtype=bool)
    occupied_before = np.asarray(vertical_occupied, dtype=bool)
    observed_before = np.asarray(vertical_observed, dtype=bool)
    occupied = occupied_before.copy()
    observed = observed_before.copy()
    evidence = dict(roomseg_ray_evidence or {})
    terminal_count = _optional_like(evidence, free.shape, "terminal_wall_count", "roomseg_terminal_wall_count", dtype=np.uint16)
    terminal_splat = _optional_like(evidence, free.shape, "terminal_wall_splat", "roomseg_terminal_wall_splat", dtype=np.uint8)
    terminal = (np.asarray(terminal_count, dtype=np.uint16) > 0) | np.asarray(terminal_splat, dtype=bool)
    terminal_no_free = terminal & ~free
    occupied |= terminal_no_free
    observed |= terminal_no_free
    return occupied.astype(bool), observed.astype(bool), {
        "terminal_wall_endpoint_cells": int(np.count_nonzero(np.asarray(terminal_count, dtype=np.uint16) > 0)),
        "terminal_wall_splat_cells": int(np.count_nonzero(np.asarray(terminal_splat, dtype=bool))),
        "terminal_wall_added_to_occupied_cells": int(np.count_nonzero(terminal_no_free & ~occupied_before)),
        "terminal_wall_added_to_observed_cells": int(np.count_nonzero(terminal_no_free & ~observed_before)),
        "terminal_wall_suppressed_by_vertical_free_cells": int(np.count_nonzero(terminal & free)),
    }


def _optional_like(
    evidence: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    *names: str,
    dtype: type,
) -> np.ndarray:
    for name in names:
        value = evidence.get(name)
        if value is None:
            continue
        arr = np.asarray(value, dtype=dtype)
        if arr.shape == shape:
            return arr
    return np.zeros(shape, dtype=dtype)


def clean_free_map(
    vertical_free_raw: np.ndarray,
    vertical_occupied_raw: np.ndarray,
    vertical_unknown_raw: np.ndarray,
    config: FreeCleanConfig,
) -> tuple[np.ndarray, dict]:
    free = np.asarray(vertical_free_raw, dtype=bool).copy()
    occupied = np.asarray(vertical_occupied_raw, dtype=bool)
    if bool(config.enabled):
        filled, holes = fill_small_holes(
            free,
            occupied=occupied,
            max_area_cells=int(config.hole_fill_max_area_cells),
            max_radius_cells=int(config.hole_fill_max_radius_cells),
            occupied_ratio_max=float(config.do_not_fill_if_occupied_ratio_gt),
        )
        free = filled
        if int(config.island_remove_max_area_cells) > 0:
            labels, count = label_components(free, 4)
            for idx in range(1, int(count) + 1):
                comp = labels == idx
                if int(np.count_nonzero(comp)) <= int(config.island_remove_max_area_cells):
                    free[comp] = False
    else:
        holes = []
    free &= ~np.asarray(vertical_unknown_raw, dtype=bool)
    return free.astype(bool), {
        "enabled": bool(config.enabled),
        "hole_fill_candidates": holes[:128],
        "hole_fill_accepted": int(sum(1 for item in holes if item.get("accepted"))),
        "free_cells_after_clean": int(np.count_nonzero(free)),
    }


def clean_wall_candidate_map(
    *,
    vertical_free_raw: np.ndarray,
    vertical_occupied_raw: np.ndarray,
    vertical_observed_raw: np.ndarray,
    free_clean: np.ndarray,
    resolution_m: float,
    config: WallCandidateConfig,
) -> tuple[np.ndarray, dict]:
    raw = (np.asarray(vertical_occupied_raw, dtype=bool) | (np.asarray(vertical_observed_raw, dtype=bool) & ~np.asarray(vertical_free_raw, dtype=bool))) & ~np.asarray(free_clean, dtype=bool)
    if not bool(config.enabled):
        return raw.astype(bool), {"enabled": False, "input_cells": int(np.count_nonzero(raw)), "kept_cells": int(np.count_nonzero(raw)), "components": []}
    if str(getattr(config, "mode", "jitter_permissive")) == "jitter_permissive":
        return _clean_wall_candidate_map_jitter_permissive(raw, float(resolution_m), config)
    labels, count = label_components(raw, 8)
    out = np.zeros_like(raw, dtype=bool)
    components: list[dict] = []
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        metrics = component_metrics(comp, float(resolution_m))
        keep = bool(
            metrics["area_cells"] >= int(config.min_component_area_cells)
            and metrics["length_m"] >= float(config.min_component_length_m)
            and metrics["thickness_m"] <= float(config.max_component_thickness_m) + 1e-6
            and metrics["elongation"] >= float(config.min_component_elongation)
        )
        if keep:
            out |= comp
        components.append({"component": int(idx), **metrics, "kept": bool(keep)})
    return out.astype(bool), {
        "enabled": True,
        "mode": "shape_gate",
        "input_cells": int(np.count_nonzero(raw)),
        "kept_cells": int(np.count_nonzero(out)),
        "components": components[:256],
    }


def _clean_wall_candidate_map_jitter_permissive(raw: np.ndarray, resolution_m: float, config: WallCandidateConfig) -> tuple[np.ndarray, dict]:
    candidate = np.asarray(raw, dtype=bool).copy()
    jitter = max(0, int(getattr(config, "jitter_tolerance_cells", 1)))
    if jitter > 0 and np.any(candidate):
        structure = np.ones((2 * jitter + 1, 2 * jitter + 1), dtype=bool)
        candidate = (candidate | ndimage.binary_closing(candidate, structure=structure, border_value=0)).astype(bool)

    labels, count = label_components(candidate, 8)
    out = candidate.copy()
    components: list[dict] = []
    removed_isolated = 0
    removed_shape_gate = 0
    max_isolated = int(getattr(config, "isolated_remove_max_area_cells", 1))
    shape_gate = bool(getattr(config, "shape_gate_enabled", False))
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        area = int(np.count_nonzero(comp))
        metrics = component_metrics(comp, float(resolution_m))
        keep = True
        reason = "kept"
        if max_isolated >= 0 and area <= max_isolated:
            keep = False
            reason = "isolated_single_cell_noise"
            removed_isolated += 1
        elif shape_gate:
            keep = bool(
                metrics["area_cells"] >= int(config.min_component_area_cells)
                and metrics["length_m"] >= float(config.min_component_length_m)
                and metrics["thickness_m"] <= float(config.max_component_thickness_m) + 1e-6
                and metrics["elongation"] >= float(config.min_component_elongation)
            )
            if not keep:
                reason = "shape_gate_rejected"
                removed_shape_gate += 1
        if not keep:
            out[comp] = False
        components.append({"component": int(idx), **metrics, "kept": bool(keep), "reject_reason": reason})
    return out.astype(bool), {
        "enabled": True,
        "mode": "jitter_permissive",
        "input_cells": int(np.count_nonzero(raw)),
        "jitter_tolerance_cells": int(jitter),
        "cells_after_jitter_closing": int(np.count_nonzero(candidate)),
        "kept_cells": int(np.count_nonzero(out)),
        "removed_isolated_component_count": int(removed_isolated),
        "removed_shape_gate_component_count": int(removed_shape_gate),
        "shape_gate_enabled": bool(shape_gate),
        "components": components[:256],
    }


def _vertical_maps(
    *,
    shape: tuple[int, int],
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    vertical_profile: VerticalProfileMap | None,
    z_min_m: float,
    z_max_m: float,
    min_free_rays: int,
    min_observed_rays: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    vp = ensure_vertical_profile(vertical_profile, shape)
    indices = [
        idx
        for idx, (_name, (lo, hi)) in enumerate(zip(vp.band_names, vp.band_ranges_m))
        if float(hi) > float(z_min_m) and float(lo) < float(z_max_m)
    ]
    if indices:
        band_names = tuple(str(vp.band_names[idx]) for idx in indices)
        free = vp.reliable_free_mask(
            min_free_rays=int(min_free_rays),
            min_observed_rays=int(min_observed_rays),
            band_names=band_names,
        ).astype(bool)
        occupied_count = np.sum(np.asarray(vp.occupied_count[indices], dtype=np.uint32), axis=0)
        observed_count = np.sum(np.asarray(vp.observed_count[indices], dtype=np.uint32), axis=0)
        occupied = (occupied_count > 0) & ~free
        observed = observed_count >= int(min_observed_rays)
        if np.any(observed):
            return free.astype(bool), occupied.astype(bool), observed.astype(bool), {
                "vertical_source": "vertical_profile_0p2_2p0",
                "vertical_band_names": list(band_names),
            }
    unknown = np.asarray(unknown_mask, dtype=bool)
    free = np.asarray(observed_free_mask, dtype=bool) & ~unknown
    occupied = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool) & ~unknown & ~free
    observed = ~unknown
    return free.astype(bool), occupied.astype(bool), observed.astype(bool), {
        "vertical_source": "fallback_observed_masks",
        "vertical_band_names": [],
    }
