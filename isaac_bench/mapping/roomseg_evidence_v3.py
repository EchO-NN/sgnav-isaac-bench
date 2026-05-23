from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.vertical_profile import VerticalProfileMap, ensure_vertical_profile


@dataclass
class RoomSegEvidenceV3:
    roomseg_free_raw: np.ndarray
    roomseg_free_clean: np.ndarray
    roomseg_unknown_raw: np.ndarray
    roomseg_unknown_clean: np.ndarray
    raw_endpoint_occupied: np.ndarray
    terminal_wall_candidate: np.ndarray
    structural_wall_candidate: np.ndarray
    structural_wall_clean: np.ndarray
    ray_free_count: np.ndarray
    ray_covered_count: np.ndarray
    endpoint_count_by_band: np.ndarray
    free_count_by_band: np.ndarray
    observed_count_by_band: np.ndarray
    navigation_free_support: np.ndarray
    navigation_reachable_support: np.ndarray
    navigation_obstacle_support: np.ndarray
    disagreement_map: np.ndarray
    debug: dict = field(default_factory=dict)


def build_roomseg_evidence_v3(
    *,
    vertical_profile: VerticalProfileMap,
    grid_free: np.ndarray,
    grid_occupied: np.ndarray,
    grid_observed: np.ndarray,
    roomseg_ray_covered_count: np.ndarray | None,
    roomseg_terminal_wall_count: np.ndarray | None,
    roomseg_terminal_wall_splat: np.ndarray | None,
    roomseg_terminal_wall_height_min: np.ndarray | None,
    roomseg_terminal_wall_height_max: np.ndarray | None,
    roomseg_terminal_wall_depth_min: np.ndarray | None,
    robot_rc: tuple[int, int] | None,
    resolution_m: float,
    config: Mapping[str, object] | object | None,
) -> RoomSegEvidenceV3:
    cfg = _root_config(config)
    ev_cfg = _section(cfg, "roomseg_evidence_v3")
    wall_cfg = _section(cfg, "structural_wall_v3")
    shape = np.asarray(grid_observed, dtype=bool).shape
    vp = ensure_vertical_profile(vertical_profile, shape)

    free_count_by_band = np.asarray(vp.free_ray_count, dtype=np.uint32)
    endpoint_count_by_band = np.asarray(vp.occupied_count, dtype=np.uint32)
    observed_count_by_band = np.asarray(vp.observed_count, dtype=np.uint32)
    ray_free_count = np.sum(free_count_by_band, axis=0, dtype=np.uint32)
    ray_covered_count = _optional_uint(roomseg_ray_covered_count, shape)
    terminal_count = _optional_uint(roomseg_terminal_wall_count, shape)
    terminal_splat = _optional_bool(roomseg_terminal_wall_splat, shape)
    terminal_height_min = _optional_float(roomseg_terminal_wall_height_min, shape, fill=np.inf)
    terminal_height_max = _optional_float(roomseg_terminal_wall_height_max, shape, fill=-np.inf)
    terminal_depth_min = _optional_float(roomseg_terminal_wall_depth_min, shape, fill=np.inf)

    min_observed_count = max(0, int(ev_cfg.get("min_observed_count", 1)))
    min_any_free = max(0, int(ev_cfg.get("min_any_free_count", 1)))
    roomseg_free_raw = _free_raw_from_bands(
        free_count_by_band=free_count_by_band,
        observed_count_by_band=observed_count_by_band,
        band_names=vp.band_names,
        ev_cfg=ev_cfg,
        min_any_free=min_any_free,
        min_observed_count=min_observed_count,
    )

    min_endpoint_raw = max(1, int(ev_cfg.get("min_endpoint_count_raw", 1)))
    endpoint_count_sum = np.sum(endpoint_count_by_band, axis=0, dtype=np.uint32)
    raw_endpoint_occupied = endpoint_count_sum >= int(min_endpoint_raw)

    min_terminal_wall_count = max(1, int(ev_cfg.get("min_terminal_wall_count", 2)))
    depth_max_m = _depth_max_m(cfg, ev_cfg)
    depth_far_margin_m = float(ev_cfg.get("depth_far_margin_m", 0.08))
    min_ray_near = max(1, int(ev_cfg.get("min_ray_support_near_endpoint", 2)))
    height_min = float(ev_cfg.get("terminal_wall_height_min_m", 0.20))
    height_max = float(ev_cfg.get("terminal_wall_height_max_m", 2.00))
    ray_support_near_endpoint = _near_count_at_least(ray_covered_count, min_ray_near, radius_cells=1)
    height_overlap = (terminal_height_max >= height_min) & (terminal_height_min <= height_max)
    depth_not_far = np.isfinite(terminal_depth_min) & (terminal_depth_min < float(depth_max_m) - float(depth_far_margin_m))

    component_support, line_support, tiny_island = _endpoint_component_support(
        raw_endpoint_occupied | (terminal_count >= min_terminal_wall_count),
        resolution_m=float(resolution_m),
        min_component_area_m2=float(ev_cfg.get("min_endpoint_component_area_m2", 0.02)),
        min_wall_component_area_m2=float(wall_cfg.get("min_wall_component_area_m2", 0.05)),
        min_wall_line_length_m=float(wall_cfg.get("min_wall_line_length_m", 0.50)),
        tiny_island_area_m2=float(wall_cfg.get("tiny_island_area_m2", 0.12)),
    )
    terminal_wall_candidate = (
        (terminal_count >= min_terminal_wall_count)
        & ~roomseg_free_raw
        & depth_not_far
        & height_overlap
        & ray_support_near_endpoint
        & (component_support["endpoint_area_ok"] | line_support)
    )
    if bool(ev_cfg.get("ray_covered_is_free", False)):
        raise ValueError("roomseg_evidence_v3.ray_covered_is_free must remain false")
    if bool(ev_cfg.get("raw_endpoint_is_wall", False)):
        raise ValueError("roomseg_evidence_v3.raw_endpoint_is_wall must remain false")
    if bool(ev_cfg.get("unknown_is_wall", False)):
        raise ValueError("roomseg_evidence_v3.unknown_is_wall must remain false")

    structural_wall_candidate, structural_wall_clean, wall_debug = _build_structural_wall_maps(
        raw_endpoint_occupied=raw_endpoint_occupied,
        terminal_wall_candidate=terminal_wall_candidate,
        terminal_count=terminal_count,
        roomseg_free_raw=roomseg_free_raw,
        free_count_by_band=free_count_by_band,
        endpoint_count_by_band=endpoint_count_by_band,
        line_support=line_support,
        component_support=component_support["wall_component_ok"],
        tiny_island=tiny_island,
        resolution_m=float(resolution_m),
        ev_cfg=ev_cfg,
        wall_cfg=wall_cfg,
    )

    roomseg_free_clean, navigation_reachable, nav_debug = clean_roomseg_free_with_navigation_support(
        roomseg_free_raw=roomseg_free_raw,
        grid_free=grid_free,
        grid_occupied=grid_occupied,
        grid_observed=grid_observed,
        structural_wall_clean=structural_wall_clean,
        robot_rc=robot_rc,
        resolution_m=float(resolution_m),
        config=cfg,
        free_count_by_band=free_count_by_band,
    )
    structural_wall_clean &= ~roomseg_free_clean
    structural_wall_candidate &= ~roomseg_free_clean

    ray_covered_observed = ray_covered_count > 0
    observed_raw = roomseg_free_raw | raw_endpoint_occupied | ray_covered_observed
    roomseg_unknown_raw = ~observed_raw
    observed_clean = roomseg_free_clean | structural_wall_clean | ray_covered_observed
    roomseg_unknown_clean = ~observed_clean
    roomseg_unknown_clean &= ~roomseg_free_clean
    roomseg_unknown_clean &= ~structural_wall_clean

    nav_free = np.asarray(grid_free, dtype=bool) & np.asarray(grid_observed, dtype=bool)
    nav_obstacle = np.asarray(grid_occupied, dtype=bool) & np.asarray(grid_observed, dtype=bool)
    disagreement_map = _disagreement_map(
        roomseg_free_raw=roomseg_free_raw,
        nav_reachable_free=navigation_reachable,
        nav_obstacle=nav_obstacle,
        resolution_m=float(resolution_m),
        config=cfg,
    )

    labels_free, free_components = label_components(roomseg_free_clean, 4)
    largest_free_cells = _largest_component_cells(labels_free, free_components)
    labels_outside_free = int(np.count_nonzero((roomseg_free_clean.astype(np.uint8) > 0) & ~roomseg_free_clean))
    debug = {
        "algorithm": "roomseg_evidence_line_closure_v3",
        "source": "roomseg_evidence_v3",
        "depth_max_m": float(depth_max_m),
        "free_raw_cells": int(np.count_nonzero(roomseg_free_raw)),
        "free_clean_cells": int(np.count_nonzero(roomseg_free_clean)),
        "unknown_raw_cells": int(np.count_nonzero(roomseg_unknown_raw)),
        "unknown_clean_cells": int(np.count_nonzero(roomseg_unknown_clean)),
        "raw_endpoint_occupied_cells": int(np.count_nonzero(raw_endpoint_occupied)),
        "terminal_wall_candidate_cells": int(np.count_nonzero(terminal_wall_candidate)),
        "structural_wall_candidate_cells": int(np.count_nonzero(structural_wall_candidate)),
        "structural_wall_clean_cells": int(np.count_nonzero(structural_wall_clean)),
        "ray_free_count_sum": int(np.sum(ray_free_count, dtype=np.uint64)),
        "ray_covered_count_sum": int(np.sum(ray_covered_count, dtype=np.uint64)),
        "ray_covered_cells": int(np.count_nonzero(ray_covered_count)),
        "terminal_wall_count_sum": int(np.sum(terminal_count, dtype=np.uint64)),
        "terminal_wall_splat_input_cells": int(np.count_nonzero(terminal_splat)),
        "roomseg_free_component_count": int(free_components),
        "largest_free_area_m2": float(largest_free_cells) * float(resolution_m) ** 2,
        "labels_outside_free_cells": int(labels_outside_free),
        "labels_in_unknown_cells": 0,
        "invariants": {
            "free_unknown_overlap_cells": int(np.count_nonzero(roomseg_free_clean & roomseg_unknown_clean)),
            "wall_free_overlap_cells": int(np.count_nonzero(structural_wall_clean & roomseg_free_clean)),
        },
        "terminal_wall_filter": {
            "min_terminal_wall_count": int(min_terminal_wall_count),
            "depth_far_margin_m": float(depth_far_margin_m),
            "min_ray_support_near_endpoint": int(min_ray_near),
            "height_min_m": float(height_min),
            "height_max_m": float(height_max),
            "depth_not_far_cells": int(np.count_nonzero(depth_not_far)),
            "height_overlap_cells": int(np.count_nonzero(height_overlap)),
            "near_ray_support_cells": int(np.count_nonzero(ray_support_near_endpoint)),
        },
        "structural_wall_v3": wall_debug,
        "navigation_consistency": nav_debug,
    }
    _assert_invariants(
        roomseg_free_clean=roomseg_free_clean,
        roomseg_unknown_clean=roomseg_unknown_clean,
        structural_wall_clean=structural_wall_clean,
    )
    return RoomSegEvidenceV3(
        roomseg_free_raw=roomseg_free_raw.astype(bool),
        roomseg_free_clean=roomseg_free_clean.astype(bool),
        roomseg_unknown_raw=roomseg_unknown_raw.astype(bool),
        roomseg_unknown_clean=roomseg_unknown_clean.astype(bool),
        raw_endpoint_occupied=raw_endpoint_occupied.astype(bool),
        terminal_wall_candidate=terminal_wall_candidate.astype(bool),
        structural_wall_candidate=structural_wall_candidate.astype(bool),
        structural_wall_clean=structural_wall_clean.astype(bool),
        ray_free_count=ray_free_count.astype(np.uint32),
        ray_covered_count=ray_covered_count.astype(np.uint16),
        endpoint_count_by_band=endpoint_count_by_band.astype(np.uint16),
        free_count_by_band=free_count_by_band.astype(np.uint16),
        observed_count_by_band=observed_count_by_band.astype(np.uint16),
        navigation_free_support=nav_free.astype(bool),
        navigation_reachable_support=navigation_reachable.astype(bool),
        navigation_obstacle_support=nav_obstacle.astype(bool),
        disagreement_map=disagreement_map.astype(np.uint8),
        debug=debug,
    )


def clean_roomseg_free_with_navigation_support(
    *,
    roomseg_free_raw: np.ndarray,
    grid_free: np.ndarray,
    grid_occupied: np.ndarray,
    grid_observed: np.ndarray,
    structural_wall_clean: np.ndarray,
    robot_rc: tuple[int, int] | None,
    resolution_m: float,
    config: Mapping[str, object] | object | None,
    free_count_by_band: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = _root_config(config)
    nav_cfg = _section(cfg, "navigation_consistency")
    raw = np.asarray(roomseg_free_raw, dtype=bool)
    wall = np.asarray(structural_wall_clean, dtype=bool)
    nav_observed = np.asarray(grid_observed, dtype=bool)
    nav_free = np.asarray(grid_free, dtype=bool) & nav_observed
    nav_obstacle = np.asarray(grid_occupied, dtype=bool) & nav_observed
    if raw.shape != nav_free.shape or wall.shape != raw.shape:
        raise ValueError("roomseg/nav evidence masks must have the same HxW shape")

    candidate = raw & ~wall
    if not bool(nav_cfg.get("enabled", True)):
        return candidate.astype(bool), np.zeros_like(candidate, dtype=bool), {
            "enabled": False,
            "candidate_free_cells": int(np.count_nonzero(candidate)),
        }

    snap_radius = _meters_to_cells(float(nav_cfg.get("robot_seed_snap_radius_m", 0.50)), float(resolution_m))
    seed, seed_debug = _snap_seed_to_free(nav_free, robot_rc, snap_radius)
    nav_reachable = _flood_fill(nav_free, seed, connectivity=8) if seed is not None else np.zeros_like(nav_free, dtype=bool)
    nav_radius = _meters_to_cells(float(nav_cfg.get("nav_support_dilation_m", 0.15)), float(resolution_m))
    nav_support = dilate(nav_reachable, nav_radius)
    strong = _strong_vertical_free(
        roomseg_free_raw=candidate,
        free_count_by_band=free_count_by_band,
        min_bands=int(nav_cfg.get("strong_vertical_free_min_bands", 2)),
        min_count_sum=int(nav_cfg.get("strong_vertical_free_min_count_sum", 3)),
    )
    keep = candidate & (nav_support | strong)

    min_area_cells = max(1, int(round(float(nav_cfg.get("min_free_component_area_m2", 0.25)) / max(float(resolution_m) ** 2, 1e-9))))
    keep, component_debug = _filter_free_components(
        keep,
        candidate=candidate,
        nav_support=nav_support,
        strong_vertical_free=strong,
        min_area_cells=min_area_cells,
        keep_largest_if_no_nav=bool(nav_cfg.get("keep_largest_component_if_no_nav_support", True)),
        remove_not_touching_nav=bool(nav_cfg.get("remove_components_not_touching_nav_free", True)),
    )
    free_nav_intersection = int(np.count_nonzero(keep & nav_reachable))
    free_nav_union = int(np.count_nonzero(keep | nav_reachable))
    nav_iou = float(free_nav_intersection / max(1, free_nav_union))
    debug = {
        "enabled": True,
        "candidate_free_cells": int(np.count_nonzero(candidate)),
        "nav_free_cells": int(np.count_nonzero(nav_free)),
        "nav_obstacle_cells": int(np.count_nonzero(nav_obstacle)),
        "nav_reachable_cells": int(np.count_nonzero(nav_reachable)),
        "nav_support_cells": int(np.count_nonzero(nav_support)),
        "strong_vertical_free_cells": int(np.count_nonzero(strong)),
        "free_clean_cells": int(np.count_nonzero(keep)),
        "nav_iou": float(nav_iou),
        "max_roomseg_nav_iou_warning": float(nav_cfg.get("max_roomseg_nav_iou_warning", 0.35)),
        "warning_low_roomseg_nav_iou": bool(
            np.any(keep) and np.any(nav_reachable) and nav_iou < float(nav_cfg.get("max_roomseg_nav_iou_warning", 0.35))
        ),
        "seed": seed_debug,
        "components": component_debug,
    }
    return keep.astype(bool), nav_reachable.astype(bool), debug


def _free_raw_from_bands(
    *,
    free_count_by_band: np.ndarray,
    observed_count_by_band: np.ndarray,
    band_names: Sequence[str],
    ev_cfg: Mapping[str, object],
    min_any_free: int,
    min_observed_count: int,
) -> np.ndarray:
    names = [str(name) for name in band_names]
    threshold_by_name = {
        "low": int(ev_cfg.get("min_free_count_low", 1)),
        "robot_body": int(ev_cfg.get("min_free_count_robot", 1)),
        "mid": int(ev_cfg.get("min_free_count_mid", 1)),
        "upper": int(ev_cfg.get("min_free_count_upper", 1)),
    }
    band_free = np.zeros(free_count_by_band.shape[1:], dtype=bool)
    for idx, name in enumerate(names):
        threshold = max(0, int(threshold_by_name.get(name, 1)))
        band_free |= (free_count_by_band[idx] >= threshold) & (observed_count_by_band[idx] >= min_observed_count)
    any_free = np.sum(free_count_by_band, axis=0, dtype=np.uint32) >= int(min_any_free)
    any_observed = np.sum(observed_count_by_band, axis=0, dtype=np.uint32) >= int(min_observed_count)
    return ((band_free | any_free) & any_observed).astype(bool)


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    if int(connectivity) == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    else:
        structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    return ndimage.label(np.asarray(mask, dtype=bool), structure=structure)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    structure = ((yy * yy + xx * xx) <= radius * radius).astype(bool)
    return ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=structure).astype(bool)


def component_metrics(mask: np.ndarray, resolution_m: float) -> dict:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return {"area_cells": 0, "area_m2": 0.0, "length_m": 0.0, "elongation": 0.0, "bbox": [0, 0, 0, 0]}
    coords = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=1)
    spans = np.ptp(coords, axis=0) + 1.0
    bbox_aspect = float(max(spans) / max(1.0, min(spans)))
    if coords.shape[0] >= 2:
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        cov = np.cov(centered.T)
        vals = np.maximum(np.sort(np.linalg.eigvalsh(cov)), 1e-6)
        elongation = float(np.sqrt(vals[-1] / vals[0]))
    else:
        elongation = 1.0
    return {
        "area_cells": int(rows.size),
        "area_m2": float(rows.size) * float(resolution_m) ** 2,
        "length_m": float(max(spans) * float(resolution_m)),
        "elongation": float(max(elongation, bbox_aspect)),
        "bbox": [int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1],
    }


def _build_structural_wall_maps(
    *,
    raw_endpoint_occupied: np.ndarray,
    terminal_wall_candidate: np.ndarray,
    terminal_count: np.ndarray,
    roomseg_free_raw: np.ndarray,
    free_count_by_band: np.ndarray,
    endpoint_count_by_band: np.ndarray,
    line_support: np.ndarray,
    component_support: np.ndarray,
    tiny_island: np.ndarray,
    resolution_m: float,
    ev_cfg: Mapping[str, object],
    wall_cfg: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, dict]:
    occ_band_count = np.count_nonzero(endpoint_count_by_band > 0, axis=0).astype(np.float32)
    free_band_count = np.count_nonzero(free_count_by_band > 0, axis=0).astype(np.float32)
    vertical_continuity = np.clip(occ_band_count / max(1.0, float(endpoint_count_by_band.shape[0])), 0.0, 1.0)
    free_suppression = np.clip(free_band_count / max(1.0, float(free_count_by_band.shape[0])), 0.0, 1.0)
    min_terminal = max(1, int(ev_cfg.get("min_terminal_wall_count", 2)))
    terminal_support = np.clip(np.asarray(terminal_count, dtype=np.float32) / float(min_terminal), 0.0, 1.0)
    terminal_support *= np.asarray(terminal_wall_candidate, dtype=np.float32)
    two_side_penalty = _two_side_free_penalty(
        roomseg_free_raw,
        radius_cells=_meters_to_cells(float(wall_cfg.get("two_side_free_check_radius_m", 0.25)), float(resolution_m)),
    )
    two_side_penalty = np.asarray(two_side_penalty, dtype=bool) & ~np.asarray(line_support, dtype=bool)
    wall_score = (
        0.30 * vertical_continuity
        + 0.25 * terminal_support
        + 0.25 * np.asarray(line_support, dtype=np.float32)
        + 0.20 * np.asarray(component_support, dtype=np.float32)
        - 0.45 * free_suppression
        - 0.25 * np.asarray(two_side_penalty, dtype=np.float32)
        - 0.20 * np.asarray(tiny_island, dtype=np.float32)
    )
    candidate_threshold = float(wall_cfg.get("candidate_score_threshold", 0.55))
    clean_threshold = float(wall_cfg.get("clean_score_threshold", 0.68))
    min_clean_bands = min(
        int(endpoint_count_by_band.shape[0]),
        max(1, int(wall_cfg.get("min_occupied_bands_for_clean_wall", 2))),
    )
    has_endpoint = np.asarray(raw_endpoint_occupied, dtype=bool) | np.asarray(terminal_wall_candidate, dtype=bool)
    structural_wall_candidate = (wall_score >= candidate_threshold) & has_endpoint & ~np.asarray(roomseg_free_raw, dtype=bool)
    structural_wall_clean = (
        (wall_score >= clean_threshold)
        & has_endpoint
        & (occ_band_count >= float(min_clean_bands))
        & ~np.asarray(roomseg_free_raw, dtype=bool)
    )
    if bool(wall_cfg.get("allow_wall_override_weak_free", False)):
        override = (wall_score >= clean_threshold) & np.asarray(line_support, dtype=bool) & (free_band_count <= 1)
        structural_wall_clean |= override
    splat_radius = max(0, int(ev_cfg.get("terminal_wall_splat_radius_cells", 0)))
    if splat_radius > 0:
        structural_wall_clean = dilate(structural_wall_clean, splat_radius) & ~np.asarray(roomseg_free_raw, dtype=bool)
    debug = {
        "candidate_score_threshold": float(candidate_threshold),
        "clean_score_threshold": float(clean_threshold),
        "min_occupied_bands_for_clean_wall": int(min_clean_bands),
        "score_max": float(np.max(wall_score)) if wall_score.size else 0.0,
        "score_mean_on_endpoints": float(np.mean(wall_score[has_endpoint])) if np.any(has_endpoint) else 0.0,
        "vertical_continuity_cells": int(np.count_nonzero(vertical_continuity > 0.0)),
        "terminal_support_cells": int(np.count_nonzero(terminal_support > 0.0)),
        "line_support_cells": int(np.count_nonzero(line_support)),
        "component_support_cells": int(np.count_nonzero(component_support)),
        "two_side_free_penalty_cells": int(np.count_nonzero(two_side_penalty)),
        "tiny_island_penalty_cells": int(np.count_nonzero(tiny_island)),
        "terminal_wall_splat_radius_cells": int(splat_radius),
    }
    return structural_wall_candidate.astype(bool), structural_wall_clean.astype(bool), debug


def _endpoint_component_support(
    endpoint: np.ndarray,
    *,
    resolution_m: float,
    min_component_area_m2: float,
    min_wall_component_area_m2: float,
    min_wall_line_length_m: float,
    tiny_island_area_m2: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    src = np.asarray(endpoint, dtype=bool)
    labels, count = label_components(src, 8)
    endpoint_area_ok = np.zeros_like(src, dtype=bool)
    wall_component_ok = np.zeros_like(src, dtype=bool)
    line_support = np.zeros_like(src, dtype=bool)
    tiny = np.zeros_like(src, dtype=bool)
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        metrics = component_metrics(comp, float(resolution_m))
        area_m2 = float(metrics.get("area_m2", 0.0))
        length_m = float(metrics.get("length_m", 0.0))
        elongation = float(metrics.get("elongation", 0.0))
        if area_m2 >= float(min_component_area_m2):
            endpoint_area_ok |= comp
        if area_m2 >= float(min_wall_component_area_m2):
            wall_component_ok |= comp
        if length_m >= float(min_wall_line_length_m) and elongation >= 2.0:
            line_support |= comp
        if area_m2 <= float(tiny_island_area_m2):
            tiny |= comp
    line_support |= _axis_run_support(src, min_cells=max(2, int(round(float(min_wall_line_length_m) / max(float(resolution_m), 1e-9)))))
    return {"endpoint_area_ok": endpoint_area_ok, "wall_component_ok": wall_component_ok}, line_support, tiny


def _axis_run_support(mask: np.ndarray, *, min_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    out = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    for r in range(h):
        c = 0
        while c < w:
            if not bool(src[r, c]):
                c += 1
                continue
            start = c
            while c < w and bool(src[r, c]):
                c += 1
            if c - start >= int(min_cells):
                out[r, start:c] = True
    for c in range(w):
        r = 0
        while r < h:
            if not bool(src[r, c]):
                r += 1
                continue
            start = r
            while r < h and bool(src[r, c]):
                r += 1
            if r - start >= int(min_cells):
                out[start:r, c] = True
    return out.astype(bool)


def _two_side_free_penalty(free: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(free, dtype=bool)
    radius = max(1, int(radius_cells))
    north = np.zeros_like(src, dtype=bool)
    south = np.zeros_like(src, dtype=bool)
    west = np.zeros_like(src, dtype=bool)
    east = np.zeros_like(src, dtype=bool)
    for offset in range(1, radius + 1):
        north[offset:, :] |= src[:-offset, :]
        south[:-offset, :] |= src[offset:, :]
        west[:, offset:] |= src[:, :-offset]
        east[:, :-offset] |= src[:, offset:]
    return ((north & south) | (west & east)).astype(bool)


def _near_count_at_least(counts: np.ndarray, threshold: int, *, radius_cells: int) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.uint32)
    radius = max(0, int(radius_cells))
    if radius <= 0:
        return arr >= int(threshold)
    size = 2 * radius + 1
    near = ndimage.maximum_filter(arr, size=size, mode="constant", cval=0)
    return near >= int(threshold)


def _strong_vertical_free(
    *,
    roomseg_free_raw: np.ndarray,
    free_count_by_band: np.ndarray | None,
    min_bands: int,
    min_count_sum: int,
) -> np.ndarray:
    raw = np.asarray(roomseg_free_raw, dtype=bool)
    if free_count_by_band is None:
        return raw.copy()
    counts = np.asarray(free_count_by_band, dtype=np.uint32)
    bands = np.count_nonzero(counts > 0, axis=0)
    summed = np.sum(counts, axis=0, dtype=np.uint32)
    return raw & (bands >= max(1, int(min_bands))) & (summed >= max(1, int(min_count_sum)))


def _filter_free_components(
    keep: np.ndarray,
    *,
    candidate: np.ndarray,
    nav_support: np.ndarray,
    strong_vertical_free: np.ndarray,
    min_area_cells: int,
    keep_largest_if_no_nav: bool,
    remove_not_touching_nav: bool,
) -> tuple[np.ndarray, list[dict]]:
    labels, count = label_components(keep, 4)
    out = np.zeros_like(keep, dtype=bool)
    debug: list[dict] = []
    largest_label = 0
    largest_area = 0
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        area = int(np.count_nonzero(comp))
        if area > largest_area:
            largest_area = area
            largest_label = idx
    any_nav_touch = bool(np.any(keep & nav_support))
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        area = int(np.count_nonzero(comp))
        touches_nav = bool(np.any(comp & nav_support))
        strong = bool(np.any(comp & strong_vertical_free))
        accepted = True
        reason = "keep"
        if area < int(min_area_cells) and not touches_nav:
            accepted = False
            reason = "remove_small_without_nav_support"
        if bool(remove_not_touching_nav) and not touches_nav and not strong:
            accepted = False
            reason = "remove_not_touching_nav_support"
        if not any_nav_touch and bool(keep_largest_if_no_nav) and idx == largest_label:
            accepted = True
            reason = "keep_largest_component_without_nav_support"
        if accepted:
            out |= comp
        debug.append(
            {
                "label": int(idx),
                "area_cells": int(area),
                "touches_nav_support": bool(touches_nav),
                "touches_strong_vertical_free": bool(strong),
                "accepted": bool(accepted),
                "reason": str(reason),
            }
        )
    if int(count) == 0 and bool(keep_largest_if_no_nav) and np.any(candidate):
        cand_labels, cand_count = label_components(candidate, 4)
        largest = _largest_label(cand_labels, cand_count)
        if largest > 0:
            out |= cand_labels == largest
            debug.append({"label": int(largest), "accepted": True, "reason": "bootstrap_largest_candidate"})
    return out.astype(bool), debug[:256]


def _snap_seed_to_free(mask: np.ndarray, robot_rc: tuple[int, int] | None, radius_cells: int) -> tuple[tuple[int, int] | None, dict]:
    free = np.asarray(mask, dtype=bool)
    if not np.any(free):
        return None, {"seed_found": False, "reason": "no_nav_free"}
    if robot_rc is None:
        labels, count = label_components(free, 8)
        largest = _largest_label(labels, count)
        rows, cols = np.nonzero(labels == largest)
        idx = int(len(rows) // 2)
        seed = (int(rows[idx]), int(cols[idx]))
        return seed, {"seed_found": True, "seed_source": "largest_nav_component", "seed_rc": [seed[0], seed[1]]}
    r, c = int(robot_rc[0]), int(robot_rc[1])
    if 0 <= r < free.shape[0] and 0 <= c < free.shape[1] and bool(free[r, c]):
        return (r, c), {"seed_found": True, "seed_source": "robot_rc", "seed_rc": [r, c], "snap_distance_cells": 0}
    rows, cols = np.nonzero(free)
    if rows.size == 0:
        return None, {"seed_found": False, "reason": "no_nav_free"}
    dist2 = (rows.astype(np.int64) - int(r)) ** 2 + (cols.astype(np.int64) - int(c)) ** 2
    best = int(np.argmin(dist2))
    best_dist = float(np.sqrt(float(dist2[best])))
    if best_dist <= float(max(0, int(radius_cells))):
        seed = (int(rows[best]), int(cols[best]))
        return seed, {
            "seed_found": True,
            "seed_source": "snapped_robot_rc",
            "seed_rc": [seed[0], seed[1]],
            "robot_rc": [int(r), int(c)],
            "snap_distance_cells": float(best_dist),
        }
    labels, count = label_components(free, 8)
    largest = _largest_label(labels, count)
    lrows, lcols = np.nonzero(labels == largest)
    idx = int(len(lrows) // 2)
    seed = (int(lrows[idx]), int(lcols[idx]))
    return seed, {
        "seed_found": True,
        "seed_source": "largest_nav_component_after_snap_failed",
        "seed_rc": [seed[0], seed[1]],
        "robot_rc": [int(r), int(c)],
        "snap_distance_cells": float(best_dist),
    }


def _flood_fill(mask: np.ndarray, seed: tuple[int, int] | None, *, connectivity: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    out = np.zeros_like(src, dtype=bool)
    if seed is None:
        return out
    sr, sc = int(seed[0]), int(seed[1])
    if not (0 <= sr < src.shape[0] and 0 <= sc < src.shape[1]) or not bool(src[sr, sc]):
        return out
    if int(connectivity) == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    queue: deque[tuple[int, int]] = deque([(sr, sc)])
    out[sr, sc] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in offsets:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < src.shape[0] and 0 <= cc < src.shape[1] and bool(src[rr, cc]) and not bool(out[rr, cc]):
                out[rr, cc] = True
                queue.append((rr, cc))
    return out


def _disagreement_map(
    *,
    roomseg_free_raw: np.ndarray,
    nav_reachable_free: np.ndarray,
    nav_obstacle: np.ndarray,
    resolution_m: float,
    config: Mapping[str, object],
) -> np.ndarray:
    nav_cfg = _section(config, "navigation_consistency")
    radius = _meters_to_cells(float(nav_cfg.get("nav_support_dilation_m", 0.15)), float(resolution_m))
    nav_support = dilate(nav_reachable_free, radius)
    out = np.zeros(np.asarray(roomseg_free_raw, dtype=bool).shape, dtype=np.uint8)
    out[np.asarray(roomseg_free_raw, dtype=bool) & ~nav_support] = 1
    out[np.asarray(nav_reachable_free, dtype=bool) & ~np.asarray(roomseg_free_raw, dtype=bool)] = 2
    out[np.asarray(nav_obstacle, dtype=bool) & np.asarray(roomseg_free_raw, dtype=bool)] = 3
    return out


def _root_config(config: Mapping[str, object] | object | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    out: dict[str, object] = {}
    for name in (
        "roomseg_evidence_v3",
        "navigation_consistency",
        "structural_wall_v3",
        "separator_v3",
        "topology_v3",
        "ray_valid_wall_inference",
    ):
        if hasattr(config, name):
            out[name] = getattr(config, name)
    return out


def _section(config: Mapping[str, object], name: str) -> dict[str, object]:
    raw = dict(config)
    section = raw.get(name)
    if isinstance(section, Mapping):
        return dict(section)
    return {}


def _depth_max_m(root_cfg: Mapping[str, object], ev_cfg: Mapping[str, object]) -> float:
    if "depth_max_m" in ev_cfg:
        return float(ev_cfg.get("depth_max_m", 3.0))
    ray_cfg = _section(root_cfg, "ray_valid_wall_inference")
    if "depth_max_m" in ray_cfg:
        return float(ray_cfg.get("depth_max_m", 3.0))
    depth_cfg = _section(root_cfg, "depth")
    if "depth_max_m" in depth_cfg:
        return float(depth_cfg.get("depth_max_m", 3.0))
    return 3.0


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("optional roomseg evidence mask has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _optional_uint(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.uint16)
    arr = np.asarray(value, dtype=np.uint32)
    if arr.shape != shape:
        raise ValueError("optional roomseg evidence count has shape %s, expected %s" % (arr.shape, shape))
    return np.minimum(arr, np.iinfo(np.uint16).max).astype(np.uint16)


def _optional_float(value: np.ndarray | None, shape: tuple[int, int], *, fill: float) -> np.ndarray:
    if value is None:
        return np.full(shape, float(fill), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError("optional roomseg evidence float array has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _meters_to_cells(distance_m: float, resolution_m: float) -> int:
    return max(0, int(round(float(distance_m) / max(float(resolution_m), 1e-9))))


def _largest_label(labels: np.ndarray, count: int) -> int:
    best = 0
    best_count = 0
    for idx in range(1, int(count) + 1):
        cells = int(np.count_nonzero(labels == idx))
        if cells > best_count:
            best = int(idx)
            best_count = int(cells)
    return int(best)


def _largest_component_cells(labels: np.ndarray, count: int) -> int:
    best = 0
    for idx in range(1, int(count) + 1):
        best = max(best, int(np.count_nonzero(labels == idx)))
    return int(best)


def _assert_invariants(
    *,
    roomseg_free_clean: np.ndarray,
    roomseg_unknown_clean: np.ndarray,
    structural_wall_clean: np.ndarray,
) -> None:
    if np.any(np.asarray(roomseg_free_clean, dtype=bool) & np.asarray(roomseg_unknown_clean, dtype=bool)):
        raise AssertionError("roomseg_free_clean overlaps roomseg_unknown_clean")
    if np.any(np.asarray(structural_wall_clean, dtype=bool) & np.asarray(roomseg_free_clean, dtype=bool)):
        raise AssertionError("structural_wall_clean overlaps roomseg_free_clean")
