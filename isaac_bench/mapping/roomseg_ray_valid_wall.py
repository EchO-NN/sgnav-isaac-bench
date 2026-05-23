from __future__ import annotations

from typing import Mapping

import numpy as np

from isaac_bench.mapping.roomseg_evidence_v3 import build_roomseg_evidence_v3
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


RAY_VALID_WALL_INFERENCE_MODE = "ray_valid_terminal_wall"
RAY_VALID_WALL_INFERENCE_MODE_V3 = "roomseg_evidence_line_closure_v3"


def build_ray_valid_wall_inference(
    *,
    vertical_free: np.ndarray,
    vertical_occupied: np.ndarray,
    vertical_observed: np.ndarray | None = None,
    terminal_wall_count: np.ndarray | None = None,
    terminal_wall_splat: np.ndarray | None = None,
    ray_covered_count: np.ndarray | None = None,
    terminal_wall_height_min: np.ndarray | None = None,
    terminal_wall_height_max: np.ndarray | None = None,
    terminal_wall_depth_min: np.ndarray | None = None,
    vertical_profile: VerticalProfileMap | None = None,
    grid_free: np.ndarray | None = None,
    grid_occupied: np.ndarray | None = None,
    grid_observed: np.ndarray | None = None,
    robot_rc: tuple[int, int] | None = None,
    resolution_m: float = 0.05,
    config: object | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the roomseg input from vertical-free and valid depth-ray endpoints.

    This intentionally does not consume navigation free/occupied masks. A cell
    becomes roomseg occupied only from vertical endpoint evidence or terminal
    wall evidence from a valid depth ray. Vertical-free wins over everything.
    """

    vf = np.asarray(vertical_free, dtype=bool)
    vo = np.asarray(vertical_occupied, dtype=bool)
    if vf.shape != vo.shape:
        raise ValueError("vertical_free and vertical_occupied must have the same HxW shape")
    shape = vf.shape
    root_cfg = _root_config(config)
    ev_cfg = _section(root_cfg, "roomseg_evidence_v3")
    force_legacy = bool(ev_cfg.get("force_legacy_ray_valid_wall", False))
    strict = bool(ev_cfg.get("strict_benchmark", root_cfg.get("strict_benchmark", False)))
    if force_legacy and strict:
        raise ValueError("strict roomseg_evidence_v3 forbids force_legacy_ray_valid_wall=true")
    if not force_legacy and bool(ev_cfg.get("enabled", True)):
        return build_ray_valid_wall_inference_v3(
            vertical_free=vf,
            vertical_occupied=vo,
            vertical_observed=vertical_observed,
            terminal_wall_count=terminal_wall_count,
            terminal_wall_splat=terminal_wall_splat,
            ray_covered_count=ray_covered_count,
            terminal_wall_height_min=terminal_wall_height_min,
            terminal_wall_height_max=terminal_wall_height_max,
            terminal_wall_depth_min=terminal_wall_depth_min,
            vertical_profile=vertical_profile,
            grid_free=grid_free,
            grid_occupied=grid_occupied,
            grid_observed=grid_observed,
            robot_rc=robot_rc,
            resolution_m=float(resolution_m),
            config=root_cfg,
        )
    cfg = _config_dict(config)
    enabled = bool(cfg.get("enabled", True))
    min_terminal_wall_count = max(1, int(cfg.get("min_terminal_wall_count", 1)))
    splat_radius = max(0, int(cfg.get("terminal_wall_splat_radius_cells", 1)))
    require_no_vertical_free = bool(cfg.get("require_no_vertical_free", True))

    tw_count = _optional_uint(terminal_wall_count, shape)
    if terminal_wall_splat is None:
        terminal = tw_count >= min_terminal_wall_count
        tw_splat = _dilate_binary(terminal, splat_radius)
    else:
        tw_splat = np.asarray(terminal_wall_splat, dtype=bool)
        if tw_splat.shape != shape:
            raise ValueError("terminal_wall_splat has shape %s, expected %s" % (tw_splat.shape, shape))
        if np.any(tw_count):
            tw_splat |= _dilate_binary(tw_count >= min_terminal_wall_count, splat_radius)

    if require_no_vertical_free:
        tw_splat &= ~vf
    if not enabled:
        tw_splat = np.zeros(shape, dtype=bool)

    occupied_before_ray_wall = vo & ~vf
    unknown_before_ray_wall = ~(vf | occupied_before_ray_wall)
    ray_valid_wall = (occupied_before_ray_wall | tw_splat) & ~vf
    initial_free = vf.copy()
    initial_occupied = ray_valid_wall.copy()
    initial_unknown = ~(initial_free | initial_occupied)
    vertical_free_overridden = initial_free & initial_occupied
    if np.any(vertical_free_overridden):
        raise AssertionError("ray-valid roomseg wall inference overrode vertical-free cells")
    unknown_removed_by_ray_wall = unknown_before_ray_wall & initial_occupied
    if np.any(initial_free & ray_valid_wall):
        raise AssertionError("ray-valid wall mask overlaps vertical-free cells")

    observed = _optional_bool(vertical_observed, shape) | initial_free | initial_occupied
    ray_covered = _optional_uint(ray_covered_count, shape)
    hmin = _optional_float(terminal_wall_height_min, shape, fill=np.inf)
    hmax = _optional_float(terminal_wall_height_max, shape, fill=-np.inf)
    dmin = _optional_float(terminal_wall_depth_min, shape, fill=np.inf)

    debug = {
        "ray_valid_wall_inference_enabled": bool(enabled),
        "ray_valid_wall_inference_mode": str(cfg.get("mode", RAY_VALID_WALL_INFERENCE_MODE) or RAY_VALID_WALL_INFERENCE_MODE),
        "depth_max_m": float(cfg.get("depth_max_m", 3.0)),
        "vertical_profile_free_min_height_m": float(cfg.get("min_endpoint_height_m", 0.10)),
        "vertical_profile_free_max_height_m": float(cfg.get("max_endpoint_height_m", 2.50)),
        "min_terminal_wall_count": int(min_terminal_wall_count),
        "terminal_wall_splat_radius_cells": int(splat_radius),
        "require_no_vertical_free": bool(require_no_vertical_free),
        "strict_no_navigation_obstacle_overlay": bool(cfg.get("strict_no_navigation_obstacle_overlay", True)),
        "strict_no_navigation_free_overlay": bool(cfg.get("strict_no_navigation_free_overlay", True)),
        "vertical_free_cells": int(np.count_nonzero(vf)),
        "vertical_occupied_cells": int(np.count_nonzero(vo)),
        "vertical_observed_cells": int(np.count_nonzero(observed)),
        "roomseg_ray_covered_cells": int(np.count_nonzero(ray_covered)),
        "roomseg_ray_covered_count_sum": int(np.sum(ray_covered, dtype=np.uint64)),
        "terminal_wall_cells": int(np.count_nonzero(tw_count >= min_terminal_wall_count)),
        "terminal_wall_count_sum": int(np.sum(tw_count, dtype=np.uint64)),
        "terminal_wall_splat_cells": int(np.count_nonzero(tw_splat)),
        "unknown_before_cells": int(np.count_nonzero(unknown_before_ray_wall)),
        "unknown_after_cells": int(np.count_nonzero(initial_unknown)),
        "unknown_removed_by_ray_wall_cells": int(np.count_nonzero(unknown_removed_by_ray_wall)),
        "vertical_free_overridden_by_wall_cells": int(np.count_nonzero(vertical_free_overridden)),
    }
    return {
        "initial_roomseg_free": initial_free.astype(bool),
        "initial_roomseg_occupied": initial_occupied.astype(bool),
        "initial_roomseg_unknown": initial_unknown.astype(bool),
        "vertical_observed_map": observed.astype(bool),
        "vertical_occupied_0p1_2p5": vo.astype(bool),
        "vertical_observed_0p1_2p5": observed.astype(bool),
        "vertical_occupied_0p2_2p0": vo.astype(bool),
        "vertical_observed_0p2_2p0": observed.astype(bool),
        "roomseg_ray_covered_count": ray_covered.astype(np.uint16),
        "roomseg_terminal_wall_count": tw_count.astype(np.uint16),
        "roomseg_terminal_wall_height_min": hmin.astype(np.float32),
        "roomseg_terminal_wall_height_max": hmax.astype(np.float32),
        "roomseg_terminal_wall_depth_min": dmin.astype(np.float32),
        "roomseg_terminal_wall_splat": tw_splat.astype(bool),
        "ray_valid_wall_inference": ray_valid_wall.astype(bool),
        "initial_roomseg_free_after_ray_wall": initial_free.astype(bool),
        "initial_roomseg_occupied_after_ray_wall": initial_occupied.astype(bool),
        "initial_roomseg_unknown_after_ray_wall": initial_unknown.astype(bool),
        "unknown_before_ray_wall": unknown_before_ray_wall.astype(bool),
        "unknown_after_ray_wall": initial_unknown.astype(bool),
        "unknown_removed_by_ray_wall": unknown_removed_by_ray_wall.astype(bool),
        "debug": debug,
    }


def build_ray_valid_wall_inference_v3(
    *,
    vertical_free: np.ndarray,
    vertical_occupied: np.ndarray,
    vertical_observed: np.ndarray | None = None,
    terminal_wall_count: np.ndarray | None = None,
    terminal_wall_splat: np.ndarray | None = None,
    ray_covered_count: np.ndarray | None = None,
    terminal_wall_height_min: np.ndarray | None = None,
    terminal_wall_height_max: np.ndarray | None = None,
    terminal_wall_depth_min: np.ndarray | None = None,
    vertical_profile: VerticalProfileMap | None = None,
    grid_free: np.ndarray | None = None,
    grid_occupied: np.ndarray | None = None,
    grid_observed: np.ndarray | None = None,
    robot_rc: tuple[int, int] | None = None,
    resolution_m: float = 0.05,
    config: object | Mapping[str, object] | None = None,
) -> dict[str, object]:
    vf = np.asarray(vertical_free, dtype=bool)
    vo = np.asarray(vertical_occupied, dtype=bool)
    if vf.shape != vo.shape:
        raise ValueError("vertical_free and vertical_occupied must have the same HxW shape")
    shape = vf.shape
    observed = _optional_bool(vertical_observed, shape) | vf | vo
    ray_covered = _optional_uint(ray_covered_count, shape)
    observed |= ray_covered > 0
    vp = vertical_profile if vertical_profile is not None else _synthetic_vertical_profile(vf, vo, observed)
    nav_free = vf.copy() if grid_free is None else _optional_bool(grid_free, shape)
    nav_occupied = vo.copy() if grid_occupied is None else _optional_bool(grid_occupied, shape)
    nav_observed = observed.copy() if grid_observed is None else _optional_bool(grid_observed, shape)
    seed = robot_rc if robot_rc is not None else _first_true(nav_free)

    evidence = build_roomseg_evidence_v3(
        vertical_profile=vp,
        grid_free=nav_free,
        grid_occupied=nav_occupied,
        grid_observed=nav_observed,
        roomseg_ray_covered_count=ray_covered,
        roomseg_terminal_wall_count=terminal_wall_count,
        roomseg_terminal_wall_splat=terminal_wall_splat,
        roomseg_terminal_wall_height_min=terminal_wall_height_min,
        roomseg_terminal_wall_height_max=terminal_wall_height_max,
        roomseg_terminal_wall_depth_min=terminal_wall_depth_min,
        robot_rc=seed,
        resolution_m=float(resolution_m),
        config=config,
    )
    tw_count = _optional_uint(terminal_wall_count, shape)
    hmin = _optional_float(terminal_wall_height_min, shape, fill=np.inf)
    hmax = _optional_float(terminal_wall_height_max, shape, fill=-np.inf)
    dmin = _optional_float(terminal_wall_depth_min, shape, fill=np.inf)
    tw_splat = _optional_bool(terminal_wall_splat, shape) | evidence.terminal_wall_candidate
    initial_free = evidence.roomseg_free_clean.astype(bool)
    initial_occupied = evidence.structural_wall_clean.astype(bool)
    initial_unknown = evidence.roomseg_unknown_clean.astype(bool)
    unknown_before = evidence.roomseg_unknown_raw.astype(bool)
    unknown_removed_by_wall = unknown_before & initial_occupied
    debug = {
        **dict(evidence.debug),
        "ray_valid_wall_inference_enabled": True,
        "ray_valid_wall_inference_mode": RAY_VALID_WALL_INFERENCE_MODE_V3,
        "force_legacy_ray_valid_wall": False,
        "vertical_free_cells": int(np.count_nonzero(vf)),
        "vertical_occupied_cells": int(np.count_nonzero(vo)),
        "vertical_observed_cells": int(np.count_nonzero(observed)),
        "roomseg_ray_covered_cells": int(np.count_nonzero(ray_covered)),
        "roomseg_ray_covered_count_sum": int(np.sum(ray_covered, dtype=np.uint64)),
        "terminal_wall_cells": int(np.count_nonzero(tw_count)),
        "terminal_wall_count_sum": int(np.sum(tw_count, dtype=np.uint64)),
        "terminal_wall_splat_cells": int(np.count_nonzero(tw_splat)),
        "unknown_before_cells": int(np.count_nonzero(unknown_before)),
        "unknown_after_cells": int(np.count_nonzero(initial_unknown)),
        "unknown_removed_by_ray_wall_cells": int(np.count_nonzero(unknown_removed_by_wall)),
        "vertical_free_overridden_by_wall_cells": int(np.count_nonzero(initial_free & initial_occupied)),
    }
    return {
        "initial_roomseg_free": initial_free.astype(bool),
        "initial_roomseg_occupied": initial_occupied.astype(bool),
        "initial_roomseg_unknown": initial_unknown.astype(bool),
        "vertical_observed_map": (~initial_unknown).astype(bool),
        "vertical_occupied_0p1_2p5": evidence.structural_wall_clean.astype(bool),
        "vertical_observed_0p1_2p5": (~evidence.roomseg_unknown_raw).astype(bool),
        "vertical_occupied_0p2_2p0": evidence.structural_wall_clean.astype(bool),
        "vertical_observed_0p2_2p0": (~evidence.roomseg_unknown_raw).astype(bool),
        "roomseg_free_raw": evidence.roomseg_free_raw.astype(bool),
        "roomseg_free_clean": evidence.roomseg_free_clean.astype(bool),
        "roomseg_unknown_raw": evidence.roomseg_unknown_raw.astype(bool),
        "roomseg_unknown_clean": evidence.roomseg_unknown_clean.astype(bool),
        "raw_endpoint_occupied": evidence.raw_endpoint_occupied.astype(bool),
        "terminal_wall_candidate": evidence.terminal_wall_candidate.astype(bool),
        "structural_wall_candidate": evidence.structural_wall_candidate.astype(bool),
        "structural_wall_clean": evidence.structural_wall_clean.astype(bool),
        "navigation_reachable_support": evidence.navigation_reachable_support.astype(bool),
        "disagreement_map": evidence.disagreement_map.astype(np.uint8),
        "roomseg_ray_covered_count": ray_covered.astype(np.uint16),
        "roomseg_terminal_wall_count": tw_count.astype(np.uint16),
        "roomseg_terminal_wall_height_min": hmin.astype(np.float32),
        "roomseg_terminal_wall_height_max": hmax.astype(np.float32),
        "roomseg_terminal_wall_depth_min": dmin.astype(np.float32),
        "roomseg_terminal_wall_splat": tw_splat.astype(bool),
        "ray_valid_wall_inference": initial_occupied.astype(bool),
        "initial_roomseg_free_after_ray_wall": initial_free.astype(bool),
        "initial_roomseg_occupied_after_ray_wall": initial_occupied.astype(bool),
        "initial_roomseg_unknown_after_ray_wall": initial_unknown.astype(bool),
        "unknown_before_ray_wall": unknown_before.astype(bool),
        "unknown_after_ray_wall": initial_unknown.astype(bool),
        "unknown_removed_by_ray_wall": unknown_removed_by_wall.astype(bool),
        "debug": debug,
    }


def _config_dict(config: object | Mapping[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        raw = dict(config)
        nested = raw.get("ray_valid_wall_inference")
        return dict(nested or raw)
    if hasattr(config, "ray_valid_wall_inference"):
        return dict(getattr(config, "ray_valid_wall_inference") or {})
    return {
        "enabled": getattr(config, "ray_valid_wall_inference_enabled", True),
        "mode": getattr(config, "ray_valid_wall_inference_mode", RAY_VALID_WALL_INFERENCE_MODE),
        "depth_max_m": getattr(config, "ray_valid_wall_depth_max_m", 3.0),
        "min_endpoint_height_m": getattr(config, "ray_valid_wall_min_endpoint_height_m", 0.10),
        "max_endpoint_height_m": getattr(config, "ray_valid_wall_max_endpoint_height_m", 2.50),
        "min_terminal_wall_count": getattr(config, "ray_valid_wall_min_terminal_wall_count", 1),
        "terminal_wall_splat_radius_cells": getattr(config, "ray_valid_wall_terminal_wall_splat_radius_cells", 1),
        "require_no_vertical_free": getattr(config, "ray_valid_wall_require_no_vertical_free", True),
        "mark_ray_covered_debug": getattr(config, "ray_valid_wall_mark_ray_covered_debug", True),
        "decouple_roomseg_rays_from_nav_clear": getattr(config, "ray_valid_wall_decouple_roomseg_rays_from_nav_clear", True),
        "strict_no_navigation_obstacle_overlay": getattr(config, "ray_valid_wall_strict_no_navigation_obstacle_overlay", True),
        "strict_no_navigation_free_overlay": getattr(config, "ray_valid_wall_strict_no_navigation_free_overlay", True),
    }


def _root_config(config: object | Mapping[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    out: dict[str, object] = {}
    for name in ("roomseg_evidence_v3", "navigation_consistency", "structural_wall_v3", "ray_valid_wall_inference"):
        if hasattr(config, name):
            out[name] = getattr(config, name)
    return out


def _section(config: Mapping[str, object], name: str) -> dict[str, object]:
    section = dict(config).get(name)
    return dict(section) if isinstance(section, Mapping) else {}


def _synthetic_vertical_profile(vertical_free: np.ndarray, vertical_occupied: np.ndarray, observed: np.ndarray) -> VerticalProfileMap:
    vf = np.asarray(vertical_free, dtype=bool)
    vo = np.asarray(vertical_occupied, dtype=bool)
    obs = np.asarray(observed, dtype=bool)
    bands = 4
    occ_count = np.zeros((bands, *vf.shape), dtype=np.uint16)
    free_count = np.zeros_like(occ_count)
    observed_count = np.zeros_like(occ_count)
    unknown_count = np.ones_like(occ_count)
    free_count[:, vf] = 1
    occ_count[:, vo] = 1
    observed_count[:, obs] = 1
    observed_count[:, vf | vo] = 1
    unknown_count[observed_count > 0] = 0
    return VerticalProfileMap.from_counts(
        occupied_count=occ_count,
        free_ray_count=free_count,
        observed_count=observed_count,
        unknown_count=unknown_count,
    )


def _first_true(mask: np.ndarray) -> tuple[int, int] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    return int(rows[0]), int(cols[0])


def _optional_bool(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("optional roomseg ray mask has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _optional_uint(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.uint16)
    arr = np.asarray(value, dtype=np.uint32)
    if arr.shape != shape:
        raise ValueError("optional roomseg ray count has shape %s, expected %s" % (arr.shape, shape))
    return np.minimum(arr, np.iinfo(np.uint16).max).astype(np.uint16)


def _optional_float(value: np.ndarray | None, shape: tuple[int, int], *, fill: float) -> np.ndarray:
    if value is None:
        return np.full(shape, float(fill), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError("optional roomseg ray float array has shape %s, expected %s" % (arr.shape, shape))
    return arr


def _dilate_binary(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = int(max(0, radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    out = src.copy()
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius:
                offsets.append((dr, dc))
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out
