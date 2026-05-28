from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import time

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import conn, label_components
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_CONFLICT,
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
)


@dataclass
class VoxelRoomsegEvidenceConfig:
    enabled: bool = True
    active_z_min_m: float = 0.10
    active_z_max_mode: str = "ceiling_90pct"
    active_z_max_ceiling_ratio: float = 0.90
    active_z_max_fallback_m: float = 2.00
    min_free_z_cells_for_xy_free: int = 3
    unknown_ratio_min_for_xy_unknown: float = 0.50
    min_observed_z_cells_for_known_column: int = 3
    wall_mode: str = "free_unknown_then_ratio_wall"
    wall_occupied_ratio_min_for_xy_wall: float = 0.90
    wall_min_occupied_z_cells_for_xy_wall: int = 3
    free_priority_over_wall: bool = True
    unknown_priority_over_wall: bool = True
    nav_assisted_free_enabled: bool = True
    nav_assisted_free_requires_observed: bool = False
    nav_assisted_free_forbidden_on_ratio_wall: bool = True
    wall_line_support_enabled: bool = True
    wall_line_support_min_occupied_z_cells: int = 1
    wall_line_support_free_exclusion_z_cells: int = 3
    wall_line_support_unknown_ratio_max: float = 0.75
    wall_line_support_unknown_ratio_hard_max: float = 0.90
    wall_line_support_min_observed_z_cells: int = 1
    wall_line_support_use_nav_edge_gate: bool = True
    wall_line_support_nav_edge_radius_cells: int = 3
    wall_line_support_free_boundary_radius_cells: int = 3
    wall_line_support_allow_free_conflict_for_projection: bool = True
    wall_line_support_conflict_weight: float = 0.35
    wall_line_support_strong_weight: float = 1.0
    wall_line_support_remove_small_area_cells: int = 3
    # Legacy/debug knobs retained for older configs; they no longer define final wall.
    wall_min_occupied_z_cells: int = 1
    wall_free_exclusion_min_z_cells: int = 3
    wall_unknown_gating_enabled: bool = True
    wall_unknown_ratio_max_for_structural: float = 0.75
    wall_min_observed_z_cells_for_structural: int = 3
    allow_unknown_dominant_wall_as_door_anchor: bool = False
    show_unknown_rejected_wall_support: bool = False
    wall_ratio_debug_enabled: bool = True
    wall_occupied_ratio_debug_threshold: float = 0.90
    projection_priority: str = "free_unknown_then_ratio_wall"
    preserve_unknown: bool = True
    fill_unknown_as_wall: bool = False
    fill_unknown_as_free: bool = False
    suppress_free_on_navigation_obstacle: bool = False
    promote_navigation_obstacle_to_wall: bool = False
    fill_small_unknown_holes_inside_vertical_free: bool = True
    small_unknown_hole_max_area_cells: int = 16
    small_unknown_hole_min_free_neighbor_ratio: float = 0.75
    free_remove_island_max_area_cells: int = 0
    wall_remove_island_max_area_cells: int = 0
    wall_micro_close_radius_cells: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VoxelRoomsegEvidenceConfig":
        if isinstance(data, cls):
            cfg = data
            for key, value in overrides.items():
                if value is not None and hasattr(cfg, key):
                    setattr(cfg, key, value)
            return cfg
        raw = dict(data or {})
        if "wall_occupied_ratio_min" in raw and "wall_occupied_ratio_min_for_xy_wall" not in raw:
            raw["wall_occupied_ratio_min_for_xy_wall"] = raw["wall_occupied_ratio_min"]
        if "wall_occupied_ratio_min" in raw and "wall_occupied_ratio_debug_threshold" not in raw:
            raw["wall_occupied_ratio_debug_threshold"] = raw["wall_occupied_ratio_min"]
        if "wall_unknown_ratio_max_for_structural" in raw and "unknown_ratio_min_for_xy_unknown" not in raw:
            raw["unknown_ratio_min_for_xy_unknown"] = raw["wall_unknown_ratio_max_for_structural"]
        if "wall_min_observed_z_cells_for_structural" in raw and "min_observed_z_cells_for_known_column" not in raw:
            raw["min_observed_z_cells_for_known_column"] = raw["wall_min_observed_z_cells_for_structural"]
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VoxelRoomsegEvidence:
    vertical_free_xy: np.ndarray
    wall_xy: np.ndarray
    unknown_xy: np.ndarray
    active_observed_xy: np.ndarray
    active_free_count_xy: np.ndarray
    active_occupied_count_xy: np.ndarray
    active_unknown_count_xy: np.ndarray
    active_observed_count_xy: np.ndarray
    active_z_bin_count_xy: np.ndarray
    occupied_any_xy: np.ndarray
    raw_occupied_wall_support_xy: np.ndarray
    strict_raw_wall_xy: np.ndarray
    wall_suppressed_by_free_xy: np.ndarray
    occupied_ratio_active_xy: np.ndarray
    unknown_ratio_active_xy: np.ndarray
    observed_ratio_active_xy: np.ndarray
    unknown_dominant_xy: np.ndarray
    wall_support_loose_xy: np.ndarray
    wall_support_unknown_gated_xy: np.ndarray
    wall_support_rejected_unknown_xy: np.ndarray
    structural_wall_seed_xy: np.ndarray
    structural_wall_ratio_xy: np.ndarray
    wall_rejected_by_free_xy: np.ndarray
    wall_rejected_by_unknown_xy: np.ndarray
    nonstructural_occupied_xy: np.ndarray
    small_unknown_hole_filled_xy: np.ndarray
    wall_line_support_xy: np.ndarray
    wall_line_support_raw_xy: np.ndarray
    wall_line_support_strong_xy: np.ndarray
    wall_line_support_conflict_xy: np.ndarray
    wall_line_support_near_free_boundary_xy: np.ndarray
    wall_line_support_rejected_furniture_xy: np.ndarray
    wall_line_support_rejected_unknown_xy: np.ndarray
    wall_line_support_weight_xy: np.ndarray
    wall_line_support_rejected_by_free_xy: np.ndarray
    wall_line_support_rejected_by_unknown_xy: np.ndarray
    wall_line_support_rejected_by_observed_xy: np.ndarray
    wall_line_support_rejected_by_nav_edge_xy: np.ndarray
    ratio_wall_debug_xy: np.ndarray
    free_wall_conflict_xy: np.ndarray
    debug: dict[str, object] = field(default_factory=dict)


def classify_voxel_columns_for_roomseg(
    *,
    state_active: np.ndarray,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    cfg: VoxelRoomsegEvidenceConfig,
    frontier_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    state = np.asarray(state_active, dtype=np.uint8)
    if state.ndim != 3:
        raise ValueError("state_active must have shape [Z, H, W]")
    shape = tuple(state.shape[1:])
    nav_free = np.asarray(navigation_free_mask, dtype=bool)
    nav_obstacle = np.asarray(navigation_obstacle_mask, dtype=bool)
    if nav_free.shape != shape or nav_obstacle.shape != shape:
        raise ValueError("navigation masks must match state_active HxW shape")
    active_count = max(1, int(state.shape[0]))
    free_count = np.sum(state == int(VOXEL_FREE), axis=0).astype(np.uint16)
    occupied_count = np.sum(state == int(VOXEL_OCCUPIED), axis=0).astype(np.uint16)
    unknown_count = np.sum(state == int(VOXEL_UNKNOWN), axis=0).astype(np.uint16)
    conflict_count = np.sum(state == int(VOXEL_CONFLICT), axis=0).astype(np.uint16)
    observed_count = np.minimum(
        free_count.astype(np.uint32) + occupied_count.astype(np.uint32) + conflict_count.astype(np.uint32),
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    active_z_bin_count = np.full(shape, active_count, dtype=np.uint16)
    free_ratio = free_count.astype(np.float32) / float(active_count)
    occupied_ratio = occupied_count.astype(np.float32) / float(active_count)
    unknown_ratio = unknown_count.astype(np.float32) / float(active_count)
    observed_ratio = observed_count.astype(np.float32) / float(active_count)

    free_raw = free_count >= max(1, int(cfg.min_free_z_cells_for_xy_free))
    wall_ratio_raw = (
        occupied_ratio >= float(cfg.wall_occupied_ratio_min_for_xy_wall)
    ) & (
        occupied_count >= max(1, int(cfg.wall_min_occupied_z_cells_for_xy_wall))
    )
    nav_assisted_free = np.zeros(shape, dtype=bool)
    if bool(cfg.nav_assisted_free_enabled):
        nav_assisted_free = nav_free.copy()
        if bool(cfg.nav_assisted_free_requires_observed):
            nav_assisted_free &= observed_count > 0
        if bool(cfg.nav_assisted_free_forbidden_on_ratio_wall):
            nav_assisted_free &= ~wall_ratio_raw
    vertical_free = free_raw | nav_assisted_free
    if bool(cfg.suppress_free_on_navigation_obstacle):
        vertical_free &= ~nav_obstacle

    unknown_dominant = (
        unknown_ratio >= float(cfg.unknown_ratio_min_for_xy_unknown)
    ) | (
        observed_count < max(1, int(cfg.min_observed_z_cells_for_known_column))
    )
    if bool(cfg.free_priority_over_wall):
        wall_rejected_by_free = wall_ratio_raw & vertical_free
    else:
        wall_rejected_by_free = np.zeros(shape, dtype=bool)
    wall_free_block = vertical_free if bool(cfg.free_priority_over_wall) else np.zeros(shape, dtype=bool)
    if bool(cfg.unknown_priority_over_wall):
        wall_rejected_by_unknown = wall_ratio_raw & ~wall_free_block & unknown_dominant
        wall_unknown_block = unknown_dominant
    else:
        wall_rejected_by_unknown = np.zeros(shape, dtype=bool)
        wall_unknown_block = np.zeros(shape, dtype=bool)
    wall = wall_ratio_raw & ~wall_free_block & ~wall_unknown_block
    promoted_nav_obstacle = np.zeros(shape, dtype=bool)
    if bool(cfg.promote_navigation_obstacle_to_wall):
        promoted_nav_obstacle = nav_obstacle & ~vertical_free & ~unknown_dominant & ~wall
        wall |= promoted_nav_obstacle

    unknown = ~(vertical_free | wall)
    occupied_any = occupied_count > 0
    wall_support_loose = occupied_any.copy()
    wall_support_unknown_gated = wall_ratio_raw & ~unknown_dominant
    wall_support_rejected_unknown = wall_rejected_by_unknown.copy()
    nonstructural_occupied = occupied_any & ~wall & ~vertical_free
    ratio_wall_debug = (
        occupied_ratio >= float(cfg.wall_occupied_ratio_debug_threshold)
    ) if bool(cfg.wall_ratio_debug_enabled) else np.zeros(shape, dtype=bool)
    wall_line_support_raw = occupied_count >= max(1, int(cfg.wall_line_support_min_occupied_z_cells))
    wall_line_support_free_block = free_count >= max(1, int(cfg.min_free_z_cells_for_xy_free))
    wall_line_support_unknown_block = unknown_ratio >= float(cfg.wall_line_support_unknown_ratio_hard_max)
    wall_line_support_observed_block = observed_count < max(1, int(cfg.wall_line_support_min_observed_z_cells))
    free_boundary = vertical_free & ndimage.binary_dilation(~vertical_free, structure=conn(8)).astype(bool)
    boundary_radius = max(0, int(getattr(cfg, "wall_line_support_free_boundary_radius_cells", 3)))
    boundary_structure = _disk(boundary_radius) if boundary_radius > 0 else conn(8)
    near_free_boundary = ndimage.binary_dilation(free_boundary, structure=boundary_structure).astype(bool)
    structural_boundary_seed = wall_ratio_raw | wall_rejected_by_free | wall_rejected_by_unknown
    if np.any(structural_boundary_seed):
        near_free_boundary |= ndimage.binary_dilation(structural_boundary_seed, structure=boundary_structure).astype(bool)
    support_raw_known = wall_line_support_raw & ~wall_line_support_unknown_block & ~wall_line_support_observed_block
    wall_line_support_near_free_boundary = support_raw_known & near_free_boundary
    wall_line_support_conflict = wall_line_support_near_free_boundary & wall_line_support_free_block
    if not bool(getattr(cfg, "wall_line_support_allow_free_conflict_for_projection", True)):
        wall_line_support_conflict = np.zeros(shape, dtype=bool)
    wall_line_support_strong = wall_line_support_near_free_boundary & ~wall_line_support_free_block
    wall_line_support_rejected_furniture = support_raw_known & ~near_free_boundary
    wall_line_support_rejected_unknown = wall_line_support_raw & wall_line_support_unknown_block
    wall_line_support_nav_edge_block = wall_line_support_rejected_furniture.copy()
    wall_line_support = wall_line_support_strong | wall_line_support_conflict
    if not bool(cfg.wall_line_support_enabled):
        wall_line_support = np.zeros(shape, dtype=bool)
        wall_line_support_strong = np.zeros(shape, dtype=bool)
        wall_line_support_conflict = np.zeros(shape, dtype=bool)
    wall_line_support = _remove_small_true_components(
        wall_line_support,
        max_area_cells=int(cfg.wall_line_support_remove_small_area_cells),
        connectivity=8,
        remove_leq=True,
    )
    wall_line_support_strong &= wall_line_support
    wall_line_support_conflict &= wall_line_support
    wall_line_support_weight = np.zeros(shape, dtype=np.float32)
    wall_line_support_weight[wall_line_support_strong] = float(getattr(cfg, "wall_line_support_strong_weight", 1.0))
    wall_line_support_weight[wall_line_support_conflict] = float(getattr(cfg, "wall_line_support_conflict_weight", 0.35))

    small_hole_filled = np.zeros(shape, dtype=bool)
    if bool(cfg.fill_small_unknown_holes_inside_vertical_free):
        frontier = None if frontier_mask is None else np.asarray(frontier_mask, dtype=bool)
        if frontier is not None and frontier.shape != shape:
            raise ValueError("frontier_mask must match state_active HxW shape")
        vertical_free, unknown, small_hole_filled = fill_small_nonstructural_unknown_holes(
            vertical_free,
            unknown,
            wall,
            max_area_cells=int(cfg.small_unknown_hole_max_area_cells),
            min_free_neighbor_ratio=float(cfg.small_unknown_hole_min_free_neighbor_ratio),
            frontier_mask=frontier,
        )
    if np.any(vertical_free & wall) or np.any(vertical_free & unknown) or np.any(wall & unknown):
        raise AssertionError("voxel roomseg v19 maps must be mutually exclusive")
    if not np.all(vertical_free | wall | unknown):
        raise AssertionError("voxel roomseg v19 maps must cover all cells")

    return {
        "free_count": free_count.astype(np.uint16),
        "occupied_count": occupied_count.astype(np.uint16),
        "unknown_count": unknown_count.astype(np.uint16),
        "conflict_count": conflict_count.astype(np.uint16),
        "observed_count": observed_count.astype(np.uint16),
        "active_z_bin_count": active_z_bin_count.astype(np.uint16),
        "free_ratio": free_ratio.astype(np.float32),
        "occupied_ratio": occupied_ratio.astype(np.float32),
        "unknown_ratio": unknown_ratio.astype(np.float32),
        "observed_ratio": observed_ratio.astype(np.float32),
        "free_raw": free_raw.astype(bool),
        "nav_assisted_free": nav_assisted_free.astype(bool),
        "vertical_free": vertical_free.astype(bool),
        "unknown_dominant": unknown_dominant.astype(bool),
        "wall_ratio_raw": wall_ratio_raw.astype(bool),
        "wall": wall.astype(bool),
        "unknown": unknown.astype(bool),
        "wall_rejected_by_free": wall_rejected_by_free.astype(bool),
        "wall_rejected_by_unknown": wall_rejected_by_unknown.astype(bool),
        "occupied_any": occupied_any.astype(bool),
        "raw_occupied_wall_support": occupied_any.astype(bool),
        "wall_support_loose": wall_support_loose.astype(bool),
        "wall_support_unknown_gated": wall_support_unknown_gated.astype(bool),
        "wall_support_rejected_unknown": wall_support_rejected_unknown.astype(bool),
        "nonstructural_occupied": nonstructural_occupied.astype(bool),
        "ratio_wall_debug": ratio_wall_debug.astype(bool),
        "small_unknown_hole_filled": small_hole_filled.astype(bool),
        "wall_line_support": wall_line_support.astype(bool),
        "wall_line_support_raw": wall_line_support_raw.astype(bool),
        "wall_line_support_strong": wall_line_support_strong.astype(bool),
        "wall_line_support_conflict": wall_line_support_conflict.astype(bool),
        "wall_line_support_near_free_boundary": wall_line_support_near_free_boundary.astype(bool),
        "wall_line_support_rejected_furniture": wall_line_support_rejected_furniture.astype(bool),
        "wall_line_support_rejected_unknown": wall_line_support_rejected_unknown.astype(bool),
        "wall_line_support_weight": wall_line_support_weight.astype(np.float32),
        "wall_line_support_rejected_by_free": wall_line_support_conflict.astype(bool),
        "wall_line_support_rejected_by_unknown": wall_line_support_rejected_unknown.astype(bool),
        "wall_line_support_rejected_by_observed": (wall_line_support_raw & ~wall_line_support_free_block & ~wall_line_support_unknown_block & wall_line_support_observed_block).astype(bool),
        "wall_line_support_rejected_by_nav_edge": wall_line_support_nav_edge_block.astype(bool),
        "promoted_nav_obstacle": promoted_nav_obstacle.astype(bool),
    }


def fill_small_nonstructural_unknown_holes(
    vertical_free: np.ndarray,
    unknown: np.ndarray,
    wall: np.ndarray,
    *,
    max_area_cells: int,
    min_free_neighbor_ratio: float,
    frontier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    free = np.asarray(vertical_free, dtype=bool).copy()
    unk = np.asarray(unknown, dtype=bool).copy()
    wall_mask = np.asarray(wall, dtype=bool)
    if int(max_area_cells) <= 0 or not np.any(unk):
        return free, unk, np.zeros_like(free, dtype=bool)
    frontier = np.zeros_like(free, dtype=bool) if frontier_mask is None else np.asarray(frontier_mask, dtype=bool)
    labels, count = ndimage.label(unk, structure=conn(8))
    filled = np.zeros_like(free, dtype=bool)
    for label in range(1, int(count) + 1):
        comp = labels == int(label)
        area = int(np.count_nonzero(comp))
        if area <= 0 or area > int(max_area_cells):
            continue
        rows, cols = np.nonzero(comp)
        if rows.size == 0:
            continue
        if int(rows.min()) == 0 or int(cols.min()) == 0 or int(rows.max()) == comp.shape[0] - 1 or int(cols.max()) == comp.shape[1] - 1:
            continue
        if np.any(comp & frontier) or np.any(ndimage.binary_dilation(comp, structure=conn(8)) & frontier):
            continue
        ring = ndimage.binary_dilation(comp, structure=conn(8)).astype(bool) & ~comp
        free_neighbors = int(np.count_nonzero(ring & free))
        wall_neighbors = int(np.count_nonzero(ring & wall_mask))
        observed_neighbors = free_neighbors + wall_neighbors
        if observed_neighbors <= 0:
            continue
        free_ratio = float(free_neighbors) / float(observed_neighbors)
        if wall_neighbors >= free_neighbors:
            continue
        if free_ratio < float(min_free_neighbor_ratio):
            continue
        free[comp] = True
        unk[comp] = False
        filled[comp] = True
    return free.astype(bool), unk.astype(bool), filled.astype(bool)


def build_voxel_roomseg_evidence(
    *,
    voxel_grid: VoxelOccupancyGrid3D,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    unknown_mask_from_navigation: np.ndarray,
    resolution_m: float,
    config: VoxelRoomsegEvidenceConfig | Mapping[str, object] | None = None,
) -> VoxelRoomsegEvidence:
    started_at = time.perf_counter()
    _ = resolution_m
    cfg = config if isinstance(config, VoxelRoomsegEvidenceConfig) else VoxelRoomsegEvidenceConfig.from_mapping(config)
    nav_free = np.asarray(navigation_free_mask, dtype=bool)
    nav_obstacle = np.asarray(navigation_obstacle_mask, dtype=bool)
    nav_unknown = np.asarray(unknown_mask_from_navigation, dtype=bool)
    shape = nav_free.shape
    if tuple(voxel_grid.shape) != tuple(shape):
        raise ValueError("voxel_grid shape must match navigation masks")
    if nav_obstacle.shape != shape or nav_unknown.shape != shape:
        raise ValueError("navigation masks must match voxel grid shape")
    if bool(cfg.fill_unknown_as_wall) or bool(cfg.fill_unknown_as_free) or not bool(cfg.preserve_unknown):
        raise ValueError("voxel roomseg evidence must preserve unknown")

    active_idx = voxel_grid.active_z_indices(
        z_min_m=float(cfg.active_z_min_m),
        z_max_m=float(voxel_grid.active_z_max_m or cfg.active_z_max_fallback_m),
    )
    if active_idx.size == 0:
        empty = np.zeros(shape, dtype=bool)
        zero_u16 = np.zeros(shape, dtype=np.uint16)
        zero_f32 = np.zeros(shape, dtype=np.float32)
        debug = _debug_aliases(
            {
                "voxel_vertical_free_xy": empty,
                "voxel_wall_xy": empty,
                "voxel_unknown_xy": np.ones(shape, dtype=bool),
                "voxel_vertical_observed_xy": empty,
                "voxel_active_free_count_xy": zero_u16,
                "voxel_active_occupied_count_xy": zero_u16,
                "voxel_active_unknown_count_xy": zero_u16,
                "voxel_active_observed_count_xy": zero_u16,
                "voxel_active_z_bin_count_xy": zero_u16,
                "voxel_occupied_ratio_active_xy": zero_f32,
                "voxel_unknown_ratio_active_xy": zero_f32,
                "voxel_observed_ratio_active_xy": zero_f32,
                "voxel_unknown_dominant_xy": empty,
                "voxel_wall_ratio_raw_xy": empty,
                "voxel_structural_wall_seed_xy": empty,
                "voxel_structural_wall_ratio_xy": empty,
                "voxel_wall_line_support_xy": empty,
                "voxel_wall_line_support_raw_xy": empty,
                "voxel_wall_line_support_strong_xy": empty,
                "voxel_wall_line_support_conflict_xy": empty,
                "voxel_wall_line_support_near_free_boundary_xy": empty,
                "voxel_wall_line_support_rejected_furniture_xy": empty,
                "voxel_wall_line_support_rejected_unknown_xy": empty,
                "voxel_wall_line_support_weight_xy": zero_f32,
                "voxel_wall_line_support_rejected_by_free_xy": empty,
                "voxel_wall_line_support_rejected_by_unknown_xy": empty,
                "voxel_wall_line_support_rejected_by_observed_xy": empty,
                "voxel_wall_line_support_rejected_by_nav_edge_xy": empty,
                "voxel_wall_rejected_by_free_xy": empty,
                "voxel_wall_rejected_by_unknown_xy": empty,
                "voxel_nonstructural_occupied_xy": empty,
                "voxel_small_unknown_hole_filled_map": empty,
                "voxel_wall_support_loose_xy": empty,
                "voxel_wall_support_unknown_gated_xy": empty,
                "voxel_wall_support_rejected_unknown_xy": empty,
                "voxel_occupied_any_xy": empty,
                "voxel_raw_occupied_wall_support_xy": empty,
                "voxel_strict_raw_wall_xy": empty,
                "voxel_wall_suppressed_by_free_xy": empty,
                "voxel_ratio_wall_debug_xy": empty,
                "voxel_free_wall_conflict_xy": empty,
                "voxel_free_wall_conflict_cells": 0,
                "voxel_wall_mode": str(cfg.wall_mode),
                "voxel_wall_ratio_used_for_final": True,
                "voxel_v19_projection_policy": "wall_line_support_then_side_validated_projection",
                "voxel_v18_projection_policy": "free_unknown_then_ratio_wall",
                "voxel_projection_priority": str(cfg.projection_priority),
                "voxel_min_free_z_cells_for_xy_free": int(cfg.min_free_z_cells_for_xy_free),
                "voxel_unknown_ratio_min_for_xy_unknown": float(cfg.unknown_ratio_min_for_xy_unknown),
                "voxel_min_observed_z_cells_for_known_column": int(cfg.min_observed_z_cells_for_known_column),
                "voxel_wall_occupied_ratio_min_for_xy_wall": float(cfg.wall_occupied_ratio_min_for_xy_wall),
                "voxel_wall_min_occupied_z_cells_for_xy_wall": int(cfg.wall_min_occupied_z_cells_for_xy_wall),
                "voxel_wall_line_support_cells": 0,
                "voxel_wall_line_support_raw_cells": 0,
                "voxel_wall_line_support_strong_cells": 0,
                "voxel_wall_line_support_conflict_cells": 0,
                "voxel_wall_line_support_near_free_boundary_cells": 0,
                "voxel_wall_line_support_rejected_furniture_cells": 0,
                "voxel_wall_line_support_rejected_unknown_cells": 0,
                "voxel_wall_line_support_rejected_by_free_cells": 0,
                "voxel_wall_line_support_rejected_by_unknown_cells": 0,
                "voxel_wall_line_support_rejected_by_observed_cells": 0,
                "voxel_wall_line_support_rejected_by_nav_edge_cells": 0,
                "voxel_wall_support_rejected_unknown_cells": 0,
                "voxel_small_unknown_hole_filled_cells": 0,
                "voxel_wall_occupied_ratio_debug_threshold": float(cfg.wall_occupied_ratio_debug_threshold),
                "voxel_project_roomseg_ms": float((time.perf_counter() - started_at) * 1000.0),
            }
        )
        return VoxelRoomsegEvidence(
            vertical_free_xy=empty,
            wall_xy=empty,
            unknown_xy=np.ones(shape, dtype=bool),
            active_observed_xy=empty,
            active_free_count_xy=zero_u16,
            active_occupied_count_xy=zero_u16,
            active_unknown_count_xy=zero_u16,
            active_observed_count_xy=zero_u16,
            active_z_bin_count_xy=zero_u16,
            occupied_any_xy=empty,
            raw_occupied_wall_support_xy=empty,
            strict_raw_wall_xy=empty,
            wall_suppressed_by_free_xy=empty,
            occupied_ratio_active_xy=zero_f32,
            unknown_ratio_active_xy=zero_f32,
            observed_ratio_active_xy=zero_f32,
            unknown_dominant_xy=empty,
            wall_support_loose_xy=empty,
            wall_support_unknown_gated_xy=empty,
            wall_support_rejected_unknown_xy=empty,
            structural_wall_seed_xy=empty,
            structural_wall_ratio_xy=empty,
            wall_rejected_by_free_xy=empty,
            wall_rejected_by_unknown_xy=empty,
            nonstructural_occupied_xy=empty,
            small_unknown_hole_filled_xy=empty,
            wall_line_support_xy=empty,
            wall_line_support_raw_xy=empty,
            wall_line_support_strong_xy=empty,
            wall_line_support_conflict_xy=empty,
            wall_line_support_near_free_boundary_xy=empty,
            wall_line_support_rejected_furniture_xy=empty,
            wall_line_support_rejected_unknown_xy=empty,
            wall_line_support_weight_xy=zero_f32,
            wall_line_support_rejected_by_free_xy=empty,
            wall_line_support_rejected_by_unknown_xy=empty,
            wall_line_support_rejected_by_observed_xy=empty,
            wall_line_support_rejected_by_nav_edge_xy=empty,
            ratio_wall_debug_xy=empty,
            free_wall_conflict_xy=empty,
            debug=debug,
        )

    active_state = np.asarray(voxel_grid.state[active_idx], dtype=np.uint8)
    classified = classify_voxel_columns_for_roomseg(
        state_active=active_state,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        cfg=cfg,
        frontier_mask=nav_unknown,
    )
    active_count = int(active_idx.size)
    free_count = np.asarray(classified["free_count"], dtype=np.uint16)
    occupied_count = np.asarray(classified["occupied_count"], dtype=np.uint16)
    unknown_count = np.asarray(classified["unknown_count"], dtype=np.uint16)
    observed_count = np.asarray(classified["observed_count"], dtype=np.uint16)
    active_observed = observed_count > 0
    active_z_bin_count = np.asarray(classified["active_z_bin_count"], dtype=np.uint16)
    occupied_ratio = np.asarray(classified["occupied_ratio"], dtype=np.float32)
    unknown_ratio = np.asarray(classified["unknown_ratio"], dtype=np.float32)
    observed_ratio = np.asarray(classified["observed_ratio"], dtype=np.float32)
    vertical_free_raw = np.asarray(classified["free_raw"], dtype=bool)
    occupied_any = np.asarray(classified["occupied_any"], dtype=bool)
    raw_occupied_wall_support = np.asarray(classified["raw_occupied_wall_support"], dtype=bool)
    wall_ratio_raw = np.asarray(classified["wall_ratio_raw"], dtype=bool)
    wall_rejected_by_free = np.asarray(classified["wall_rejected_by_free"], dtype=bool)
    wall_rejected_by_unknown = np.asarray(classified["wall_rejected_by_unknown"], dtype=bool)
    wall_suppressed_by_free = wall_rejected_by_free.copy()
    wall_support_loose = np.asarray(classified["wall_support_loose"], dtype=bool)
    wall_support_unknown_gated = np.asarray(classified["wall_support_unknown_gated"], dtype=bool)
    wall_support_rejected_unknown = np.asarray(classified["wall_support_rejected_unknown"], dtype=bool)
    strict_raw_wall = np.asarray(classified["wall"], dtype=bool)
    ratio_wall_debug = np.asarray(classified["ratio_wall_debug"], dtype=bool)
    free_wall_conflict = wall_rejected_by_free.copy()
    vertical_free = np.asarray(classified["vertical_free"], dtype=bool)
    wall = np.asarray(classified["wall"], dtype=bool)
    unknown = np.asarray(classified["unknown"], dtype=bool)
    unknown_dominant = np.asarray(classified["unknown_dominant"], dtype=bool)
    nonstructural_occupied = np.asarray(classified["nonstructural_occupied"], dtype=bool)
    small_unknown_hole_filled = np.asarray(classified["small_unknown_hole_filled"], dtype=bool)
    wall_line_support = np.asarray(classified["wall_line_support"], dtype=bool)
    wall_line_support_raw = np.asarray(classified["wall_line_support_raw"], dtype=bool)
    wall_line_support_strong = np.asarray(classified["wall_line_support_strong"], dtype=bool)
    wall_line_support_conflict = np.asarray(classified["wall_line_support_conflict"], dtype=bool)
    wall_line_support_near_free_boundary = np.asarray(classified["wall_line_support_near_free_boundary"], dtype=bool)
    wall_line_support_rejected_furniture = np.asarray(classified["wall_line_support_rejected_furniture"], dtype=bool)
    wall_line_support_rejected_unknown = np.asarray(classified["wall_line_support_rejected_unknown"], dtype=bool)
    wall_line_support_weight = np.asarray(classified["wall_line_support_weight"], dtype=np.float32)
    wall_line_support_rejected_by_free = np.asarray(classified["wall_line_support_rejected_by_free"], dtype=bool)
    wall_line_support_rejected_by_unknown = np.asarray(classified["wall_line_support_rejected_by_unknown"], dtype=bool)
    wall_line_support_rejected_by_observed = np.asarray(classified["wall_line_support_rejected_by_observed"], dtype=bool)
    wall_line_support_rejected_by_nav_edge = np.asarray(classified["wall_line_support_rejected_by_nav_edge"], dtype=bool)
    promoted_nav_obstacle = np.asarray(classified["promoted_nav_obstacle"], dtype=bool)
    nav_obstacle_overlap_free = vertical_free & nav_obstacle
    wall_before_cleanup = wall.copy()
    debug_summary = {
        "voxel_free_clean_cells": int(np.count_nonzero(vertical_free)),
        "voxel_free_raw_cells": int(np.count_nonzero(vertical_free_raw)),
        "voxel_wall_clean_cells": int(np.count_nonzero(wall)),
        "voxel_wall_raw_cells": int(np.count_nonzero(wall_ratio_raw)),
        "voxel_wall_ratio_raw_cells": int(np.count_nonzero(wall_ratio_raw)),
        "voxel_occupied_any_cells": int(np.count_nonzero(occupied_any)),
        "voxel_raw_occupied_wall_support_cells": int(np.count_nonzero(raw_occupied_wall_support)),
        "voxel_wall_support_loose_cells": int(np.count_nonzero(wall_support_loose)),
        "voxel_wall_support_unknown_gated_cells": int(np.count_nonzero(wall_support_unknown_gated)),
        "voxel_wall_support_rejected_unknown_cells": int(np.count_nonzero(wall_support_rejected_unknown)),
        "voxel_wall_rejected_by_free_cells": int(np.count_nonzero(wall_rejected_by_free)),
        "voxel_wall_rejected_by_unknown_cells": int(np.count_nonzero(wall_rejected_by_unknown)),
        "voxel_nonstructural_occupied_cells": int(np.count_nonzero(nonstructural_occupied)),
        "voxel_wall_line_support_cells": int(np.count_nonzero(wall_line_support)),
        "voxel_wall_line_support_raw_cells": int(np.count_nonzero(wall_line_support_raw)),
        "voxel_wall_line_support_strong_cells": int(np.count_nonzero(wall_line_support_strong)),
        "voxel_wall_line_support_conflict_cells": int(np.count_nonzero(wall_line_support_conflict)),
        "voxel_wall_line_support_near_free_boundary_cells": int(np.count_nonzero(wall_line_support_near_free_boundary)),
        "voxel_wall_line_support_rejected_furniture_cells": int(np.count_nonzero(wall_line_support_rejected_furniture)),
        "voxel_wall_line_support_rejected_unknown_cells": int(np.count_nonzero(wall_line_support_rejected_unknown)),
        "voxel_wall_line_support_rejected_by_free_cells": int(np.count_nonzero(wall_line_support_rejected_by_free)),
        "voxel_wall_line_support_rejected_by_unknown_cells": int(np.count_nonzero(wall_line_support_rejected_by_unknown)),
        "voxel_wall_line_support_rejected_by_observed_cells": int(np.count_nonzero(wall_line_support_rejected_by_observed)),
        "voxel_wall_line_support_rejected_by_nav_edge_cells": int(np.count_nonzero(wall_line_support_rejected_by_nav_edge)),
        "voxel_strict_raw_wall_cells": int(np.count_nonzero(strict_raw_wall)),
        "voxel_unknown_dominant_cells": int(np.count_nonzero(unknown_dominant)),
        "voxel_wall_suppressed_by_free_cells": int(np.count_nonzero(wall_suppressed_by_free)),
        "voxel_ratio_wall_debug_cells": int(np.count_nonzero(ratio_wall_debug)),
        "voxel_small_unknown_hole_filled_cells": int(np.count_nonzero(small_unknown_hole_filled)),
        "voxel_unknown_clean_cells": int(np.count_nonzero(unknown)),
        "voxel_free_wall_conflict_cells": int(np.count_nonzero(free_wall_conflict)),
        "voxel_vertical_observed_cells": int(np.count_nonzero(active_observed)),
        "voxel_navigation_obstacle_cells": int(np.count_nonzero(nav_obstacle)),
        "voxel_nav_obstacle_overlap_free_cells": int(np.count_nonzero(nav_obstacle_overlap_free)),
        "voxel_nav_obstacle_suppressed_free_cells": int(np.count_nonzero(nav_obstacle_overlap_free)) if bool(cfg.suppress_free_on_navigation_obstacle) else 0,
        "voxel_nav_obstacle_added_wall_cells": int(np.count_nonzero(promoted_nav_obstacle)),
        "voxel_navigation_unknown_cells": int(np.count_nonzero(nav_unknown)),
        "voxel_unknown_preserved_cells": int(np.count_nonzero(unknown & nav_unknown)),
    }
    debug = {
        "voxel_active_z_min_m": float(cfg.active_z_min_m),
        "voxel_active_z_max_m": float(voxel_grid.active_z_max_m or cfg.active_z_max_fallback_m),
        "voxel_ceiling_height_m": None if voxel_grid.ceiling_height_m is None else float(voxel_grid.ceiling_height_m),
        "voxel_ceiling_estimate_status": str(voxel_grid.ceiling_estimate_status),
        "voxel_active_z_bin_count": int(active_count),
        "voxel_wall_mode": str(cfg.wall_mode),
        "voxel_wall_ratio_used_for_final": True,
        "voxel_v19_projection_policy": "wall_line_support_then_side_validated_projection",
        "voxel_v18_projection_policy": "free_unknown_then_ratio_wall",
        "voxel_projection_priority": str(cfg.projection_priority),
        "voxel_min_free_z_cells_for_xy_free": int(cfg.min_free_z_cells_for_xy_free),
        "voxel_unknown_ratio_min_for_xy_unknown": float(cfg.unknown_ratio_min_for_xy_unknown),
        "voxel_min_observed_z_cells_for_known_column": int(cfg.min_observed_z_cells_for_known_column),
        "voxel_wall_occupied_ratio_min_for_xy_wall": float(cfg.wall_occupied_ratio_min_for_xy_wall),
        "voxel_wall_min_occupied_z_cells_for_xy_wall": int(cfg.wall_min_occupied_z_cells_for_xy_wall),
        "voxel_wall_line_support_enabled": bool(cfg.wall_line_support_enabled),
        "voxel_wall_line_support_min_occupied_z_cells": int(cfg.wall_line_support_min_occupied_z_cells),
        "voxel_wall_line_support_free_exclusion_z_cells": int(cfg.wall_line_support_free_exclusion_z_cells),
        "voxel_wall_line_support_unknown_ratio_max": float(cfg.wall_line_support_unknown_ratio_max),
        "voxel_wall_line_support_min_observed_z_cells": int(cfg.wall_line_support_min_observed_z_cells),
        "voxel_wall_line_support_use_nav_edge_gate": bool(cfg.wall_line_support_use_nav_edge_gate),
        "voxel_wall_line_support_nav_edge_radius_cells": int(cfg.wall_line_support_nav_edge_radius_cells),
        "voxel_wall_line_support_remove_small_area_cells": int(cfg.wall_line_support_remove_small_area_cells),
        "voxel_wall_min_occupied_z_cells": int(cfg.wall_min_occupied_z_cells),
        "voxel_wall_free_exclusion_min_z_cells": int(cfg.wall_free_exclusion_min_z_cells),
        "voxel_wall_unknown_gating_enabled": bool(cfg.wall_unknown_gating_enabled),
        "voxel_wall_unknown_ratio_max_for_structural": float(cfg.wall_unknown_ratio_max_for_structural),
        "voxel_wall_min_observed_z_cells_for_structural": int(cfg.wall_min_observed_z_cells_for_structural),
        "voxel_allow_unknown_dominant_wall_as_door_anchor": bool(cfg.allow_unknown_dominant_wall_as_door_anchor),
        "voxel_wall_occupied_ratio_debug_threshold": float(cfg.wall_occupied_ratio_debug_threshold),
        "voxel_free_raw_xy": vertical_free_raw.astype(bool),
        "voxel_wall_raw_xy": wall_ratio_raw.astype(bool),
        "voxel_wall_ratio_raw_xy": wall_ratio_raw.astype(bool),
        "voxel_structural_wall_seed_xy": wall.astype(bool),
        "voxel_structural_wall_ratio_xy": wall_ratio_raw.astype(bool),
        "voxel_wall_line_support_xy": wall_line_support.astype(bool),
        "voxel_wall_line_support_raw_xy": wall_line_support_raw.astype(bool),
        "voxel_wall_line_support_strong_xy": wall_line_support_strong.astype(bool),
        "voxel_wall_line_support_conflict_xy": wall_line_support_conflict.astype(bool),
        "voxel_wall_line_support_near_free_boundary_xy": wall_line_support_near_free_boundary.astype(bool),
        "voxel_wall_line_support_rejected_furniture_xy": wall_line_support_rejected_furniture.astype(bool),
        "voxel_wall_line_support_rejected_unknown_xy": wall_line_support_rejected_unknown.astype(bool),
        "voxel_wall_line_support_weight_xy": wall_line_support_weight.astype(np.float32),
        "voxel_wall_line_support_rejected_by_free_xy": wall_line_support_rejected_by_free.astype(bool),
        "voxel_wall_line_support_rejected_by_unknown_xy": wall_line_support_rejected_by_unknown.astype(bool),
        "voxel_wall_line_support_rejected_by_observed_xy": wall_line_support_rejected_by_observed.astype(bool),
        "voxel_wall_line_support_rejected_by_nav_edge_xy": wall_line_support_rejected_by_nav_edge.astype(bool),
        "voxel_wall_rejected_by_free_xy": wall_rejected_by_free.astype(bool),
        "voxel_wall_rejected_by_unknown_xy": wall_rejected_by_unknown.astype(bool),
        "voxel_nonstructural_occupied_xy": nonstructural_occupied.astype(bool),
        "voxel_small_unknown_hole_filled_map": small_unknown_hole_filled.astype(bool),
        "voxel_occupied_any_xy": occupied_any.astype(bool),
        "voxel_raw_occupied_wall_support_xy": raw_occupied_wall_support.astype(bool),
        "voxel_unknown_ratio_active_xy": unknown_ratio.astype(np.float32),
        "voxel_observed_ratio_active_xy": observed_ratio.astype(np.float32),
        "voxel_unknown_dominant_xy": unknown_dominant.astype(bool),
        "voxel_wall_support_loose_xy": wall_support_loose.astype(bool),
        "voxel_wall_support_unknown_gated_xy": wall_support_unknown_gated.astype(bool),
        "voxel_wall_support_rejected_unknown_xy": wall_support_rejected_unknown.astype(bool),
        "voxel_strict_raw_wall_xy": strict_raw_wall.astype(bool),
        "voxel_wall_suppressed_by_free_xy": wall_suppressed_by_free.astype(bool),
        "voxel_ratio_wall_debug_xy": ratio_wall_debug.astype(bool),
        "voxel_vertical_free_xy": vertical_free.astype(bool),
        "voxel_wall_xy": wall.astype(bool),
        "voxel_unknown_xy": unknown.astype(bool),
        "voxel_vertical_observed_xy": active_observed.astype(bool),
        "voxel_active_observed_xy": active_observed.astype(bool),
        "voxel_active_free_count_xy": free_count.astype(np.uint16),
        "voxel_active_occupied_count_xy": occupied_count.astype(np.uint16),
        "voxel_active_unknown_count_xy": unknown_count.astype(np.uint16),
        "voxel_active_observed_count_xy": observed_count.astype(np.uint16),
        "voxel_active_z_bin_count_xy": active_z_bin_count.astype(np.uint16),
        "voxel_occupied_ratio_active_xy": occupied_ratio.astype(np.float32),
        "voxel_unknown_ratio_active_xy": unknown_ratio.astype(np.float32),
        "voxel_observed_ratio_active_xy": observed_ratio.astype(np.float32),
        "voxel_free_wall_conflict_xy": free_wall_conflict.astype(bool),
        "voxel_free_wall_conflict_cells": int(np.count_nonzero(free_wall_conflict)),
        "voxel_occupied_any_cells": int(np.count_nonzero(occupied_any)),
        "voxel_raw_occupied_wall_support_cells": int(np.count_nonzero(raw_occupied_wall_support)),
        "voxel_wall_ratio_raw_cells": int(np.count_nonzero(wall_ratio_raw)),
        "voxel_structural_wall_seed_cells": int(np.count_nonzero(wall)),
        "voxel_structural_wall_ratio_cells": int(np.count_nonzero(wall_ratio_raw)),
        "voxel_wall_rejected_by_free_cells": int(np.count_nonzero(wall_rejected_by_free)),
        "voxel_wall_rejected_by_unknown_cells": int(np.count_nonzero(wall_rejected_by_unknown)),
        "voxel_nonstructural_occupied_cells": int(np.count_nonzero(nonstructural_occupied)),
        "voxel_wall_line_support_cells": int(np.count_nonzero(wall_line_support)),
        "voxel_wall_line_support_raw_cells": int(np.count_nonzero(wall_line_support_raw)),
        "voxel_wall_line_support_strong_cells": int(np.count_nonzero(wall_line_support_strong)),
        "voxel_wall_line_support_conflict_cells": int(np.count_nonzero(wall_line_support_conflict)),
        "voxel_wall_line_support_near_free_boundary_cells": int(np.count_nonzero(wall_line_support_near_free_boundary)),
        "voxel_wall_line_support_rejected_furniture_cells": int(np.count_nonzero(wall_line_support_rejected_furniture)),
        "voxel_wall_line_support_rejected_unknown_cells": int(np.count_nonzero(wall_line_support_rejected_unknown)),
        "voxel_wall_line_support_rejected_by_free_cells": int(np.count_nonzero(wall_line_support_rejected_by_free)),
        "voxel_wall_line_support_rejected_by_unknown_cells": int(np.count_nonzero(wall_line_support_rejected_by_unknown)),
        "voxel_wall_line_support_rejected_by_observed_cells": int(np.count_nonzero(wall_line_support_rejected_by_observed)),
        "voxel_wall_line_support_rejected_by_nav_edge_cells": int(np.count_nonzero(wall_line_support_rejected_by_nav_edge)),
        "voxel_small_unknown_hole_filled_cells": int(np.count_nonzero(small_unknown_hole_filled)),
        "voxel_wall_support_loose_cells": int(np.count_nonzero(wall_support_loose)),
        "voxel_wall_support_unknown_gated_cells": int(np.count_nonzero(wall_support_unknown_gated)),
        "voxel_wall_support_rejected_unknown_cells": int(np.count_nonzero(wall_support_rejected_unknown)),
        "voxel_unknown_dominant_cells": int(np.count_nonzero(unknown_dominant)),
        "voxel_strict_raw_wall_cells": int(np.count_nonzero(strict_raw_wall)),
        "voxel_wall_suppressed_by_free_cells": int(np.count_nonzero(wall_suppressed_by_free)),
        "voxel_ratio_wall_debug_cells": int(np.count_nonzero(ratio_wall_debug)),
        "voxel_vertical_free_cells": int(np.count_nonzero(vertical_free)),
        "voxel_unknown_cells": int(np.count_nonzero(unknown)),
        "voxel_free_raw_cells": int(np.count_nonzero(vertical_free_raw)),
        "voxel_wall_raw_cells": int(np.count_nonzero(wall_ratio_raw)),
        "voxel_free_clean_cells": int(np.count_nonzero(vertical_free)),
        "voxel_wall_clean_cells": int(np.count_nonzero(wall)),
        "voxel_unknown_clean_cells": int(np.count_nonzero(unknown)),
        "voxel_v19_vertical_free_cells": int(np.count_nonzero(vertical_free)),
        "voxel_v19_wall_cells": int(np.count_nonzero(wall)),
        "voxel_v19_unknown_cells": int(np.count_nonzero(unknown)),
        "voxel_v18_vertical_free_cells": int(np.count_nonzero(vertical_free)),
        "voxel_v18_wall_cells": int(np.count_nonzero(wall)),
        "voxel_v18_unknown_cells": int(np.count_nonzero(unknown)),
        "voxel_evidence_wall_before_cleanup": wall_before_cleanup.astype(bool),
        "voxel_evidence_nav_obstacle_overlap_free": nav_obstacle_overlap_free.astype(bool),
        "voxel_evidence_navigation_obstacle_promoted_wall": promoted_nav_obstacle.astype(bool),
        "voxel_evidence_debug_summary": debug_summary,
        "voxel_project_roomseg_ms": float((time.perf_counter() - started_at) * 1000.0),
    }
    debug = _debug_aliases(debug)
    return VoxelRoomsegEvidence(
        vertical_free_xy=vertical_free.astype(bool),
        wall_xy=wall.astype(bool),
        unknown_xy=unknown.astype(bool),
        active_observed_xy=active_observed.astype(bool),
        active_free_count_xy=free_count.astype(np.uint16),
        active_occupied_count_xy=occupied_count.astype(np.uint16),
        active_unknown_count_xy=unknown_count.astype(np.uint16),
        active_observed_count_xy=observed_count.astype(np.uint16),
        active_z_bin_count_xy=active_z_bin_count.astype(np.uint16),
        occupied_any_xy=occupied_any.astype(bool),
        raw_occupied_wall_support_xy=raw_occupied_wall_support.astype(bool),
        strict_raw_wall_xy=strict_raw_wall.astype(bool),
        wall_suppressed_by_free_xy=wall_suppressed_by_free.astype(bool),
        occupied_ratio_active_xy=occupied_ratio.astype(np.float32),
        unknown_ratio_active_xy=unknown_ratio.astype(np.float32),
        observed_ratio_active_xy=observed_ratio.astype(np.float32),
        unknown_dominant_xy=unknown_dominant.astype(bool),
        wall_support_loose_xy=wall_support_loose.astype(bool),
        wall_support_unknown_gated_xy=wall_support_unknown_gated.astype(bool),
        wall_support_rejected_unknown_xy=wall_support_rejected_unknown.astype(bool),
        structural_wall_seed_xy=wall.astype(bool),
        structural_wall_ratio_xy=wall_ratio_raw.astype(bool),
        wall_rejected_by_free_xy=wall_rejected_by_free.astype(bool),
        wall_rejected_by_unknown_xy=wall_rejected_by_unknown.astype(bool),
        nonstructural_occupied_xy=nonstructural_occupied.astype(bool),
        small_unknown_hole_filled_xy=small_unknown_hole_filled.astype(bool),
        wall_line_support_xy=wall_line_support.astype(bool),
        wall_line_support_raw_xy=wall_line_support_raw.astype(bool),
        wall_line_support_strong_xy=wall_line_support_strong.astype(bool),
        wall_line_support_conflict_xy=wall_line_support_conflict.astype(bool),
        wall_line_support_near_free_boundary_xy=wall_line_support_near_free_boundary.astype(bool),
        wall_line_support_rejected_furniture_xy=wall_line_support_rejected_furniture.astype(bool),
        wall_line_support_rejected_unknown_xy=wall_line_support_rejected_unknown.astype(bool),
        wall_line_support_weight_xy=wall_line_support_weight.astype(np.float32),
        wall_line_support_rejected_by_free_xy=wall_line_support_rejected_by_free.astype(bool),
        wall_line_support_rejected_by_unknown_xy=wall_line_support_rejected_by_unknown.astype(bool),
        wall_line_support_rejected_by_observed_xy=wall_line_support_rejected_by_observed.astype(bool),
        wall_line_support_rejected_by_nav_edge_xy=wall_line_support_rejected_by_nav_edge.astype(bool),
        ratio_wall_debug_xy=ratio_wall_debug.astype(bool),
        free_wall_conflict_xy=free_wall_conflict.astype(bool),
        debug=debug,
    )


def _debug_aliases(debug: dict[str, object]) -> dict[str, object]:
    aliased = dict(debug)
    alias_map = {
        "height_profile_vertical_free_xy": "voxel_vertical_free_xy",
        "height_profile_wall_xy": "voxel_wall_xy",
        "height_profile_unknown_xy": "voxel_unknown_xy",
        "height_profile_vertical_observed_xy": "voxel_vertical_observed_xy",
        "height_evidence_free_clean": "voxel_vertical_free_xy",
        "height_evidence_wall_clean": "voxel_wall_xy",
        "height_evidence_unknown_clean": "voxel_unknown_xy",
        "height_evidence_vertical_observed_xy": "voxel_vertical_observed_xy",
        "height_profile_free_wall_conflict_xy": "voxel_free_wall_conflict_xy",
    }
    for dst, src in alias_map.items():
        if src in aliased:
            aliased[dst] = aliased[src]
    return aliased


def voxel_column_debug(
    voxel_grid: VoxelOccupancyGrid3D,
    row: int,
    col: int,
    *,
    active_only: bool = True,
    config: VoxelRoomsegEvidenceConfig | Mapping[str, object] | None = None,
) -> dict[str, object]:
    cfg = config if isinstance(config, VoxelRoomsegEvidenceConfig) else VoxelRoomsegEvidenceConfig.from_mapping(config)
    r = int(row)
    c = int(col)
    if r < 0 or r >= voxel_grid.state.shape[1] or c < 0 or c >= voxel_grid.state.shape[2]:
        raise IndexError("voxel column row/col out of bounds")
    if bool(active_only):
        idx = voxel_grid.active_z_indices(
            z_min_m=float(cfg.active_z_min_m),
            z_max_m=float(voxel_grid.active_z_max_m or cfg.active_z_max_fallback_m),
        )
    else:
        idx = np.arange(voxel_grid.z_bin_count, dtype=np.int32)
    states = np.asarray(voxel_grid.state[idx, r, c], dtype=np.uint8)
    log = np.asarray(voxel_grid.log_odds[idx, r, c], dtype=np.int16)
    z = np.asarray(voxel_grid.z_centers_m[idx], dtype=np.float32)
    active_count = int(states.size)
    classified = classify_voxel_columns_for_roomseg(
        state_active=states.reshape((active_count, 1, 1)),
        navigation_free_mask=np.zeros((1, 1), dtype=bool),
        navigation_obstacle_mask=np.zeros((1, 1), dtype=bool),
        cfg=cfg,
    )

    def scalar(name: str) -> object:
        arr = np.asarray(classified[name])
        return arr[0, 0].item()

    free_count = int(scalar("free_count"))
    occupied_count = int(scalar("occupied_count"))
    unknown_count = int(scalar("unknown_count"))
    conflict_count = int(scalar("conflict_count"))
    observed_count = int(scalar("observed_count"))
    free_ratio = float(scalar("free_ratio"))
    occupied_ratio = float(scalar("occupied_ratio"))
    unknown_ratio = float(scalar("unknown_ratio"))
    observed_ratio = float(scalar("observed_ratio"))
    free_raw = bool(scalar("free_raw"))
    occupied_any = bool(scalar("occupied_any"))
    raw_occupied_wall_support = bool(scalar("raw_occupied_wall_support"))
    wall_ratio_raw = bool(scalar("wall_ratio_raw"))
    wall_rejected_by_free = bool(scalar("wall_rejected_by_free"))
    wall_rejected_by_unknown = bool(scalar("wall_rejected_by_unknown"))
    wall_suppressed_by_free = bool(wall_rejected_by_free)
    unknown_dominant = bool(scalar("unknown_dominant"))
    wall_support_loose = bool(scalar("wall_support_loose"))
    wall_support_unknown_gated = bool(scalar("wall_support_unknown_gated"))
    wall_support_rejected_unknown = bool(scalar("wall_support_rejected_unknown"))
    strict_raw_wall = bool(scalar("wall"))
    ratio_wall_debug = bool(scalar("ratio_wall_debug"))
    nonstructural_occupied = bool(scalar("nonstructural_occupied"))
    small_unknown_hole_filled = bool(scalar("small_unknown_hole_filled"))
    wall_line_support = bool(scalar("wall_line_support"))
    wall_line_support_raw = bool(scalar("wall_line_support_raw"))
    wall_line_support_rejected_by_free = bool(scalar("wall_line_support_rejected_by_free"))
    wall_line_support_rejected_by_unknown = bool(scalar("wall_line_support_rejected_by_unknown"))
    wall_line_support_rejected_by_observed = bool(scalar("wall_line_support_rejected_by_observed"))
    wall_line_support_rejected_by_nav_edge = bool(scalar("wall_line_support_rejected_by_nav_edge"))
    final_vertical_free = bool(scalar("vertical_free"))
    final_wall = bool(scalar("wall"))
    final_unknown = bool(scalar("unknown"))
    return {
        "row": int(r),
        "col": int(c),
        "active_only": bool(active_only),
        "z_centers_m": [float(v) for v in z.tolist()],
        "state_codes": [int(v) for v in states.tolist()],
        "state_names": [_state_name(int(v)) for v in states.tolist()],
        "log_odds": [int(v) for v in log.tolist()],
        "free_count_active": int(free_count),
        "occupied_count_active": int(occupied_count),
        "unknown_count_active": int(unknown_count),
        "conflict_count_active": int(conflict_count),
        "observed_count_active": int(observed_count),
        "active_z_bin_count": int(active_count),
        "free_ratio_active": float(free_ratio),
        "occupied_ratio_active": float(occupied_ratio),
        "unknown_ratio_active": float(unknown_ratio),
        "observed_ratio_active": float(observed_ratio),
        "free_raw": bool(free_raw),
        "occupied_any": bool(occupied_any),
        "raw_occupied_wall_support": bool(raw_occupied_wall_support),
        "wall_ratio_raw": bool(wall_ratio_raw),
        "unknown_dominant": bool(unknown_dominant),
        "wall_rejected_by_free": bool(wall_rejected_by_free),
        "wall_rejected_by_unknown": bool(wall_rejected_by_unknown),
        "wall_support_loose": bool(wall_support_loose),
        "wall_support_unknown_gated": bool(wall_support_unknown_gated),
        "wall_support_rejected_unknown": bool(wall_support_rejected_unknown),
        "strict_raw_wall": bool(strict_raw_wall),
        "wall_suppressed_by_free": bool(wall_suppressed_by_free),
        "ratio_wall_debug": bool(ratio_wall_debug),
        "nonstructural_occupied": bool(nonstructural_occupied),
        "small_unknown_hole_filled": bool(small_unknown_hole_filled),
        "wall_line_support": bool(wall_line_support),
        "wall_line_support_raw": bool(wall_line_support_raw),
        "wall_line_support_rejected_by_free": bool(wall_line_support_rejected_by_free),
        "wall_line_support_rejected_by_unknown": bool(wall_line_support_rejected_by_unknown),
        "wall_line_support_rejected_by_observed": bool(wall_line_support_rejected_by_observed),
        "wall_line_support_rejected_by_nav_edge": bool(wall_line_support_rejected_by_nav_edge),
        "wall_raw": bool(wall_ratio_raw),
        "final_vertical_free": bool(final_vertical_free),
        "final_wall": bool(final_wall),
        "final_unknown": bool(final_unknown),
        "projection_priority": str(cfg.projection_priority),
        "wall_mode": str(cfg.wall_mode),
        "wall_ratio_used_for_final": True,
        "min_free_z_cells_for_xy_free": int(cfg.min_free_z_cells_for_xy_free),
        "unknown_ratio_min_for_xy_unknown": float(cfg.unknown_ratio_min_for_xy_unknown),
        "min_observed_z_cells_for_known_column": int(cfg.min_observed_z_cells_for_known_column),
        "wall_occupied_ratio_min_for_xy_wall": float(cfg.wall_occupied_ratio_min_for_xy_wall),
        "wall_min_occupied_z_cells_for_xy_wall": int(cfg.wall_min_occupied_z_cells_for_xy_wall),
        "wall_line_support_enabled": bool(cfg.wall_line_support_enabled),
        "wall_line_support_min_occupied_z_cells": int(cfg.wall_line_support_min_occupied_z_cells),
        "wall_line_support_free_exclusion_z_cells": int(cfg.wall_line_support_free_exclusion_z_cells),
        "wall_line_support_unknown_ratio_max": float(cfg.wall_line_support_unknown_ratio_max),
        "wall_line_support_min_observed_z_cells": int(cfg.wall_line_support_min_observed_z_cells),
        "wall_min_occupied_z_cells": int(cfg.wall_min_occupied_z_cells),
        "wall_free_exclusion_min_z_cells": int(cfg.wall_free_exclusion_min_z_cells),
        "wall_unknown_gating_enabled": bool(cfg.wall_unknown_gating_enabled),
        "wall_unknown_ratio_max_for_structural": float(cfg.wall_unknown_ratio_max_for_structural),
        "wall_min_observed_z_cells_for_structural": int(cfg.wall_min_observed_z_cells_for_structural),
        "wall_occupied_ratio_debug_threshold": float(cfg.wall_occupied_ratio_debug_threshold),
    }


def _state_name(value: int) -> str:
    if int(value) == int(VOXEL_FREE):
        return "free"
    if int(value) == int(VOXEL_OCCUPIED):
        return "occupied"
    if int(value) == int(VOXEL_UNKNOWN):
        return "unknown"
    return "conflict"


def _remove_small_true_components(mask: np.ndarray, *, max_area_cells: int, connectivity: int, remove_leq: bool) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if int(max_area_cells) <= 0 or not np.any(src):
        return src.copy()
    labels, count = label_components(src, connectivity)
    out = src.copy()
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        area = int(np.count_nonzero(comp))
        if (area <= int(max_area_cells)) if remove_leq else (area < int(max_area_cells)):
            out[comp] = False
    return out.astype(bool)


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return conn(4).astype(bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)
