from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    build_voxel_partition_maps,
)
from isaac_bench.mapping.voxel_roomseg_evidence import VoxelRoomsegEvidence


def _evidence(shape: tuple[int, int]) -> VoxelRoomsegEvidence:
    vertical_free = np.zeros(shape, dtype=bool)
    raw_support = np.zeros(shape, dtype=bool)
    strict_wall = np.zeros(shape, dtype=bool)
    vertical_free[3, 4] = True
    raw_support[3, 4] = True
    strict_wall[2, 4] = True
    zero_u16 = np.zeros(shape, dtype=np.uint16)
    zero_f32 = np.zeros(shape, dtype=np.float32)
    return VoxelRoomsegEvidence(
        vertical_free_xy=vertical_free,
        wall_xy=strict_wall.copy(),
        unknown_xy=~(vertical_free | strict_wall),
        active_observed_xy=vertical_free | raw_support | strict_wall,
        active_free_count_xy=zero_u16.copy(),
        active_occupied_count_xy=zero_u16.copy(),
        active_unknown_count_xy=zero_u16.copy(),
        active_observed_count_xy=zero_u16.copy(),
        active_z_bin_count_xy=zero_u16.copy(),
        occupied_any_xy=raw_support | strict_wall,
        raw_occupied_wall_support_xy=raw_support | strict_wall,
        strict_raw_wall_xy=strict_wall,
        wall_suppressed_by_free_xy=np.zeros(shape, dtype=bool),
        occupied_ratio_active_xy=zero_f32,
        unknown_ratio_active_xy=zero_f32.copy(),
        observed_ratio_active_xy=zero_f32.copy(),
        unknown_dominant_xy=np.zeros(shape, dtype=bool),
        wall_support_loose_xy=raw_support | strict_wall,
        wall_support_unknown_gated_xy=raw_support | strict_wall,
        wall_support_rejected_unknown_xy=np.zeros(shape, dtype=bool),
        structural_wall_seed_xy=strict_wall.copy(),
        structural_wall_ratio_xy=strict_wall.copy(),
        wall_rejected_by_free_xy=vertical_free & raw_support,
        wall_rejected_by_unknown_xy=np.zeros(shape, dtype=bool),
        nonstructural_occupied_xy=raw_support & ~strict_wall & ~vertical_free,
        small_unknown_hole_filled_xy=np.zeros(shape, dtype=bool),
        wall_line_support_xy=raw_support | strict_wall,
        wall_line_support_raw_xy=raw_support | strict_wall,
        wall_line_support_rejected_by_free_xy=vertical_free & raw_support,
        wall_line_support_rejected_by_unknown_xy=np.zeros(shape, dtype=bool),
        wall_line_support_rejected_by_observed_xy=np.zeros(shape, dtype=bool),
        wall_line_support_rejected_by_nav_edge_xy=np.zeros(shape, dtype=bool),
        ratio_wall_debug_xy=np.zeros(shape, dtype=bool),
        free_wall_conflict_xy=vertical_free & raw_support,
        debug={},
    )


def test_raw_support_overlap_does_not_remove_partition_free_or_become_anchor() -> None:
    shape = (8, 9)
    cfg = VoxelOccupancyDoorWallRoomSegConfig()
    bundle = build_voxel_partition_maps(
        evidence=_evidence(shape),
        door_seed_mask=np.zeros(shape, dtype=bool),
        strict_raw_wall=np.asarray(_evidence(shape).strict_raw_wall_xy, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        step1_gap_fill_map=np.zeros(shape, dtype=bool),
        cfg=cfg,
    )

    assert not bundle.wall_anchor_support_map[3, 4]
    assert not bundle.door_anchor_wall_map[3, 4]
    assert bundle.base_partition_free[3, 4]
    assert not bundle.partition_real_wall_map[3, 4]
    assert bundle.debug["voxel_partition_maps_stage"] == "v19_wall_recovery_partition_sources"


def test_door_seed_carves_partition_wall_but_not_anchor_support() -> None:
    shape = (8, 9)
    evidence = _evidence(shape)
    door_seed = np.zeros(shape, dtype=bool)
    door_seed[2, 4] = True
    cfg = VoxelOccupancyDoorWallRoomSegConfig(door_seed_wall_carve_radius_cells=0)
    bundle = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_mask=door_seed,
        strict_raw_wall=evidence.strict_raw_wall_xy,
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        step1_gap_fill_map=np.zeros(shape, dtype=bool),
        cfg=cfg,
    )

    assert bundle.wall_anchor_support_map[2, 4]
    assert bundle.door_anchor_wall_map[2, 4]
    assert not bundle.partition_real_wall_map[2, 4]
    assert bundle.removed_by_seed_carve_map[2, 4]


def test_extension_seed_line_not_partition_wall() -> None:
    shape = (8, 9)
    evidence = _evidence(shape)
    evidence.vertical_free_xy[:, :] = True
    evidence.wall_xy[:, :] = False
    evidence.strict_raw_wall_xy[:, :] = False
    extension_seed = np.zeros(shape, dtype=bool)
    extension_seed[4, 2:7] = True

    bundle = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_mask=np.zeros(shape, dtype=bool),
        strict_raw_wall=evidence.strict_raw_wall_xy,
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=extension_seed,
        step1_gap_fill_map=np.zeros(shape, dtype=bool),
        cfg=VoxelOccupancyDoorWallRoomSegConfig(),
    )

    assert not np.any(bundle.partition_real_wall_map & extension_seed)
    assert np.all(bundle.base_partition_free[extension_seed])
