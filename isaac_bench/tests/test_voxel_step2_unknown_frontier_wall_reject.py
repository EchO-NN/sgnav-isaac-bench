from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, build_voxel_partition_maps
from isaac_bench.mapping.voxel_roomseg_evidence import VoxelRoomsegEvidence


def _evidence(shape: tuple[int, int]) -> VoxelRoomsegEvidence:
    zero_u16 = np.zeros(shape, dtype=np.uint16)
    zero_f32 = np.zeros(shape, dtype=np.float32)
    vertical_free = np.zeros(shape, dtype=bool)
    vertical_free[2, 2] = True
    unknown_dominant = np.zeros(shape, dtype=bool)
    unknown_dominant[2, 2] = True
    rejected = np.zeros(shape, dtype=bool)
    rejected[2, 2] = True
    return VoxelRoomsegEvidence(
        vertical_free_xy=vertical_free,
        wall_xy=np.zeros(shape, dtype=bool),
        unknown_xy=~vertical_free,
        active_observed_xy=vertical_free,
        active_free_count_xy=zero_u16.copy(),
        active_occupied_count_xy=zero_u16.copy(),
        active_unknown_count_xy=zero_u16.copy(),
        active_observed_count_xy=zero_u16.copy(),
        active_z_bin_count_xy=zero_u16.copy(),
        occupied_any_xy=rejected.copy(),
        raw_occupied_wall_support_xy=rejected.copy(),
        strict_raw_wall_xy=np.zeros(shape, dtype=bool),
        wall_suppressed_by_free_xy=np.zeros(shape, dtype=bool),
        occupied_ratio_active_xy=zero_f32.copy(),
        unknown_ratio_active_xy=zero_f32.copy(),
        observed_ratio_active_xy=zero_f32.copy(),
        unknown_dominant_xy=unknown_dominant,
        wall_support_loose_xy=rejected.copy(),
        wall_support_unknown_gated_xy=np.zeros(shape, dtype=bool),
        wall_support_rejected_unknown_xy=rejected.copy(),
        structural_wall_seed_xy=np.zeros(shape, dtype=bool),
        structural_wall_ratio_xy=rejected.copy(),
        wall_rejected_by_free_xy=np.zeros(shape, dtype=bool),
        wall_rejected_by_unknown_xy=rejected.copy(),
        nonstructural_occupied_xy=rejected.copy(),
        small_unknown_hole_filled_xy=np.zeros(shape, dtype=bool),
        wall_line_support_xy=rejected.copy(),
        wall_line_support_raw_xy=rejected.copy(),
        wall_line_support_strong_xy=np.zeros(shape, dtype=bool),
        wall_line_support_conflict_xy=np.zeros(shape, dtype=bool),
        wall_line_support_near_free_boundary_xy=rejected.copy(),
        wall_line_support_rejected_furniture_xy=np.zeros(shape, dtype=bool),
        wall_line_support_rejected_unknown_xy=rejected.copy(),
        wall_line_support_weight_xy=np.zeros(shape, dtype=np.float32),
        wall_line_support_rejected_by_free_xy=np.zeros(shape, dtype=bool),
        wall_line_support_rejected_by_unknown_xy=rejected.copy(),
        wall_line_support_rejected_by_observed_xy=np.zeros(shape, dtype=bool),
        wall_line_support_rejected_by_nav_edge_xy=np.zeros(shape, dtype=bool),
        ratio_wall_debug_xy=np.zeros(shape, dtype=bool),
        free_wall_conflict_xy=np.zeros(shape, dtype=bool),
        debug={},
    )


def test_step2_target_excludes_unknown_dominant_false_frontier_wall() -> None:
    shape = (6, 6)
    fake_wall = np.zeros(shape, dtype=bool)
    fake_wall[2, 2] = True

    bundle = build_voxel_partition_maps(
        evidence=_evidence(shape),
        door_seed_mask=np.zeros(shape, dtype=bool),
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=fake_wall,
        extension_seed_line_map=fake_wall,
        step1_gap_fill_map=np.zeros(shape, dtype=bool),
        cfg=VoxelOccupancyDoorWallRoomSegConfig(),
    )

    assert not bundle.door_anchor_wall_map[2, 2]
    assert not bundle.step2_target_wall_map[2, 2]
    assert bundle.debug["voxel_step2_false_frontier_wall_rejected_map"][2, 2]
