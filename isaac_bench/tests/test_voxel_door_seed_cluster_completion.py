from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import VoxelDoorDetectorConfig, VoxelDoorSeedResult, complete_voxel_doors_from_seeds


def _seed_result(shape: tuple[int, int], components: list[list[tuple[int, int]]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for cid, cells in enumerate(components, start=1):
        for r, c in cells:
            seed[r, c] = True
            labels[r, c] = cid
    return VoxelDoorSeedResult(
        door_seed_mask=seed,
        door_seed_component_map=labels,
        door_seed_reject_reason_map=np.zeros(shape, dtype=np.uint8),
        lower_free_cells_xy=np.zeros(shape, dtype=np.uint16),
        top_occupied_cells_xy=np.zeros(shape, dtype=np.uint16),
        first_occupied_z_xy=np.full(shape, np.nan, dtype=np.float32),
        unknown_tail_cells_xy=np.zeros(shape, dtype=np.uint16),
        seed_evidence=[],
        debug={},
    )


def _legacy_completion_config(**overrides: object) -> VoxelDoorDetectorConfig:
    data = {
        "min_seed_cells_for_accepted_extension": 1,
        "min_seed_line_length_cells_for_accepted_extension": 1,
        "min_seed_elongation_for_direction": 1.0,
        "accepted_orientation_mode": "legacy",
        "local_free_neck_orientation_debug_only": False,
    }
    data.update(overrides)
    return VoxelDoorDetectorConfig(**data)


def test_split_same_door_seed_components_complete_as_one_cluster() -> None:
    shape = (24, 28)
    seed_result = _seed_result(shape, [[(12, 10), (12, 11)], [(12, 14), (12, 15)]])
    free = np.zeros(shape, dtype=bool)
    free[12, 5:22] = True
    anchor_wall = np.zeros(shape, dtype=bool)
    anchor_wall[12, 5] = True
    anchor_wall[12, 21] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed_result,
        free_map=free,
        base_partition_free=free & ~anchor_wall,
        anchor_wall_map=anchor_wall,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(wall_anchor_radius_cells=0, seed_cluster_merge_distance_cells=3, partition_topology_enabled=False),
        real_wall_barrier_map=anchor_wall,
    )

    assert result.debug["voxel_door_seed_component_count"] == 2
    assert result.debug["voxel_door_seed_cluster_count"] == 1
    assert result.debug["voxel_door_visual_accepted_count"] == 1
    assert result.debug["voxel_door_partition_accepted_count"] == 1
    assert np.any(result.door_centerline_visual_mask)
    assert np.any(result.door_cut_mask_for_partition)
    assert "extension_hits_door_instead_of_wall" not in result.debug["voxel_door_reject_reason_counts"]
    assert "extension_intersects_other_door_candidate" not in result.debug["voxel_door_reject_reason_counts"]


def test_different_door_clusters_that_cross_are_still_rejected() -> None:
    shape = (24, 28)
    seed_result = _seed_result(shape, [[(12, 8), (12, 9)], [(8, 12), (9, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[12, 2:19] = True
    free[2:19, 12] = True
    anchor_wall = np.zeros(shape, dtype=bool)
    anchor_wall[12, 2] = True
    anchor_wall[12, 18] = True
    anchor_wall[2, 12] = True
    anchor_wall[18, 12] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed_result,
        free_map=free,
        base_partition_free=free & ~anchor_wall,
        anchor_wall_map=anchor_wall,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(wall_anchor_radius_cells=0, seed_cluster_merge_distance_cells=0, partition_topology_enabled=False),
        real_wall_barrier_map=anchor_wall,
    )

    assert result.debug["voxel_door_seed_cluster_count"] == 2
    assert "extension_intersects_other_door_candidate" not in result.debug["voxel_door_reject_reason_counts"]
    assert "partition_intersects_other_door_candidate" in result.debug["voxel_door_partition_reject_reason_counts"]
