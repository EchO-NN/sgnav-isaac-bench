from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import VoxelDoorDetectorConfig, VoxelDoorSeedResult, complete_voxel_doors_from_seeds


def _seed_result(shape: tuple[int, int], components: list[list[tuple[int, int]]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for label, cells in enumerate(components, start=1):
        for r, c in cells:
            seed[r, c] = True
            labels[r, c] = int(label)
    return VoxelDoorSeedResult(
        door_seed_mask=seed,
        door_seed_component_map=labels,
        door_seed_reject_reason_map=np.zeros(shape, dtype=np.uint8),
        lower_free_cells_xy=np.zeros(shape, dtype=np.uint16),
        top_occupied_cells_xy=np.zeros(shape, dtype=np.uint16),
        first_occupied_z_xy=np.full(shape, np.nan, dtype=np.float32),
        unknown_tail_cells_xy=np.zeros(shape, dtype=np.uint16),
        seed_evidence=[],
        debug={"voxel_door_seed_mask": seed, "voxel_door_seed_component_map": labels},
    )


def test_same_physical_door_seed_fragments_merge_before_completion() -> None:
    shape = (24, 32)
    seed = _seed_result(shape, [[(12, 12), (12, 13)], [(12, 18), (12, 19)]])
    free = np.zeros(shape, dtype=bool)
    free[12, 6:24] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[12, 6] = True
    anchors[12, 23] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(seed_cluster_merge_distance_cells=6, seed_cluster_morph_close_radius_cells=2, partition_topology_enabled=False),
    )

    assert result.debug["voxel_door_seed_component_count"] == 2
    assert result.debug["voxel_door_seed_cluster_count"] == 1
    assert result.debug["voxel_door_visual_accepted_count"] == 1
    assert result.debug["voxel_door_partition_accepted_count"] == 1
    assert "extension_hits_other_door_cluster" not in result.debug["voxel_door_trial_reject_reason_counts"]


def test_door_visual_line_drawn_even_when_partition_cut_empty() -> None:
    shape = (20, 28)
    seed = _seed_result(shape, [[(10, 13), (10, 14)]])
    visual_free = np.zeros(shape, dtype=bool)
    visual_free[10, 6:22] = True
    partition_free = np.zeros(shape, dtype=bool)
    anchors = np.zeros(shape, dtype=bool)
    anchors[10, 6] = True
    anchors[10, 21] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=visual_free,
        free_map_for_visual_validation=visual_free,
        base_partition_free=partition_free,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(partition_topology_reject_mode="reject"),
    )

    assert result.debug["voxel_door_visual_accepted_count"] == 1
    assert result.debug["voxel_door_partition_accepted_count"] == 0
    assert not np.any(result.door_centerline_visual_mask)
    assert np.any(result.door_visual_only_mask)
    assert np.any(result.debug["voxel_door_provisional_accepted_visual_mask"])
    assert not np.any(result.door_cut_mask_for_partition)
    assert result.debug["voxel_door_topology_reject_reason_counts"]["door_cut_no_topology_gain"] == 1


def test_door_unknown_ratio_does_not_count_real_wall_as_unknown() -> None:
    shape = (20, 28)
    seed = _seed_result(shape, [[(10, 13), (10, 14)]])
    free = np.zeros(shape, dtype=bool)
    free[10, 7:21] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[10, 6] = True
    anchors[10, 21] = True
    real_wall = anchors.copy()

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        free_map_for_visual_validation=free,
        base_partition_free=free,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        real_wall_barrier_map=real_wall,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(partition_topology_enabled=False),
    )

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["inner_unknown_ratio"] == 0.0
    assert result.debug["voxel_door_partition_accepted_count"] == 1
    assert np.any(result.door_cut_mask_for_partition)
