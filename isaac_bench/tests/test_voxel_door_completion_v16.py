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


def test_v16_seed_cluster_without_wall_still_outputs_extension_attempts() -> None:
    shape = (21, 21)
    seed = _seed_result(shape, [[(10, 10)]])

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=np.ones(shape, dtype=bool),
        base_partition_free=np.ones(shape, dtype=bool),
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(seed_cluster_morph_close_radius_cells=0),
    )

    assert result.debug["voxel_door_seed_cluster_count"] == 1
    assert result.debug["voxel_door_trial_candidate_count"] >= 1
    assert np.any(result.door_extension_attempt_all_mask)
    assert np.any(result.door_extension_attempt_rejected_mask)
    assert not np.any(result.door_cut_mask_for_partition)


def test_v16_two_wall_seed_has_visual_and_partition_cut_maps() -> None:
    shape = (18, 26)
    seed = _seed_result(shape, [[(8, 12), (9, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[4:14, 5:21] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[3, 12] = True
    anchors[14, 12] = True
    partition_free = free & ~anchors

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=partition_free,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(wall_anchor_radius_cells=0, seed_cluster_morph_close_radius_cells=0),
        real_wall_barrier_map=anchors,
    )

    assert np.any(result.door_centerline_visual_mask)
    assert np.any(result.door_partition_cut_candidate_mask)
    assert np.any(result.door_cut_mask_for_partition)
    assert np.array_equal(result.debug["voxel_door_partition_cut_accepted_mask"], result.door_cut_mask_for_partition)
    assert result.debug["voxel_door_partition_accepted_count"] == 1


def test_v16_visual_only_door_keeps_partition_reject_auditable() -> None:
    shape = (20, 28)
    seed = _seed_result(shape, [[(10, 13), (10, 14)]])
    visual_free = np.zeros(shape, dtype=bool)
    visual_free[10, 6:22] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[10, 6] = True
    anchors[10, 21] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=visual_free,
        free_map_for_visual_validation=visual_free,
        base_partition_free=np.zeros(shape, dtype=bool),
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(wall_anchor_radius_cells=0, partition_topology_reject_mode="reject"),
    )

    assert not np.any(result.door_centerline_visual_mask)
    assert np.any(result.door_visual_only_mask)
    assert np.any(result.debug["voxel_door_provisional_accepted_visual_mask"])
    assert np.any(result.door_partition_cut_candidate_mask)
    assert np.any(result.debug["voxel_door_partition_cut_rejected_mask"])
    assert not np.any(result.door_cut_mask_for_partition)
    assert result.debug["voxel_door_topology_reject_reason_counts"]["door_cut_no_topology_gain"] == 1


def test_v16_split_seed_clusters_do_not_hard_reject_visual_crossing_other_seed() -> None:
    shape = (24, 28)
    seed = _seed_result(shape, [[(12, 8), (12, 9)], [(8, 12), (9, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[12, 2:19] = True
    free[2:19, 12] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[12, 2] = True
    anchors[12, 18] = True
    anchors[2, 12] = True
    anchors[18, 12] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free & ~anchors,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(wall_anchor_radius_cells=0, seed_cluster_merge_distance_cells=0, partition_topology_enabled=False),
        real_wall_barrier_map=anchors,
    )

    assert result.debug["voxel_door_seed_cluster_count"] == 2
    assert "extension_intersects_other_door_candidate" not in result.debug["voxel_door_reject_reason_counts"]
    assert "partition_intersects_other_door_candidate" in result.debug["voxel_door_partition_reject_reason_counts"]
