from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    DOOR_COMPLETION_ONE_SEED_ONE_WALL,
    DOOR_COMPLETION_SEED_PAIR_BRIDGE,
    VoxelDoorDetectorConfig,
    VoxelDoorSeedResult,
    complete_voxel_doors_from_seeds,
)


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


def test_one_seed_one_wall_completion_generates_partition_cut() -> None:
    shape = (18, 24)
    seed = _seed_result(shape, [[(9, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[9, 6:13] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[9, 6] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free & ~anchors,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(
            wall_anchor_radius_cells=0,
            seed_cluster_morph_close_radius_cells=0,
            partition_topology_enabled=False,
        ),
        real_wall_barrier_map=anchors,
    )

    accepted = [candidate for candidate in result.candidates if candidate.accepted]
    assert accepted
    assert accepted[0].debug["completion_mode"] == DOOR_COMPLETION_ONE_SEED_ONE_WALL
    assert np.any(result.door_cut_mask_for_partition)


def test_seed_pair_bridge_supports_two_offset_seed_blobs_without_wall_anchor() -> None:
    shape = (18, 24)
    seed = _seed_result(shape, [[(9, 10)], [(10, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[9:11, 10:13] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free,
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_legacy_completion_config(
            wall_anchor_radius_cells=0,
            seed_cluster_morph_close_radius_cells=0,
            seed_cluster_merge_distance_cells=0,
            seed_cluster_max_perpendicular_gap_cells=0,
            partition_topology_enabled=False,
        ),
    )

    assert result.debug["voxel_door_seed_pair_group_count"] >= 1
    accepted = [candidate for candidate in result.candidates if candidate.accepted]
    assert any(candidate.debug["completion_mode"] == DOOR_COMPLETION_SEED_PAIR_BRIDGE for candidate in accepted)
    assert np.any(result.door_cut_mask_for_partition)


def test_topology_no_gain_is_warning_by_default_for_geometry_valid_door_cut() -> None:
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
        config=_legacy_completion_config(wall_anchor_radius_cells=0),
    )

    assert np.any(result.door_cut_mask_for_partition)
    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["partition_accepted"] is True
    assert candidate.debug["partition_topology_accepted"] is False
    assert candidate.debug["door_topology_warning"] is True
    assert np.any(result.debug["voxel_door_topology_warning_cut_mask"])
