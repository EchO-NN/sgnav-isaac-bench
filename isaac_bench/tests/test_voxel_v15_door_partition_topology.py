from __future__ import annotations

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import conn
from isaac_bench.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    VoxelDoorSeedResult,
    complete_voxel_doors_from_seeds,
)


def _seed_result(shape: tuple[int, int], cells: list[tuple[int, int]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for r, c in cells:
        seed[r, c] = True
        labels[r, c] = 1
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


def test_door_visual_line_does_not_become_partition_without_topology_gain() -> None:
    shape = (20, 28)
    seed = _seed_result(shape, [(10, 13), (10, 14)])
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
        config=_legacy_completion_config(wall_anchor_radius_cells=0, partition_topology_reject_mode="reject"),
    )

    assert not np.any(result.debug["voxel_door_centerline_visual_mask"])
    assert np.any(result.debug["voxel_door_visual_only_mask"])
    assert np.any(result.debug["voxel_door_provisional_accepted_visual_mask"])
    assert not np.any(result.debug["voxel_door_cut_mask"])
    assert result.debug["voxel_door_topology_reject_reason_counts"]["door_cut_no_topology_gain"] == 1


def test_valid_door_cut_must_split_partition_free() -> None:
    shape = (18, 26)
    seed = _seed_result(shape, [(8, 12), (9, 12)])
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

    door_cut = np.asarray(result.debug["voxel_door_cut_mask"], dtype=bool)
    before_count = ndimage.label(partition_free, structure=conn(4))[1]
    after_count = ndimage.label((partition_free | np.asarray(seed.door_seed_mask, dtype=bool)) & ~door_cut, structure=conn(4))[1]

    assert np.any(door_cut)
    assert after_count > before_count
    assert result.debug["voxel_door_partition_accepted_count"] == 1
    assert result.candidates[0].debug["door_topology_accepted"] is True
