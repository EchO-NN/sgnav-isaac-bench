from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    VoxelDoorSeedResult,
    complete_voxel_doors_from_seeds,
)


def _seed_result(shape: tuple[int, int], cells: list[tuple[int, int]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for idx, (row, col) in enumerate(cells, start=1):
        seed[int(row), int(col)] = True
        labels[int(row), int(col)] = int(idx)
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


def test_v32_raw_seed_and_visual_only_do_not_become_partition_cut() -> None:
    shape = (18, 24)
    free = np.zeros(shape, dtype=bool)
    free[9, 5:18] = True
    result = complete_voxel_doors_from_seeds(
        seed_result=_seed_result(shape, [(9, 10), (9, 11), (9, 12)]),
        free_map=free,
        base_partition_free=free,
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(partition_topology_enabled=True),
    )

    raw_seed = np.asarray(result.debug["voxel_door_raw_seed_mask"], dtype=bool)
    assert np.any(raw_seed)
    assert np.any(result.debug["voxel_door_seed_line_primitive_mask"])
    assert not np.any(result.door_cut_mask_for_partition)
    assert not np.any(result.debug["voxel_door_topology_effective_cut_mask"])


def test_v32_small_known_side_gate_does_not_override_topology_disabled_legacy_path() -> None:
    shape = (18, 24)
    free = np.zeros(shape, dtype=bool)
    free[8:11, 9:14] = True
    result = complete_voxel_doors_from_seeds(
        seed_result=_seed_result(shape, [(9, 10), (10, 12)]),
        free_map=free,
        base_partition_free=free,
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            min_seed_cells_for_accepted_extension=1,
            min_seed_line_length_cells_for_accepted_extension=1,
            min_seed_elongation_for_direction=1.0,
            seed_cluster_morph_close_radius_cells=0,
            seed_cluster_merge_distance_cells=0,
            seed_cluster_max_perpendicular_gap_cells=0,
            partition_topology_enabled=False,
            partition_reject_small_known_side_enabled=True,
        ),
    )

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["partition_small_known_side_topology_enabled"] is False
    assert candidate.debug["partition_small_known_side_rejected"] is False
    assert np.any(result.door_cut_mask_for_partition)
