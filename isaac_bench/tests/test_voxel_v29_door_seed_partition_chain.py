from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    DOOR_COMPLETION_STRONG_SEED_CENTERLINE,
    VoxelDoorDetectorConfig,
    VoxelDoorLineCandidate,
    VoxelDoorSeedResult,
    _batch_reject_conflicting_doors,
    complete_voxel_doors_from_seeds,
)
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import accepted_seed_mask_from_candidates


def _seed_result(shape: tuple[int, int], components: list[list[tuple[int, int]]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for label, cells in enumerate(components, start=1):
        for r, c in cells:
            seed[int(r), int(c)] = True
            labels[int(r), int(c)] = int(label)
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


def _completion_config(**overrides: object) -> VoxelDoorDetectorConfig:
    data = {
        "min_seed_cells_for_accepted_extension": 1,
        "min_seed_line_length_cells_for_accepted_extension": 1,
        "min_seed_elongation_for_direction": 1.0,
        "accepted_orientation_mode": "legacy",
        "local_free_neck_orientation_debug_only": False,
        "wall_anchor_radius_cells": 0,
        "seed_cluster_morph_close_radius_cells": 0,
    }
    data.update(overrides)
    return VoxelDoorDetectorConfig(**data)


def test_v29_geometry_valid_seed_line_stays_visual_only_without_topology_effect() -> None:
    shape = (20, 28)
    seed = _seed_result(shape, [[(10, 13), (10, 14)]])
    free = np.zeros(shape, dtype=bool)
    free[10, 6:22] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[10, 6] = True
    anchors[10, 21] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        free_map_for_visual_validation=free,
        base_partition_free=np.zeros(shape, dtype=bool),
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_completion_config(),
        real_wall_barrier_map=anchors,
    )

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["partition_geometry_accepted"] is True
    assert candidate.debug["partition_topology_effective"] is False
    assert candidate.debug["partition_accepted"] is False
    assert not np.any(result.door_cut_mask_for_partition)
    assert not np.any(result.door_topology_effective_cut_mask)
    assert np.any(result.door_geometry_warning_cut_mask)
    assert np.any(result.door_visual_only_mask)


def test_v29_accepted_seed_mask_ignores_visual_only_candidates() -> None:
    shape = (8, 8)
    cluster_map = np.zeros(shape, dtype=np.int32)
    cluster_map[3, 2:5] = 1

    visual_only = SimpleNamespace(debug={"seed_group_id": 1, "visual_accepted": True, "partition_accepted": False, "partition_topology_effective": False})
    effective = SimpleNamespace(debug={"seed_group_id": 1, "visual_accepted": True, "partition_accepted": False, "partition_topology_effective": True})

    assert not np.any(accepted_seed_mask_from_candidates([visual_only], cluster_map, shape))
    assert np.any(accepted_seed_mask_from_candidates([effective], cluster_map, shape))


def test_v29_raw_seed_overlap_is_debugged_but_not_used_as_partition_conflict() -> None:
    shape = (12, 14)
    all_seed = np.zeros(shape, dtype=bool)
    all_seed[5, 7] = True
    candidate = VoxelDoorLineCandidate(
        candidate_id=1,
        seed_component_id=1,
        seed_cells=[(5, 3)],
        center_rc=(5.0, 4.0),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(1.0, 0.0),
        seed_projected_centerline_cells=[(5, 3)],
        extended_centerline_cells=[(5, c) for c in range(3, 9)],
        door_cut_cells=[(5, c) for c in range(3, 9)],
        wall_anchor_a=(5, 2),
        wall_anchor_b=(5, 9),
        width_m=0.60,
        accepted=True,
        reject_reason=None,
        debug={"partition_accepted": True, "partition_topology_effective": True, "score": 1.0},
    )

    debug = _batch_reject_conflicting_doors([candidate], all_seed, shape, VoxelDoorDetectorConfig(seed_cluster_merge_distance_cells=0))

    assert candidate.accepted is True
    assert candidate.debug["partition_accepted"] is True
    assert candidate.debug["raw_seed_conflict_ignored_cells"] == 1
    assert debug["voxel_door_partition_reject_raw_seed_conflict_count"] == 0
    assert debug["voxel_door_raw_seed_conflict_ignored_cells"] == 1


def test_v29_strong_seed_centerline_fallback_must_be_topology_effective_to_partition() -> None:
    shape = (20, 40)
    seed_cells = [(10, col) for col in range(16, 20)]
    seed = _seed_result(shape, [seed_cells])
    free = np.zeros(shape, dtype=bool)
    free[10, 5:33] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free,
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_completion_config(
            partition_cut_bridge_unknown_max_cells=0,
            partition_cut_bridge_nonfree_max_cells=0,
            partition_topology_min_side_area_cells=1,
            strong_seed_centerline_min_elongation=1.0,
        ),
    )

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["completion_mode"] == DOOR_COMPLETION_STRONG_SEED_CENTERLINE
    assert candidate.debug["door_wall_attached"] is False
    assert candidate.debug["partition_topology_effective"] is True
    assert candidate.debug["partition_accepted"] is True
    assert np.any(result.door_topology_effective_cut_mask)
    assert np.array_equal(result.door_cut_mask_for_partition, result.door_topology_effective_cut_mask)
