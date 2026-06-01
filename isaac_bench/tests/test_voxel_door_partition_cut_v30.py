from __future__ import annotations

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import conn
from isaac_bench.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    VoxelDoorLineCandidate,
    VoxelDoorMemory,
    VoxelDoorSeedResult,
    build_door_partition_cut_v30,
    complete_voxel_doors_from_seeds,
)


def _seed_result(shape: tuple[int, int], cells: list[tuple[int, int]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for r, c in cells:
        seed[int(r), int(c)] = True
        labels[int(r), int(c)] = 1
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
        "partition_topology_min_side_area_cells": 1,
    }
    data.update(overrides)
    return VoxelDoorDetectorConfig(**data)


def test_v30_anchor_to_anchor_cut_closes_to_wall_and_splits_free_component() -> None:
    shape = (18, 26)
    free = np.zeros(shape, dtype=bool)
    free[4:14, 5:21] = True
    wall = np.zeros(shape, dtype=bool)
    wall[3, 12] = True
    wall[14, 12] = True
    seed = np.zeros(shape, dtype=bool)
    seed[8:10, 12] = True
    full_line = [(row, 12) for row in range(3, 15)]

    result = build_door_partition_cut_v30(
        full_line_cells=full_line,
        seed_mask=seed,
        accepted_seed_mask=seed,
        partition_free=free,
        partition_unknown=np.zeros(shape, dtype=bool),
        real_wall_barrier=wall,
        anchor_a=(3, 12),
        anchor_b=(14, 12),
        max_unknown_bridge_gap_cells=1,
        max_nonfree_bridge_gap_cells=0,
        max_endpoint_wall_gap_cells=1,
        seed_dilation_cells=0,
        min_cut_cells=1,
    )

    before = ndimage.label(free & ~wall, structure=conn(4))[1]
    after = ndimage.label((free & ~wall) & ~result.mask, structure=conn(4))[1]
    assert result.debug["door_partition_cut_v30_closed_to_wall"] is True
    assert int(np.count_nonzero(result.mask)) > int(np.count_nonzero(seed))
    assert after > before


def test_v30_attachment_only_candidate_is_not_effective_or_stable() -> None:
    shape = (16, 20)
    seed_cells = [(8, 7), (8, 8), (8, 9)]
    seed = _seed_result(shape, seed_cells)
    free = np.zeros(shape, dtype=bool)
    free[8, 7:10] = True
    wall = np.zeros(shape, dtype=bool)
    wall[8, 6] = True
    wall[8, 10] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        free_map_for_visual_validation=free,
        base_partition_free=free,
        anchor_wall_map=wall,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_completion_config(),
        real_wall_barrier_map=wall,
    )

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["partition_closure_attached"] is True
    assert candidate.debug["partition_topology_gain"] is False
    assert candidate.debug["partition_effective_verified"] is False
    assert not np.any(result.door_cut_mask_for_partition)

    memory = VoxelDoorMemory(_completion_config())
    memory_result = memory.update(result.candidates, step=1, shape=shape)
    assert not np.any(memory_result.stable_door_cut_mask)
    assert memory_result.debug["voxel_door_memory_verified_create_count"] == 0


def test_v30_door_partition_rejects_small_known_side() -> None:
    shape = (28, 32)
    seed_cells = [(12, 8), (13, 8)]
    seed = _seed_result(shape, seed_cells)
    free = np.zeros(shape, dtype=bool)
    free[5:21, 5:25] = True
    wall = np.zeros(shape, dtype=bool)
    wall[4, 8] = True
    wall[21, 8] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        free_map_for_visual_validation=free,
        base_partition_free=free,
        anchor_wall_map=wall,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_completion_config(
            wall_anchor_radius_cells=0,
            seed_cluster_morph_close_radius_cells=0,
            partition_topology_min_side_area_cells=1,
            partition_reject_small_known_side_enabled=True,
            partition_small_known_side_area_m2=2.0,
            partition_small_known_side_unknown_ratio_max=0.20,
        ),
        real_wall_barrier_map=wall,
    )

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["partition_small_known_side_rejected"] is True
    assert candidate.debug["reject_reason_partition"] == "door_partition_small_known_side_low_unknown"
    assert candidate.debug["partition_accepted"] is False
    assert not np.any(result.door_cut_mask_for_partition)


def _candidate(candidate_id: int, *, verified: bool) -> VoxelDoorLineCandidate:
    cut_cells = [(5, col) for col in range(3, 9)] if verified else []
    visual_cells = [(5, col) for col in range(3, 9)]
    return VoxelDoorLineCandidate(
        candidate_id=candidate_id,
        seed_component_id=1,
        seed_cells=[(5, 5), (5, 6)],
        center_rc=(5.0, 5.5),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(1.0, 0.0),
        seed_projected_centerline_cells=[(5, 5), (5, 6)],
        extended_centerline_cells=visual_cells,
        door_cut_cells=cut_cells,
        wall_anchor_a=(5, 2),
        wall_anchor_b=(5, 9),
        width_m=0.60,
        accepted=True,
        reject_reason=None,
        debug={
            "partition_accepted": bool(verified),
            "partition_effective_verified": bool(verified),
            "partition_geometry_accepted": bool(verified),
            "stable_memory_refresh_eligible": True,
            "score": 1.0,
        },
    )


def test_v30_stable_door_memory_weak_refresh_keeps_verified_cut() -> None:
    shape = (12, 14)
    memory = VoxelDoorMemory(VoxelDoorDetectorConfig(door_memory_decay_per_update=0.20, door_memory_weak_refresh_increment=0.15))

    first = memory.update([_candidate(1, verified=True)], step=1, shape=shape)
    assert np.any(first.stable_door_cut_mask)
    weak_refresh_count = 0
    for step in range(2, 6):
        result = memory.update([_candidate(step, verified=False)], step=step, shape=shape)
        weak_refresh_count += int(result.debug["voxel_door_memory_weak_refresh_count"])
        assert np.any(result.stable_door_cut_mask)
        assert result.debug["voxel_door_memory_track_count"] == 1

    assert weak_refresh_count >= 1
