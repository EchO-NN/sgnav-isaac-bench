from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    VoxelDoorMemory,
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


def test_stable_door_memory_keeps_last_valid_cut_after_candidate_disappears() -> None:
    shape = (18, 26)
    seed = _seed_result(shape, [(8, 12), (9, 12)])
    free = np.zeros(shape, dtype=bool)
    free[4:14, 5:21] = True
    anchors = np.zeros(shape, dtype=bool)
    anchors[3, 12] = True
    anchors[14, 12] = True
    cfg = _legacy_completion_config(wall_anchor_radius_cells=0, seed_cluster_morph_close_radius_cells=0)
    completion = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free & ~anchors,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
        real_wall_barrier_map=anchors,
    )
    assert np.any(completion.door_cut_mask_for_partition)

    memory = VoxelDoorMemory(cfg)
    first = memory.update(completion.candidates, step=1, shape=shape)
    second = memory.update([], step=2, shape=shape)

    assert np.any(first.stable_door_cut_mask)
    assert np.array_equal(first.stable_door_cut_mask, second.stable_door_cut_mask)
    assert second.debug["voxel_door_memory_track_count"] == 1
