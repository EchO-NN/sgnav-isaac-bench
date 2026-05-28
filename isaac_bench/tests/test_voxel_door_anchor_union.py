from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    DOOR_ANCHOR_PROJECTED,
    DOOR_ANCHOR_STEP1,
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
        debug={"voxel_door_seed_mask": seed, "voxel_door_seed_component_map": labels},
    )


def _complete_with_anchor_source(source_code: int):
    shape = (24, 28)
    seed_result = _seed_result(shape, [(12, 12), (12, 13), (12, 14)])
    free = np.zeros(shape, dtype=bool)
    free[12, 6:22] = True
    anchor_wall = np.zeros(shape, dtype=bool)
    anchor_wall[12, 6] = True
    anchor_wall[12, 21] = True
    source = np.zeros(shape, dtype=np.uint8)
    source[anchor_wall] = int(source_code)
    result = complete_voxel_doors_from_seeds(
        seed_result=seed_result,
        free_map=free,
        anchor_wall_map=anchor_wall,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            min_seed_component_cells=1,
            extend_max_m=2.0,
            wall_anchor_radius_cells=0,
            line_fit_max_residual_cells=0.75,
            inner_unknown_ratio_max=0.20,
            inner_wall_ratio_max=0.10,
            inner_free_or_seed_ratio_min=0.40,
            partition_topology_enabled=False,
        ),
        anchor_source_map=source,
    )
    assert int(result.debug["voxel_door_accepted_count"]) == 1
    return result


def test_door_seed_can_complete_against_projected_anchor_only() -> None:
    result = _complete_with_anchor_source(DOOR_ANCHOR_PROJECTED)

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["anchor_a_source"] == "projected"
    assert candidate.debug["anchor_b_source"] == "projected"
    assert np.any(result.door_centerline_visual_mask)
    assert np.any(result.door_cut_mask_for_partition)
    assert int(np.count_nonzero(result.door_centerline_visual_mask)) > int(np.count_nonzero(result.door_cut_mask_for_partition))


def test_door_completion_reports_step1_anchor_source() -> None:
    result = _complete_with_anchor_source(DOOR_ANCHOR_STEP1)

    candidate = next(item for item in result.candidates if item.accepted)
    assert candidate.debug["anchor_a_source"] == "step1"
    assert candidate.debug["anchor_b_source"] == "step1"
    assert result.debug["voxel_door_anchor_source_counts"]["step1"] == 2
