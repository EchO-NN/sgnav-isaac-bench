from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import build_step2_door_reject_mask, build_step2_line_pool
from isaac_bench.mapping.wall_projection import ProjectedWallLine


def _projected_line() -> ProjectedWallLine:
    return ProjectedWallLine(
        line_id=1,
        axis="h",
        line=8,
        start=3,
        end=12,
        support_cell_count=8,
        projected_cell_count=10,
        support_ratio=0.80,
        lateral_std_cells=0.0,
        source="axis_accumulator",
        reject_reason=None,
        side_free_ratio_a=0.0,
        side_free_ratio_b=1.0,
        side_unknown_ratio_a=1.0,
        side_unknown_ratio_b=0.0,
        side_nonfree_ratio_a=1.0,
        side_nonfree_ratio_b=0.0,
        structural_side_score=0.75,
    )


def test_v21_step2_source_uses_projected_wall_line_objects() -> None:
    shape = (18, 18)
    projected_source_map = np.zeros(shape, dtype=bool)
    projected_source_map[8, 3:13] = True

    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        projected_display_lines=[],
        projected_source_lines=[_projected_line()],
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        projected_source_line_map=projected_source_map,
        shape=shape,
        resolution_m=0.10,
    )

    assert pool.debug["voxel_step2_projected_source_line_count"] == 1
    assert len(pool.source_lines) == 1
    assert np.any(pool.source_line_map[8, 3:13])
    assert np.all(pool.target_wall_map[8, 3:13])
    assert np.all(pool.target_source_map[8, 3:13] == 7)


def test_v21_step2_door_reject_mask_ignores_seed_and_visual_by_default() -> None:
    shape = (12, 12)
    seed = np.zeros(shape, dtype=bool)
    seed[5, 5] = True
    visual = np.zeros(shape, dtype=bool)
    visual[5, 6] = True
    cut = np.zeros(shape, dtype=bool)

    default_mask = build_step2_door_reject_mask(
        current_door_cut_mask=cut,
        stable_door_cut_mask=np.zeros(shape, dtype=bool),
        accepted_door_visual_mask=visual,
        accepted_seed_cluster_mask=seed,
        door_intersection_dilation_cells=1,
    )

    assert not np.any(default_mask)

    opt_in_mask = build_step2_door_reject_mask(
        current_door_cut_mask=cut,
        stable_door_cut_mask=np.zeros(shape, dtype=bool),
        accepted_door_visual_mask=visual,
        accepted_seed_cluster_mask=seed,
        door_intersection_dilation_cells=1,
        reject_if_intersects_door_visual=True,
        reject_if_intersects_door_seed=True,
    )

    assert opt_in_mask[5, 5]
    assert opt_in_mask[5, 6]
