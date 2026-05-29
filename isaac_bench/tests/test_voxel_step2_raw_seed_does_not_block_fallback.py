from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import build_step2_line_pool
from isaac_bench.mapping.wall_projection import ProjectedWallLine


def test_v25_raw_seed_band_does_not_block_main_step2_source() -> None:
    shape = (16, 18)
    line = _projected_line(line_id=1, row=8, start=3, end=12, support_ratio=1.0)
    support = np.zeros(shape, dtype=bool)
    support[8, 3:13] = True
    raw_seed_band = np.zeros(shape, dtype=bool)
    raw_seed_band[8, 7] = True

    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        projected_display_lines=[],
        projected_source_lines=[line],
        source_support_map=support,
        door_suppression_band=np.zeros(shape, dtype=bool),
        raw_door_seed_band=raw_seed_band,
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        projected_source_line_map=support,
        shape=shape,
        resolution_m=0.10,
    )

    assert pool.debug["voxel_step2_blocked_by_raw_seed_count"] == 0
    assert pool.debug["voxel_step2_source_line_count"] > 0
    assert np.any(pool.source_line_map[8, 3:13])


def test_v25_corridor_neck_source_ignores_raw_seed_but_not_accepted_door_band() -> None:
    shape = (16, 18)
    short_line = _projected_line(line_id=2, row=8, start=6, end=9, support_ratio=1.0)
    support = np.zeros(shape, dtype=bool)
    support[8, 6:10] = True
    raw_seed_band = np.zeros(shape, dtype=bool)
    raw_seed_band[8, 7] = True
    free = np.zeros(shape, dtype=bool)
    free[6:11, 4:13] = True

    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        projected_display_lines=[],
        projected_source_lines=[],
        projected_corridor_neck_source_lines=[short_line],
        source_support_map=support,
        door_suppression_band=np.zeros(shape, dtype=bool),
        raw_door_seed_band=raw_seed_band,
        vertical_free_map=free,
        unknown_ratio_map=np.zeros(shape, dtype=np.float32),
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        projected_source_line_map=np.zeros(shape, dtype=bool),
        shape=shape,
        resolution_m=0.10,
        corridor_neck_source_enabled=True,
        corridor_neck_source_min_line_length_m=0.35,
        corridor_neck_source_min_support_cells=4,
        corridor_neck_source_min_support_ratio=0.45,
        corridor_neck_source_min_adjacent_free_area_cells=20,
        corridor_neck_source_forbid_raw_seed_band=False,
        corridor_neck_source_forbid_accepted_door_band=True,
    )

    assert pool.debug["voxel_step2_blocked_by_raw_seed_count"] == 0
    assert pool.debug["voxel_step2_source_line_count_corridor_neck"] == 1
    assert np.any(pool.source_line_map[8, 6:10])


def _projected_line(*, line_id: int, row: int, start: int, end: int, support_ratio: float) -> ProjectedWallLine:
    return ProjectedWallLine(
        line_id=int(line_id),
        axis="h",
        line=int(row),
        start=int(start),
        end=int(end),
        support_cell_count=max(0, int(end) - int(start) + 1),
        projected_cell_count=max(0, int(end) - int(start) + 1),
        support_ratio=float(support_ratio),
        lateral_std_cells=0.0,
        source="axis_accumulator",
        reject_reason=None,
        side_free_ratio_a=0.20,
        side_free_ratio_b=0.80,
        side_unknown_ratio_a=0.0,
        side_unknown_ratio_b=0.0,
        side_nonfree_ratio_a=0.50,
        side_nonfree_ratio_b=0.0,
        structural_side_score=0.50,
    )
