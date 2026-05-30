from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import merge_small_enclosed_single_neighbor_regions
from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


def test_v28_projected_display_wall_stays_inside_known_domain() -> None:
    shape = (10, 16)
    seed = np.zeros(shape, dtype=bool)
    seed[4, 2:6] = True
    seed[4, 9:13] = True
    outside = np.zeros(shape, dtype=bool)
    outside[4, 6:9] = True
    known = np.ones(shape, dtype=bool)
    known[outside] = False

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=np.zeros(shape, dtype=bool),
        vertical_free_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        projection_known_domain_map=known,
        projection_gap_forbidden_unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(
            side_validation_enabled=False,
            min_projected_line_length_m=0.10,
            min_projected_support_ratio=0.05,
            anchor_min_projected_line_length_m=0.10,
            anchor_min_projected_support_ratio=0.05,
            max_fill_gap_m=0.50,
            min_seed_support_cells_per_projected_line=1,
            min_seed_support_ratio_per_projected_line=0.0,
        ),
    )

    projected = result.projected_wall_display_map
    assert projected is not None
    assert not np.any(projected & ~known)
    assert result.debug["voxel_projected_wall_outside_known_gap_reject_count"] >= 1


def test_v28_small_single_neighbor_merge_is_label_only() -> None:
    labels = np.ones((24, 24), dtype=np.int32)
    labels[:2, :] = 0
    labels[-2:, :] = 0
    labels[:, :2] = 0
    labels[:, -2:] = 0
    labels[11:13, 11:13] = 2
    free = labels > 0
    free_before = free.copy()

    merged, debug = merge_small_enclosed_single_neighbor_regions(
        labels,
        partition_free=free,
        partition_unknown=np.zeros_like(free),
        final_separator_map=np.zeros_like(free),
        resolution_m=0.10,
        max_area_m2=1.50,
    )

    assert np.array_equal(free, free_before)
    assert not np.any(merged == 2)
    assert debug["voxel_small_enclosed_merge_merged_count"] == 1
