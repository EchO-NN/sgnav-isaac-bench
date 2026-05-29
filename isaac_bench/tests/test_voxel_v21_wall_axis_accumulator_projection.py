from __future__ import annotations

import numpy as np

from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


def test_v21_axis_accumulator_merges_fragmented_wall_support() -> None:
    shape = (20, 24)
    support = np.zeros(shape, dtype=bool)
    support[10, 4:7] = True
    support[10, 10:15] = True
    weight = support.astype(np.float32)
    free = np.zeros(shape, dtype=bool)
    free[11:15, 4:15] = True
    unknown = ~free & ~support

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_map=support,
        support_weight=weight,
        vertical_free_map=free,
        unknown_map=unknown,
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(
            max_fill_gap_m=0.40,
            min_projected_line_length_m=0.30,
            min_projected_support_ratio=0.25,
        ),
    )

    assert np.all(result.projected_wall_display_map[10, 4:15])
    assert result.debug["voxel_wall_projection_mode"] == "axis_accumulator"
    assert result.debug["voxel_projected_wall_step2_source_cells"] >= 11


def test_v23_axis_accumulator_keeps_both_sides_free_as_debug_only() -> None:
    shape = (18, 18)
    support = np.zeros(shape, dtype=bool)
    support[8, 5:12] = True
    free = np.ones(shape, dtype=bool)

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_map=support,
        support_weight=support.astype(np.float32),
        vertical_free_map=free,
        unknown_map=np.zeros(shape, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.30),
    )

    assert np.any(result.projected_wall_display_map)
    assert result.debug["voxel_wall_projection_reject_reason_counts"].get("projected_wall_both_sides_free_furniture_like", 0) == 0
    assert any(bool(line.debug.get("both_sides_free_like", False)) for line in result.projected_display_lines)
