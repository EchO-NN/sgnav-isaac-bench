from __future__ import annotations

import numpy as np

from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


def _cfg(**overrides: object) -> WallProjectionConfig:
    values = {
        "side_validation_enabled": False,
        "min_projected_line_length_m": 0.10,
        "min_projected_support_ratio": 0.05,
        "anchor_min_projected_line_length_m": 0.10,
        "anchor_min_projected_support_ratio": 0.05,
        "max_fill_gap_m": 0.50,
        "max_free_gap_ratio": 0.20,
        "min_seed_support_cells_per_projected_line": 1,
        "min_seed_support_ratio_per_projected_line": 0.0,
    }
    values.update(overrides)
    return WallProjectionConfig(**values)


def test_projected_wall_gap_fill_does_not_cross_unknown_or_outside_known() -> None:
    shape = (12, 20)
    seed = np.zeros(shape, dtype=bool)
    seed[5, 2:5] = True
    seed[5, 8:12] = True
    unknown_gap = np.zeros(shape, dtype=bool)
    unknown_gap[5, 5:8] = True

    result_unknown = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=np.zeros(shape, dtype=bool),
        vertical_free_map=np.zeros(shape, dtype=bool),
        unknown_map=unknown_gap,
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        projection_known_domain_map=np.ones(shape, dtype=bool),
        projection_gap_forbidden_unknown_map=unknown_gap,
        resolution_m=0.10,
        config=_cfg(),
    )
    projected_unknown = result_unknown.projected_wall_display_map
    assert projected_unknown is not None
    assert not np.any(projected_unknown[unknown_gap])
    assert result_unknown.debug["voxel_projected_wall_unknown_gap_reject_count"] >= 1

    outside_gap = np.zeros(shape, dtype=bool)
    outside_gap[5, 5:8] = True
    known = np.ones(shape, dtype=bool)
    known[outside_gap] = False
    result_outside = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=np.zeros(shape, dtype=bool),
        vertical_free_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        projection_known_domain_map=known,
        projection_gap_forbidden_unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_cfg(),
    )
    projected_outside = result_outside.projected_wall_display_map
    assert projected_outside is not None
    assert not np.any(projected_outside[outside_gap])
    assert result_outside.debug["voxel_projected_wall_outside_known_gap_reject_count"] >= 1


def test_projection_does_not_draw_middle_line_between_parallel_walls() -> None:
    shape = (14, 28)
    seed = np.zeros(shape, dtype=bool)
    seed[4, 2:24] = True
    seed[8, 2:24] = True
    seed[6, 12] = True
    free = np.zeros(shape, dtype=bool)
    free[5:8, 2:24] = True

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=np.zeros(shape, dtype=bool),
        vertical_free_map=free,
        unknown_map=np.zeros(shape, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        projection_known_domain_map=np.ones(shape, dtype=bool),
        projection_gap_forbidden_unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_cfg(
            projection_line_nms_lateral_radius_cells=2,
            projection_valley_min_parallel_peak_cells=3,
            projection_valley_min_peak_separation_cells=3,
            projection_valley_direct_seed_min_cells=2,
            projection_valley_direct_seed_ratio_min=0.05,
        ),
    )

    projected = result.projected_wall_display_map
    assert projected is not None
    assert int(np.count_nonzero(projected[6, 2:24])) <= 1
    assert np.any(projected[4, 2:24])
    assert np.any(projected[8, 2:24])
    assert result.debug["voxel_projected_wall_parallel_valley_reject_count"] >= 1


def test_normal_room_boundary_wall_projection_still_kept() -> None:
    shape = (12, 20)
    seed = np.zeros(shape, dtype=bool)
    seed[5, 2:5] = True
    seed[5, 6:11] = True

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=np.zeros(shape, dtype=bool),
        vertical_free_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        projection_known_domain_map=np.ones(shape, dtype=bool),
        projection_gap_forbidden_unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_cfg(),
    )

    projected = result.projected_wall_display_map
    assert projected is not None
    assert np.all(projected[5, 2:11])
    assert result.debug["voxel_projected_wall_unknown_gap_reject_count"] == 0
    assert result.debug["voxel_projected_wall_outside_known_gap_reject_count"] == 0
