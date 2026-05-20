from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.wall_lines import (
    LineFilteringConfig,
    LineWallsConfig,
    extract_line_supported_walls,
    filter_and_snap_wall_lines,
)


def test_line_filtering_merges_broken_wall_and_rejects_speckles():
    wall = np.zeros((50, 80), dtype=bool)
    wall[20, 5:30] = True
    wall[20, 35:65] = True
    wall[8:10, 8:10] = True
    wall[35, 50] = True
    free = np.zeros_like(wall)
    free[21:35, 5:65] = True

    raw, _ = extract_line_supported_walls(wall, resolution_m=0.05)
    filtered, debug = filter_and_snap_wall_lines(
        raw,
        wall_candidate_clean=wall,
        free_clean=free,
        resolution_m=0.05,
        config=LineFilteringConfig(
            merge_collinear_gap_m=0.40,
            min_filtered_line_length_m=0.60,
            min_filtered_support_ratio=0.35,
            endpoint_min_wall_support_m=0.05,
            min_confidence=0.30,
        ),
    )

    assert len(raw) > len(filtered)
    assert any(line.length_m >= 2.5 for line in filtered)
    assert all(line.length_m >= 0.60 for line in filtered)
    assert debug["filtered_wall_line_count"] == len(filtered)


def test_line_filtering_counts_endpoint_cells_for_threshold_length():
    wall = np.zeros((24, 32), dtype=bool)
    wall[10, 5:17] = True  # 12 cells at 5 cm resolution is exactly 0.60 m.
    free = np.zeros_like(wall)
    free[11:16, 5:17] = True

    raw, _ = extract_line_supported_walls(wall, resolution_m=0.05)
    filtered, debug = filter_and_snap_wall_lines(
        raw,
        wall_candidate_clean=wall,
        free_clean=free,
        resolution_m=0.05,
        config=LineFilteringConfig(
            min_filtered_line_length_m=0.60,
            min_filtered_support_ratio=0.35,
            endpoint_min_wall_support_m=0.05,
            min_confidence=0.30,
        ),
    )

    assert debug["filtered_wall_line_count"] == 1
    assert 0.599 <= filtered[0].length_m <= 0.601


def test_line_filtering_treats_nearby_cells_as_same_wall_line():
    wall = np.zeros((32, 48), dtype=bool)
    wall[10, 5:13] = True
    wall[11, 13:22] = True
    wall[9, 22:30] = True
    free = np.zeros_like(wall)
    free[12:20, 5:30] = True

    raw, raw_debug = extract_line_supported_walls(
        wall,
        resolution_m=0.05,
        config=LineWallsConfig(
            hough_enabled=True,
            pca_enabled=False,
            min_line_length_m=0.90,
            min_support_ratio=0.35,
            axis_snap_lateral_tolerance_cells=2,
        ),
    )
    filtered, debug = filter_and_snap_wall_lines(
        raw,
        wall_candidate_clean=wall,
        free_clean=free,
        resolution_m=0.05,
        config=LineFilteringConfig(
            min_filtered_line_length_m=0.90,
            min_filtered_support_ratio=0.35,
            endpoint_min_wall_support_m=0.05,
            max_wall_thickness_m=0.20,
            min_confidence=0.30,
        ),
    )

    assert raw_debug["segment_count"] >= 1
    assert any(segment["source"] == "hough_axis_run_lateral_snap" for segment in raw_debug["segments"])
    assert debug["filtered_wall_line_count"] == 1
    line = filtered[0]
    assert int(round(float(line.p0_rc[0]))) == int(round(float(line.p1_rc[0])))
    assert line.length_m >= 1.20
