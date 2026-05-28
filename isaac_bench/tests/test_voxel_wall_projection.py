from __future__ import annotations

import numpy as np

from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_lines


def test_wall_projection_snaps_decorative_noise_to_main_wall_line() -> None:
    shape = (32, 40)
    raw = np.zeros(shape, dtype=bool)
    raw[11, 5:26] = True
    raw[10, 6:26:4] = True
    raw[12, 8:26:4] = True
    free = np.zeros(shape, dtype=bool)
    ratio = raw.astype(np.float32)

    result = project_wall_evidence_to_lines(
        wall_raw=raw,
        free_map=free,
        occupied_ratio=ratio,
        resolution_m=0.05,
        config=WallProjectionConfig(min_projected_line_length_m=0.30, side_validation_enabled=False),
    )

    rows, cols = np.nonzero(result.projected_wall_map)
    assert rows.size > 0
    assert int(rows.max() - rows.min()) == 0
    assert int(cols.max() - cols.min()) >= 18
    assert np.count_nonzero(result.support_map) == np.count_nonzero(raw)


def test_parallel_near_walls_are_not_merged() -> None:
    shape = (32, 40)
    raw = np.zeros(shape, dtype=bool)
    raw[10, 5:28] = True
    raw[15, 5:28] = True
    free = np.zeros(shape, dtype=bool)
    free[12:14, 5:28] = True

    result = project_wall_evidence_to_lines(
        wall_raw=raw,
        free_map=free,
        occupied_ratio=raw.astype(np.float32),
        resolution_m=0.05,
        config=WallProjectionConfig(min_projected_line_length_m=0.30, side_validation_enabled=False),
    )

    lines = [line for line in result.projected_lines if line.reject_reason is None]
    assert len(lines) == 2
    assert sorted(line.line for line in lines) == [10, 15]


def test_small_unknown_gap_fills_but_free_gap_does_not() -> None:
    shape = (24, 48)
    raw = np.zeros(shape, dtype=bool)
    raw[10, 2:8] = True
    raw[10, 10:16] = True
    raw[10, 22:27] = True
    raw[10, 32:38] = True
    free = np.zeros(shape, dtype=bool)
    free[10, 27:32] = True

    result = project_wall_evidence_to_lines(
        wall_raw=raw,
        free_map=free,
        occupied_ratio=raw.astype(np.float32),
        resolution_m=0.05,
        config=WallProjectionConfig(min_projected_line_length_m=0.20, max_fill_gap_m=0.25, max_free_gap_ratio=0.20, side_validation_enabled=False),
    )

    projected = result.projected_wall_map
    assert np.all(projected[10, 8:10])
    assert not np.any(projected[10, 27:32])
