from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.wall_lines import FilteredWallLine
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import build_step2_line_pool


def _line(line_id: int, p0: tuple[int, int], p1: tuple[int, int], *, confidence: float = 0.8) -> FilteredWallLine:
    a = np.asarray(p0, dtype=np.float32)
    b = np.asarray(p1, dtype=np.float32)
    delta = b - a
    theta = float(np.arctan2(float(delta[0]), float(delta[1])))
    return FilteredWallLine(
        line_id=int(line_id),
        p0_rc=a,
        p1_rc=b,
        theta=theta,
        normal_theta=float(theta + np.pi / 2.0),
        length_m=float(np.linalg.norm(delta) * 0.10),
        support_ratio=0.9,
        mean_wall_score=0.9,
        thickness_m=0.10,
        source_segment_ids=[int(line_id)],
        confidence=float(confidence),
    )


def test_step2_line_pool_uses_filtered_and_extension_seed_lines() -> None:
    shape = (20, 20)
    filtered = [_line(1, (5, 2), (5, 8))]
    relaxed = [_line(2, (12, 2), (12, 7))]

    pool = build_step2_line_pool(
        filtered_lines=filtered,
        extension_seed_lines=relaxed,
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        shape=shape,
        resolution_m=0.10,
    )

    assert len(pool.source_lines) == 2
    assert pool.debug["voxel_step2_filtered_line_count"] == 1
    assert pool.debug["voxel_step2_extension_seed_line_count"] == 1
    assert np.any(pool.source_line_map[5, 2:9])
    assert np.any(pool.source_line_map[12, 2:8])


def test_step2_target_wall_excludes_extension_seed_line_map_by_default() -> None:
    shape = (20, 20)
    extension_seed_map = np.zeros(shape, dtype=bool)
    extension_seed_map[10, 4:12] = True

    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=extension_seed_map,
        shape=shape,
        resolution_m=0.10,
    )

    assert not np.any(pool.target_wall_map[10, 4:12])
    assert not np.any(pool.target_source_map[10, 4:12])
    assert pool.debug["voxel_step2_target_source_counts"]["extension_seed_line"] == 0
