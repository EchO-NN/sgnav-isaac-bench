from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.corridor_l_corner_v6_2 import detect_corridor_l_corners_v6_2


def test_l_corner_extension_accepts_wall_to_wall_separator():
    wall, free = _l_corner_scene()

    result = detect_corridor_l_corners_v6_2(
        wall_line_support_mask=wall,
        stable_free_mask=free,
        door_mask=np.zeros_like(wall),
        resolution_m=0.10,
        config={"corridor": {"extension_free_ratio_min": 0.85, "min_split_area_m2": 0.25}},
    )

    assert len(result.accepted) >= 1
    assert int(np.count_nonzero(result.accepted_separator_mask)) > 0


def test_l_corner_extension_rejects_low_free_ratio():
    wall, free = _l_corner_scene()
    free[10, 13:18] = False

    result = detect_corridor_l_corners_v6_2(
        wall_line_support_mask=wall,
        stable_free_mask=free,
        door_mask=np.zeros_like(wall),
        resolution_m=0.10,
        config={"corridor": {"extension_free_ratio_min": 0.85, "min_split_area_m2": 0.25}},
    )

    assert len(result.accepted) == 0
    assert len(result.rejected) >= 1


def _l_corner_scene() -> tuple[np.ndarray, np.ndarray]:
    wall = np.zeros((30, 30), dtype=bool)
    wall[10, 0:11] = True
    wall[0:11, 10] = True
    wall[10:30, 22] = True
    free = np.ones_like(wall, dtype=bool) & ~wall
    return wall, free
