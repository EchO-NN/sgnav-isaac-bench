from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.door_wall_repair_v6_2 import repair_door_wall_v6_2


def test_repair_keeps_door_and_wall_separate_and_does_not_fill_unknown():
    wall = np.zeros((9, 10), dtype=bool)
    door = np.zeros_like(wall)
    unknown = np.zeros_like(wall)
    wall[4, 2] = True
    wall[4, 4] = True
    door[4, 6:8] = True
    unknown[4, 5] = True

    result = repair_door_wall_v6_2(
        wall_mask_raw=wall,
        door_mask_raw=door,
        unknown_mask=unknown,
        resolution_m=0.10,
        config={
            "repair": {
                "enabled": True,
                "close_kernel_m": 0.10,
                "remove_island_max_area_m2": 0.0,
                "preserve_door_class": True,
            }
        },
    )

    assert bool(result.wall_mask[4, 3])
    assert not bool(result.wall_mask[4, 5])
    assert not bool(result.door_mask[4, 5])
    assert bool(result.door_mask[4, 6])
    assert not np.any(result.wall_mask & result.door_mask)
    assert np.array_equal(result.wall_line_support_mask, result.wall_mask | result.door_mask)
