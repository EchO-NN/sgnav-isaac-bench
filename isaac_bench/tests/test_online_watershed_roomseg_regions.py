from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_watershed_roomseg.online_watershed_room_segmenter import (
    OnlineWatershedRoomSegConfig,
    run_online_watershed_roomseg,
)
from isaac_bench.mapping.online_watershed_roomseg.seeds import WatershedSeedConfig


def test_narrow_neck_splits_two_room_lobes():
    free = np.zeros((50, 90), dtype=bool)
    free[8:42, 8:35] = True
    free[8:42, 55:82] = True
    free[22:28, 35:55] = True
    wall = np.zeros_like(free)
    result = run_online_watershed_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=~free,
        vertical_profile=None,
        config=OnlineWatershedRoomSegConfig(
            resolution_m=0.1,
            min_observed_free_cells=1,
            seeds=WatershedSeedConfig(confirmed_min_clearance_m=0.4, confirmed_min_distance_m=2.0, confirmed_min_component_area_m2=0.5),
        ),
    )

    left = int(result.room_label_map[25, 20])
    right = int(result.room_label_map[25, 68])
    assert left > 0 and right > 0 and left != right
    assert result.debug["marker_controlled_watershed"] is True


def test_large_open_room_not_oversegmented():
    free = np.zeros((60, 60), dtype=bool)
    free[10:50, 10:50] = True
    wall = np.zeros_like(free)
    result = run_online_watershed_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=~free,
        vertical_profile=None,
        config=OnlineWatershedRoomSegConfig(
            resolution_m=0.1,
            min_observed_free_cells=1,
            seeds=WatershedSeedConfig(confirmed_min_clearance_m=0.4, confirmed_min_distance_m=1.0, confirmed_min_component_area_m2=0.5),
        ),
    )

    labels = [int(v) for v in np.unique(result.room_label_map) if int(v) > 0]
    assert len(labels) == 1


def test_corridor_region_is_not_merged_into_room():
    free = np.zeros((50, 90), dtype=bool)
    free[10:40, 8:35] = True
    free[22:28, 35:80] = True
    wall = np.zeros_like(free)
    result = run_online_watershed_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=~free,
        vertical_profile=None,
        config=OnlineWatershedRoomSegConfig(resolution_m=0.1, min_observed_free_cells=1),
    )

    types = {str(info["region_type"]) for info in result.region_infos}
    assert "confirmed_room" in types
    assert "corridor" in types
