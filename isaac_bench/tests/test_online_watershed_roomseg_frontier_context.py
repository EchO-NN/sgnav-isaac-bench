from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_watershed_roomseg.online_watershed_room_segmenter import (
    OnlineWatershedRoomSegConfig,
    run_online_watershed_roomseg,
)
from isaac_bench.mapping.online_watershed_roomseg.seeds import WatershedSeedConfig


def test_frontier_context_marks_room_entry_for_provisional_room_seed():
    free = np.zeros((40, 50), dtype=bool)
    free[8:32, 8:34] = True
    free[14:26, 34:42] = True
    wall = np.zeros_like(free)
    frontier = {"frontier_id": "door_like_frontier", "cells": [[19, 42], [20, 42], [21, 42]]}
    result = run_online_watershed_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=~free,
        vertical_profile=None,
        frontier_clusters=[frontier],
        config=OnlineWatershedRoomSegConfig(
            resolution_m=0.1,
            min_observed_free_cells=1,
            seeds=WatershedSeedConfig(
                confirmed_min_clearance_m=0.8,
                confirmed_min_distance_m=2.0,
                confirmed_min_component_area_m2=0.5,
                provisional_min_clearance_m=0.1,
                provisional_min_distance_from_confirmed_m=0.5,
            ),
        ),
    )

    contexts = result.frontier_room_context["frontiers"]
    assert len(contexts) == 1
    assert contexts[0]["room_label"] > 0
    assert contexts[0]["region_type"] in {"confirmed_room", "provisional_room"}
    assert contexts[0]["is_room_entry"] is True
