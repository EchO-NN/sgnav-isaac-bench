from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_watershed_roomseg.corridor import WatershedCorridorConfig, compute_watershed_corridors
from isaac_bench.mapping.online_watershed_roomseg.distance import compute_watershed_distance_fields
from isaac_bench.mapping.online_watershed_roomseg.online_watershed_room_segmenter import (
    OnlineWatershedRoomSegConfig,
    run_online_watershed_roomseg,
)


def test_long_thin_component_becomes_corridor_region_without_fragmentation():
    free = np.zeros((30, 80), dtype=bool)
    free[12:18, 5:75] = True
    wall = np.zeros_like(free)
    fields = compute_watershed_distance_fields(free_clean=free, wall_candidate_clean=wall, resolution_m=0.1)
    corridor = compute_watershed_corridors(
        free_clean=free,
        dist_free_extent_m=fields.dist_free_extent_m,
        resolution_m=0.1,
        config=WatershedCorridorConfig(max_width_m=0.9, min_length_m=2.0, min_aspect_ratio=4.0),
    )

    assert np.any(corridor.corridor_core)
    assert np.any(corridor.corridor_seeds)

    result = run_online_watershed_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=~(free | wall),
        vertical_profile=None,
        config=OnlineWatershedRoomSegConfig(
            resolution_m=0.1,
            corridor=WatershedCorridorConfig(max_width_m=0.9, min_length_m=2.0, min_aspect_ratio=4.0),
            min_observed_free_cells=1,
        ),
    )
    corridor_infos = [info for info in result.region_infos if info["region_type"] == "corridor"]
    assert len(corridor_infos) == 1
