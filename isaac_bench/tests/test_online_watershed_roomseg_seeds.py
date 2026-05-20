from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_watershed_roomseg.corridor import WatershedCorridorConfig, compute_watershed_corridors
from isaac_bench.mapping.online_watershed_roomseg.distance import compute_watershed_distance_fields
from isaac_bench.mapping.online_watershed_roomseg.seeds import WatershedSeedConfig, build_watershed_markers


def test_seed_priority_corridor_confirmed_then_provisional():
    free = np.zeros((40, 60), dtype=bool)
    free[5:35, 5:35] = True
    free[17:23, 35:55] = True
    wall = np.zeros_like(free)
    wall[4, 5:35] = wall[35, 5:35] = True
    wall[5:35, 4] = True
    fields = compute_watershed_distance_fields(free_clean=free, wall_candidate_clean=wall, resolution_m=0.1)
    corridor = compute_watershed_corridors(
        free_clean=free,
        dist_free_extent_m=fields.dist_free_extent_m,
        resolution_m=0.1,
        config=WatershedCorridorConfig(max_width_m=0.9, min_length_m=1.0, min_aspect_ratio=2.0),
    )
    frontier = {"frontier_id": "f0", "cells": [[20, 55], [21, 55]]}
    seeds = build_watershed_markers(
        free_clean=free,
        dist_struct_m=fields.dist_struct_m,
        corridor_seed_mask=corridor.corridor_seeds,
        frontier_clusters=[frontier],
        resolution_m=0.1,
        config=WatershedSeedConfig(
            confirmed_min_clearance_m=0.4,
            confirmed_min_distance_m=1.0,
            confirmed_min_component_area_m2=0.5,
            provisional_min_clearance_m=0.1,
            provisional_min_distance_from_confirmed_m=0.3,
        ),
    )

    assert np.any(seeds.corridor_seeds)
    assert np.any(seeds.confirmed_room_seeds)
    assert any(v == "corridor" for v in seeds.seed_type_by_label.values())
    assert any(v == "confirmed_room" for v in seeds.seed_type_by_label.values())
    assert seeds.report["priority"] == ["corridor", "confirmed_room", "provisional_room"]


def test_provisional_seed_rejects_corridor_only_frontier():
    free = np.zeros((20, 50), dtype=bool)
    free[8:12, 3:45] = True
    wall = np.zeros_like(free)
    fields = compute_watershed_distance_fields(free_clean=free, wall_candidate_clean=wall, resolution_m=0.1)
    corridor = np.zeros_like(free)
    corridor[8:12, 3:45] = True
    seeds = build_watershed_markers(
        free_clean=free,
        dist_struct_m=fields.dist_struct_m,
        corridor_seed_mask=corridor,
        frontier_clusters=[{"id": "corridor_exit", "cells": [[9, 45], [10, 45]]}],
        resolution_m=0.1,
        config=WatershedSeedConfig(provisional_min_clearance_m=0.1),
    )

    states = [item["state"] for item in seeds.report["frontier_room_seeds"]]
    assert states == ["rejected"]
    assert not np.any(seeds.frontier_room_seeds)
