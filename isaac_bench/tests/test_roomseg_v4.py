from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier_stability_v4 import stable_frontier_cells_v4
from isaac_bench.mapping.online_roomseg.corridor_axis_v4 import detect_corridor_axis_v4
from isaac_bench.mapping.online_roomseg.evidence_v4 import build_roomseg_evidence_v4
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


def test_evidence_v4_keeps_reachable_single_ray_free_and_rejects_disconnected_spur():
    shape = (20, 20)
    vertical_free = np.zeros(shape, dtype=bool)
    vertical_free[8:12, 8:12] = True
    vertical_free[2, 2:6] = True
    observed_free = np.zeros(shape, dtype=bool)
    observed_free[8:12, 8:12] = True
    traversible = observed_free.copy()
    profile = _profile_from_free(vertical_free)

    evidence = build_roomseg_evidence_v4(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=observed_free,
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=~vertical_free,
        vertical_profile=profile,
        roomseg_ray_evidence=None,
        traversible=traversible,
        agent_grid=(10, 10),
        map_info=MapInfo(resolution_m=0.1, min_x=0.0, max_x=2.0, min_y=0.0, max_y=2.0, width=20, height=20),
        config={
            "resolution_m": 0.1,
            "roomseg_evidence_v4": {
                "min_free_rays_for_stable_free": 2,
                "min_observed_rays_for_stable_cell": 2,
                "allow_single_ray_free_if_nav_reachable": True,
                "require_nav_reachable_for_roomseg_free": True,
                "min_free_component_area_m2": 0.01,
            },
            "navigation_consistency": {"min_free_component_area_m2": 0.01},
        },
    )

    assert np.any(evidence.roomseg_free_clean[8:12, 8:12])
    assert not np.any(evidence.roomseg_free_clean[2, 2:6])
    assert np.any(evidence.free_noise_rejected[2, 2:6])
    assert evidence.debug["algorithm"] == "online_line_extend_roomseg_v4"


def test_corridor_axis_v4_finds_local_axis_and_neck_candidates():
    free = np.zeros((30, 45), dtype=bool)
    free[12:17, 3:38] = True
    free[5:13, 20:27] = True
    result = detect_corridor_axis_v4(
        free,
        structural_wall_clean=~free,
        unknown_clean=np.zeros_like(free),
        resolution_m=0.1,
        config={
            "min_axis_width_m": 0.30,
            "max_axis_width_m": 1.10,
            "max_mean_width_m": 1.10,
            "min_length_m": 0.60,
            "max_neck_width_m": 1.10,
        },
    )

    assert int(np.count_nonzero(result.corridor_axis)) > 5
    assert int(np.count_nonzero(result.corridor_junctions)) >= 1
    assert len(result.corridor_neck_candidates) >= 1


def test_frontier_stability_v4_rejects_roomseg_noise_frontier_cells():
    free = np.zeros((15, 20), dtype=bool)
    free[4:11, 4:12] = True
    observed = free.copy()
    occupancy = np.zeros_like(free)
    stable_roomseg_free = free.copy()
    unknown = ~observed
    noise = np.zeros_like(free)
    noise[7, 11] = True

    result = stable_frontier_cells_v4(
        reachable_free=free,
        observed=observed,
        stable_roomseg_free=stable_roomseg_free,
        occupancy=occupancy,
        roomseg_unknown_clean=unknown,
        roomseg_free_noise_rejected=noise,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
        config={"enabled": True, "noise_reject_dilation_cells": 1},
    )

    assert result.raw_frontier[7, 11]
    assert result.rejected_noise[7, 11]
    assert not result.frontier[7, 11]
    assert int(result.debug["stable_frontier_cells"]) < int(result.debug["raw_frontier_cells"])


def _profile_from_free(free: np.ndarray) -> VerticalProfileMap:
    bands = 4
    free_count = np.zeros((bands, *free.shape), dtype=np.uint16)
    observed_count = np.zeros_like(free_count)
    occupied_count = np.zeros_like(free_count)
    unknown_count = np.ones_like(free_count)
    free_count[:, np.asarray(free, dtype=bool)] = 1
    observed_count[:, np.asarray(free, dtype=bool)] = 1
    unknown_count[observed_count > 0] = 0
    return VerticalProfileMap.from_counts(
        occupied_count=occupied_count,
        free_ray_count=free_count,
        observed_count=observed_count,
        unknown_count=unknown_count,
    )
