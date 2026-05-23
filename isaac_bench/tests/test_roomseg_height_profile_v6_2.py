from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.height_profile_v6_2 import build_height_profile_v6_2


def test_terminal_splat_without_height_range_is_debug_evidence_not_wall_bins():
    shape = (4, 4)
    terminal = np.zeros(shape, dtype=np.uint16)
    terminal[2, 2] = 3

    profile = build_height_profile_v6_2(
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        roomseg_ray_evidence={"terminal_wall_count": terminal},
        config={"vertical_range": {"z_bin_m": 0.10}},
    )

    assert profile.debug["raw_endpoint_evidence_2d_cells"] == 1
    assert profile.debug["terminal_height_profile_endpoint_cells"] == 0
    assert int(np.count_nonzero(profile.endpoint_supported[:, 2, 2])) == 0


def test_terminal_height_range_enters_endpoint_bins_but_still_uses_profile_support():
    shape = (3, 3)
    terminal = np.zeros(shape, dtype=np.uint16)
    z_min = np.full(shape, np.inf, dtype=np.float32)
    z_max = np.full(shape, -np.inf, dtype=np.float32)
    terminal[1, 1] = 2
    z_min[1, 1] = 1.90
    z_max[1, 1] = 2.10

    profile = build_height_profile_v6_2(
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        roomseg_ray_evidence={
            "terminal_wall_count": terminal,
            "terminal_wall_height_min": z_min,
            "terminal_wall_height_max": z_max,
        },
        config={"vertical_range": {"z_bin_m": 0.05}},
    )

    assert profile.debug["terminal_height_profile_endpoint_cells"] == 1
    assert int(np.count_nonzero(profile.endpoint_supported[:, 1, 1])) > 0
    assert int(np.count_nonzero(profile.endpoint_supported[:, 1, 1])) < int(profile.endpoint_supported.shape[0])
