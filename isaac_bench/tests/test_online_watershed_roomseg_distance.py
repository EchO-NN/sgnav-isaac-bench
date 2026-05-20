from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_watershed_roomseg.distance import compute_watershed_distance_fields


def test_structural_distance_ignores_unknown_and_uses_only_wall_core():
    free = np.zeros((30, 40), dtype=bool)
    free[5:25, 5:28] = True
    wall = np.zeros_like(free)
    wall[5:25, 4] = True
    wall[5:25, 28] = True

    fields_a = compute_watershed_distance_fields(free_clean=free, wall_candidate_clean=wall, resolution_m=0.05)

    unknown_changed_wall = wall.copy()
    # Unknown is intentionally not passed as a barrier. A caller changing unknown
    # outside the structural wall core must not alter dist_struct_m.
    fields_b = compute_watershed_distance_fields(free_clean=free, wall_candidate_clean=unknown_changed_wall, resolution_m=0.05)

    assert np.allclose(fields_a.dist_struct_m, fields_b.dist_struct_m)
    assert np.all(fields_a.dist_struct_m[~free] == 0.0)
    assert np.isinf(fields_a.elevation[~free]).all()
    assert float(fields_a.dist_free_extent_m[15, 15]) > float(fields_a.dist_free_extent_m[6, 6])


def test_clean_wall_candidates_are_the_only_structural_minima():
    free = np.zeros((20, 20), dtype=bool)
    free[2:18, 2:18] = True
    wall = np.zeros_like(free)
    wall[2:18, 1] = True
    fields = compute_watershed_distance_fields(free_clean=free, wall_candidate_clean=wall, resolution_m=0.1)

    near_wall = float(fields.dist_struct_m[10, 2])
    center = float(fields.dist_struct_m[10, 10])
    assert center > near_wall
    assert float(fields.elevation[10, 10]) < float(fields.elevation[10, 2])
