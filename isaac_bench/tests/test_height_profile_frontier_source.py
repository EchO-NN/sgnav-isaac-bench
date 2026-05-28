import numpy as np

from isaac_bench.scripts.run_one_episode import resolve_frontier_source_layers


def test_frontier_free_source_is_vertical_free_not_navigation_free():
    nav_free = np.zeros((6, 6), dtype=bool)
    nav_free[1:5, 1:5] = True
    vertical_free = np.zeros_like(nav_free)
    vertical_free[2, 2] = True
    vertical_free[2, 3] = True
    debug = {
        "height_profile_vertical_free_xy": vertical_free,
        "height_profile_vertical_observed_xy": vertical_free.copy(),
        "height_profile_wall_xy": np.zeros_like(nav_free),
    }

    frontier_free, _observed, _wall, meta = resolve_frontier_source_layers(
        room_debug=debug,
        mapper=None,
        navigation_free=nav_free,
        navigation_observed=nav_free,
        navigation_occupancy=np.zeros_like(nav_free),
        frontier_traversible=np.ones_like(nav_free),
        source="vertical_free",
    )

    assert np.array_equal(frontier_free, vertical_free)
    assert meta["frontier_source"] == "vertical_free"
    assert meta["frontier_cells_from_vertical_free"] == int(np.count_nonzero(vertical_free))


def test_frontier_reachability_still_uses_navigation_traversible():
    vertical_free = np.zeros((6, 6), dtype=bool)
    vertical_free[2, 2] = True
    vertical_free[2, 3] = True
    traversible = np.zeros_like(vertical_free)
    traversible[2, 2] = True
    debug = {
        "height_profile_vertical_free_xy": vertical_free,
        "height_profile_vertical_observed_xy": vertical_free.copy(),
        "height_profile_wall_xy": np.zeros_like(vertical_free),
    }

    frontier_free, _observed, _wall, meta = resolve_frontier_source_layers(
        room_debug=debug,
        mapper=None,
        navigation_free=np.ones_like(vertical_free),
        navigation_observed=np.ones_like(vertical_free),
        navigation_occupancy=np.zeros_like(vertical_free),
        frontier_traversible=traversible,
        source="vertical_free",
        require_navigation_reachable=True,
    )

    assert frontier_free[2, 2]
    assert not frontier_free[2, 3]
    assert meta["frontier_cells_removed_by_navigation_unreachable"] == 1


def test_vertical_unknown_used_for_frontier_unknown_boundary():
    vertical_free = np.zeros((5, 5), dtype=bool)
    vertical_free[2, 2] = True
    observed = np.zeros_like(vertical_free)
    observed[2, 2] = True
    wall = np.zeros_like(vertical_free)
    wall[1, 1] = True
    debug = {
        "height_profile_vertical_free_xy": vertical_free,
        "height_profile_vertical_observed_xy": observed,
        "height_profile_wall_xy": wall,
    }

    _frontier_free, frontier_observed, frontier_wall, meta = resolve_frontier_source_layers(
        room_debug=debug,
        mapper=None,
        navigation_free=np.ones_like(vertical_free),
        navigation_observed=np.ones_like(vertical_free),
        navigation_occupancy=np.zeros_like(vertical_free),
        frontier_traversible=np.ones_like(vertical_free),
        source="vertical_free",
    )

    assert np.array_equal(frontier_observed, observed)
    assert np.array_equal(frontier_wall, wall)
    assert meta["frontier_vertical_observed_cells"] == 1
    assert meta["frontier_vertical_wall_cells"] == 1
