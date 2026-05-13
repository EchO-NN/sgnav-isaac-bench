import math

import numpy as np

from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.dataset.episode_generator import dijkstra_distance_from_region


def test_astar_region_goal_nearest():
    grid = np.ones((10, 10), dtype=bool)
    grid[5, 1:8] = False
    grid[5, 4] = True
    planner = GridAStarPlanner(grid, resolution_m=1.0, allow_diagonal=False)
    result = planner.plan((1, 1), [(8, 8), (1, 4)])
    assert result.reached_goal == (1, 4)
    assert math.isclose(result.length_m, 3.0)


def test_dijkstra_distance_from_region_matches_diagonal_cost():
    grid = np.ones((5, 5), dtype=bool)
    dist = dijkstra_distance_from_region(grid, [(4, 4)], resolution_m=1.0, allow_diagonal=True)
    assert math.isclose(float(dist[1, 1]), 3.0 * math.sqrt(2.0))


def test_astar_diagonal_does_not_cut_blocked_corner():
    grid = np.array([[True, False], [False, True]], dtype=bool)
    planner = GridAStarPlanner(grid, resolution_m=1.0, allow_diagonal=True)
    result = planner.plan((0, 0), (1, 1))
    assert result.path == []
    dist = dijkstra_distance_from_region(grid, [(1, 1)], resolution_m=1.0, allow_diagonal=True)
    assert math.isinf(float(dist[0, 0]))
