import math

import numpy as np

from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.dataset.episode_generator import dijkstra_distance_from_region
from isaac_bench.scripts.run_one_episode import (
    apply_dynamic_astar_edge_clearance,
    navigation_target_reached,
    planning_target_cells_within_radius,
)


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


def test_dynamic_astar_edge_clearance_adds_margin_from_raw_occupied_cells():
    traversible = np.ones((9, 9), dtype=bool)
    inflated_occupied = np.zeros_like(traversible)
    inflated_occupied[4, 4] = True

    out = apply_dynamic_astar_edge_clearance(
        traversible,
        inflated_occupied,
        resolution_m=0.05,
        extra_clearance_m=0.10,
        current_grid=(4, 5),
    )

    assert not out[4, 4]
    assert out[4, 5]  # current pose is allowed to start inside the added safety band
    assert not out[4, 6]
    assert out[4, 7]


def test_navigation_target_reached_uses_configured_radius():
    decision = NavigationDecision(
        mode="frontier",
        target_cells=[(12, 10)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="selected_new_frontier",
    )

    reached, distance_m = navigation_target_reached(
        current_grid=(10, 10),
        nav_decision=decision,
        resolution_m=0.1,
        reached_radius_m=0.2,
    )

    assert reached
    assert math.isclose(distance_m, 0.2)

    reached, _ = navigation_target_reached(
        current_grid=(10, 10),
        nav_decision=decision,
        resolution_m=0.1,
        reached_radius_m=0.19,
    )

    assert not reached


def test_planning_target_cells_within_radius_uses_nearby_safe_cells():
    traversible = np.zeros((7, 7), dtype=bool)
    traversible[3, 2] = True
    traversible[3, 4] = True

    goals = planning_target_cells_within_radius(
        [(3, 3)],
        traversible,
        resolution_m=0.1,
        radius_m=0.11,
    )

    assert goals == [(3, 2), (3, 4)]
