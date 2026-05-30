from __future__ import annotations

import numpy as np

from isaac_bench.navigation.astar import ClearanceAStarPlanner, GridAStarPlanner


def _path_min_clearance(path: list[tuple[int, int]], clearance_m: np.ndarray) -> float:
    return min(float(clearance_m[row, col]) for row, col in path)


def test_clearance_astar_prefers_corridor_center_over_wall_hugging_shortest_path() -> None:
    grid = np.ones((12, 40), dtype=bool)
    grid[0, :] = False
    grid[11, :] = False
    start = (1, 1)
    goal = (1, 38)

    binary = GridAStarPlanner(grid, resolution_m=0.05, allow_diagonal=True).plan(start, goal)
    clearance = ClearanceAStarPlanner(
        grid,
        resolution_m=0.05,
        allow_diagonal=True,
        clearance_desired_m=0.25,
        clearance_weight=8.0,
        clearance_power=2.0,
        clearance_hard_min_m=0.0,
    )
    result = clearance.plan(start, goal)

    assert result.path
    assert _path_min_clearance(result.path[4:-4], clearance.clearance_m) > _path_min_clearance(binary.path[4:-4], clearance.clearance_m)


def test_clearance_astar_soft_cost_still_passes_narrow_door() -> None:
    grid = np.ones((11, 21), dtype=bool)
    grid[:, 10] = False
    grid[5, 10] = True

    planner = ClearanceAStarPlanner(
        grid,
        resolution_m=0.05,
        allow_diagonal=False,
        clearance_desired_m=0.30,
        clearance_weight=5.0,
        clearance_power=2.0,
        clearance_hard_min_m=0.0,
    )
    result = planner.plan((5, 2), (5, 18))

    assert result.path
    assert (5, 10) in result.path
