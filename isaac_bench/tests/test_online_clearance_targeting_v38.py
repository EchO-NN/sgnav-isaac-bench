from __future__ import annotations

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.navigation.frontier_commitment import FrontierCommitmentManager
from isaac_bench.navigation.frontier_targeting import resolve_frontier_target
from isaac_bench.scripts.run_one_episode import (
    effective_lookahead_min_clearance_m,
    filter_goal_cells_by_clearance_result,
    guard_kinematic_cmd_with_clearance,
)


def _frontier(center: tuple[int, int], members: list[tuple[int, int]] | None = None) -> FrontierCluster:
    return FrontierCluster(
        center_grid=center,
        center_world=(float(center[1]), float(center[0])),
        members=list(members or [center]),
        size=len(members or [center]),
        path_distance_from_agent=1.0,
    )


def test_effective_lookahead_never_reduces_configured_clearance() -> None:
    assert effective_lookahead_min_clearance_m(
        0.14,
        robot_radius_m=0.05,
        runtime_planning_clearance_m=0.02,
        astar_clearance_hard_min_m=0.0,
    ) == pytest.approx(0.14)
    assert effective_lookahead_min_clearance_m(
        0.14,
        robot_radius_m=0.14,
        runtime_planning_clearance_m=0.02,
        astar_clearance_hard_min_m=0.0,
    ) == pytest.approx(0.16)


def test_goal_clearance_filter_does_not_fallback_to_unsafe_original() -> None:
    traversible = np.ones((7, 7), dtype=bool)
    clearance = np.zeros((7, 7), dtype=np.float32)

    result = filter_goal_cells_by_clearance_result(
        [(3, 3)],
        traversible,
        clearance,
        min_clearance_m=0.18,
        search_radius_cells=2,
    )

    assert result.goals == []
    assert result.no_safe_goal_count == 1


def test_frontier_target_resolver_selects_clearance_safe_approach_cell() -> None:
    traversible = np.ones((7, 7), dtype=bool)
    clearance = np.full((7, 7), 0.05, dtype=np.float32)
    clearance[3, 3] = 0.05
    safe_cell = (3, 4)
    clearance[safe_cell] = 0.30
    planner = GridAStarPlanner(traversible, resolution_m=0.10, allow_diagonal=True)

    resolution = resolve_frontier_target(
        _frontier((3, 3), [(3, 3)]),
        current_grid=(1, 1),
        planner=planner,
        traversible=traversible,
        clearance_m=clearance,
        resolution_m=0.10,
        min_clearance_m=0.18,
        search_radius_m=0.20,
        reached_radius_m=0.20,
    )

    assert resolution.actual_target_grid == safe_cell
    assert resolution.mode in {"approach_cell", "member_neighbor"}
    assert resolution.selected_clearance_m is not None
    assert resolution.selected_clearance_m >= 0.18


def test_commitment_keep_existing_does_not_accept_proposed_frontier_target() -> None:
    manager = FrontierCommitmentManager(resolution_m=0.05, min_commit_steps=100)
    active_frontier = _frontier((10, 10), [(10, 10), (10, 11)])
    proposed_frontier = _frontier((30, 30), [(30, 30), (30, 31)])

    first = manager.select(
        [active_frontier],
        active_frontier,
        proposed_score=1.0,
        current_grid=(0, 0),
        step=0,
        target_cells=[(10, 11)],
        target_frontier=active_frontier,
        target_metadata={"frontier_target_clearance_m": 0.25},
        scores_by_index=[1.0],
    )
    assert first.target_cells == [(10, 11)]

    second = manager.select(
        [active_frontier, proposed_frontier],
        proposed_frontier,
        proposed_score=2.0,
        current_grid=(1, 1),
        step=1,
        target_cells=[(30, 31)],
        target_frontier=proposed_frontier,
        target_metadata={"frontier_target_clearance_m": 0.30},
        scores_by_index=[1.0, 2.0],
    )

    assert second.keep_existing is True
    assert manager.active is not None
    assert manager.active.center_grid == (10, 10)
    assert manager.active.target_cells == [(10, 11)]
    assert manager.active.target_cells != [(30, 31)]
    assert second.metadata["frontier_commitment_target_consistent"] is True
    assert second.metadata["active_frontier_actual_target_grid"] == [10, 11]


def test_swept_clearance_guard_blocks_low_clearance_motion() -> None:
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    traversible = np.ones((5, 5), dtype=bool)
    clearance = np.full((5, 5), 0.30, dtype=np.float32)
    clearance[2, 2] = 0.05
    pose = (1.99, 2.5, 0.05, 0.0)
    cmd = (1.0, 0.0, 0.0)

    guarded, blocked, debug = guard_kinematic_cmd_with_clearance(
        pose,
        cmd,
        1.0,
        traversible,
        clearance,
        info,
        min_clearance_m=0.14,
        camera_forward_offset_m=0.0,
    )

    assert blocked is True
    assert guarded == pytest.approx((0.0, 0.0, 0.0))
    assert debug["guard_block_reason"] == "low_clearance"
