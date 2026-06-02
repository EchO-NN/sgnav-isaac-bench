from __future__ import annotations

import numpy as np

from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.navigation.frontier_commitment import FrontierCommitmentManager
from isaac_bench.navigation.frontier_execution_state import CommittedFrontierExecutionState
from isaac_bench.navigation.frontier_recovery import (
    FrontierRecoveryConfig,
    find_best_reachable_frontier_approach,
)
from isaac_bench.scripts.run_one_episode import (
    FrontierArrivalUpdateState,
    FrontierRoomsegUpdateGateState,
    consume_frontier_refresh,
    request_frontier_refresh,
    should_update_roomseg_frontiers,
)


def _frontier(center: tuple[int, int], score_dist: float = 5.0) -> FrontierCluster:
    r, c = center
    members = [(r, c), (r, c + 1), (r + 1, c)]
    return FrontierCluster(center, (float(c), float(r)), members, len(members), score_dist)


def test_recovery_selects_reachable_approach_instead_of_exact_unreachable_target() -> None:
    traversible = np.ones((12, 12), dtype=bool)
    traversible[:, 5] = False
    clearance = np.full(traversible.shape, 0.20, dtype=np.float32)
    planner = GridAStarPlanner(traversible, resolution_m=0.10, allow_diagonal=False)

    recovery = find_best_reachable_frontier_approach(
        current_grid=(5, 1),
        frontier_center=(5, 8),
        frontier_members=[(5, 8), (5, 9)],
        original_target=(5, 8),
        traversible=traversible,
        clearance_m=clearance,
        planner=planner,
        resolution_m=0.10,
        config=FrontierRecoveryConfig(
            local_search_radius_m=0.20,
            global_search_enabled=True,
            global_max_frontier_distance_m=1.20,
            min_clearance_m=0.05,
            min_approach_improvement_m=0.10,
            max_debug_candidates=16,
        ),
    )

    assert recovery.reachable is True
    assert recovery.target_cell is not None
    assert recovery.target_cell[1] < 5
    assert recovery.path
    assert recovery.distance_to_frontier_m < recovery.current_distance_to_frontier_m
    assert recovery.metadata()["frontier_target_planning_stage"] == "recovery"


def test_frontier_refresh_pending_updates_once_then_unblocks_reselect() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=0, last_update_reason="initial")
    arrival = FrontierArrivalUpdateState(initialized=True)

    request_frontier_refresh(
        refresh_state=arrival,
        step=12,
        reason="frontier_confirmed_unreachable_no_path",
        frontier_key=("frontier", 3, (10, 10), (10, 12)),
        block_reselect=True,
    )

    update, reason = should_update_roomseg_frontiers(
        step=12,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )
    assert update is True
    assert reason == "frontier_confirmed_unreachable_no_path"
    assert arrival.block_reselect_until_refresh is True

    consume_frontier_refresh(refresh_state=arrival, step=12, reason=reason, cooldown_steps=3)
    assert arrival.refresh_pending is False
    assert arrival.block_reselect_until_refresh is False

    request_frontier_refresh(
        refresh_state=arrival,
        step=13,
        reason="frontier_confirmed_unreachable_no_path",
        frontier_key=("frontier", 3, (10, 10), (10, 12)),
        block_reselect=True,
    )
    update, reason = should_update_roomseg_frontiers(
        step=14,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )
    assert update is True
    assert reason == "frontier_confirmed_unreachable_no_path"

    consume_frontier_refresh(refresh_state=arrival, step=14, reason=reason, cooldown_steps=3)
    arrival.refresh_pending = True
    arrival.refresh_frontier_key = arrival.last_consumed_key
    arrival.refresh_reason = "frontier_confirmed_unreachable_no_path"
    update, reason = should_update_roomseg_frontiers(
        step=15,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )
    assert update is False
    assert reason == "frontier_refresh_already_consumed"


def test_commitment_switch_requires_refresh_before_reselect() -> None:
    planner = GridAStarPlanner(np.ones((80, 80), dtype=bool), resolution_m=0.10)
    manager = FrontierCommitmentManager(
        resolution_m=0.10,
        min_commit_steps=1,
        no_progress_steps=999,
        allow_score_switch=True,
        switch_requires_refresh=True,
    )
    a = _frontier((10, 10))
    b = _frontier((60, 60))

    first = manager.select([a, b], a, 1.0, (0, 0), 0, planner=planner, scores_by_index=[1.0, 0.1])
    assert first.keep_existing is False

    switched = manager.select([a, b], b, 10.0, (1, 1), 10, planner=planner, scores_by_index=[1.0, 10.0])

    assert switched.keep_existing is True
    assert switched.reason == "frontier_switch_requires_refresh"
    assert switched.metadata["frontier_refresh_required_before_reselect"] is True
    assert manager.active is not None


def test_execution_recovery_state_tracks_original_target_and_consumes_refresh() -> None:
    execution = CommittedFrontierExecutionState()
    frontier = _frontier((10, 10))
    from isaac_bench.graph.decision import DecisionResult, NavigationDecision

    decision = NavigationDecision(
        "frontier",
        [(10, 10)],
        False,
        None,
        DecisionResult(0, frontier, [0.0], [1.0], [1.0], "selected"),
        "selected_new_frontier",
        metadata={"frontier_center_grid": [10, 10], "frontier_actual_target_grid": [10, 10]},
    )
    execution.start_from_decision(decision, step=2, commitment_metadata={"active_frontier_id": 4})
    execution.start_recovery(step=5, target_grid=(10, 8), path=[(0, 0), (5, 5), (10, 8)], reason="frontier_no_path")

    assert execution.recovery_active is True
    assert execution.original_actual_target_grid == (10, 10)
    assert execution.actual_target_grid == (10, 8)
    assert execution.recovery_attempts == 1

    execution.mark_partial_arrival_pending(step=8, key=("frontier", 4, (10, 10), (10, 8)), reason="frontier_recovery_arrival")
    assert execution.partial_arrival_pending_update is True
    execution.consume_arrival_update(step=9)

    assert execution.exists() is False
    assert execution.recovery_active is False
    assert execution.arrival_consumed_step == 9
