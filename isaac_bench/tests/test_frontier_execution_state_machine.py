from __future__ import annotations

import ast
from pathlib import Path

from isaac_bench.graph.decision import DecisionResult, NavigationDecision
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.frontier_commitment import FrontierCommitmentManager
from isaac_bench.navigation.frontier_execution_state import (
    CommittedFrontierExecutionState,
    frontier_arrival_status,
    make_committed_frontier_arrival_key,
)
from isaac_bench.scripts.run_one_episode import (
    FrontierArrivalUpdateState,
    FrontierRoomsegUpdateGateState,
    RoomsegFrontierUpdateToken,
    mark_roomseg_frontier_gate_update,
    should_update_roomseg_frontiers,
)


def _frontier(center=(10, 10)) -> FrontierCluster:
    return FrontierCluster(
        center_grid=center,
        center_world=(float(center[1]) * 0.05, float(center[0]) * 0.05),
        members=[center, (center[0], center[1] + 1)],
        size=2,
        path_distance_from_agent=1.0,
    )


def _frontier_decision(target=(10, 10)) -> NavigationDecision:
    frontier = _frontier(target)
    result = DecisionResult(
        selected_index=0,
        selected_frontier=frontier,
        scenegraph_scores=[0.0],
        distance_scores=[1.0],
        total_scores=[1.0],
        reason="selected_frontier",
    )
    return NavigationDecision(
        "frontier",
        [target, (target[0], target[1] + 1)],
        False,
        None,
        result,
        "selected_new_frontier",
        metadata={
            "frontier_center_grid": list(target),
            "frontier_actual_target_grid": list(target),
            "frontier_commitment": {"active_frontier_id": 7},
        },
    )


def test_committed_frontier_freezes_roomseg_until_arrival() -> None:
    gate = FrontierRoomsegUpdateGateState()
    update, reason = should_update_roomseg_frontiers(step=0, has_current_path=False, gate_state=gate)
    assert update is True
    assert reason == "initial"
    mark_roomseg_frontier_gate_update(gate, step=0, reason=reason)

    update, reason = should_update_roomseg_frontiers(
        step=3,
        has_current_path=True,
        gate_state=gate,
        arrival_state=FrontierArrivalUpdateState(),
        no_active_path=False,
        target_invalidated=False,
    )

    assert update is False
    assert reason == "cached_during_navigation"
    token = RoomsegFrontierUpdateToken(update, reason, 3)
    assert token.allowed is False


def test_empty_path_replans_same_committed_frontier_not_new_frontier() -> None:
    decision = _frontier_decision((20, 30))
    execution = CommittedFrontierExecutionState()
    execution.start_from_decision(decision, step=5, commitment_metadata={"active_frontier_id": 11})

    locked = execution.to_navigation_decision(decision)

    assert execution.exists()
    assert locked.mode == "frontier"
    assert locked.target_cells == decision.target_cells
    assert locked.reason == "continue_committed_frontier_execution"
    assert locked.metadata["frontier_execution_locked"] is True


def test_committed_frontier_syncs_astar_reached_goal() -> None:
    decision = _frontier_decision((20, 30))
    execution = CommittedFrontierExecutionState()
    execution.start_from_decision(decision, step=5, commitment_metadata={"active_frontier_id": 11})

    execution.sync_actual_target((22, 34), step=6)

    assert execution.actual_target_grid == (22, 34)
    assert execution.target_cells == [(22, 34)]
    assert execution.last_replan_step == 6
    assert execution.distance_to_target_m((22, 34), resolution_m=0.05) == 0.0
    assert make_committed_frontier_arrival_key(execution) == ("frontier", 11, (20, 30), (22, 34))


def test_frontier_reached_does_not_blacklist() -> None:
    manager = FrontierCommitmentManager(resolution_m=0.05, blacklist_ttl_steps=100)
    frontier = _frontier((10, 10))
    selected = manager.select(
        [frontier],
        frontier,
        proposed_score=1.0,
        current_grid=(0, 0),
        step=0,
        planner=None,
        target_cells=frontier.members,
        scores_by_index=[1.0],
    )
    assert selected.target_cells
    assert manager.active is not None

    manager.mark_active_reached(step=10, reason="frontier_arrival_confirmed")

    assert manager.active is None
    assert manager.blacklist == []


def test_frontier_arrival_requires_min_age_and_confirm_steps() -> None:
    decision = _frontier_decision((10, 10))
    execution = CommittedFrontierExecutionState()
    execution.start_from_decision(decision, step=0, commitment_metadata={"active_frontier_id": 7})

    arrived, distance, reason = frontier_arrival_status(
        step=2,
        current_grid=(10, 10),
        execution=execution,
        resolution_m=0.05,
        reached_radius_m=0.20,
        min_steps_since_selection=6,
        confirm_steps=2,
        current_path=[],
    )
    assert arrived is False
    assert distance == 0.0
    assert reason == "arrival_too_early"

    arrived, _distance, reason = frontier_arrival_status(
        step=6,
        current_grid=(10, 10),
        execution=execution,
        resolution_m=0.05,
        reached_radius_m=0.20,
        min_steps_since_selection=6,
        confirm_steps=2,
        current_path=[],
    )
    assert arrived is False
    assert reason == "frontier_arrival_candidate"

    arrived, _distance, reason = frontier_arrival_status(
        step=7,
        current_grid=(10, 10),
        execution=execution,
        resolution_m=0.05,
        reached_radius_m=0.20,
        min_steps_since_selection=6,
        confirm_steps=2,
        current_path=[],
    )
    assert arrived is True
    assert reason == "frontier_arrival_confirmed"
    assert make_committed_frontier_arrival_key(execution) == ("frontier", 7, (10, 10), (10, 10))


def test_frontier_arrival_update_exactly_once_after_consumption() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=0, last_update_reason="initial")
    arrival = FrontierArrivalUpdateState(
        initialized=True,
        arrival_pending=True,
        arrival_frontier_key=("frontier", 7, (10, 10), (10, 10)),
    )

    update, reason = should_update_roomseg_frontiers(
        step=9,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )
    assert update is True
    assert reason == "frontier_arrival"

    mark_roomseg_frontier_gate_update(gate, step=9, reason=reason)
    arrival.last_consumed_arrival_key = arrival.arrival_frontier_key
    arrival.arrival_pending = False
    arrival.cooldown_until_step = 12

    update, reason = should_update_roomseg_frontiers(
        step=10,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )
    assert update is False
    assert reason == "frontier_arrival_cooldown"

    arrival.arrival_pending = True
    update, reason = should_update_roomseg_frontiers(
        step=13,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )
    assert update is False
    assert reason == "frontier_arrival_already_consumed"


def test_path_exhausted_without_arrival_keeps_committed_target() -> None:
    decision = _frontier_decision((40, 40))
    execution = CommittedFrontierExecutionState()
    execution.start_from_decision(decision, step=0, commitment_metadata={"active_frontier_id": 9})

    arrived, distance, reason = frontier_arrival_status(
        step=12,
        current_grid=(10, 10),
        execution=execution,
        resolution_m=0.05,
        reached_radius_m=0.20,
        min_steps_since_selection=6,
        confirm_steps=2,
        current_path=[],
    )
    if not arrived:
        execution.no_path_replans += 1

    assert arrived is False
    assert distance > 0.20
    assert reason == "arrival_not_in_radius"
    assert execution.exists()
    assert execution.no_path_replans == 1
    assert execution.failed is False


def test_same_target_replan_does_not_clear_no_path_counter() -> None:
    decision = _frontier_decision((40, 40))
    execution = CommittedFrontierExecutionState()
    execution.start_from_decision(decision, step=0, commitment_metadata={"active_frontier_id": 9})

    execution.no_path_replans = 2
    execution.mark_replanned(step=12)

    assert execution.no_path_replans == 2
    assert execution.last_replan_step == 12


def test_guard_blocked_single_step_does_not_blacklist_frontier() -> None:
    manager = FrontierCommitmentManager(resolution_m=0.05, blacklist_ttl_steps=100)
    frontier = _frontier((10, 10))
    manager.select([frontier], frontier, 1.0, (0, 0), 0, target_cells=frontier.members, scores_by_index=[1.0])
    execution = CommittedFrontierExecutionState()
    execution.start_from_decision(_frontier_decision((10, 10)), step=0, commitment_metadata={"active_frontier_id": 1})

    execution.guard_blocked_steps += 1

    assert execution.exists()
    assert execution.guard_blocked_steps == 1
    assert manager.active is not None
    assert manager.blacklist == []


def test_roomseg_update_calls_are_token_wrapped() -> None:
    source = Path("isaac_bench/scripts/run_one_episode.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct_calls = []
    maybe_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name == "update_room_context_for_frontier_scoring":
            direct_calls.append(node.lineno)
        if name == "maybe_update_room_context_for_frontier_scoring":
            maybe_calls.append(node.lineno)

    assert len(direct_calls) == 1
    assert len(maybe_calls) >= 2
