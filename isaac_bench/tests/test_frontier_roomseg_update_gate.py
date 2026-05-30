from __future__ import annotations

from isaac_bench.scripts.run_one_episode import (
    FrontierRoomsegUpdateGateState,
    mark_roomseg_frontier_gate_update,
    should_update_roomseg_frontiers,
)


def test_frontier_roomseg_update_gate_freezes_during_navigation() -> None:
    gate = FrontierRoomsegUpdateGateState()

    update, reason = should_update_roomseg_frontiers(step=0, has_current_path=False, gate_state=gate)
    assert update is True
    assert reason == "initial"
    mark_roomseg_frontier_gate_update(gate, step=0, reason=reason)

    update, reason = should_update_roomseg_frontiers(step=10, has_current_path=True, gate_state=gate)
    assert update is False
    assert reason == "cached_during_navigation"

    update, reason = should_update_roomseg_frontiers(step=11, has_current_path=False, gate_state=gate, target_invalidated=True)
    assert update is False
    assert reason == "target_invalidated_cached"

    update, reason = should_update_roomseg_frontiers(
        step=11,
        has_current_path=False,
        gate_state=gate,
        target_invalidated=True,
        update_on_target_invalidated=True,
    )
    assert update is True
    assert reason == "target_invalidated"


def test_frontier_roomseg_update_gate_can_be_forced_for_debug() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=5, last_update_reason="initial")

    update, reason = should_update_roomseg_frontiers(
        step=6,
        has_current_path=True,
        gate_state=gate,
        debug_force=True,
    )

    assert update is True
    assert reason == "debug_force"
