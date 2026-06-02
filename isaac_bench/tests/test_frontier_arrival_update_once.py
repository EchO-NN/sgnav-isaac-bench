from __future__ import annotations

from isaac_bench.scripts.run_one_episode import (
    FrontierArrivalUpdateState,
    FrontierRoomsegUpdateGateState,
    mark_roomseg_frontier_gate_update,
    should_update_roomseg_frontiers,
)


def test_frontier_arrival_latch_updates_once_then_freezes() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=0, last_update_reason="initial")
    arrival = FrontierArrivalUpdateState(
        initialized=True,
        arrival_pending=True,
        arrival_frontier_key=("frontier", 1, (10, 10)),
    )

    update, reason = should_update_roomseg_frontiers(
        step=3,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )

    assert update is True
    assert reason == "frontier_arrival"

    mark_roomseg_frontier_gate_update(gate, step=3, reason=reason)
    arrival.last_consumed_arrival_key = arrival.arrival_frontier_key
    arrival.arrival_pending = True
    arrival.cooldown_until_step = 6

    update, reason = should_update_roomseg_frontiers(
        step=4,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )

    assert update is False
    assert reason == "frontier_arrival_cooldown"

    update, reason = should_update_roomseg_frontiers(
        step=7,
        has_current_path=False,
        gate_state=gate,
        arrival_state=arrival,
    )

    assert update is False
    assert reason == "frontier_arrival_already_consumed"
