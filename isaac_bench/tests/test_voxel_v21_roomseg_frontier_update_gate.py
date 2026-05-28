from __future__ import annotations

from isaac_bench.scripts.run_one_episode import FrontierRoomsegUpdateGateState, should_update_roomseg_frontiers


def test_v21_roomseg_frontier_gate_defaults_to_cached_during_motion_failures() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=1, last_update_reason="initial")

    for has_current_path, kwargs in (
        (True, {"target_invalidated": True}),
        (True, {"no_progress": True}),
        (False, {}),
    ):
        update, reason = should_update_roomseg_frontiers(
            step=10,
            has_current_path=has_current_path,
            gate_state=gate,
            **kwargs,
        )
        assert update is False
        assert reason == "cached_during_navigation"


def test_v21_roomseg_frontier_gate_updates_on_target_reached_and_debug_force() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=1, last_update_reason="initial")

    update, reason = should_update_roomseg_frontiers(step=11, has_current_path=True, gate_state=gate, target_reached=True)
    assert update is True
    assert reason == "target_reached"

    update, reason = should_update_roomseg_frontiers(step=12, has_current_path=True, gate_state=gate, debug_force=True)
    assert update is True
    assert reason == "debug_force"
