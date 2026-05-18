from __future__ import annotations

from argparse import Namespace

from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.metrics.result_schema import complete_result_row
from isaac_bench.scripts.run_one_episode import (
    build_stop_state_payload,
    final_log_row,
    success_region_can_finish,
)


def _strict_args() -> Namespace:
    return Namespace(
        strict_benchmark=True,
        ablation_name="local_deterministic_llm",
        detector="grounding_dino",
        segmenter="sam2",
        sim_backend="isaac",
        planner="astar",
        sgnav_mode="paper",
        llm_enabled=True,
        vllm_frontier_scoring=False,
        seed_gt_object_memory=False,
        allow_gt_goal_fallback=False,
        static_nearfield_map=False,
        frontier_allow_near_fallback=False,
    )


def test_success_radius_does_not_finish_strict_run_without_sgnav_stop():
    assert not success_region_can_finish(
        0.0,
        0.2,
        require_sgnav_stop=True,
        policy_stop_confirmed=False,
    )
    assert success_region_can_finish(
        0.0,
        0.2,
        require_sgnav_stop=True,
        policy_stop_confirmed=True,
    )
    assert success_region_can_finish(
        0.0,
        0.2,
        require_sgnav_stop=False,
        policy_stop_confirmed=False,
    )


def test_result_schema_removes_success_without_policy_stop_confirmation():
    row = complete_result_row(
        {
            "success": True,
            "spl": 1.0,
            "softspl": 1.0,
            "distance_to_goal": 0.0,
            "path_length": 3.0,
            "success_requires_sgnav_stop": True,
            "policy_stop_confirmed": False,
            "stop_called": True,
            "sgnav_decision_mode": "frontier",
            "sgnav_decision_reason": "selected_new_frontier",
        },
        _strict_args(),
    )

    assert row["success"] is False
    assert row["spl"] == 0.0
    assert row["stop_reason"] == "sgnav_stop_required"
    assert row["failure_reason"] == "sgnav_stop_required"


def test_stop_state_reports_blocked_gt_radius_stop():
    nav_decision = NavigationDecision(
        mode="frontier",
        target_cells=[(10, 11)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="selected_new_frontier",
    )
    stop_state = build_stop_state_payload(
        nav_decision,
        {
            "success": False,
            "distance_to_goal": 0.0,
            "success_requires_sgnav_stop": True,
            "policy_stop_confirmed": False,
            "gt_success_region_reached": True,
            "stop_blocked_reason": "sgnav_stop_required",
        },
    )

    assert stop_state["stop_allowed"] is False
    assert stop_state["policy_stop_confirmed"] is False
    assert stop_state["gt_success_region_reached"] is True
    assert stop_state["stop_reason"] == "sgnav_stop_required"


def test_final_log_exposes_strict_stop_guard_fields():
    out = final_log_row(
        {
            "goal_category": "ceiling_light",
            "success": False,
            "distance_to_goal": 0.0,
            "spl": 0.0,
            "failure_reason": "max_control_steps",
            "sgnav_decision_mode": "frontier",
            "sgnav_decision_reason": "selected_new_frontier",
            "stop_called": False,
            "policy_stop_confirmed": False,
            "success_requires_sgnav_stop": True,
            "gt_success_region_reached": True,
            "gt_success_without_sgnav_stop_steps": 4,
            "stop_blocked_reason": "sgnav_stop_required",
        }
    )

    assert out["success"] is False
    assert out["stop_reason"] == "max_control_steps"
    assert out["sgnav_decision_mode"] == "frontier"
    assert out["policy_stop_confirmed"] is False
    assert out["gt_success_region_reached"] is True
    assert out["stop_blocked_reason"] == "sgnav_stop_required"
