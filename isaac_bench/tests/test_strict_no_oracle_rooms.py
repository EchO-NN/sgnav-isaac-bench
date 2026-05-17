from argparse import Namespace

import pytest

from isaac_bench.metrics.result_schema import BenchmarkAssetError, complete_result_row, validate_strict_benchmark_assets


def _args(**overrides):
    data = {
        "strict_benchmark": True,
        "detector": "dry_run",
        "segmenter": "none",
        "sim_backend": "isaac",
        "planner": "astar",
        "sgnav_mode": "paper",
        "llm_enabled": True,
        "room_map_mode": "observed_rooms_json",
        "ablation_name": None,
        "seed_gt_object_memory": False,
        "allow_gt_goal_fallback": False,
        "static_nearfield_map": False,
        "frontier_allow_near_fallback": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_strict_metric_rejects_rooms_json_oracle_room_map():
    with pytest.raises(BenchmarkAssetError, match="rooms.json/oracle room maps are not allowed"):
        validate_strict_benchmark_assets(_args())


def test_oracle_room_map_marks_result_non_metric_when_not_named_ablation():
    row = complete_result_row(
        {
            "success": False,
            "spl": 0.0,
            "softspl": 0.0,
            "distance_to_goal": 1.0,
            "path_length": 0.0,
            "room_map_mode": "observed_rooms_json",
        },
        _args(),
    )

    assert row["metric_valid"] is False
    assert "oracle_room_map" in row["fallbacks_used"]


def test_named_oracle_room_ablation_can_be_explicitly_non_main():
    validate_strict_benchmark_assets(_args(ablation_name="oracle_room_ablation"))
