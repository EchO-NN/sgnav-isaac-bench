import json

import numpy as np

from isaac_bench.metrics.episode_logger import JsonlEpisodeLogger, make_jsonable
from isaac_bench.metrics.summarize import summarize_rows
from isaac_bench.metrics.spl import compute_softspl, compute_spl


def test_spl():
    assert compute_spl(False, 5.0, 5.0) == 0.0
    assert compute_spl(True, 5.0, 5.0) == 1.0
    assert compute_spl(True, 5.0, 10.0) == 0.5


def test_softspl():
    assert compute_softspl(10.0, 0.0, 10.0, 10.0) == 1.0
    assert compute_softspl(10.0, 5.0, 10.0, 10.0) == 0.5


def test_jsonl_logger_serializes_numpy_values(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {
        "score": np.float32(0.5),
        "nested": {"count": np.int64(2), "arr": np.asarray([np.float32(1.5)])},
    }

    JsonlEpisodeLogger(str(path)).log(row)
    parsed = json.loads(path.read_text(encoding="utf-8"))

    assert parsed == {"score": 0.5, "nested": {"count": 2, "arr": [1.5]}}
    assert json.dumps(make_jsonable(row))


def test_summarize_tracks_metric_validity_and_fallback_counts():
    rows = [
        {
            "planner": "astar",
            "detector": "dry_run",
            "metric_valid": False,
            "success": True,
            "spl": 1.0,
            "softspl": 1.0,
            "distance_to_goal": 0.0,
            "path_length": 2.0,
            "fallbacks_used": ["dry_run_detector"],
        },
        {
            "planner": "astar",
            "detector": "dry_run",
            "metric_valid": False,
            "success": False,
            "spl": 0.0,
            "softspl": 0.5,
            "distance_to_goal": 1.0,
            "path_length": 3.0,
            "fallbacks_used": ["dry_run_detector", "static_map_planning"],
        },
    ]

    summary = summarize_rows(rows, ["planner", "detector", "metric_valid"])

    assert summary[0]["episodes"] == 2
    assert summary[0]["metric_valid_episodes"] == 0
    assert summary[0]["non_metric_episodes"] == 2
    assert summary[0]["fallback_counts"] == {"dry_run_detector": 2, "static_map_planning": 1}
