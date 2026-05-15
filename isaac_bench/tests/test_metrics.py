import json

import numpy as np

from isaac_bench.metrics.episode_logger import JsonlEpisodeLogger, make_jsonable
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

