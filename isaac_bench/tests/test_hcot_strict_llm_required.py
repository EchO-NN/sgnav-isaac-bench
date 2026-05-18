from argparse import Namespace

import numpy as np
import pytest

from isaac_bench.graph.hcot_scorer import HCoTSubgraphScorer
from isaac_bench.graph.subgraph_builder import Subgraph
from isaac_bench.metrics.result_schema import BenchmarkAssetError, complete_result_row, validate_strict_benchmark_assets


def _subgraph():
    return Subgraph(
        id="sg_object_chair",
        central_object_id="object:chair",
        central_object_category="chair",
        parent_room_id=None,
        parent_group_id=None,
        directly_connected_object_ids=[],
        nodes=[{"id": "object:chair", "type": "object", "category": "chair"}],
        edges=[],
        central_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )


def _args(**overrides):
    data = {
        "strict_benchmark": True,
        "detector": "dry_run",
        "segmenter": "none",
        "sim_backend": "isaac",
        "planner": "astar",
        "sgnav_mode": "paper",
        "llm_enabled": False,
        "room_map_mode": "upstream_rose2_vertical_or_free",
        "room_segmentation_config": {"require_upstream_source_for_strict": False},
        "ablation_name": None,
        "seed_gt_object_memory": False,
        "allow_gt_goal_fallback": False,
        "static_nearfield_map": False,
        "frontier_allow_near_fallback": False,
    }
    data.update(overrides)
    return Namespace(**data)


def test_strict_hcot_without_llm_client_raises():
    scorer = HCoTSubgraphScorer(llm_client=None, allow_deterministic_fallback=False)

    with pytest.raises(RuntimeError, match="requires an enabled"):
        scorer.score_subgraph(_subgraph(), "table")


def test_strict_asset_contract_requires_llm_enabled_for_paper_mode():
    with pytest.raises(BenchmarkAssetError, match="requires --llm-enabled true"):
        validate_strict_benchmark_assets(_args())


def test_hcot_fallback_count_marks_result_non_metric():
    row = complete_result_row(
        {
            "success": False,
            "spl": 0.0,
            "softspl": 0.0,
            "distance_to_goal": 2.0,
            "path_length": 0.0,
            "hcot_llm_fallback_count": 1,
        },
        _args(llm_enabled=True, ablation_name=None),
    )

    assert row["metric_valid"] is False
    assert "llm_deterministic_local" in row["fallbacks_used"]
