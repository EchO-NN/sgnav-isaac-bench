import numpy as np

from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.mapping.frontier import FrontierCluster


class FakeSceneGraph:
    def __init__(self, scores):
        self.scores = scores

    def score(self, locs, n):
        return np.asarray(self.scores[:n], dtype=np.float32)


def test_near_frontier_under_one_meter_is_ignored_when_eligible_exists():
    decision = SGNavDecision(
        FakeSceneGraph([100.0, 1.0]),
        frontier_distance_weight=0.2,
        frontier_min_select_distance_m=1.0,
        frontier_scenegraph_score_norm="none",
    )
    frontiers = [
        FrontierCluster((1, 1), (0.0, 0.0), [(1, 1)], 1, 0.8),
        FrontierCluster((2, 2), (1.0, 0.0), [(2, 2)], 1, 1.2),
    ]

    out = decision.choose_frontier(frontiers)

    assert out.selected_index == 1
    assert out.metadata["filtered_near_frontiers"] == 1
    assert out.metadata["frontier_min_select_distance_m"] == 1.0


def test_all_near_frontiers_without_fallback_return_clear_reason():
    decision = SGNavDecision(FakeSceneGraph([1.0]), frontier_min_select_distance_m=1.0, frontier_allow_near_fallback=False)

    out = decision.choose_frontier([FrontierCluster((1, 1), (0.0, 0.0), [(1, 1)], 1, 0.8)])

    assert out.selected_frontier is None
    assert out.reason == "all_frontiers_within_min_distance"


def test_explicit_near_fallback_records_metadata():
    decision = SGNavDecision(FakeSceneGraph([1.0]), frontier_min_select_distance_m=1.0, frontier_allow_near_fallback=True)

    out = decision.choose_frontier([FrontierCluster((1, 1), (0.0, 0.0), [(1, 1)], 1, 0.8)])

    assert out.selected_index == 0
    assert out.reason == "near_frontier_fallback"
    assert out.metadata["used_near_frontier_fallback"] is True
