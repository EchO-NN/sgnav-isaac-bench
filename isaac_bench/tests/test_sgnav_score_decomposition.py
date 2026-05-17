import numpy as np

from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.mapping.frontier import FrontierCluster


class FakeSceneGraph:
    def score(self, locs, n):
        return np.asarray([0.2, 3.0], dtype=np.float32)


def test_farther_hcot_scenegraph_evidence_can_beat_closer_frontier():
    decision = SGNavDecision(
        FakeSceneGraph(),
        frontier_distance_weight=0.2,
        frontier_min_select_distance_m=1.0,
        frontier_scenegraph_score_norm="none",
    )
    frontiers = [
        FrontierCluster((1, 1), (0.0, 0.0), [(1, 1)], 1, 1.1, distance_inverse=0.99),
        FrontierCluster((2, 2), (5.0, 0.0), [(2, 2)], 1, 5.0, distance_inverse=0.60),
    ]

    out = decision.choose_frontier(frontiers)

    assert out.selected_index == 1
    assert out.metadata["raw_scenegraph_scores"] == [0.20000000298023224, 3.0]
    assert out.metadata["distance_scores"] == [0.9900000095367432, 0.6000000238418579]
    assert out.metadata["frontier_distance_weight"] == 0.2
    assert out.metadata["total_scores"][1] > out.metadata["total_scores"][0]
