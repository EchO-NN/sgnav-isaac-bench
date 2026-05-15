import numpy as np

from isaac_bench.graph.decision import SGNavDecision, normalize_scores
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.frontier import FrontierCluster


def test_normalize_scores_minmax_and_flat_values():
    assert np.allclose(normalize_scores([2.0, 4.0, 6.0], "minmax"), [0.0, 0.5, 1.0])
    assert np.allclose(normalize_scores([3.0, 3.0], "minmax"), [0.0, 0.0])


def test_frontier_decision_stores_raw_normalized_and_total_scores():
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.score = lambda locs, n: np.asarray([10.0, 20.0], dtype=np.float32)
    decision = SGNavDecision(scenegraph, frontier_distance_weight=0.5, frontier_scenegraph_score_norm="minmax")
    frontiers = [
        FrontierCluster((1, 1), (1.0, 1.0), [(1, 1)], 1, 2.0, distance_inverse=0.4),
        FrontierCluster((2, 2), (2.0, 2.0), [(2, 2)], 1, 3.0, distance_inverse=0.2),
    ]

    out = decision.choose_frontier(frontiers)

    assert out.scenegraph_scores == [0.0, 1.0]
    assert out.metadata["raw_scenegraph_scores"] == [10.0, 20.0]
    assert out.metadata["scenegraph_score_norm"] == "minmax"
    assert out.selected_index == 1
