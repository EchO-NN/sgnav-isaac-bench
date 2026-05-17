import numpy as np

from isaac_bench.graph.frontier_interpolation import score_frontiers_by_subgraphs
from isaac_bench.graph.hcot_scorer import SubgraphScore
from isaac_bench.mapping.frontier import FrontierCluster


def test_frontier_interpolation_uses_hcot_psub_inverse_distance():
    frontiers = [
        FrontierCluster((0, 0), (0.0, 0.0), [(0, 0)], 1, 2.0),
        FrontierCluster((0, 5), (5.0, 0.0), [(0, 5)], 1, 5.0),
    ]
    subgraph_scores = [
        SubgraphScore(
            subgraph_id="sg_a",
            central_object_id="object:a",
            goal_category="sink",
            estimated_distance_m=2.0,
            p_sub=10.0,
            summary_reason="strong HCoT probability",
            raw_llm_response={"paper_hcot": True},
            central_world=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
        ),
        SubgraphScore(
            subgraph_id="sg_b",
            central_object_id="object:b",
            goal_category="sink",
            estimated_distance_m=0.5,
            p_sub=1.0,
            summary_reason="weaker HCoT probability",
            raw_llm_response={"paper_hcot": True},
            central_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        ),
    ]

    scores = score_frontiers_by_subgraphs(frontiers, subgraph_scores, eps=0.25)

    assert scores[1].score > scores[0].score
    assert scores[1].top_supporting_subgraphs[0]["p_sub"] == 10.0
