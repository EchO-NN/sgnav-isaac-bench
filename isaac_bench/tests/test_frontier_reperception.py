import numpy as np

from isaac_bench.graph.frontier_interpolation import frontier_debug_payload, score_frontiers_by_subgraphs, select_highest_score_frontier
from isaac_bench.graph.hcot_scorer import SubgraphScore
from isaac_bench.graph.reperception import GraphReperceptionManager, ReperceptionDecision, compute_goal_candidate_credibility
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.tests.test_edge_builder import _graph_with_table_chair


def _score(subgraph_id: str, center, p_sub: float) -> SubgraphScore:
    return SubgraphScore(
        subgraph_id=subgraph_id,
        central_object_id="object:%s" % subgraph_id,
        goal_category="chair",
        estimated_distance_m=1.0,
        p_sub=p_sub,
        summary_reason="near useful context",
        raw_llm_response={},
        central_world=np.asarray(center, dtype=np.float32),
    )


def test_score_frontiers_by_subgraphs_uses_sum_p_sub_over_distance():
    frontiers = [
        FrontierCluster((0, 0), (0.0, 0.0), [(0, 0)], 1, 1.0),
        FrontierCluster((5, 5), (5.0, 5.0), [(5, 5)], 1, 1.0),
    ]
    scores = [_score("near", (0.5, 0.0, 0.0), 1.0)]

    frontier_scores = score_frontiers_by_subgraphs(frontiers, scores)
    selected = select_highest_score_frontier(frontier_scores)

    assert selected is not None
    assert selected.frontier_id == "frontier_0"
    assert frontier_debug_payload(frontier_scores)["selected_frontier_id"] == "frontier_0"


def test_compute_goal_candidate_credibility_accumulates_subgraph_support():
    graph = _graph_with_table_chair()
    candidate = graph.object_nodes["object:chair"]
    scores = [_score("table", (1.0, 1.0, 0.5), 1.0)]

    s_k = compute_goal_candidate_credibility(candidate, 0.8, scores)

    assert s_k > 0.0


def test_reperception_manager_accepts_or_rejects_by_threshold_and_nmax():
    graph = _graph_with_table_chair()
    candidate = graph.object_nodes["object:chair"]
    manager = GraphReperceptionManager(n_max=3, s_thres=0.8)
    high = [_score("table", candidate.center_world, 1.0)]

    accepted = manager.update(candidate, 0.9, high)

    assert accepted.decision == ReperceptionDecision.ACCEPT_GOAL

    manager = GraphReperceptionManager(n_max=2, s_thres=10.0)
    low = [_score("far", (10.0, 10.0, 0.0), 0.1)]
    first = manager.update(candidate, 0.1, low)
    second = manager.update(candidate, 0.1, low)

    assert first.decision == ReperceptionDecision.CONTINUE_OBSERVING
    assert second.decision == ReperceptionDecision.REJECT_GOAL
