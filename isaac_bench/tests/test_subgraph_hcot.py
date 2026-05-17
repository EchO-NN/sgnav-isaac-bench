from isaac_bench.graph.edge_builder import apply_edge_proposals, propose_object_edges_with_llm
from isaac_bench.graph.hcot_scorer import HCoTSubgraphScorer, OpenAICompatibleJSONClient, score_subgraph_with_hcot
from isaac_bench.graph.subgraph_builder import build_object_centered_subgraphs
from isaac_bench.tests.test_edge_builder import FakeLLM, _graph_with_table_chair


class FakeHCoTLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt: str):
        self.calls += 1
        if "Predict the most likely distance" in prompt:
            return {"prior_distance_m": 1.0, "reason": "chairs are often near tables"}
        if "Ask useful questions" in prompt:
            return {"questions": ["Is there a table next to the chair?"]}
        if "Answer the questions" in prompt:
            return {"answers": [{"question": "Is there a table next to the chair?", "answer": "Yes"}]}
        return {"estimated_distance_m": 0.8, "confidence": 0.9, "summary_reason": "subgraph supports the goal nearby"}


def test_build_object_centered_subgraphs_includes_neighbors_and_parents():
    graph = _graph_with_table_chair()
    proposals = propose_object_edges_with_llm([graph.object_nodes["object:table"]], list(graph.object_nodes.values()), FakeLLM())
    apply_edge_proposals(graph, proposals)
    graph.update_group_nodes()

    subgraphs = build_object_centered_subgraphs(graph)
    table_subgraph = next(item for item in subgraphs if item.central_object_id == "object:table")

    assert len(subgraphs) == 2
    assert table_subgraph.parent_room_id == "room:unknown_room"
    assert table_subgraph.parent_group_id is not None
    assert table_subgraph.directly_connected_object_ids == ["object:chair"]


def test_score_subgraph_with_hcot_returns_inverse_distance_probability():
    graph = _graph_with_table_chair()
    subgraph = build_object_centered_subgraphs(graph)[0]

    score = score_subgraph_with_hcot(subgraph, "chair", FakeHCoTLLM())

    assert score.estimated_distance_m == 0.8
    assert score.p_sub > 1.0
    assert "supports" in score.summary_reason


def test_hcot_scorer_caches_by_graph_goal_and_subgraph():
    graph = _graph_with_table_chair()
    subgraph = build_object_centered_subgraphs(graph)[0]
    llm = FakeHCoTLLM()
    scorer = HCoTSubgraphScorer(llm_client=llm)

    scorer.score([subgraph], "chair", graph_version=1)
    scorer.score([subgraph], "chair", graph_version=1)

    assert llm.calls == 4


def test_openai_json_client_default_output_budget_fits_qwen_context():
    client = OpenAICompatibleJSONClient({"enabled": True})

    assert client.max_tokens == 512
