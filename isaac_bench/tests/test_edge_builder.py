import numpy as np

from isaac_bench.graph.edge_builder import (
    apply_edge_proposals,
    propose_object_edges_with_llm,
    verify_long_edge_geometrically,
    verify_short_edge_with_vlm,
)
from isaac_bench.graph.paper_scene_graph import PaperSceneGraph
from isaac_bench.tests.test_paper_scene_graph import _instance


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def complete_json(self, prompt: str):
        self.prompts.append(prompt)
        return [{"src": "object:table", "dst": "object:chair", "relation": "next to", "confidence": 0.8, "reason": "common pair"}]


class FakeVLM:
    def complete_json(self, prompt: str, image=None):
        return {"exists": True, "confidence": 0.9, "reason": "visible"}


def _graph_with_table_chair():
    graph = PaperSceneGraph()
    graph.update_object_and_room_nodes(
        [
            _instance("table", "table", (1.0, 1.0, 0.5)),
            _instance("chair", "chair", (1.5, 1.0, 0.5)),
        ]
    )
    return graph


def test_propose_object_edges_batches_pairs_in_one_llm_call():
    graph = _graph_with_table_chair()
    llm = FakeLLM()

    proposals = propose_object_edges_with_llm([graph.object_nodes["object:table"]], list(graph.object_nodes.values()), llm)

    assert len(llm.prompts) == 1
    assert len(proposals) == 1
    assert proposals[0].relation == "next to"


def test_apply_edge_proposals_adds_unique_edges():
    graph = _graph_with_table_chair()
    proposals = propose_object_edges_with_llm([graph.object_nodes["object:table"]], list(graph.object_nodes.values()), FakeLLM())

    apply_edge_proposals(graph, proposals)
    apply_edge_proposals(graph, proposals)

    assert len(graph.object_edges) == 1


def test_verify_short_edge_with_vlm_uses_json_decision():
    proposal = propose_object_edges_with_llm([_graph_with_table_chair().object_nodes["object:table"]], list(_graph_with_table_chair().object_nodes.values()), FakeLLM())[0]

    kept = verify_short_edge_with_vlm(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool), proposal, FakeVLM())

    assert kept
    assert proposal.pruning_debug["kept"] is True


def test_verify_long_edge_geometrically_records_debug():
    graph = _graph_with_table_chair()
    proposal = propose_object_edges_with_llm([graph.object_nodes["object:table"]], list(graph.object_nodes.values()), FakeLLM())[0]
    occupancy = np.zeros((10, 10), dtype=bool)

    kept = verify_long_edge_geometrically(proposal, graph, occupancy, resolution_m=0.5)

    assert kept
    assert proposal.pruning_debug["same_room"] is True
    assert proposal.pruning_debug["unobstructed"] is True
