import numpy as np

from isaac_bench.graph.reperception import GraphReperceptionManager, ReperceptionDecision
from isaac_bench.graph.hcot_scorer import SubgraphScore
from isaac_bench.graph.paper_scene_graph import ObjectNode
from isaac_bench.scripts.run_one_episode import build_stop_state_payload
from isaac_bench.graph.decision import NavigationDecision


def _candidate():
    return ObjectNode(
        id="object:1",
        category="chair",
        confidence=1.0,
        point_cloud_world=np.zeros((1, 3), dtype=np.float32),
        bbox_world=np.zeros((2, 3), dtype=np.float32),
        center_world=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        observed_count=1,
        last_seen_step=0,
    )


def _score(p_sub=1.0):
    return SubgraphScore(
        subgraph_id="sg_object_2",
        central_object_id="object:2",
        goal_category="chair",
        estimated_distance_m=1.0,
        p_sub=p_sub,
        summary_reason="supporting object nearby",
        raw_llm_response={},
        central_world=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_reperception_accepts_before_max_observations():
    manager = GraphReperceptionManager(n_max=3, s_thres=0.5)

    result = manager.update(_candidate(), detector_confidence=1.0, subgraph_scores=[_score()])

    assert result.decision == ReperceptionDecision.ACCEPT_GOAL
    assert result.to_dict()["last_s_k"] >= 0.5
    assert result.to_dict()["supporting_subgraphs"][0]["subgraph_id"] == "sg_object_2"


def test_reperception_accepts_when_threshold_reached_exactly_at_n_max():
    manager = GraphReperceptionManager(n_max=2, s_thres=2.0)

    first = manager.update(_candidate(), detector_confidence=1.0, subgraph_scores=[_score()])
    second = manager.update(_candidate(), detector_confidence=1.0, subgraph_scores=[_score()])

    assert first.decision == ReperceptionDecision.CONTINUE_OBSERVING
    assert second.num_reperception_steps == 2
    assert second.accumulated_credibility >= 2.0
    assert second.decision == ReperceptionDecision.ACCEPT_GOAL


def test_reperception_rejects_below_threshold_at_n_max():
    manager = GraphReperceptionManager(n_max=2, s_thres=3.0)

    manager.update(_candidate(), detector_confidence=1.0, subgraph_scores=[_score()])
    second = manager.update(_candidate(), detector_confidence=1.0, subgraph_scores=[_score()])

    assert second.num_reperception_steps == 2
    assert second.accumulated_credibility < 3.0
    assert second.decision == ReperceptionDecision.REJECT_GOAL


def test_reperception_update_does_not_trigger_room_segmentation_or_vlm():
    class RoomSegmenter:
        update_count = 0

        def update(self, *args, **kwargs):
            self.update_count += 1
            raise AssertionError("room segmentation must not run during re-perception")

    class RoomLabeler:
        request_count = 0

        def label_room(self, *args, **kwargs):
            self.request_count += 1
            raise AssertionError("room VLM must not run during re-perception")

    segmenter = RoomSegmenter()
    labeler = RoomLabeler()
    manager = GraphReperceptionManager(n_max=2, s_thres=0.5)

    result = manager.update(_candidate(), detector_confidence=1.0, subgraph_scores=[_score()])

    assert result.decision == ReperceptionDecision.ACCEPT_GOAL
    assert segmenter.update_count == 0
    assert labeler.request_count == 0


def test_stop_state_blocks_stop_before_candidate_confirmation():
    nav_decision = NavigationDecision(
        mode="reperception",
        target_cells=[(1, 1)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="reperception_collecting",
        metadata={"candidate_accepted": False},
    )

    stop_state = build_stop_state_payload(nav_decision, {"success": False, "distance_to_goal": 2.0})

    assert stop_state["stop_allowed"] is False
    assert stop_state["stop_reason"] == "candidate_not_confirmed"
