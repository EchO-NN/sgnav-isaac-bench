import numpy as np

from isaac_bench.graph.hcot_scorer import (
    build_hcot_answer_prompt,
    build_hcot_final_distance_prompt,
    build_hcot_prior_distance_prompt,
    build_hcot_question_prompt,
    score_subgraph_with_hcot,
)
from isaac_bench.graph.subgraph_builder import Subgraph


class RecordingHCoTLLM:
    def __init__(self):
        self.prompts = []

    def complete_json(self, prompt: str):
        self.prompts.append(prompt)
        if "stage 1/4" in prompt:
            return {"prior_distance_m": 2.0, "reason": "table and sink are usually separated"}
        if "stage 2/4" in prompt:
            return {"questions": ["Is the table in a kitchen?", "Which objects connect the table to the sink?"]}
        if "stage 3/4" in prompt:
            return {
                "answers": [
                    {"question": "Is the table in a kitchen?", "answer": "The parent room is kitchen."},
                    {"question": "Which objects connect the table to the sink?", "answer": "A counter is nearby."},
                ]
            }
        if "stage 4/4" in prompt:
            return {"estimated_distance_m": 0.8, "confidence": 0.9, "summary_reason": "kitchen context supports sink nearby"}
        raise AssertionError("unexpected prompt")


def _subgraph():
    return Subgraph(
        id="sg_object_table",
        central_object_id="object:table",
        central_object_category="table",
        parent_room_id="room:room_0001",
        parent_group_id="group:kitchen_table",
        directly_connected_object_ids=["object:counter"],
        nodes=[
            {"id": "object:table", "type": "object", "category": "table", "confidence": 0.9},
            {"id": "object:counter", "type": "object", "category": "counter", "confidence": 0.8},
            {
                "id": "room:room_0001",
                "type": "room",
                "category": "kitchen",
                "confidence": 0.72,
                "mask_id": "room_0001",
                "mask_source": "online_geometry_watershed",
            },
            {"id": "group:kitchen_table", "type": "group", "category": "counter, table", "object_ids": ["object:table", "object:counter"]},
        ],
        edges=[
            {"src": "object:table", "dst": "object:counter", "relation": "next to", "confidence": 0.8},
            {"src": "object:table", "dst": "room:room_0001", "relation": "belongs_to", "confidence": 0.9},
        ],
        central_world=np.asarray([1.0, 2.0, 0.5], dtype=np.float32),
    )


def test_prompt_builders_include_room_group_edges_and_schema():
    sg = _subgraph()
    prompts = [
        build_hcot_prior_distance_prompt(sg, "sink"),
        build_hcot_question_prompt(sg, "sink", 2.0),
        build_hcot_answer_prompt(sg, "sink", ["Is the table in a kitchen?"]),
        build_hcot_final_distance_prompt(sg, "sink", 2.0, ["q"], ["a"]),
    ]

    text = "\n".join(prompts)
    assert "object:table" in text
    assert "sink" in text
    assert "room_0001" in text
    assert "kitchen" in text
    assert "group:kitchen_table" in text
    assert "object:counter" in text
    assert "next to" in text
    assert "strict JSON" in text


def test_hcot_paper_prompt_sequence_and_probability():
    llm = RecordingHCoTLLM()
    score = score_subgraph_with_hcot(_subgraph(), "sink", llm, min_distance_m=0.25)

    assert [("stage %d/4" % i) in prompt for i, prompt in enumerate(llm.prompts, start=1)] == [True, True, True, True]
    assert score.estimated_distance_m == 0.8
    assert score.p_sub == 1.0 / 0.8
    assert score.raw_llm_response["paper_hcot"] is True
    assert score.raw_llm_response["fallback"] is False
    assert score.raw_llm_response["stage1_prior_distance_m"] == 2.0
    assert score.raw_llm_response["stage4_final_distance_m"] == 0.8
    assert score.raw_llm_response["p_sub_formula"] == "1 / max(stage4_final_distance_m, min_distance_m)"
