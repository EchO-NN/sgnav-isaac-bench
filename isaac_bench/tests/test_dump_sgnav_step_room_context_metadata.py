import json

from isaac_bench.scripts.dump_sgnav_step import main as dump_sgnav_step_main


def test_dump_sgnav_step_includes_room_context_reperception_and_stop_state(tmp_path):
    graph_debug = tmp_path / "graph_step_000001.json"
    result_row = tmp_path / "results.jsonl"
    out = tmp_path / "sgnav_step.json"
    room_context = {
        "room_context_source": "online_geometry_watershed_vlm",
        "room_update_invoked_for_frontier_scoring": True,
        "room_segmentation_ran": True,
        "room_labeling_ran": False,
        "room_context_cache_hit": False,
        "room_mask_count": 1,
        "room_label_count": 1,
        "room_label_requests": 0,
        "room_label_cache_hits": 1,
        "room_call_order_trace": [
            "frontier_extraction",
            "room_context_for_frontier_scoring",
            "scenegraph_update",
            "hcot_subgraph_scoring",
            "frontier_interpolation",
            "frontier_selection",
        ],
    }
    graph_debug.write_text(
        json.dumps(
            {
                "objects": [],
                "groups": [],
                "rooms": [],
                "edges": [],
                "room_context": room_context,
                "decision": {"mode": "frontier", "reason": "selected_frontier", "target_cells": [[1, 2]]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result_row.write_text(
        json.dumps(
            {
                "reperception_state": {
                    "candidate_id": "object:1",
                    "num_reperception_steps": 2,
                    "accumulated_credibility": 0.8,
                    "last_s_k": 0.4,
                    "detector_confidence": 0.9,
                    "supporting_subgraphs": [],
                    "decision": "ACCEPT_GOAL",
                },
                "stop_state": {"stop_allowed": False, "stop_reason": "candidate_not_confirmed"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert dump_sgnav_step_main(
        [
            "--output",
            str(out),
            "--graph-debug-dump",
            str(graph_debug),
            "--result-row",
            str(result_row),
        ]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["room_context"] == room_context
    assert payload["reperception_state"]["decision"] == "ACCEPT_GOAL"
    assert payload["stop_state"]["stop_allowed"] is False
