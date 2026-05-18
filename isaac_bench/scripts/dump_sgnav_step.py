from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from isaac_bench.metrics.episode_logger import make_jsonable
from isaac_bench.metrics.result_schema import empty_sgnav_step_dump


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an SG-Nav decision dump JSON artifact with the contract-required keys."
    )
    parser.add_argument("--output", "--out", dest="output", required=True, help="Path to the JSON artifact to write.")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--goal-category", default=None)
    parser.add_argument("--episode-file", default=None, help="Optional episode JSONL used only for metadata compatibility.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode row index for --episode-file metadata.")
    parser.add_argument("--graph-debug-dump", default=None, help="Optional graph_step_*.json from --debug-graph-dump.")
    parser.add_argument("--result-row", default=None, help="Optional JSON/JSONL result row to enrich candidate and STOP state.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    graph_debug = _read_json_artifact(args.graph_debug_dump) if args.graph_debug_dump else None
    result_row = _read_json_artifact(args.result_row) if args.result_row else None
    episode_metadata = _read_episode_metadata(args.episode_file, int(args.episode_index)) if args.episode_file else {}
    metadata = {
        key: value
        for key, value in {
            "episode_id": args.episode_id or episode_metadata.get("episode_id"),
            "scene_id": args.scene_id or episode_metadata.get("scene_id"),
            "goal_category": args.goal_category or episode_metadata.get("goal_category"),
            "schema_only": graph_debug is None and result_row is None,
            "graph_debug_dump": args.graph_debug_dump,
            "result_row": args.result_row,
            "episode_file": args.episode_file,
            "episode_index": int(args.episode_index) if args.episode_file else None,
        }.items()
        if value is not None
    }
    payload = empty_sgnav_step_dump(metadata)
    if graph_debug is not None or result_row is not None:
        payload.update(_decision_dump_from_runtime_artifacts(graph_debug or {}, result_row or {}))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(make_jsonable(payload), ensure_ascii=False, indent=2 if args.pretty else None)
    out.write_text(text + "\n", encoding="utf-8")
    print(str(out))
    return 0


def _read_json_artifact(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            return value if isinstance(value, dict) else {}
    return {}


def _read_episode_metadata(path: str, index: int) -> dict:
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return {}
        row = json.loads(lines[max(0, min(int(index), len(lines) - 1))])
        if not isinstance(row, dict):
            return {}
        return {
            "episode_id": row.get("episode_id") or row.get("id"),
            "scene_id": row.get("scene_id"),
            "goal_category": row.get("goal_category") or row.get("goal"),
        }
    except Exception:
        return {}


def _decision_dump_from_runtime_artifacts(graph_debug: dict, result_row: dict) -> dict:
    score_debug = dict(graph_debug.get("score_debug") or result_row.get("paper_frontier_interpolation") or {})
    frontiers = list(graph_debug.get("frontiers") or [])
    decision = dict(graph_debug.get("decision") or {})
    decision_metadata = dict(decision.get("metadata") or result_row.get("sgnav_decision_metadata") or {})
    supporting_subgraphs = list(score_debug.get("top_supporting_subgraphs") or [])
    frontier_scores = list(score_debug.get("frontier_scores") or _frontier_scores_from_debug(frontiers))
    return {
        "objects": list(graph_debug.get("objects") or []),
        "groups": list(graph_debug.get("groups") or []),
        "rooms": list(graph_debug.get("rooms") or []),
        "room_context": dict(graph_debug.get("room_context") or result_row.get("room_context") or _room_context_from_row(result_row)),
        "room_segmentation": dict(
            graph_debug.get("room_segmentation")
            or result_row.get("room_segmentation")
            or {"source": "rose2_structure", "algorithm": "rose2_structure", "room_count": 0, "rooms": []}
        ),
        "object_memory": dict(result_row.get("object_memory_gnn_snapshot") or _object_memory_summary_from_row(result_row)),
        "room_semantics": dict(
            graph_debug.get("room_semantics")
            or result_row.get("room_semantics")
            or {"backend": result_row.get("room_vlm_backend", "unavailable"), "allowed_categories": [], "labels": []}
        ),
        "edges": list(graph_debug.get("edges") or []),
        "subgraphs": supporting_subgraphs,
        "subgraph_texts_or_payloads": [
            {
                "subgraph_id": item.get("subgraph_id"),
                "central_object_id": item.get("central_object_id"),
                "summary_reason": item.get("summary_reason"),
            }
            for item in supporting_subgraphs
        ],
        "llm_scores": {
            "score_backend": graph_debug.get("score_backend") or score_debug.get("mode"),
            "paper_llm_enabled": result_row.get("paper_llm_enabled"),
            "paper_llm_requests": result_row.get("paper_llm_requests"),
            "hcot_llm_enabled": result_row.get("hcot_llm_enabled"),
            "hcot_llm_attempts": result_row.get("hcot_llm_attempts"),
            "hcot_llm_failures": result_row.get("hcot_llm_failures"),
            "scores": [
                {
                    "subgraph_id": item.get("subgraph_id"),
                    "estimated_distance_m": item.get("estimated_distance_m"),
                    "p_sub": item.get("p_sub"),
                    "reason": item.get("summary_reason"),
                }
                for item in supporting_subgraphs
            ],
        },
        "subgraph_probabilities": [
            {
                "subgraph_id": item.get("subgraph_id"),
                "central_object_id": item.get("central_object_id"),
                "p_sub": item.get("p_sub"),
            }
            for item in supporting_subgraphs
        ],
        "frontier_scores": frontier_scores,
        "selected_frontier": _selected_frontier(score_debug, frontier_scores, frontiers, result_row),
        "candidate_goals": list(result_row.get("object_memory_goal_candidates") or []),
        "reperception_state": dict(result_row.get("reperception_state") or _reperception_state_from_metadata(decision_metadata, result_row)),
        "stop_state": dict(result_row.get("stop_state") or _stop_state_from_artifacts(decision, result_row)),
    }


def _frontier_scores_from_debug(frontiers: list[dict]) -> list[dict]:
    out = []
    for item in frontiers:
        out.append(
            {
                "frontier_id": "frontier_%s" % item.get("index"),
                "center_grid": item.get("center"),
                "score": item.get("total"),
                "scenegraph_score": item.get("sg_score"),
                "distance_score": item.get("dist_score"),
                "selected": bool(item.get("selected", False)),
            }
        )
    return out


def _selected_frontier(score_debug: dict, frontier_scores: list[dict], frontiers: list[dict], result_row: dict):
    selected_id = score_debug.get("selected_frontier_id")
    if selected_id is not None:
        for item in frontier_scores:
            if item.get("frontier_id") == selected_id:
                return item
        return {"frontier_id": selected_id}
    for item in frontier_scores:
        if item.get("selected"):
            return item
    for item in frontiers:
        if item.get("selected"):
            return item
    return result_row.get("selected_frontier")


def _room_context_from_row(result_row: dict) -> dict:
    keys = (
        "room_context_source",
        "room_update_invoked_for_frontier_scoring",
        "room_segmentation_ran",
        "room_labeling_ran",
        "room_context_cache_hit",
        "room_mask_count",
        "room_label_count",
        "room_label_requests",
        "room_label_cache_hits",
        "room_call_order_trace",
    )
    return {key: result_row.get(key) for key in keys if key in result_row}


def _object_memory_summary_from_row(result_row: dict) -> dict:
    tracks = list(result_row.get("object_memory_tracks") or [])
    raw = list(result_row.get("raw_detection_log") or [])
    return {
        "raw_detection_count": int(len(raw)),
        "stable_track_count": int(len([item for item in tracks if bool(item.get("used_for_policy_graph", True))])),
        "tentative_track_count": int(len([item for item in tracks if not bool(item.get("used_for_policy_graph", True))])),
        "mask_association_count": int(len([item for item in raw if item.get("associated_track_id")])),
        "partial_edge_count": int(len([item for item in raw if item.get("visibility_status") == "partial_edge"])),
        "contained_child_count": int(len([item for item in tracks if item.get("parent_track_id")])),
        "raw_detections": raw[:128],
        "object_tracks": tracks[:128],
    }


def _reperception_state_from_metadata(decision_metadata: dict, result_row: dict) -> dict:
    reperception = dict(decision_metadata.get("reperception") or {})
    candidate_id = reperception.get("candidate_id", decision_metadata.get("selected_candidate_id"))
    decision = reperception.get("decision")
    if decision is None:
        if bool(result_row.get("candidate_accepted", decision_metadata.get("candidate_accepted", False))):
            decision = "ACCEPT_GOAL"
        elif bool(result_row.get("candidate_rejected", decision_metadata.get("candidate_rejected", False))):
            decision = "REJECT_GOAL"
        elif candidate_id is not None:
            decision = "CONTINUE_OBSERVING"
    return {
        "candidate_id": candidate_id,
        "num_reperception_steps": int(
            reperception.get(
                "num_reperception_steps",
                result_row.get("candidate_reperception_steps", decision_metadata.get("candidate_reperception_steps", 0)),
            )
            or 0
        ),
        "accumulated_credibility": float(
            reperception.get(
                "accumulated_credibility",
                result_row.get("candidate_credibility", decision_metadata.get("candidate_credibility", 0.0)),
            )
            or 0.0
        ),
        "last_s_k": float(reperception.get("last_s_k", reperception.get("s_k", 0.0)) or 0.0),
        "detector_confidence": float(reperception.get("detector_confidence", 0.0) or 0.0),
        "supporting_subgraphs": list(reperception.get("supporting_subgraphs") or []),
        "decision": decision,
    }


def _stop_state_from_artifacts(decision: dict, result_row: dict) -> dict:
    mode = decision.get("mode") or result_row.get("sgnav_decision_mode")
    stop_allowed = bool(mode == "stop" or result_row.get("stop_called", False))
    return {
        "stop_allowed": stop_allowed,
        "stop_reason": result_row.get("stop_reason") or decision.get("reason"),
        "candidate_confirmed": bool(stop_allowed),
        "mode": mode,
        "reason": decision.get("reason") or result_row.get("sgnav_decision_reason"),
        "success": result_row.get("success"),
        "distance_to_goal": result_row.get("distance_to_goal"),
        "target_cells": decision.get("target_cells") or [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
