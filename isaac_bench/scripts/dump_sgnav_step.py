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
    parser.add_argument("--output", required=True, help="Path to the JSON artifact to write.")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--goal-category", default=None)
    parser.add_argument("--graph-debug-dump", default=None, help="Optional graph_step_*.json from --debug-graph-dump.")
    parser.add_argument("--result-row", default=None, help="Optional JSON/JSONL result row to enrich candidate and STOP state.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    graph_debug = _read_json_artifact(args.graph_debug_dump) if args.graph_debug_dump else None
    result_row = _read_json_artifact(args.result_row) if args.result_row else None
    metadata = {
        key: value
        for key, value in {
            "episode_id": args.episode_id,
            "scene_id": args.scene_id,
            "goal_category": args.goal_category,
            "schema_only": graph_debug is None and result_row is None,
            "graph_debug_dump": args.graph_debug_dump,
            "result_row": args.result_row,
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
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
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
        "reperception_state": {
            "candidate_credibility": result_row.get("candidate_credibility"),
            "candidate_reperception_steps": result_row.get("candidate_reperception_steps"),
            "candidate_rejected": result_row.get("candidate_rejected"),
            "candidate_accepted": result_row.get("candidate_accepted"),
            "metadata": decision_metadata.get("reperception"),
        },
        "stop_state": {
            "mode": decision.get("mode") or result_row.get("sgnav_decision_mode"),
            "reason": decision.get("reason") or result_row.get("sgnav_decision_reason"),
            "stop_reason": result_row.get("stop_reason"),
            "success": result_row.get("success"),
            "distance_to_goal": result_row.get("distance_to_goal"),
            "target_cells": decision.get("target_cells") or [],
        },
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


if __name__ == "__main__":
    raise SystemExit(main())
