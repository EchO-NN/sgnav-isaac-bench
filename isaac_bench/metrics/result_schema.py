from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

from isaac_bench.mapping.upstream_rose2_pure_python_adapter import validate_upstream_rose2_source_root


REQUIRED_RESULT_FIELDS = [
    "metric_valid",
    "strict_benchmark",
    "fallbacks_used",
    "detector_backend",
    "segmenter_backend",
    "llm_backend",
    "sim_backend",
    "map_source",
    "policy_name",
    "sgnav_decision_mode",
    "sgnav_decision_reason",
    "object_memory_count",
    "goal_candidate_count",
    "frontier_count",
    "selected_frontier",
    "stop_reason",
    "success",
    "spl",
    "softspl",
    "distance_to_goal",
    "path_length",
    "steps",
    "collisions",
    "stuck_events",
    "timeout",
    "perception_latency_ms",
    "mapping_latency_ms",
    "graph_latency_ms",
    "llm_latency_ms",
    "planning_latency_ms",
]


SGNAV_STEP_DUMP_KEYS = [
    "objects",
    "groups",
    "rooms",
    "edges",
    "room_context",
    "room_segmentation",
    "room_semantics",
    "subgraphs",
    "subgraph_texts_or_payloads",
    "llm_scores",
    "subgraph_probabilities",
    "frontier_scores",
    "selected_frontier",
    "candidate_goals",
    "reperception_state",
    "stop_state",
]


class BenchmarkAssetError(RuntimeError):
    """Raised when a strict benchmark run is missing a required asset."""


def complete_result_row(row: Mapping[str, object], args: object | None = None) -> dict:
    out = dict(row)
    strict_benchmark = bool(_get_arg(args, "strict_benchmark", out.get("strict_benchmark", False)))
    ablation_name = _get_arg(args, "ablation_name", out.get("ablation_name"))
    detector_backend = str(_get_arg(args, "detector", out.get("detector_backend", out.get("detector", "none"))))
    segmenter_backend = str(_get_arg(args, "segmenter", out.get("segmenter_backend", out.get("segmenter", "none"))))
    sim_backend = str(_get_arg(args, "sim_backend", out.get("sim_backend", "map")))
    llm_backend = _infer_llm_backend(out, args)
    map_source = _infer_map_source(out, sim_backend)
    policy_name = _infer_policy_name(out, args, sim_backend)

    out["strict_benchmark"] = strict_benchmark
    out["ablation_name"] = ablation_name
    out["detector_backend"] = detector_backend
    out["segmenter_backend"] = segmenter_backend
    out["llm_backend"] = llm_backend
    out["sim_backend"] = sim_backend
    out["map_source"] = map_source
    out["policy_name"] = policy_name
    out["steps"] = int(out.get("steps", out.get("num_steps", 0)) or 0)
    out["collisions"] = int(out.get("collisions", out.get("num_collisions", 0)) or 0)
    out["stuck_events"] = int(out.get("stuck_events", out.get("num_stuck_events", 0)) or 0)
    out["timeout"] = bool(
        out.get("timeout", False)
        or out.get("failure_reason") == "max_control_steps"
        or out.get("terminal_reason") == "timeout"
    )
    out["object_memory_count"] = int(out.get("object_memory_count", 0) or 0)
    out["goal_candidate_count"] = int(
        out.get("goal_candidate_count", len(out.get("object_memory_goal_candidates", []) or [])) or 0
    )
    out["frontier_count"] = int(out.get("frontier_count", out.get("frontier_clusters", 0)) or 0)
    out["selected_frontier"] = out.get("selected_frontier", out.get("frontier_center_grid"))
    out["stop_reason"] = _infer_stop_reason(out)

    for key in (
        "sgnav_decision_mode",
        "sgnav_decision_reason",
        "success",
        "spl",
        "softspl",
        "distance_to_goal",
        "path_length",
    ):
        out.setdefault(key, None)
    for key in (
        "perception_latency_ms",
        "mapping_latency_ms",
        "graph_latency_ms",
        "llm_latency_ms",
        "planning_latency_ms",
    ):
        out[key] = float(out.get(key, 0.0) or 0.0)

    if (
        bool(out.get("success", False))
        and bool(out.get("success_requires_sgnav_stop", False))
        and not bool(out.get("policy_stop_confirmed", False))
    ):
        out["success"] = False
        out["spl"] = 0.0
        out["failure_reason"] = out.get("failure_reason") or "sgnav_stop_required"
        out["stop_reason"] = "sgnav_stop_required"

    fallbacks = _stable_unique([*list(out.get("fallbacks_used", []) or []), *_infer_fallbacks(out, args)])
    out["fallbacks_used"] = fallbacks
    explicit_invalid = out.get("metric_valid") is False
    out["metric_valid"] = bool((not fallbacks) and not explicit_invalid)
    if strict_benchmark and fallbacks:
        out["metric_valid"] = False

    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in out]
    if missing:
        raise ValueError("result row missing required fields after completion: %s" % ", ".join(missing))
    return out


def validate_strict_benchmark_assets(args: object) -> None:
    if not bool(_get_arg(args, "strict_benchmark", False)):
        return
    detector = str(_get_arg(args, "detector", "") or "").strip().lower()
    segmenter = str(_get_arg(args, "segmenter", "") or "").strip().lower()
    ablation_name = str(_get_arg(args, "ablation_name", "") or "").strip().lower()
    if detector == "yolo_world" and ablation_name != "legacy_yolo_world_ablation":
        raise BenchmarkAssetError(
            "strict SG-Nav metric path now requires GroundingDINO-B/Swin-B; YOLO-World is legacy/debug unless --ablation-name legacy_yolo_world_ablation"
        )
    if detector == "grounding_dino":
        _require_existing_asset("GroundingDINO-B/Swin-B checkpoint", _get_arg(args, "grounding_dino_checkpoint", ""))
        _require_existing_asset("GroundingDINO-B/Swin-B config", _get_arg(args, "grounding_dino_config", ""))
        if segmenter == "sam2":
            _require_existing_asset("SAM2 checkpoint", _get_arg(args, "sam2_checkpoint", ""))
    if detector == "yolo_world":
        _require_existing_asset("YOLO-World model", _get_arg(args, "yolo_world_model", ""))
        if segmenter == "sam2":
            _require_existing_asset("SAM2 checkpoint", _get_arg(args, "sam2_checkpoint", ""))
    sgnav_mode = str(_get_arg(args, "sgnav_mode", "") or "").strip().lower()
    if sgnav_mode == "paper" and not bool(_get_arg(args, "llm_enabled", False)):
        raise BenchmarkAssetError("strict SG-Nav paper mode requires --llm-enabled true and a reachable LLM/VLM endpoint")
    room_map_mode = str(_get_arg(args, "room_map_mode", "") or "").strip().lower()
    if room_map_mode in {"observed_rooms_json", "rooms_json", "observed"} and ablation_name != "oracle_room_ablation":
        raise BenchmarkAssetError(
            "strict SG-Nav metric path requires upstream_rose2_vertical_or_free room masks; rooms.json/oracle room maps are not allowed"
        )
    if room_map_mode in {"online_geometry_watershed", "online_geometry_watershed_vlm"} and ablation_name != "legacy_watershed_room_ablation":
        raise BenchmarkAssetError(
            "strict SG-Nav metric path requires upstream_rose2_vertical_or_free room masks; watershed is debug/ablation-only"
        )
    if room_map_mode in {"online_rose2_structure", "rose2_structure", "online_rose2_structure_vlm"} and ablation_name != "local_rose2_lite_room_ablation":
        raise BenchmarkAssetError(
            "strict SG-Nav metric path requires upstream_rose2_vertical_or_free room masks; local ROSE2-lite is debug/ablation-only"
        )
    if room_map_mode in {"upstream_rose2_vertical_or_free", "upstream_rose2_vertical_or_free_vlm", "upstream_rose2_pure_python", "upstream_rose2_pure_python_vlm"}:
        room_cfg = dict(_get_arg(args, "room_segmentation_config", {}) or {})
        if bool(room_cfg.get("require_upstream_source_for_strict", True)) and not bool(_get_arg(args, "allow_debug_fallbacks", False)):
            try:
                validate_upstream_rose2_source_root(
                    room_cfg.get("source_root"),
                    env_name=str(room_cfg.get("upstream_repo_env", "ROSE2_SOURCE_ROOT") or "ROSE2_SOURCE_ROOT"),
                    fail=True,
                )
            except FileNotFoundError as exc:
                raise BenchmarkAssetError(str(exc)) from exc


def empty_sgnav_step_dump(metadata: Optional[Mapping[str, object]] = None) -> dict:
    payload = {
        "objects": [],
        "groups": [],
        "rooms": [],
        "room_context": {},
        "room_segmentation": {
            "source": "upstream_rose2_vertical_or_free",
            "algorithm": "upstream_rose2_vertical_or_free",
            "room_count": 0,
            "rooms": [],
        },
        "object_memory": {
            "raw_detection_count": 0,
            "stable_track_count": 0,
            "tentative_track_count": 0,
            "mask_association_count": 0,
            "partial_edge_count": 0,
            "contained_child_count": 0,
        },
        "room_semantics": {
            "backend": "unavailable",
            "allowed_categories": [],
            "labels": [],
        },
        "edges": [],
        "subgraphs": [],
        "subgraph_texts_or_payloads": [],
        "llm_scores": [],
        "subgraph_probabilities": [],
        "frontier_scores": [],
        "selected_frontier": None,
        "candidate_goals": [],
        "reperception_state": {},
        "stop_state": {},
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    payload["contract_version"] = "sgnav_step_dump_v1"
    return payload


def _get_arg(args: object | None, name: str, default: object = None) -> object:
    if args is None:
        return default
    return getattr(args, name, default)


def _infer_llm_backend(row: Mapping[str, object], args: object | None) -> str:
    if row.get("llm_backend"):
        return str(row["llm_backend"])
    if bool(_get_arg(args, "vllm_frontier_scoring", row.get("vllm_frontier_scoring", False))):
        return "vllm_frontier"
    if bool(_get_arg(args, "llm_enabled", row.get("hcot_llm_enabled", row.get("paper_llm_enabled", False)))):
        return "openai_compatible"
    return "deterministic_local"


def _infer_map_source(row: Mapping[str, object], sim_backend: str) -> str:
    if row.get("map_source"):
        return str(row["map_source"])
    if row.get("mapping_source"):
        return str(row["mapping_source"])
    if sim_backend == "isaac":
        return "online_rgbd_depth"
    return "static_preprocessed"


def _infer_policy_name(row: Mapping[str, object], args: object | None, sim_backend: str) -> str:
    if row.get("policy_name"):
        return str(row["policy_name"])
    policy = _get_arg(args, "policy", None)
    if policy:
        return str(policy)
    planner = str(_get_arg(args, "planner", row.get("planner", "astar")))
    sgnav_mode = str(_get_arg(args, "sgnav_mode", row.get("sgnav_mode", "legacy")))
    if sim_backend == "isaac":
        return "sgnav_%s_%s" % (sgnav_mode, planner)
    return "%s_static_map_baseline" % planner


def _infer_stop_reason(row: Mapping[str, object]) -> str:
    if row.get("stop_reason"):
        return str(row["stop_reason"])
    if row.get("failure_reason"):
        return str(row["failure_reason"])
    if bool(row.get("success", False)):
        return "success"
    if row.get("terminal_reason"):
        return str(row["terminal_reason"])
    if bool(row.get("timeout", False)):
        return "timeout"
    return "not_success"


def _infer_fallbacks(row: Mapping[str, object], args: object | None) -> list[str]:
    detector = str(row.get("detector_backend", "")).strip().lower()
    segmenter = str(row.get("segmenter_backend", "")).strip().lower()
    llm_backend = str(row.get("llm_backend", "")).strip().lower()
    sim_backend = str(row.get("sim_backend", "")).strip().lower()
    ablation_name = row.get("ablation_name")
    fallbacks: list[str] = []
    if detector == "dry_run":
        fallbacks.append("dry_run_detector")
    if detector == "none":
        fallbacks.append("detector_none")
    if detector in {"yolo_world", "grounding_dino"} and segmenter != "sam2":
        fallbacks.append("sam2_missing_or_disabled")
    if detector == "yolo_world" and str(ablation_name or "") != "legacy_yolo_world_ablation":
        fallbacks.append("legacy_yolo_world_detector")
    if bool(_get_arg(args, "seed_gt_object_memory", row.get("seed_gt_object_memory", False))):
        fallbacks.append("seeded_gt_object_memory")
    if int(row.get("seeded_object_memory_count", 0) or 0) > 0:
        fallbacks.append("seeded_gt_object_memory")
    if bool(_get_arg(args, "allow_gt_goal_fallback", row.get("allow_gt_goal_fallback", False))):
        fallbacks.append("gt_goal_fallback_allowed")
    if row.get("sgnav_decision_mode") == "gt_goal_fallback":
        fallbacks.append("gt_goal_fallback_used")
    if llm_backend == "deterministic_local" and not ablation_name:
        fallbacks.append("llm_deterministic_local")
    if int(row.get("hcot_llm_fallback_count", 0) or 0) > 0:
        fallbacks.append("llm_deterministic_local")
    room_vlm_backend = str(row.get("room_vlm_backend", "") or "").strip().lower()
    if room_vlm_backend in {"deterministic_debug", "unavailable"} and not ablation_name:
        fallbacks.append("room_vlm_unavailable" if room_vlm_backend == "unavailable" else "room_vlm_deterministic_debug")
    if bool(row.get("room_vlm_invalid_json", False)):
        fallbacks.append("room_vlm_invalid_json")
    room_map_mode = str(row.get("room_map_mode", _get_arg(args, "room_map_mode", "")) or "").strip().lower()
    if room_map_mode in {"observed_rooms_json", "rooms_json", "observed"} and str(ablation_name or "") != "oracle_room_ablation":
        fallbacks.append("oracle_room_map")
    if room_map_mode in {"online_rose2_structure", "rose2_structure", "online_rose2_structure_vlm"} and str(ablation_name or "") != "local_rose2_lite_room_ablation":
        fallbacks.append("local_rose2_lite_room_segmentation")
    if room_map_mode in {"online_geometry_watershed", "online_geometry_watershed_vlm"} and str(ablation_name or "") != "legacy_watershed_room_ablation":
        fallbacks.append("legacy_watershed_room_segmentation")
    if sim_backend == "map":
        fallbacks.append("static_map_planning")
    if bool(_get_arg(args, "static_nearfield_map", row.get("static_nearfield_map", False))):
        fallbacks.append("static_nearfield_map")
    if bool(_get_arg(args, "frontier_allow_near_fallback", row.get("frontier_allow_near_fallback", False))):
        fallbacks.append("frontier_near_fallback")
    return fallbacks


def _stable_unique(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        item = str(value)
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _require_existing_asset(kind: str, value: object) -> None:
    path = "" if value is None else str(value)
    if not path:
        raise BenchmarkAssetError("Missing %s path for strict benchmark run" % kind)
    if not Path(path).expanduser().exists():
        raise BenchmarkAssetError("Missing %s for strict benchmark run: %s" % (kind, path))
