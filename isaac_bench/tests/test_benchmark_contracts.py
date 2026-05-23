from __future__ import annotations

import json
from argparse import Namespace

import pytest

from isaac_bench.config import load_config
from isaac_bench.metrics.result_schema import (
    REQUIRED_RESULT_FIELDS,
    SGNAV_STEP_DUMP_KEYS,
    BenchmarkAssetError,
    complete_result_row,
    validate_strict_benchmark_assets,
)
from isaac_bench.scripts.check_assets import _path_from_env_or_config, main as check_assets_main
from isaac_bench.scripts.dump_sgnav_step import main as dump_sgnav_step_main
from isaac_bench.scripts.run_benchmark import main as run_benchmark_main
from isaac_bench.scripts.run_one_episode import main as run_one_episode_main


def _args(**overrides):
    data = {
        "strict_benchmark": True,
        "ablation_name": None,
        "detector": "grounding_dino",
        "segmenter": "sam2",
        "sim_backend": "isaac",
        "planner": "astar",
        "sgnav_mode": "paper",
        "llm_enabled": False,
        "vllm_frontier_scoring": False,
        "seed_gt_object_memory": False,
        "allow_gt_goal_fallback": False,
        "static_nearfield_map": False,
        "frontier_allow_near_fallback": False,
        "room_map_mode": "vertical_free_gap_closure_v1_vlm",
        "room_segmentation_config": {
            "backend": "vertical_free_gap_closure_v1",
            "require_upstream_source_for_strict": False,
        },
        "yolo_world_model": "data/models/yolov8l-worldv2.pt",
        "grounding_dino_checkpoint": "data/models/groundingdino_swinb_cogcoor.pth",
        "grounding_dino_config": "data/models/GroundingDINO_SwinB.cfg.py",
        "sam2_checkpoint": "data/models/sam2.1_hiera_small.pt",
    }
    data.update(overrides)
    return Namespace(**data)


def _base_row():
    return {
        "success": False,
        "spl": 0.0,
        "softspl": 0.0,
        "distance_to_goal": 1.0,
        "path_length": 0.0,
    }


def test_result_schema_completes_required_fields():
    row = complete_result_row(_base_row(), _args(ablation_name="local_llm_ablation"))

    assert all(field in row for field in REQUIRED_RESULT_FIELDS)
    assert row["strict_benchmark"] is True
    assert row["fallbacks_used"] == []
    assert row["metric_valid"] is True
    assert row["steps"] == 0
    assert row["collisions"] == 0
    assert row["timeout"] is False


def test_dry_run_detector_marks_row_non_metric():
    row = complete_result_row(
        _base_row(),
        _args(detector="dry_run", segmenter="none", sim_backend="map"),
    )

    assert row["metric_valid"] is False
    assert "dry_run_detector" in row["fallbacks_used"]
    assert "static_map_planning" in row["fallbacks_used"]


def test_seeded_gt_memory_marks_row_non_metric():
    row = complete_result_row(_base_row(), _args(seed_gt_object_memory=True))

    assert row["metric_valid"] is False
    assert "seeded_gt_object_memory" in row["fallbacks_used"]


def test_metric_pose_outside_static_map_marks_row_non_metric():
    row = complete_result_row(
        {**_base_row(), "metric_pose_outside_static_map_count": 2},
        _args(),
    )

    assert row["metric_valid"] is False
    assert "metric_pose_outside_static_map" in row["fallbacks_used"]


def test_unmarked_local_deterministic_llm_marks_row_non_metric():
    row = complete_result_row(_base_row(), _args())

    assert row["metric_valid"] is False
    assert "llm_deterministic_local" in row["fallbacks_used"]


def test_named_ablation_can_carry_local_deterministic_scorer():
    row = complete_result_row(_base_row(), _args(ablation_name="local_deterministic_llm"))

    assert "llm_deterministic_local" not in row["fallbacks_used"]
    assert row["metric_valid"] is True
    assert row["ablation_name"] == "local_deterministic_llm"


def test_strict_missing_grounding_dino_checkpoint_fails_clearly(tmp_path):
    args = _args(grounding_dino_checkpoint=str(tmp_path / "missing-grounding-dino.pth"))

    with pytest.raises(BenchmarkAssetError, match="Missing GroundingDINO-B/Swin-B checkpoint"):
        validate_strict_benchmark_assets(args)


def test_strict_yolo_world_requires_named_legacy_ablation(tmp_path):
    yolo = tmp_path / "yolo.pt"
    yolo.write_bytes(b"placeholder")
    args = _args(detector="yolo_world", yolo_world_model=str(yolo), llm_enabled=True)

    with pytest.raises(BenchmarkAssetError, match="requires GroundingDINO-B/Swin-B"):
        validate_strict_benchmark_assets(args)


def test_strict_missing_grounding_dino_cli_fails_without_traceback(tmp_path, capsys):
    status = run_one_episode_main(
        [
            "--episode-file",
            str(tmp_path / "episodes.jsonl"),
            "--detector",
            "grounding_dino",
            "--segmenter",
            "sam2",
            "--sim-backend",
            "map",
            "--strict-benchmark",
            "true",
            "--grounding-dino-checkpoint",
            str(tmp_path / "missing-grounding-dino.pth"),
            "--grounding-dino-config",
            str(tmp_path / "missing-grounding-dino.py"),
        ]
    )
    captured = capsys.readouterr()

    assert status == 2
    assert "Missing GroundingDINO-B/Swin-B checkpoint" in captured.err
    assert "Traceback" not in captured.err


def test_strict_missing_grounding_dino_batch_cli_fails_without_traceback(tmp_path, capsys):
    status = run_benchmark_main(
        [
            "--episode-file",
            str(tmp_path / "episodes.jsonl"),
            "--detector",
            "grounding_dino",
            "--sim-backend",
            "isaac",
            "--strict-benchmark",
            "true",
            "--grounding-dino-checkpoint",
            str(tmp_path / "missing-grounding-dino.pth"),
            "--grounding-dino-config",
            str(tmp_path / "missing-grounding-dino.py"),
        ]
    )
    captured = capsys.readouterr()

    assert status == 2
    assert "Missing GroundingDINO-B/Swin-B checkpoint" in captured.err
    assert "Traceback" not in captured.err


def test_strict_missing_sam2_checkpoint_fails_clearly(tmp_path):
    gdino = tmp_path / "groundingdino.pth"
    gdino.write_bytes(b"placeholder")
    cfg = tmp_path / "GroundingDINO_SwinB.cfg.py"
    cfg.write_text("modelname='groundingdino'\n", encoding="utf-8")
    args = _args(grounding_dino_checkpoint=str(gdino), grounding_dino_config=str(cfg), sam2_checkpoint=str(tmp_path / "missing-sam2.pt"))

    with pytest.raises(BenchmarkAssetError, match="Missing SAM2 checkpoint"):
        validate_strict_benchmark_assets(args)


def test_config_defaults_match_benchmark_contract():
    cfg = load_config("isaac_bench/configs/isaac_bench.yaml")

    assert cfg["benchmark"]["strict_benchmark"] is True
    assert cfg["benchmark"]["allow_debug_fallbacks"] is False
    assert cfg["benchmark"]["policy"] == "sgnav_original"
    assert cfg["repo"]["detector"] == "grounding_dino"
    assert cfg["perception"]["grounding_dino"]["variant"] == "GroundingDINO-B/Swin-B"
    assert cfg["perception"]["grounding_dino"]["checkpoint"].endswith("groundingdino_swinb_cogcoor.pth")
    assert cfg["perception"]["grounding_dino"]["config"].endswith("GroundingDINO_SwinB.cfg.py")
    assert cfg["perception"]["segmenter"] == "sam2"
    assert cfg["perception"]["confidence_threshold"] == 0.45
    assert cfg["perception"]["min_valid_detection_confidence"] == 0.45
    assert cfg["sgnav"]["seed_gt_object_memory"] is False
    assert cfg["sgnav"]["allow_gt_goal_fallback"] is False
    assert cfg["nearfield_static_map"]["enabled"] is False
    assert cfg["mapping"]["frontier_allow_near_fallback"] is False
    assert cfg["mapping"]["frontier_min_distance_m"] == 1.0
    assert cfg["mapping"]["room_map_mode"] == "roomseg_evidence_line_closure_v3_vlm"
    assert cfg["mapping"]["strict_no_oracle_rooms"] is True
    assert cfg["mapping"]["room_segmentation"]["algorithm"] == "roomseg_evidence_line_closure_v3"
    assert cfg["mapping"]["room_segmentation"]["backend"] == "roomseg_evidence_line_closure_v3"
    assert cfg["mapping"]["room_segmentation"]["source_mode"] == "declutter_reconstruct_external"
    assert cfg["mapping"]["room_segmentation"]["legacy_watershed_allowed"] == "debug_only"
    assert cfg["mapping"]["room_segmentation"]["local_rose2_lite_allowed"] == "debug_only"
    assert cfg["mapping"]["room_segmentation"]["require_upstream_source_for_strict"] is False
    assert cfg["mapping"]["room_segmentation"]["allow_source_form_in_metric"] is False
    assert cfg["mapping"]["room_segmentation"]["allow_silent_fallback"] is False
    assert cfg["mapping"]["room_segmentation"]["upstream_repo_env"] == "ROSE2_SOURCE_ROOT"
    assert cfg["mapping"]["room_segmentation"]["run_only_before_frontier_scoring"] is True
    assert cfg["mapping"]["room_segmentation"]["finalization_mode"] == "no_merge_until_geometry_verified"
    assert cfg["mapping"]["room_segmentation"]["strict_disallow_legacy_fallback"] is True
    assert cfg["mapping"]["room_segmentation"]["source_form"]["min_cell_area_m2"] == 0.35
    assert cfg["mapping"]["room_segmentation"]["open_boundary_merge"] is True
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["z_min_m"] == 0.10
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["z_max_m"] == 2.50
    assert cfg["mapping"]["room_segmentation"]["vertical_free_roomseg"]["enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["vertical_free_roomseg"]["doorway_width_max_m"] == 1.60
    assert cfg["mapping"]["room_segmentation"]["vertical_free_gap_closure"]["enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["vertical_free_gap_closure"]["close_max_gap_m"] == 1.50
    assert cfg["mapping"]["room_segmentation"]["vertical_free_gap_closure"]["topology_verify_enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["wall_confidence_threshold"] == 0.55
    assert cfg["mapping"]["room_segmentation"]["vertical_free_suppression_weight"] == 0.35
    assert cfg["mapping"]["room_segmentation"]["furniture_suppression_radius_m"] == 0.90
    assert cfg["mapping"]["room_segmentation"]["wall_like_aspect_ratio_min"] == 4.0
    assert cfg["mapping"]["room_segmentation"]["rose2"]["wall_extension_enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["rose2"]["wall_extension_band_m"] == 0.45
    assert cfg["mapping"]["room_segmentation"]["rose2"]["wall_extension_margin_m"] == 0.15
    assert cfg["room_semantics"]["use_premerge_labels_for_open_plan_merge"] is True
    assert cfg["perception"]["yolo_world"]["reject_edge_touching_bboxes"] is False
    assert cfg["perception"]["yolo_world"]["mask_aware_partial_tracking"] is True
    assert cfg["object_memory"]["use_edge_touching_detections_for_policy"] is False
    assert cfg["sgnav"]["frontier_distance_weight"] == 0.2
    assert cfg["llm"]["enabled"] is True


def test_dump_sgnav_step_writes_contract_artifact(tmp_path):
    out = tmp_path / "step.json"

    assert dump_sgnav_step_main(["--output", str(out), "--episode-id", "episode-0"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert all(key in payload for key in SGNAV_STEP_DUMP_KEYS)
    assert payload["metadata"]["episode_id"] == "episode-0"
    assert payload["metadata"]["schema_only"] is True
    assert payload["contract_version"] == "sgnav_step_dump_v1"


def test_dump_sgnav_step_can_convert_runtime_debug_artifacts(tmp_path):
    graph_debug = tmp_path / "graph_step_000001.json"
    result_row = tmp_path / "results.jsonl"
    out = tmp_path / "step.json"
    graph_debug.write_text(
        json.dumps(
            {
                "objects": [{"id": "object:1", "cat": "chair"}],
                "groups": [{"id": "group:1", "members": ["object:1"]}],
                "rooms": [{"id": "room:kitchen", "caption": "kitchen"}],
                "edges": [{"src": "object:1", "dst": "room:kitchen", "rel": "in"}],
                "frontiers": [{"index": 0, "selected": True, "center": [4, 5], "total": 1.2}],
                "decision": {"mode": "frontier", "reason": "selected_new_frontier", "target_cells": [[4, 5]]},
                "score_debug": {
                    "mode": "paper_subgraph_interpolation",
                    "selected_frontier_id": "frontier_0",
                    "frontier_scores": [{"frontier_id": "frontier_0", "score": 1.2}],
                    "top_supporting_subgraphs": [
                        {"subgraph_id": "sg_object_1", "central_object_id": "object:1", "p_sub": 0.8}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    result_row.write_text(
        json.dumps(
            {
                "object_memory_goal_candidates": [{"node_id": 1, "category": "chair"}],
                "candidate_accepted": False,
                "stop_reason": "max_control_steps",
                "success": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert dump_sgnav_step_main(["--output", str(out), "--graph-debug-dump", str(graph_debug), "--result-row", str(result_row)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["metadata"]["schema_only"] is False
    assert payload["objects"][0]["cat"] == "chair"
    assert payload["subgraph_probabilities"][0]["p_sub"] == 0.8
    assert payload["selected_frontier"]["frontier_id"] == "frontier_0"
    assert payload["candidate_goals"][0]["category"] == "chair"
    assert payload["stop_state"]["stop_reason"] == "max_control_steps"


def test_check_assets_reports_required_missing_paths(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GROUNDING_DINO_CHECKPOINT", str(tmp_path / "missing-grounding-dino.pth"))

    status = check_assets_main(["--require-grounding-dino"])
    captured = capsys.readouterr()

    assert status == 2
    assert "MISSING grounding_dino_checkpoint" in captured.out


def test_check_assets_does_not_require_rose2_source_for_gap_closure_default(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ROSE2_SOURCE_ROOT", str(tmp_path / "missing-rose2"))

    status = check_assets_main(["--require-rose2-source"])
    captured = capsys.readouterr()

    assert status == 0
    assert "OPTIONAL_MISSING rose2_source_root" in captured.out


def test_asset_path_lookup_accepts_legacy_isaac_root_env(monkeypatch):
    monkeypatch.delenv("ISAAC_SIM_ROOT", raising=False)
    monkeypatch.setenv("ISAAC_ROOT", "/tmp/legacy-isaac-root")

    resolved = _path_from_env_or_config(["ISAAC_SIM_ROOT", "ISAAC_ROOT"], {}, "missing", "default")

    assert resolved == "/tmp/legacy-isaac-root"


def test_policy_and_allow_debug_fallback_flags_parse_before_asset_check(tmp_path, capsys):
    status = run_one_episode_main(
        [
            "--episode-file",
            str(tmp_path / "episodes.jsonl"),
            "--detector",
            "grounding_dino",
            "--segmenter",
            "sam2",
            "--sim-backend",
            "map",
            "--policy",
            "sgnav_original",
            "--allow-debug-fallbacks",
            "--grounding-dino-checkpoint",
            str(tmp_path / "missing-grounding-dino.pth"),
            "--grounding-dino-config",
            str(tmp_path / "missing-grounding-dino.py"),
        ]
    )
    captured = capsys.readouterr()

    assert status == 2
    assert "Missing GroundingDINO-B/Swin-B checkpoint" in captured.err
