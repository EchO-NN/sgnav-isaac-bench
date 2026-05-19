from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from isaac_bench.config import get_nested, load_config, str_to_bool
from isaac_bench.dataset.episode_generator import read_jsonl
from isaac_bench.metrics.result_schema import BenchmarkAssetError, validate_strict_benchmark_assets
from isaac_bench.scripts.run_one_episode import main as run_one_episode_main


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--planner", default=None, choices=["astar", "nav2"])
    parser.add_argument("--detector", default=None, choices=["dry_run", "yolo_world", "grounding_dino", "none"])
    parser.add_argument("--headless", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict-benchmark", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-strict-benchmark", dest="strict_benchmark", action="store_false")
    parser.add_argument("--allow-debug-fallbacks", action="store_true", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--ablation-name", default=None)
    parser.add_argument("--sgnav-repo", default=None)
    parser.add_argument("--use-original-scenegraph", action="store_true", default=None)
    parser.add_argument("--sim-backend", default=None, choices=["map", "isaac"])
    parser.add_argument("--yolo-world-model", default=None)
    parser.add_argument("--grounding-dino-checkpoint", default=None)
    parser.add_argument("--grounding-dino-config", default=None)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=None)
    parser.add_argument("--grounding-dino-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--segmenter", default=None, choices=["none", "auto", "sam2"])
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--debug-map", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    args.planner = args.planner or get_nested(cfg, "repo.planner", "astar")
    args.detector = args.detector or get_nested(cfg, "repo.detector", "dry_run")
    args.sim_backend = args.sim_backend or get_nested(cfg, "repo.backend", "map")
    args.strict_benchmark = bool(
        args.strict_benchmark
        if args.strict_benchmark is not None
        else get_nested(cfg, "benchmark.strict_benchmark", True)
    )
    args.allow_debug_fallbacks = bool(
        args.allow_debug_fallbacks
        if args.allow_debug_fallbacks is not None
        else get_nested(cfg, "benchmark.allow_debug_fallbacks", False)
    )
    args.policy = args.policy or get_nested(cfg, "benchmark.policy", None)
    args.ablation_name = args.ablation_name or get_nested(cfg, "benchmark.ablation_name", None)
    args.headless = bool(get_nested(cfg, "isaac.headless", True) if args.headless is None else args.headless)
    if args.output_dir and not args.output:
        args.output = str(Path(args.output_dir) / "results.jsonl")
    args.output = args.output or str(Path(get_nested(cfg, "project.output_dir", "data/isaac_bench_runs")) / "astar_grounding_dino" / "results.jsonl")
    args.sgnav_repo = args.sgnav_repo or get_nested(cfg, "paths.sgnav_repo", "/home/echo/SG-Nav")
    args.yolo_world_model = args.yolo_world_model or get_nested(cfg, "paths.yolo_world_model", get_nested(cfg, "perception.yolo_world_model", "data/models/yolov8l-worldv2.pt"))
    args.grounding_dino_checkpoint = args.grounding_dino_checkpoint or os.environ.get("GROUNDING_DINO_CHECKPOINT") or get_nested(cfg, "paths.grounding_dino_checkpoint", get_nested(cfg, "perception.grounding_dino.checkpoint", "data/models/groundingdino_swinb_cogcoor.pth"))
    args.grounding_dino_config = args.grounding_dino_config or os.environ.get("GROUNDING_DINO_CONFIG") or get_nested(cfg, "paths.grounding_dino_config", get_nested(cfg, "perception.grounding_dino.config", ""))
    args.grounding_dino_text_threshold = float(args.grounding_dino_text_threshold if args.grounding_dino_text_threshold is not None else get_nested(cfg, "perception.grounding_dino.text_threshold", 0.25))
    args.grounding_dino_device = str(args.grounding_dino_device or get_nested(cfg, "perception.grounding_dino.device", "cuda"))
    args.segmenter = args.segmenter or get_nested(cfg, "perception.segmenter", "none")
    args.sam2_checkpoint = args.sam2_checkpoint or get_nested(cfg, "perception.sam2_checkpoint", "")
    args.sgnav_mode = str(get_nested(cfg, "sgnav.mode", "legacy"))
    args.room_map_mode = str(get_nested(cfg, "mapping.room_map_mode", "rose2_source_form_v2_vlm"))
    args.room_segmentation_config = dict(get_nested(cfg, "mapping.room_segmentation", {}) or {})
    args.llm_enabled = bool(get_nested(cfg, "llm.enabled", True))
    args.vllm_frontier_scoring = bool(get_nested(cfg, "vllm.frontier_scoring", False))
    args.seed_gt_object_memory = bool(get_nested(cfg, "sgnav.seed_gt_object_memory", False))
    args.allow_gt_goal_fallback = bool(get_nested(cfg, "sgnav.allow_gt_goal_fallback", False))
    args.static_nearfield_map = bool(get_nested(cfg, "nearfield_static_map.enabled", False))
    args.frontier_allow_near_fallback = bool(get_nested(cfg, "mapping.frontier_allow_near_fallback", False))
    if args.use_original_scenegraph is None:
        args.use_original_scenegraph = bool(get_nested(cfg, "repo.use_original_scenegraph", False))
    try:
        validate_strict_benchmark_assets(args)
    except BenchmarkAssetError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 2

    if args.planner == "nav2":
        from isaac_bench.navigation.nav2_client import Nav2NavigateToPoseClient

        Nav2NavigateToPoseClient()
    out = Path(args.output)
    if out.exists():
        out.unlink()
    episodes = read_jsonl(args.episode_file)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    for idx, episode in enumerate(episodes):
        _ = episode
        child_args = [
            "--config",
            args.config,
            "--episode-file",
            args.episode_file,
            "--episode-index",
            str(idx),
            "--planner",
            args.planner,
            "--detector",
            args.detector,
            "--sim-backend",
            args.sim_backend,
            "--output",
            args.output,
            "--yolo-world-model",
            args.yolo_world_model,
            "--grounding-dino-checkpoint",
            args.grounding_dino_checkpoint,
            "--grounding-dino-config",
            args.grounding_dino_config,
            "--grounding-dino-text-threshold",
            str(args.grounding_dino_text_threshold),
            "--grounding-dino-device",
            args.grounding_dino_device,
            "--segmenter",
            args.segmenter,
            "--sam2-checkpoint",
            args.sam2_checkpoint,
            "--policy",
            str(args.policy or ""),
        ]
        child_args.append("--strict-benchmark" if args.strict_benchmark else "--no-strict-benchmark")
        if args.allow_debug_fallbacks:
            child_args.append("--allow-debug-fallbacks")
        if args.ablation_name:
            child_args.extend(["--ablation-name", str(args.ablation_name)])
        if args.headless is not None:
            child_args.extend(["--headless", "true" if args.headless else "false"])
        if args.debug_map:
            child_args.extend(["--debug-map", args.debug_map])
        if args.save_debug_video:
            child_args.append("--save-debug-video")
        status = run_one_episode_main(child_args)
        if status != 0:
            return int(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
