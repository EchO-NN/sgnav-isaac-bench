from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from isaac_bench.config import get_nested, load_config, str_to_bool
from isaac_bench.dataset.episode_generator import read_jsonl
from isaac_bench.metrics.episode_logger import JsonlEpisodeLogger
from isaac_bench.scripts.run_one_episode import run_episode_map_sim


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--planner", default=None, choices=["astar", "nav2"])
    parser.add_argument("--detector", default=None, choices=["dry_run", "yolo_world", "none"])
    parser.add_argument("--headless", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sgnav-repo", default=None)
    parser.add_argument("--use-original-scenegraph", action="store_true", default=None)
    parser.add_argument("--sim-backend", default="map", choices=["map"])
    parser.add_argument("--yolo-world-model", default=None)
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--debug-map", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    args.planner = args.planner or get_nested(cfg, "repo.planner", "astar")
    args.detector = args.detector or get_nested(cfg, "repo.detector", "dry_run")
    args.headless = bool(get_nested(cfg, "isaac.headless", True) if args.headless is None else args.headless)
    args.output = args.output or str(Path(get_nested(cfg, "project.output_dir", "data/isaac_bench_runs")) / "astar_yoloworld" / "results.jsonl")
    args.sgnav_repo = args.sgnav_repo or get_nested(cfg, "paths.sgnav_repo", "/home/echo/SG-Nav")
    args.yolo_world_model = args.yolo_world_model or get_nested(cfg, "paths.yolo_world_model", get_nested(cfg, "perception.yolo_world_model", "data/models/yolov8s-worldv2.pt"))
    if args.use_original_scenegraph is None:
        args.use_original_scenegraph = bool(get_nested(cfg, "repo.use_original_scenegraph", False))

    if args.planner == "nav2":
        from isaac_bench.navigation.nav2_client import Nav2NavigateToPoseClient

        Nav2NavigateToPoseClient()
    out = Path(args.output)
    if out.exists():
        out.unlink()
    logger = JsonlEpisodeLogger(args.output)
    episodes = read_jsonl(args.episode_file)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    for idx, episode in enumerate(episodes):
        args.episode_index = idx
        row = run_episode_map_sim(episode, args)
        logger.log(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
