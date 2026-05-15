import random
import json

import numpy as np

from isaac_bench.dataset.episode_generator import EpisodeGenerationConfig, generate_scene_episodes, sample_start_pose
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.scripts.generate_episodes import main as generate_episodes_main


def test_sample_start_pose_respects_object_clearance():
    navigable = np.ones((8, 8), dtype=bool)
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=8.0, min_y=0.0, max_y=8.0, width=8, height=8)
    planner = GridAStarPlanner(navigable, resolution_m=1.0, allow_diagonal=False)
    objects = [{"bbox_min_world": [2.0, 2.0, 0.0], "bbox_max_world": [6.0, 6.0, 1.0]}]
    cfg = EpisodeGenerationConfig(
        min_start_goal_distance_m=1.0,
        max_start_goal_distance_m=20.0,
        min_start_object_clearance_m=1.0,
        min_start_goal_bbox_distance_m=0.0,
        max_attempts_per_episode=10000,
    )

    sampled = sample_start_pose(
        navigable,
        [(0, 0)],
        planner,
        map_info,
        random.Random(3),
        cfg,
        all_objects=objects,
        goal_objects=[],
    )

    assert sampled is not None
    _pose, _grid, _shortest, object_clearance, _goal_clearance = sampled
    assert object_clearance >= 1.0


def _write_synthetic_scene(root):
    scene_dir = root / "kujiale_test"
    scene_dir.mkdir(parents=True)
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=12.0, min_y=0.0, max_y=12.0, width=12, height=12)
    navigable = np.ones((12, 12), dtype=bool)
    objects = [
        {
            "instance_id": "lamp_1",
            "category": "lamp",
            "center_world": [8.1, 8.1, 0.5],
            "bbox_min_world": [8.0, 8.0, 0.0],
            "bbox_max_world": [8.2, 8.2, 1.0],
        }
    ]
    (scene_dir / "map_info.json").write_text(json.dumps(map_info.to_dict()), encoding="utf-8")
    (scene_dir / "objects.json").write_text(json.dumps(objects), encoding="utf-8")
    (scene_dir / "objects_all.json").write_text(json.dumps(objects), encoding="utf-8")
    (scene_dir / "scene_metadata.json").write_text(
        json.dumps({"usd_path": "/tmp/scene.usd", "rooms_json_path": "/tmp/rooms.json"}),
        encoding="utf-8",
    )
    np.save(scene_dir / "navigable.npy", navigable)
    return scene_dir


def test_generate_scene_episodes_is_deterministic(tmp_path):
    scene_dir = _write_synthetic_scene(tmp_path)

    first = generate_scene_episodes(
        str(scene_dir),
        episodes_per_scene=3,
        seed=7,
        success_distance_m=1.5,
        min_start_goal_distance_m=2.0,
        min_start_object_clearance_m=0.0,
        min_start_goal_bbox_distance_m=0.0,
    )
    second = generate_scene_episodes(
        str(scene_dir),
        episodes_per_scene=3,
        seed=7,
        success_distance_m=1.5,
        min_start_goal_distance_m=2.0,
        min_start_object_clearance_m=0.0,
        min_start_goal_bbox_distance_m=0.0,
    )

    assert first == second
    assert len(first) == 3
    assert first[0]["version"] == "interioragent_objectnav_episode_v1"
    assert first[0]["goal_category"] == "lamp"
    assert first[0]["metadata"]["generator_seed"] == 7


def test_generate_episodes_cli_fails_clearly_for_missing_preprocessed_dir(tmp_path, capsys):
    status = generate_episodes_main(["--preprocessed-dir", str(tmp_path / "missing"), "--out", str(tmp_path / "out.jsonl")])
    captured = capsys.readouterr()

    assert status == 2
    assert "preprocessed directory not found" in captured.err
