import random

import numpy as np

from isaac_bench.dataset.episode_generator import EpisodeGenerationConfig, sample_start_pose
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.navigation.astar import GridAStarPlanner


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
