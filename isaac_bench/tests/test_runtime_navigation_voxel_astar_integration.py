from __future__ import annotations

import math

import numpy as np

from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED
from isaac_bench.metrics.evaluator import EpisodeEvaluator
from isaac_bench.navigation.astar import ClearanceAStarPlanner, GridAStarPlanner
from isaac_bench.scripts.run_one_episode import (
    collision_checked_path_world,
    effective_lookahead_min_clearance_m,
    path_clearance_debug,
    update_metric_evaluator_pose_or_mark_invalid,
)


def test_runtime_navigation_debug_contains_projection_override_and_astar_clearance() -> None:
    mapper = OnlineMapper(size_m=2.0, resolution_m=0.10)
    mapper.reset((0.0, 0.0))
    row, col = mapper.grid.world_to_grid(0.0, 0.0)
    mapper.voxel_grid.state[2, row, col] = int(VOXEL_FREE)
    mapper.voxel_nav_occupied_endpoint_count[row, col] = 2
    projection = mapper.voxel_grid.project_navigation(
        **mapper.voxel_navigation_projection_config,
        nav_endpoint_count_xy=mapper.voxel_nav_occupied_endpoint_count,
    )
    mapper.grid.free[:, :] = projection.free.astype(np.uint8)
    mapper.grid.occupied[:, :] = projection.occupied.astype(np.uint8)
    mapper.grid.observed[:, :] = projection.observed.astype(np.uint8)

    assert projection.occupied[row, col]
    assert not projection.free[row, col]

    mapper.grid.occupied[row, col] = 1
    mapper.roomseg_static_structural_occupied[row, col] = 0
    mapper._apply_voxel_navigation_blind_zone((0.0, 0.0, 0.0, 0.0), write_to_voxel=False, write_to_grid=True)
    nav = mapper.traversible(unknown_is_obstacle=True)
    planner = ClearanceAStarPlanner(nav, resolution_m=0.10, allow_diagonal=True)
    path_debug = path_clearance_debug([(row, col)], planner.clearance_m, robot_radius_m=0.10)
    debug = mapper.navigation_debug_layers(include_arrays=True)
    debug.update(
        {
            "astar_clearance_m": planner.clearance_m,
            **path_debug,
        }
    )

    assert "voxel_nav_final_occupied_xy" in debug
    assert "current_pose_navigation_override_mask" in debug
    assert "astar_clearance_m" in debug
    assert "astar_path_min_clearance_m" in debug


def test_lookahead_clearance_tracks_small_robot_astar_policy() -> None:
    configured = 0.14
    effective = effective_lookahead_min_clearance_m(
        configured,
        robot_radius_m=0.05,
        runtime_planning_clearance_m=0.02,
        astar_clearance_hard_min_m=0.0,
    )
    assert abs(effective - 0.14) < 1e-6

    map_info = MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.25, min_y=0.0, max_y=0.25, width=5, height=5)
    traversible = np.ones((5, 5), dtype=bool)
    clearance = np.full((5, 5), 0.10, dtype=np.float32)
    start_xy = grid_to_world_xy(2, 1, map_info)
    pose = (start_xy[0], start_xy[1], 0.05, 0.0)
    path_cells = [(2, 1), (2, 2), (2, 3)]

    assert not collision_checked_path_world(
        pose,
        path_cells,
        map_info,
        traversible,
        clearance,
        lookahead_m=0.05,
        min_clearance_m=configured,
        max_skip_cells=8,
    )
    assert not collision_checked_path_world(
        pose,
        path_cells,
        map_info,
        traversible,
        clearance,
        lookahead_m=0.05,
        min_clearance_m=effective,
        max_skip_cells=8,
    )


def test_static_metric_off_map_marks_invalid_without_stopping_evaluator_accounting() -> None:
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=3.0, min_y=0.0, max_y=3.0, width=3, height=3)
    traversible = np.zeros((3, 3), dtype=bool)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    episode = {
        "goal_regions_grid": [(1, 1)],
        "success_distance_m": 0.2,
        "shortest_path_distance_m": 1.0,
    }
    evaluator = EpisodeEvaluator(episode, planner)

    valid, metric_grid = update_metric_evaluator_pose_or_mark_invalid(
        evaluator,
        planner,
        (1.5, 1.5, 0.05, 0.0),
        map_info,
        collided=True,
    )

    assert valid is False
    assert metric_grid is None
    assert evaluator.num_steps == 1
    assert evaluator.num_collisions == 1
    assert math.isinf(evaluator.final_distance_to_goal)
