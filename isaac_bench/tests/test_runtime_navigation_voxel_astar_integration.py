from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED
from isaac_bench.navigation.astar import ClearanceAStarPlanner
from isaac_bench.scripts.run_one_episode import path_clearance_debug


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
    debug = mapper.navigation_debug_layers()
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
