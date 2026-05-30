from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def _grid(shape: tuple[int, int] = (9, 9)) -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, width=shape[1], height=shape[0])
    return VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=1.0, z_resolution_m=0.10, active_z_min_m=0.0),
    )


def test_navigation_projection_occupied_voxel_wins_over_free_same_xy() -> None:
    grid = _grid()
    row, col = 4, 4
    grid.state[1, row, col] = int(VOXEL_FREE)
    grid.state[3, row, col] = int(VOXEL_OCCUPIED)
    grid.state[5, row, col] = int(VOXEL_FREE)

    nav = grid.project_navigation(obstacle_z_min_m=0.20, obstacle_z_max_m=0.90, free_z_min_m=0.10, free_z_max_m=0.90)

    assert nav.occupied[row, col]
    assert not nav.free[row, col]
    assert nav.debug["voxel_nav_free_suppressed_by_occupied_cells"] >= 1


def test_navigation_projection_endpoint_hysteresis_blocks_free_hole() -> None:
    grid = _grid()
    row, col = 4, 4
    grid.state[2, row, col] = int(VOXEL_FREE)
    endpoint = np.zeros(grid.shape, dtype=np.uint16)
    endpoint[row, col] = 2

    nav = grid.project_navigation(
        obstacle_z_min_m=0.20,
        obstacle_z_max_m=0.90,
        free_z_min_m=0.10,
        free_z_max_m=0.90,
        nav_endpoint_count_xy=endpoint,
    )

    assert nav.occupied[row, col]
    assert not nav.free[row, col]
    assert nav.debug["voxel_nav_occupied_from_endpoint_cells"] == 1


def test_navigation_projection_fills_tiny_occupied_hole() -> None:
    grid = _grid()
    for row in range(3, 6):
        for col in range(3, 6):
            if (row, col) != (4, 4):
                grid.state[3, row, col] = int(VOXEL_OCCUPIED)
    center = (4, 4)
    grid.state[2, center[0], center[1]] = int(VOXEL_FREE)

    nav = grid.project_navigation(
        obstacle_z_min_m=0.20,
        obstacle_z_max_m=0.90,
        free_z_min_m=0.10,
        free_z_max_m=0.90,
        occupied_close_radius_cells=0,
        occupied_fill_small_holes_max_area_cells=4,
    )

    assert nav.occupied[center]
    assert int(nav.debug["voxel_nav_occupied_hole_filled_cells"]) >= 1


def test_navigation_projection_keeps_unobserved_unknown() -> None:
    grid = _grid()

    nav = grid.project_navigation(obstacle_z_min_m=0.20, obstacle_z_max_m=0.90, free_z_min_m=0.10, free_z_max_m=0.90)

    assert nav.unknown[4, 4]
    assert not nav.free[4, 4]
    assert not nav.occupied[4, 4]
