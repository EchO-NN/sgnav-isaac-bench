from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def _grid() -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, width=20, height=20)
    return VoxelOccupancyGrid3D.zeros(
        (20, 20),
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.0, z_resolution_m=0.10),
    )


def test_3d_ray_marks_free_before_endpoint_and_occupied_endpoint() -> None:
    grid = _grid()
    origin = np.asarray([-0.5, 0.0, 1.0], dtype=np.float32)
    endpoint = np.asarray([0.5, 0.0, 1.0], dtype=np.float32)
    voxels = grid.ray_voxels_3d(origin, endpoint, floor_z=0.0)

    grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoint.reshape(1, 3), floor_z=0.0)

    assert len(voxels) > 3
    assert int(grid.state[voxels[-1]]) == int(VOXEL_OCCUPIED)
    for voxel in voxels[:-1]:
        assert int(grid.state[voxel]) == int(VOXEL_FREE)
    z, r, c = voxels[-1]
    if c + 1 < grid.state.shape[2]:
        assert int(grid.state[z, r, c + 1]) == int(VOXEL_UNKNOWN)


def test_occupied_endpoint_not_erased_by_single_free_ray() -> None:
    grid = _grid()
    voxel = (10, 10, 10)
    grid.mark_occupied_voxels([voxel])
    grid.refresh_state()
    assert int(grid.state[voxel]) == int(VOXEL_OCCUPIED)

    grid.mark_free_voxels([voxel])
    grid.refresh_state()
    assert int(grid.state[voxel]) == int(VOXEL_OCCUPIED)


def test_repeated_free_can_clear_old_occupied_if_needed() -> None:
    grid = _grid()
    voxel = (10, 10, 10)
    grid.mark_occupied_voxels([voxel])
    for _ in range(5):
        grid.mark_free_voxels([voxel])
    grid.refresh_state()

    assert int(grid.state[voxel]) in {int(VOXEL_FREE), int(VOXEL_UNKNOWN)}


def test_navigation_projection_from_voxel_bands() -> None:
    grid = _grid()
    free_voxel = (2, 8, 8)
    occ_voxel = (5, 8, 9)
    grid.mark_free_voxels([free_voxel])
    grid.mark_occupied_voxels([occ_voxel])
    grid.refresh_state()

    projection = grid.project_navigation(
        obstacle_z_min_m=0.20,
        obstacle_z_max_m=0.90,
        free_z_min_m=0.10,
        free_z_max_m=0.90,
        min_free_voxels=1,
    )

    assert projection.free[8, 8]
    assert projection.occupied[8, 9]
    assert projection.observed[8, 8]
    assert projection.observed[8, 9]
