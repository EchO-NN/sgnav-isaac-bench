from __future__ import annotations

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid(backend: str = "cpu_vectorized", *, python_allowed: bool = False) -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, width=20, height=20)
    return VoxelOccupancyGrid3D.zeros(
        (20, 20),
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=2.0,
            z_resolution_m=0.10,
            integration_backend=backend,
            python_debug_backend_allowed=python_allowed,
        ),
    )


def test_cpu_vectorized_backend_integrates_without_python_debug() -> None:
    grid = _grid("cpu_vectorized")
    origin = np.asarray([-0.5, 0.0, 1.0], dtype=np.float32)
    endpoints = np.asarray([[0.5, 0.0, 1.0], [0.4, 0.1, 0.8]], dtype=np.float32)

    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoints, floor_z=0.0)

    assert stats.integration_backend == "cpu_vectorized"
    assert not stats.python_debug_backend_used
    assert stats.depth_rays_integrated == 2
    assert stats.occupied_update_count == 2
    assert int(np.count_nonzero(grid.state == int(VOXEL_OCCUPIED))) >= 1


def test_python_debug_backend_is_blocked_unless_explicitly_allowed() -> None:
    grid = _grid("python_debug", python_allowed=False)
    origin = np.asarray([-0.5, 0.0, 1.0], dtype=np.float32)
    endpoint = np.asarray([[0.5, 0.0, 1.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="debug-only"):
        grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoint, floor_z=0.0)
