from __future__ import annotations

import time

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, run_voxel_occupancy_door_wall_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_cpu_vectorized_voxel_integration_uses_changed_refresh_and_stays_fast() -> None:
    shape = (256, 256)
    info = MapInfo(resolution_m=0.05, min_x=-6.4, max_x=6.4, min_y=-6.4, max_y=6.4, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=2.4,
            z_resolution_m=0.10,
            integration_backend="cpu_vectorized",
            cuda_chunk_rays=2048,
            cuda_ray_step_voxels=1.0,
            cuda_max_samples_per_ray=160,
            python_debug_backend_allowed=False,
        ),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.4
    xs = np.linspace(-2.8, 2.8, 64, dtype=np.float32)
    ys = np.linspace(0.6, 4.6, 64, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, 0.8, dtype=np.float32)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).astype(np.float32)
    origin = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    elapsed = []
    stats = None
    for _ in range(20):
        started = time.perf_counter()
        stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)
        elapsed.append((time.perf_counter() - started) * 1000.0)

    assert stats is not None
    assert stats.integration_backend in {"cpu_vectorized", "cpu_numba", "cuda_persistent"}
    assert not stats.python_debug_backend_used
    assert stats.refresh_mode in {"changed_indices", "full"}
    assert stats.refresh_changed_voxels > 0
    assert float(np.mean(elapsed)) < 80.0

    projection = grid.project_navigation()
    result = run_voxel_occupancy_door_wall_roomseg(
        occupancy_map=projection.occupied,
        observed_free_mask=projection.free,
        obstacle_mask=projection.occupied,
        unknown_mask=projection.unknown,
        voxel_grid=grid,
        navigation_free_mask=projection.free,
        navigation_obstacle_mask=projection.occupied,
        resolution_m=0.05,
        config=VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
            {"voxel_wall_projection": {"min_projected_line_length_m": 0.20}},
            resolution_m=0.05,
            map_info=info,
        ),
    )

    debug = result.debug
    for key in (
        "voxel_integrate_total_ms",
        "voxel_integrate_ray_sample_ms",
        "voxel_integrate_unique_ms",
        "voxel_integrate_scatter_ms",
        "voxel_refresh_state_ms",
        "voxel_project_navigation_ms",
        "voxel_project_roomseg_ms",
        "voxel_door_seed_ms",
        "voxel_wall_projection_ms",
    ):
        assert key in debug
