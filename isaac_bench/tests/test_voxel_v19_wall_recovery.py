from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    run_voxel_occupancy_door_wall_roomseg,
)
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_CONFLICT,
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)
from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_lines


def _grid(shape: tuple[int, int] = (32, 32), z_bins: int = 20) -> VoxelOccupancyGrid3D:
    info = MapInfo(
        resolution_m=0.10,
        min_x=0.0,
        max_x=float(shape[1]) * 0.10,
        min_y=0.0,
        max_y=float(shape[0]) * 0.10,
        width=shape[1],
        height=shape[0],
    )
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=float(z_bins) * 0.10,
            z_resolution_m=0.10,
            active_z_min_m=0.0,
            active_z_max_fallback_m=float(z_bins) * 0.10,
        ),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = float(z_bins) * 0.10
    return grid


def test_sparse_occupied_support_recovers_projected_display_wall() -> None:
    grid = _grid()
    shape = grid.shape
    grid.state[4:10, 12, 6:26] = int(VOXEL_OCCUPIED)
    grid.state[10:15, 12, 6:26] = int(VOXEL_CONFLICT)
    grid.state[1:4, 7:12, 6:26] = int(VOXEL_FREE)

    nav_free = np.zeros(shape, dtype=bool)
    nav_free[7:12, 6:26] = True
    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_door": {"enabled": False},
            "voxel_step1": {"gap_fill_enabled": False},
            "voxel_step2": {"enabled": False},
        },
        resolution_m=0.10,
        map_info=grid.map_info,
    )

    result = run_voxel_occupancy_door_wall_roomseg(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=nav_free,
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.zeros(shape, dtype=bool),
        voxel_grid=grid,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
    )

    strict_wall = np.asarray(result.layers["voxel_wall_xy"], dtype=bool)
    wall_support = np.asarray(result.layers["voxel_wall_line_support_xy"], dtype=bool)
    projected = np.asarray(result.layers["voxel_projected_structural_wall_map"], dtype=bool)
    display = np.asarray(result.layers["voxel_display_wall_xy"], dtype=bool)

    assert not np.any(strict_wall[12, 6:26])
    assert np.any(wall_support[12, 6:26])
    assert np.any(projected[12, 6:26])
    assert int(np.count_nonzero(display)) > int(np.count_nonzero(strict_wall))


def test_furniture_like_support_is_rejected_by_side_validation() -> None:
    shape = (24, 36)
    raw = np.zeros(shape, dtype=bool)
    raw[10, 6:30] = True
    free = np.zeros(shape, dtype=bool)
    free[7:14, 6:30] = True

    result = project_wall_evidence_to_lines(
        wall_raw=raw,
        free_map=free,
        occupied_ratio=raw.astype(np.float32),
        vertical_free_map=free,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.30),
    )

    assert np.any(result.raw_wall_map)
    assert not np.any(result.projected_wall_map)
    assert result.debug["voxel_wall_projection_side_reject_reason_counts"]["projected_wall_both_sides_free_furniture_like"] >= 1


def test_v19_parallel_recovered_walls_remain_separate() -> None:
    shape = (32, 40)
    raw = np.zeros(shape, dtype=bool)
    raw[10, 5:28] = True
    raw[15, 5:28] = True
    free = np.zeros(shape, dtype=bool)
    free[12:14, 5:28] = True

    result = project_wall_evidence_to_lines(
        wall_raw=raw,
        free_map=free,
        occupied_ratio=raw.astype(np.float32),
        vertical_free_map=free,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.05,
        config=WallProjectionConfig(min_projected_line_length_m=0.30),
    )

    lines = [line for line in result.projected_lines if line.reject_reason is None]
    assert len(lines) == 2
    assert sorted(line.line for line in lines) == [10, 15]
    assert not np.any(result.projected_wall_map[12:14, 5:28])
