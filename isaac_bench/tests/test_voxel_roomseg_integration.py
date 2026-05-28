from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    run_voxel_occupancy_door_wall_roomseg,
)
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid() -> VoxelOccupancyGrid3D:
    shape = (24, 24)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=2.4, min_y=0.0, max_y=2.4, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.0, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.0),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.0
    return grid


def _two_rooms_grid() -> VoxelOccupancyGrid3D:
    grid = _grid()
    grid.state[:5, 4:20, 4:20] = int(VOXEL_FREE)
    grid.state[:, :, 12] = int(VOXEL_OCCUPIED)
    return grid


def test_final_separator_includes_wall_door_step1_step2() -> None:
    grid = _two_rooms_grid()
    shape = grid.shape
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
        observed_free_mask=np.ones(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.zeros(shape, dtype=bool),
        voxel_grid=grid,
        navigation_free_mask=np.ones(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
    )

    expected = (
        np.asarray(result.layers["voxel_wall_after_step1_map"], dtype=bool)
        | np.asarray(result.layers["voxel_door_cut_mask"], dtype=bool)
        | np.asarray(result.layers["voxel_step2_extension_separator_map"], dtype=bool)
    )
    assert np.array_equal(result.separator_map, expected)


def test_no_small_room_merge_by_default_keeps_wall_split() -> None:
    grid = _two_rooms_grid()
    shape = grid.shape
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
        observed_free_mask=np.ones(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.zeros(shape, dtype=bool),
        voxel_grid=grid,
        navigation_free_mask=np.ones(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
    )

    labels = [int(v) for v in np.unique(result.room_label_map) if int(v) > 0]
    assert not cfg.merge_small_components_enabled
    assert len(labels) >= 2
