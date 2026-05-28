from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, run_voxel_occupancy_door_wall_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_real_wall_barrier_is_removed_from_partition_free_and_splits_rooms() -> None:
    shape = (24, 24)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=2.4, min_y=0.0, max_y=2.4, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.40, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.40),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.40
    grid.state[2:4, 4:20, 4:20] = int(VOXEL_FREE)
    grid.state[:, 12, :] = int(VOXEL_OCCUPIED)

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {"voxel_door": {"enabled": False}, "voxel_step1": {"gap_fill_enabled": False}, "voxel_step2": {"enabled": False}},
        resolution_m=0.10,
        map_info=info,
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

    assert np.any(result.layers["voxel_real_wall_barrier_map"][12, 4:20])
    assert not np.any(result.layers["voxel_base_partition_free"] & result.layers["voxel_real_wall_barrier_map"])
    assert int(result.room_label_map.max()) >= 2
