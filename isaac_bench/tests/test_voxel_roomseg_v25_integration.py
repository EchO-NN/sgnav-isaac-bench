from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    run_voxel_occupancy_door_wall_roomseg,
)
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def test_v25_sensor_aware_door_seed_and_raw_seed_free_step2_debug() -> None:
    shape = (36, 40)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=4.0, min_y=0.0, max_y=3.6, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.60, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.60),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.60
    grid.state[1:4, 4:32, 4:36] = int(VOXEL_FREE)

    grid.state[:, 0:17, 20] = int(VOXEL_OCCUPIED)
    grid.state[:, 22:36, 20] = int(VOXEL_OCCUPIED)
    z = grid.z_centers_m
    door_rows = np.arange(17, 22)
    lower_idx = np.nonzero((z >= 0.10) & (z < 1.70))[0]
    upper_actual = np.nonzero((z >= 1.70) & (z < 1.80))[0]
    upper_unknown = np.nonzero((z >= 1.80) & (z < 2.10))[0]
    grid.state[np.ix_(lower_idx, door_rows, np.asarray([20]))] = int(VOXEL_FREE)
    grid.state[np.ix_(upper_actual, door_rows, np.asarray([20]))] = int(VOXEL_OCCUPIED)
    grid.state[np.ix_(upper_unknown, door_rows, np.asarray([20]))] = int(VOXEL_UNKNOWN)
    grid.sensor_range_count[np.ix_(upper_unknown, door_rows, np.asarray([20]))] = 1

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_wall_projection": {"min_projected_line_length_m": 0.30, "side_validation_enabled": False},
            "voxel_step1": {"gap_fill_enabled": False},
        },
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

    assert result.debug["voxel_door_seed_cells"] > 0
    assert result.debug["voxel_door_partition_accepted_count"] > 0
    assert result.debug["voxel_step2_blocked_by_raw_seed_count"] == 0
    assert result.debug["voxel_room_count"] >= 2
    assert np.any(result.layers["voxel_door_seed_mask"][door_rows, 20])
