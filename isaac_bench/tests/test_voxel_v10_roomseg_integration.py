from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, run_voxel_occupancy_door_wall_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_voxel_v10_roomseg_uses_projected_wall_and_centroid_door_seed() -> None:
    shape = (36, 40)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=4.0, min_y=0.0, max_y=3.6, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.40, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.40),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.40
    grid.state[1:4, 4:32, 4:36] = int(VOXEL_FREE)

    # Dividing wall with a vertical door opening.
    grid.state[:, 0:17, 20] = int(VOXEL_OCCUPIED)
    grid.state[:, 22:36, 20] = int(VOXEL_OCCUPIED)
    z = grid.z_centers_m
    door_rows = np.arange(17, 22)
    lower_idx = np.nonzero((z >= 0.10) & (z < 1.80))[0]
    upper_idx = np.nonzero((z >= 1.80) & (z < 2.20))[0]
    grid.state[np.ix_(lower_idx, door_rows, np.asarray([20]))] = int(VOXEL_FREE)
    grid.state[np.ix_(upper_idx, door_rows, np.asarray([20]))] = int(VOXEL_OCCUPIED)

    # Low furniture should be vertical-free, not a room wall.
    grid.state[:7, 10, 10] = int(VOXEL_OCCUPIED)
    grid.state[10:12, 10, 10] = int(VOXEL_FREE)

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_roomseg_evidence": {"wall_unknown_ratio_max_for_structural": 0.50},
            "voxel_wall_projection": {"min_projected_line_length_m": 0.30, "side_validation_enabled": False},
            "voxel_step1": {"gap_fill_enabled": False},
            "voxel_step2": {"enabled": False},
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

    assert result.layers["voxel_vertical_free_xy"][10, 10]
    assert result.layers["voxel_raw_occupied_wall_support_xy"][10, 10]
    assert not result.layers["voxel_strict_raw_wall_xy"][10, 10]
    assert not result.layers["voxel_wall_xy"][10, 10]
    assert np.any(result.layers["voxel_wall_projected_xy"][5:17, 20])
    assert np.any(result.layers["voxel_door_seed_mask"][door_rows, 20])
    assert np.any(result.layers["voxel_door_cut_mask"])
    assert result.debug["voxel_door_seed_method"] == "centroid_ratio"
    assert int(result.room_label_map.max()) >= 2


def test_voxel_v10_door_detection_anchors_to_raw_wall_not_projected_wall() -> None:
    shape = (24, 60)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=6.0, min_y=0.0, max_y=2.4, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.40, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.40),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.40
    grid.state[1:4, :, :] = int(VOXEL_FREE)

    z = grid.z_centers_m
    lower_idx = np.nonzero((z >= 0.10) & (z < 1.80))[0]
    upper_idx = np.nonzero((z >= 1.80) & (z < 2.20))[0]
    seed_cols = np.arange(20, 51)
    grid.state[:, 10, [10, 55]] = int(VOXEL_OCCUPIED)
    grid.state[np.ix_(lower_idx, np.asarray([10]), seed_cols)] = int(VOXEL_FREE)
    grid.state[np.ix_(upper_idx, np.asarray([10]), seed_cols)] = int(VOXEL_OCCUPIED)

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_wall_projection": {"min_projected_line_length_m": 5.00},
            "voxel_door": {
                "door_width_max_m": 0.60,
                "visual_width_max_m": 5.00,
                "extend_max_m": 5.00,
                "seed_cluster_max_width_m": 5.00,
                "partition_topology_enabled": False,
            },
            "voxel_step1": {"gap_fill_enabled": False},
            "voxel_step2": {"enabled": False},
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

    assert int(np.count_nonzero(result.layers["voxel_wall_projected_xy"])) == 0
    assert int(np.count_nonzero(result.layers["voxel_door_anchor_wall_xy"])) >= 2
    assert int(result.debug["voxel_door_accepted_count"]) == 1
    assert np.any(result.layers["voxel_door_cut_mask"][10, 10:56])
