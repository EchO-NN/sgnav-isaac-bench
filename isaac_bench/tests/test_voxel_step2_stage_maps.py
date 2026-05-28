from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, run_voxel_occupancy_door_wall_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_step2_stage_maps_are_exposed_even_when_no_separator_is_accepted() -> None:
    shape = (26, 30)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=3.0, min_y=0.0, max_y=2.6, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.40, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.40),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.40
    grid.state[2:4, 4:22, 4:26] = int(VOXEL_FREE)
    grid.state[:, 12, 5:11] = int(VOXEL_OCCUPIED)
    grid.state[:, 12, 18:24] = int(VOXEL_OCCUPIED)

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {"voxel_door": {"enabled": False}, "voxel_step1": {"gap_fill_enabled": False}},
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

    for key in (
        "voxel_step2_extension_hits_all_map",
        "voxel_step2_extension_hits_pre_topology_map",
        "voxel_step2_separator_candidates_pre_topology_map",
        "voxel_step2_topology_rejected_separator_map",
        "voxel_step2_accepted_separator_map",
        "voxel_step2_extension_separator_map",
    ):
        assert key in result.layers
        assert result.layers[key].shape == shape
    assert "voxel_step2_extension_reject_reason_counts" in result.debug
    assert "voxel_step2_topology_reject_reason_counts" in result.debug
