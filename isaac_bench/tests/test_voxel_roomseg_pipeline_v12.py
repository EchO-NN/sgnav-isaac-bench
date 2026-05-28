from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, run_voxel_occupancy_door_wall_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_voxel_v12_pipeline_order_anchor_union_and_door_visual_cut_layers() -> None:
    shape = (30, 42)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=4.2, min_y=0.0, max_y=3.0, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.40, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.40),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.40
    grid.state[1:4, 4:26, 4:38] = int(VOXEL_FREE)

    grid.state[:, 4:12, 21] = int(VOXEL_OCCUPIED)
    grid.state[:, 19:26, 21] = int(VOXEL_OCCUPIED)
    z = grid.z_centers_m
    door_rows = np.arange(12, 19)
    lower_idx = np.nonzero((z >= 0.10) & (z < 1.80))[0]
    upper_idx = np.nonzero((z >= 1.80) & (z < 2.20))[0]
    grid.state[np.ix_(lower_idx, door_rows, np.asarray([21]))] = int(VOXEL_FREE)
    grid.state[np.ix_(upper_idx, door_rows, np.asarray([21]))] = int(VOXEL_OCCUPIED)

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_wall_projection": {"min_projected_line_length_m": 0.30, "anchor_min_projected_line_length_m": 0.15},
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

    assert result.debug["separator_report"]["stage_order"] == [
        "voxel_evidence",
        "voxel_door_seed",
        "wall_projection",
        "step1_real_wall_gap_fill",
        "voxel_door_completion",
        "step2_wall_line_extension",
        "final_4conn_labels",
    ]
    assert np.any(result.layers["voxel_door_seed_mask"][door_rows, 21])
    assert np.any(result.layers["voxel_door_anchor_wall_union_map"])
    assert np.any(result.layers["voxel_door_centerline_visual_mask"])
    assert np.any(result.layers["voxel_door_cut_mask"])
    assert int(np.count_nonzero(result.layers["voxel_door_centerline_visual_mask"])) >= int(np.count_nonzero(result.layers["voxel_door_cut_mask"]))
    assert int(result.debug["voxel_door_accepted_count"]) >= 1
