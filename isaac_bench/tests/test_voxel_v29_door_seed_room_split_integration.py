from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    run_voxel_occupancy_door_wall_roomseg,
)
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid(shape: tuple[int, int]) -> VoxelOccupancyGrid3D:
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
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.30, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.30),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.30
    return grid


def _write_vertical_free(grid: VoxelOccupancyGrid3D, free_xy: np.ndarray) -> None:
    centers = grid.z_centers_m
    rows, cols = np.nonzero(np.asarray(free_xy, dtype=bool))
    for row, col in zip(rows, cols):
        grid.state[(centers >= 0.10) & (centers < 1.80), int(row), int(col)] = int(VOXEL_FREE)


def _write_door_seed_columns(grid: VoxelOccupancyGrid3D, cells: list[tuple[int, int]]) -> None:
    centers = grid.z_centers_m
    for row, col in cells:
        grid.state[(centers >= 0.10) & (centers < 1.80), int(row), int(col)] = int(VOXEL_FREE)
        grid.state[(centers >= 1.80) & (centers < 2.15), int(row), int(col)] = int(VOXEL_OCCUPIED)


def _v29_test_config(grid: VoxelOccupancyGrid3D) -> VoxelOccupancyDoorWallRoomSegConfig:
    return VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_step1": {"gap_fill_enabled": False},
            "voxel_wall_projection": {"enabled": False},
            "voxel_door": {
                "min_seed_cells_for_accepted_extension": 1,
                "min_seed_line_length_cells_for_accepted_extension": 1,
                "min_seed_elongation_for_direction": 1.0,
                "accepted_orientation_mode": "legacy",
                "local_free_neck_orientation_debug_only": False,
                "wall_anchor_radius_cells": 0,
                "seed_cluster_morph_close_radius_cells": 0,
                "partition_cut_bridge_unknown_max_cells": 0,
                "partition_cut_bridge_nonfree_max_cells": 0,
                "partition_topology_min_side_area_cells": 1,
                "strong_seed_centerline_min_elongation": 1.0,
            },
        },
        resolution_m=0.10,
        map_info=grid.map_info,
    )


def _run(grid: VoxelOccupancyGrid3D, free_xy: np.ndarray):
    shape = tuple(grid.shape)
    return run_voxel_occupancy_door_wall_roomseg(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=np.asarray(free_xy, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.zeros(shape, dtype=bool),
        voxel_grid=grid,
        navigation_free_mask=np.asarray(free_xy, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_v29_test_config(grid),
    )


def test_v29_raw_seed_cells_are_not_added_back_to_partition_free() -> None:
    shape = (20, 40)
    grid = _grid(shape)
    free_xy = np.zeros(shape, dtype=bool)
    free_xy[10, 5:33] = True
    seed_cells = [(10, col) for col in range(16, 20)]
    _write_vertical_free(grid, free_xy)
    _write_door_seed_columns(grid, seed_cells)

    result = _run(grid, free_xy)
    door_seed_mask = np.asarray(result.layers["voxel_door_seed_mask"], dtype=bool)
    final_cut = np.asarray(result.layers["voxel_door_final_cut_mask"], dtype=bool)
    partition_free = np.asarray(result.debug["partition_free_for_label"], dtype=bool)

    assert result.debug["voxel_door_seed_cells"] > 0
    assert result.debug["voxel_seed_not_added_to_partition_free"] is True
    assert result.debug["voxel_door_topology_effective_cells"] > 0
    assert result.debug["room_count"] >= 2
    assert not np.any(partition_free & door_seed_mask & ~final_cut)


def test_v29_visual_only_door_seed_does_not_block_step2() -> None:
    shape = (20, 30)
    grid = _grid(shape)
    free_xy = np.zeros(shape, dtype=bool)
    free_xy[7:14, 5:24] = True
    seed_cells = [(10, col) for col in range(13, 17)]
    _write_vertical_free(grid, free_xy)
    _write_door_seed_columns(grid, seed_cells)

    result = _run(grid, free_xy)
    visual_only = np.asarray(result.layers["voxel_door_visual_only_mask"], dtype=bool)
    step2_block = np.asarray(result.layers["voxel_step2_door_reject_mask"], dtype=bool)
    raw_seed = np.asarray(result.layers["voxel_door_seed_mask"], dtype=bool)

    assert result.debug["voxel_door_seed_cells"] > 0
    assert np.any(visual_only)
    assert result.debug["voxel_door_topology_effective_cells"] == 0
    assert not np.any(step2_block & visual_only)
    assert not np.any(step2_block & raw_seed)
    assert result.debug["voxel_step2_raw_seed_not_blocking_cells"] > 0
