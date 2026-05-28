from __future__ import annotations

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig
from isaac_bench.mapping.voxel_roomseg_evidence import build_voxel_roomseg_evidence, voxel_column_debug


def _grid(z_bins: int = 10) -> VoxelOccupancyGrid3D:
    shape = (3, 3)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=0.3, min_y=0.0, max_y=0.3, width=3, height=3)
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=z_bins * 0.10, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=z_bins * 0.10),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = z_bins * 0.10
    return grid


def _evidence(grid: VoxelOccupancyGrid3D, extra_config: dict | None = None):
    shape = grid.shape
    config = {"active_z_min_m": 0.0, "min_free_z_cells_for_xy_free": 3, "wall_occupied_ratio_min_for_xy_wall": 0.90}
    config.update(extra_config or {})
    return build_voxel_roomseg_evidence(
        voxel_grid=grid,
        navigation_free_mask=np.zeros(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask_from_navigation=np.ones(shape, dtype=bool),
        resolution_m=0.10,
        config=config,
    )


def test_wall_and_free_overlap_keeps_vertical_free_after_v18_policy() -> None:
    grid = _grid(z_bins=20)
    grid.state[:, 1, 1] = int(VOXEL_OCCUPIED)
    grid.state[17:20, 1, 1] = int(VOXEL_FREE)

    evidence = _evidence(grid, {"wall_occupied_ratio_min_for_xy_wall": 0.80})

    assert evidence.vertical_free_xy[1, 1]
    assert evidence.occupied_any_xy[1, 1]
    assert evidence.raw_occupied_wall_support_xy[1, 1]
    assert not evidence.wall_xy[1, 1]
    assert evidence.wall_rejected_by_free_xy[1, 1]
    assert evidence.debug["voxel_free_wall_conflict_xy"][1, 1]


def test_single_occupied_is_not_wall_when_not_free() -> None:
    grid = _grid()
    grid.state[4, 1, 1] = int(VOXEL_OCCUPIED)

    evidence = _evidence(grid)

    assert not evidence.wall_xy[1, 1]
    assert not evidence.vertical_free_xy[1, 1]
    assert not evidence.ratio_wall_debug_xy[1, 1]
    assert evidence.unknown_xy[1, 1]


def test_unknown_preserved_when_not_free_not_wall() -> None:
    grid = _grid()
    grid.state[0, 1, 1] = int(VOXEL_FREE)

    evidence = _evidence(grid)

    assert evidence.unknown_xy[1, 1]
    assert not evidence.wall_xy[1, 1]
    assert not evidence.vertical_free_xy[1, 1]


def test_voxel_column_debug_explains_final_classification() -> None:
    grid = _grid()
    grid.state[4, 1, 1] = int(VOXEL_OCCUPIED)

    debug = voxel_column_debug(grid, 1, 1, config={"active_z_min_m": 0.0, "wall_occupied_ratio_min_for_xy_wall": 0.90})

    assert debug["occupied_count_active"] == 1
    assert debug["occupied_ratio_active"] == pytest.approx(0.1)
    assert debug["occupied_any"] is True
    assert debug["ratio_wall_debug"] is False
    assert debug["wall_raw"] is False
    assert debug["final_wall"] is False
    assert debug["final_unknown"] is True
    assert debug["wall_ratio_used_for_final"] is True


def test_voxel_column_debug_reports_free_priority_for_wall_overlap() -> None:
    grid = _grid()
    grid.state[:, 1, 1] = int(VOXEL_OCCUPIED)
    grid.state[7:10, 1, 1] = int(VOXEL_FREE)

    debug = voxel_column_debug(grid, 1, 1, config={"active_z_min_m": 0.0, "wall_occupied_ratio_min_for_xy_wall": 0.60})

    assert debug["free_raw"] is True
    assert debug["wall_raw"] is True
    assert debug["wall_rejected_by_free"] is True
    assert debug["final_wall"] is False
    assert debug["final_vertical_free"] is True
    assert debug["projection_priority"] == "free_unknown_then_ratio_wall"
