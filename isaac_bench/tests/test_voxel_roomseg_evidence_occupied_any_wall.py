from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig
from isaac_bench.mapping.voxel_roomseg_evidence import build_voxel_roomseg_evidence, voxel_column_debug


def _grid(z_bins: int = 40, shape: tuple[int, int] = (4, 4), z_res: float = 0.05) -> VoxelOccupancyGrid3D:
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
            z_max_m=float(z_bins) * z_res,
            z_resolution_m=z_res,
            active_z_min_m=0.0,
            active_z_max_fallback_m=float(z_bins) * z_res,
        ),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = float(z_bins) * z_res
    return grid


def _evidence(grid: VoxelOccupancyGrid3D, extra_config: dict | None = None):
    shape = grid.shape
    config = {"active_z_min_m": 0.0}
    config.update(extra_config or {})
    return build_voxel_roomseg_evidence(
        voxel_grid=grid,
        navigation_free_mask=np.zeros(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask_from_navigation=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=config,
    )


def test_v18_single_occupied_bin_is_not_wall() -> None:
    grid = _grid()
    grid.state[11, 1, 1] = int(VOXEL_OCCUPIED)

    evidence = _evidence(grid)

    assert evidence.occupied_any_xy[1, 1]
    assert evidence.raw_occupied_wall_support_xy[1, 1]
    assert evidence.unknown_dominant_xy[1, 1]
    assert evidence.nonstructural_occupied_xy[1, 1]
    assert not evidence.structural_wall_ratio_xy[1, 1]
    assert not evidence.strict_raw_wall_xy[1, 1]
    assert not evidence.wall_xy[1, 1]
    assert not evidence.vertical_free_xy[1, 1]
    assert evidence.unknown_xy[1, 1]
    assert not evidence.ratio_wall_debug_xy[1, 1]
    assert evidence.debug["voxel_wall_ratio_used_for_final"] is True


def test_v18_free_wins_over_ratio_wall_conflict() -> None:
    grid = _grid(z_bins=20)
    grid.state[:, 1, 1] = int(VOXEL_OCCUPIED)
    grid.state[0:3, 1, 1] = int(VOXEL_FREE)

    evidence = _evidence(grid, {"wall_occupied_ratio_min_for_xy_wall": 0.80})

    assert evidence.vertical_free_xy[1, 1]
    assert evidence.occupied_any_xy[1, 1]
    assert evidence.raw_occupied_wall_support_xy[1, 1]
    assert evidence.structural_wall_ratio_xy[1, 1]
    assert evidence.wall_suppressed_by_free_xy[1, 1]
    assert evidence.wall_rejected_by_free_xy[1, 1]
    assert not evidence.strict_raw_wall_xy[1, 1]
    assert not evidence.wall_xy[1, 1]
    assert not evidence.unknown_xy[1, 1]


def test_vertical_free_requires_three_free_z_cells() -> None:
    grid = _grid()
    grid.state[8:9, 1, 1] = int(VOXEL_FREE)
    assert not _evidence(grid).vertical_free_xy[1, 1]

    grid.state[9:10, 1, 1] = int(VOXEL_FREE)
    assert not _evidence(grid).vertical_free_xy[1, 1]

    grid.state[10:11, 1, 1] = int(VOXEL_FREE)
    evidence = _evidence(grid)

    assert evidence.vertical_free_xy[1, 1]


def test_v18_ratio_wall_when_not_free_not_unknown() -> None:
    grid = _grid(z_bins=20)
    grid.state[0:18, 1, 1] = int(VOXEL_OCCUPIED)

    evidence = _evidence(grid)

    assert evidence.structural_wall_ratio_xy[1, 1]
    assert evidence.wall_xy[1, 1]
    assert evidence.strict_raw_wall_xy[1, 1]
    assert not evidence.vertical_free_xy[1, 1]
    assert not evidence.unknown_xy[1, 1]


def test_voxel_column_debug_reports_v18_ratio_wall_policy() -> None:
    grid = _grid()
    grid.state[11, 1, 1] = int(VOXEL_OCCUPIED)

    debug = voxel_column_debug(grid, 1, 1, config={"active_z_min_m": 0.0})

    assert debug["occupied_count_active"] == 1
    assert debug["occupied_any"] is True
    assert debug["ratio_wall_debug"] is False
    assert debug["wall_raw"] is False
    assert debug["wall_ratio_raw"] is False
    assert debug["unknown_dominant"] is True
    assert debug["final_wall"] is False
    assert debug["final_unknown"] is True
    assert debug["wall_ratio_used_for_final"] is True
