from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig
from isaac_bench.mapping.voxel_roomseg_evidence import build_voxel_roomseg_evidence


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
        unknown_mask_from_navigation=np.ones(shape, dtype=bool),
        resolution_m=0.10,
        config=config,
    )


def test_v18_unknown_wins_over_occupied_when_unknown_dominant() -> None:
    grid = _grid(z_bins=20)
    grid.state[0:8, 1, 1] = int(VOXEL_OCCUPIED)

    evidence = _evidence(
        grid,
        {
            "wall_occupied_ratio_min_for_xy_wall": 0.30,
            "unknown_ratio_min_for_xy_unknown": 0.50,
        },
    )

    assert evidence.occupied_any_xy[1, 1]
    assert evidence.wall_support_loose_xy[1, 1]
    assert evidence.unknown_dominant_xy[1, 1]
    assert evidence.structural_wall_ratio_xy[1, 1]
    assert evidence.wall_rejected_by_unknown_xy[1, 1]
    assert evidence.wall_support_rejected_unknown_xy[1, 1]
    assert not evidence.wall_support_unknown_gated_xy[1, 1]
    assert not evidence.strict_raw_wall_xy[1, 1]
    assert not evidence.wall_xy[1, 1]
    assert evidence.unknown_xy[1, 1]


def test_v18_free_wins_over_ratio_wall_when_unknown_gate_passes() -> None:
    grid = _grid(z_bins=10)
    grid.state[:, 2, 2] = int(VOXEL_OCCUPIED)
    grid.state[0:3, 2, 2] = int(VOXEL_FREE)

    evidence = _evidence(grid, {"wall_occupied_ratio_min_for_xy_wall": 0.60})

    assert evidence.vertical_free_xy[2, 2]
    assert evidence.occupied_any_xy[2, 2]
    assert not evidence.unknown_dominant_xy[2, 2]
    assert evidence.wall_support_unknown_gated_xy[2, 2]
    assert evidence.wall_rejected_by_free_xy[2, 2]
    assert not evidence.strict_raw_wall_xy[2, 2]
    assert not evidence.wall_xy[2, 2]
