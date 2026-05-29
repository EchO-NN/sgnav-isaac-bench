from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    _seed_reject_reason_from_code,
    classify_voxel_door_seeds,
)
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def test_v25_in_range_unknown_can_complete_upper_solid_for_door_seed() -> None:
    grid = _grid(shape=(8, 8), z_bins=26)
    rows = np.asarray([2, 3, 4], dtype=np.int32)
    col = 3
    lower = _z_between(grid, 1.40, 1.70)
    upper_actual = _z_between(grid, 1.70, 1.80)
    upper_unknown = _z_between(grid, 1.80, 2.10)

    grid.state[np.ix_(lower, rows, np.asarray([col]))] = int(VOXEL_FREE)
    grid.state[np.ix_(upper_actual, rows, np.asarray([col]))] = int(VOXEL_OCCUPIED)
    grid.state[np.ix_(upper_unknown, rows, np.asarray([col]))] = int(VOXEL_UNKNOWN)
    grid.sensor_range_count[np.ix_(upper_unknown, rows, np.asarray([col]))] = 1

    result = classify_voxel_door_seeds(voxel_grid=grid, config=VoxelDoorDetectorConfig())

    assert result.door_seed_mask[3, col]
    assert result.debug["voxel_door_upper_solid_ratio_effective_xy"][3, col] >= 0.75
    assert result.debug["voxel_door_upper_actual_occupied_count_xy"][3, col] >= 1


def test_v25_outside_range_unknown_cannot_form_door_seed_without_actual_occupied() -> None:
    grid = _grid(shape=(5, 5), z_bins=26)
    row, col = 2, 2
    lower = _z_between(grid, 1.40, 1.70)
    upper_unknown = _z_between(grid, 1.70, 2.10)
    grid.state[lower, row, col] = int(VOXEL_FREE)
    grid.state[upper_unknown, row, col] = int(VOXEL_UNKNOWN)
    grid.sensor_range_count[:, row, col] = 0

    result = classify_voxel_door_seeds(voxel_grid=grid, config=VoxelDoorDetectorConfig())
    reason = _seed_reject_reason_from_code(int(result.door_seed_reject_reason_map[row, col]))

    assert not result.door_seed_mask[row, col]
    assert reason == "upper_actual_occupied_cells_too_few"


def test_v25_lower_ratio_uses_effective_observed_denominator() -> None:
    grid = _grid(shape=(5, 5), z_bins=26)
    row, col = 2, 2
    lower = _z_between(grid, 1.50, 1.70)[:2]
    upper_actual = _z_between(grid, 1.70, 1.80)
    upper_unknown = _z_between(grid, 1.80, 2.00)

    grid.state[lower, row, col] = int(VOXEL_FREE)
    grid.state[upper_actual, row, col] = int(VOXEL_OCCUPIED)
    grid.state[upper_unknown, row, col] = int(VOXEL_UNKNOWN)
    grid.sensor_range_count[upper_unknown, row, col] = 1

    result = classify_voxel_door_seeds(
        voxel_grid=grid,
        config=VoxelDoorDetectorConfig(upper_actual_occupied_min_cells_for_cluster=1),
    )

    assert result.door_seed_mask[row, col]
    assert result.debug["voxel_door_lower_effective_count_xy"][row, col] == 2
    assert result.debug["voxel_door_lower_free_ratio_effective_xy"][row, col] == 1.0


def _grid(*, shape: tuple[int, int], z_bins: int) -> VoxelOccupancyGrid3D:
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
            z_max_m=float(z_bins) * 0.10,
            z_resolution_m=0.10,
            active_z_min_m=0.0,
            active_z_max_fallback_m=float(z_bins) * 0.10,
        ),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = float(z_bins) * 0.10
    return grid


def _z_between(grid: VoxelOccupancyGrid3D, z_min: float, z_max: float) -> np.ndarray:
    z = grid.z_centers_m
    return np.nonzero((z >= float(z_min)) & (z < float(z_max)))[0].astype(np.int32)
