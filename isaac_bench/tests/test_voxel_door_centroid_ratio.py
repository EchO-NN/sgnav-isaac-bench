from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_door_detector import VoxelDoorDetectorConfig, classify_voxel_door_seed_column, classify_voxel_door_seeds_vectorized
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_UNKNOWN, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid() -> VoxelOccupancyGrid3D:
    shape = (3, 3)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=0.3, min_y=0.0, max_y=0.3, width=3, height=3)
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=3.00, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=3.00),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 3.00
    return grid


def _door_state(grid: VoxelOccupancyGrid3D) -> np.ndarray:
    z = grid.z_centers_m
    state = np.full(grid.z_bin_count, int(VOXEL_UNKNOWN), dtype=np.uint8)
    state[(z >= 0.10) & (z < 1.80)] = int(VOXEL_FREE)
    state[(z >= 1.80) & (z < 2.20)] = int(VOXEL_OCCUPIED)
    return state


def _classify(state: np.ndarray):
    grid = _grid()
    grid.state[:, 1, 1] = state
    cfg = VoxelDoorDetectorConfig(seed_method="centroid_ratio")
    return classify_voxel_door_seed_column(grid.state[:, 1, 1], grid.z_centers_m, grid.active_z_indices(), cfg, row=1, col=1)


def test_centroid_ratio_accepts_standard_door_column() -> None:
    grid = _grid()
    state = _door_state(grid)

    ev = _classify(state)

    assert ev.accepted
    assert ev.turn_z_m is not None and ev.turn_z_m >= 1.799
    assert ev.lower_free_ratio is not None and ev.lower_free_ratio >= 0.90
    assert ev.upper_occupied_ratio_observed is not None and ev.upper_occupied_ratio_observed >= 0.80


def test_centroid_ratio_tolerates_one_lower_unknown_noise_cell() -> None:
    grid = _grid()
    state = _door_state(grid)
    state[5] = int(VOXEL_UNKNOWN)

    ev = _classify(state)

    assert ev.accepted
    assert ev.lower_free_ratio is not None and ev.lower_free_ratio >= 0.90


def test_centroid_ratio_rejects_low_lower_free_ratio() -> None:
    grid = _grid()
    state = _door_state(grid)
    state[3:7] = int(VOXEL_UNKNOWN)

    ev = _classify(state)

    assert not ev.accepted
    assert ev.reject_reason == "lower_free_ratio_too_low"


def test_centroid_ratio_rejects_low_upper_occupied_ratio() -> None:
    grid = _grid()
    state = _door_state(grid)
    z = grid.z_centers_m
    state[(z >= 2.20) & (z < 2.50)] = int(VOXEL_FREE)
    state[(z >= 2.50) & (z < 2.90)] = int(VOXEL_OCCUPIED)
    grid.state[:, 1, 1] = state
    cfg = VoxelDoorDetectorConfig(seed_method="centroid_ratio", lower_free_ratio_min=0.50, upper_occupied_ratio_min_observed=0.90)

    ev = classify_voxel_door_seed_column(grid.state[:, 1, 1], grid.z_centers_m, grid.active_z_indices(), cfg, row=1, col=1)

    assert not ev.accepted
    assert ev.reject_reason == "upper_occupied_ratio_too_low"


def test_centroid_ratio_rejects_low_turn_z() -> None:
    grid = _grid()
    z = grid.z_centers_m
    state = np.full(grid.z_bin_count, int(VOXEL_UNKNOWN), dtype=np.uint8)
    state[(z >= 0.10) & (z < 0.80)] = int(VOXEL_FREE)
    state[(z >= 1.80) & (z < 2.20)] = int(VOXEL_OCCUPIED)

    ev = _classify(state)

    assert not ev.accepted
    assert ev.reject_reason == "turn_z_below_1p8"


def test_centroid_ratio_vectorized_outputs_debug_maps() -> None:
    grid = _grid()
    grid.state[:, 1, 1] = _door_state(grid)
    cfg = VoxelDoorDetectorConfig(seed_method="centroid_ratio")

    seed, reason_map, _lower, _upper, _tail, _first, counts, _evidence, maps = classify_voxel_door_seeds_vectorized(
        grid.state,
        grid.z_centers_m,
        grid.active_z_indices(),
        cfg,
        shape=grid.shape,
        return_debug=True,
    )

    assert seed[1, 1]
    assert int(reason_map[1, 1]) == 1
    assert int(counts.get("lower_free_cells_too_few", 0)) >= 0
    assert maps["voxel_door_turn_z_estimate_xy"].shape == grid.shape
    assert maps["voxel_door_upper_occupied_ratio_observed_xy"][1, 1] >= 0.80
