from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_door_detector import VoxelDoorDetectorConfig, classify_voxel_door_seed_column, classify_voxel_door_seeds_vectorized, detect_voxel_doors
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_CONFLICT, VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_UNKNOWN, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid(shape: tuple[int, int] = (24, 24)) -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=2.4, min_y=0.0, max_y=2.4, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.30, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.30),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.30
    return grid


def _write_door_column(grid: VoxelOccupancyGrid3D, row: int, col: int) -> None:
    centers = grid.z_centers_m
    grid.state[(centers >= 0.10) & (centers < 1.80), row, col] = int(VOXEL_FREE)
    grid.state[(centers >= 1.80) & (centers < 2.15), row, col] = int(VOXEL_OCCUPIED)


def test_door_seed_accepted() -> None:
    grid = _grid()
    _write_door_column(grid, 5, 5)

    result = detect_voxel_doors(
        voxel_grid=grid,
        free_map=np.ones(grid.shape, dtype=bool),
        wall_map=np.zeros(grid.shape, dtype=bool),
        unknown_map=np.zeros(grid.shape, dtype=bool),
        resolution_m=0.10,
    )

    assert result.door_seed_mask[5, 5]


def test_door_seed_accepted_at_active_range_end() -> None:
    grid = _grid()
    centers = grid.z_centers_m
    col_state = np.zeros(grid.z_bin_count, dtype=np.uint8)
    col_state[(centers >= 0.10) & (centers < 1.80)] = int(VOXEL_FREE)
    col_state[(centers >= 1.80)] = int(VOXEL_OCCUPIED)

    ev = classify_voxel_door_seed_column(col_state, centers, grid.active_z_indices(), VoxelDoorDetectorConfig(seed_method="strict_contiguous"))

    assert ev.accepted


def test_unknown_below_top_occupied_rejected() -> None:
    grid = _grid()
    centers = grid.z_centers_m
    col_state = np.full(grid.z_bin_count, int(VOXEL_UNKNOWN), dtype=np.uint8)
    col_state[(centers >= 0.10) & (centers < 0.40)] = int(VOXEL_FREE)
    col_state[(centers >= 1.80) & (centers < 2.20)] = int(VOXEL_OCCUPIED)

    ev = classify_voxel_door_seed_column(col_state, centers, grid.active_z_indices(), VoxelDoorDetectorConfig(seed_method="strict_contiguous"))

    assert not ev.accepted
    assert ev.reject_reason == "unknown_before_top_occupied"


def test_occupied_too_low_rejected() -> None:
    grid = _grid()
    centers = grid.z_centers_m
    col_state = np.full(grid.z_bin_count, int(VOXEL_UNKNOWN), dtype=np.uint8)
    col_state[(centers >= 0.10) & (centers < 0.80)] = int(VOXEL_FREE)
    col_state[(centers >= 0.80) & (centers < 1.20)] = int(VOXEL_OCCUPIED)

    ev = classify_voxel_door_seed_column(col_state, centers, grid.active_z_indices(), VoxelDoorDetectorConfig(seed_method="strict_contiguous"))

    assert not ev.accepted
    assert ev.reject_reason == "top_occupied_too_low"


def test_top_occupied_run_too_short_rejected() -> None:
    grid = _grid()
    centers = grid.z_centers_m
    col_state = np.full(grid.z_bin_count, int(VOXEL_UNKNOWN), dtype=np.uint8)
    col_state[(centers >= 0.10) & (centers < 1.80)] = int(VOXEL_FREE)
    col_state[(centers >= 1.80) & (centers < 1.95)] = int(VOXEL_OCCUPIED)

    ev = classify_voxel_door_seed_column(col_state, centers, grid.active_z_indices(), VoxelDoorDetectorConfig(seed_method="strict_contiguous"))

    assert not ev.accepted
    assert ev.reject_reason == "top_occupied_run_too_short"


def test_seed_blob_projects_to_single_centerline_and_hits_two_walls() -> None:
    grid = _grid()
    seed_rows = range(9, 12)
    seed_cols = range(7, 14)
    for row in seed_rows:
        for col in seed_cols:
            _write_door_column(grid, row, col)
    free = np.zeros(grid.shape, dtype=bool)
    free[10, 5:16] = True
    free[9:12, 7:14] = True
    wall = np.zeros(grid.shape, dtype=bool)
    wall[10, 5] = True
    wall[10, 15] = True
    unknown = np.zeros(grid.shape, dtype=bool)

    result = detect_voxel_doors(
        voxel_grid=grid,
        free_map=free,
        wall_map=wall,
        unknown_map=unknown,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(partition_reject_small_known_side_enabled=False),
    )

    assert int(result.debug["voxel_door_accepted_count"]) >= 1
    rows, _cols = np.nonzero(result.accepted_door_centerline_mask)
    assert rows.size > 0
    assert int(rows.max() - rows.min()) <= 1


def test_seed_door_acceptance_does_not_require_width_limit_by_default() -> None:
    grid = _grid(shape=(24, 60))
    for col in range(20, 51):
        _write_door_column(grid, 10, col)
    free = np.zeros(grid.shape, dtype=bool)
    free[10, 10:56] = True
    wall = np.zeros(grid.shape, dtype=bool)
    wall[10, 10] = True
    wall[10, 55] = True
    unknown = np.zeros(grid.shape, dtype=bool)

    result = detect_voxel_doors(
        voxel_grid=grid,
        free_map=free,
        wall_map=wall,
        unknown_map=unknown,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            door_width_max_m=0.60,
            visual_width_max_m=5.00,
            extend_max_m=5.00,
            seed_cluster_max_width_m=5.00,
            partition_topology_enabled=False,
        ),
    )

    assert int(result.debug["voxel_door_accepted_count"]) == 1
    assert result.candidates[0].width_m > 0.60
    assert result.candidates[0].reject_reason is None


def test_partition_cut_acceptance_ignores_visual_width_limit_when_extension_is_bounded() -> None:
    grid = _grid(shape=(24, 70))
    for col in range(20, 51):
        _write_door_column(grid, 10, col)
    free = np.zeros(grid.shape, dtype=bool)
    free[10, 18:55] = True
    wall = np.zeros(grid.shape, dtype=bool)
    wall[10, 18] = True
    wall[10, 54] = True
    unknown = np.zeros(grid.shape, dtype=bool)

    result = detect_voxel_doors(
        voxel_grid=grid,
        free_map=free,
        wall_map=wall,
        unknown_map=unknown,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            visual_width_max_m=0.60,
            partition_cut_max_total_extension_m=1.60,
            partition_topology_enabled=False,
        ),
    )

    candidate = result.candidates[0]
    assert int(result.debug["voxel_door_accepted_count"]) == 1
    assert candidate.width_m > 1.60
    assert candidate.reject_reason is None
    assert candidate.debug["partition_accepted"] is True
    assert candidate.debug["door_partition_width_limit_enforced"] is False
    assert candidate.debug["door_extension_total_m"] <= 1.60
    limit_cells = {tuple(cell) for cell in candidate.debug["door_extension_limit_cells"]}
    assert not any(20 <= col <= 50 for _row, col in limit_cells)
    assert candidate.debug["door_extension_limit_excludes_seed_cells"] is True


def test_partition_cut_rejects_total_extension_over_limit() -> None:
    grid = _grid(shape=(24, 70))
    for col in range(30, 32):
        _write_door_column(grid, 10, col)
    free = np.zeros(grid.shape, dtype=bool)
    free[10, 10:56] = True
    wall = np.zeros(grid.shape, dtype=bool)
    wall[10, 10] = True
    wall[10, 55] = True
    unknown = np.zeros(grid.shape, dtype=bool)

    result = detect_voxel_doors(
        voxel_grid=grid,
        free_map=free,
        wall_map=wall,
        unknown_map=unknown,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            visual_width_max_m=10.0,
            one_seed_one_wall_visual_width_max_m=10.0,
            seed_pair_bridge_visual_width_max_m=10.0,
            extend_max_m=5.0,
            partition_cut_max_total_extension_m=1.60,
            seed_cluster_max_width_m=10.0,
            min_seed_cells_for_accepted_extension=1,
            min_seed_line_length_cells_for_accepted_extension=1,
            min_seed_elongation_for_direction=1.0,
            partition_topology_enabled=False,
        ),
    )

    candidate = result.candidates[0]
    assert int(result.debug["voxel_door_accepted_count"]) == 0
    assert candidate.reject_reason == "door_extension_total_too_long"
    assert candidate.debug["door_extension_total_m"] > 1.60
    assert not np.any(result.door_cut_mask)


def test_seed_door_width_limit_can_still_be_enforced_for_compatibility() -> None:
    grid = _grid(shape=(24, 60))
    for col in range(20, 51):
        _write_door_column(grid, 10, col)
    free = np.zeros(grid.shape, dtype=bool)
    free[10, 10:56] = True
    wall = np.zeros(grid.shape, dtype=bool)
    wall[10, 10] = True
    wall[10, 55] = True
    unknown = np.zeros(grid.shape, dtype=bool)

    result = detect_voxel_doors(
        voxel_grid=grid,
        free_map=free,
        wall_map=wall,
        unknown_map=unknown,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(door_width_max_m=0.60, enforce_seed_door_width_limits=True),
    )

    assert int(result.debug["voxel_door_accepted_count"]) == 0
    assert result.candidates[0].reject_reason == "door_width_out_of_range"


def test_vectorized_seed_matches_single_column_debug_function() -> None:
    grid = _grid(shape=(5, 6))
    rng = np.random.default_rng(4)
    choices = np.asarray([VOXEL_UNKNOWN, VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_CONFLICT], dtype=np.uint8)
    grid.state[:, :, :] = rng.choice(choices, size=grid.state.shape)
    _write_door_column(grid, 2, 3)
    cfg = VoxelDoorDetectorConfig()

    seed, reason_map, lower_free, top_occ, unknown_tail, first_occ_z, _counts, _evidence = classify_voxel_door_seeds_vectorized(
        grid.state,
        grid.z_centers_m,
        grid.active_z_indices(),
        cfg,
        shape=grid.shape,
    )

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            ev = classify_voxel_door_seed_column(grid.state[:, row, col], grid.z_centers_m, grid.active_z_indices(), cfg, row=row, col=col)
            assert bool(seed[row, col]) == bool(ev.accepted)
            assert int(lower_free[row, col]) == int(ev.lower_free_cells)
            assert int(top_occ[row, col]) == int(ev.top_occupied_cells)
            assert int(unknown_tail[row, col]) == int(ev.unknown_tail_cells)
            if ev.first_occupied_z_m is None:
                assert np.isnan(first_occ_z[row, col])
            else:
                assert np.isclose(first_occ_z[row, col], ev.first_occupied_z_m)
            if ev.accepted:
                assert int(reason_map[row, col]) == 1
