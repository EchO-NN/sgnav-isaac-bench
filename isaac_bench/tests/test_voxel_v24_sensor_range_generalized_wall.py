from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig
from isaac_bench.mapping.voxel_roomseg_evidence import VoxelRoomsegEvidenceConfig, build_voxel_roomseg_evidence, classify_voxel_columns_for_roomseg


def _classify_column(state_values: np.ndarray, sensor_values: np.ndarray | None = None) -> dict[str, np.ndarray]:
    state = np.asarray(state_values, dtype=np.uint8).reshape((-1, 1, 1))
    sensor = None if sensor_values is None else np.asarray(sensor_values, dtype=np.uint8).reshape((-1, 1, 1))
    return classify_voxel_columns_for_roomseg(
        state_active=state,
        sensor_range_active=sensor,
        navigation_free_mask=np.zeros((1, 1), dtype=bool),
        navigation_obstacle_mask=np.zeros((1, 1), dtype=bool),
        navigation_unknown_mask=np.zeros((1, 1), dtype=bool),
        cfg=VoxelRoomsegEvidenceConfig(
            min_free_z_cells_for_xy_free=3,
            wall_use_generalized_occupied_ratio=True,
            count_in_range_unknown_as_occupied_for_wall=True,
            wall_generalized_occupied_ratio_min_for_xy_wall=0.90,
            wall_min_actual_occupied_z_cells_for_xy_wall=3,
            outside_unknown_ratio_min_for_xy_unknown=0.50,
            min_effective_range_z_cells_for_known_column=3,
            fill_small_unknown_holes_inside_vertical_free=False,
        ),
    )


def _cell(classified: dict[str, np.ndarray], key: str):
    return np.asarray(classified[key])[0, 0].item()


def test_in_range_unknown_can_complete_generalized_wall_ratio() -> None:
    state = np.zeros(10, dtype=np.uint8)
    state[:3] = int(VOXEL_OCCUPIED)
    sensor = np.ones(10, dtype=np.uint8)

    classified = _classify_column(state, sensor)

    assert bool(_cell(classified, "wall"))
    assert not bool(_cell(classified, "vertical_free"))
    assert not bool(_cell(classified, "unknown"))
    assert float(_cell(classified, "generalized_occupied_ratio")) == 1.0
    assert bool(_cell(classified, "wall_from_in_range_unknown"))


def test_outside_range_unknown_does_not_complete_wall_ratio() -> None:
    state = np.zeros(10, dtype=np.uint8)
    state[:3] = int(VOXEL_OCCUPIED)
    sensor = np.zeros(10, dtype=np.uint8)
    sensor[:3] = 1

    classified = _classify_column(state, sensor)

    assert not bool(_cell(classified, "wall"))
    assert bool(_cell(classified, "unknown"))
    assert bool(_cell(classified, "outside_unknown_dominant"))


def test_free_keeps_highest_priority_over_generalized_wall() -> None:
    state = np.zeros(10, dtype=np.uint8)
    state[:3] = int(VOXEL_FREE)
    state[3:6] = int(VOXEL_OCCUPIED)
    sensor = np.ones(10, dtype=np.uint8)

    classified = _classify_column(state, sensor)

    assert bool(_cell(classified, "vertical_free"))
    assert not bool(_cell(classified, "wall"))
    assert not bool(_cell(classified, "unknown"))


def test_pure_in_range_unknown_cannot_be_wall_without_actual_occupied() -> None:
    state = np.zeros(10, dtype=np.uint8)
    sensor = np.ones(10, dtype=np.uint8)

    classified = _classify_column(state, sensor)

    assert not bool(_cell(classified, "wall"))
    assert bool(_cell(classified, "unknown"))
    assert not bool(_cell(classified, "wall_actual_occupied_requirement"))


def test_actual_occupied_below_threshold_cannot_be_wall() -> None:
    state = np.zeros(10, dtype=np.uint8)
    state[:2] = int(VOXEL_OCCUPIED)
    sensor = np.ones(10, dtype=np.uint8)

    classified = _classify_column(state, sensor)

    assert not bool(_cell(classified, "wall"))
    assert bool(_cell(classified, "unknown"))
    assert not bool(_cell(classified, "wall_actual_occupied_requirement"))


def test_sensor_range_marking_does_not_change_log_odds_or_state() -> None:
    grid = _grid(shape=(4, 4), z_bins=6)
    before_log = grid.log_odds.copy()
    before_state = grid.state.copy()

    count, changed = grid.mark_sensor_range_voxels_array(np.asarray([[1, 2, 3], [1, 2, 3], [4, 2, 3]], dtype=np.int32))

    assert count == 2
    assert changed.size == 2
    assert grid.sensor_range_count[1, 2, 3] == 1
    assert grid.sensor_range_count[4, 2, 3] == 1
    assert np.array_equal(grid.log_odds, before_log)
    assert np.array_equal(grid.state, before_state)


def test_cpu_vectorized_endpoint_column_marks_active_z_sensor_range() -> None:
    grid = _grid(shape=(20, 20), z_bins=10)
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 0.46
    origin = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
    endpoint = np.asarray([[0.5, 0.0, 0.5]], dtype=np.float32)

    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoint, floor_z=0.0)

    row = int((grid.map_info.max_y - 0.0) / grid.map_info.resolution_m)
    col = int((0.5 - grid.map_info.min_x) / grid.map_info.resolution_m)
    active_z = grid.active_z_indices()
    assert stats.integration_backend == "cpu_vectorized"
    assert stats.sensor_range_update_count > 0
    assert active_z.size == 5
    assert np.all(grid.sensor_range_count[active_z, row, col] > 0)


def test_build_evidence_wall_from_in_range_unknown_and_outside_unknown_stays_unknown() -> None:
    grid = _grid(shape=(6, 6), z_bins=10)
    grid.state[:3, 2, 2] = int(VOXEL_OCCUPIED)
    grid.sensor_range_count[:, 2, 2] = 1
    grid.state[:3, 3, 3] = int(VOXEL_OCCUPIED)
    grid.sensor_range_count[:3, 3, 3] = 1

    evidence = build_voxel_roomseg_evidence(
        voxel_grid=grid,
        navigation_free_mask=np.zeros(grid.shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(grid.shape, dtype=bool),
        unknown_mask_from_navigation=np.zeros(grid.shape, dtype=bool),
        resolution_m=0.10,
        config={"active_z_min_m": 0.0, "fill_small_unknown_holes_inside_vertical_free": False},
    )

    assert evidence.wall_xy[2, 2]
    assert evidence.debug["voxel_wall_from_in_range_unknown_xy"][2, 2]
    assert not evidence.wall_xy[3, 3]
    assert evidence.unknown_xy[3, 3]
    assert evidence.debug["voxel_outside_unknown_dominant_xy"][3, 3]


def _grid(*, shape: tuple[int, int], z_bins: int) -> VoxelOccupancyGrid3D:
    info = MapInfo(
        resolution_m=0.10,
        min_x=-1.0,
        max_x=-1.0 + shape[1] * 0.10,
        min_y=-1.0,
        max_y=-1.0 + shape[0] * 0.10,
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
            integration_backend="cpu_vectorized",
        ),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = float(z_bins) * 0.10
    return grid
