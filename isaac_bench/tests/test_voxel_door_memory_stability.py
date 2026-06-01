from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_door_detector import (
    DoorMemoryObservationMaps,
    VoxelDoorDetectorConfig,
    VoxelDoorLineCandidate,
    VoxelDoorMemory,
    classify_voxel_door_seeds,
)
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def _candidate(
    candidate_id: int,
    *,
    row: int = 10,
    start_col: int = 5,
    end_col: int = 24,
    verified: bool = True,
    visual_start_col: int | None = None,
    visual_end_col: int | None = None,
) -> VoxelDoorLineCandidate:
    cut_cells = [(int(row), int(c)) for c in range(int(start_col), int(end_col) + 1)] if verified else []
    vs = int(start_col) if visual_start_col is None else int(visual_start_col)
    ve = int(end_col) if visual_end_col is None else int(visual_end_col)
    visual_cells = [(int(row), int(c)) for c in range(vs, ve + 1)]
    seed_cells = [(int(row), int(c)) for c in range(max(int(start_col), 10), min(int(end_col), 13) + 1)]
    return VoxelDoorLineCandidate(
        candidate_id=int(candidate_id),
        seed_component_id=1,
        seed_cells=seed_cells,
        center_rc=(float(row), 0.5 * float(start_col + end_col)),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(1.0, 0.0),
        seed_projected_centerline_cells=seed_cells,
        extended_centerline_cells=visual_cells,
        door_cut_cells=cut_cells,
        wall_anchor_a=(int(row), int(start_col) - 1),
        wall_anchor_b=(int(row), int(end_col) + 1),
        width_m=0.1 * float(max(1, end_col - start_col + 1)),
        accepted=True,
        reject_reason=None,
        debug={
            "partition_accepted": bool(verified),
            "partition_effective_verified": bool(verified),
            "partition_geometry_accepted": bool(verified),
            "stable_memory_refresh_eligible": True,
            "partition_inner_unknown_ratio": 0.0,
            "partition_inner_wall_ratio": 0.0,
        },
    )


def _observation(
    shape: tuple[int, int],
    *,
    sensor_range: np.ndarray | None = None,
    wall: np.ndarray | None = None,
    raw_seed: np.ndarray | None = None,
    verified_cut: np.ndarray | None = None,
) -> DoorMemoryObservationMaps:
    zeros = np.zeros(shape, dtype=bool)
    sensor = zeros.copy() if sensor_range is None else np.asarray(sensor_range, dtype=bool)
    return DoorMemoryObservationMaps(
        observed_xy=sensor.copy(),
        sensor_range_xy=sensor.copy(),
        vertical_free_xy=zeros.copy(),
        wall_xy=zeros.copy() if wall is None else np.asarray(wall, dtype=bool),
        raw_seed_mask=zeros.copy() if raw_seed is None else np.asarray(raw_seed, dtype=bool),
        current_verified_cut_mask=zeros.copy() if verified_cut is None else np.asarray(verified_cut, dtype=bool),
    )


def test_v31_unobserved_door_memory_does_not_decay() -> None:
    shape = (30, 36)
    cfg = VoxelDoorDetectorConfig(door_memory_decay_per_update=0.10, door_memory_ttl_updates=2)
    memory = VoxelDoorMemory(cfg)

    first = memory.update([_candidate(1)], step=1, shape=shape)
    start_conf = float(first.tracks[0].confidence)
    for step in range(2, 102):
        result = memory.update([], step=step, shape=shape, observation=_observation(shape))

    assert result.debug["voxel_door_memory_track_count"] == 1
    assert np.array_equal(first.stable_door_cut_mask, result.stable_door_cut_mask)
    assert float(result.tracks[0].confidence) == start_conf
    assert result.debug["voxel_door_memory_tracks_unobserved_count"] == 1


def test_v31_weak_refresh_does_not_overwrite_stable_cut() -> None:
    shape = (30, 36)
    memory = VoxelDoorMemory(VoxelDoorDetectorConfig(door_memory_decay_per_update=0.0, door_memory_weak_refresh_updates_visual=False))
    first = memory.update([_candidate(1, end_col=24)], step=1, shape=shape)

    short = _candidate(2, start_col=12, end_col=16, verified=False, visual_start_col=12, visual_end_col=16)
    result = memory.update([short], step=2, shape=shape)

    assert result.debug["voxel_door_memory_track_count"] == 1
    assert int(np.count_nonzero(result.stable_door_cut_mask)) == int(np.count_nonzero(first.stable_door_cut_mask))
    assert np.array_equal(result.stable_door_cut_mask, first.stable_door_cut_mask)
    assert result.debug["voxel_door_memory_tracks_weak_refreshed_no_geometry_update"] == 1


def test_v31_one_cell_shift_matches_same_track_with_dilated_iou() -> None:
    shape = (30, 36)
    cfg = VoxelDoorDetectorConfig(
        door_memory_decay_per_update=0.0,
        door_memory_match_distance_cells=0,
        door_memory_anchor_match_distance_cells=0,
        door_memory_seed_overlap_min=0.95,
        door_memory_match_dilation_cells=2,
        door_memory_dilated_iou_min=0.10,
    )
    memory = VoxelDoorMemory(cfg)
    memory.update([_candidate(1, row=10, start_col=5, end_col=24)], step=1, shape=shape)
    shifted = _candidate(2, row=11, start_col=5, end_col=24)
    result = memory.update([shifted], step=2, shape=shape)

    assert result.debug["voxel_door_memory_track_count"] == 1
    assert result.debug["voxel_door_memory_verified_update_count"] == 1


def test_v31_contradiction_requires_repeated_observed_wall_conflict() -> None:
    shape = (30, 36)
    cfg = VoxelDoorDetectorConfig(
        door_memory_decay_per_update=0.0,
        door_memory_contradictions_to_prune=3,
        door_memory_min_observed_cells_for_decay=1,
        door_memory_min_observed_cells_for_contradiction=1,
    )
    memory = VoxelDoorMemory(cfg)
    first = memory.update([_candidate(1)], step=1, shape=shape)
    sensor = np.ones(shape, dtype=bool)

    no_wall = memory.update([], step=2, shape=shape, observation=_observation(shape, sensor_range=sensor))
    assert no_wall.debug["voxel_door_memory_track_count"] == 1
    assert np.array_equal(no_wall.stable_door_cut_mask, first.stable_door_cut_mask)

    wall = np.ones(shape, dtype=bool)
    for step in range(3, 6):
        result = memory.update([], step=step, shape=shape, observation=_observation(shape, sensor_range=sensor, wall=wall))

    assert result.debug["voxel_door_memory_track_count"] == 0
    assert not np.any(result.stable_door_cut_mask)


def test_v31_sensor_range_unknown_counts_as_upper_solid_but_not_actual_occ() -> None:
    shape = (1, 1)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=0.1, min_y=0.0, max_y=0.1, width=1, height=1)
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.50, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.50),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.50
    z = grid.z_centers_m
    grid.state[:, 0, 0] = int(VOXEL_UNKNOWN)
    grid.state[(z >= 0.10) & (z < 1.80), 0, 0] = int(VOXEL_FREE)
    upper_occ = np.flatnonzero((z >= 1.80) & (z < 2.10))[:3]
    upper_unknown = np.flatnonzero((z >= 2.10) & (z < 2.40))[:2]
    grid.state[upper_occ, 0, 0] = int(VOXEL_OCCUPIED)
    grid.sensor_range_count[upper_unknown, 0, 0] = 1
    cfg = VoxelDoorDetectorConfig(
        seed_method="centroid_ratio",
        min_upper_observed_cells=5,
        min_upper_occupied_cells=3,
        upper_occupied_ratio_min_observed=0.95,
    )

    result = classify_voxel_door_seeds(voxel_grid=grid, config=cfg, sensor_range_count=grid.sensor_range_count)

    assert result.door_seed_mask[0, 0]
    assert int(result.debug["voxel_door_upper_actual_occupied_count_xy"][0, 0]) == 3
    assert int(result.debug["voxel_door_upper_solid_count_xy"][0, 0]) == 5
