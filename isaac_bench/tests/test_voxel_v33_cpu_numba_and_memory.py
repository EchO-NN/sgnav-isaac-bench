from __future__ import annotations

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
import isaac_bench.mapping.voxel_cpu_numba_backend as numba_backend
from isaac_bench.mapping.voxel_door_detector import StableDoorTrack, VoxelDoorMemory
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import StableSeparatorMemory, StableSeparatorTrack
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_cpu_numba_backend_is_real_or_reports_explicit_vectorized_fallback() -> None:
    info = MapInfo(resolution_m=0.10, min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5, width=30, height=30)
    grid = VoxelOccupancyGrid3D.zeros(
        (30, 30),
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=1.5,
            z_resolution_m=0.10,
            integration_backend="cpu_numba",
            cpu_numba_threads=2,
            cpu_numba_chunk_rays=16,
            cpu_numba_max_samples_per_ray=64,
            cuda_ray_step_voxels=1.0,
            sensor_range_tracking_enabled=True,
            cpu_numba_strict_required=True,
        ),
    )
    origin = np.asarray([-0.6, 0.0, 0.7], dtype=np.float32)
    endpoints = np.asarray([[0.6, 0.0, 0.7], [0.7, 0.2, 0.8], [0.5, -0.2, 0.6]], dtype=np.float32)

    if not numba_backend.numba_available():
        with pytest.raises(RuntimeError, match="numba is unavailable"):
            grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoints, floor_z=0.0)
        return

    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoints, floor_z=0.0)

    assert stats.integration_backend == "cpu_numba"
    assert stats.voxel_integrate_backend_thread_count >= 2
    assert stats.voxel_integrate_sample_kernel_ms >= 0.0
    assert stats.voxel_integrate_pass1_ms >= 0.0
    assert stats.voxel_integrate_pass2_ms >= 0.0
    assert stats.voxel_integrate_event_bucket_ms >= 0.0
    assert stats.voxel_numba_threading_layer
    assert stats.depth_rays_integrated == 3
    assert int(np.count_nonzero(grid.state == int(VOXEL_OCCUPIED))) >= 1


def test_cpu_numba_unavailable_requires_explicit_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numba_backend, "numba_available", lambda: False)
    info = MapInfo(resolution_m=0.10, min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5, width=30, height=30)
    origin = np.asarray([-0.6, 0.0, 0.7], dtype=np.float32)
    endpoints = np.asarray([[0.6, 0.0, 0.7]], dtype=np.float32)

    strict_grid = VoxelOccupancyGrid3D.zeros(
        (30, 30),
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=1.5,
            z_resolution_m=0.10,
            integration_backend="cpu_numba",
            cpu_numba_strict_required=True,
        ),
    )
    with pytest.raises(RuntimeError, match="numba is unavailable"):
        strict_grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoints, floor_z=0.0)

    fallback_grid = VoxelOccupancyGrid3D.zeros(
        (30, 30),
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=1.5,
            z_resolution_m=0.10,
            integration_backend="cpu_numba",
            cpu_numba_strict_required=False,
        ),
    )
    stats = fallback_grid.integrate_depth_points(camera_origin_world=origin, points_world=endpoints, floor_z=0.0)
    assert stats.integration_backend == "cpu_vectorized"
    assert stats.voxel_numba_requested_unavailable is True


def test_voxel_door_memory_round_trip_state_dict() -> None:
    memory = VoxelDoorMemory()
    memory._tracks = [
        StableDoorTrack(
            track_id=2,
            first_seen_step=1,
            last_seen_step=5,
            confidence=0.9,
            center_rc=(4.5, 6.0),
            major_dir_rc=(0.0, 1.0),
            cut_cells=[(4, 5), (4, 6)],
            visual_cells=[(4, 5), (4, 6), (4, 7)],
            stable_seed_cells=[(4, 6)],
            anchor_a_rc=(4, 4),
            anchor_b_rc=(4, 8),
            source_candidate_ids=[11, 12],
            update_count=3,
        )
    ]
    memory._next_track_id = 7

    restored = VoxelDoorMemory.from_state_dict(memory.to_state_dict())

    assert restored.to_state_dict()["next_track_id"] == 7
    assert restored.to_state_dict()["tracks"] == memory.to_state_dict()["tracks"]


def test_stable_separator_memory_round_trip_state_dict() -> None:
    memory = StableSeparatorMemory(ttl_updates=12, decay_per_update=0.1, min_confidence_to_keep=0.25)
    memory._tracks = [
        StableSeparatorTrack(
            track_id=4,
            confidence=0.8,
            first_seen_step=2,
            last_seen_step=6,
            line_cells=[(2, 3), (2, 4), (2, 5)],
            p0_rc=(2.0, 3.0),
            p1_rc=(2.0, 5.0),
            source_candidate_ids=[9],
        )
    ]
    memory._next_track_id = 8

    restored = StableSeparatorMemory.from_state_dict(memory.to_state_dict())

    assert restored.to_state_dict()["next_track_id"] == 8
    assert restored.to_state_dict()["tracks"] == memory.to_state_dict()["tracks"]
