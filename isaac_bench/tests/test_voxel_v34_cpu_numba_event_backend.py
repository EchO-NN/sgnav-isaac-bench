from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
import isaac_bench.mapping.voxel_cpu_numba_backend as numba_backend
from isaac_bench.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid(backend: str) -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=-2.0, max_x=2.0, min_y=-2.0, max_y=2.0, width=40, height=40)
    return VoxelOccupancyGrid3D.zeros(
        (40, 40),
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=1.5,
            z_resolution_m=0.10,
            integration_backend=backend,
            python_debug_backend_allowed=(backend == "python_debug"),
            cuda_ray_step_voxels=1.0,
            cuda_max_samples_per_ray=64,
            cpu_numba_threads=2,
            cpu_numba_chunk_rays=8,
            cpu_numba_max_samples_per_ray=64,
            cpu_numba_event_block_size=128,
            sensor_range_tracking_enabled=True,
            sensor_range_mark_endpoint_column_enabled=True,
            sensor_range_mark_ray_samples_enabled=True,
        ),
    )


def test_cpu_numba_event_backend_matches_runtime_vectorized_small_grid() -> None:
    if not numba_backend.numba_available():
        pytest.skip("numba unavailable")
    origin = np.asarray([0.0, 0.0, 0.7], dtype=np.float32)
    points = np.asarray(
        [
            [0.6, 0.0, 0.7],
            [0.7, 0.2, 0.8],
            [0.5, -0.2, 0.6],
            [-0.3, 0.5, 1.0],
            [1.2, -0.7, 0.3],
        ],
        dtype=np.float32,
    )

    baseline = _grid("cpu_vectorized")
    optimized = _grid("cpu_numba")
    baseline_stats = baseline.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)
    optimized_stats = optimized.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)

    assert optimized_stats.integration_backend == "cpu_numba"
    assert optimized_stats.voxel_integrate_backend_thread_count >= 2
    assert baseline_stats.depth_rays_integrated == optimized_stats.depth_rays_integrated
    assert np.array_equal(baseline.log_odds, optimized.log_odds)
    assert np.array_equal(baseline.state, optimized.state)
    assert np.array_equal(baseline.sensor_range_count, optimized.sensor_range_count)


def test_benchmark_script_outputs_v34_stage_profile(tmp_path: Path) -> None:
    if not numba_backend.numba_available():
        pytest.skip("numba unavailable")
    profile_json = tmp_path / "voxel_perf.json"
    cmd = [
        sys.executable,
        "isaac_bench/scripts/benchmark_voxel_grid_update.py",
        "--backend",
        "cpu_numba",
        "--threads",
        "2",
        "--rays",
        "2000",
        "--size-cells",
        "80",
        "--repeat",
        "2",
        "--warmup",
        "1",
        "--profile-json",
        str(profile_json),
        "--assert-backend",
        "cpu_numba",
        "--assert-max-ms",
        "1000",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True, text=True, capture_output=True)

    data = json.loads(profile_json.read_text(encoding="utf-8"))
    assert data["backend_actual"] == "cpu_numba"
    assert int(data["numba_threads"]) >= 2
    for key in (
        "pass1_ms",
        "pass2_ms",
        "event_bucket_ms",
        "log_apply_ms",
        "sensor_apply_ms",
        "endpoint_column_ms",
        "refresh_ms",
        "project_navigation_ms",
        "total_ms",
    ):
        assert key + "_mean" in data["summary"]
