from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
import isaac_bench.mapping.voxel_projection_numba as projection_numba
from isaac_bench.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def _grid(backend: str = "cpu_numba", **overrides: object) -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=-2.0, max_x=2.0, min_y=-2.0, max_y=2.0, width=40, height=40)
    raw = {
        "z_min_m": 0.0,
        "z_max_m": 1.5,
        "z_resolution_m": 0.10,
        "integration_backend": backend,
        "python_debug_backend_allowed": (backend == "python_debug"),
        "cuda_ray_step_voxels": 1.0,
        "cuda_max_samples_per_ray": 64,
        "cpu_numba_threads": 2,
        "cpu_numba_threads_mode": "manual",
        "cpu_numba_chunk_rays": 8,
        "cpu_numba_max_samples_per_ray": 64,
        "cpu_numba_event_block_size": 128,
        "sensor_range_tracking_enabled": True,
        "sensor_range_mark_endpoint_column_enabled": True,
        "sensor_range_mark_ray_samples_enabled": False,
    }
    raw.update(overrides)
    cfg = VoxelOccupancyGridConfig(**raw)
    grid = VoxelOccupancyGrid3D.zeros((40, 40), info, cfg)
    grid.active_z_min_m = 0.10
    grid.active_z_max_m = 1.20
    return grid


def _points() -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray([0.0, 0.0, 0.7], dtype=np.float32)
    points = np.asarray(
        [[0.6, 0.0, 0.7], [0.7, 0.2, 0.8], [0.5, -0.2, 0.6], [-0.3, 0.5, 1.0]],
        dtype=np.float32,
    )
    return origin, points


def test_sensor_ray_events_disabled_but_endpoint_column_kept() -> None:
    origin, points = _points()
    grid = _grid()

    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)

    assert stats.integration_backend == "cpu_numba"
    assert stats.voxel_integrate_total_sensor_events == 0
    assert stats.voxel_integrate_apply_sensor_ms == 0.0
    assert int(np.count_nonzero(grid.sensor_range_count)) > 0


def test_inline_apply_refresh_matches_vectorized_without_changed_scan() -> None:
    origin, points = _points()
    reference = _grid("cpu_vectorized")
    optimized = _grid("cpu_numba")

    reference.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)
    stats = optimized.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)

    assert np.array_equal(reference.log_odds, optimized.log_odds)
    assert np.array_equal(reference.state, optimized.state)
    assert stats.refresh_mode == "inline_apply"
    assert stats.voxel_integrate_changed_scan_ms == 0.0


def test_autotune_selects_candidate_and_reports_effective_threads(tmp_path: Path) -> None:
    origin, points = _points()
    grid = _grid(
        cpu_numba_threads=4,
        cpu_numba_threads_mode="auto",
        cpu_numba_autotune_candidates=(2, 4),
        cpu_numba_autotune_repeat=1,
        cpu_numba_autotune_rays=4,
        cpu_numba_autotune_cache_path=str(tmp_path / "autotune.json"),
    )

    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)

    assert stats.voxel_integrate_numba_threads_mode == "auto"
    assert stats.voxel_integrate_backend_effective_thread_count in {2, 4}
    assert stats.voxel_integrate_numba_requested_thread_count == 4


def test_navigation_projection_numba_matches_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _grid("cpu_vectorized")
    grid.state[3:7, 10, 10] = 2
    grid.state[2:8, 12, 12] = 1
    endpoint = np.zeros(grid.shape, dtype=np.uint16)
    endpoint[8, 8] = 1

    fast = grid.project_navigation(nav_endpoint_count_xy=endpoint)
    assert fast.debug["voxel_project_navigation_backend"] == "numba_column"

    monkeypatch.setattr(projection_numba, "project_navigation_columns", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced")))
    fallback = grid.project_navigation(nav_endpoint_count_xy=endpoint)
    assert fallback.debug["voxel_project_navigation_backend"] == "numpy"

    assert np.array_equal(fast.free, fallback.free)
    assert np.array_equal(fast.occupied, fallback.occupied)
    assert np.array_equal(fast.observed, fallback.observed)
    assert np.array_equal(fast.unknown, fallback.unknown)


def test_benchmark_threads_list_outputs_per_thread_profiles(tmp_path: Path) -> None:
    out = tmp_path / "threads.json"
    cmd = [
        sys.executable,
        "isaac_bench/scripts/benchmark_voxel_grid_update.py",
        "--backend",
        "cpu_numba",
        "--threads-list",
        "2,4",
        "--rays",
        "500",
        "--size-cells",
        "80",
        "--repeat",
        "1",
        "--warmup",
        "0",
        "--json-out",
        str(out),
        "--assert-backend",
        "cpu_numba",
        "--assert-max-ms",
        "1000",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True, text=True, capture_output=True)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["best_threads"] in {2, 4}
    assert len(data["threads_results"]) == 2
    for result in data["threads_results"]:
        assert result["backend_actual"] == "cpu_numba"
        assert "changed_scan_ms_mean" in result["summary"]
