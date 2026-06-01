from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_cuda_backend import VoxelCudaBackend
from isaac_bench.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark voxel occupancy grid integration backends.")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda_torch", "cpu_numba", "cpu_vectorized", "python_debug"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rays", type=int, default=76000)
    parser.add_argument("--size-cells", type=int, default=240)
    parser.add_argument("--chunk-rays", type=int, default=131072)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--threads-mode", default="manual", choices=["manual", "auto"])
    parser.add_argument("--threads-list", default=None)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--profile-json", default=None)
    parser.add_argument("--assert-backend", default=None)
    parser.add_argument("--assert-max-ms", type=float, default=None)
    parser.add_argument("--show-stage-table", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    if args.backend == "cuda_torch" and not VoxelCudaBackend.is_available(args.device):
        print("voxel_integration_backend=cuda_torch unavailable=true python_debug_backend_used=false")
        return 0

    if args.threads_list:
        thread_values = [int(v.strip()) for v in str(args.threads_list).split(",") if v.strip()]
        results = []
        for thread_count in thread_values:
            result = _run_benchmark_for_threads(args, int(thread_count), force_threads_mode="manual", include_objects=False)
            results.append(result)
            print(
                "threads=%d backend=%s total_mean=%.3fms total_max=%.3fms integrate_mean=%.3fms"
                % (
                    int(thread_count),
                    str(result["backend_actual"]),
                    float(result["summary"]["total_ms_mean"]),
                    float(result["summary"]["total_ms_max"]),
                    float(result["summary"]["integrate_ms_mean"]),
                )
            )
        best = min(results, key=lambda item: float(item["summary"]["total_ms_mean"])) if results else None
        row = {
            "backend_requested": str(args.backend),
            "threads_results": results,
            "best_threads": None if best is None else int(best["thread_count"]),
            "best_total_ms_mean": None if best is None else float(best["summary"]["total_ms_mean"]),
        }
        if bool(args.show_stage_table):
            for result in results:
                print("\nthreads=%d" % int(result["thread_count"]))
                _print_stage_table(result["runs"])
        if args.assert_backend:
            for result in results:
                if str(result["backend_actual"]) != str(args.assert_backend):
                    raise SystemExit("backend assertion failed for threads=%d: expected %s got %s" % (int(result["thread_count"]), str(args.assert_backend), str(result["backend_actual"])))
        if args.assert_max_ms is not None:
            for result in results:
                if float(result["summary"]["total_ms_max"]) > float(args.assert_max_ms):
                    raise SystemExit("max total ms assertion failed for threads=%d: %.3f > %.3f" % (int(result["thread_count"]), float(result["summary"]["total_ms_max"]), float(args.assert_max_ms)))
        if args.json_out:
            Path(args.json_out).expanduser().write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.profile_json:
            Path(args.profile_json).expanduser().write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    row = _run_benchmark_for_threads(args, int(args.threads), force_threads_mode=str(args.threads_mode), include_objects=True)
    stats = row["stats_obj"]
    projection = row["projection_obj"]
    if str(args.backend) == "cpu_numba" and not bool(args.no_warmup):
        from isaac_bench.mapping.voxel_cpu_numba_backend import VoxelCpuNumbaBackend

        _ = VoxelCpuNumbaBackend
    rows = row["runs"]
    summary = row["summary"]
    print(
        "backend=%s rays=%d free_updates=%d occ_updates=%d integrate=%.3fms projection=%.3fms total=%.3fms"
        % (
            str(stats.integration_backend),
            int(stats.depth_rays_integrated),
            int(stats.free_update_count),
            int(stats.occupied_update_count),
            float(row["integrate_ms"]),
            float(row["projection_ms"]),
            float(row["total_ms"]),
        )
    )
    print("voxel_integration_backend=%s" % str(stats.integration_backend))
    print("python_debug_backend_used=%s" % ("true" if bool(stats.python_debug_backend_used) else "false"))
    print(
        "nav_free=%d nav_occ=%d nav_unknown=%d"
        % (
            int(np.count_nonzero(projection.free)),
            int(np.count_nonzero(projection.occupied)),
            int(np.count_nonzero(projection.unknown)),
        )
    )
    if bool(args.show_stage_table):
        _print_stage_table(rows)
    row.pop("stats_obj", None)
    row.pop("projection_obj", None)
    if args.assert_backend and str(stats.integration_backend) != str(args.assert_backend):
        raise SystemExit("backend assertion failed: expected %s got %s" % (str(args.assert_backend), str(stats.integration_backend)))
    if args.assert_max_ms is not None and float(summary["total_ms_max"]) > float(args.assert_max_ms):
        raise SystemExit("max total ms assertion failed: %.3f > %.3f" % (float(summary["total_ms_max"]), float(args.assert_max_ms)))
    if args.json_out:
        Path(args.json_out).expanduser().write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.profile_json:
        Path(args.profile_json).expanduser().write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _numba_available() -> bool:
    try:
        from isaac_bench.mapping.voxel_cpu_numba_backend import numba_available

        return bool(numba_available())
    except Exception:
        return False


def _run_benchmark_for_threads(args: argparse.Namespace, thread_count: int, *, force_threads_mode: str, include_objects: bool) -> dict[str, object]:
    grid, origin, points, floor_z = _load_benchmark_case(args, int(thread_count), str(force_threads_mode))
    if str(args.backend) == "cpu_numba" and not bool(args.no_warmup):
        from isaac_bench.mapping.voxel_cpu_numba_backend import VoxelCpuNumbaBackend

        VoxelCpuNumbaBackend.warmup(grid)
    warmup_iters = 0 if bool(args.no_warmup) else max(0, int(args.warmup))
    for _ in range(warmup_iters):
        _run_once(grid, origin, points, floor_z=floor_z)

    rows = []
    projection = None
    stats = None
    integration_ms = 0.0
    project_ms = 0.0
    for _ in range(max(1, int(args.repeat))):
        stats, projection, integration_ms, project_ms = _run_once(grid, origin, points, floor_z=floor_z)
        stats.voxel_integrate_projection_ms = float(project_ms)
        rows.append(_profile_row(args.backend, stats, integration_ms, project_ms))
    assert stats is not None and projection is not None
    summary = _summarize_rows(rows)
    row: dict[str, object] = {
        "backend_requested": str(args.backend),
        "backend_actual": str(stats.integration_backend),
        "numba_available": bool(_numba_available()),
        "rays": int(stats.depth_rays_integrated),
        "free_updates": int(stats.free_update_count),
        "occupied_updates": int(stats.occupied_update_count),
        "integrate_ms": float(integration_ms),
        "projection_ms": float(project_ms),
        "total_ms": float(integration_ms + project_ms),
        "thread_count": int(thread_count),
        "numba_threads": int(stats.voxel_integrate_backend_thread_count),
        "numba_effective_threads": int(getattr(stats, "voxel_integrate_backend_effective_thread_count", stats.voxel_integrate_backend_thread_count)),
        "numba_requested_threads": int(getattr(stats, "voxel_integrate_numba_requested_thread_count", thread_count)),
        "numba_threads_mode": str(getattr(stats, "voxel_integrate_numba_threads_mode", force_threads_mode)),
        "numba_threading_layer": str(getattr(stats, "voxel_numba_threading_layer", "unknown")),
        "numba_requested_unavailable": bool(stats.voxel_numba_requested_unavailable),
        "stats": stats.to_dict(),
        "summary": summary,
        "runs": rows,
    }
    if include_objects:
        row["stats_obj"] = stats
        row["projection_obj"] = projection
    return row


def _run_once(grid: VoxelOccupancyGrid3D, origin: np.ndarray, points: np.ndarray, *, floor_z: float):
    started = time.perf_counter()
    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=float(floor_z))
    integration_ms = (time.perf_counter() - started) * 1000.0
    project_started = time.perf_counter()
    projection = grid.project_navigation()
    project_ms = (time.perf_counter() - project_started) * 1000.0
    return stats, projection, float(integration_ms), float(project_ms)


def _profile_row(backend: str, stats, integration_ms: float, project_ms: float) -> dict[str, object]:
    data = stats.to_dict()
    return {
        "backend_requested": str(backend),
        "backend_actual": str(stats.integration_backend),
        "numba_threads": int(stats.voxel_integrate_backend_thread_count),
        "numba_effective_threads": int(getattr(stats, "voxel_integrate_backend_effective_thread_count", stats.voxel_integrate_backend_thread_count)),
        "numba_requested_threads": int(getattr(stats, "voxel_integrate_numba_requested_thread_count", stats.voxel_integrate_backend_thread_count)),
        "numba_threads_mode": str(getattr(stats, "voxel_integrate_numba_threads_mode", "manual")),
        "numba_threading_layer": str(getattr(stats, "voxel_numba_threading_layer", "unknown")),
        "rays": int(stats.depth_rays_integrated),
        "free_events": int(getattr(stats, "voxel_integrate_total_samples", 0)),
        "sensor_events": int(getattr(stats, "voxel_integrate_total_sensor_events", 0)),
        "occ_events": int(getattr(stats, "voxel_integrate_total_occ_events", 0)),
        "pass1_ms": float(getattr(stats, "voxel_integrate_pass1_ms", 0.0)),
        "pass2_ms": float(getattr(stats, "voxel_integrate_pass2_ms", 0.0)),
        "event_bucket_ms": float(getattr(stats, "voxel_integrate_event_bucket_ms", 0.0)),
        "bucket_free_ms": float(getattr(stats, "voxel_integrate_bucket_free_ms", 0.0)),
        "bucket_occ_ms": float(getattr(stats, "voxel_integrate_bucket_occ_ms", 0.0)),
        "bucket_sensor_ms": float(getattr(stats, "voxel_integrate_bucket_sensor_ms", 0.0)),
        "log_apply_ms": float(getattr(stats, "voxel_integrate_apply_logodds_ms", 0.0)),
        "sensor_apply_ms": float(getattr(stats, "voxel_integrate_apply_sensor_ms", 0.0)),
        "endpoint_column_ms": float(getattr(stats, "voxel_integrate_endpoint_column_ms", 0.0)),
        "changed_scan_ms": float(getattr(stats, "voxel_integrate_changed_scan_ms", 0.0)),
        "refresh_ms": float(stats.refresh_state_ms),
        "refresh_mode": str(getattr(stats, "refresh_mode", "unknown")),
        "project_navigation_ms": float(project_ms),
        "integrate_ms": float(integration_ms),
        "total_ms": float(integration_ms + project_ms),
        "stats": data,
    }


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in (
        "integrate_ms",
        "project_navigation_ms",
        "total_ms",
        "pass1_ms",
        "pass2_ms",
        "event_bucket_ms",
        "bucket_free_ms",
        "bucket_occ_ms",
        "bucket_sensor_ms",
        "log_apply_ms",
        "sensor_apply_ms",
        "endpoint_column_ms",
        "changed_scan_ms",
        "refresh_ms",
    ):
        values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
        out[key + "_mean"] = float(np.mean(values)) if values.size else 0.0
        out[key + "_max"] = float(np.max(values)) if values.size else 0.0
        out[key + "_min"] = float(np.min(values)) if values.size else 0.0
    return out


def _print_stage_table(rows: list[dict[str, object]]) -> None:
    keys = (
        "integrate_ms",
        "project_navigation_ms",
        "total_ms",
        "pass1_ms",
        "pass2_ms",
        "event_bucket_ms",
        "bucket_free_ms",
        "bucket_occ_ms",
        "bucket_sensor_ms",
        "log_apply_ms",
        "sensor_apply_ms",
        "endpoint_column_ms",
        "changed_scan_ms",
        "refresh_ms",
    )
    summary = _summarize_rows(rows)
    print("stage                 mean_ms    min_ms    max_ms")
    for key in keys:
        print("%-21s %8.3f %8.3f %8.3f" % (key, summary[key + "_mean"], summary[key + "_min"], summary[key + "_max"]))


def _load_benchmark_case(args: argparse.Namespace, threads: int, threads_mode: str) -> tuple[VoxelOccupancyGrid3D, np.ndarray, np.ndarray, float]:
    if args.snapshot:
        return _load_snapshot_case(args, threads, threads_mode)
    grid = _make_grid(args.backend, args.device, int(args.size_cells), int(args.chunk_rays), int(threads), str(threads_mode))
    origin = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    points = _synthetic_points(int(args.rays))
    return grid, origin, points, 0.0


def _make_grid(backend: str, device: str, size_cells: int, chunk_rays: int, threads: int = 28, threads_mode: str = "manual") -> VoxelOccupancyGrid3D:
    resolution = 0.05
    half = float(size_cells) * resolution * 0.5
    info = MapInfo(
        resolution_m=resolution,
        min_x=-half,
        max_x=half,
        min_y=-half,
        max_y=half,
        width=int(size_cells),
        height=int(size_cells),
    )
    cfg = VoxelOccupancyGridConfig(
        z_min_m=0.0,
        z_max_m=3.2,
        z_resolution_m=0.05,
        integration_backend=str(backend),
        cuda_device=str(device),
        cuda_chunk_rays=int(chunk_rays),
        cuda_ray_step_voxels=1.00,
        cuda_max_samples_per_ray=320,
        cpu_numba_threads=int(threads),
        cpu_numba_threads_mode=str(threads_mode),
        cpu_numba_autotune_candidates=(2, 4, 8, 14, 28),
        cpu_numba_chunk_rays=int(chunk_rays),
        cpu_numba_max_samples_per_ray=320,
        python_debug_backend_allowed=(str(backend) == "python_debug"),
    )
    grid = VoxelOccupancyGrid3D.zeros((int(size_cells), int(size_cells)), info, cfg)
    grid.active_z_min_m = 0.10
    grid.active_z_max_m = 2.00
    return grid


def _load_snapshot_case(args: argparse.Namespace, threads: int, threads_mode: str) -> tuple[VoxelOccupancyGrid3D, np.ndarray, np.ndarray, float]:
    path = Path(str(args.snapshot)).expanduser()
    if not path.exists():
        raise SystemExit("snapshot not found: %s" % str(path))
    data = np.load(path, allow_pickle=True)
    point_key = _first_npz_key(data, ("points_world", "depth_points_world", "voxel_points_world", "roomseg_points_world"))
    if point_key is None:
        raise SystemExit("snapshot %s lacks points_world/depth_points_world; refusing synthetic fallback" % str(path))
    origin_key = _first_npz_key(data, ("camera_origin_world", "camera_origin", "camera_position_world"))
    if origin_key is None:
        raise SystemExit("snapshot %s lacks camera_origin_world; refusing synthetic fallback" % str(path))
    points = np.asarray(data[point_key], dtype=np.float32).reshape(-1, 3)
    origin = np.asarray(data[origin_key], dtype=np.float32).reshape(-1)[:3]
    floor_z = float(np.asarray(data["floor_z"]).reshape(-1)[0]) if "floor_z" in data.files else 0.0
    state_key = _first_npz_key(data, ("voxel_state", "state", "voxel_state_zyx"))
    if state_key is not None:
        state = np.asarray(data[state_key], dtype=np.uint8)
        if state.ndim != 3:
            raise SystemExit("snapshot voxel_state must be z,y,x")
        shape = (int(state.shape[1]), int(state.shape[2]))
        z_max = float(state.shape[0]) * float(np.asarray(data["z_resolution_m"]).reshape(-1)[0]) if "z_resolution_m" in data.files else 3.2
    else:
        shape = (int(args.size_cells), int(args.size_cells))
        z_max = 3.2
        state = None
    resolution = float(np.asarray(data["resolution_m"]).reshape(-1)[0]) if "resolution_m" in data.files else 0.05
    min_x = float(np.asarray(data["min_x"]).reshape(-1)[0]) if "min_x" in data.files else -0.5 * shape[1] * resolution
    max_x = float(np.asarray(data["max_x"]).reshape(-1)[0]) if "max_x" in data.files else min_x + shape[1] * resolution
    max_y = float(np.asarray(data["max_y"]).reshape(-1)[0]) if "max_y" in data.files else 0.5 * shape[0] * resolution
    min_y = float(np.asarray(data["min_y"]).reshape(-1)[0]) if "min_y" in data.files else max_y - shape[0] * resolution
    info = MapInfo(resolution_m=resolution, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, width=shape[1], height=shape[0])
    cfg = VoxelOccupancyGridConfig(
        z_min_m=float(np.asarray(data["z_min_m"]).reshape(-1)[0]) if "z_min_m" in data.files else 0.0,
        z_max_m=z_max,
        z_resolution_m=float(np.asarray(data["z_resolution_m"]).reshape(-1)[0]) if "z_resolution_m" in data.files else 0.05,
        integration_backend=str(args.backend),
        cuda_device=str(args.device),
        cpu_numba_threads=int(threads),
        cpu_numba_threads_mode=str(threads_mode),
        cpu_numba_chunk_rays=int(args.chunk_rays),
        python_debug_backend_allowed=(str(args.backend) == "python_debug"),
    )
    grid = VoxelOccupancyGrid3D.zeros(shape, info, cfg)
    if state is not None and state.shape == grid.state.shape:
        grid.state[...] = state
    if "voxel_log_odds" in data.files and np.asarray(data["voxel_log_odds"]).shape == grid.log_odds.shape:
        grid.log_odds[...] = np.asarray(data["voxel_log_odds"], dtype=np.int16)
    if "voxel_sensor_range_count" in data.files and np.asarray(data["voxel_sensor_range_count"]).shape == grid.sensor_range_count.shape:
        grid.sensor_range_count[...] = np.asarray(data["voxel_sensor_range_count"], dtype=np.uint8)
    grid.active_z_min_m = float(np.asarray(data["active_z_min_m"]).reshape(-1)[0]) if "active_z_min_m" in data.files else 0.10
    grid.active_z_max_m = float(np.asarray(data["active_z_max_m"]).reshape(-1)[0]) if "active_z_max_m" in data.files else 2.00
    return grid, origin, points, floor_z


def _first_npz_key(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data.files:
            return key
    return None


def _synthetic_points(count: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    angle = rng.uniform(-1.2, 1.2, size=int(count)).astype(np.float32)
    dist = rng.uniform(0.6, 5.5, size=int(count)).astype(np.float32)
    height = rng.uniform(0.05, 1.8, size=int(count)).astype(np.float32)
    x = np.cos(angle) * dist
    y = np.sin(angle) * dist
    return np.stack([x, y, height], axis=1).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
