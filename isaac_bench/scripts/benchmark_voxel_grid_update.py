from __future__ import annotations

import argparse
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
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda_torch", "cpu_vectorized", "python_debug"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rays", type=int, default=76000)
    parser.add_argument("--size-cells", type=int, default=240)
    parser.add_argument("--chunk-rays", type=int, default=8192)
    args = parser.parse_args(argv)

    if args.backend == "cuda_torch" and not VoxelCudaBackend.is_available(args.device):
        print("voxel_integration_backend=cuda_torch unavailable=true python_debug_backend_used=false")
        return 0

    grid = _make_grid(args.backend, args.device, int(args.size_cells), int(args.chunk_rays))
    origin = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    points = _synthetic_points(int(args.rays))

    started = time.perf_counter()
    stats = grid.integrate_depth_points(camera_origin_world=origin, points_world=points, floor_z=0.0)
    integration_ms = (time.perf_counter() - started) * 1000.0
    project_started = time.perf_counter()
    projection = grid.project_navigation()
    project_ms = (time.perf_counter() - project_started) * 1000.0

    print(
        "backend=%s rays=%d free_updates=%d occ_updates=%d integrate=%.3fms projection=%.3fms"
        % (
            str(stats.integration_backend),
            int(stats.depth_rays_integrated),
            int(stats.free_update_count),
            int(stats.occupied_update_count),
            float(integration_ms),
            float(project_ms),
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
    return 0


def _make_grid(backend: str, device: str, size_cells: int, chunk_rays: int) -> VoxelOccupancyGrid3D:
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
        python_debug_backend_allowed=(str(backend) == "python_debug"),
    )
    grid = VoxelOccupancyGrid3D.zeros((int(size_cells), int(size_cells)), info, cfg)
    grid.active_z_min_m = 0.10
    grid.active_z_max_m = 2.00
    return grid


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
