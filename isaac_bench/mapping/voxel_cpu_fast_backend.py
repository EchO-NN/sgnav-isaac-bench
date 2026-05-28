from __future__ import annotations

import time

import numpy as np

from isaac_bench.mapping.voxel_occupancy_grid import VoxelIntegrationStats


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


class VoxelCpuVectorizedBackend:
    @staticmethod
    def integrate(
        grid,
        *,
        camera_origin_world,
        points_world: np.ndarray,
        floor_z: float,
        valid_mask: np.ndarray,
        backend_name: str = "cpu_vectorized",
    ) -> VoxelIntegrationStats:
        started_at = time.perf_counter()
        stats = VoxelIntegrationStats(integration_backend=str(backend_name), python_debug_backend_used=False)
        points = np.asarray(points_world, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if points.ndim != 2 or points.shape[1] < 3 or valid.size != points.shape[0]:
            return stats

        points = points[valid]
        finite = np.all(np.isfinite(points[:, :3]), axis=1) if points.size else np.zeros(0, dtype=bool)
        points = points[finite]
        if points.size == 0:
            stats.skipped_empty_rays = int(np.count_nonzero(valid))
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            return stats

        origin = _world_to_voxel_float_array(
            grid,
            np.asarray(camera_origin_world, dtype=np.float32).reshape(1, -1)[:, :3],
            float(floor_z),
        )[0]
        endpoints = _world_to_voxel_float_array(grid, points[:, :3], float(floor_z))
        finite_endpoint = np.all(np.isfinite(endpoints), axis=1) & np.all(np.isfinite(origin))
        endpoints = endpoints[finite_endpoint]
        if endpoints.size == 0:
            stats.skipped_empty_rays = int(points.shape[0])
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            return stats

        chunk_rays = max(1, min(int(getattr(grid.config, "cuda_chunk_rays", 8192)), 4096))
        step_voxels = max(0.10, float(getattr(grid.config, "cuda_ray_step_voxels", 0.50)))
        max_samples = max(1, int(getattr(grid.config, "cuda_max_samples_per_ray", 320)))
        stats.cuda_chunk_rays = int(chunk_rays)
        stats.cuda_max_samples_per_ray = int(max_samples)
        total_voxels = int(grid.state.size)
        height, width = int(grid.state.shape[1]), int(grid.state.shape[2])
        z_bins = int(grid.state.shape[0])
        log_flat = grid.log_odds.reshape(-1)
        log_min = int(grid.config.logodds_min)
        log_max = int(grid.config.logodds_max)
        free_delta = int(grid.config.free_logodds_delta)
        occ_delta = int(grid.config.occupied_logodds_delta)
        exclude_n = max(0, int(grid.config.free_excludes_last_n_voxels_before_endpoint))
        changed_chunks: list[np.ndarray] = []

        integrated = 0
        for start in range(0, int(endpoints.shape[0]), int(chunk_rays)):
            end = min(int(endpoints.shape[0]), start + int(chunk_rays))
            p1 = endpoints[start:end]
            sample_started = time.perf_counter()
            delta = p1 - origin[None, :]
            ray_len = np.max(np.abs(delta), axis=1)
            steps = np.ceil(ray_len / step_voxels).astype(np.int32)
            steps = np.clip(steps, 1, max_samples)
            s = np.arange(int(np.max(steps)) + 1, dtype=np.float32)
            valid_step = s[None, :] <= steps[:, None]
            t = s[None, :] / steps[:, None].astype(np.float32)
            coords = origin[None, None, :] + t[:, :, None] * delta[:, None, :]
            vox = np.floor(coords).astype(np.int32)
            z = vox[:, :, 0]
            r = vox[:, :, 1]
            c = vox[:, :, 2]
            in_bounds = (
                valid_step
                & (z >= 0)
                & (z < z_bins)
                & (r >= 0)
                & (r < height)
                & (c >= 0)
                & (c < width)
            )
            ray_has_sample = np.any(in_bounds, axis=1)
            integrated += int(np.count_nonzero(ray_has_sample))
            lin = z.astype(np.int64) * int(height * width) + r.astype(np.int64) * int(width) + c.astype(np.int64)
            ray_id = np.arange(end - start, dtype=np.int64)[:, None]
            combined = ray_id * int(total_voxels) + lin
            stats.ray_sample_ms += float((time.perf_counter() - sample_started) * 1000.0)

            unique_started = time.perf_counter()
            if bool(grid.config.free_excludes_endpoint):
                free_step = s[None, :] < np.maximum(steps - exclude_n, 0)[:, None]
                free_mask = in_bounds & free_step
            else:
                free_mask = in_bounds
            free_combined = combined[free_mask]

            endpoint_vox = np.floor(p1).astype(np.int32)
            endpoint_valid = (
                (endpoint_vox[:, 0] >= 0)
                & (endpoint_vox[:, 0] < z_bins)
                & (endpoint_vox[:, 1] >= 0)
                & (endpoint_vox[:, 1] < height)
                & (endpoint_vox[:, 2] >= 0)
                & (endpoint_vox[:, 2] < width)
            )
            endpoint_lin = (
                endpoint_vox[:, 0].astype(np.int64) * int(height * width)
                + endpoint_vox[:, 1].astype(np.int64) * int(width)
                + endpoint_vox[:, 2].astype(np.int64)
            )
            endpoint_ids = np.arange(end - start, dtype=np.int64)[endpoint_valid]
            endpoint_lin_valid = endpoint_lin[endpoint_valid]
            if bool(grid.config.free_excludes_endpoint) and free_combined.size and endpoint_lin_valid.size:
                endpoint_combined = endpoint_ids * int(total_voxels) + endpoint_lin_valid
                free_combined = free_combined[~np.isin(free_combined, endpoint_combined, assume_unique=False)]
            if free_combined.size:
                free_combined = np.unique(free_combined)
                free_lin = np.asarray(free_combined % int(total_voxels), dtype=np.int64)
                free_unique, free_counts = np.unique(free_lin, return_counts=True)
            else:
                free_unique = np.zeros(0, dtype=np.int64)
                free_counts = np.zeros(0, dtype=np.int32)

            if bool(grid.config.mark_endpoint_occupied) and endpoint_lin_valid.size:
                occ_lin = _endpoint_splat_lines(grid, endpoint_vox[endpoint_valid], height=height, width=width, z_bins=z_bins)
                occ_unique, occ_counts = np.unique(occ_lin, return_counts=True) if occ_lin.size else (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32))
            else:
                occ_unique = np.zeros(0, dtype=np.int64)
                occ_counts = np.zeros(0, dtype=np.int32)
            stats.unique_ms += float((time.perf_counter() - unique_started) * 1000.0)

            scatter_started = time.perf_counter()
            stats.free_update_count += _apply_flat_updates(log_flat, free_unique, free_counts, free_delta, log_min, log_max)
            stats.occupied_update_count += _apply_flat_updates(log_flat, occ_unique, occ_counts, occ_delta, log_min, log_max)
            if free_unique.size:
                changed_chunks.append(np.asarray(free_unique, dtype=np.int64))
            if occ_unique.size:
                changed_chunks.append(np.asarray(occ_unique, dtype=np.int64))
            stats.scatter_ms += float((time.perf_counter() - scatter_started) * 1000.0)

        stats.depth_rays_integrated = int(integrated)
        stats.skipped_empty_rays = max(0, int(endpoints.shape[0]) - int(integrated))
        refresh_started = time.perf_counter()
        if changed_chunks:
            changed = np.unique(np.concatenate(changed_chunks).astype(np.int64, copy=False))
        else:
            changed = np.zeros(0, dtype=np.int64)
        stats.refresh_changed_voxels = int(changed.size)
        if changed.size > int(grid.state.size) * 0.35:
            grid.refresh_state()
            stats.refresh_mode = "full"
        else:
            grid.refresh_state_indices(changed)
            stats.refresh_mode = "changed_indices"
        stats.refresh_state_ms = float((time.perf_counter() - refresh_started) * 1000.0)
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats


def _world_to_voxel_float_array(grid, points_world_xyz: np.ndarray, floor_z: float) -> np.ndarray:
    pts = np.asarray(points_world_xyz, dtype=np.float32)
    out = np.empty((pts.shape[0], 3), dtype=np.float32)
    out[:, 0] = (pts[:, 2] - float(floor_z) - float(grid.z_min_m)) / float(grid.z_resolution_m)
    out[:, 1] = (float(grid.map_info.max_y) - pts[:, 1]) / float(grid.map_info.resolution_m)
    out[:, 2] = (pts[:, 0] - float(grid.map_info.min_x)) / float(grid.map_info.resolution_m)
    return out


def _endpoint_splat_lines(grid, endpoint_vox: np.ndarray, *, height: int, width: int, z_bins: int) -> np.ndarray:
    rz = max(0, int(grid.config.endpoint_splat_z_radius_cells))
    rxy = max(0, int(grid.config.endpoint_splat_xy_radius_cells))
    if endpoint_vox.size == 0:
        return np.zeros(0, dtype=np.int64)
    if rz == 0 and rxy == 0:
        return (
            endpoint_vox[:, 0].astype(np.int64) * int(height * width)
            + endpoint_vox[:, 1].astype(np.int64) * int(width)
            + endpoint_vox[:, 2].astype(np.int64)
        )
    offsets = np.asarray(
        [(dz, dr, dc) for dz in range(-rz, rz + 1) for dr in range(-rxy, rxy + 1) for dc in range(-rxy, rxy + 1)],
        dtype=np.int32,
    )
    splat = endpoint_vox[:, None, :] + offsets[None, :, :]
    z = splat[:, :, 0].reshape(-1)
    r = splat[:, :, 1].reshape(-1)
    c = splat[:, :, 2].reshape(-1)
    valid = (z >= 0) & (z < z_bins) & (r >= 0) & (r < height) & (c >= 0) & (c < width)
    if not np.any(valid):
        return np.zeros(0, dtype=np.int64)
    return z[valid].astype(np.int64) * int(height * width) + r[valid].astype(np.int64) * int(width) + c[valid].astype(np.int64)


def _apply_flat_updates(
    log_flat: np.ndarray,
    indices: np.ndarray,
    counts: np.ndarray,
    delta: int,
    log_min: int,
    log_max: int,
) -> int:
    if indices.size == 0:
        return 0
    idx = np.asarray(indices, dtype=np.int64)
    count = np.asarray(counts, dtype=np.int32)
    values = log_flat[idx].astype(np.int32) + count * int(delta)
    values = np.clip(values, int(log_min), int(log_max)).astype(np.int16)
    log_flat[idx] = values
    return int(np.sum(count, dtype=np.int64))
