from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo


VOXEL_UNKNOWN = np.uint8(0)
VOXEL_FREE = np.uint8(1)
VOXEL_OCCUPIED = np.uint8(2)
VOXEL_CONFLICT = np.uint8(3)


@dataclass
class VoxelOccupancyGridConfig:
    enabled: bool = True
    z_min_m: float = 0.00
    z_max_m: float = 3.20
    z_resolution_m: float = 0.05
    active_z_min_m: float = 0.10
    active_z_max_mode: str = "ceiling_90pct"
    active_z_max_ceiling_ratio: float = 0.90
    active_z_max_fallback_m: float = 2.00
    integration_backend: str = "cpu_vectorized"
    cuda_device: str = "cuda:0"
    cuda_chunk_rays: int = 8192
    cuda_ray_step_voxels: float = 1.00
    cuda_max_samples_per_ray: int = 320
    cuda_keep_logodds_on_device: bool = True
    python_debug_backend_allowed: bool = False
    depth_stride_px: int = 2
    depth_min_m: float = 0.20
    depth_max_m: float = 5.00
    free_logodds_delta: int = -1
    occupied_logodds_delta: int = 3
    free_logodds_threshold: int = -1
    occupied_logodds_threshold: int = 1
    logodds_min: int = -20
    logodds_max: int = 20
    mark_endpoint_occupied: bool = True
    free_excludes_endpoint: bool = True
    free_excludes_last_n_voxels_before_endpoint: int = 0
    endpoint_splat_xy_radius_cells: int = 0
    endpoint_splat_z_radius_cells: int = 0
    conflict_enabled: bool = False
    conflict_margin: int = 0
    voxel_grid_drives_navigation: bool = True
    sensor_range_tracking_enabled: bool = True
    sensor_range_count_delta: int = 1
    sensor_range_count_max: int = 255
    sensor_range_count_threshold: int = 1
    sensor_range_mark_endpoint_column_enabled: bool = True
    sensor_range_endpoint_column_xy_radius_cells: int = 0
    sensor_range_mark_active_z_only: bool = True
    sensor_range_mark_ray_samples_enabled: bool = True
    sensor_range_behind_endpoint_margin_m: float = 0.00
    sensor_range_count_decay_per_update: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VoxelOccupancyGridConfig":
        if isinstance(data, cls):
            cfg = data
            for key, value in overrides.items():
                if value is not None and hasattr(cfg, key):
                    setattr(cfg, key, value)
            return cfg
        raw = dict(data or {})
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class NavigationProjection:
    free: np.ndarray
    occupied: np.ndarray
    observed: np.ndarray
    unknown: np.ndarray
    debug: dict[str, object] = field(default_factory=dict)


@dataclass
class NavigationProjectionConfig:
    obstacle_z_min_m: float = 0.20
    obstacle_z_max_m: float = 0.90
    free_z_min_m: float = 0.10
    free_z_max_m: float = 0.90
    min_free_voxels: int = 1
    occupied_any_voxel_wins: bool = True
    occupied_use_endpoint_hysteresis: bool = True
    occupied_endpoint_count_threshold: int = 1
    occupied_endpoint_decay_per_free_ray: int = 1
    occupied_endpoint_increment: int = 2
    occupied_endpoint_xy_splat_radius_cells: int = 1
    occupied_endpoint_z_splat_radius_cells: int = 0
    occupied_close_radius_cells: int = 1
    occupied_fill_small_holes_max_area_cells: int = 4
    occupied_priority_over_free: bool = True
    unknown_preserve_when_no_observation: bool = True
    debug_navigation_projection_layers: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "NavigationProjectionConfig":
        raw = dict(data or {})
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VoxelIntegrationStats:
    depth_rays_integrated: int = 0
    skipped_empty_rays: int = 0
    free_update_count: int = 0
    occupied_update_count: int = 0
    integration_backend: str = "none"
    python_debug_backend_used: bool = False
    integrate_total_ms: float = 0.0
    ray_sample_ms: float = 0.0
    unique_ms: float = 0.0
    scatter_ms: float = 0.0
    refresh_state_ms: float = 0.0
    refresh_mode: str = "none"
    refresh_changed_voxels: int = 0
    cuda_chunk_rays: int = 0
    cuda_max_samples_per_ray: int = 0
    sensor_range_update_count: int = 0
    sensor_range_decay_applied: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "voxel_depth_rays_integrated": int(self.depth_rays_integrated),
            "voxel_depth_rays_skipped_empty": int(self.skipped_empty_rays),
            "voxel_free_update_count": int(self.free_update_count),
            "voxel_occupied_update_count": int(self.occupied_update_count),
            "voxel_integration_backend": str(self.integration_backend),
            "voxel_python_debug_backend_used": bool(self.python_debug_backend_used),
            "python_debug_backend_used": bool(self.python_debug_backend_used),
            "voxel_integrate_total_ms": float(self.integrate_total_ms),
            "voxel_integrate_ray_sample_ms": float(self.ray_sample_ms),
            "voxel_integrate_unique_ms": float(self.unique_ms),
            "voxel_integrate_scatter_ms": float(self.scatter_ms),
            "voxel_refresh_state_ms": float(self.refresh_state_ms),
            "voxel_refresh_mode": str(self.refresh_mode),
            "voxel_refresh_changed_voxels": int(self.refresh_changed_voxels),
            "voxel_cuda_chunk_rays": int(self.cuda_chunk_rays),
            "voxel_cuda_max_samples_per_ray": int(self.cuda_max_samples_per_ray),
            "voxel_sensor_range_update_count": int(self.sensor_range_update_count),
            "voxel_sensor_range_decay_applied": int(self.sensor_range_decay_applied),
        }


@dataclass
class VoxelOccupancyGrid3D:
    log_odds: np.ndarray
    state: np.ndarray
    sensor_range_count: np.ndarray
    z_min_m: float
    z_max_m: float
    z_resolution_m: float
    map_info: MapInfo
    config: VoxelOccupancyGridConfig = field(default_factory=VoxelOccupancyGridConfig)
    active_z_min_m: float = 0.10
    active_z_max_m: float | None = None
    ceiling_height_m: float | None = None
    ceiling_estimate_status: str = "unavailable"
    last_integration_stats: VoxelIntegrationStats = field(default_factory=VoxelIntegrationStats)
    last_navigation_debug: dict[str, object] = field(default_factory=dict)

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        map_info: MapInfo,
        cfg: VoxelOccupancyGridConfig | Mapping[str, object] | None = None,
    ) -> "VoxelOccupancyGrid3D":
        config = cfg if isinstance(cfg, VoxelOccupancyGridConfig) else VoxelOccupancyGridConfig.from_mapping(cfg)
        height, width = int(shape[0]), int(shape[1])
        z_bins = int(math.ceil((float(config.z_max_m) - float(config.z_min_m)) / float(config.z_resolution_m)))
        if z_bins <= 0:
            raise ValueError("voxel grid requires at least one z bin")
        log_odds = np.zeros((z_bins, height, width), dtype=np.int16)
        state = np.zeros((z_bins, height, width), dtype=np.uint8)
        sensor_range_count = np.zeros((z_bins, height, width), dtype=np.uint8)
        return cls(
            log_odds=log_odds,
            state=state,
            sensor_range_count=sensor_range_count,
            z_min_m=float(config.z_min_m),
            z_max_m=float(config.z_max_m),
            z_resolution_m=float(config.z_resolution_m),
            map_info=map_info,
            config=config,
            active_z_min_m=float(config.active_z_min_m),
            active_z_max_m=float(config.active_z_max_fallback_m),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.state.shape[1]), int(self.state.shape[2])

    @property
    def z_bin_count(self) -> int:
        return int(self.state.shape[0])

    @property
    def z_centers_m(self) -> np.ndarray:
        return self.z_min_m + (np.arange(self.z_bin_count, dtype=np.float32) + 0.5) * self.z_resolution_m

    def reset_shape(self, shape: tuple[int, int], map_info: MapInfo | None = None) -> None:
        if map_info is not None:
            self.map_info = map_info
        height, width = int(shape[0]), int(shape[1])
        self.log_odds = np.zeros((self.z_bin_count, height, width), dtype=np.int16)
        self.state = np.zeros((self.z_bin_count, height, width), dtype=np.uint8)
        self.sensor_range_count = np.zeros((self.z_bin_count, height, width), dtype=np.uint8)
        self.last_integration_stats = VoxelIntegrationStats()
        self.last_navigation_debug = {}

    def z_index_for_height(self, rel_z_m: float) -> int | None:
        zf = (float(rel_z_m) - self.z_min_m) / self.z_resolution_m
        if not np.isfinite(zf):
            return None
        zi = int(math.floor(zf))
        if zi < 0 or zi >= self.z_bin_count:
            return None
        return zi

    def voxel_center_z(self, z_idx: int) -> float:
        return float(self.z_min_m + (int(z_idx) + 0.5) * self.z_resolution_m)

    def world_to_voxel_float(self, point_world_xyz: Sequence[float], floor_z: float) -> tuple[float, float, float]:
        point = np.asarray(point_world_xyz, dtype=np.float64).reshape(-1)
        if point.size < 3:
            raise ValueError("point_world_xyz must have at least 3 values")
        z_float = (float(point[2]) - float(floor_z) - self.z_min_m) / self.z_resolution_m
        row_float = (float(self.map_info.max_y) - float(point[1])) / float(self.map_info.resolution_m)
        col_float = (float(point[0]) - float(self.map_info.min_x)) / float(self.map_info.resolution_m)
        return float(z_float), float(row_float), float(col_float)

    def ray_voxels_3d(
        self,
        origin_world_xyz: Sequence[float],
        endpoint_world_xyz: Sequence[float],
        *,
        floor_z: float,
        include_endpoint: bool = True,
    ) -> list[tuple[int, int, int]]:
        p0 = np.asarray(self.world_to_voxel_float(origin_world_xyz, floor_z), dtype=np.float64)
        p1 = np.asarray(self.world_to_voxel_float(endpoint_world_xyz, floor_z), dtype=np.float64)
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            return []
        clipped = self._clip_segment_to_volume(p0, p1)
        if clipped is None:
            return []
        p0, p1 = clipped
        cells = self._dda_voxels(p0, p1)
        if not include_endpoint and cells:
            return cells[:-1]
        return cells

    def integrate_depth_points(
        self,
        *,
        camera_origin_world: Sequence[float],
        points_world: np.ndarray,
        depths_m: np.ndarray | None = None,
        floor_z: float,
        valid_mask: np.ndarray | None = None,
    ) -> VoxelIntegrationStats:
        started_at = time.perf_counter()
        stats = VoxelIntegrationStats()
        if not bool(self.config.enabled):
            stats.integration_backend = "disabled"
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            self.last_integration_stats = stats
            return stats
        sensor_range_decay_applied = self.decay_sensor_range_count()
        stats.sensor_range_decay_applied = int(sensor_range_decay_applied)
        points = np.asarray(points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or points.size == 0:
            stats.integration_backend = self._resolve_integration_backend()
            stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
            self.last_integration_stats = stats
            return stats
        if valid_mask is None:
            valid = np.ones(points.shape[0], dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
            if valid.size != points.shape[0]:
                raise ValueError("valid_mask must match points_world length")
        backend = self._resolve_integration_backend()
        if backend == "cuda_torch":
            from isaac_bench.mapping.voxel_cuda_backend import VoxelCudaBackend

            stats = VoxelCudaBackend.integrate(
                self,
                camera_origin_world=camera_origin_world,
                points_world=points,
                floor_z=float(floor_z),
                valid_mask=valid,
            )
        elif backend in {"cpu_vectorized", "cpu_numba"}:
            from isaac_bench.mapping.voxel_cpu_fast_backend import VoxelCpuVectorizedBackend

            stats = VoxelCpuVectorizedBackend.integrate(
                self,
                camera_origin_world=camera_origin_world,
                points_world=points,
                floor_z=float(floor_z),
                valid_mask=valid,
                backend_name=backend,
            )
        elif backend == "python_debug":
            if not bool(self.config.python_debug_backend_allowed):
                raise RuntimeError("Python voxel DDA backend is debug-only and disabled for runtime")
            stats = self._integrate_depth_points_python_debug(
                camera_origin_world=camera_origin_world,
                points_world=points,
                depths_m=depths_m,
                floor_z=float(floor_z),
                valid_mask=valid,
            )
        else:
            raise RuntimeError("unsupported voxel integration backend: %s" % backend)
        stats.sensor_range_decay_applied = int(sensor_range_decay_applied)
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        self.last_integration_stats = stats
        return stats

    def _resolve_integration_backend(self) -> str:
        requested = str(getattr(self.config, "integration_backend", "auto") or "auto").strip().lower()
        valid = {"auto", "cuda_torch", "cpu_numba", "cpu_vectorized", "python_debug"}
        if requested not in valid:
            raise ValueError("unsupported voxel integration backend: %s" % requested)
        if requested == "auto":
            try:
                from isaac_bench.mapping.voxel_cuda_backend import VoxelCudaBackend

                if VoxelCudaBackend.is_available(str(self.config.cuda_device)):
                    return "cuda_torch"
            except Exception:
                pass
            try:
                from isaac_bench.mapping.voxel_cpu_fast_backend import numba_available

                if numba_available():
                    return "cpu_numba"
            except Exception:
                pass
            return "cpu_vectorized"
        if requested == "cuda_torch":
            from isaac_bench.mapping.voxel_cuda_backend import VoxelCudaBackend

            if not VoxelCudaBackend.is_available(str(self.config.cuda_device)):
                raise RuntimeError("cuda_torch voxel backend requested but torch CUDA is unavailable")
            return requested
        if requested == "cpu_numba":
            from isaac_bench.mapping.voxel_cpu_fast_backend import numba_available

            if not numba_available():
                return "cpu_vectorized"
            return requested
        return requested

    def _integrate_depth_points_python_debug(
        self,
        *,
        camera_origin_world: Sequence[float],
        points_world: np.ndarray,
        depths_m: np.ndarray | None = None,
        floor_z: float,
        valid_mask: np.ndarray | None = None,
    ) -> VoxelIntegrationStats:
        _ = depths_m
        started_at = time.perf_counter()
        stats = VoxelIntegrationStats(integration_backend="python_debug", python_debug_backend_used=True)
        points = np.asarray(points_world, dtype=np.float32)
        if valid_mask is None:
            valid = np.ones(points.shape[0], dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        origin = np.asarray(camera_origin_world, dtype=np.float32).reshape(-1)[:3]
        free_voxels: list[tuple[int, int, int]] = []
        occupied_voxels: list[tuple[int, int, int]] = []
        sensor_range_voxels: list[tuple[int, int, int]] = []
        flush_threshold = 250_000
        exclude_n = max(0, int(self.config.free_excludes_last_n_voxels_before_endpoint))
        sensor_enabled = bool(getattr(self.config, "sensor_range_tracking_enabled", True))
        sensor_ray_enabled = sensor_enabled and bool(getattr(self.config, "sensor_range_mark_ray_samples_enabled", True))
        sensor_endpoint_enabled = sensor_enabled and bool(getattr(self.config, "sensor_range_mark_endpoint_column_enabled", True))
        for point in points[valid]:
            voxels = self.ray_voxels_3d(origin, point[:3], floor_z=float(floor_z), include_endpoint=True)
            if not voxels:
                stats.skipped_empty_rays += 1
                continue
            stats.depth_rays_integrated += 1
            if sensor_ray_enabled:
                sensor_range_voxels.extend(voxels)
            if sensor_endpoint_enabled:
                sensor_range_voxels.extend(self._sensor_endpoint_column_voxels(voxels[-1]))
            if bool(self.config.mark_endpoint_occupied):
                occupied_voxels.extend(self._endpoint_splat(voxels[-1]))
            if bool(self.config.free_excludes_endpoint):
                free_end = max(0, len(voxels) - 1 - exclude_n)
                free_voxels.extend(voxels[:free_end])
            else:
                free_voxels.extend(voxels)
            if len(free_voxels) >= flush_threshold:
                stats.free_update_count += self.mark_free_voxels(free_voxels)
                free_voxels.clear()
            if len(occupied_voxels) >= flush_threshold:
                stats.occupied_update_count += self.mark_occupied_voxels(occupied_voxels)
                occupied_voxels.clear()
            if len(sensor_range_voxels) >= flush_threshold:
                count, _changed = self.mark_sensor_range_voxels_array(sensor_range_voxels)
                stats.sensor_range_update_count += int(count)
                sensor_range_voxels.clear()
        stats.free_update_count += self.mark_free_voxels(free_voxels)
        stats.occupied_update_count += self.mark_occupied_voxels(occupied_voxels)
        count, _changed = self.mark_sensor_range_voxels_array(sensor_range_voxels)
        stats.sensor_range_update_count += int(count)
        refresh_started_at = time.perf_counter()
        self.refresh_state()
        stats.refresh_state_ms = float((time.perf_counter() - refresh_started_at) * 1000.0)
        stats.integrate_total_ms = float((time.perf_counter() - started_at) * 1000.0)
        return stats

    def mark_free_voxels(self, voxels: Iterable[Sequence[int]]) -> int:
        count, _changed = self._add_logodds_array(voxels, int(self.config.free_logodds_delta))
        return int(count)

    def mark_occupied_voxels(self, voxels: Iterable[Sequence[int]]) -> int:
        count, _changed = self._add_logodds_array(voxels, int(self.config.occupied_logodds_delta))
        return int(count)

    def mark_sensor_range_voxels_array(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0, np.zeros(0, dtype=np.int64)
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return 0, np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return 0, np.zeros(0, dtype=np.int64)
        arr = np.unique(arr[valid], axis=0)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        delta = max(0, int(getattr(self.config, "sensor_range_count_delta", 1)))
        max_value = int(np.clip(int(getattr(self.config, "sensor_range_count_max", 255)), 0, 255))
        if delta <= 0 or max_value <= 0:
            return int(arr.shape[0]), self.flat_indices_from_voxels(arr)
        current = self.sensor_range_count[z, r, c].astype(np.uint16) + int(delta)
        self.sensor_range_count[z, r, c] = np.minimum(current, int(max_value)).astype(np.uint8)
        return int(arr.shape[0]), self.flat_indices_from_voxels(arr)

    def mark_sensor_range_flat_indices(self, flat_indices: np.ndarray | Iterable[int]) -> int:
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0
        idx = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return 0
        idx = np.unique(idx[(idx >= 0) & (idx < self.sensor_range_count.size)])
        if idx.size == 0:
            return 0
        delta = max(0, int(getattr(self.config, "sensor_range_count_delta", 1)))
        max_value = int(np.clip(int(getattr(self.config, "sensor_range_count_max", 255)), 0, 255))
        if delta <= 0 or max_value <= 0:
            return int(idx.size)
        flat = self.sensor_range_count.reshape(-1)
        updated = flat[idx].astype(np.uint16) + int(delta)
        flat[idx] = np.minimum(updated, int(max_value)).astype(np.uint8)
        return int(idx.size)

    def decay_sensor_range_count(self) -> int:
        if not bool(getattr(self.config, "sensor_range_tracking_enabled", True)):
            return 0
        decay = max(0, int(getattr(self.config, "sensor_range_count_decay_per_update", 0)))
        if decay <= 0:
            return 0
        before = int(np.count_nonzero(self.sensor_range_count))
        values = self.sensor_range_count.astype(np.int16) - int(decay)
        self.sensor_range_count[:, :, :] = np.maximum(values, 0).astype(np.uint8)
        return int(before - np.count_nonzero(self.sensor_range_count))

    def mark_free_voxels_array(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        return self._add_logodds_array(voxels, int(self.config.free_logodds_delta))

    def mark_occupied_voxels_array(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        return self._add_logodds_array(voxels, int(self.config.occupied_logodds_delta))

    def force_free_voxels(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> tuple[int, np.ndarray]:
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return 0, np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return 0, np.zeros(0, dtype=np.int64)
        arr = np.unique(arr[valid], axis=0)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        value = min(int(self.config.free_logodds_threshold), int(self.config.free_logodds_delta))
        self.log_odds[z, r, c] = np.minimum(self.log_odds[z, r, c].astype(np.int32), int(value)).astype(np.int16)
        return int(arr.shape[0]), self.flat_indices_from_voxels(arr)

    def refresh_state(self) -> None:
        self.state.fill(int(VOXEL_UNKNOWN))
        self.state[self.log_odds <= int(self.config.free_logodds_threshold)] = int(VOXEL_FREE)
        self.state[self.log_odds >= int(self.config.occupied_logodds_threshold)] = int(VOXEL_OCCUPIED)

    def refresh_state_indices(self, flat_indices: np.ndarray | Iterable[int]) -> int:
        idx = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return 0
        idx = np.unique(idx[(idx >= 0) & (idx < self.state.size)])
        if idx.size == 0:
            return 0
        log = self.log_odds.reshape(-1)
        state = self.state.reshape(-1)
        state[idx] = int(VOXEL_UNKNOWN)
        free = log[idx] <= int(self.config.free_logodds_threshold)
        occupied = log[idx] >= int(self.config.occupied_logodds_threshold)
        state[idx[free]] = int(VOXEL_FREE)
        state[idx[occupied]] = int(VOXEL_OCCUPIED)
        return int(idx.size)

    def flat_indices_from_voxels(self, voxels: np.ndarray | Iterable[Sequence[int]]) -> np.ndarray:
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return np.zeros(0, dtype=np.int64)
        z = z[valid]
        r = r[valid]
        c = c[valid]
        return z.astype(np.int64) * int(self.state.shape[1] * self.state.shape[2]) + r.astype(np.int64) * int(self.state.shape[2]) + c.astype(np.int64)

    def set_active_z_from_ceiling(self, ceiling_height_m: float | None, status: str = "unknown") -> None:
        self.ceiling_height_m = None if ceiling_height_m is None or not np.isfinite(ceiling_height_m) else float(ceiling_height_m)
        self.ceiling_estimate_status = str(status or "unknown")
        if str(self.config.active_z_max_mode).strip().lower() == "ceiling_90pct" and self.ceiling_height_m is not None:
            active = float(self.ceiling_height_m) * float(self.config.active_z_max_ceiling_ratio)
        else:
            active = float(self.config.active_z_max_fallback_m)
        self.active_z_min_m = float(self.config.active_z_min_m)
        self.active_z_max_m = float(np.clip(active, self.active_z_min_m, self.z_max_m))

    def active_z_indices(self, *, z_min_m: float | None = None, z_max_m: float | None = None) -> np.ndarray:
        zmin = self.active_z_min_m if z_min_m is None else float(z_min_m)
        zmax = self.active_z_max_m if z_max_m is None else float(z_max_m)
        if zmax is None:
            zmax = float(self.config.active_z_max_fallback_m)
        z = self.z_centers_m
        idx = np.nonzero((z >= float(zmin)) & (z <= float(zmax)))[0].astype(np.int32)
        if idx.size == 0:
            fallback = self.z_index_for_height(float(zmin))
            if fallback is not None:
                idx = np.asarray([fallback], dtype=np.int32)
        return idx

    def project_navigation(
        self,
        *,
        obstacle_z_min_m: float = 0.20,
        obstacle_z_max_m: float = 0.90,
        free_z_min_m: float = 0.10,
        free_z_max_m: float = 0.90,
        min_free_voxels: int = 1,
        nav_endpoint_count_xy: np.ndarray | None = None,
        config: NavigationProjectionConfig | Mapping[str, object] | None = None,
        occupied_any_voxel_wins: bool = True,
        occupied_use_endpoint_hysteresis: bool = True,
        occupied_endpoint_count_threshold: int = 1,
        occupied_endpoint_decay_per_free_ray: int = 1,
        occupied_endpoint_increment: int = 2,
        occupied_endpoint_xy_splat_radius_cells: int = 1,
        occupied_endpoint_z_splat_radius_cells: int = 0,
        occupied_close_radius_cells: int = 1,
        occupied_fill_small_holes_max_area_cells: int = 4,
        occupied_priority_over_free: bool = True,
        unknown_preserve_when_no_observation: bool = True,
        debug_navigation_projection_layers: bool = True,
    ) -> NavigationProjection:
        started_at = time.perf_counter()
        cfg = (
            config
            if isinstance(config, NavigationProjectionConfig)
            else NavigationProjectionConfig.from_mapping(
                config,
                obstacle_z_min_m=obstacle_z_min_m,
                obstacle_z_max_m=obstacle_z_max_m,
                free_z_min_m=free_z_min_m,
                free_z_max_m=free_z_max_m,
                min_free_voxels=min_free_voxels,
                occupied_any_voxel_wins=occupied_any_voxel_wins,
                occupied_use_endpoint_hysteresis=occupied_use_endpoint_hysteresis,
                occupied_endpoint_count_threshold=occupied_endpoint_count_threshold,
                occupied_endpoint_decay_per_free_ray=occupied_endpoint_decay_per_free_ray,
                occupied_endpoint_increment=occupied_endpoint_increment,
                occupied_endpoint_xy_splat_radius_cells=occupied_endpoint_xy_splat_radius_cells,
                occupied_endpoint_z_splat_radius_cells=occupied_endpoint_z_splat_radius_cells,
                occupied_close_radius_cells=occupied_close_radius_cells,
                occupied_fill_small_holes_max_area_cells=occupied_fill_small_holes_max_area_cells,
                occupied_priority_over_free=occupied_priority_over_free,
                unknown_preserve_when_no_observation=unknown_preserve_when_no_observation,
                debug_navigation_projection_layers=debug_navigation_projection_layers,
            )
        )
        occ_idx = self.active_z_indices(z_min_m=float(cfg.obstacle_z_min_m), z_max_m=float(cfg.obstacle_z_max_m))
        free_idx = self.active_z_indices(z_min_m=float(cfg.free_z_min_m), z_max_m=float(cfg.free_z_max_m))
        union_idx = np.unique(np.concatenate([occ_idx, free_idx])).astype(np.int32) if occ_idx.size or free_idx.size else np.asarray([], dtype=np.int32)
        if occ_idx.size and bool(cfg.occupied_any_voxel_wins):
            occupied_from_voxel = np.any(self.state[occ_idx] == int(VOXEL_OCCUPIED), axis=0)
        else:
            occupied_from_voxel = np.zeros(self.shape, dtype=bool)
        if free_idx.size:
            free_count = np.sum(self.state[free_idx] == int(VOXEL_FREE), axis=0)
            free_raw = free_count >= max(1, int(cfg.min_free_voxels))
        else:
            free_count = np.zeros(self.shape, dtype=np.int16)
            free_raw = np.zeros(self.shape, dtype=bool)
        if nav_endpoint_count_xy is not None and bool(cfg.occupied_use_endpoint_hysteresis):
            endpoint_count = np.asarray(nav_endpoint_count_xy)
            if endpoint_count.shape != self.shape:
                raise ValueError("nav_endpoint_count_xy shape %s does not match grid shape %s" % (endpoint_count.shape, self.shape))
            occupied_from_endpoint = endpoint_count.astype(np.uint16) >= max(1, int(cfg.occupied_endpoint_count_threshold))
        else:
            occupied_from_endpoint = np.zeros(self.shape, dtype=bool)
        occupied_raw = np.asarray(occupied_from_voxel | occupied_from_endpoint, dtype=bool)
        occupied_closed = occupied_raw
        if int(cfg.occupied_close_radius_cells) > 0:
            occupied_closed = _binary_close_disk(occupied_closed, int(cfg.occupied_close_radius_cells))
        hole_filled_mask = np.zeros(self.shape, dtype=bool)
        if int(cfg.occupied_fill_small_holes_max_area_cells) > 0:
            occupied_closed, hole_filled_mask = _fill_small_false_holes(
                occupied_closed,
                max_area_cells=int(cfg.occupied_fill_small_holes_max_area_cells),
            )
        occupied = np.asarray(occupied_closed, dtype=bool)
        free_suppressed_by_occupied = np.asarray(free_raw, dtype=bool) & occupied
        if bool(cfg.occupied_priority_over_free):
            free = np.asarray(free_raw, dtype=bool) & ~occupied
        else:
            free = np.asarray(free_raw, dtype=bool)
            occupied = occupied & ~free
        if union_idx.size:
            observed_from_voxel = np.any(self.state[union_idx] != int(VOXEL_UNKNOWN), axis=0)
        else:
            observed_from_voxel = np.zeros(self.shape, dtype=bool)
        observed = np.asarray(observed_from_voxel | free | occupied, dtype=bool)
        unknown = ~observed
        free &= ~unknown
        occupied &= ~unknown
        debug = {
            "voxel_nav_obstacle_z_min_m": float(cfg.obstacle_z_min_m),
            "voxel_nav_obstacle_z_max_m": float(cfg.obstacle_z_max_m),
            "voxel_nav_free_z_min_m": float(cfg.free_z_min_m),
            "voxel_nav_free_z_max_m": float(cfg.free_z_max_m),
            "voxel_nav_min_free_voxels": int(cfg.min_free_voxels),
            "voxel_nav_occupied_any_voxel_wins": bool(cfg.occupied_any_voxel_wins),
            "voxel_nav_occupied_use_endpoint_hysteresis": bool(cfg.occupied_use_endpoint_hysteresis),
            "voxel_nav_occupied_endpoint_count_threshold": int(cfg.occupied_endpoint_count_threshold),
            "voxel_nav_occupied_endpoint_decay_per_free_ray": int(cfg.occupied_endpoint_decay_per_free_ray),
            "voxel_nav_occupied_endpoint_increment": int(cfg.occupied_endpoint_increment),
            "voxel_nav_occupied_endpoint_xy_splat_radius_cells": int(cfg.occupied_endpoint_xy_splat_radius_cells),
            "voxel_nav_occupied_close_radius_cells": int(cfg.occupied_close_radius_cells),
            "voxel_nav_occupied_fill_small_holes_max_area_cells": int(cfg.occupied_fill_small_holes_max_area_cells),
            "voxel_nav_obstacle_z_bins": int(occ_idx.size),
            "voxel_nav_free_z_bins": int(free_idx.size),
            "voxel_nav_free_cells": int(np.count_nonzero(free)),
            "voxel_nav_occupied_cells": int(np.count_nonzero(occupied)),
            "voxel_nav_occupied_from_voxel_cells": int(np.count_nonzero(occupied_from_voxel)),
            "voxel_nav_occupied_from_endpoint_cells": int(np.count_nonzero(occupied_from_endpoint)),
            "voxel_nav_occupied_raw_cells": int(np.count_nonzero(occupied_raw)),
            "voxel_nav_occupied_closed_cells": int(np.count_nonzero(occupied_closed)),
            "voxel_nav_free_raw_cells": int(np.count_nonzero(free_raw)),
            "voxel_nav_free_suppressed_by_occupied_cells": int(np.count_nonzero(free_suppressed_by_occupied)),
            "voxel_nav_occupied_hole_filled_cells": int(np.count_nonzero(hole_filled_mask)),
            "voxel_nav_observed_cells": int(np.count_nonzero(observed)),
            "voxel_nav_unknown_cells": int(np.count_nonzero(unknown)),
            "voxel_project_navigation_ms": float((time.perf_counter() - started_at) * 1000.0),
        }
        if bool(cfg.debug_navigation_projection_layers):
            debug.update(
                {
                    "voxel_nav_occupied_from_voxel_xy": occupied_from_voxel.astype(bool),
                    "voxel_nav_occupied_from_endpoint_xy": occupied_from_endpoint.astype(bool),
                    "voxel_nav_occupied_raw_xy": occupied_raw.astype(bool),
                    "voxel_nav_occupied_closed_xy": occupied_closed.astype(bool),
                    "voxel_nav_free_raw_xy": np.asarray(free_raw, dtype=bool),
                    "voxel_nav_free_suppressed_by_occupied_xy": free_suppressed_by_occupied.astype(bool),
                    "voxel_nav_final_free_xy": free.astype(bool),
                    "voxel_nav_final_occupied_xy": occupied.astype(bool),
                    "voxel_nav_final_unknown_xy": unknown.astype(bool),
                }
            )
        self.last_navigation_debug = dict(debug)
        return NavigationProjection(
            free=free.astype(bool),
            occupied=occupied.astype(bool),
            observed=observed.astype(bool),
            unknown=unknown.astype(bool),
            debug=debug,
        )

    def to_debug_dict(self) -> dict[str, object]:
        z_count, height, width = self.state.shape
        debug = {
            "voxel_backend": "voxel_occupancy_door_wall_v29",
            "voxel_grid_enabled": bool(self.config.enabled),
            "voxel_grid_shape_zyx": [int(z_count), int(height), int(width)],
            "voxel_z_min_m": float(self.z_min_m),
            "voxel_z_max_m": float(self.z_max_m),
            "voxel_z_resolution_m": float(self.z_resolution_m),
            "voxel_active_z_min_m": float(self.active_z_min_m),
            "voxel_active_z_max_m": None if self.active_z_max_m is None else float(self.active_z_max_m),
            "voxel_ceiling_height_m": None if self.ceiling_height_m is None else float(self.ceiling_height_m),
            "voxel_ceiling_estimate_status": str(self.ceiling_estimate_status),
            "voxel_active_z_bin_count": int(self.active_z_indices().size),
            "voxel_integration_backend_requested": str(self.config.integration_backend),
            "voxel_python_debug_backend_allowed": bool(self.config.python_debug_backend_allowed),
            "voxel_state_free_count_3d": int(np.count_nonzero(self.state == int(VOXEL_FREE))),
            "voxel_state_occupied_count_3d": int(np.count_nonzero(self.state == int(VOXEL_OCCUPIED))),
            "voxel_state_unknown_count_3d": int(np.count_nonzero(self.state == int(VOXEL_UNKNOWN))),
            "voxel_state_conflict_count_3d": int(np.count_nonzero(self.state == int(VOXEL_CONFLICT))),
            "voxel_sensor_range_tracking_enabled": bool(getattr(self.config, "sensor_range_tracking_enabled", True)),
            "voxel_sensor_range_count_nonzero_3d": int(np.count_nonzero(self.sensor_range_count)),
            "voxel_sensor_range_count_threshold": int(getattr(self.config, "sensor_range_count_threshold", 1)),
            "voxel_sensor_range_endpoint_column_enabled": bool(getattr(self.config, "sensor_range_mark_endpoint_column_enabled", True)),
            "voxel_sensor_range_mark_ray_samples_enabled": bool(getattr(self.config, "sensor_range_mark_ray_samples_enabled", True)),
            "voxel_sensor_range_behind_endpoint_margin_m": float(getattr(self.config, "sensor_range_behind_endpoint_margin_m", 0.0)),
        }
        debug.update(self.last_integration_stats.to_dict())
        debug.update({key: value for key, value in dict(self.last_navigation_debug).items() if not isinstance(value, np.ndarray)})
        return debug

    def _add_logodds(self, voxels: Iterable[Sequence[int]], delta: int) -> int:
        count, _changed = self._add_logodds_array(voxels, int(delta))
        return int(count)

    def _add_logodds_array(self, voxels: np.ndarray | Iterable[Sequence[int]], delta: int) -> tuple[int, np.ndarray]:
        if isinstance(voxels, np.ndarray):
            arr = np.asarray(voxels, dtype=np.int64)
        else:
            arr = np.asarray(list(voxels), dtype=np.int64)
        if arr.size == 0:
            return 0, np.zeros(0, dtype=np.int64)
        arr = arr.reshape(-1, 3)
        z = arr[:, 0]
        r = arr[:, 1]
        c = arr[:, 2]
        valid = (z >= 0) & (z < self.z_bin_count) & (r >= 0) & (r < self.state.shape[1]) & (c >= 0) & (c < self.state.shape[2])
        if not np.any(valid):
            return 0, np.zeros(0, dtype=np.int64)
        z = z[valid]
        r = r[valid]
        c = c[valid]
        np.add.at(self.log_odds, (z, r, c), int(delta))
        np.clip(self.log_odds, int(self.config.logodds_min), int(self.config.logodds_max), out=self.log_odds)
        changed = z.astype(np.int64) * int(self.state.shape[1] * self.state.shape[2]) + r.astype(np.int64) * int(self.state.shape[2]) + c.astype(np.int64)
        return int(z.size), np.unique(changed)

    def _endpoint_splat(self, voxel: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        z0, r0, c0 = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        rz = max(0, int(self.config.endpoint_splat_z_radius_cells))
        rxy = max(0, int(self.config.endpoint_splat_xy_radius_cells))
        out: list[tuple[int, int, int]] = []
        for dz in range(-rz, rz + 1):
            for dr in range(-rxy, rxy + 1):
                for dc in range(-rxy, rxy + 1):
                    z, r, c = z0 + dz, r0 + dr, c0 + dc
                    if 0 <= z < self.z_bin_count and 0 <= r < self.state.shape[1] and 0 <= c < self.state.shape[2]:
                        out.append((z, r, c))
        return out

    def _sensor_endpoint_column_voxels(self, voxel: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        _z0, r0, c0 = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        radius = max(0, int(getattr(self.config, "sensor_range_endpoint_column_xy_radius_cells", 0)))
        if bool(getattr(self.config, "sensor_range_mark_active_z_only", True)):
            z_values = [int(v) for v in self.active_z_indices().tolist()]
        else:
            z_values = list(range(int(self.z_bin_count)))
        out: list[tuple[int, int, int]] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < self.state.shape[1] and 0 <= c < self.state.shape[2]):
                    continue
                out.extend((int(z), int(r), int(c)) for z in z_values)
        return out

    def _clip_segment_to_volume(self, p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        bounds = np.asarray([self.z_bin_count, self.state.shape[1], self.state.shape[2]], dtype=np.float64)
        direction = p1 - p0
        t_min = 0.0
        t_max = 1.0
        eps = 1e-9
        for axis in range(3):
            lower = 0.0
            upper = max(0.0, float(bounds[axis]) - eps)
            if abs(float(direction[axis])) < eps:
                if p0[axis] < lower or p0[axis] > upper:
                    return None
                continue
            inv = 1.0 / float(direction[axis])
            ta = (lower - float(p0[axis])) * inv
            tb = (upper - float(p0[axis])) * inv
            if ta > tb:
                ta, tb = tb, ta
            t_min = max(t_min, ta)
            t_max = min(t_max, tb)
            if t_min > t_max:
                return None
        q0 = p0 + direction * t_min
        q1 = p0 + direction * t_max
        q0 = np.minimum(np.maximum(q0, 0.0), bounds - eps)
        q1 = np.minimum(np.maximum(q1, 0.0), bounds - eps)
        return q0, q1

    def _dda_voxels(self, p0: np.ndarray, p1: np.ndarray) -> list[tuple[int, int, int]]:
        bounds = np.asarray([self.z_bin_count, self.state.shape[1], self.state.shape[2]], dtype=np.int32)
        direction = p1 - p0
        cell = np.floor(p0).astype(np.int32)
        target = np.floor(p1).astype(np.int32)
        cell = np.minimum(np.maximum(cell, 0), bounds - 1)
        target = np.minimum(np.maximum(target, 0), bounds - 1)
        step = np.sign(direction).astype(np.int32)
        t_delta = np.full(3, np.inf, dtype=np.float64)
        t_max = np.full(3, np.inf, dtype=np.float64)
        eps = 1e-12
        for axis in range(3):
            d = float(direction[axis])
            if abs(d) < eps:
                continue
            t_delta[axis] = abs(1.0 / d)
            if step[axis] > 0:
                boundary = math.floor(float(p0[axis])) + 1.0
                t_max[axis] = (boundary - float(p0[axis])) / d
            elif step[axis] < 0:
                boundary = math.floor(float(p0[axis]))
                t_max[axis] = (float(p0[axis]) - boundary) / (-d)
        out: list[tuple[int, int, int]] = []
        max_steps = int(np.sum(bounds) * 4 + 16)
        for _ in range(max_steps):
            if np.any(cell < 0) or np.any(cell >= bounds):
                break
            out.append((int(cell[0]), int(cell[1]), int(cell[2])))
            if np.array_equal(cell, target):
                break
            axis = int(np.argmin(t_max))
            if not np.isfinite(t_max[axis]):
                break
            cell[axis] += int(step[axis])
            t_max[axis] += t_delta[axis]
        return out


def _disk_structure(radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)


def _binary_close_disk(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    try:
        from scipy import ndimage

        return ndimage.binary_closing(src, structure=_disk_structure(radius)).astype(bool)
    except Exception:
        dilated = _dilate_bool(src, radius)
        return ~_dilate_bool(~dilated, radius)


def _fill_small_false_holes(mask: np.ndarray, *, max_area_cells: int) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(mask, dtype=bool)
    max_area = max(0, int(max_area_cells))
    filled = src.copy()
    filled_mask = np.zeros(src.shape, dtype=bool)
    if max_area <= 0 or not np.any(~src):
        return filled, filled_mask
    try:
        from scipy import ndimage

        labels, count = ndimage.label(~src)
        for label in range(1, int(count) + 1):
            component = labels == label
            area = int(np.count_nonzero(component))
            if area == 0 or area > max_area:
                continue
            rows, cols = np.nonzero(component)
            if int(rows.min()) == 0 or int(cols.min()) == 0 or int(rows.max()) == src.shape[0] - 1 or int(cols.max()) == src.shape[1] - 1:
                continue
            filled[component] = True
            filled_mask[component] = True
        return filled, filled_mask
    except Exception:
        visited = np.zeros(src.shape, dtype=bool)
        h, w = src.shape
        inv = ~src
        for row in range(h):
            for col in range(w):
                if visited[row, col] or not inv[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                cells: list[tuple[int, int]] = []
                touches_border = False
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    touches_border = touches_border or rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and inv[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if not touches_border and len(cells) <= max_area:
                    for rr, cc in cells:
                        filled[rr, cc] = True
                        filled_mask[rr, cc] = True
        return filled, filled_mask


def _dilate_bool(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    out = src.copy()
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= radius * radius
    ]
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out
