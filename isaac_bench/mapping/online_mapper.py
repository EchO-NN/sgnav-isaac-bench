from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from isaac_bench.mapping.grid_map import OnlineGridMap
from isaac_bench.mapping.vertical_profile import VerticalProfileMap
from isaac_bench.sensors.camera_geometry import CameraIntrinsics
from isaac_bench.sensors.depth_backproject import backproject_pixels, transform_points


class OnlineMapper:
    def __init__(
        self,
        size_m: float,
        resolution_m: float,
        depth_max_m: float = 6.0,
        depth_min_m: float = 0.20,
        depth_stride_px: int = 8,
        ray_step_m: float | None = None,
        obstacle_min_height_m: float = 0.20,
        obstacle_max_height_m: float = 0.90,
        free_min_height_m: float = -1.50,
        free_max_height_m: float = 0.10,
        vertical_profile_free_min_height_m: float = 0.10,
        vertical_profile_free_max_height_m: float = 2.50,
        splat_point_threshold: int = 6,
        free_splat_point_threshold: int | None = None,
        robot_radius_m: float = 0.10,
        inflation_radius_m: float = 0.0,
    ):
        self.size_m = float(size_m)
        self.resolution_m = float(resolution_m)
        self.depth_max_m = float(depth_max_m)
        self.depth_min_m = float(depth_min_m)
        self.depth_stride_px = max(1, int(depth_stride_px))
        # Kept for old callers. The online depth map now uses grid ray
        # casting, so this is only retained as a compatibility field.
        self.ray_step_m = float(ray_step_m) if ray_step_m is not None else max(self.resolution_m, 0.05)
        self.obstacle_min_height_m = float(obstacle_min_height_m)
        self.obstacle_max_height_m = float(obstacle_max_height_m)
        self.free_min_height_m = float(free_min_height_m)
        self.free_max_height_m = float(free_max_height_m)
        self.vertical_profile_free_min_height_m = float(vertical_profile_free_min_height_m)
        self.vertical_profile_free_max_height_m = float(vertical_profile_free_max_height_m)
        self.splat_point_threshold = max(1, int(splat_point_threshold))
        self.free_splat_point_threshold = (
            max(1, int(free_splat_point_threshold))
            if free_splat_point_threshold is not None
            else 1
        )
        self.robot_radius_m = float(robot_radius_m)
        # Retained for config/CLI compatibility. Runtime traversal inflation is
        # intentionally limited to the robot footprint radius only.
        self.inflation_radius_m = float(inflation_radius_m)
        self.grid = OnlineGridMap.centered(0.0, 0.0, self.size_m, self.resolution_m)
        self.last_debug_stats: dict = {"reason": "not_updated"}
        self.last_nearfield_debug_stats: dict = {"reason": "not_updated"}
        self.last_static_nearfield_debug_stats: dict = {"reason": "not_updated"}
        self.last_timing_stats: dict = {"reason": "not_updated"}
        self.static_nearfield_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.roomseg_static_structural_occupied = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_free_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_obstacle_endpoint_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self._obstacle_endpoint_evidence_increment = 2
        self._free_ray_obstacle_endpoint_decay = 1
        self._roomseg_frame_index = 0
        self._current_roomseg_pose_bin = 0
        self.last_inflated_occupied = np.zeros_like(self.grid.free, dtype=bool)
        self.vertical_profile = VerticalProfileMap.zeros(self.grid.free.shape)
        self._reset_roomseg_ray_evidence()

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self.grid = OnlineGridMap.centered(start_xy[0], start_xy[1], self.size_m, self.resolution_m)
        self.last_debug_stats = {"reason": "reset"}
        self.last_nearfield_debug_stats = {"reason": "reset"}
        self.last_static_nearfield_debug_stats = {"reason": "reset"}
        self.last_timing_stats = {"reason": "reset"}
        self.static_nearfield_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.roomseg_static_structural_occupied = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_free_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_obstacle_endpoint_count = np.zeros_like(self.grid.free, dtype=np.uint16)
        self._roomseg_frame_index = 0
        self._current_roomseg_pose_bin = 0
        self.last_inflated_occupied = np.zeros_like(self.grid.free, dtype=bool)
        self.vertical_profile = VerticalProfileMap.zeros(self.grid.free.shape)
        self._reset_roomseg_ray_evidence()

    def update_simple_radius(self, base_pose_world: Tuple[float, float, float, float], radius_m: float = 1.5) -> OnlineGridMap:
        # Conservative fallback mapping for smoke tests: mark a local disk free.
        rr = int(radius_m / self.grid.map_info.resolution_m)
        center = self.grid.world_to_grid(base_pose_world[0], base_pose_world[1])
        for dr in range(-rr, rr + 1):
            for dc in range(-rr, rr + 1):
                if dr * dr + dc * dc <= rr * rr:
                    self.grid.mark_free(center[0] + dr, center[1] + dc)
        return self.grid

    def update(self, depth: np.ndarray, intr: CameraIntrinsics, base_pose_world: Tuple[float, float, float, float], camera_pose_world: Tuple[float, float, float, float]) -> OnlineGridMap:
        total_started_at = time.perf_counter()
        self._roomseg_frame_index += 1
        self._current_roomseg_pose_bin = _pose_bin(base_pose_world)
        timings: dict[str, float] = {}
        stage_started_at = total_started_at
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
            stage_started_at = time.perf_counter()
            self.last_debug_stats = {"reason": "invalid_depth_shape", "depth_shape": list(depth_arr.shape)}
            self._mark_robot_footprint_free(base_pose_world)
            timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
            self._finish_timing_stats(timings, total_started_at, reason="invalid_depth_shape")
            return self.grid

        stride = self.depth_stride_px
        vs = np.arange(stride // 2, min(depth_arr.shape[0], intr.height), stride, dtype=np.int32)
        us = np.arange(stride // 2, min(depth_arr.shape[1], intr.width), stride, dtype=np.int32)
        if len(vs) == 0 or len(us) == 0:
            timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
            stage_started_at = time.perf_counter()
            self.last_debug_stats = {"reason": "no_sample_pixels", "depth_shape": list(depth_arr.shape)}
            self._mark_robot_footprint_free(base_pose_world)
            timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
            self._finish_timing_stats(timings, total_started_at, reason="no_sample_pixels")
            return self.grid

        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
        sampled_depth = depth_arr[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        valid = np.isfinite(sampled_depth) & (sampled_depth > self.depth_min_m) & (sampled_depth < self.depth_max_m)
        if int(valid.sum()) == 0:
            timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
            stage_started_at = time.perf_counter()
            self.last_debug_stats = {
                "reason": "no_valid_depth",
                "sampled_pixels": int(len(pixels)),
                "depth_shape": list(depth_arr.shape),
            }
            self._mark_robot_footprint_free(base_pose_world)
            timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
            self._finish_timing_stats(timings, total_started_at, reason="no_valid_depth")
            return self.grid

        valid_pixels = pixels[valid]
        valid_depth = sampled_depth[valid]
        timings["depth_prepare_ms"] = _elapsed_ms(stage_started_at)
        stage_started_at = time.perf_counter()
        points_cam = backproject_pixels(depth_arr, valid_pixels, intr)
        points_world = transform_points(points_cam, camera_pose_world)
        floor_z = float(base_pose_world[2])

        rel_z = points_world[:, 2].astype(np.float32) - floor_z
        rows_cols = _world_points_to_grid(points_world, self.grid.map_info)
        in_bounds = (
            (rows_cols[:, 0] >= 0)
            & (rows_cols[:, 0] < self.grid.map_info.height)
            & (rows_cols[:, 1] >= 0)
            & (rows_cols[:, 1] < self.grid.map_info.width)
        )
        timings["depth_project_ms"] = _elapsed_ms(stage_started_at)
        if int(np.count_nonzero(in_bounds)) == 0:
            self.last_debug_stats = {
                "reason": "no_points_in_map_bounds",
                "valid_points": int(len(points_world)),
                "depth_shape": list(depth_arr.shape),
                "depth_m_percentiles": _percentiles(valid_depth),
            }
            self._finish_timing_stats(timings, total_started_at, reason="no_points_in_map_bounds")
            return self.grid

        origin_cell = self.grid.world_to_grid(float(camera_pose_world[0]), float(camera_pose_world[1]))
        if not is_inside_grid(origin_cell[0], origin_cell[1], self.grid.map_info):
            origin_cell = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        if not is_inside_grid(origin_cell[0], origin_cell[1], self.grid.map_info):
            self.last_debug_stats = {
                "reason": "ray_origin_out_of_bounds",
                "valid_points": int(len(points_world)),
                "depth_shape": list(depth_arr.shape),
                "depth_m_percentiles": _percentiles(valid_depth),
                "camera_pose_world": [float(v) for v in camera_pose_world],
                "base_pose_world": [float(v) for v in base_pose_world],
            }
            self._finish_timing_stats(timings, total_started_at, reason="ray_origin_out_of_bounds")
            return self.grid

        in_bounds_pixels = valid_pixels[in_bounds]
        in_bounds_depth = valid_depth[in_bounds]
        rows_cols = rows_cols[in_bounds]
        rel_z = rel_z[in_bounds]
        stage_started_at = time.perf_counter()
        self.vertical_profile.mark_occupied_points(rows_cols, rel_z)
        timings["vertical_profile_occupied_ms"] = _elapsed_ms(stage_started_at)
        obstacle_mask = (rel_z >= self.obstacle_min_height_m) & (rel_z <= self.obstacle_max_height_m)
        free_mask = (rel_z >= self.free_min_height_m) & (rel_z <= self.free_max_height_m)
        ray_clear_mask = (rel_z >= self.free_min_height_m) & (rel_z <= self.obstacle_max_height_m)
        camera_rel_z = float(camera_pose_world[2]) - floor_z
        map_width = int(self.grid.map_info.width)
        stage_started_at = time.perf_counter()
        ray_result = _collect_ray_cast_evidence(
            origin_cell=(int(origin_cell[0]), int(origin_cell[1])),
            endpoints=rows_cols,
            endpoint_rel_z=rel_z,
            endpoint_depth_m=in_bounds_depth,
            endpoint_is_obstacle=obstacle_mask,
            ray_can_clear=ray_clear_mask,
            map_width=map_width,
            depth_min_m=float(self.depth_min_m),
            depth_max_m=float(self.depth_max_m),
            camera_rel_z_m=float(camera_rel_z),
            z_min_m=float(self.vertical_profile_free_min_height_m),
            z_max_m=float(self.vertical_profile_free_max_height_m),
            band_ranges_m=self.vertical_profile.band_ranges_m,
        )
        self._mark_vertical_profile_free_weighted_flat(
            ray_result.free_flat_by_band,
            ray_result.free_weight_by_band,
        )
        self._mark_roomseg_ray_covered_weighted_flat(
            ray_result.roomseg_ray_covered_flat,
            ray_result.roomseg_ray_covered_weight,
        )
        self._mark_roomseg_terminal_wall_flat(
            ray_result.terminal_wall_flat,
            ray_result.terminal_wall_depth_m,
            ray_result.terminal_wall_rel_z_m,
        )
        self._refresh_roomseg_terminal_wall_splat(radius_cells=1)
        timings["ray_cast_ms"] = _elapsed_ms(stage_started_at)

        stage_started_at = time.perf_counter()
        free_unique = _unique_flat_array(ray_result.free_flat)
        free_endpoint_unique = _unique_flat_array(ray_result.free_endpoint_flat)
        free_protected_by_obstacle_endpoint = 0
        stale_obstacle_endpoint_cells_cleared = 0
        free_ray_occupied_cells_skipped = 0
        if free_unique.size:
            free_flat = np.asarray(free_unique, dtype=np.int64)
            occupied_flat_now = self.grid.occupied.reshape(-1)
            occupied_by_ray = np.asarray(occupied_flat_now[free_flat], dtype=np.uint8) > 0
            free_ray_occupied_cells_skipped = int(np.count_nonzero(occupied_by_ray))
            free_flat = free_flat[~occupied_by_ray]

            endpoint_counts = self.depth_obstacle_endpoint_count.reshape(-1)
            endpoint_clear_flat = np.zeros((0,), dtype=np.int64)
            if free_endpoint_unique.size:
                endpoint_free_flat = np.asarray(free_endpoint_unique, dtype=np.int64)
                endpoint_occupied = np.asarray(occupied_flat_now[endpoint_free_flat], dtype=np.uint8) > 0
                endpoint_free_flat = endpoint_free_flat[endpoint_occupied]
                protected = np.asarray(endpoint_counts[endpoint_free_flat], dtype=np.uint16) > 0
                free_protected_by_obstacle_endpoint = int(np.count_nonzero(protected))
                protected_flat = endpoint_free_flat[protected]
                if protected_flat.size:
                    decayed = np.maximum(
                        endpoint_counts[protected_flat].astype(np.int32) - int(self._free_ray_obstacle_endpoint_decay),
                        0,
                    ).astype(np.uint16)
                    endpoint_counts[protected_flat] = decayed
                    endpoint_clear_flat = protected_flat[decayed == 0]
                    stale_obstacle_endpoint_cells_cleared = int(endpoint_clear_flat.size)
            if endpoint_clear_flat.size:
                free_flat = np.concatenate([free_flat, endpoint_clear_flat])
            if free_flat.size:
                free_flat = _unique_flat_array(free_flat)
                rows = free_flat // map_width
                cols = free_flat % map_width
                self.grid.free[rows, cols] = 1
                self.grid.occupied[rows, cols] = 0
                self.grid.observed[rows, cols] = 1
                self.depth_free_mask[rows, cols] = 1
        occupied_unique = _unique_flat_array(ray_result.occupied_flat)
        if occupied_unique.size:
            rows = occupied_unique // map_width
            cols = occupied_unique % map_width
            uint16_max = int(np.iinfo(np.uint16).max)
            flat = np.asarray(occupied_unique, dtype=np.int64)
            current = self.depth_obstacle_endpoint_count.reshape(-1).astype(np.uint32)
            current[flat] = np.minimum(current[flat] + int(self._obstacle_endpoint_evidence_increment), uint16_max)
            self.depth_obstacle_endpoint_count.reshape(-1)[:] = current.astype(np.uint16)
            self.grid.free[rows, cols] = 0
            self.grid.occupied[rows, cols] = 1
            self.grid.observed[rows, cols] = 1
        timings["grid_write_ms"] = _elapsed_ms(stage_started_at)
        stage_started_at = time.perf_counter()
        self._mark_robot_footprint_free(base_pose_world)
        timings["robot_footprint_ms"] = _elapsed_ms(stage_started_at)
        stage_started_at = time.perf_counter()
        self.last_debug_stats = self._build_debug_stats(
            depth_arr=depth_arr,
            valid_depth=in_bounds_depth,
            valid_pixels=in_bounds_pixels,
            rel_z=rel_z,
            obstacle_mask=obstacle_mask,
            free_mask=free_mask,
            occupied_endpoint_cells=int(occupied_unique.size),
            free_ray_cells=int(free_unique.size),
            free_protected_by_obstacle_endpoint=int(free_protected_by_obstacle_endpoint),
            stale_obstacle_endpoint_cells_cleared=int(stale_obstacle_endpoint_cells_cleared),
            free_ray_occupied_cells_skipped=int(free_ray_occupied_cells_skipped),
            obstacle_ray_interior_free_suppressed_cells=int(ray_result.obstacle_ray_interior_free_suppressed_cells),
            ray_count=int(ray_result.nav_ray_count),
            skipped_height_rays=int(ray_result.nav_skipped_height_rays),
            vertical_profile_ray_count=int(ray_result.vertical_profile_ray_count),
            vertical_profile_skipped_height_rays=int(ray_result.vertical_profile_skipped_height_rays),
            ray_cast_backend=str(ray_result.backend),
            ray_unique_endpoint_cells=int(ray_result.unique_endpoint_cells),
            base_pose_world=base_pose_world,
            camera_pose_world=camera_pose_world,
            ray_origin_cell=origin_cell,
        )
        timings["debug_stats_ms"] = _elapsed_ms(stage_started_at)
        self._finish_timing_stats(timings, total_started_at, reason="ok")
        return self.grid

    def update_nearfield_topdown(
        self,
        depth: np.ndarray,
        intr: CameraIntrinsics,
        base_pose_world: Tuple[float, float, float, float],
        camera_pose_world: Tuple[float, float, float, float],
        radius_m: float = 0.75,
        ignore_radius_m: float = 0.10,
        depth_stride_px: int = 3,
        floor_tolerance_m: float = 0.12,
        obstacle_min_height_m: float | None = None,
        obstacle_max_height_m: float | None = None,
        splat_point_threshold: int | None = None,
        free_splat_point_threshold: int | None = None,
    ) -> dict:
        """Fuse a downward near-field depth camera into the 2D map.

        The overhead camera is only used to fill the ground blind spot around
        the robot. It is treated as a top-down depth sensor whose optical axis
        points toward -Z; pixels are projected into the robot-local XY plane.
        """
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            stats = {"reason": "invalid_depth_shape", "depth_shape": list(depth_arr.shape)}
            self.last_nearfield_debug_stats = stats
            return stats

        stride = max(1, int(depth_stride_px))
        vs = np.arange(stride // 2, min(depth_arr.shape[0], intr.height), stride, dtype=np.int32)
        us = np.arange(stride // 2, min(depth_arr.shape[1], intr.width), stride, dtype=np.int32)
        if len(vs) == 0 or len(us) == 0:
            stats = {"reason": "no_sample_pixels", "depth_shape": list(depth_arr.shape)}
            self.last_nearfield_debug_stats = stats
            return stats

        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
        sampled_depth = depth_arr[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        cam_z = float(camera_pose_world[2])
        floor_z = float(base_pose_world[2])
        camera_height = max(1e-3, cam_z - floor_z)
        max_depth = camera_height + max(0.20, float(floor_tolerance_m) * 2.0)
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.01) & (sampled_depth < max_depth)
        if int(valid.sum()) == 0:
            stats = {
                "reason": "no_valid_depth",
                "sampled_pixels": int(len(pixels)),
                "camera_height_m": float(camera_height),
                "depth_m_percentiles": _percentiles(sampled_depth),
            }
            self.last_nearfield_debug_stats = stats
            return stats

        pixels = pixels[valid]
        sampled_depth = sampled_depth[valid]
        x_right = (pixels[:, 0] - intr.cx) * sampled_depth / intr.fx
        y_down = (pixels[:, 1] - intr.cy) * sampled_depth / intr.fy
        local_forward = -y_down
        local_left = -x_right
        local_range = np.sqrt(local_forward * local_forward + local_left * local_left)
        radius = max(0.0, float(radius_m))
        ignore_radius = max(0.0, float(ignore_radius_m))
        range_mask = (local_range <= radius) & (local_range >= ignore_radius)
        if int(np.count_nonzero(range_mask)) == 0:
            stats = {
                "reason": "no_points_in_nearfield_radius",
                "valid_points": int(len(sampled_depth)),
                "camera_height_m": float(camera_height),
                "radius_m": float(radius),
                "ignore_radius_m": float(ignore_radius),
            }
            self.last_nearfield_debug_stats = stats
            return stats

        local_forward = local_forward[range_mask]
        local_left = local_left[range_mask]
        local_range = local_range[range_mask]
        sampled_depth = sampled_depth[range_mask]
        rel_height = camera_height - sampled_depth
        yaw = float(base_pose_world[3])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        world_x = float(base_pose_world[0]) + cos_yaw * local_forward - sin_yaw * local_left
        world_y = float(base_pose_world[1]) + sin_yaw * local_forward + cos_yaw * local_left
        rows_cols = np.asarray(
            [self.grid.world_to_grid(float(x), float(y)) for x, y in zip(world_x, world_y)],
            dtype=np.int32,
        )
        in_bounds = (
            (rows_cols[:, 0] >= 0)
            & (rows_cols[:, 0] < self.grid.map_info.height)
            & (rows_cols[:, 1] >= 0)
            & (rows_cols[:, 1] < self.grid.map_info.width)
        )
        if int(np.count_nonzero(in_bounds)) == 0:
            stats = {
                "reason": "no_points_in_map_bounds",
                "valid_points": int(len(sampled_depth)),
                "camera_height_m": float(camera_height),
            }
            self.last_nearfield_debug_stats = stats
            return stats

        rows_cols = rows_cols[in_bounds]
        rel_height = rel_height[in_bounds]
        local_range = local_range[in_bounds]
        floor_tol = max(0.01, float(floor_tolerance_m))
        obs_min = self.obstacle_min_height_m if obstacle_min_height_m is None else float(obstacle_min_height_m)
        obs_max = self.obstacle_max_height_m if obstacle_max_height_m is None else float(obstacle_max_height_m)
        free_mask = np.abs(rel_height) <= floor_tol
        obstacle_mask = (rel_height >= obs_min) & (rel_height <= obs_max)
        obstacle_splats = self._splat_cells(
            rows_cols[obstacle_mask],
            occupied=True,
            point_threshold=self.splat_point_threshold if splat_point_threshold is None else int(splat_point_threshold),
        )
        free_splats = self._splat_cells(
            rows_cols[free_mask],
            occupied=False,
            point_threshold=(
                self.free_splat_point_threshold
                if free_splat_point_threshold is None
                else int(free_splat_point_threshold)
            ),
        )
        self._mark_robot_footprint_free(base_pose_world)
        stats = {
            "reason": "ok",
            "depth_shape": [int(v) for v in depth_arr.shape],
            "camera_pose_world": [float(v) for v in camera_pose_world],
            "camera_height_m": float(camera_height),
            "radius_m": float(radius),
            "ignore_radius_m": float(ignore_radius),
            "valid_points": int(len(rel_height)),
            "free_band_points": int(np.count_nonzero(free_mask)),
            "obstacle_band_points": int(np.count_nonzero(obstacle_mask)),
            "free_splat_cells": int(free_splats),
            "obstacle_splat_cells": int(obstacle_splats),
            "depth_m_percentiles": _percentiles(sampled_depth),
            "rel_height_m_percentiles": _percentiles(rel_height),
            "local_range_m_percentiles": _percentiles(local_range),
            "height_filters_m": {
                "floor_tolerance": float(floor_tol),
                "obstacle_min": float(obs_min),
                "obstacle_max": float(obs_max),
            },
        }
        self.last_nearfield_debug_stats = stats
        return stats

    def update_static_nearfield(
        self,
        static_occupancy: np.ndarray,
        static_navigable: np.ndarray,
        static_map_info: MapInfo,
        base_pose_world: Tuple[float, float, float, float],
        radius_m: float = 1.0,
        static_openings: np.ndarray | None = None,
    ) -> dict:
        """Use the preprocessed map as a local blind-spot fill around the robot."""
        occupancy = np.asarray(static_occupancy).astype(bool)
        navigable = np.asarray(static_navigable).astype(bool)
        openings = None if static_openings is None else np.asarray(static_openings).astype(bool)
        if occupancy.shape != navigable.shape:
            stats = {
                "reason": "shape_mismatch",
                "occupancy_shape": [int(v) for v in occupancy.shape],
                "navigable_shape": [int(v) for v in navigable.shape],
            }
            self.last_static_nearfield_debug_stats = stats
            return stats
        if openings is not None and openings.shape != navigable.shape:
            stats = {
                "reason": "opening_shape_mismatch",
                "opening_shape": [int(v) for v in openings.shape],
                "navigable_shape": [int(v) for v in navigable.shape],
            }
            self.last_static_nearfield_debug_stats = stats
            return stats
        if occupancy.shape != (int(static_map_info.height), int(static_map_info.width)):
            stats = {
                "reason": "map_info_shape_mismatch",
                "map_shape": [int(v) for v in occupancy.shape],
                "map_info_shape": [int(static_map_info.height), int(static_map_info.width)],
            }
            self.last_static_nearfield_debug_stats = stats
            return stats

        radius = max(0.0, float(radius_m))
        radius_cells = int(math.ceil(radius / self.grid.map_info.resolution_m))
        center = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        free_cells = 0
        occupied_cells = 0
        blocked_clearance_cells = 0
        roomseg_static_structural_occupied_cells = 0
        skipped_outside_dynamic = 0
        skipped_outside_static = 0
        sampled_cells = 0
        opening_cells = 0
        for dr, dc in _disk_offsets(radius_cells):
            row, col = int(center[0] + dr), int(center[1] + dc)
            if not is_inside_grid(row, col, self.grid.map_info):
                skipped_outside_dynamic += 1
                continue
            wx, wy = grid_to_world_xy(row, col, self.grid.map_info)
            static_row, static_col = world_xy_to_grid(wx, wy, static_map_info)
            if not is_inside_grid(static_row, static_col, static_map_info):
                skipped_outside_static += 1
                continue
            sampled_cells += 1
            self.static_nearfield_mask[row, col] = 1
            is_opening = openings is not None and bool(openings[static_row, static_col])
            if bool(navigable[static_row, static_col]) or is_opening:
                self.grid.free[row, col] = 1
                self.grid.occupied[row, col] = 0
                self.grid.observed[row, col] = 1
                self.roomseg_static_structural_occupied[row, col] = 0
                free_cells += 1
                if is_opening:
                    opening_cells += 1
            else:
                self.grid.free[row, col] = 0
                self.grid.occupied[row, col] = 0
                if bool(occupancy[static_row, static_col]):
                    occupied_cells += 1
                    self.roomseg_static_structural_occupied[row, col] = 1
                    roomseg_static_structural_occupied_cells += 1
                else:
                    blocked_clearance_cells += 1
                    self.roomseg_static_structural_occupied[row, col] = 0
                self.grid.observed[row, col] = 1

        stats = {
            "reason": "ok",
            "source": "preprocessed_static_map",
            "radius_m": float(radius),
            "radius_cells": int(radius_cells),
            "sampled_cells": int(sampled_cells),
            "free_cells": int(free_cells),
            "occupied_cells": int(occupied_cells),
            "blocked_clearance_cells": int(blocked_clearance_cells),
            "opening_cells": int(opening_cells),
            "roomseg_static_structural_occupied_cells": int(roomseg_static_structural_occupied_cells),
            "skipped_outside_dynamic": int(skipped_outside_dynamic),
            "skipped_outside_static": int(skipped_outside_static),
            "base_pose_world": [float(v) for v in base_pose_world],
        }
        self.last_static_nearfield_debug_stats = stats
        return stats

    def traversible(self, unknown_is_obstacle: bool = True) -> np.ndarray:
        occupied = self.inflated_occupied()
        self.last_inflated_occupied = np.asarray(occupied, dtype=bool)
        if unknown_is_obstacle:
            free = self.grid.free.astype(bool)
        else:
            free = np.ones_like(occupied, dtype=bool)
        return free & ~occupied

    def inflated_occupied(self) -> np.ndarray:
        radius_cells = self._robot_footprint_radius_cells()
        return _dilate_binary(self.grid.occupied.astype(bool), radius_cells)

    def _mark_robot_footprint_free(self, base_pose_world: Tuple[float, float, float, float]) -> None:
        radius_cells = self._robot_footprint_free_radius_cells()
        center = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        profile_cells: List[Tuple[int, int]] = []
        for dr, dc in _disk_offsets(radius_cells):
            row, col = int(center[0] + dr), int(center[1] + dc)
            self._set_free_cell(row, col, mark_mask=self.depth_free_mask)
            if 0 <= row < self.grid.map_info.height and 0 <= col < self.grid.map_info.width:
                profile_cells.append((row, col))
        if profile_cells:
            free_z = min(
                max(float(self.vertical_profile_free_min_height_m), 0.80),
                max(float(self.vertical_profile_free_min_height_m), float(self.vertical_profile_free_max_height_m) - 1e-3),
            )
            self.vertical_profile.mark_free_ray_cells(profile_cells, rel_z_m=free_z)

    def _set_free_cell(
        self,
        row: int,
        col: int,
        mark_mask: np.ndarray | None = None,
        *,
        clear_occupied: bool = False,
    ) -> None:
        if 0 <= int(row) < self.grid.map_info.height and 0 <= int(col) < self.grid.map_info.width:
            rr, cc = int(row), int(col)
            self.grid.free[rr, cc] = 1
            if bool(clear_occupied):
                self.grid.occupied[rr, cc] = 0
            self.grid.observed[rr, cc] = 1
            if mark_mask is not None:
                mark_mask[rr, cc] = 1

    def _set_occupied_cell(self, row: int, col: int) -> None:
        if 0 <= int(row) < self.grid.map_info.height and 0 <= int(col) < self.grid.map_info.width:
            rr, cc = int(row), int(col)
            self.grid.free[rr, cc] = 0
            self.grid.occupied[rr, cc] = 1
            self.grid.observed[rr, cc] = 1

    def _finish_timing_stats(self, timings: dict[str, float], total_started_at: float, *, reason: str) -> None:
        out = {str(key): float(value) for key, value in timings.items()}
        out["update_total_ms"] = _elapsed_ms(total_started_at)
        out["reason"] = str(reason)
        self.last_timing_stats = out

    def _reset_roomseg_ray_evidence(self) -> None:
        shape = self.grid.free.shape
        self.roomseg_ray_covered_count = np.zeros(shape, dtype=np.uint16)
        self.roomseg_free_view_count = np.zeros(shape, dtype=np.uint16)
        self.roomseg_terminal_wall_count = np.zeros(shape, dtype=np.uint16)
        self.roomseg_terminal_wall_view_count = np.zeros(shape, dtype=np.uint16)
        self.roomseg_terminal_wall_height_min = np.full(shape, np.inf, dtype=np.float32)
        self.roomseg_terminal_wall_height_max = np.full(shape, -np.inf, dtype=np.float32)
        self.roomseg_terminal_wall_depth_min = np.full(shape, np.inf, dtype=np.float32)
        self.roomseg_terminal_wall_splat = np.zeros(shape, dtype=np.uint8)
        self.roomseg_last_seen_frame = np.zeros(shape, dtype=np.int32)
        self.roomseg_last_seen_pose_bin = np.zeros(shape, dtype=np.int32)
        self.roomseg_ray_fan_candidate = np.zeros(shape, dtype=np.uint8)

    def roomseg_ray_evidence(self) -> dict[str, np.ndarray]:
        ray_fan_candidate = (
            (np.asarray(self.roomseg_ray_covered_count, dtype=np.uint32) > 0)
            & (np.asarray(self.roomseg_free_view_count, dtype=np.uint32) <= 1)
            & (np.asarray(self.roomseg_terminal_wall_count, dtype=np.uint32) == 0)
        )
        self.roomseg_ray_fan_candidate[:, :] = ray_fan_candidate.astype(np.uint8)
        return {
            "ray_covered_count": np.asarray(self.roomseg_ray_covered_count, dtype=np.uint16),
            "free_view_count": np.asarray(self.roomseg_free_view_count, dtype=np.uint16),
            "terminal_wall_count": np.asarray(self.roomseg_terminal_wall_count, dtype=np.uint16),
            "terminal_wall_view_count": np.asarray(self.roomseg_terminal_wall_view_count, dtype=np.uint16),
            "terminal_wall_height_min": np.asarray(self.roomseg_terminal_wall_height_min, dtype=np.float32),
            "terminal_wall_height_max": np.asarray(self.roomseg_terminal_wall_height_max, dtype=np.float32),
            "terminal_wall_depth_min": np.asarray(self.roomseg_terminal_wall_depth_min, dtype=np.float32),
            "terminal_wall_splat": np.asarray(self.roomseg_terminal_wall_splat, dtype=np.uint8),
            "last_seen_frame": np.asarray(self.roomseg_last_seen_frame, dtype=np.int32),
            "last_seen_pose_bin": np.asarray(self.roomseg_last_seen_pose_bin, dtype=np.int32),
            "ray_fan_candidate": np.asarray(self.roomseg_ray_fan_candidate, dtype=np.uint8),
        }

    def _mark_roomseg_ray_covered_flat(self, flat_values: List[int]) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat = _valid_flat_array(flat_values, total_cells)
        if flat.size == 0:
            return
        uint16_max = int(np.iinfo(np.uint16).max)
        counts = np.bincount(flat, minlength=total_cells).reshape(h, w).astype(np.uint32)
        updated = np.asarray(self.roomseg_ray_covered_count, dtype=np.uint32) + counts
        self.roomseg_ray_covered_count[:, :] = np.minimum(updated, uint16_max).astype(np.uint16)
        self._mark_roomseg_view_flat(flat)

    def _mark_roomseg_ray_covered_weighted_flat(self, flat_values: np.ndarray, weights: np.ndarray) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat, weight = _valid_flat_weight_arrays(flat_values, weights, total_cells)
        if flat.size == 0:
            return
        uint16_max = int(np.iinfo(np.uint16).max)
        counts = np.bincount(flat, weights=weight, minlength=total_cells).reshape(h, w).astype(np.uint32)
        updated = np.asarray(self.roomseg_ray_covered_count, dtype=np.uint32) + counts
        self.roomseg_ray_covered_count[:, :] = np.minimum(updated, uint16_max).astype(np.uint16)
        self._mark_roomseg_view_flat(flat)

    def _mark_roomseg_terminal_wall_cell(
        self,
        row: int,
        col: int,
        *,
        endpoint_depth_m: float,
        endpoint_rel_z_m: float,
    ) -> None:
        rr, cc = int(row), int(col)
        if not is_inside_grid(rr, cc, self.grid.map_info):
            return
        depth = float(endpoint_depth_m)
        rel_z = float(endpoint_rel_z_m)
        if not (np.isfinite(depth) and self.depth_min_m < depth < self.depth_max_m):
            return
        if not (
            float(self.vertical_profile_free_min_height_m)
            <= rel_z
            <= float(self.vertical_profile_free_max_height_m)
        ):
            return
        uint16_max = int(np.iinfo(np.uint16).max)
        self.roomseg_terminal_wall_count[rr, cc] = min(
            uint16_max,
            int(self.roomseg_terminal_wall_count[rr, cc]) + 1,
        )
        self.roomseg_terminal_wall_height_min[rr, cc] = min(
            float(self.roomseg_terminal_wall_height_min[rr, cc]),
            rel_z,
        )
        self.roomseg_terminal_wall_height_max[rr, cc] = max(
            float(self.roomseg_terminal_wall_height_max[rr, cc]),
            rel_z,
        )
        self.roomseg_terminal_wall_depth_min[rr, cc] = min(
            float(self.roomseg_terminal_wall_depth_min[rr, cc]),
            depth,
        )
        self._mark_roomseg_terminal_view_flat(np.asarray([rr * int(self.grid.map_info.width) + cc], dtype=np.int64))

    def _mark_roomseg_terminal_wall_flat(
        self,
        flat_values: np.ndarray,
        endpoint_depth_m: np.ndarray,
        endpoint_rel_z_m: np.ndarray,
    ) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat = np.asarray(flat_values, dtype=np.int64).reshape(-1)
        depth = np.asarray(endpoint_depth_m, dtype=np.float32).reshape(-1)
        rel_z = np.asarray(endpoint_rel_z_m, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return
        if depth.shape != flat.shape or rel_z.shape != flat.shape:
            raise ValueError("terminal wall flat/depth/height arrays must have the same shape")
        valid = (
            (flat >= 0)
            & (flat < total_cells)
            & np.isfinite(depth)
            & (depth > float(self.depth_min_m))
            & (depth < float(self.depth_max_m))
            & np.isfinite(rel_z)
            & (rel_z >= float(self.vertical_profile_free_min_height_m))
            & (rel_z <= float(self.vertical_profile_free_max_height_m))
        )
        if int(np.count_nonzero(valid)) == 0:
            return
        flat = flat[valid]
        depth = depth[valid]
        rel_z = rel_z[valid]
        uint16_max = int(np.iinfo(np.uint16).max)
        counts = np.bincount(flat, minlength=total_cells).astype(np.uint32)
        current = self.roomseg_terminal_wall_count.reshape(-1).astype(np.uint32)
        self.roomseg_terminal_wall_count.reshape(-1)[:] = np.minimum(current + counts, uint16_max).astype(np.uint16)
        self._mark_roomseg_terminal_view_flat(flat)
        np.minimum.at(self.roomseg_terminal_wall_height_min.reshape(-1), flat, rel_z)
        np.maximum.at(self.roomseg_terminal_wall_height_max.reshape(-1), flat, rel_z)
        np.minimum.at(self.roomseg_terminal_wall_depth_min.reshape(-1), flat, depth)

    def _mark_roomseg_view_flat(self, flat_values: np.ndarray) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat = _valid_flat_array(np.asarray(flat_values, dtype=np.int64), total_cells)
        if flat.size == 0:
            return
        unique = np.unique(flat)
        uint16_max = int(np.iinfo(np.uint16).max)
        view_flat = self.roomseg_free_view_count.reshape(-1).astype(np.uint32)
        view_flat[unique] = np.minimum(view_flat[unique] + 1, uint16_max)
        self.roomseg_free_view_count.reshape(-1)[:] = view_flat.astype(np.uint16)
        self.roomseg_last_seen_frame.reshape(-1)[unique] = int(self._roomseg_frame_index)
        self.roomseg_last_seen_pose_bin.reshape(-1)[unique] = int(self._current_roomseg_pose_bin)

    def _mark_roomseg_terminal_view_flat(self, flat_values: np.ndarray) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        flat = _valid_flat_array(np.asarray(flat_values, dtype=np.int64), total_cells)
        if flat.size == 0:
            return
        unique = np.unique(flat)
        uint16_max = int(np.iinfo(np.uint16).max)
        view_flat = self.roomseg_terminal_wall_view_count.reshape(-1).astype(np.uint32)
        view_flat[unique] = np.minimum(view_flat[unique] + 1, uint16_max)
        self.roomseg_terminal_wall_view_count.reshape(-1)[:] = view_flat.astype(np.uint16)
        self.roomseg_last_seen_frame.reshape(-1)[unique] = int(self._roomseg_frame_index)
        self.roomseg_last_seen_pose_bin.reshape(-1)[unique] = int(self._current_roomseg_pose_bin)

    def _refresh_roomseg_terminal_wall_splat(self, radius_cells: int = 1, min_count: int = 1) -> None:
        terminal = np.asarray(self.roomseg_terminal_wall_count, dtype=np.uint32) >= max(1, int(min_count))
        self.roomseg_terminal_wall_splat[:, :] = _dilate_binary(terminal, int(radius_cells)).astype(np.uint8)

    def _mark_vertical_profile_free_flat(self, flat_indices_by_band: List[List[int]]) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        uint16_max = int(np.iinfo(np.uint16).max)
        for band_idx, flat_values in enumerate(flat_indices_by_band):
            flat = _valid_flat_array(flat_values, total_cells)
            if flat.size == 0:
                continue
            counts = np.bincount(flat, minlength=total_cells).reshape(h, w).astype(np.uint32)
            free_updated = np.asarray(self.vertical_profile.free_ray_count[band_idx], dtype=np.uint32) + counts
            observed_updated = np.asarray(self.vertical_profile.observed_count[band_idx], dtype=np.uint32) + counts
            self.vertical_profile.free_ray_count[band_idx][:, :] = np.minimum(free_updated, uint16_max).astype(np.uint16)
            self.vertical_profile.observed_count[band_idx][:, :] = np.minimum(observed_updated, uint16_max).astype(np.uint16)
            touched = np.flatnonzero(counts.reshape(-1) > 0)
            if touched.size:
                self.vertical_profile.unknown_count[band_idx].reshape(-1)[touched] = 0

    def _mark_vertical_profile_free_weighted_flat(
        self,
        flat_indices_by_band: List[np.ndarray],
        weights_by_band: List[np.ndarray],
    ) -> None:
        h, w = self.grid.occupied.shape
        total_cells = int(h * w)
        uint16_max = int(np.iinfo(np.uint16).max)
        for band_idx, (flat_values, weights) in enumerate(zip(flat_indices_by_band, weights_by_band)):
            flat, weight = _valid_flat_weight_arrays(flat_values, weights, total_cells)
            if flat.size == 0:
                continue
            counts = np.bincount(flat, weights=weight, minlength=total_cells).reshape(h, w).astype(np.uint32)
            free_updated = np.asarray(self.vertical_profile.free_ray_count[band_idx], dtype=np.uint32) + counts
            observed_updated = np.asarray(self.vertical_profile.observed_count[band_idx], dtype=np.uint32) + counts
            self.vertical_profile.free_ray_count[band_idx][:, :] = np.minimum(free_updated, uint16_max).astype(np.uint16)
            self.vertical_profile.observed_count[band_idx][:, :] = np.minimum(observed_updated, uint16_max).astype(np.uint16)
            touched = np.flatnonzero(counts.reshape(-1) > 0)
            if touched.size:
                self.vertical_profile.unknown_count[band_idx].reshape(-1)[touched] = 0

    def _robot_footprint_radius_cells(self) -> int:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        return int(math.ceil(max(0.0, self.robot_radius_m) / self.resolution_m))

    def _robot_footprint_free_radius_cells(self) -> int:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        # This is only for marking the robot's own occupied footprint as
        # observed free. Collision inflation remains conservative via ceil().
        return int(math.floor(max(0.0, self.robot_radius_m) / self.resolution_m + 1e-9))

    def _splat_cells(
        self,
        rows_cols: np.ndarray,
        occupied: bool,
        point_threshold: int | None = None,
        mark_mask: np.ndarray | None = None,
    ) -> int:
        if rows_cols.size == 0:
            return 0
        threshold = self.splat_point_threshold if point_threshold is None else max(1, int(point_threshold))
        h, w = self.grid.occupied.shape
        flat = rows_cols[:, 0].astype(np.int64) * int(w) + rows_cols[:, 1].astype(np.int64)
        counts = np.bincount(flat, minlength=int(h * w))
        selected = np.flatnonzero(counts >= threshold)
        if len(selected) == 0:
            return 0
        rows = selected // int(w)
        cols = selected % int(w)
        for row, col in zip(rows, cols):
            if occupied:
                self._set_occupied_cell(int(row), int(col))
            else:
                self._set_free_cell(int(row), int(col))
            if mark_mask is not None:
                mark_mask[int(row), int(col)] = 1
        return int(len(selected))

    def _build_debug_stats(
        self,
        *,
        depth_arr: np.ndarray,
        valid_depth: np.ndarray,
        valid_pixels: np.ndarray,
        rel_z: np.ndarray,
        obstacle_mask: np.ndarray,
        free_mask: np.ndarray,
        occupied_endpoint_cells: int,
        free_ray_cells: int,
        free_protected_by_obstacle_endpoint: int,
        stale_obstacle_endpoint_cells_cleared: int,
        free_ray_occupied_cells_skipped: int,
        obstacle_ray_interior_free_suppressed_cells: int,
        ray_count: int,
        skipped_height_rays: int,
        vertical_profile_ray_count: int,
        vertical_profile_skipped_height_rays: int,
        ray_cast_backend: str,
        ray_unique_endpoint_cells: int,
        base_pose_world: Tuple[float, float, float, float],
        camera_pose_world: Tuple[float, float, float, float],
        ray_origin_cell: Tuple[int, int],
    ) -> dict:
        height = int(depth_arr.shape[0])
        rows = valid_pixels[:, 1].astype(np.float32) if len(valid_pixels) else np.zeros((0,), dtype=np.float32)
        top = rows < height / 3.0
        middle = (rows >= height / 3.0) & (rows < 2.0 * height / 3.0)
        bottom = rows >= 2.0 * height / 3.0
        return {
            "reason": "ok",
            "mapping_mode": "depth_ray_cast",
            "depth_shape": [int(v) for v in depth_arr.shape],
            "depth_stride_px": int(self.depth_stride_px),
            "vertical_profile_depth_stride_px": int(self.depth_stride_px),
            "ray_cast_backend": str(ray_cast_backend),
            "ray_unique_endpoint_cells": int(ray_unique_endpoint_cells),
            "base_pose_world": [float(v) for v in base_pose_world],
            "camera_pose_world": [float(v) for v in camera_pose_world],
            "ray_origin_cell": [int(ray_origin_cell[0]), int(ray_origin_cell[1])],
            "depth_m_percentiles": _percentiles(valid_depth),
            "rel_z_m_percentiles": _percentiles(rel_z),
            "valid_points": int(len(rel_z)),
            "ray_count": int(ray_count),
            "skipped_height_rays": int(skipped_height_rays),
            "vertical_profile_ray_count": int(vertical_profile_ray_count),
            "vertical_profile_skipped_height_rays": int(vertical_profile_skipped_height_rays),
            "free_band_points": int(np.count_nonzero(free_mask)),
            "obstacle_band_points": int(np.count_nonzero(obstacle_mask)),
            "below_free_min_points": int(np.count_nonzero(rel_z < self.free_min_height_m)),
            "between_free_and_obstacle_points": int(
                np.count_nonzero((rel_z > self.free_max_height_m) & (rel_z < self.obstacle_min_height_m))
            ),
            "above_obstacle_max_points": int(np.count_nonzero(rel_z > self.obstacle_max_height_m)),
            "ceiling_like_points": int(np.count_nonzero(rel_z > 1.8)),
            "negative_height_points": int(np.count_nonzero(rel_z < -0.05)),
            "occupied_endpoint_cells": int(occupied_endpoint_cells),
            "free_ray_cells": int(free_ray_cells),
            "free_ray_cells_protected_by_obstacle_endpoint": int(free_protected_by_obstacle_endpoint),
            "stale_obstacle_endpoint_cells_cleared_by_free_rays": int(stale_obstacle_endpoint_cells_cleared),
            "free_ray_occupied_cells_skipped": int(free_ray_occupied_cells_skipped),
            "obstacle_ray_interior_free_suppressed_cells": int(obstacle_ray_interior_free_suppressed_cells),
            "obstacle_endpoint_evidence_increment": int(self._obstacle_endpoint_evidence_increment),
            "free_ray_obstacle_endpoint_decay": int(self._free_ray_obstacle_endpoint_decay),
            "depth_obstacle_endpoint_cells": int(np.count_nonzero(self.depth_obstacle_endpoint_count)),
            "depth_obstacle_endpoint_count_sum": int(np.sum(self.depth_obstacle_endpoint_count, dtype=np.uint64)),
            "obstacle_splat_cells": int(occupied_endpoint_cells),
            "free_splat_cells": int(free_ray_cells),
            "image_bands": {
                "top": self._band_debug(rel_z, obstacle_mask, free_mask, top),
                "middle": self._band_debug(rel_z, obstacle_mask, free_mask, middle),
                "bottom": self._band_debug(rel_z, obstacle_mask, free_mask, bottom),
            },
            "height_filters_m": {
                "free_min": float(self.free_min_height_m),
                "free_max": float(self.free_max_height_m),
                "obstacle_min": float(self.obstacle_min_height_m),
                "obstacle_max": float(self.obstacle_max_height_m),
                "vertical_profile_free_min": float(self.vertical_profile_free_min_height_m),
                "vertical_profile_free_max": float(self.vertical_profile_free_max_height_m),
            },
            "vertical_profile": self.vertical_profile.to_debug_dict(),
            "roomseg_ray_evidence": {
                "ray_covered_cells": int(np.count_nonzero(self.roomseg_ray_covered_count)),
                "ray_covered_count_sum": int(np.sum(self.roomseg_ray_covered_count, dtype=np.uint64)),
                "free_view_cells": int(np.count_nonzero(self.roomseg_free_view_count)),
                "free_view_count_sum": int(np.sum(self.roomseg_free_view_count, dtype=np.uint64)),
                "terminal_wall_cells": int(np.count_nonzero(self.roomseg_terminal_wall_count)),
                "terminal_wall_count_sum": int(np.sum(self.roomseg_terminal_wall_count, dtype=np.uint64)),
                "terminal_wall_view_cells": int(np.count_nonzero(self.roomseg_terminal_wall_view_count)),
                "terminal_wall_view_count_sum": int(np.sum(self.roomseg_terminal_wall_view_count, dtype=np.uint64)),
                "terminal_wall_splat_cells": int(np.count_nonzero(self.roomseg_terminal_wall_splat)),
                "terminal_wall_depth_min_m_percentiles": _percentiles(
                    self.roomseg_terminal_wall_depth_min[np.isfinite(self.roomseg_terminal_wall_depth_min)]
                ),
                "terminal_wall_height_min_m_percentiles": _percentiles(
                    self.roomseg_terminal_wall_height_min[np.isfinite(self.roomseg_terminal_wall_height_min)]
                ),
                "terminal_wall_height_max_m_percentiles": _percentiles(
                    self.roomseg_terminal_wall_height_max[np.isfinite(self.roomseg_terminal_wall_height_max)]
                ),
            },
            "splat_thresholds": {
                "free": int(self.free_splat_point_threshold),
                "obstacle": int(self.splat_point_threshold),
                "unused_for_mapping_mode": "depth_ray_cast",
            },
        }

    @staticmethod
    def _band_debug(rel_z: np.ndarray, obstacle_mask: np.ndarray, free_mask: np.ndarray, band_mask: np.ndarray) -> dict:
        rel = rel_z[band_mask]
        return {
            "points": int(len(rel)),
            "rel_z_m_percentiles": _percentiles(rel),
            "free_band_points": int(np.count_nonzero(free_mask & band_mask)),
            "obstacle_band_points": int(np.count_nonzero(obstacle_mask & band_mask)),
            "ceiling_like_points": int(np.count_nonzero(rel > 1.8)),
            "negative_height_points": int(np.count_nonzero(rel < -0.05)),
        }


def _percentiles(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    keys = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(arr, keys)
    return {"p%d" % int(key): float(val) for key, val in zip(keys, vals)}


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - float(started_at)) * 1000.0)


@dataclass
class _RayCastEvidence:
    backend: str
    free_flat: np.ndarray
    free_endpoint_flat: np.ndarray
    occupied_flat: np.ndarray
    free_flat_by_band: List[np.ndarray]
    free_weight_by_band: List[np.ndarray]
    roomseg_ray_covered_flat: np.ndarray
    roomseg_ray_covered_weight: np.ndarray
    terminal_wall_flat: np.ndarray
    terminal_wall_depth_m: np.ndarray
    terminal_wall_rel_z_m: np.ndarray
    nav_ray_count: int
    nav_skipped_height_rays: int
    vertical_profile_ray_count: int
    vertical_profile_skipped_height_rays: int
    obstacle_ray_interior_free_suppressed_cells: int
    unique_endpoint_cells: int


def _world_points_to_grid(points_world: np.ndarray, info: MapInfo) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    cols = np.floor((points[:, 0].astype(np.float64) - float(info.min_x)) / float(info.resolution_m)).astype(np.int32)
    rows = np.floor((float(info.max_y) - points[:, 1].astype(np.float64)) / float(info.resolution_m)).astype(np.int32)
    return np.stack([rows, cols], axis=1)


def _collect_ray_cast_evidence(
    *,
    origin_cell: Tuple[int, int],
    endpoints: np.ndarray,
    endpoint_rel_z: np.ndarray,
    endpoint_depth_m: np.ndarray,
    endpoint_is_obstacle: np.ndarray,
    ray_can_clear: np.ndarray,
    map_width: int,
    depth_min_m: float,
    depth_max_m: float,
    camera_rel_z_m: float,
    z_min_m: float,
    z_max_m: float,
    band_ranges_m: Tuple[Tuple[float, float], ...],
) -> _RayCastEvidence:
    cells = np.asarray(endpoints, dtype=np.int32).reshape(-1, 2)
    rel_z = np.asarray(endpoint_rel_z, dtype=np.float32).reshape(-1)
    depth = np.asarray(endpoint_depth_m, dtype=np.float32).reshape(-1)
    obstacle = np.asarray(endpoint_is_obstacle, dtype=bool).reshape(-1)
    clear = np.asarray(ray_can_clear, dtype=bool).reshape(-1)
    n = int(cells.shape[0])
    if not (len(rel_z) == len(depth) == len(obstacle) == len(clear) == n):
        raise ValueError("ray evidence arrays must have matching lengths")
    band_count = len(tuple(band_ranges_m))
    empty_i64 = np.zeros((0,), dtype=np.int64)
    empty_f32 = np.zeros((0,), dtype=np.float32)
    if n == 0:
        return _RayCastEvidence(
            backend="endpoint_grouped_bresenham",
            free_flat=empty_i64,
            free_endpoint_flat=empty_i64,
            occupied_flat=empty_i64,
            free_flat_by_band=[empty_i64.copy() for _ in range(band_count)],
            free_weight_by_band=[empty_i64.copy() for _ in range(band_count)],
            roomseg_ray_covered_flat=empty_i64,
            roomseg_ray_covered_weight=empty_i64,
            terminal_wall_flat=empty_i64,
            terminal_wall_depth_m=empty_f32,
            terminal_wall_rel_z_m=empty_f32,
            nav_ray_count=0,
            nav_skipped_height_rays=0,
            vertical_profile_ray_count=0,
            vertical_profile_skipped_height_rays=0,
            obstacle_ray_interior_free_suppressed_cells=0,
            unique_endpoint_cells=0,
        )

    width = int(map_width)
    endpoint_flat = cells[:, 0].astype(np.int64) * width + cells[:, 1].astype(np.int64)
    order = np.argsort(endpoint_flat, kind="stable")
    sorted_flat = endpoint_flat[order]
    unique_flat, starts, counts = np.unique(sorted_flat, return_index=True, return_counts=True)

    lo = np.maximum(float(z_min_m), np.minimum(float(camera_rel_z_m), rel_z.astype(np.float32)))
    hi = np.minimum(float(z_max_m), np.maximum(float(camera_rel_z_m), rel_z.astype(np.float32)))
    valid_interval = hi >= lo
    band_bits = np.zeros((n,), dtype=np.uint8)
    for band_idx, (band_lo, band_hi) in enumerate(band_ranges_m):
        band_mask = valid_interval & (float(band_hi) > lo) & (float(band_lo) < hi)
        if np.any(band_mask):
            band_bits[band_mask] |= np.uint8(1 << int(band_idx))

    roomseg_ray_valid = np.isfinite(depth) & (depth > float(depth_min_m)) & (depth < float(depth_max_m))
    terminal_valid = roomseg_ray_valid & np.isfinite(rel_z) & (rel_z >= float(z_min_m)) & (rel_z <= float(z_max_m))

    free_chunks: list[np.ndarray] = []
    covered_chunks: list[np.ndarray] = []
    covered_weight_chunks: list[np.ndarray] = []
    band_flat_chunks: list[list[np.ndarray]] = [[] for _ in range(band_count)]
    band_weight_chunks: list[list[np.ndarray]] = [[] for _ in range(band_count)]
    nav_ray_count = 0
    vertical_profile_ray_count = 0
    obstacle_ray_interior_free_suppressed_cells = 0

    for endpoint, start, count in zip(unique_flat, starts, counts):
        group_indices = order[int(start) : int(start) + int(count)]
        end_row = int(endpoint // width)
        end_col = int(endpoint % width)
        line = _bresenham_cells(origin_cell, (end_row, end_col))
        if not line:
            continue
        line_flat = np.fromiter((int(row) * width + int(col) for row, col in line), dtype=np.int64)
        group_clear = group_indices[clear[group_indices]]
        if group_clear.size:
            nav_ray_count += int(group_clear.size)
            if np.any(~obstacle[group_clear]):
                free_chunks.append(line_flat)
            elif line_flat.size > 1:
                obstacle_ray_interior_free_suppressed_cells += int(line_flat.size - 1)

        profile_line_flat = line_flat[:-1]
        if profile_line_flat.size == 0:
            continue
        valid_group = group_indices[roomseg_ray_valid[group_indices] & (band_bits[group_indices] > 0)]
        if valid_group.size == 0:
            continue
        group_bits = band_bits[valid_group]
        unique_bits, bit_counts = np.unique(group_bits, return_counts=True)
        for bits, bits_count in zip(unique_bits, bit_counts):
            count_int = int(bits_count)
            if count_int <= 0:
                continue
            vertical_profile_ray_count += count_int
            covered_chunks.append(profile_line_flat)
            covered_weight_chunks.append(np.full(profile_line_flat.shape, count_int, dtype=np.int64))
            for band_idx in range(band_count):
                if int(bits) & (1 << int(band_idx)):
                    band_flat_chunks[band_idx].append(profile_line_flat)
                    band_weight_chunks[band_idx].append(np.full(profile_line_flat.shape, count_int, dtype=np.int64))

    occupied_flat = endpoint_flat[obstacle]
    free_endpoint_flat = endpoint_flat[clear & ~obstacle]
    terminal_flat = endpoint_flat[terminal_valid]
    return _RayCastEvidence(
        backend="endpoint_grouped_bresenham",
        free_flat=_concat_i64(free_chunks),
        free_endpoint_flat=np.asarray(free_endpoint_flat, dtype=np.int64),
        occupied_flat=np.asarray(occupied_flat, dtype=np.int64),
        free_flat_by_band=[_concat_i64(chunks) for chunks in band_flat_chunks],
        free_weight_by_band=[_concat_i64(chunks) for chunks in band_weight_chunks],
        roomseg_ray_covered_flat=_concat_i64(covered_chunks),
        roomseg_ray_covered_weight=_concat_i64(covered_weight_chunks),
        terminal_wall_flat=np.asarray(terminal_flat, dtype=np.int64),
        terminal_wall_depth_m=np.asarray(depth[terminal_valid], dtype=np.float32),
        terminal_wall_rel_z_m=np.asarray(rel_z[terminal_valid], dtype=np.float32),
        nav_ray_count=int(nav_ray_count),
        nav_skipped_height_rays=int(n - nav_ray_count),
        vertical_profile_ray_count=int(vertical_profile_ray_count),
        vertical_profile_skipped_height_rays=int(n - vertical_profile_ray_count),
        obstacle_ray_interior_free_suppressed_cells=int(obstacle_ray_interior_free_suppressed_cells),
        unique_endpoint_cells=int(len(unique_flat)),
    )


def _concat_i64(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate([np.asarray(chunk, dtype=np.int64).reshape(-1) for chunk in chunks]).astype(np.int64, copy=False)


def _valid_flat_array(values: List[int] | np.ndarray, total_cells: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return arr[(arr >= 0) & (arr < int(total_cells))]


def _pose_bin(base_pose_world: Tuple[float, float, float, float]) -> int:
    if len(base_pose_world) < 4:
        return 0
    x_bin = int(round(float(base_pose_world[0]) / 0.25))
    y_bin = int(round(float(base_pose_world[1]) / 0.25))
    yaw_bin = int(round(float(base_pose_world[3]) / (math.pi / 8.0)))
    return int(((x_bin & 0x3FF) << 20) ^ ((y_bin & 0x3FF) << 10) ^ (yaw_bin & 0x3FF))


def _valid_flat_weight_arrays(values: np.ndarray, weights: np.ndarray, total_cells: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    if arr.shape != weight.shape:
        raise ValueError("flat value and weight arrays must have the same shape")
    valid = (arr >= 0) & (arr < int(total_cells)) & np.isfinite(weight) & (weight > 0.0)
    return arr[valid], weight[valid]


def _unique_flat(values: List[int]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.asarray(values, dtype=np.int64))


def _unique_flat_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(arr)


def _append_vertical_profile_free_ray_cells(
    flat_indices_by_band: List[List[int]],
    free_cells: List[Tuple[int, int]],
    *,
    map_width: int,
    origin_rel_z_m: float,
    endpoint_rel_z_m: float,
    z_min_m: float,
    z_max_m: float,
    vertical_profile: VerticalProfileMap,
) -> int:
    """Mark vertical free evidence for xy columns crossed by a free ray.

    The room-segmentation vertical profile asks: for this xy column, did any
    ray pass through free space between z_min and z_max? Using only the depth
    endpoint height drops valid rays that hit the floor or a low object, even
    though the same ray crossed 0.1-2.5 m free space before the hit. Therefore
    each crossed xy cell receives the clipped free vertical interval spanned by
    the camera origin and the depth endpoint.
    """

    if not free_cells:
        return 0
    width = int(map_width)
    lo = max(float(z_min_m), min(float(origin_rel_z_m), float(endpoint_rel_z_m)))
    hi = min(float(z_max_m), max(float(origin_rel_z_m), float(endpoint_rel_z_m)))
    if hi < lo:
        return 0
    band_indices: list[int] = []
    for band_idx, (_name, (band_lo, band_hi)) in enumerate(zip(vertical_profile.band_names, vertical_profile.band_ranges_m)):
        if float(band_hi) <= lo or float(band_lo) >= hi:
            continue
        band_indices.append(int(band_idx))
    if not band_indices:
        return 0
    added = 0
    for row, col in free_cells:
        flat = int(row) * width + int(col)
        for band_idx in band_indices:
            flat_indices_by_band[int(band_idx)].append(flat)
            added += 1
    return int(added)


def _disk_offsets(radius_cells: int) -> List[Tuple[int, int]]:
    rr = int(max(0, radius_cells))
    out: List[Tuple[int, int]] = []
    for dr in range(-rr, rr + 1):
        for dc in range(-rr, rr + 1):
            if dr * dr + dc * dc <= rr * rr:
                out.append((dr, dc))
    return out


def _bresenham_cells(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    row0, col0 = int(start[0]), int(start[1])
    row1, col1 = int(end[0]), int(end[1])
    drow = abs(row1 - row0)
    dcol = abs(col1 - col0)
    srow = 1 if row0 < row1 else -1
    scol = 1 if col0 < col1 else -1
    row, col = row0, col0
    cells: List[Tuple[int, int]] = []
    if dcol > drow:
        err = dcol // 2
        while col != col1:
            cells.append((row, col))
            col += scol
            err -= drow
            if err < 0:
                row += srow
                err += dcol
        cells.append((row, col))
    else:
        err = drow // 2
        while row != row1:
            cells.append((row, col))
            row += srow
            err -= dcol
            if err < 0:
                col += scol
                err += drow
        cells.append((row, col))
    return cells


def _dilate_binary(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask).astype(bool)
    if radius_cells <= 0 or not np.any(src):
        return src
    out = np.array(src, copy=True)
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = _disk_offsets(radius_cells)
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out
