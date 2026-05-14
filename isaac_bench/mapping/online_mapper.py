from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from isaac_bench.mapping.grid_map import OnlineGridMap
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
        splat_point_threshold: int = 6,
        free_splat_point_threshold: int | None = None,
        robot_radius_m: float = 0.14,
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
        self.static_nearfield_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_free_mask = np.zeros_like(self.grid.free, dtype=np.uint8)

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self.grid = OnlineGridMap.centered(start_xy[0], start_xy[1], self.size_m, self.resolution_m)
        self.last_debug_stats = {"reason": "reset"}
        self.last_nearfield_debug_stats = {"reason": "reset"}
        self.last_static_nearfield_debug_stats = {"reason": "reset"}
        self.static_nearfield_mask = np.zeros_like(self.grid.free, dtype=np.uint8)
        self.depth_free_mask = np.zeros_like(self.grid.free, dtype=np.uint8)

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
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]
        if depth_arr.ndim != 2 or depth_arr.size == 0:
            self.last_debug_stats = {"reason": "invalid_depth_shape", "depth_shape": list(depth_arr.shape)}
            self._mark_robot_footprint_free(base_pose_world)
            return self.grid

        stride = self.depth_stride_px
        vs = np.arange(stride // 2, min(depth_arr.shape[0], intr.height), stride, dtype=np.int32)
        us = np.arange(stride // 2, min(depth_arr.shape[1], intr.width), stride, dtype=np.int32)
        if len(vs) == 0 or len(us) == 0:
            self.last_debug_stats = {"reason": "no_sample_pixels", "depth_shape": list(depth_arr.shape)}
            self._mark_robot_footprint_free(base_pose_world)
            return self.grid

        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
        sampled_depth = depth_arr[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        valid = np.isfinite(sampled_depth) & (sampled_depth > self.depth_min_m) & (sampled_depth < self.depth_max_m)
        if int(valid.sum()) == 0:
            self.last_debug_stats = {
                "reason": "no_valid_depth",
                "sampled_pixels": int(len(pixels)),
                "depth_shape": list(depth_arr.shape),
            }
            self._mark_robot_footprint_free(base_pose_world)
            return self.grid

        valid_pixels = pixels[valid]
        valid_depth = sampled_depth[valid]
        points_cam = backproject_pixels(depth_arr, valid_pixels, intr)
        points_world = transform_points(points_cam, camera_pose_world)
        floor_z = float(base_pose_world[2])

        rel_z = points_world[:, 2].astype(np.float32) - floor_z
        rows_cols = np.asarray(
            [self.grid.world_to_grid(float(point[0]), float(point[1])) for point in points_world],
            dtype=np.int32,
        )
        in_bounds = (
            (rows_cols[:, 0] >= 0)
            & (rows_cols[:, 0] < self.grid.map_info.height)
            & (rows_cols[:, 1] >= 0)
            & (rows_cols[:, 1] < self.grid.map_info.width)
        )
        if int(np.count_nonzero(in_bounds)) == 0:
            self.last_debug_stats = {
                "reason": "no_points_in_map_bounds",
                "valid_points": int(len(points_world)),
                "depth_shape": list(depth_arr.shape),
                "depth_m_percentiles": _percentiles(valid_depth),
            }
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
            return self.grid

        in_bounds_pixels = valid_pixels[in_bounds]
        in_bounds_depth = valid_depth[in_bounds]
        rows_cols = rows_cols[in_bounds]
        rel_z = rel_z[in_bounds]
        obstacle_mask = (rel_z >= self.obstacle_min_height_m) & (rel_z <= self.obstacle_max_height_m)
        free_mask = (rel_z >= self.free_min_height_m) & (rel_z <= self.free_max_height_m)
        ray_clear_mask = (rel_z >= self.free_min_height_m) & (rel_z <= self.obstacle_max_height_m)
        free_cells: set[Tuple[int, int]] = set()
        occupied_cells: set[Tuple[int, int]] = set()
        ray_count = 0
        skipped_height_rays = 0
        for endpoint, endpoint_is_obstacle, ray_can_clear in zip(rows_cols, obstacle_mask, ray_clear_mask):
            if not bool(ray_can_clear):
                skipped_height_rays += 1
                continue
            end_cell = (int(endpoint[0]), int(endpoint[1]))
            line = _bresenham_cells((int(origin_cell[0]), int(origin_cell[1])), end_cell)
            if not line:
                continue
            ray_count += 1
            free_line = line[:-1] if bool(endpoint_is_obstacle) else line
            for row, col in free_line:
                if is_inside_grid(row, col, self.grid.map_info):
                    free_cells.add((int(row), int(col)))
            if bool(endpoint_is_obstacle):
                for dr, dc in _disk_offsets(0):
                    row, col = int(end_cell[0] + dr), int(end_cell[1] + dc)
                    if is_inside_grid(row, col, self.grid.map_info):
                        occupied_cells.add((row, col))

        for row, col in free_cells:
            self._set_free_cell(row, col, mark_mask=self.depth_free_mask)
        for row, col in occupied_cells:
            self._set_occupied_cell(row, col)
        self._mark_robot_footprint_free(base_pose_world)
        self.last_debug_stats = self._build_debug_stats(
            depth_arr=depth_arr,
            valid_depth=in_bounds_depth,
            valid_pixels=in_bounds_pixels,
            rel_z=rel_z,
            obstacle_mask=obstacle_mask,
            free_mask=free_mask,
            occupied_endpoint_cells=len(occupied_cells),
            free_ray_cells=len(free_cells),
            ray_count=ray_count,
            skipped_height_rays=skipped_height_rays,
            base_pose_world=base_pose_world,
            camera_pose_world=camera_pose_world,
            ray_origin_cell=origin_cell,
        )
        return self.grid

    def update_nearfield_topdown(
        self,
        depth: np.ndarray,
        intr: CameraIntrinsics,
        base_pose_world: Tuple[float, float, float, float],
        camera_pose_world: Tuple[float, float, float, float],
        radius_m: float = 0.75,
        ignore_radius_m: float = 0.22,
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
                free_cells += 1
                if is_opening:
                    opening_cells += 1
            else:
                self.grid.free[row, col] = 0
                self.grid.occupied[row, col] = 0
                if bool(occupancy[static_row, static_col]):
                    occupied_cells += 1
                else:
                    blocked_clearance_cells += 1
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
            "skipped_outside_dynamic": int(skipped_outside_dynamic),
            "skipped_outside_static": int(skipped_outside_static),
            "base_pose_world": [float(v) for v in base_pose_world],
        }
        self.last_static_nearfield_debug_stats = stats
        return stats

    def traversible(self, unknown_is_obstacle: bool = True) -> np.ndarray:
        occupied = self.inflated_occupied()
        if unknown_is_obstacle:
            free = self.grid.free.astype(bool)
        else:
            free = np.ones_like(occupied, dtype=bool)
        return free & ~occupied

    def inflated_occupied(self) -> np.ndarray:
        radius_cells = self._robot_footprint_radius_cells()
        return _dilate_binary(self.grid.occupied.astype(bool), radius_cells)

    def _mark_robot_footprint_free(self, base_pose_world: Tuple[float, float, float, float]) -> None:
        radius_cells = max(1, self._robot_footprint_radius_cells())
        center = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        for dr, dc in _disk_offsets(radius_cells):
            self._set_free_cell(center[0] + dr, center[1] + dc, mark_mask=self.depth_free_mask)

    def _set_free_cell(self, row: int, col: int, mark_mask: np.ndarray | None = None) -> None:
        if 0 <= int(row) < self.grid.map_info.height and 0 <= int(col) < self.grid.map_info.width:
            rr, cc = int(row), int(col)
            self.grid.free[rr, cc] = 1
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

    def _robot_footprint_radius_cells(self) -> int:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        return int(math.ceil(max(0.0, self.robot_radius_m) / self.resolution_m))

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
        ray_count: int,
        skipped_height_rays: int,
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
            "base_pose_world": [float(v) for v in base_pose_world],
            "camera_pose_world": [float(v) for v in camera_pose_world],
            "ray_origin_cell": [int(ray_origin_cell[0]), int(ray_origin_cell[1])],
            "depth_m_percentiles": _percentiles(valid_depth),
            "rel_z_m_percentiles": _percentiles(rel_z),
            "valid_points": int(len(rel_z)),
            "ray_count": int(ray_count),
            "skipped_height_rays": int(skipped_height_rays),
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
