from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

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
        obstacle_min_height_m: float = 0.05,
        obstacle_max_height_m: float = 1.50,
        robot_radius_m: float = 0.28,
        inflation_radius_m: float = 0.0,
    ):
        self.size_m = float(size_m)
        self.resolution_m = float(resolution_m)
        self.depth_max_m = float(depth_max_m)
        self.depth_min_m = float(depth_min_m)
        self.depth_stride_px = max(1, int(depth_stride_px))
        self.ray_step_m = float(ray_step_m) if ray_step_m is not None else max(self.resolution_m * 2.0, 0.10)
        self.obstacle_min_height_m = float(obstacle_min_height_m)
        self.obstacle_max_height_m = float(obstacle_max_height_m)
        self.robot_radius_m = float(robot_radius_m)
        self.inflation_radius_m = float(inflation_radius_m)
        self.grid = OnlineGridMap.centered(0.0, 0.0, self.size_m, self.resolution_m)

    def reset(self, start_xy: Tuple[float, float]) -> None:
        self.grid = OnlineGridMap.centered(start_xy[0], start_xy[1], self.size_m, self.resolution_m)

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
            return self.update_simple_radius(base_pose_world)

        stride = self.depth_stride_px
        vs = np.arange(stride // 2, min(depth_arr.shape[0], intr.height), stride, dtype=np.int32)
        us = np.arange(stride // 2, min(depth_arr.shape[1], intr.width), stride, dtype=np.int32)
        if len(vs) == 0 or len(us) == 0:
            return self.update_simple_radius(base_pose_world)

        uu, vv = np.meshgrid(us, vs)
        pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1).astype(np.float32)
        sampled_depth = depth_arr[pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
        valid = np.isfinite(sampled_depth) & (sampled_depth > self.depth_min_m) & (sampled_depth < self.depth_max_m)
        if int(valid.sum()) == 0:
            return self.update_simple_radius(base_pose_world)

        valid_pixels = pixels[valid]
        valid_depth = sampled_depth[valid]
        points_cam = backproject_pixels(depth_arr, valid_pixels, intr)
        points_world = transform_points(points_cam, camera_pose_world)
        cam_x, cam_y = float(camera_pose_world[0]), float(camera_pose_world[1])
        floor_z = float(base_pose_world[2])

        self._mark_robot_footprint_free(base_pose_world)
        for point, range_m in zip(points_world, valid_depth):
            end_x, end_y, end_z = float(point[0]), float(point[1]), float(point[2])
            rel_z = end_z - floor_z
            hit_obstacle = (
                float(range_m) < self.depth_max_m - self.ray_step_m
                and self.obstacle_min_height_m <= rel_z <= self.obstacle_max_height_m
            )
            self._mark_ray_free(cam_x, cam_y, end_x, end_y, stop_short=hit_obstacle)
            if hit_obstacle:
                row, col = self.grid.world_to_grid(end_x, end_y)
                self.grid.mark_occupied(row, col)
            else:
                row, col = self.grid.world_to_grid(end_x, end_y)
                self.grid.mark_free(row, col)
        return self.grid

    def traversible(self, unknown_is_obstacle: bool = True) -> np.ndarray:
        occupied = self.inflated_occupied()
        if unknown_is_obstacle:
            free = self.grid.free.astype(bool)
        else:
            free = np.ones_like(occupied, dtype=bool)
        return free & ~occupied

    def inflated_occupied(self) -> np.ndarray:
        radius_cells = int(math.ceil((self.robot_radius_m + self.inflation_radius_m) / self.resolution_m))
        return _dilate_binary(self.grid.occupied.astype(bool), radius_cells)

    def _mark_robot_footprint_free(self, base_pose_world: Tuple[float, float, float, float]) -> None:
        radius_cells = max(1, int(math.ceil((self.robot_radius_m + self.resolution_m) / self.resolution_m)))
        center = self.grid.world_to_grid(float(base_pose_world[0]), float(base_pose_world[1]))
        for dr, dc in _disk_offsets(radius_cells):
            self.grid.mark_free(center[0] + dr, center[1] + dc)

    def _mark_ray_free(self, start_x: float, start_y: float, end_x: float, end_y: float, stop_short: bool) -> None:
        dx = float(end_x) - float(start_x)
        dy = float(end_y) - float(start_y)
        dist = float(math.hypot(dx, dy))
        if dist <= 1e-6:
            return
        usable = max(0.0, dist - (self.resolution_m * 1.5 if stop_short else 0.0))
        steps = max(1, int(math.ceil(usable / self.ray_step_m)))
        for idx in range(steps + 1):
            t = min(1.0, (idx * self.ray_step_m) / max(usable, 1e-6))
            row, col = self.grid.world_to_grid(start_x + dx * t, start_y + dy * t)
            self.grid.mark_free(row, col)


def _disk_offsets(radius_cells: int) -> List[Tuple[int, int]]:
    rr = int(max(0, radius_cells))
    out: List[Tuple[int, int]] = []
    for dr in range(-rr, rr + 1):
        for dc in range(-rr, rr + 1):
            if dr * dr + dc * dc <= rr * rr:
                out.append((dr, dc))
    return out


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
