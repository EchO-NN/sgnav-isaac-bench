from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .config import DepthConfig, MapConfig, VerticalProjectionConfig
from .data_types import GridSpec, VerticalMapState
from .utils import in_bounds, sigmoid, world_to_grid


def prob_to_logodds(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return float(np.log(p / (1.0 - p)))


def logodds_to_prob(values: np.ndarray) -> np.ndarray:
    return np.asarray(sigmoid(values), dtype=np.float32)


class VerticalMapBuilder:
    def __init__(
        self,
        config: MapConfig | Mapping[str, object] | None,
        depth_config: DepthConfig | Mapping[str, object] | None,
        projection_config: VerticalProjectionConfig | Mapping[str, object] | None,
        grid_spec: GridSpec,
    ):
        self.map_config = config if isinstance(config, MapConfig) else MapConfig.from_mapping(config or {})
        self.depth_config = depth_config if isinstance(depth_config, DepthConfig) else DepthConfig.from_mapping(depth_config or {})
        self.projection_config = (
            projection_config
            if isinstance(projection_config, VerticalProjectionConfig)
            else VerticalProjectionConfig.from_mapping(projection_config or {})
        )
        self.grid_spec = grid_spec
        z_count = max(1, int(math.ceil((float(self.map_config.z_max_m) - float(self.map_config.z_min_m)) / max(float(self.map_config.z_resolution_m), 1e-9))))
        shape3 = (z_count, int(grid_spec.height), int(grid_spec.width))
        self.free_log_odds = np.zeros(shape3, dtype=np.float32)
        self.occupied_log_odds = np.zeros(shape3, dtype=np.float32)
        self.endpoint_log_odds = np.zeros((int(grid_spec.height), int(grid_spec.width)), dtype=np.float32)
        self.observation_count_3d = np.zeros(shape3, dtype=np.uint16)
        self.last_observed_frame_3d = np.full(shape3, -1, dtype=np.int32)
        self._last_state = self._project(frame_id=0)

    def reset(self) -> None:
        self.free_log_odds.fill(0.0)
        self.occupied_log_odds.fill(0.0)
        self.endpoint_log_odds.fill(0.0)
        self.observation_count_3d.fill(0)
        self.last_observed_frame_3d.fill(-1)
        self._last_state = self._project(frame_id=0)

    def update_from_depth(
        self,
        depth: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_pose_world: np.ndarray,
        frame_id: int,
    ) -> VerticalMapState:
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim != 2:
            raise ValueError("depth must have shape [H, W]")
        k = np.asarray(camera_intrinsics, dtype=np.float64)
        t_wc = np.asarray(camera_pose_world, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError("camera_intrinsics must be 3x3")
        if t_wc.shape != (4, 4):
            raise ValueError("camera_pose_world must be 4x4")

        cfg = self.depth_config
        pcfg = self.projection_config
        self.free_log_odds *= float(pcfg.decay_per_update)
        self.occupied_log_odds *= float(pcfg.decay_per_update)
        self.endpoint_log_odds *= float(pcfg.decay_per_update)

        stride = max(1, int(cfg.depth_stride_px))
        rows = np.arange(0, depth_arr.shape[0], stride, dtype=np.int32)
        cols = np.arange(0, depth_arr.shape[1], stride, dtype=np.int32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        pix_rows = rr.reshape(-1)
        pix_cols = cc.reshape(-1)
        valid_depth = depth_arr[pix_rows, pix_cols]
        valid = np.isfinite(valid_depth) & (valid_depth >= float(cfg.depth_min_m)) & (valid_depth <= float(cfg.depth_max_m))
        pix_rows = pix_rows[valid]
        pix_cols = pix_cols[valid]
        valid_depth = valid_depth[valid]
        max_rays = max(1, int(cfg.max_ray_points_per_frame))
        if valid_depth.size > max_rays:
            idx = np.linspace(0, valid_depth.size - 1, max_rays).astype(np.int64)
            pix_rows = pix_rows[idx]
            pix_cols = pix_cols[idx]
            valid_depth = valid_depth[idx]

        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        rot = t_wc[:3, :3]
        trans = t_wc[:3, 3]
        for row, col, dist in zip(pix_rows.tolist(), pix_cols.tolist(), valid_depth.tolist()):
            x = (float(col) - cx) * float(dist) / max(fx, 1e-9)
            y = (float(row) - cy) * float(dist) / max(fy, 1e-9)
            endpoint_cam = np.asarray([x, y, float(dist)], dtype=np.float64)
            endpoint_world = rot @ endpoint_cam + trans
            endpoint_distance = float(np.linalg.norm(endpoint_cam))
            if endpoint_distance <= 1e-6:
                continue
            direction_world = (endpoint_world - trans) / endpoint_distance
            free_limit = max(0.0, endpoint_distance - float(cfg.endpoint_margin_m))
            n_steps = int(math.floor(free_limit / max(float(cfg.ray_step_m), 1e-9)))
            free_cells: set[tuple[int, int, int]] = set()
            for step in range(1, n_steps + 1):
                point = trans + direction_world * (float(step) * float(cfg.ray_step_m))
                voxel = self._world_to_voxel(point)
                if voxel is not None:
                    free_cells.add(voxel)
            if free_cells:
                z_idx, rs, cs = zip(*free_cells)
                self.free_log_odds[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)] += float(pcfg.free_log_odds_update)
                self.observation_count_3d[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)] = np.minimum(
                    self.observation_count_3d[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)].astype(np.uint32) + 1,
                    np.iinfo(np.uint16).max,
                )
                self.last_observed_frame_3d[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)] = int(frame_id)

            endpoint_cells: set[tuple[int, int, int]] = set()
            for delta in (-float(cfg.endpoint_margin_m), 0.0, float(cfg.endpoint_margin_m)):
                point = endpoint_world + direction_world * delta
                voxel = self._world_to_voxel(point)
                if voxel is not None:
                    endpoint_cells.add(voxel)
            if endpoint_cells:
                z_idx, rs, cs = zip(*endpoint_cells)
                self.occupied_log_odds[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)] += float(pcfg.occupied_log_odds_update)
                self.endpoint_log_odds[np.asarray(rs), np.asarray(cs)] += float(pcfg.endpoint_log_odds_update)
                self.observation_count_3d[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)] = np.minimum(
                    self.observation_count_3d[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)].astype(np.uint32) + 1,
                    np.iinfo(np.uint16).max,
                )
                self.last_observed_frame_3d[np.asarray(z_idx), np.asarray(rs), np.asarray(cs)] = int(frame_id)

        np.clip(self.free_log_odds, float(pcfg.min_log_odds), float(pcfg.max_log_odds), out=self.free_log_odds)
        np.clip(self.occupied_log_odds, float(pcfg.min_log_odds), float(pcfg.max_log_odds), out=self.occupied_log_odds)
        np.clip(self.endpoint_log_odds, float(pcfg.min_log_odds), float(pcfg.max_log_odds), out=self.endpoint_log_odds)
        self._last_state = self._project(frame_id=int(frame_id))
        return self._last_state

    def from_probability_layers(
        self,
        *,
        p_free: np.ndarray,
        p_occupied: np.ndarray | None = None,
        p_unknown: np.ndarray | None = None,
        p_endpoint: np.ndarray | None = None,
        frame_id: int = 0,
    ) -> VerticalMapState:
        free = np.asarray(p_free, dtype=np.float32)
        occ = np.zeros_like(free, dtype=np.float32) if p_occupied is None else np.asarray(p_occupied, dtype=np.float32)
        unk = np.clip(1.0 - np.maximum(free, occ), 0.0, 1.0) if p_unknown is None else np.asarray(p_unknown, dtype=np.float32)
        endpoint = np.zeros_like(free, dtype=np.float32) if p_endpoint is None else np.asarray(p_endpoint, dtype=np.float32)
        if free.shape != (self.grid_spec.height, self.grid_spec.width):
            raise ValueError("probability layers must match grid_spec shape")
        observation = (((free > 0.5) | (occ > 0.5)).astype(np.uint16) * np.uint16(5)).astype(np.uint16)
        last = np.where(observation > 0, int(frame_id), -1).astype(np.int32)
        structural = np.clip(0.55 * occ + 0.35 * endpoint - 0.25 * free, 0.0, 1.0).astype(np.float32)
        self._last_state = VerticalMapState(
            p_free=np.clip(free, 0.0, 1.0).astype(np.float32),
            p_occupied=np.clip(occ, 0.0, 1.0).astype(np.float32),
            p_unknown=np.clip(unk, 0.0, 1.0).astype(np.float32),
            p_endpoint=np.clip(endpoint, 0.0, 1.0).astype(np.float32),
            p_structural_wall=structural,
            observation_count=observation,
            last_observed_frame=last,
        )
        return self._last_state

    def from_masks(
        self,
        *,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray | None = None,
        unknown_mask: np.ndarray | None = None,
        frame_id: int = 0,
    ) -> VerticalMapState:
        free = np.asarray(observed_free_mask, dtype=bool)
        occ = np.zeros_like(free, dtype=bool) if obstacle_mask is None else np.asarray(obstacle_mask, dtype=bool)
        unk = ~(free | occ) if unknown_mask is None else np.asarray(unknown_mask, dtype=bool)
        p_free = np.where(free, 0.90, 0.05).astype(np.float32)
        p_occ = np.where(occ & ~free, 0.85, 0.05).astype(np.float32)
        p_unknown = np.where(unk, 0.92, 0.05).astype(np.float32)
        p_endpoint = np.where(occ & ~free, 0.75, 0.05).astype(np.float32)
        return self.from_probability_layers(
            p_free=p_free,
            p_occupied=p_occ,
            p_unknown=p_unknown,
            p_endpoint=p_endpoint,
            frame_id=int(frame_id),
        )

    @property
    def last_state(self) -> VerticalMapState:
        return self._last_state

    def _world_to_voxel(self, point_world: np.ndarray) -> tuple[int, int, int] | None:
        row, col = world_to_grid(float(point_world[0]), float(point_world[1]), self.grid_spec)
        if not in_bounds(row, col, (self.grid_spec.height, self.grid_spec.width)):
            return None
        z_idx = int(math.floor((float(point_world[2]) - float(self.map_config.z_min_m)) / max(float(self.map_config.z_resolution_m), 1e-9)))
        if z_idx < 0 or z_idx >= self.free_log_odds.shape[0]:
            return None
        return z_idx, row, col

    def _project(self, frame_id: int) -> VerticalMapState:
        pcfg = self.projection_config
        observed_z = self.observation_count_3d > 0
        p_free_z = (1.0 - logodds_to_prob(self.free_log_odds)) * observed_z.astype(np.float32)
        p_occ_z = logodds_to_prob(self.occupied_log_odds) * observed_z.astype(np.float32)
        free_z = (p_free_z >= float(pcfg.through_ray_free_confidence)) & observed_z
        occ_z = (p_occ_z >= 0.55) & observed_z
        free_count = np.sum(free_z, axis=0)
        occ_count = np.sum(occ_z, axis=0)
        obs_count = np.sum(observed_z, axis=0)
        z_count = max(1, self.free_log_odds.shape[0])
        free_ratio = free_count.astype(np.float32) / float(z_count)
        occ_ratio = occ_count.astype(np.float32) / float(z_count)
        unknown_ratio = (z_count - obs_count).astype(np.float32) / float(z_count)
        p_free = np.max(p_free_z, axis=0)
        p_free = np.where(
            free_count >= int(pcfg.min_free_count_for_xy_free),
            np.maximum(p_free, float(pcfg.through_ray_free_confidence)),
            p_free * 0.45,
        )
        endpoint = logodds_to_prob(self.endpoint_log_odds)
        occupied_ok = (
            (occ_count >= int(pcfg.min_occupied_count_for_xy_occupied))
            & (occ_ratio >= float(pcfg.occupied_height_ratio_threshold))
            & (free_count == 0)
        )
        p_occupied = np.where(occupied_ok, np.maximum(np.max(p_occ_z, axis=0), endpoint), np.max(p_occ_z, axis=0) * 0.35)
        p_unknown = np.clip(unknown_ratio, 0.0, 1.0)
        p_unknown = np.where((occ_count > 0) & (free_count == 0) & ~occupied_ok, np.maximum(p_unknown, 0.70), p_unknown)
        p_unknown = np.where(free_count > 0, np.minimum(p_unknown, 0.35), p_unknown)
        structural = np.clip(
            0.45 * p_occupied
            + 0.25 * endpoint
            + 0.20 * occ_ratio
            - 0.25 * p_free
            - 0.10 * free_ratio,
            0.0,
            1.0,
        )
        observation_count = np.asarray(obs_count, dtype=np.uint16)
        last_observed = np.max(self.last_observed_frame_3d, axis=0).astype(np.int32)
        last_observed = np.where(observation_count > 0, last_observed, -1).astype(np.int32)
        return VerticalMapState(
            p_free=np.clip(p_free, 0.0, 1.0).astype(np.float32),
            p_occupied=np.clip(p_occupied, 0.0, 1.0).astype(np.float32),
            p_unknown=np.clip(p_unknown, 0.0, 1.0).astype(np.float32),
            p_endpoint=np.clip(endpoint, 0.0, 1.0).astype(np.float32),
            p_structural_wall=structural.astype(np.float32),
            observation_count=observation_count,
            last_observed_frame=np.where(last_observed >= 0, last_observed, int(frame_id)).astype(np.int32),
        )
