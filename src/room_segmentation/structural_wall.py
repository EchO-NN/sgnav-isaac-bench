from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import StructuralWallConfig
from .data_types import StructuralMap, VerticalMapState
from .utils import (
    binary_close,
    binary_open,
    clamp,
    dilate,
    label_components,
    pca_direction_rc,
    radius_cells,
    remove_small_components,
    sigmoid,
)


class StructuralWallEstimator:
    def __init__(self, config: StructuralWallConfig | Mapping[str, object] | None = None, resolution_m: float = 0.05):
        self.config = config if isinstance(config, StructuralWallConfig) else StructuralWallConfig.from_mapping(config or {})
        self.resolution_m = float(resolution_m)

    def update(self, vertical_state: VerticalMapState, existing_2d_map: np.ndarray | None = None) -> StructuralMap:
        cfg = self.config
        p_free = np.asarray(vertical_state.p_free, dtype=np.float32)
        p_occ = np.asarray(vertical_state.p_occupied, dtype=np.float32)
        p_unknown = np.asarray(vertical_state.p_unknown, dtype=np.float32)
        p_endpoint = np.asarray(vertical_state.p_endpoint, dtype=np.float32)
        observation = np.asarray(vertical_state.observation_count, dtype=np.float32)
        if existing_2d_map is not None:
            nav_occ = np.asarray(existing_2d_map)
            if nav_occ.shape == p_occ.shape:
                p_occ = np.maximum(p_occ, (nav_occ > 0).astype(np.float32) * 0.45)

        candidate = (p_occ > 0.55) & (p_unknown < 0.92)
        candidate = remove_small_components(
            candidate,
            min_cells=max(1, int(round(float(cfg.min_structural_component_area_m2) / max(self.resolution_m * self.resolution_m, 1e-9)))),
            connectivity=8,
        )
        close_r = radius_cells(float(cfg.close_kernel_m), self.resolution_m)
        open_r = radius_cells(float(cfg.open_kernel_m), self.resolution_m)
        if close_r:
            candidate = binary_close(candidate, close_r)
        if open_r > 1:
            candidate = binary_open(candidate, open_r)

        line_support = np.zeros_like(p_occ, dtype=np.float32)
        island_penalty = np.zeros_like(p_occ, dtype=np.float32)
        two_side_free = np.zeros_like(p_occ, dtype=np.float32)
        labels, count = label_components(candidate, connectivity=8)
        min_line_cells = max(2, int(round(float(cfg.min_wall_line_length_m) / max(self.resolution_m, 1e-9))))
        max_island_cells = max(1, int(round(float(cfg.max_island_area_m2) / max(self.resolution_m * self.resolution_m, 1e-9))))
        for label in range(1, int(count) + 1):
            comp = labels == label
            rows, cols = np.nonzero(comp)
            if rows.size == 0:
                continue
            cells = list(zip(rows.tolist(), cols.tolist()))
            span = max(int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1))
            direction = pca_direction_rc(cells)
            centered = np.asarray(cells, dtype=np.float32)
            if centered.shape[0] >= 2:
                centered -= np.mean(centered, axis=0, keepdims=True)
                vals = np.linalg.eigvalsh(np.cov(centered.T))
                elongation = float(np.sqrt(max(vals[-1], 1e-6) / max(vals[0], 1e-6))) if vals.size else 1.0
            else:
                elongation = 1.0
            if span >= min_line_cells and elongation >= 2.0:
                line_support[comp] = 1.0
                line_support[dilate(comp, 1) & candidate] = np.maximum(line_support[dilate(comp, 1) & candidate], 0.65)
            if rows.size <= max_island_cells and span < min_line_cells:
                island_penalty[comp] = 1.0
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            n_norm = float(np.linalg.norm(normal))
            if n_norm <= 1e-6:
                continue
            normal /= n_norm
            for row, col in cells:
                support = _two_side_free_score(int(row), int(col), normal, p_free, self.resolution_m)
                two_side_free[row, col] = support

        two_side_free = two_side_free * (1.0 - 0.70 * line_support)
        persistence_score = np.clip(observation / 5.0, 0.0, 1.0)
        base = (
            0.35 * p_occ
            + 0.25 * p_endpoint
            + 0.20 * persistence_score
            + 0.20 * line_support
            - 0.30 * two_side_free
            - 0.20 * island_penalty
        )
        p_wall = sigmoid((base - 0.50) / 0.15).astype(np.float32)
        p_wall = np.where(p_unknown >= float(cfg.unknown_probability_threshold), np.minimum(p_wall, 0.35), p_wall)
        p_wall = np.where(p_free >= float(cfg.free_probability_threshold), p_wall * (1.0 - 0.45 * p_free), p_wall)
        p_wall = np.where(island_penalty > 0.5, np.minimum(p_wall, 0.60), p_wall)
        p_wall = np.where((p_occ > 0.55) & ~candidate, np.minimum(p_wall, 0.50), p_wall)
        p_wall = np.maximum(p_wall, np.asarray(vertical_state.p_structural_wall, dtype=np.float32) * 0.65)
        p_wall = np.clip(p_wall, 0.0, 1.0).astype(np.float32)
        hard_wall = (p_wall >= float(cfg.hard_wall_probability_threshold)) & (p_unknown < float(cfg.unknown_probability_threshold))
        observed_free = (p_free >= float(cfg.free_probability_threshold)) & (p_unknown < float(cfg.unknown_probability_threshold))
        frontier = observed_free & dilate(p_unknown >= float(cfg.unknown_probability_threshold), radius_cells(0.25, self.resolution_m))
        debug = {
            "p_occupied": p_occ.astype(np.float32),
            "wall_candidate_mask": candidate.astype(np.uint8),
            "line_support_score": line_support.astype(np.float32),
            "two_side_free_score": two_side_free.astype(np.float32),
            "island_penalty": island_penalty.astype(np.float32),
            "persistence_score": persistence_score.astype(np.float32),
            "base_wall_score": base.astype(np.float32),
        }
        return StructuralMap(
            p_wall=p_wall,
            p_free=np.clip(p_free, 0.0, 1.0).astype(np.float32),
            p_unknown=np.clip(p_unknown, 0.0, 1.0).astype(np.float32),
            hard_wall_mask=hard_wall.astype(bool),
            observed_free_mask=observed_free.astype(bool),
            frontier_mask=frontier.astype(bool),
            debug=debug,
        )


def _two_side_free_score(row: int, col: int, normal_rc: np.ndarray, p_free: np.ndarray, resolution_m: float) -> float:
    distances_m = (0.15, 0.25, 0.35)
    hits = [False, False]
    h, w = p_free.shape
    for side_idx, sign in enumerate((-1.0, 1.0)):
        for dist_m in distances_m:
            step = float(dist_m) / max(float(resolution_m), 1e-9)
            rr = int(round(float(row) + sign * float(normal_rc[0]) * step))
            cc = int(round(float(col) + sign * float(normal_rc[1]) * step))
            if 0 <= rr < h and 0 <= cc < w and float(p_free[rr, cc]) >= 0.55:
                hits[side_idx] = True
                break
    if hits[0] and hits[1]:
        return 1.0
    if hits[0] or hits[1]:
        return 0.35
    return 0.0


def wall_support_near(p_wall: np.ndarray, cells: list[tuple[int, int]], radius_cells_value: int) -> float:
    if not cells:
        return 0.0
    src = np.asarray(p_wall, dtype=np.float32)
    best = 0.0
    for row, col in cells:
        r0 = max(0, int(row) - radius_cells_value)
        r1 = min(src.shape[0], int(row) + radius_cells_value + 1)
        c0 = max(0, int(col) - radius_cells_value)
        c1 = min(src.shape[1], int(col) + radius_cells_value + 1)
        if r0 < r1 and c0 < c1:
            best = max(best, float(np.max(src[r0:r1, c0:c1])))
    return clamp(best)
