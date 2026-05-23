from __future__ import annotations

from collections import OrderedDict
import math
from typing import Iterable, Mapping, Set

import numpy as np

from .config import StructuralWallConfig, VisibilityConfig
from .data_types import StructuralMap
from .utils import in_bounds


GridCell = tuple[int, int]


class VisibilityComputer:
    def __init__(
        self,
        config: VisibilityConfig | Mapping[str, object] | None = None,
        structural_config: StructuralWallConfig | Mapping[str, object] | None = None,
        resolution_m: float = 0.05,
    ):
        self.config = config if isinstance(config, VisibilityConfig) else VisibilityConfig.from_mapping(config or {})
        self.structural_config = (
            structural_config
            if isinstance(structural_config, StructuralWallConfig)
            else StructuralWallConfig.from_mapping(structural_config or {})
        )
        self.resolution_m = float(resolution_m)
        self._cache: OrderedDict[tuple[int, int, int], Set[GridCell]] = OrderedDict()

    def set_frame_id(self, frame_id: int, full_recompute_interval: int = 10) -> None:
        self._version = int(frame_id) // max(1, int(full_recompute_interval))

    def compute_visibility_signature(
        self,
        seed_cells: Iterable[GridCell],
        structural_map: StructuralMap,
        max_range_m: float | None = None,
        ray_count: int | None = None,
        ray_step_m: float | None = None,
    ) -> Set[GridCell]:
        out: set[GridCell] = set()
        for seed in seed_cells:
            out |= self._visibility_from_seed(
                (int(seed[0]), int(seed[1])),
                structural_map,
                max_range_m=float(max_range_m if max_range_m is not None else self.config.ray_max_range_m),
                ray_count=int(ray_count if ray_count is not None else self.config.ray_count_per_node),
                ray_step_m=float(ray_step_m if ray_step_m is not None else self.config.ray_step_m),
            )
            limit = int(self.config.visibility_cell_sample_limit)
            if len(out) > limit:
                out = set(list(out)[:: max(1, len(out) // limit)][:limit])
                break
        return out

    def visibility_drop(self, left_cells: Iterable[GridCell], right_cells: Iterable[GridCell], structural_map: StructuralMap) -> float:
        if not bool(self.config.enabled):
            return 0.0
        left = self.compute_visibility_signature(left_cells, structural_map)
        right = self.compute_visibility_signature(right_cells, structural_map)
        score = 1.0 - jaccard(left, right, float(self.config.jaccard_epsilon))
        if len(left) < int(self.config.min_visibility_cells) or len(right) < int(self.config.min_visibility_cells):
            score *= 0.5
        return float(np.clip(score, 0.0, 1.0))

    def _visibility_from_seed(
        self,
        seed: GridCell,
        structural_map: StructuralMap,
        *,
        max_range_m: float,
        ray_count: int,
        ray_step_m: float,
    ) -> Set[GridCell]:
        version = int(getattr(self, "_version", 0))
        key = (int(seed[0]), int(seed[1]), version)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return set(cached)
        visible: set[GridCell] = set()
        p_wall = np.asarray(structural_map.p_wall, dtype=np.float32)
        hard = np.asarray(structural_map.hard_wall_mask, dtype=bool)
        p_unknown = np.asarray(structural_map.p_unknown, dtype=np.float32)
        shape = p_wall.shape
        if not in_bounds(seed[0], seed[1], shape):
            return visible
        step_px = max(float(ray_step_m) / max(self.resolution_m, 1e-9), 0.25)
        max_px = float(max_range_m) / max(self.resolution_m, 1e-9)
        angle_span = math.radians(float(self.config.ray_angle_span_deg))
        for idx in range(max(1, int(ray_count))):
            angle = (float(idx) / max(1, int(ray_count))) * angle_span
            dr = math.sin(angle)
            dc = math.cos(angle)
            cur = step_px
            while cur <= max_px:
                row = int(round(float(seed[0]) + dr * cur))
                col = int(round(float(seed[1]) + dc * cur))
                if not in_bounds(row, col, shape):
                    break
                if hard[row, col] or float(p_wall[row, col]) >= float(self.structural_config.hard_wall_probability_threshold):
                    break
                if float(p_unknown[row, col]) >= float(self.structural_config.unknown_probability_threshold):
                    break
                visible.add((row, col))
                cur += step_px
        self._cache[key] = set(visible)
        self._cache.move_to_end(key)
        while len(self._cache) > int(self.config.visibility_cache_size):
            self._cache.popitem(last=False)
        return visible


def compute_visibility_signature(
    seed_cells: list[GridCell],
    structural_map: StructuralMap,
    max_range_m: float,
    ray_count: int,
    ray_step_m: float,
) -> Set[GridCell]:
    comp = VisibilityComputer(resolution_m=0.05)
    return comp.compute_visibility_signature(seed_cells, structural_map, max_range_m, ray_count, ray_step_m)


def jaccard(a: Set[GridCell], b: Set[GridCell], eps: float = 1e-6) -> float:
    return float(len(a & b) / (len(a | b) + float(eps)))

