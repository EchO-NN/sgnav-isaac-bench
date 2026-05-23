from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import FreeSpaceConfig, StructuralWallConfig
from .data_types import FreeSpaceState, StructuralMap
from .utils import dilate, distance_transform, erode, radius_cells, remove_small_components


class FreeSpaceExtractor:
    def __init__(
        self,
        config: FreeSpaceConfig | Mapping[str, object] | None = None,
        structural_config: StructuralWallConfig | Mapping[str, object] | None = None,
        resolution_m: float = 0.05,
    ):
        self.config = config if isinstance(config, FreeSpaceConfig) else FreeSpaceConfig.from_mapping(config or {})
        self.structural_config = (
            structural_config
            if isinstance(structural_config, StructuralWallConfig)
            else StructuralWallConfig.from_mapping(structural_config or {})
        )
        self.resolution_m = float(resolution_m)

    def extract(self, structural_map: StructuralMap) -> FreeSpaceState:
        cfg = self.config
        scfg = self.structural_config
        observed_free = (
            (np.asarray(structural_map.p_free, dtype=np.float32) >= float(scfg.free_probability_threshold))
            & (np.asarray(structural_map.p_unknown, dtype=np.float32) < float(scfg.unknown_probability_threshold))
        )
        observed_free &= ~np.asarray(structural_map.hard_wall_mask, dtype=bool)
        min_cells = max(1, int(round(float(cfg.min_free_component_area_m2) / max(self.resolution_m * self.resolution_m, 1e-9))))
        observed_free = remove_small_components(observed_free, min_cells=min_cells, connectivity=8)
        default_nav_erosion_m = max(float(cfg.erosion_radius_m), float(cfg.robot_radius_m) + float(cfg.safety_margin_m))
        roomseg_erosion_m = float(cfg.roomseg_erosion_radius_m) if cfg.roomseg_erosion_radius_m is not None else float(default_nav_erosion_m)
        navigation_erosion_m = (
            float(cfg.navigation_erosion_radius_m) if cfg.navigation_erosion_radius_m is not None else float(default_nav_erosion_m)
        )
        roomseg_erode_radius = radius_cells(roomseg_erosion_m, self.resolution_m)
        navigation_erode_radius = radius_cells(navigation_erosion_m, self.resolution_m)
        free_eroded = erode(observed_free, roomseg_erode_radius)
        if not np.any(free_eroded) and np.any(observed_free):
            free_eroded = observed_free.copy()
        navigation_free_eroded = erode(observed_free, navigation_erode_radius)
        if not np.any(navigation_free_eroded) and np.any(observed_free):
            navigation_free_eroded = observed_free.copy()
        unknown_high = np.asarray(structural_map.p_unknown, dtype=np.float32) >= float(scfg.unknown_probability_threshold)
        frontier = observed_free & dilate(unknown_high, radius_cells(float(cfg.frontier_band_m), self.resolution_m))
        distance_input = free_eroded.copy()
        distance_m = distance_transform(distance_input) * float(self.resolution_m)
        debug = {
            "distance_boundary_mask": (~distance_input).astype(np.uint8),
            "unknown_distance_boundary": (unknown_high & bool(cfg.unknown_as_boundary_for_distance)).astype(np.uint8),
            "roomseg_erosion_radius_m": np.asarray([roomseg_erosion_m], dtype=np.float32),
            "navigation_erosion_radius_m": np.asarray([navigation_erosion_m], dtype=np.float32),
            "navigation_free_eroded_mask": navigation_free_eroded.astype(np.uint8),
        }
        return FreeSpaceState(
            observed_free_mask=observed_free.astype(bool),
            free_eroded_mask=free_eroded.astype(bool),
            frontier_mask=frontier.astype(bool),
            distance_transform_m=distance_m.astype(np.float32),
            debug=debug,
        )
