from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import component_metrics, dilate, label_components, remove_small_components


@dataclass
class WatershedCorridorConfig:
    enabled: bool = True
    max_width_m: float = 1.40
    min_length_m: float = 1.50
    min_aspect_ratio: float = 2.8
    seed_dilate_radius_m: float = 0.12
    min_component_cells: int = 8

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WatershedCorridorConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WatershedCorridorResult:
    corridor_core: np.ndarray
    corridor_seeds: np.ndarray
    report: dict


def compute_watershed_corridors(
    *,
    free_clean: np.ndarray,
    dist_free_extent_m: np.ndarray,
    resolution_m: float,
    config: WatershedCorridorConfig,
) -> WatershedCorridorResult:
    free = np.asarray(free_clean, dtype=bool)
    if not bool(config.enabled) or not np.any(free):
        empty = np.zeros_like(free, dtype=bool)
        return WatershedCorridorResult(empty, empty, {"enabled": bool(config.enabled), "components": []})

    width_m = 2.0 * np.asarray(dist_free_extent_m, dtype=np.float32)
    skeleton = _skeletonize(free)
    narrow_medial = skeleton & (width_m <= float(config.max_width_m) + 1e-6)
    narrow_medial = remove_small_components(narrow_medial, int(config.min_component_cells), 8)
    labels, count = label_components(narrow_medial, 8)
    core = np.zeros_like(free, dtype=bool)
    report: list[dict] = []
    for label in range(1, int(count) + 1):
        comp = labels == label
        metrics = component_metrics(comp, float(resolution_m))
        rows, cols = np.nonzero(comp)
        local_width = float(np.median(width_m[comp])) if rows.size else 0.0
        keep = bool(
            metrics["length_m"] >= float(config.min_length_m)
            and metrics["aspect_ratio"] >= float(config.min_aspect_ratio)
            and local_width <= float(config.max_width_m) + 1e-6
        )
        if keep:
            core |= comp
        report.append({**metrics, "component": int(label), "median_width_m": float(local_width), "kept": bool(keep)})
    radius = max(0, int(round(float(config.seed_dilate_radius_m) / max(1e-6, float(resolution_m)))))
    seeds = dilate(core, radius) & free
    return WatershedCorridorResult(core.astype(bool), seeds.astype(bool), {"enabled": True, "components": report})


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return np.asarray(skeletonize(src), dtype=bool)
    except Exception:
        dist = ndimage.distance_transform_edt(src)
        maxed = ndimage.maximum_filter(dist, size=3, mode="constant", cval=0.0)
        ridge = src & (dist >= maxed - 1e-6) & (dist > 0.0)
        return ridge.astype(bool)
