from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage


@dataclass
class FrontierStabilityV4Result:
    frontier: np.ndarray
    raw_frontier: np.ndarray
    stable_free: np.ndarray
    stable_unknown: np.ndarray
    rejected_noise: np.ndarray
    rejected_morphology: np.ndarray
    debug: dict = field(default_factory=dict)


def stable_frontier_cells_v4(
    *,
    reachable_free: np.ndarray,
    observed: np.ndarray,
    stable_roomseg_free: np.ndarray,
    occupancy: np.ndarray | None = None,
    roomseg_unknown_clean: np.ndarray | None = None,
    roomseg_free_noise_rejected: np.ndarray | None = None,
    ray_fan_spur_rejected: np.ndarray | None = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: np.ndarray | None = None,
    config: Mapping[str, object] | None = None,
) -> FrontierStabilityV4Result:
    cfg = dict(config or {})
    free = np.asarray(reachable_free, dtype=bool) & np.asarray(stable_roomseg_free, dtype=bool)
    observed_bool = np.asarray(observed, dtype=bool)
    occ = np.zeros_like(free, dtype=bool) if occupancy is None else np.asarray(occupancy, dtype=bool)
    if occ.shape != free.shape or observed_bool.shape != free.shape:
        raise ValueError("frontier v4 masks must have the same HxW shape")
    unknown = (
        np.asarray(roomseg_unknown_clean, dtype=bool)
        if roomseg_unknown_clean is not None and np.asarray(roomseg_unknown_clean).shape == free.shape
        else ((~observed_bool) & ~free & ~occ)
    )
    obstacle_dilated = _disk_dilate(occ, int(obstacle_dilation_radius_cells))
    stable_free = free & ~obstacle_dilated
    unknown_dilated = _disk_dilate(unknown, int(unknown_dilation_radius_cells))
    raw_frontier = stable_free & unknown_dilated
    noise = np.zeros_like(free, dtype=bool)
    for source in (roomseg_free_noise_rejected, ray_fan_spur_rejected):
        if source is None:
            continue
        arr = np.asarray(source, dtype=bool)
        if arr.shape == free.shape:
            noise |= _disk_dilate(arr, int(cfg.get("noise_reject_dilation_cells", 1)))
    frontier = raw_frontier & ~noise
    if exclude_mask is not None:
        excluded = np.asarray(exclude_mask, dtype=bool)
        if excluded.shape != free.shape:
            raise ValueError("exclude_mask and frontier free must have the same shape")
        frontier &= ~excluded
    rejected_noise = raw_frontier & ~frontier
    morph_open_cells = max(0, int(cfg.get("morphology_open_cells", 0)))
    rejected_morphology = np.zeros_like(frontier, dtype=bool)
    if morph_open_cells > 0 and np.any(frontier):
        opened = ndimage.binary_opening(frontier, structure=_disk(morph_open_cells)).astype(bool)
        rejected_morphology = frontier & ~opened
        frontier = opened
    debug = {
        "frontier_v4_enabled": bool(cfg.get("enabled", True)),
        "raw_frontier_cells": int(np.count_nonzero(raw_frontier)),
        "stable_frontier_cells": int(np.count_nonzero(frontier)),
        "stable_free_cells": int(np.count_nonzero(stable_free)),
        "stable_unknown_cells": int(np.count_nonzero(unknown)),
        "noise_rejected_cells": int(np.count_nonzero(rejected_noise)),
        "morphology_rejected_cells": int(np.count_nonzero(rejected_morphology)),
        "obstacle_dilation_radius_cells": int(obstacle_dilation_radius_cells),
        "unknown_dilation_radius_cells": int(unknown_dilation_radius_cells),
    }
    return FrontierStabilityV4Result(
        frontier=frontier.astype(bool),
        raw_frontier=raw_frontier.astype(bool),
        stable_free=stable_free.astype(bool),
        stable_unknown=unknown.astype(bool),
        rejected_noise=rejected_noise.astype(bool),
        rejected_morphology=rejected_morphology.astype(bool),
        debug=debug,
    )


def _disk_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    src = np.asarray(mask, dtype=bool)
    if radius <= 0 or not np.any(src):
        return src.copy()
    return ndimage.binary_dilation(src, structure=_disk(radius)).astype(bool)


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((yy * yy + xx * xx) <= radius * radius).astype(bool)
