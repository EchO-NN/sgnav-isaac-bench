from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage

from .utils import disk, label_components


@dataclass
class DoorWallRepairV62Result:
    wall_mask: np.ndarray
    door_mask: np.ndarray
    wall_line_support_mask: np.ndarray
    room_boundary_mask: np.ndarray
    debug: dict[str, object] = field(default_factory=dict)


def repair_door_wall_v6_2(
    *,
    wall_mask_raw: np.ndarray,
    door_mask_raw: np.ndarray,
    unknown_mask: np.ndarray,
    resolution_m: float,
    config: Mapping[str, object] | None = None,
) -> DoorWallRepairV62Result:
    cfg = dict(config or {})
    repair_cfg = dict(cfg.get("repair", cfg) or {})
    enabled = bool(repair_cfg.get("enabled", True))
    wall_raw = np.asarray(wall_mask_raw, dtype=bool)
    door_raw = np.asarray(door_mask_raw, dtype=bool) & ~wall_raw
    unknown = np.asarray(unknown_mask, dtype=bool)
    if not enabled:
        wall = wall_raw & ~unknown
        door = door_raw & ~unknown & ~wall
        return _result(wall, door, {"enabled": False, "source": "door_wall_repair_v6_2"})

    max_island_area_m2 = float(repair_cfg.get("remove_island_max_area_m2", 0.02))
    wall = _remove_small_islands(wall_raw, max_area_m2=max_island_area_m2, resolution_m=float(resolution_m))
    door = _remove_small_islands(door_raw, max_area_m2=max_island_area_m2, resolution_m=float(resolution_m))
    removed_wall = int(np.count_nonzero(wall_raw & ~wall))
    removed_door = int(np.count_nonzero(door_raw & ~door))

    combined = (wall | door) & ~unknown
    radius = max(0, int(round(float(repair_cfg.get("close_kernel_m", 0.10)) / max(1e-6, float(resolution_m)))))
    if radius > 0 and np.any(combined):
        combined_closed = ndimage.binary_closing(combined, structure=disk(radius)).astype(bool)
    else:
        combined_closed = combined.copy()
    gap_cells = max(0, int(np.floor(float(repair_cfg.get("fill_gap_max_m", 0.15)) / max(1e-6, float(resolution_m)))))
    if gap_cells > 0:
        combined_closed |= _fill_axis_gaps(combined, max_gap_cells=gap_cells)
    added = combined_closed & ~combined & ~unknown
    if np.any(added):
        door_neighbors = ndimage.convolve(door.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
        wall_neighbors = ndimage.convolve(wall.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
        added_door = added & (door_neighbors > wall_neighbors)
        added_wall = added & ~added_door
        if bool(repair_cfg.get("preserve_door_class", True)):
            door |= added_door
            wall |= added_wall
        else:
            wall |= added
    wall &= ~unknown
    door &= ~unknown
    door &= ~wall
    debug = {
        "enabled": True,
        "source": "door_wall_repair_v6_2",
        "remove_island_max_area_m2": float(max_island_area_m2),
        "close_kernel_m": float(repair_cfg.get("close_kernel_m", 0.10)),
        "close_radius_cells": int(radius),
        "wall_island_removed_cells": int(removed_wall),
        "door_island_removed_cells": int(removed_door),
        "closing_added_cells": int(np.count_nonzero(added)),
        "wall_mask_cells": int(np.count_nonzero(wall)),
        "door_mask_cells": int(np.count_nonzero(door)),
        "wall_door_overlap_cells_after_repair": int(np.count_nonzero(wall & door)),
    }
    return _result(wall, door, debug)


def _result(wall: np.ndarray, door: np.ndarray, debug: dict[str, object]) -> DoorWallRepairV62Result:
    wall_arr = np.asarray(wall, dtype=bool)
    door_arr = np.asarray(door, dtype=bool) & ~wall_arr
    support = wall_arr | door_arr
    return DoorWallRepairV62Result(
        wall_mask=wall_arr.astype(bool),
        door_mask=door_arr.astype(bool),
        wall_line_support_mask=support.astype(bool),
        room_boundary_mask=support.astype(bool),
        debug=debug,
    )


def _remove_small_islands(mask: np.ndarray, *, max_area_m2: float, resolution_m: float) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if float(max_area_m2) <= 0.0 or not np.any(src):
        return src.copy()
    max_cells = int(np.floor(float(max_area_m2) / max(1e-9, float(resolution_m) ** 2)))
    if max_cells <= 0:
        return src.copy()
    labels, count = label_components(src, 8)
    out = src.copy()
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        if int(np.count_nonzero(comp)) <= max_cells:
            out[comp] = False
    return out.astype(bool)


def _fill_axis_gaps(mask: np.ndarray, *, max_gap_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    out = src.copy()
    h, w = src.shape
    for r in range(h):
        cols = np.flatnonzero(src[r])
        for left, right in zip(cols[:-1].tolist(), cols[1:].tolist()):
            gap = int(right) - int(left) - 1
            if 0 < gap <= int(max_gap_cells):
                out[r, int(left) + 1 : int(right)] = True
    for c in range(w):
        rows = np.flatnonzero(src[:, c])
        for top, bottom in zip(rows[:-1].tolist(), rows[1:].tolist()):
            gap = int(bottom) - int(top) - 1
            if 0 < gap <= int(max_gap_cells):
                out[int(top) + 1 : int(bottom), c] = True
    return out.astype(bool)
