from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class WatershedDistanceFields:
    dist_struct_m: np.ndarray
    dist_free_extent_m: np.ndarray
    elevation: np.ndarray


def compute_watershed_distance_fields(
    *,
    free_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    resolution_m: float,
    alpha_narrow: float = 0.35,
) -> WatershedDistanceFields:
    """Build watershed fields from free space and structural wall candidates.

    Unknown cells are intentionally absent from the structural distance
    computation. Only confirmed wall candidates reduce ``dist_struct_m``; the
    final labels are still masked to confirmed free cells.
    """

    free = np.asarray(free_clean, dtype=bool)
    wall_core = np.asarray(wall_candidate_clean, dtype=bool)
    if free.shape != wall_core.shape:
        raise ValueError("free_clean and wall_candidate_clean must have the same HxW shape")
    dist_struct_m = ndimage.distance_transform_edt(~wall_core).astype(np.float32) * float(resolution_m)
    dist_struct_m[~free] = 0.0
    dist_free_extent_m = ndimage.distance_transform_edt(free).astype(np.float32) * float(resolution_m)
    struct_norm = normalize_field(dist_struct_m, mask=free)
    elevation = (-struct_norm + float(alpha_narrow) * (1.0 - struct_norm)).astype(np.float32)
    elevation[~free] = np.inf
    return WatershedDistanceFields(
        dist_struct_m=dist_struct_m.astype(np.float32),
        dist_free_extent_m=dist_free_extent_m.astype(np.float32),
        elevation=elevation.astype(np.float32),
    )


def normalize_field(values: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) if mask is not None else np.isfinite(arr)
    valid &= np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(valid):
        return out
    lo = float(np.min(arr[valid]))
    hi = float(np.max(arr[valid]))
    if hi <= lo + 1e-6:
        out[valid] = 1.0
        return out
    out[valid] = (arr[valid] - lo) / (hi - lo)
    return out
