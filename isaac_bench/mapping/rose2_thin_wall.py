from __future__ import annotations

from typing import Mapping

import numpy as np

from isaac_bench.mapping.rose2_separator_detection import detect_thin_wall_separator_candidates as _detect


def detect_thin_wall_separators(
    *,
    vertical_free: np.ndarray,
    vertical_observed: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    wall_confidence_map: np.ndarray | None = None,
    resolution_m: float = 0.05,
    min_length_m: float = 0.35,
    max_width_m: float = 0.30,
    free_support_band_m: float = 0.30,
    min_free_support_ratio: float = 0.20,
    min_aspect_ratio: float = 3.0,
    max_unknown_overlap_ratio: float = 0.25,
) -> tuple[list[dict], np.ndarray, dict]:
    """Promote thin room separators from observed non-free vertical profiles.

    The input vertical-free mask is not modified.  A cell can become a
    candidate only when it is observed, non-free, and not unknown, or when it
    has occupied support.  This is the source-faithful T0 path used to catch
    thin walls that are visible in the 0.2--2.0 m vertical profile but missing
    from the ordinary occupied wall-line map.
    """

    lines, mask, debug = _detect(
        vertical_free=vertical_free,
        vertical_observed=vertical_observed,
        occupied=occupied,
        unknown=unknown,
        wall_confidence_map=wall_confidence_map,
        resolution_m=float(resolution_m),
        min_length_m=float(min_length_m),
        max_width_m=float(max_width_m),
        free_support_band_m=float(free_support_band_m),
        min_free_support_ratio=float(min_free_support_ratio),
        min_aspect_ratio=float(min_aspect_ratio),
    )
    converted = []
    for line in lines:
        row = dict(line)
        row["source"] = "thin_wall_from_nonfree_observed"
        converted.append(row)
    debug = dict(debug)
    debug["source"] = "thin_wall_from_nonfree_observed"
    debug["max_unknown_overlap_ratio"] = float(max_unknown_overlap_ratio)
    debug["nonfree_observed_separator_count"] = int(len(converted))
    return converted, mask, debug


def summarize_thin_wall_support(line: Mapping[str, object]) -> dict:
    return {
        "source": str(line.get("source", "")),
        "length_m": float(line.get("length_m", 0.0) or 0.0),
        "width_m": float(line.get("width_m", 0.0) or 0.0),
        "free_support_ratio": max(
            float(line.get("free_support_left", 0.0) or 0.0),
            float(line.get("free_support_right", 0.0) or 0.0),
            float(line.get("free_support_top", 0.0) or 0.0),
            float(line.get("free_support_bottom", 0.0) or 0.0),
        ),
        "occupied_support_ratio": float(line.get("wall_confidence_mean", 0.0) or 0.0),
    }
