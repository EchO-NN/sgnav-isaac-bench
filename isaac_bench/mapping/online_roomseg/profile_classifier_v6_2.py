from __future__ import annotations

from typing import Mapping

import numpy as np

from .height_profile_v6_2 import ColumnProfileClassification, HeightProfileState


def classify_height_profile_v6_2(
    profile: HeightProfileState,
    config: Mapping[str, object] | None = None,
) -> ColumnProfileClassification:
    cfg = dict(config or {})
    classifier_cfg = dict(cfg.get("classifier", cfg) or {})
    wall_ratio_min = float(classifier_cfg.get("wall_endpoint_ratio_min", 0.95))
    door_lower_min = float(classifier_cfg.get("door_lower_free_ratio_min", 0.95))
    door_upper_min = float(classifier_cfg.get("door_upper_occupied_ratio_min", 0.95))
    transition_min_z = float(profile.floor_ceiling.floor_z) + float(classifier_cfg.get("door_transition_min_height_m", 1.80))
    min_upper_span_m = float(classifier_cfg.get("min_upper_span_m", 0.15))
    min_lower_span_m = float(classifier_cfg.get("min_lower_span_m", 1.20))

    free_supported = np.asarray(profile.free_supported, dtype=bool)
    occupied_supported = np.asarray(profile.occupied_supported, dtype=bool)
    if free_supported.ndim != 3 or free_supported.shape != occupied_supported.shape:
        raise ValueError("HeightProfileState support arrays must have shape (Z, H, W)")
    k_bins = int(free_supported.shape[0])
    if k_bins <= 0:
        raise ValueError("HeightProfileState must contain at least one z bin")

    wall_endpoint_ratio = np.sum(occupied_supported, axis=0, dtype=np.float32) / float(k_bins)
    wall_mask_raw = wall_endpoint_ratio >= wall_ratio_min
    free_ratio_all = np.sum(free_supported, axis=0, dtype=np.float32) / float(k_bins)

    best_lower_free = np.zeros(free_supported.shape[1:], dtype=np.float32)
    best_upper_occ = np.zeros_like(best_lower_free)
    best_transition_z = np.zeros_like(best_lower_free, dtype=np.float32)
    best_score = np.full_like(best_lower_free, -1.0, dtype=np.float32)
    z_edges = np.asarray(profile.z_edges_m, dtype=np.float32)

    for split_idx in range(1, k_bins):
        transition_z = float(z_edges[split_idx])
        lower_span = transition_z - float(z_edges[0])
        upper_span = float(z_edges[-1]) - transition_z
        if transition_z < transition_min_z or lower_span < min_lower_span_m or upper_span < min_upper_span_m:
            continue
        lower_free_ratio = np.sum(free_supported[:split_idx], axis=0, dtype=np.float32) / float(split_idx)
        upper_occ_ratio = np.sum(occupied_supported[split_idx:], axis=0, dtype=np.float32) / float(k_bins - split_idx)
        score = np.minimum(lower_free_ratio, upper_occ_ratio)
        better = score > best_score
        best_score[better] = score[better]
        best_lower_free[better] = lower_free_ratio[better]
        best_upper_occ[better] = upper_occ_ratio[better]
        best_transition_z[better] = float(transition_z)

    door_condition = (
        (best_lower_free >= door_lower_min)
        & (best_upper_occ >= door_upper_min)
        & (best_transition_z >= transition_min_z)
    )
    door_mask_raw = door_condition & ~wall_mask_raw
    free_mask_raw = (free_ratio_all >= 0.50) & ~wall_mask_raw
    unknown_mask = ~(wall_mask_raw | door_mask_raw | free_mask_raw)
    room_boundary = wall_mask_raw | door_mask_raw
    debug = {
        "source": "profile_classifier_v6_2",
        "z_bin_count": int(k_bins),
        "wall_endpoint_ratio_min": float(wall_ratio_min),
        "door_lower_free_ratio_min": float(door_lower_min),
        "door_upper_occupied_ratio_min": float(door_upper_min),
        "door_transition_min_z": float(transition_min_z),
        "min_lower_span_m": float(min_lower_span_m),
        "min_upper_span_m": float(min_upper_span_m),
        "wall_mask_raw_cells": int(np.count_nonzero(wall_mask_raw)),
        "door_mask_raw_cells": int(np.count_nonzero(door_mask_raw)),
        "free_mask_raw_cells": int(np.count_nonzero(free_mask_raw)),
        "unknown_mask_cells": int(np.count_nonzero(unknown_mask)),
        "wall_door_overlap_suppressed_cells": int(np.count_nonzero(door_condition & wall_mask_raw)),
    }
    return ColumnProfileClassification(
        wall_mask_raw=wall_mask_raw.astype(bool),
        door_mask_raw=door_mask_raw.astype(bool),
        free_mask_raw=free_mask_raw.astype(bool),
        unknown_mask=unknown_mask.astype(bool),
        wall_endpoint_ratio=wall_endpoint_ratio.astype(np.float32),
        best_door_lower_free_ratio=best_lower_free.astype(np.float32),
        best_door_upper_occupied_ratio=best_upper_occ.astype(np.float32),
        best_door_transition_z=best_transition_z.astype(np.float32),
        wall_mask=wall_mask_raw.astype(bool),
        door_mask=door_mask_raw.astype(bool),
        wall_line_support_mask=room_boundary.astype(bool),
        room_boundary_mask=room_boundary.astype(bool),
        debug=debug,
    )
