from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .utils import label_components, relabel_compact


@dataclass
class RoomLabelV3Result:
    raw_room_labels: np.ndarray
    final_room_labels: np.ndarray
    architectural_room_label_map: np.ndarray
    functional_zone_label_map: np.ndarray
    debug: dict = field(default_factory=dict)


def label_rooms_from_accepted_boundaries(
    *,
    roomseg_free_clean: np.ndarray,
    accepted_virtual_boundary_map: np.ndarray,
    structural_wall_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    config: Mapping[str, object] | object | None,
) -> RoomLabelV3Result:
    _ = structural_wall_clean, config
    free = np.asarray(roomseg_free_clean, dtype=bool)
    accepted = np.asarray(accepted_virtual_boundary_map, dtype=bool) & free
    unknown = np.asarray(unknown_clean, dtype=bool)
    room_core = free & ~accepted
    raw_labels, _count = label_components(room_core, 4)
    raw_labels = relabel_compact(raw_labels)
    final_labels = _merge_tiny_fragments(
        raw_labels,
        free=free,
        min_area_cells=max(1, int(round(0.20 / max(float(resolution_m) ** 2, 1e-9)))),
    )
    final_labels[~free] = 0
    final_labels[unknown] = 0
    final_labels[accepted] = 0
    final_labels = relabel_compact(final_labels)
    labels_outside_free = int(np.count_nonzero((final_labels > 0) & ~free))
    labels_in_unknown = int(np.count_nonzero((final_labels > 0) & unknown))
    if labels_outside_free:
        raise AssertionError("room labels escaped roomseg_free_clean")
    if labels_in_unknown:
        raise AssertionError("room labels overlap roomseg_unknown_clean")
    room_count = int(len([v for v in np.unique(final_labels) if int(v) > 0]))
    return RoomLabelV3Result(
        raw_room_labels=raw_labels.astype(np.int32),
        final_room_labels=final_labels.astype(np.int32),
        architectural_room_label_map=final_labels.astype(np.int32),
        functional_zone_label_map=np.zeros_like(final_labels, dtype=np.int32),
        debug={
            "algorithm": "roomseg_evidence_line_closure_v3",
            "connected_components_formula": "raw_room_labels = connected_components(roomseg_free_clean & ~accepted_virtual_boundary_map, connectivity=4)",
            "raw_component_count": int(len([v for v in np.unique(raw_labels) if int(v) > 0])),
            "final_room_count": int(room_count),
            "accepted_virtual_boundary_cells": int(np.count_nonzero(accepted)),
            "labels_outside_free_cells": int(labels_outside_free),
            "labels_in_unknown_cells": int(labels_in_unknown),
            "largest_room_area_m2": _largest_label_area_m2(final_labels, float(resolution_m)),
        },
    )


def _merge_tiny_fragments(labels: np.ndarray, *, free: np.ndarray, min_area_cells: int) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if int(min_area_cells) <= 1:
        return arr
    labels_present = [int(v) for v in np.unique(arr) if int(v) > 0]
    if len(labels_present) <= 1:
        return arr
    for label in labels_present:
        comp = arr == int(label)
        if int(np.count_nonzero(comp)) >= int(min_area_cells):
            continue
        neighbor_labels = arr[_neighbor_ring(comp) & free]
        neighbor_labels = [int(v) for v in neighbor_labels.tolist() if int(v) > 0 and int(v) != int(label)]
        if not neighbor_labels:
            continue
        replacement = max(set(neighbor_labels), key=neighbor_labels.count)
        arr[comp] = int(replacement)
    return relabel_compact(arr)


def _neighbor_ring(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    padded = np.pad(src, 1, mode="constant", constant_values=False)
    ring = (
        padded[0:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, 0:-2]
        | padded[1:-1, 2:]
    )
    return ring & ~src


def _largest_label_area_m2(labels: np.ndarray, resolution_m: float) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    best = 0
    for label in np.unique(arr):
        if int(label) <= 0:
            continue
        best = max(best, int(np.count_nonzero(arr == int(label))))
    return float(best) * float(resolution_m) ** 2
