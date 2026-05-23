from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping

import numpy as np
from scipy import ndimage

from .utils import label_components, relabel_compact


@dataclass
class RoomLabelV62Result:
    room_label_map: np.ndarray
    raw_room_labels: np.ndarray
    room_core: np.ndarray
    room_cut_mask: np.ndarray
    room_graph_edges: list[dict[str, object]] = field(default_factory=list)
    debug: dict[str, object] = field(default_factory=dict)


def label_rooms_v6_2(
    *,
    free_for_navigation: np.ndarray,
    wall_mask: np.ndarray,
    door_mask: np.ndarray,
    corridor_separator_mask: np.ndarray,
    unknown_mask: np.ndarray,
    resolution_m: float,
    config: Mapping[str, object] | None = None,
    door_confidence_map: np.ndarray | None = None,
) -> RoomLabelV62Result:
    cfg = dict(config or {})
    labels_cfg = dict(cfg.get("labels", cfg) or {})
    connectivity = int(labels_cfg.get("connectivity", 4))
    min_room_area_m2 = float(labels_cfg.get("min_room_area_m2", 0.25))
    merge_small_regions = bool(labels_cfg.get("merge_small_regions", True))
    free_nav = np.asarray(free_for_navigation, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    wall = np.asarray(wall_mask, dtype=bool)
    door = np.asarray(door_mask, dtype=bool) & ~wall
    corridor_sep = np.asarray(corridor_separator_mask, dtype=bool)
    room_cut = wall | door | corridor_sep
    room_core = free_nav & ~room_cut & ~unknown
    raw_labels, _count = label_components(room_core, connectivity)
    raw_labels = relabel_compact(raw_labels)
    final_labels = raw_labels.copy()
    merge_events: list[dict[str, object]] = []
    if merge_small_regions:
        final_labels, merge_events = _merge_small_regions(
            final_labels,
            min_area_m2=min_room_area_m2,
            resolution_m=float(resolution_m),
            connectivity=connectivity,
        )
    final_labels = relabel_compact(final_labels)
    final_labels[unknown] = -1
    final_labels[room_cut & ~unknown] = 0
    edges = _room_graph_edges(
        final_labels,
        door_mask=door,
        corridor_separator_mask=corridor_sep,
        door_confidence_map=door_confidence_map,
    )
    room_count = int(len([v for v in np.unique(final_labels) if int(v) > 0]))
    debug = {
        "source": "room_labeler_v6_2",
        "connectivity": int(connectivity),
        "min_room_area_m2": float(min_room_area_m2),
        "merge_small_regions": bool(merge_small_regions),
        "raw_room_count": int(len([v for v in np.unique(raw_labels) if int(v) > 0])),
        "final_room_count": int(room_count),
        "room_core_cells": int(np.count_nonzero(room_core)),
        "room_cut_cells": int(np.count_nonzero(room_cut)),
        "unknown_label_cells": int(np.count_nonzero(final_labels == -1)),
        "door_boundary_cells": int(np.count_nonzero(door)),
        "wall_boundary_cells": int(np.count_nonzero(wall)),
        "corridor_separator_cells": int(np.count_nonzero(corridor_sep)),
        "room_graph_edge_count": int(len(edges)),
        "merge_events": merge_events[:256],
    }
    return RoomLabelV62Result(
        room_label_map=final_labels.astype(np.int32),
        raw_room_labels=raw_labels.astype(np.int32),
        room_core=room_core.astype(bool),
        room_cut_mask=room_cut.astype(bool),
        room_graph_edges=edges,
        debug=debug,
    )


def _merge_small_regions(
    labels: np.ndarray,
    *,
    min_area_m2: float,
    resolution_m: float,
    connectivity: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.asarray(labels, dtype=np.int32).copy()
    min_cells = max(1, int(np.ceil(float(min_area_m2) / max(1e-9, float(resolution_m) ** 2))))
    structure = np.ones((3, 3), dtype=np.uint8) if int(connectivity) == 8 else np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    events: list[dict[str, object]] = []
    for label in sorted(int(v) for v in np.unique(out) if int(v) > 0):
        comp = out == label
        area = int(np.count_nonzero(comp))
        if area >= min_cells:
            continue
        border = ndimage.binary_dilation(comp, structure=structure) & ~comp
        neighbor_labels, counts = np.unique(out[border & (out > 0)], return_counts=True)
        if neighbor_labels.size > 0:
            target = int(neighbor_labels[int(np.argmax(counts))])
            out[comp] = target
            action = "merged_to_neighbor"
        else:
            target = 0
            out[comp] = 0
            action = "dropped_no_neighbor"
        events.append(
            {
                "label": int(label),
                "area_cells": int(area),
                "target_label": int(target),
                "action": action,
            }
        )
    return out.astype(np.int32), events


def _room_graph_edges(
    labels: np.ndarray,
    *,
    door_mask: np.ndarray,
    corridor_separator_mask: np.ndarray,
    door_confidence_map: np.ndarray | None,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    seen: set[tuple[int, int, str, int]] = set()
    for edge_type, mask in (("door", door_mask), ("corridor_separator", corridor_separator_mask)):
        comp_labels, count = label_components(mask, 8)
        for comp_id in range(1, int(count) + 1):
            comp = comp_labels == comp_id
            adjacent = _adjacent_positive_labels(labels, comp)
            for room_a, room_b in combinations(adjacent, 2):
                key = (int(room_a), int(room_b), edge_type, int(comp_id))
                if key in seen:
                    continue
                seen.add(key)
                confidence = 1.0
                if edge_type == "door" and door_confidence_map is not None and np.any(comp):
                    confidence = float(np.nanmax(np.asarray(door_confidence_map, dtype=np.float32)[comp]))
                edges.append(
                    {
                        "room_a": int(room_a),
                        "room_b": int(room_b),
                        "edge_type": edge_type,
                        "component_id": int(comp_id),
                        "confidence": float(confidence),
                    }
                )
    return edges


def _adjacent_positive_labels(labels: np.ndarray, mask: np.ndarray) -> list[int]:
    dilated = ndimage.binary_dilation(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8))
    values = sorted(int(v) for v in np.unique(np.asarray(labels, dtype=np.int32)[dilated]) if int(v) > 0)
    return values
