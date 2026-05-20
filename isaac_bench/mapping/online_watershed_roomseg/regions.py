from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import component_metrics, relabel_compact


@dataclass
class WatershedRegionConfig:
    min_region_area_m2: float = 0.40
    merge_small_regions: bool = True
    corridor_overlap_min_ratio: float = 0.15

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WatershedRegionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WatershedRegionResult:
    raw_labels: np.ndarray
    final_labels: np.ndarray
    region_type_map: np.ndarray
    region_infos: list[dict]
    graph: dict
    report: dict


REGION_TYPE_IDS = {
    "unassigned_free": 1,
    "confirmed_room": 2,
    "provisional_room": 3,
    "corridor": 4,
}


def run_marker_controlled_watershed(
    *,
    free_clean: np.ndarray,
    elevation: np.ndarray,
    marker_labels: np.ndarray,
) -> np.ndarray:
    free = np.asarray(free_clean, dtype=bool)
    elev = np.asarray(elevation, dtype=np.float32)
    markers = np.asarray(marker_labels, dtype=np.int32)
    if free.shape != elev.shape or free.shape != markers.shape:
        raise ValueError("watershed inputs must have matching HxW shapes")
    labels = np.zeros_like(markers, dtype=np.int32)
    heap: list[tuple[float, int, int, int]] = []
    order = 0
    for r, c in zip(*np.nonzero((markers > 0) & free)):
        label = int(markers[int(r), int(c)])
        labels[int(r), int(c)] = label
        heapq.heappush(heap, (float(elev[int(r), int(c)]), order, int(r), int(c)))
        order += 1
    if not heap and np.any(free):
        row, col = np.unravel_index(int(np.argmin(np.where(free, elev, np.inf))), elev.shape)
        labels[int(row), int(col)] = 1
        heapq.heappush(heap, (float(elev[int(row), int(col)]), order, int(row), int(col)))
    h, w = free.shape
    while heap:
        _value, _order, r, c = heapq.heappop(heap)
        label = int(labels[r, c])
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not free[nr, nc] or labels[nr, nc] > 0:
                continue
            labels[nr, nc] = label
            heapq.heappush(heap, (float(elev[nr, nc]), order, nr, nc))
            order += 1
    return labels.astype(np.int32)


def postprocess_watershed_regions(
    *,
    raw_labels: np.ndarray,
    free_clean: np.ndarray,
    marker_labels: np.ndarray,
    seed_type_by_label: Mapping[int, str],
    corridor_core: np.ndarray,
    dist_free_extent_m: np.ndarray,
    resolution_m: float,
    config: WatershedRegionConfig,
) -> WatershedRegionResult:
    raw = relabel_compact(np.asarray(raw_labels, dtype=np.int32))
    labels = raw.copy()
    seed_types = {int(k): str(v) for k, v in seed_type_by_label.items()}
    type_by_label = _classify_regions(raw, marker_labels, seed_types, corridor_core, config)
    if bool(config.merge_small_regions):
        labels = _merge_small_regions(labels, free_clean, type_by_label, resolution_m, config)
        labels = relabel_compact(labels)
        type_by_label = _reclassify_after_merge(labels, raw, type_by_label, free_clean, marker_labels, seed_types, corridor_core, config)
    graph = build_region_graph(labels)
    infos = _region_infos(labels, type_by_label, resolution_m, dist_free_extent_m, graph)
    region_type_map = np.zeros_like(labels, dtype=np.int32)
    for label, kind in type_by_label.items():
        region_type_map[labels == int(label)] = int(REGION_TYPE_IDS.get(str(kind), 1))
    report = {
        "raw_region_count": int(len([v for v in np.unique(raw) if int(v) > 0])),
        "final_region_count": int(len([v for v in np.unique(labels) if int(v) > 0])),
        "region_type_counts": dict(_type_counts(type_by_label)),
        "marker_controlled_watershed": True,
    }
    return WatershedRegionResult(raw, labels, region_type_map, infos, graph, report)


def build_region_graph(labels: np.ndarray) -> dict:
    arr = np.asarray(labels, dtype=np.int32)
    edges: set[tuple[int, int]] = set()
    pairs = (
        (arr[:-1, :], arr[1:, :]),
        (arr[:, :-1], arr[:, 1:]),
    )
    for a_arr, b_arr in pairs:
        mask = (a_arr > 0) & (b_arr > 0) & (a_arr != b_arr)
        for a, b in zip(a_arr[mask].tolist(), b_arr[mask].tolist()):
            edge = tuple(sorted((int(a), int(b))))
            if edge[0] > 0 and edge[1] > 0:
                edges.add(edge)
    labels_out = [int(v) for v in sorted(np.unique(arr).tolist()) if int(v) > 0]
    return {
        "nodes": [{"label": int(label)} for label in labels_out],
        "edges": [{"a": int(a), "b": int(b)} for a, b in sorted(edges)],
    }


def build_frontier_room_context(
    *,
    frontier_clusters: Sequence[object] | None,
    final_labels: np.ndarray,
    region_infos: Sequence[Mapping[str, object]],
    free_clean: np.ndarray,
    resolution_m: float,
    search_radius_m: float = 0.45,
) -> dict:
    type_by_label = {int(info["label"]): str(info.get("region_type", "unassigned_free")) for info in region_infos}
    radius = max(1, int(round(float(search_radius_m) / max(1e-6, float(resolution_m)))))
    out: list[dict] = []
    h, w = final_labels.shape
    for idx, frontier in enumerate(frontier_clusters or []):
        cells = _frontier_cells(frontier)
        mask = np.zeros_like(free_clean, dtype=bool)
        for r, c in cells:
            if 0 <= int(r) < h and 0 <= int(c) < w:
                mask[int(r), int(c)] = True
        nearby = ndimage.binary_dilation(mask, iterations=radius) & np.asarray(free_clean, dtype=bool)
        labels = [int(v) for v in np.unique(final_labels[nearby]) if int(v) > 0]
        counts = {int(label): int(np.count_nonzero(nearby & (final_labels == int(label)))) for label in labels}
        main_label = max(counts, key=counts.get) if counts else 0
        region_type = type_by_label.get(int(main_label), "unassigned_free")
        out.append(
            {
                "frontier_index": int(idx),
                "frontier_id": _frontier_id(frontier, idx),
                "room_label": int(main_label),
                "region_type": region_type,
                "is_room_entry": bool(region_type in {"confirmed_room", "provisional_room"}),
                "is_corridor_frontier": bool(region_type == "corridor"),
                "nearby_region_counts": {str(k): int(v) for k, v in counts.items()},
            }
        )
    return {"frontiers": out, "search_radius_m": float(search_radius_m)}


def _classify_regions(
    labels: np.ndarray,
    marker_labels: np.ndarray,
    seed_type_by_label: Mapping[int, str],
    corridor_core: np.ndarray,
    config: WatershedRegionConfig,
) -> dict[int, str]:
    out: dict[int, str] = {}
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        marker_values = [int(v) for v in np.unique(marker_labels[mask]) if int(v) > 0]
        seed_types = {str(seed_type_by_label.get(v, "")) for v in marker_values}
        corridor_ratio = float(np.count_nonzero(mask & corridor_core)) / float(max(1, np.count_nonzero(mask)))
        if "corridor" in seed_types or corridor_ratio >= float(config.corridor_overlap_min_ratio):
            kind = "corridor"
        elif "confirmed_room" in seed_types:
            kind = "confirmed_room"
        elif "provisional_room" in seed_types:
            kind = "provisional_room"
        else:
            kind = "unassigned_free"
        out[int(label)] = kind
    return out


def _merge_small_regions(
    labels: np.ndarray,
    free_clean: np.ndarray,
    type_by_label: Mapping[int, str],
    resolution_m: float,
    config: WatershedRegionConfig,
) -> np.ndarray:
    out = labels.copy()
    min_cells = max(1, int(round(float(config.min_region_area_m2) / max(1e-6, float(resolution_m) ** 2))))
    graph = build_region_graph(out)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for edge in graph["edges"]:
        a, b = int(edge["a"]), int(edge["b"])
        neighbors[a].add(b)
        neighbors[b].add(a)
    for label in sorted(int(v) for v in np.unique(out) if int(v) > 0):
        if str(type_by_label.get(label)) == "corridor":
            continue
        mask = out == label
        if int(np.count_nonzero(mask)) >= min_cells:
            continue
        nbrs = sorted(neighbors.get(label, []), key=lambda n: int(np.count_nonzero(out == int(n))), reverse=True)
        if not nbrs:
            continue
        target = next((n for n in nbrs if str(type_by_label.get(n)) != "corridor"), nbrs[0])
        out[mask] = int(target)
    out[~np.asarray(free_clean, dtype=bool)] = 0
    return out.astype(np.int32)


def _reclassify_after_merge(
    labels: np.ndarray,
    raw_labels: np.ndarray,
    old_types: Mapping[int, str],
    free_clean: np.ndarray,
    marker_labels: np.ndarray,
    seed_type_by_label: Mapping[int, str],
    corridor_core: np.ndarray,
    config: WatershedRegionConfig,
) -> dict[int, str]:
    _ = raw_labels, old_types, free_clean
    return _classify_regions(labels, marker_labels, seed_type_by_label, corridor_core, config)


def _region_infos(
    labels: np.ndarray,
    type_by_label: Mapping[int, str],
    resolution_m: float,
    dist_free_extent_m: np.ndarray,
    graph: Mapping[str, object],
) -> list[dict]:
    neighbor_map: dict[int, list[int]] = defaultdict(list)
    for edge in graph.get("edges", []):
        a, b = int(edge["a"]), int(edge["b"])
        neighbor_map[a].append(b)
        neighbor_map[b].append(a)
    out: list[dict] = []
    width_cells = np.asarray(dist_free_extent_m, dtype=np.float32) / max(1e-6, float(resolution_m))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        metrics = component_metrics(mask, float(resolution_m), distance_m=width_cells)
        out.append(
            {
                "label": int(label),
                "region_type": str(type_by_label.get(label, "unassigned_free")),
                "area_cells": int(metrics["area_cells"]),
                "area_m2": float(metrics["area_m2"]),
                "aspect_ratio": float(metrics["aspect_ratio"]),
                "median_width_m": float(metrics["median_width_m"]),
                "neighbors": [int(v) for v in sorted(neighbor_map.get(label, []))],
            }
        )
    return out


def _type_counts(type_by_label: Mapping[int, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for kind in type_by_label.values():
        out[str(kind)] = out.get(str(kind), 0) + 1
    return out


def _frontier_cells(frontier: object) -> list[tuple[int, int]]:
    if isinstance(frontier, Mapping):
        raw = frontier.get("cells", frontier.get("members", []))
    else:
        raw = getattr(frontier, "cells", getattr(frontier, "members", []))
    out: list[tuple[int, int]] = []
    for cell in raw or []:
        if len(cell) >= 2:
            out.append((int(cell[0]), int(cell[1])))
    return out


def _frontier_id(frontier: object, idx: int) -> str:
    if isinstance(frontier, Mapping):
        return str(frontier.get("frontier_id", frontier.get("id", idx)))
    return str(getattr(frontier, "frontier_id", getattr(frontier, "id", idx)))
