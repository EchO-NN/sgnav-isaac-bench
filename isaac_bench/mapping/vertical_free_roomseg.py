from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.rose2_source_form import ROSE2SourceResult


VERTICAL_FREE_ROOMSEG_BACKEND = "vertical_free_geodesic_watershed_v1"
VERTICAL_FREE_ROOMSEG_ALGORITHM = "vertical_free_roomseg_v1"
VERTICAL_FREE_ROOMSEG_CONTEXT = "vertical_free_roomseg_v1_vlm"


@dataclass
class VerticalFreeRoomSegConfig:
    resolution_m: float = 0.05
    min_room_area_m2: float = 1.20
    min_room_free_cells: int = 50
    free_close_radius_m: float = 0.05
    free_open_radius_m: float = 0.0
    wall_close_radius_m: float = 0.05
    min_free_component_area_m2: float = 0.25
    fill_tiny_unknown_holes_m2: float = 0.15
    distance_smooth_sigma_cells: float = 1.0
    seed_min_clearance_m: float = 0.45
    seed_h_max_m: float = 0.12
    seed_min_distance_m: float = 0.80
    seed_min_area_m2: float = 0.20
    force_one_seed_per_free_component: bool = True
    connectivity: int = 8
    conflict_boundary_enabled: bool = True
    conflict_boundary_max_clearance_m: float = 0.85
    conflict_min_neighbor_labels: int = 2
    keep_boundary_unlabeled: bool = False
    doorway_partition_enabled: bool = True
    doorway_width_min_m: float = 0.45
    doorway_width_max_m: float = 1.60
    bottleneck_max_clearance_m: float = 0.80
    bottleneck_min_side_area_m2: float = 1.00
    bottleneck_band_radius_m: float = 0.30
    open_region_merge_enabled: bool = True
    open_merge_min_boundary_width_m: float = 1.80
    open_merge_min_contact_length_m: float = 0.60
    open_merge_max_boundary_fraction: float = 0.35
    merge_small_area_m2: float = 1.20
    merge_thin_sliver_area_m2: float = 0.50
    corridor_keep_enabled: bool = True
    corridor_as_room_min_area_m2: float = 1.00
    corridor_as_room_aspect_ratio: float = 4.0
    debug_dump: bool = False
    debug_dir: str = "debug/vertical_free_roomseg"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VerticalFreeRoomSegConfig":
        raw = dict(data or {})
        raw.update({key: value for key, value in overrides.items() if value is not None})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VerticalFreeRoomSegResult:
    room_label_map: np.ndarray
    boundary_map: np.ndarray
    distance_map_m: np.ndarray
    seed_label_map: np.ndarray
    initial_label_map: np.ndarray
    label_map_before_merge: np.ndarray
    label_map_after_merge: np.ndarray
    room_stats: list[dict]
    adjacency_edges: list[dict]
    debug: dict[str, Any] = field(default_factory=dict)


def run_vertical_free_roomseg(
    *,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    resolution_m: float,
    config: VerticalFreeRoomSegConfig | None = None,
) -> VerticalFreeRoomSegResult:
    t0 = time.perf_counter()
    cfg = config or VerticalFreeRoomSegConfig(resolution_m=float(resolution_m))
    cfg.resolution_m = float(resolution_m)
    free_input = np.asarray(observed_free, dtype=bool)
    wall_input = np.asarray(observed_occupied, dtype=bool)
    unknown_input = np.asarray(unknown, dtype=bool)
    if free_input.shape != wall_input.shape or free_input.shape != unknown_input.shape:
        raise ValueError("vertical-free roomseg inputs must have same HxW shape")

    free = free_input & ~unknown_input
    wall = wall_input & ~free & ~unknown_input
    clean_free, clean_wall, clean_unknown, preprocess_debug = preprocess_vertical_free_inputs(free, wall, unknown_input, cfg)
    distance_m, distance_debug = compute_distance_map(clean_free, cfg)
    seeds, seed_regions, seed_debug = compute_room_seeds(clean_free, distance_m, cfg)
    initial_labels, conflict_boundary, conflict_events, flood_debug = priority_geodesic_watershed(clean_free, distance_m, seeds, cfg)
    contact_boundary, contact_debug = label_contact_boundary(initial_labels, clean_free, distance_m, cfg)
    raw_boundary = conflict_boundary | contact_boundary
    doorway_boundary, doorway_cuts, doorway_debug = refine_bottleneck_boundaries(initial_labels, raw_boundary, clean_free, distance_m, cfg)
    edges = build_region_adjacency(initial_labels, doorway_boundary, clean_free, distance_m, cfg)
    before_merge = initial_labels.copy()
    merged_labels, merge_debug = merge_open_regions(initial_labels, edges, cfg)
    small_merged, small_debug = cleanup_small_segments(merged_labels, clean_free, distance_m, cfg)
    final_labels, absorption_debug = absorb_boundary_pixels(small_merged, doorway_boundary, clean_free, cfg)
    final_labels = _relabel_compact(final_labels)
    final_labels[~clean_free] = 0
    final_labels[~free_input] = 0
    final_labels[clean_unknown] = 0
    boundary = doorway_boundary & label_contact_boundary(final_labels, clean_free, distance_m, cfg)[0] & clean_free & free_input
    stats = compute_room_stats(final_labels, distance_m, cfg)
    labels_outside_free = int(np.count_nonzero((final_labels > 0) & ~free_input))
    labels_in_unknown = int(np.count_nonzero((final_labels > 0) & unknown_input))
    debug: dict[str, Any] = {
        "backend": VERTICAL_FREE_ROOMSEG_BACKEND,
        "actual_backend": VERTICAL_FREE_ROOMSEG_BACKEND,
        "source_backend": VERTICAL_FREE_ROOMSEG_BACKEND,
        "roomseg_backend": VERTICAL_FREE_ROOMSEG_BACKEND,
        "algorithm": VERTICAL_FREE_ROOMSEG_ALGORITHM,
        "source": VERTICAL_FREE_ROOMSEG_ALGORITHM,
        "source_repository": None,
        "source_root_required": False,
        "source_provenance": {"available": False, "required_files": [], "files": []},
        "context_source": VERTICAL_FREE_ROOMSEG_CONTEXT,
        "room_map_mode": VERTICAL_FREE_ROOMSEG_CONTEXT,
        "resolution_m": float(cfg.resolution_m),
        "source_form_used": False,
        "source_form_used_for_final": False,
        "legacy_style_used": False,
        "legacy_style_used_for_final": False,
        "silent_fallback_used": False,
        "legacy_connected_component_rooms_used": False,
        "strict_fallback_used": False,
        "free_cells": int(np.count_nonzero(clean_free)),
        "wall_cells": int(np.count_nonzero(clean_wall)),
        "unknown_cells": int(np.count_nonzero(clean_unknown)),
        "seed_count": int(_label_count(seeds)),
        "initial_room_count": int(_label_count(initial_labels)),
        "room_count_before_merge": int(_label_count(before_merge)),
        "room_count_after_merge": int(_label_count(merged_labels)),
        "final_room_count": int(_label_count(final_labels)),
        "room_count": int(_label_count(final_labels)),
        "doorway_boundary_count": int(np.count_nonzero(doorway_boundary)),
        "virtual_boundary_cells": int(np.count_nonzero(boundary)),
        "open_merge_count": int(len(merge_debug.get("merge_operations", []))),
        "small_merge_count": int(len(small_debug.get("small_merge_operations", []))),
        "labels_outside_vertical_free_cells": labels_outside_free,
        "labels_in_unknown_cells": labels_in_unknown,
        "largest_room_area_m2": max([float(s["area_m2"]) for s in stats], default=0.0),
        "room_stats": stats,
        "seed_regions": seed_regions,
        "conflict_events": conflict_events[:200],
        "doorway_cuts": doorway_cuts,
        "adjacency_edges": edges,
        "timing_ms": {},
        "input_free": free_input,
        "input_wall": wall_input,
        "input_unknown": unknown_input,
        "clean_free": clean_free,
        "clean_wall": clean_wall,
        "distance_map_m": distance_m,
        "seed_label_map": seeds,
        "initial_watershed_labels": before_merge,
        "conflict_boundary_map": conflict_boundary,
        "label_contact_boundary_map": contact_boundary,
        "doorway_boundary_map": doorway_boundary,
        "label_map_before_merge": before_merge,
        "label_map_after_open_merge": merged_labels,
        "label_map_after_small_merge": small_merged,
        "final_room_label_map": final_labels,
        "virtual_boundary_map": boundary,
    }
    debug.update(preprocess_debug)
    debug.update(distance_debug)
    debug.update(seed_debug)
    debug.update(flood_debug)
    debug.update(contact_debug)
    debug.update(doorway_debug)
    debug.update(merge_debug)
    debug.update(small_debug)
    debug.update(absorption_debug)
    debug["timing_ms"]["total"] = float((time.perf_counter() - t0) * 1000.0)
    result = VerticalFreeRoomSegResult(
        room_label_map=final_labels.astype(np.int32),
        boundary_map=boundary.astype(bool),
        distance_map_m=distance_m.astype(np.float32),
        seed_label_map=seeds.astype(np.int32),
        initial_label_map=initial_labels.astype(np.int32),
        label_map_before_merge=before_merge.astype(np.int32),
        label_map_after_merge=merged_labels.astype(np.int32),
        room_stats=stats,
        adjacency_edges=edges,
        debug=debug,
    )
    if labels_outside_free or labels_in_unknown:
        raise AssertionError("vertical-free roomseg labels escaped free/unknown constraints")
    if cfg.debug_dump:
        save_vertical_free_roomseg_debug(result=result, out_dir=cfg.debug_dir, step=0)
    return result


def preprocess_vertical_free_inputs(
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    cfg: VerticalFreeRoomSegConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    clean_unknown = np.asarray(unknown, dtype=bool).copy()
    clean_free = np.asarray(free, dtype=bool) & ~clean_unknown
    clean_wall = np.asarray(wall, dtype=bool) & ~clean_free & ~clean_unknown
    removed_small_free = 0
    min_free_cells = max(1, int(round(float(cfg.min_free_component_area_m2) / max(cfg.resolution_m**2, 1e-9))))
    for comp in _components(clean_free, cfg.connectivity):
        if len(comp) < min_free_cells:
            removed_small_free += len(comp)
            for r, c in comp:
                clean_free[r, c] = False
    tiny_unknown_limit = max(0, int(round(float(cfg.fill_tiny_unknown_holes_m2) / max(cfg.resolution_m**2, 1e-9))))
    filled_unknown = 0
    filled_wall_noise = 0
    if tiny_unknown_limit > 0:
        for comp in _components(clean_unknown, cfg.connectivity):
            if len(comp) > tiny_unknown_limit or _touches_border(comp, clean_unknown.shape):
                continue
            if _component_touches(comp, clean_free):
                for r, c in comp:
                    clean_unknown[r, c] = False
                    clean_free[r, c] = True
                filled_unknown += len(comp)
        tiny_blocked = (~clean_free) & ~clean_unknown
        for comp in _components(tiny_blocked, cfg.connectivity):
            if len(comp) > tiny_unknown_limit or _touches_border(comp, clean_unknown.shape):
                continue
            if _component_touches(comp, clean_free):
                for r, c in comp:
                    clean_free[r, c] = True
                    clean_wall[r, c] = False
                filled_wall_noise += len(comp)
    wall_close_cells = _radius_cells(cfg.wall_close_radius_m, cfg)
    if wall_close_cells > 0 and np.any(clean_wall):
        closed = ndimage.binary_closing(clean_wall, structure=_disk(wall_close_cells))
        clean_wall = (clean_wall | (closed & ~clean_free & ~clean_unknown)).astype(bool)
    return clean_free, clean_wall, clean_unknown, {
        "preprocess_removed_small_free_cells": int(removed_small_free),
        "preprocess_filled_tiny_unknown_cells": int(filled_unknown),
        "preprocess_filled_tiny_wall_noise_cells": int(filled_wall_noise),
        "preprocess_wall_close_radius_cells": int(wall_close_cells),
    }


def compute_distance_map(free: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> tuple[np.ndarray, dict]:
    distance = ndimage.distance_transform_edt(np.asarray(free, dtype=bool)) * float(cfg.resolution_m)
    if float(cfg.distance_smooth_sigma_cells) > 0.0:
        smoothed = ndimage.gaussian_filter(distance, sigma=float(cfg.distance_smooth_sigma_cells))
        smoothed[~np.asarray(free, dtype=bool)] = 0.0
    else:
        smoothed = distance
    return smoothed.astype(np.float32), {
        "distance_max_m": float(np.max(smoothed)) if smoothed.size else 0.0,
        "distance_mean_free_m": float(np.mean(smoothed[np.asarray(free, dtype=bool)])) if np.any(free) else 0.0,
    }


def compute_room_seeds(
    free: np.ndarray,
    distance_m: np.ndarray,
    cfg: VerticalFreeRoomSegConfig,
) -> tuple[np.ndarray, list[dict], dict]:
    free_arr = np.asarray(free, dtype=bool)
    d = np.asarray(distance_m, dtype=np.float32)
    radius = max(1, _radius_cells(cfg.seed_min_distance_m, cfg))
    local_max = d == ndimage.maximum_filter(d, footprint=_disk(radius))
    peaks = local_max & free_arr & (d >= float(cfg.seed_min_clearance_m))
    seed_cc, n_seed_cc = ndimage.label(peaks, structure=_conn(cfg.connectivity))
    seeds = np.zeros_like(seed_cc, dtype=np.int32)
    regions: list[dict] = []
    next_id = 1
    for sid in range(1, int(n_seed_cc) + 1):
        mask = seed_cc == sid
        if not np.any(mask):
            continue
        coords = np.argwhere(mask)
        best = coords[int(np.argmax(d[mask]))]
        blob = _seed_blob(tuple(int(v) for v in best), free_arr, d, cfg)
        if np.any(seeds[blob] > 0):
            continue
        seeds[blob] = next_id
        regions.append(_seed_region(next_id, blob, d, forced=False))
        next_id += 1
    forced = 0
    if cfg.force_one_seed_per_free_component:
        free_cc, n_free = ndimage.label(free_arr, structure=_conn(cfg.connectivity))
        for fid in range(1, int(n_free) + 1):
            comp = free_cc == fid
            if np.any(seeds[comp] > 0):
                continue
            coords = np.argwhere(comp)
            if coords.size == 0:
                continue
            best = coords[int(np.argmax(d[comp]))]
            blob = _seed_blob(tuple(int(v) for v in best), free_arr, d, cfg)
            seeds[blob] = next_id
            regions.append(_seed_region(next_id, blob, d, forced=True))
            next_id += 1
            forced += 1
    return seeds, regions, {
        "seed_peak_component_count": int(n_seed_cc),
        "forced_seed_count": int(forced),
    }


def priority_geodesic_watershed(
    free: np.ndarray,
    distance_m: np.ndarray,
    seeds: np.ndarray,
    cfg: VerticalFreeRoomSegConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    free_arr = np.asarray(free, dtype=bool)
    d = np.asarray(distance_m, dtype=np.float32)
    labels = np.asarray(seeds, dtype=np.int32).copy()
    boundary = np.zeros_like(free_arr, dtype=bool)
    heap: list[tuple[float, int, int, int]] = []
    for r, c in np.argwhere(labels > 0):
        lab = int(labels[int(r), int(c)])
        for nr, nc in _neighbors(int(r), int(c), labels.shape, cfg.connectivity):
            if free_arr[nr, nc] and labels[nr, nc] == 0:
                heapq.heappush(heap, (-float(d[nr, nc]), lab, nr, nc))
    events: list[dict] = []
    visits = 0
    while heap:
        _neg_score, lab, r, c = heapq.heappop(heap)
        if not free_arr[r, c] or boundary[r, c] or labels[r, c] > 0:
            continue
        neighbor_labels = sorted({int(labels[nr, nc]) for nr, nc in _neighbors(r, c, labels.shape, cfg.connectivity) if int(labels[nr, nc]) > 0})
        if len(neighbor_labels) >= int(cfg.conflict_min_neighbor_labels):
            width_m = 2.0 * float(d[r, c])
            if bool(cfg.conflict_boundary_enabled) and float(d[r, c]) <= float(cfg.conflict_boundary_max_clearance_m) and width_m <= float(cfg.doorway_width_max_m):
                boundary[r, c] = True
                events.append(
                    {
                        "rc": [int(r), int(c)],
                        "neighbor_labels": neighbor_labels,
                        "clearance_m": float(d[r, c]),
                        "width_m": float(width_m),
                        "type": "narrow_conflict_boundary",
                    }
                )
                continue
            labels[r, c] = _majority_neighbor_label(labels, r, c, lab, cfg)
        elif len(neighbor_labels) == 1:
            labels[r, c] = int(neighbor_labels[0])
        else:
            labels[r, c] = int(lab)
        visits += 1
        for nr, nc in _neighbors(r, c, labels.shape, cfg.connectivity):
            if free_arr[nr, nc] and labels[nr, nc] == 0 and not boundary[nr, nc]:
                heapq.heappush(heap, (-float(d[nr, nc]), int(labels[r, c]), nr, nc))
    return labels, boundary, events, {"watershed_assigned_cells": int(visits)}


def refine_bottleneck_boundaries(
    labels: np.ndarray,
    boundary: np.ndarray,
    free: np.ndarray,
    distance_m: np.ndarray,
    cfg: VerticalFreeRoomSegConfig,
) -> tuple[np.ndarray, list[dict], dict]:
    refined = np.asarray(boundary, dtype=bool) & np.asarray(free, dtype=bool)
    cuts: list[dict] = []
    cc, n = ndimage.label(refined, structure=_conn(cfg.connectivity))
    for cid in range(1, int(n) + 1):
        comp = cc == cid
        labs = _boundary_adjacent_labels(labels, comp, radius_cells=max(1, _radius_cells(cfg.bottleneck_band_radius_m, cfg)))
        clear = np.asarray(distance_m, dtype=np.float32)[comp]
        mean_clear = float(np.mean(clear)) if clear.size else 0.0
        width = 2.0 * mean_clear
        keep = (
            len(labs) >= 2
            and width <= float(cfg.doorway_width_max_m)
        )
        if keep:
            cuts.append(
                {
                    "component_id": int(cid),
                    "labels": labs,
                    "boundary_cells": int(np.count_nonzero(comp)),
                    "mean_clearance_m": mean_clear,
                    "estimated_width_m": float(width),
                    "type": "doorway_bottleneck",
                }
            )
        else:
            refined[comp] = False
    return refined, cuts, {"doorway_cut_count": int(len(cuts))}


def label_contact_boundary(
    labels: np.ndarray,
    free: np.ndarray,
    distance_m: np.ndarray,
    cfg: VerticalFreeRoomSegConfig,
) -> tuple[np.ndarray, dict]:
    lab = np.asarray(labels, dtype=np.int32)
    free_arr = np.asarray(free, dtype=bool)
    d = np.asarray(distance_m, dtype=np.float32)
    out = np.zeros_like(free_arr, dtype=bool)
    cells = 0
    for r, c in np.argwhere(free_arr):
        labels_here = {int(lab[int(r), int(c)])} if int(lab[int(r), int(c)]) > 0 else set()
        for nr, nc in _neighbors(int(r), int(c), lab.shape, cfg.connectivity):
            neighbor_label = int(lab[nr, nc])
            if neighbor_label > 0:
                labels_here.add(neighbor_label)
        if len(labels_here) < int(cfg.conflict_min_neighbor_labels):
            continue
        width_m = 2.0 * float(d[int(r), int(c)])
        if width_m <= float(cfg.doorway_width_max_m) and float(d[int(r), int(c)]) <= float(cfg.conflict_boundary_max_clearance_m):
            out[int(r), int(c)] = True
            cells += 1
    return out, {"label_contact_boundary_cells": int(cells)}


def build_region_adjacency(
    labels: np.ndarray,
    boundary: np.ndarray,
    free: np.ndarray,
    distance_m: np.ndarray,
    cfg: VerticalFreeRoomSegConfig,
) -> list[dict]:
    del free
    lab = np.asarray(labels, dtype=np.int32)
    boundary_arr = np.asarray(boundary, dtype=bool)
    d = np.asarray(distance_m, dtype=np.float32)
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    h, w = lab.shape
    for r in range(h):
        for c in range(w):
            neighbors = [int(lab[nr, nc]) for nr, nc in _neighbors(r, c, lab.shape, cfg.connectivity) if int(lab[nr, nc]) > 0]
            if int(lab[r, c]) > 0:
                neighbors.append(int(lab[r, c]))
            uniq = sorted(set(neighbors))
            if len(uniq) < 2:
                continue
            for i, a in enumerate(uniq):
                for b in uniq[i + 1 :]:
                    key = (min(a, b), max(a, b))
                    row = rows.setdefault(
                        key,
                        {"label_a": int(key[0]), "label_b": int(key[1]), "contact_cells": 0, "boundary_cells": 0, "clearances": []},
                    )
                    row["contact_cells"] += 1
                    if boundary_arr[r, c]:
                        row["boundary_cells"] += 1
                    row["clearances"].append(float(d[r, c]))
    out = []
    for row in rows.values():
        vals = np.asarray(row.pop("clearances", []), dtype=np.float32)
        med = float(np.median(vals)) if vals.size else 0.0
        contact_m = float(row["contact_cells"]) * float(cfg.resolution_m)
        width = 2.0 * med
        is_narrow = width <= float(cfg.doorway_width_max_m)
        row.update(
            {
                "mean_clearance_m": float(np.mean(vals)) if vals.size else 0.0,
                "max_clearance_m": float(np.max(vals)) if vals.size else 0.0,
                "estimated_opening_width_m": float(width),
                "contact_length_m": contact_m,
                "is_narrow_bottleneck": bool(is_narrow),
                "is_wide_open": bool(width >= float(cfg.open_merge_min_boundary_width_m)),
            }
        )
        out.append(dict(row))
    return out


def merge_open_regions(labels: np.ndarray, adjacency_edges: Sequence[Mapping[str, object]], cfg: VerticalFreeRoomSegConfig) -> tuple[np.ndarray, dict]:
    parent = {int(v): int(v) for v in np.unique(labels) if int(v) > 0}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ops = []
    if bool(cfg.open_region_merge_enabled):
        for edge in adjacency_edges:
            width = float(edge.get("estimated_opening_width_m", 0.0) or 0.0)
            contact = float(edge.get("contact_length_m", 0.0) or 0.0)
            if width >= float(cfg.open_merge_min_boundary_width_m) and contact >= float(cfg.open_merge_min_contact_length_m) and not bool(edge.get("is_narrow_bottleneck", False)):
                a, b = int(edge["label_a"]), int(edge["label_b"])
                union(a, b)
                ops.append({"label_a": a, "label_b": b, "reason": "wide_open_region", "estimated_opening_width_m": width})
    out = np.zeros_like(labels, dtype=np.int32)
    remap: dict[int, int] = {}
    next_id = 1
    for old in sorted(parent):
        root = find(old)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        out[np.asarray(labels) == old] = remap[root]
    return out, {"merge_operations": ops}


def cleanup_small_segments(labels: np.ndarray, free: np.ndarray, distance_m: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> tuple[np.ndarray, dict]:
    out = np.asarray(labels, dtype=np.int32).copy()
    ops = []
    min_cells = max(1, int(round(float(cfg.merge_small_area_m2) / max(cfg.resolution_m**2, 1e-9))))
    for label in sorted(int(v) for v in np.unique(out) if int(v) > 0):
        mask = out == label
        if int(np.count_nonzero(mask)) >= min_cells or _is_corridor_like(mask, distance_m, cfg):
            continue
        target = _best_neighbor_label(out, mask, cfg)
        if target > 0:
            out[mask] = target
            ops.append({"from_label": int(label), "to_label": int(target), "reason": "small_segment_merge"})
    out = _relabel_compact(out)
    out[~np.asarray(free, dtype=bool)] = 0
    return out, {"small_merge_operations": ops}


def absorb_boundary_pixels(labels: np.ndarray, boundary: np.ndarray, free: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> tuple[np.ndarray, dict]:
    out = np.asarray(labels, dtype=np.int32).copy()
    if bool(cfg.keep_boundary_unlabeled):
        return out, {"boundary_absorbed_cells": 0, "strict_boundary_map": np.asarray(boundary, dtype=bool)}
    target = np.asarray(boundary, dtype=bool) & np.asarray(free, dtype=bool) & (out == 0)
    if not np.any(target) or not np.any(out > 0):
        return out, {"boundary_absorbed_cells": 0, "strict_boundary_map": np.asarray(boundary, dtype=bool)}
    _, inds = ndimage.distance_transform_edt(out <= 0, return_indices=True)
    rr = inds[0][target]
    cc = inds[1][target]
    out[target] = out[rr, cc]
    return out, {"boundary_absorbed_cells": int(np.count_nonzero(target)), "strict_boundary_map": np.asarray(boundary, dtype=bool)}


def compute_room_stats(labels: np.ndarray, distance_m: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> list[dict]:
    out = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = np.asarray(labels) == label
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            continue
        bbox = [int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1]
        height = max(1, bbox[1] - bbox[0])
        width = max(1, bbox[3] - bbox[2])
        aspect = float(max(height, width)) / float(max(1, min(height, width)))
        area_m2 = float(np.count_nonzero(mask)) * float(cfg.resolution_m) ** 2
        mean_clear = float(np.mean(distance_m[mask])) if np.any(mask) else 0.0
        max_clear = float(np.max(distance_m[mask])) if np.any(mask) else 0.0
        corridor = bool(area_m2 >= cfg.corridor_as_room_min_area_m2 and aspect >= cfg.corridor_as_room_aspect_ratio)
        out.append(
            {
                "label_id": int(label),
                "area_m2": area_m2,
                "centroid_rc": [float(np.mean(rr)), float(np.mean(cc))],
                "bbox_rc": bbox,
                "mean_clearance_m": mean_clear,
                "max_clearance_m": max_clear,
                "aspect_ratio": aspect,
                "is_corridor_like": corridor,
                "reliability": float(min(1.0, max(0.2, max_clear / max(float(cfg.seed_min_clearance_m), 1e-6)))),
            }
        )
    return out


def vertical_free_result_to_source_result(result: VerticalFreeRoomSegResult) -> ROSE2SourceResult:
    labels = np.asarray(result.room_label_map, dtype=np.int32)
    debug = dict(result.debug)
    return ROSE2SourceResult(
        backend=VERTICAL_FREE_ROOMSEG_BACKEND,
        room_label_map=labels,
        source_room_label_map=labels.copy(),
        clean_structure_map=np.asarray(result.boundary_map, dtype=bool),
        structural_score=np.asarray(result.distance_map_m, dtype=np.float32),
        boundary_map=np.asarray(result.boundary_map, dtype=bool),
        dominant_directions_rad=[],
        hough_segments=[],
        wall_clusters=[],
        representative_lines=[],
        extended_lines=[],
        faces=_faces_from_labels_simple(labels, float(debug.get("resolution_m", 0.05) or 0.05)),
        face_adjacency_edges=[dict(edge) for edge in result.adjacency_edges],
        cell_edges=[dict(edge) for edge in result.adjacency_edges],
        cell_polygons=_room_polygons_debug(labels),
        timing_ms=dict(debug.get("timing_ms", {})),
        debug=debug,
    )


def save_vertical_free_roomseg_debug(
    *,
    result: VerticalFreeRoomSegResult,
    out_dir: str | Path,
    step: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "vertical_free_step_%06d" % int(step)
    npz_path = out / ("%s.npz" % stem)
    summary_path = out / ("%s.summary.json" % stem)
    overlay_path = out / ("%s.overlay.png" % stem)
    layers_path = out / ("%s.layers.png" % stem)
    d = result.debug
    np.savez_compressed(
        npz_path,
        input_free=np.asarray(d.get("input_free"), dtype=np.uint8),
        input_wall=np.asarray(d.get("input_wall"), dtype=np.uint8),
        input_unknown=np.asarray(d.get("input_unknown"), dtype=np.uint8),
        clean_free=np.asarray(d.get("clean_free"), dtype=np.uint8),
        clean_wall=np.asarray(d.get("clean_wall"), dtype=np.uint8),
        distance_map_m=np.asarray(result.distance_map_m, dtype=np.float32),
        seed_label_map=np.asarray(result.seed_label_map, dtype=np.int32),
        initial_watershed_labels=np.asarray(result.initial_label_map, dtype=np.int32),
        conflict_boundary_map=np.asarray(d.get("conflict_boundary_map"), dtype=np.uint8),
        doorway_boundary_map=np.asarray(d.get("doorway_boundary_map"), dtype=np.uint8),
        label_map_before_merge=np.asarray(result.label_map_before_merge, dtype=np.int32),
        label_map_after_open_merge=np.asarray(result.label_map_after_merge, dtype=np.int32),
        label_map_after_small_merge=np.asarray(d.get("label_map_after_small_merge"), dtype=np.int32),
        final_room_label_map=np.asarray(result.room_label_map, dtype=np.int32),
        virtual_boundary_map=np.asarray(result.boundary_map, dtype=np.uint8),
    )
    summary = _json_ready(_summary_from_result(result))
    summary["debug_npz"] = str(npz_path)
    summary["overlay_png"] = str(overlay_path)
    summary["layers_png"] = str(layers_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        from PIL import Image, ImageDraw

        Image.fromarray(_overlay_rgb(result), mode="RGB").save(overlay_path)
        panels = [
            _bool_rgb(np.asarray(d.get("input_free"), dtype=bool), (230, 230, 230)),
            _bool_rgb(np.asarray(d.get("input_wall"), dtype=bool), (20, 20, 20)),
            _heat_rgb(result.distance_map_m),
            _label_rgb(result.seed_label_map),
            _label_rgb(result.initial_label_map),
            _bool_rgb(np.asarray(d.get("conflict_boundary_map"), dtype=bool), (255, 80, 0)),
            _bool_rgb(result.boundary_map, (255, 220, 0)),
            _label_rgb(result.label_map_before_merge),
            _label_rgb(result.label_map_after_merge),
            _label_rgb(result.room_label_map),
        ]
        h, w = panels[0].shape[:2]
        canvas = Image.new("RGB", (w * 5, h * 2), (0, 0, 0))
        for idx, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel, mode="RGB"), ((idx % 5) * w, (idx // 5) * h))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), "VFGW RoomSeg layers", fill=(255, 255, 255))
        canvas.save(layers_path)
    except Exception:
        pass
    result.debug["vertical_free_roomseg_layers"] = {
        "backend": VERTICAL_FREE_ROOMSEG_BACKEND,
        "debug_npz": str(npz_path),
        "overlay_png": str(overlay_path),
        "layers_png": str(layers_path),
        "room_count": int(_label_count(result.room_label_map)),
        "seed_count": int(_label_count(result.seed_label_map)),
        "virtual_boundary_cells": int(np.count_nonzero(result.boundary_map)),
        "doorway_boundary_cells": int(np.count_nonzero(result.boundary_map)),
    }
    return {"paths": result.debug["vertical_free_roomseg_layers"], "summary": summary}


def _summary_from_result(result: VerticalFreeRoomSegResult) -> dict:
    d = result.debug
    return {
        "backend": VERTICAL_FREE_ROOMSEG_BACKEND,
        "free_cells": int(d.get("free_cells", 0) or 0),
        "wall_cells": int(d.get("wall_cells", 0) or 0),
        "unknown_cells": int(d.get("unknown_cells", 0) or 0),
        "seed_count": int(d.get("seed_count", 0) or 0),
        "initial_room_count": int(d.get("initial_room_count", 0) or 0),
        "room_count_before_merge": int(d.get("room_count_before_merge", 0) or 0),
        "room_count_after_merge": int(d.get("room_count_after_merge", 0) or 0),
        "final_room_count": int(d.get("final_room_count", 0) or 0),
        "doorway_boundary_count": int(d.get("doorway_boundary_count", 0) or 0),
        "open_merge_count": int(d.get("open_merge_count", 0) or 0),
        "small_merge_count": int(d.get("small_merge_count", 0) or 0),
        "labels_outside_vertical_free_cells": int(d.get("labels_outside_vertical_free_cells", 0) or 0),
        "largest_room_area_m2": float(d.get("largest_room_area_m2", 0.0) or 0.0),
        "room_stats": list(result.room_stats),
    }


def _seed_blob(rc: tuple[int, int], free: np.ndarray, d: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> np.ndarray:
    r, c = rc
    radius = max(1, int(round(np.sqrt(max(float(cfg.seed_min_area_m2), cfg.resolution_m**2) / np.pi) / max(cfg.resolution_m, 1e-9))))
    yy, xx = np.ogrid[: free.shape[0], : free.shape[1]]
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= radius**2
    threshold = max(0.0, float(d[r, c]) - float(cfg.seed_h_max_m))
    blob = disk & free & (d >= threshold)
    if not np.any(blob):
        blob = np.zeros_like(free, dtype=bool)
        blob[r, c] = True
    return blob


def _seed_region(label: int, blob: np.ndarray, d: np.ndarray, *, forced: bool) -> dict:
    rr, cc = np.nonzero(blob)
    return {
        "label_id": int(label),
        "cells": int(rr.size),
        "centroid_rc": [float(np.mean(rr)) if rr.size else 0.0, float(np.mean(cc)) if cc.size else 0.0],
        "max_clearance_m": float(np.max(d[blob])) if np.any(blob) else 0.0,
        "forced": bool(forced),
    }


def _boundary_adjacent_labels(labels: np.ndarray, component: np.ndarray, radius_cells: int) -> list[int]:
    dil = ndimage.binary_dilation(component, structure=_disk(radius_cells))
    return sorted(int(v) for v in np.unique(np.asarray(labels, dtype=np.int32)[dil]) if int(v) > 0)


def _best_neighbor_label(labels: np.ndarray, mask: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> int:
    counts: dict[int, int] = {}
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors(int(r), int(c), mask.shape, cfg.connectivity):
            lab = int(labels[nr, nc])
            if lab > 0 and not mask[nr, nc]:
                counts[lab] = counts.get(lab, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else 0


def _is_corridor_like(mask: np.ndarray, distance_m: np.ndarray, cfg: VerticalFreeRoomSegConfig) -> bool:
    if not bool(cfg.corridor_keep_enabled):
        return False
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        return False
    area_m2 = float(rr.size) * float(cfg.resolution_m) ** 2
    if area_m2 < float(cfg.corridor_as_room_min_area_m2):
        return False
    aspect = float(max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)) / float(max(1, min(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)))
    return bool(aspect >= float(cfg.corridor_as_room_aspect_ratio) and float(np.mean(distance_m[mask])) <= float(cfg.bottleneck_max_clearance_m))


def _majority_neighbor_label(labels: np.ndarray, r: int, c: int, fallback: int, cfg: VerticalFreeRoomSegConfig) -> int:
    counts: dict[int, int] = {}
    for nr, nc in _neighbors(r, c, labels.shape, cfg.connectivity):
        lab = int(labels[nr, nc])
        if lab > 0:
            counts[lab] = counts.get(lab, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else int(fallback)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    labels, n = ndimage.label(np.asarray(mask, dtype=bool), structure=_conn(connectivity))
    comps: list[list[tuple[int, int]]] = []
    for label in range(1, int(n) + 1):
        rr, cc = np.nonzero(labels == label)
        comps.append([(int(r), int(c)) for r, c in zip(rr, cc)])
    return comps


def _component_touches(comp: Sequence[tuple[int, int]], mask: np.ndarray) -> bool:
    for r, c in comp:
        for nr, nc in _neighbors(r, c, mask.shape, 8):
            if bool(mask[nr, nc]):
                return True
    return False


def _touches_border(comp: Sequence[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1 for r, c in comp)


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int = 8):
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = shape
    for dr, dc in offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _conn(connectivity: int) -> np.ndarray:
    return np.ones((3, 3), dtype=bool) if int(connectivity) == 8 else np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _disk(radius_cells: int) -> np.ndarray:
    r = int(radius_cells)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _radius_cells(radius_m: float, cfg: VerticalFreeRoomSegConfig) -> int:
    return max(0, int(round(float(radius_m) / max(float(cfg.resolution_m), 1e-9))))


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


def _relabel_compact(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    for new_id, old_id in enumerate(sorted(int(v) for v in np.unique(arr) if int(v) > 0), start=1):
        out[arr == old_id] = int(new_id)
    return out


def _faces_from_labels_simple(labels: np.ndarray, resolution_m: float) -> list[dict]:
    faces = []
    for item in compute_room_stats(labels, np.asarray(labels > 0, dtype=np.float32) * float(resolution_m), VerticalFreeRoomSegConfig(resolution_m=resolution_m)):
        faces.append({"face_id": int(item["label_id"]), **item})
    return faces


def _room_polygons_debug(labels: np.ndarray) -> list[dict]:
    rows = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        rr, cc = np.nonzero(labels == label)
        if rr.size:
            rows.append({"label_id": int(label), "bbox_rc": [int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1]})
    return rows


def _bool_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[np.asarray(mask, dtype=bool)] = np.asarray(color, dtype=np.uint8)
    return out


def _label_rgb(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        color = np.asarray(_color(label), dtype=np.uint8)
        out[arr == label] = color
    return out


def _heat_rgb(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mx = float(np.max(arr)) if arr.size else 0.0
    norm = np.clip(arr / mx, 0.0, 1.0) if mx > 0 else arr
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[..., 0] = (norm * 255).astype(np.uint8)
    out[..., 1] = (np.sqrt(norm) * 180).astype(np.uint8)
    out[..., 2] = ((1.0 - norm) * 80).astype(np.uint8)
    return out


def _overlay_rgb(result: VerticalFreeRoomSegResult) -> np.ndarray:
    free = np.asarray(result.debug.get("clean_free"), dtype=bool)
    unknown = np.asarray(result.debug.get("input_unknown"), dtype=bool)
    out = np.zeros((*result.room_label_map.shape, 3), dtype=np.uint8)
    out[unknown] = (70, 70, 70)
    out[free] = (210, 210, 210)
    colors = _label_rgb(result.room_label_map)
    mask = result.room_label_map > 0
    out[mask] = (0.45 * out[mask] + 0.55 * colors[mask]).astype(np.uint8)
    out[result.boundary_map] = (255, 220, 0)
    return out


def _color(label: int) -> tuple[int, int, int]:
    x = (int(label) * 1103515245 + 12345) & 0xFFFFFFFF
    return (80 + (x & 127), 80 + ((x >> 8) & 127), 80 + ((x >> 16) & 127))


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value
