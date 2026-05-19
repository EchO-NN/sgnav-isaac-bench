from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from isaac_bench.mapping.structure_extraction import (
    connected_components,
    face_adjacency_edges,
    faces_from_labels,
    rasterize_representative_lines,
)


STRONG_BOUNDARY_SOURCES = {
    "thin_wall_from_vertical_free_nonfree",
    "thin_wall_from_nonfree_observed",
    "verified_doorway_partition_cut",
    "doorway_partition_cut",
    "topology_effective_separator",
    "rose2_representative_wall",
}


def select_topology_effective_separators(
    *,
    free: np.ndarray,
    unknown: np.ndarray,
    base_boundary: np.ndarray,
    candidate_lines: list[dict],
    resolution_m: float,
    min_room_area_m2: float,
    wall_raster_radius_cells: int,
    max_candidates: int = 128,
    min_largest_component_drop: float = 0.08,
) -> tuple[np.ndarray, list[dict], dict]:
    """Greedily keep only separator lines that create valid room-size splits."""

    free_arr = np.asarray(free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    boundary = np.asarray(base_boundary, dtype=bool).copy()
    if free_arr.shape != unknown_arr.shape or boundary.shape != free_arr.shape:
        raise ValueError("partition graph masks must have the same shape")
    min_cells = max(1, int(round(float(min_room_area_m2) / max(float(resolution_m) ** 2, 1e-9))))
    accepted: list[dict] = []
    rejected: list[dict] = []
    before_comps = _valid_components(free_arr & ~boundary, min_cells=1)
    candidates = sorted(
        [dict(line) for line in candidate_lines],
        key=lambda row: (-_line_priority(row), -float(row.get("length_m", 0.0) or 0.0)),
    )[: max(0, int(max_candidates))]

    for idx, line in enumerate(candidates):
        line["candidate_rank"] = int(idx)
        line_mask = rasterize_representative_lines([line], free_arr.shape, int(wall_raster_radius_cells))
        line_mask &= ~unknown_arr
        if not np.any(line_mask):
            rejected.append(_reject(line, "rejected_unknown_only", before_comps, before_comps))
            continue
        if not _line_has_free_on_both_sides(line_mask, free_arr):
            rejected.append(_reject(line, "rejected_no_free_on_both_sides", before_comps, before_comps))
            continue
        test_boundary = boundary | line_mask
        comps_before = _valid_components(free_arr & ~boundary, min_cells=1)
        comps_after = _valid_components(free_arr & ~test_boundary, min_cells=1)
        large_after = [comp for comp in comps_after if len(comp) >= min_cells]
        if len(comps_after) <= len(comps_before):
            large_before = [comp for comp in comps_before if len(comp) >= min_cells]
            if str(line.get("source", "")) in {"thin_wall_from_vertical_free_nonfree", "thin_wall_from_nonfree_observed", "rose2_representative_wall", "axis_support_from_structural_occupancy"} and (
                len(large_before) >= 2 or len(comps_before) >= 2
            ):
                row = dict(line)
                row.update(
                    {
                        "topology_effective": True,
                        "accepted_reason": "existing_nonfree_valid_split",
                        "components_before": int(len(comps_before)),
                        "components_after": int(len(comps_after)),
                        "largest_component_before": int(max([len(comp) for comp in comps_before] or [0])),
                        "largest_component_after": int(max([len(comp) for comp in comps_after] or [0])),
                        "largest_component_drop": 0.0,
                        "source": _strong_source_name(line),
                    }
                )
                boundary = test_boundary
                accepted.append(row)
                before_comps = comps_after
                continue
            rejected.append(_reject(line, "rejected_no_split", comps_before, comps_after))
            continue
        if len(large_after) < 2:
            rejected.append(_reject(line, "rejected_too_small_room", comps_before, comps_after))
            continue
        drop = _largest_component_drop(comps_before, comps_after)
        if drop < float(min_largest_component_drop) and _line_priority(line) < 90.0:
            rejected.append(_reject(line, "rejected_largest_component_drop_too_small", comps_before, comps_after, drop=drop))
            continue
        row = dict(line)
        row.update(
            {
                "topology_effective": True,
                "accepted_reason": "valid_split",
                "components_before": int(len(comps_before)),
                "components_after": int(len(comps_after)),
                "largest_component_before": int(max([len(comp) for comp in comps_before] or [0])),
                "largest_component_after": int(max([len(comp) for comp in comps_after] or [0])),
                "largest_component_drop": float(drop),
                "source": _strong_source_name(line),
            }
        )
        boundary = test_boundary
        accepted.append(row)
        before_comps = comps_after

    accepted_mask = rasterize_representative_lines(accepted, free_arr.shape, int(wall_raster_radius_cells)) & ~unknown_arr
    rejected_mask = rasterize_representative_lines(rejected, free_arr.shape, int(wall_raster_radius_cells)) & ~unknown_arr
    debug = {
        "topology_effective_separator_selection": True,
        "candidate_separator_count": int(len(candidate_lines)),
        "candidate_separator_count_limited": int(len(candidates)),
        "accepted_topology_separator_count": int(len(accepted)),
        "rejected_separator_count": int(len(rejected)),
        "accepted_separator_mask_cells": int(np.count_nonzero(accepted_mask)),
        "rejected_separator_mask_cells": int(np.count_nonzero(rejected_mask & ~accepted_mask)),
        "accepted_separators": accepted[:128],
        "rejected_separators": rejected[:128],
        "min_room_area_cells": int(min_cells),
        "largest_component_drop_threshold": float(min_largest_component_drop),
    }
    return boundary.astype(bool), accepted, debug


def generate_doorway_partition_cuts(
    *,
    free: np.ndarray,
    wall_boundary: np.ndarray,
    selected_separator_boundary: np.ndarray,
    candidate_lines: list[dict],
    resolution_m: float,
    doorway_width_min_m: float = 0.45,
    doorway_width_max_m: float = 1.60,
    min_wall_support_on_sides_m: float = 0.35,
) -> tuple[list[dict], dict]:
    """Generate virtual room-boundary cuts across narrow free openings."""

    free_arr = np.asarray(free, dtype=bool)
    walls = np.asarray(wall_boundary, dtype=bool) | np.asarray(selected_separator_boundary, dtype=bool)
    resolution = max(float(resolution_m), 1e-6)
    min_gap = max(1, int(round(float(doorway_width_min_m) / resolution)))
    max_gap = max(min_gap, int(round(float(doorway_width_max_m) / resolution)))
    min_support = max(1, int(round(float(min_wall_support_on_sides_m) / resolution)))
    cuts: list[dict] = []

    def add_cut(axis: str, index: int, start: int, end: int, support_a: int, support_b: int) -> None:
        width_cells = int(end - start + 1)
        if not (min_gap <= width_cells <= max_gap):
            return
        if support_a < min_support or support_b < min_support:
            return
        if axis == "vertical":
            p0 = [int(start), int(index)]
            p1 = [int(end), int(index)]
            orientation = math.pi / 2.0
        else:
            p0 = [int(index), int(start)]
            p1 = [int(index), int(end)]
            orientation = 0.0
        cuts.append(
            {
                "line_id": int(len(cuts)),
                "p0": p0,
                "p1": p1,
                "orientation_rad": float(orientation),
                "length_m": float(width_cells * resolution),
                "source": "verified_doorway_partition_cut",
                "doorway_width_m": float(width_cells * resolution),
                "wall_support_before_cells": int(support_a),
                "wall_support_after_cells": int(support_b),
                "topology_effective": False,
            }
        )

    for c in range(walls.shape[1]):
        _scan_doorway_axis(
            vec_wall=walls[:, c],
            vec_free=free_arr[:, c],
            axis="vertical",
            index=c,
            min_gap=min_gap,
            max_gap=max_gap,
            min_support=min_support,
            add_cut=add_cut,
        )
    for r in range(walls.shape[0]):
        _scan_doorway_axis(
            vec_wall=walls[r, :],
            vec_free=free_arr[r, :],
            axis="horizontal",
            index=r,
            min_gap=min_gap,
            max_gap=max_gap,
            min_support=min_support,
            add_cut=add_cut,
        )
    # Candidate lines with merged gaps already encode virtual cuts. Keep these
    # explicit records mainly for debug/merge guard evidence.
    for line in candidate_lines:
        if str(line.get("source", "")) in {"thin_wall_from_vertical_free_nonfree", "thin_wall_from_nonfree_observed"} and int(line.get("thin_wall_gap_count", 0) or 0) > 0:
            row = dict(line)
            row["source"] = "verified_doorway_partition_cut"
            row["line_id"] = int(len(cuts))
            cuts.append(row)
    cuts = _dedupe_by_geometry(cuts)
    debug = {
        "doorway_partition_cut_count": int(len(cuts)),
        "doorway_width_min_m": float(doorway_width_min_m),
        "doorway_width_max_m": float(doorway_width_max_m),
        "min_wall_support_on_sides_m": float(min_wall_support_on_sides_m),
        "doorway_partition_cuts": cuts[:128],
    }
    return cuts, debug


def labels_from_partition_boundary_v2(
    *,
    free: np.ndarray,
    partition_boundary: np.ndarray,
    unknown: np.ndarray,
    resolution_m: float,
    min_room_area_m2: float,
) -> tuple[np.ndarray, list[dict]]:
    free_arr = np.asarray(free, dtype=bool)
    boundary_arr = np.asarray(partition_boundary, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    unknown_bridge = _narrow_unknown_connectivity_bridge(
        free=free_arr,
        unknown=unknown_arr,
        partition_boundary=boundary_arr,
    )
    traversible = (free_arr | unknown_bridge) & ~boundary_arr
    labels = np.zeros_like(traversible, dtype=np.int32)
    min_cells = max(1, int(round(float(min_room_area_m2) / max(float(resolution_m) ** 2, 1e-9))))
    next_label = 1
    small: list[tuple[int, int]] = []
    for comp in connected_components(traversible):
        free_comp = [(int(r), int(c)) for r, c in comp if bool(free_arr[int(r), int(c)])]
        if len(free_comp) < min_cells and np.any(labels > 0):
            small.extend(comp)
            continue
        for r, c in free_comp:
            labels[int(r), int(c)] = int(next_label)
        next_label += 1
    for r, c in small:
        if not bool(free_arr[int(r), int(c)]):
            continue
        near = _nearest_label(labels, int(r), int(c), radius=8)
        if near:
            labels[int(r), int(c)] = int(near)
    faces = _faces_from_labels_simple(labels, resolution_m=resolution_m)
    return labels.astype(np.int32), faces


def absorb_partition_boundary_pixels_without_merging(
    labels: np.ndarray,
    *,
    free: np.ndarray,
    partition_boundary: np.ndarray,
    unknown: np.ndarray,
) -> tuple[np.ndarray, dict]:
    out = np.asarray(labels, dtype=np.int32).copy()
    boundary_free = np.asarray(free, dtype=bool) & np.asarray(partition_boundary, dtype=bool) & ~np.asarray(unknown, dtype=bool)
    absorbed = 0
    ambiguous = 0
    for r, c in zip(*np.nonzero(boundary_free & (out <= 0))):
        neighbors = {
            int(out[nr, nc])
            for nr in range(max(0, int(r) - 1), min(out.shape[0], int(r) + 2))
            for nc in range(max(0, int(c) - 1), min(out.shape[1], int(c) + 2))
            if int(out[nr, nc]) > 0
        }
        if len(neighbors) == 1:
            out[int(r), int(c)] = int(next(iter(neighbors)))
            absorbed += 1
        elif len(neighbors) > 1:
            ambiguous += 1
    return out, {
        "boundary_free_cells": int(np.count_nonzero(boundary_free)),
        "absorbed_boundary_pixels": int(absorbed),
        "ambiguous_boundary_pixels_left_unlabeled": int(ambiguous),
    }


def should_block_room_merge(
    *,
    room_a: int,
    room_b: int,
    adjacency_edge: Mapping[str, object],
    source_result: object,
    strong_boundary_sources: set[str] = STRONG_BOUNDARY_SOURCES,
) -> tuple[bool, str]:
    edge_weight = float(adjacency_edge.get("edge_wall_weight", 0.0) or 0.0)
    if edge_weight >= 0.25:
        return True, "high_confidence_partition_boundary_between_rooms"
    debug = dict(getattr(source_result, "debug", {}) or {})
    for line in list(debug.get("accepted_separators") or []) + list(debug.get("doorway_partition_cuts") or []):
        source = str(line.get("source", ""))
        if source in strong_boundary_sources or bool(line.get("topology_effective", False)):
            return True, "%s_between_rooms" % (source or "topology_effective_separator")
    return False, ""


def _scan_doorway_axis(*, vec_wall: np.ndarray, vec_free: np.ndarray, axis: str, index: int, min_gap: int, max_gap: int, min_support: int, add_cut) -> None:
    wall_idx = np.flatnonzero(np.asarray(vec_wall, dtype=bool))
    if wall_idx.size < 2:
        return
    for left, right in zip(wall_idx[:-1], wall_idx[1:]):
        gap = int(right) - int(left) - 1
        if gap < min_gap or gap > max_gap:
            continue
        if not np.all(np.asarray(vec_free, dtype=bool)[int(left) + 1 : int(right)]):
            continue
        support_a = _run_support(vec_wall, int(left), direction=-1)
        support_b = _run_support(vec_wall, int(right), direction=1)
        add_cut(axis, int(index), int(left) + 1, int(right) - 1, int(support_a), int(support_b))


def _run_support(vec: np.ndarray, idx: int, *, direction: int) -> int:
    arr = np.asarray(vec, dtype=bool)
    count = 0
    pos = int(idx)
    while 0 <= pos < arr.size and arr[pos]:
        count += 1
        pos += int(direction)
    return int(count)


def _line_priority(line: Mapping[str, object]) -> float:
    source = str(line.get("source", ""))
    if source in {"thin_wall_from_vertical_free_nonfree", "thin_wall_from_nonfree_observed"}:
        return 100.0
    if source in {"verified_doorway_partition_cut", "doorway_partition_cut"}:
        return 90.0
    if source in {"rose2_representative_wall", "vertical_profile_window_door_repair"}:
        return 70.0
    if source == "axis_support_from_structural_occupancy":
        return 50.0
    return 10.0


def _strong_source_name(line: Mapping[str, object]) -> str:
    source = str(line.get("source", ""))
    if source in {"thin_wall_from_vertical_free_nonfree", "thin_wall_from_nonfree_observed", "verified_doorway_partition_cut", "doorway_partition_cut"}:
        return source
    if bool(line.get("topology_effective", False)):
        return "topology_effective_separator"
    return source or "topology_effective_separator"


def _reject(line: Mapping[str, object], reason: str, comps_before: Sequence[Sequence[tuple[int, int]]], comps_after: Sequence[Sequence[tuple[int, int]]], *, drop: float = 0.0) -> dict:
    row = dict(line)
    row.update(
        {
            "topology_effective": False,
            "rejected_reason": reason,
            "components_before": int(len(comps_before)),
            "components_after": int(len(comps_after)),
            "largest_component_before": int(max([len(comp) for comp in comps_before] or [0])),
            "largest_component_after": int(max([len(comp) for comp in comps_after] or [0])),
            "largest_component_drop": float(drop),
        }
    )
    return row


def _valid_components(mask: np.ndarray, *, min_cells: int) -> list[list[tuple[int, int]]]:
    return [comp for comp in connected_components(mask) if len(comp) >= int(min_cells)]


def _largest_component_drop(before: Sequence[Sequence[tuple[int, int]]], after: Sequence[Sequence[tuple[int, int]]]) -> float:
    b = float(max([len(comp) for comp in before] or [0]))
    a = float(max([len(comp) for comp in after] or [0]))
    return 0.0 if b <= 0 else max(0.0, (b - a) / b)


def _line_has_free_on_both_sides(line_mask: np.ndarray, free: np.ndarray) -> bool:
    rr, cc = np.nonzero(line_mask)
    if rr.size == 0:
        return False
    h, w = free.shape
    vertical = (rr.max() - rr.min()) >= (cc.max() - cc.min())
    if vertical:
        left = free[np.clip(rr, 0, h - 1), np.clip(cc - 1, 0, w - 1)]
        right = free[np.clip(rr, 0, h - 1), np.clip(cc + 1, 0, w - 1)]
        return bool(np.count_nonzero(left) > 0 and np.count_nonzero(right) > 0)
    top = free[np.clip(rr - 1, 0, h - 1), np.clip(cc, 0, w - 1)]
    bottom = free[np.clip(rr + 1, 0, h - 1), np.clip(cc, 0, w - 1)]
    return bool(np.count_nonzero(top) > 0 and np.count_nonzero(bottom) > 0)


def _narrow_unknown_connectivity_bridge(
    *,
    free: np.ndarray,
    unknown: np.ndarray,
    partition_boundary: np.ndarray,
    max_width_cells: int = 2,
    support_band_cells: int = 2,
) -> np.ndarray:
    """Let narrow unobserved slits preserve room connectivity without labels.

    A black line in the vertical-free debug image is not necessarily a wall.
    When it is only unknown/unobserved and has free evidence on both sides, it
    should not split a room.  The returned mask is used only for connected
    component topology; those unknown pixels remain unlabeled in the final room
    mask and cannot override an accepted structural/doorway partition boundary.
    """

    free_arr = np.asarray(free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool) & ~np.asarray(partition_boundary, dtype=bool)
    out = np.zeros_like(free_arr, dtype=bool)
    for comp in connected_components(unknown_arr):
        rr = np.asarray([int(r) for r, _c in comp], dtype=np.int32)
        cc = np.asarray([int(c) for _r, c in comp], dtype=np.int32)
        if rr.size == 0:
            continue
        r0, r1 = int(rr.min()), int(rr.max())
        c0, c1 = int(cc.min()), int(cc.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if min(height, width) > int(max_width_cells):
            continue
        axis = "vertical" if height >= width else "horizontal"
        support = _free_support_for_bridge(
            free_arr,
            axis=axis,
            r0=r0,
            r1=r1,
            c0=c0,
            c1=c1,
            band=int(support_band_cells),
        )
        if support["side_a"] <= 0.0 or support["side_b"] <= 0.0:
            continue
        out[rr, cc] = True
    return out


def _free_support_for_bridge(
    free: np.ndarray,
    *,
    axis: str,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    band: int,
) -> dict:
    h, w = free.shape
    if axis == "vertical":
        rows = slice(max(0, r0), min(h, r1 + 1))
        left = free[rows, max(0, c0 - band) : max(0, c0)]
        right = free[rows, min(w, c1 + 1) : min(w, c1 + 1 + band)]
        return {"side_a": float(np.count_nonzero(left)), "side_b": float(np.count_nonzero(right))}
    cols = slice(max(0, c0), min(w, c1 + 1))
    top = free[max(0, r0 - band) : max(0, r0), cols]
    bottom = free[min(h, r1 + 1) : min(h, r1 + 1 + band), cols]
    return {"side_a": float(np.count_nonzero(top)), "side_b": float(np.count_nonzero(bottom))}


def _nearest_label(labels: np.ndarray, r: int, c: int, *, radius: int) -> int:
    for rad in range(1, int(radius) + 1):
        r0, r1 = max(0, r - rad), min(labels.shape[0], r + rad + 1)
        c0, c1 = max(0, c - rad), min(labels.shape[1], c + rad + 1)
        vals = [int(v) for v in np.unique(labels[r0:r1, c0:c1]) if int(v) > 0]
        if vals:
            return vals[0]
    return 0


def _faces_from_labels_simple(labels: np.ndarray, *, resolution_m: float) -> list[dict]:
    out: list[dict] = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        rr, cc = np.nonzero(labels == label)
        if rr.size == 0:
            continue
        out.append(
            {
                "face_id": int(label),
                "label": int(label),
                "area_cells": int(rr.size),
                "area_m2": float(rr.size * float(resolution_m) ** 2),
                "centroid_rc": [float(np.mean(rr)), float(np.mean(cc))],
                "bbox_rc": [int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())],
            }
        )
    return out


def _dedupe_by_geometry(lines: Sequence[Mapping[str, object]]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for line in lines:
        p0 = tuple(int(round(float(v))) for v in line.get("p0", (0, 0)))
        p1 = tuple(int(round(float(v))) for v in line.get("p1", (0, 0)))
        key = (p0[0], p0[1], p1[0], p1[1], str(line.get("source", "")))
        rev = (p1[0], p1[1], p0[0], p0[1], str(line.get("source", "")))
        if key in seen or rev in seen:
            continue
        row = dict(line)
        row["line_id"] = int(len(out))
        out.append(row)
        seen.add(key)
    return out
