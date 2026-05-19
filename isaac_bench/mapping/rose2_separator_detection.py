from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from isaac_bench.mapping.structure_extraction import connected_components


def detect_thin_wall_separator_candidates(
    *,
    vertical_free: np.ndarray,
    vertical_observed: np.ndarray,
    occupied: np.ndarray,
    unknown: np.ndarray,
    wall_confidence_map: np.ndarray | None,
    resolution_m: float,
    min_length_m: float = 0.45,
    max_width_m: float = 0.25,
    free_support_band_m: float = 0.30,
    min_free_support_ratio: float = 0.25,
    min_aspect_ratio: float = 3.0,
    doorway_width_max_m: float = 1.60,
) -> tuple[list[dict], np.ndarray, dict]:
    """Promote slender non-free vertical-profile structures to separators.

    This does not mutate the strict vertical-free map. It inspects the
    complement of vertical-free inside the observed vertical domain and returns
    line hypotheses that can later be tested for topology-effective splits.
    """

    free = np.asarray(vertical_free, dtype=bool)
    observed = np.asarray(vertical_observed, dtype=bool)
    occ = np.asarray(occupied, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    if free.shape != observed.shape or free.shape != occ.shape or free.shape != unknown_arr.shape:
        raise ValueError("thin-wall separator masks must have the same shape")

    resolution = max(float(resolution_m), 1e-6)
    min_len_cells = max(2, int(round(float(min_length_m) / resolution)))
    max_width_cells = max(1, int(round(float(max_width_m) / resolution)))
    support_band_cells = max(1, int(round(float(free_support_band_m) / resolution)))
    max_gap_cells = max(0, int(round(float(doorway_width_max_m) / resolution)))

    candidate = ((observed & ~free & ~unknown_arr) | occ) & ~unknown_arr
    separator_mask = np.zeros_like(candidate, dtype=bool)
    lines: list[dict] = []
    rejected: list[dict] = []

    component_lines = _component_separator_lines(
        candidate,
        free,
        wall_confidence_map=wall_confidence_map,
        resolution_m=resolution,
        min_len_cells=min_len_cells,
        max_width_cells=max_width_cells,
        support_band_cells=support_band_cells,
        min_free_support_ratio=float(min_free_support_ratio),
        min_aspect_ratio=float(min_aspect_ratio),
        rejected=rejected,
        separator_mask=separator_mask,
    )
    lines.extend(component_lines)
    axis_lines = _axis_merged_separator_lines(
        candidate,
        free,
        resolution_m=resolution,
        min_len_cells=min_len_cells if component_lines else max(min_len_cells, int(round(2.0 / resolution))),
        max_width_cells=max_width_cells,
        support_band_cells=support_band_cells,
        min_free_support_ratio=float(min_free_support_ratio),
        max_gap_cells=max_gap_cells,
        separator_mask=separator_mask,
        source_suffix="" if component_lines else "_long_axis_scan",
    )
    lines = _dedupe_lines(component_lines + axis_lines, tolerance_cells=max(1, max_width_cells))
    for idx, line in enumerate(lines):
        line["line_id"] = int(idx)
        line.setdefault("source", "thin_wall_from_vertical_free_nonfree")

    debug = {
        "vertical_free_source": "vertical_profile_0p2_2p0",
        "candidate_cells": int(np.count_nonzero(candidate)),
        "thin_wall_separator_count": int(len(lines)),
        "component_separator_count": int(len(component_lines)),
        "axis_merged_separator_count": int(len(axis_lines)),
        "separator_mask_cells": int(np.count_nonzero(separator_mask)),
        "rejected_candidate_count": int(len(rejected)),
        "rejected_candidates": rejected[:128],
        "min_length_m": float(min_length_m),
        "max_width_m": float(max_width_m),
        "free_support_band_m": float(free_support_band_m),
        "min_free_support_ratio": float(min_free_support_ratio),
    }
    return lines, separator_mask, debug


def _component_separator_lines(
    candidate: np.ndarray,
    free: np.ndarray,
    *,
    wall_confidence_map: np.ndarray | None,
    resolution_m: float,
    min_len_cells: int,
    max_width_cells: int,
    support_band_cells: int,
    min_free_support_ratio: float,
    min_aspect_ratio: float,
    rejected: list[dict],
    separator_mask: np.ndarray,
) -> list[dict]:
    lines: list[dict] = []
    confidence = np.asarray(wall_confidence_map, dtype=np.float32) if wall_confidence_map is not None else None
    for comp_id, comp in enumerate(connected_components(candidate), start=1):
        rr = np.asarray([r for r, _c in comp], dtype=np.int32)
        cc = np.asarray([c for _r, c in comp], dtype=np.int32)
        r0, r1 = int(rr.min()), int(rr.max())
        c0, c1 = int(cc.min()), int(cc.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        long_axis = max(height, width)
        short_axis = max(1, min(height, width))
        aspect = float(long_axis) / float(short_axis)
        if height >= width:
            axis = "vertical"
            length_cells = height
            width_cells = width
            center = int(round(float(cc.mean())))
        else:
            axis = "horizontal"
            length_cells = width
            width_cells = height
            center = int(round(float(rr.mean())))
        reason = ""
        if length_cells < min_len_cells:
            reason = "too_short"
        elif width_cells > max_width_cells:
            reason = "too_wide"
        elif aspect < float(min_aspect_ratio):
            reason = "low_aspect_ratio"
        support = _free_support(free, axis=axis, r0=r0, r1=r1, c0=c0, c1=c1, band=support_band_cells)
        if not reason and min(support["side_a"], support["side_b"]) < min_free_support_ratio:
            reason = "no_free_on_both_sides"
        if reason:
            rejected.append(
                {
                    "component_id": int(comp_id),
                    "reason": reason,
                    "bbox_rc": [r0, c0, r1, c1],
                    "length_cells": int(length_cells),
                    "width_cells": int(width_cells),
                    "aspect_ratio": float(aspect),
                    "free_support": support,
                }
            )
            continue
        if axis == "vertical":
            p0, p1 = [r0, center], [r1, center]
            orientation = math.pi / 2.0
        else:
            p0, p1 = [center, c0], [center, c1]
            orientation = 0.0
        for r, c in comp:
            separator_mask[int(r), int(c)] = True
        line = {
            "p0": p0,
            "p1": p1,
            "orientation_rad": float(orientation),
            "length_m": float(length_cells * resolution_m),
            "width_m": float(width_cells * resolution_m),
            "support_ratio": float(len(comp)) / float(max(1, length_cells * width_cells)),
            "source": "thin_wall_from_vertical_free_nonfree",
            "thin_wall_component_id": int(comp_id),
            "component_bbox_rc": [r0, c0, r1, c1],
            "free_support_left": float(support["side_a"]) if axis == "vertical" else None,
            "free_support_right": float(support["side_b"]) if axis == "vertical" else None,
            "free_support_top": float(support["side_a"]) if axis == "horizontal" else None,
            "free_support_bottom": float(support["side_b"]) if axis == "horizontal" else None,
            "wall_confidence_mean": float(np.mean(confidence[rr, cc])) if confidence is not None and rr.size else 0.0,
        }
        lines.append(line)
    return lines


def _axis_merged_separator_lines(
    candidate: np.ndarray,
    free: np.ndarray,
    *,
    resolution_m: float,
    min_len_cells: int,
    max_width_cells: int,
    support_band_cells: int,
    min_free_support_ratio: float,
    max_gap_cells: int,
    separator_mask: np.ndarray,
    source_suffix: str = "",
) -> list[dict]:
    out: list[dict] = []

    def emit(axis: str, idx: int, start: int, end: int, support_cells: int, gap_count: int) -> None:
        if end - start + 1 < min_len_cells:
            return
        if axis == "vertical":
            r0, r1, c0, c1 = int(start), int(end), int(idx), int(idx)
        else:
            r0, r1, c0, c1 = int(idx), int(idx), int(start), int(end)
        support = _free_support(free, axis=axis, r0=r0, r1=r1, c0=c0, c1=c1, band=support_band_cells)
        if min(support["side_a"], support["side_b"]) < min_free_support_ratio:
            return
        if axis == "vertical":
            p0, p1 = [r0, c0], [r1, c0]
            orientation = math.pi / 2.0
            separator_mask[r0 : r1 + 1, c0] |= candidate[r0 : r1 + 1, c0]
        else:
            p0, p1 = [r0, c0], [r0, c1]
            orientation = 0.0
            separator_mask[r0, c0 : c1 + 1] |= candidate[r0, c0 : c1 + 1]
        out.append(
            {
                "p0": p0,
                "p1": p1,
                "orientation_rad": float(orientation),
                "length_m": float((end - start + 1) * resolution_m),
                "width_m": float(max_width_cells * resolution_m),
                "support_ratio": float(support_cells) / float(max(1, end - start + 1)),
                "source": "thin_wall_from_vertical_free_nonfree",
                "thin_wall_detection_mode": "axis_merged%s" % str(source_suffix),
                "thin_wall_axis_merged": True,
                "thin_wall_gap_count": int(gap_count),
                "free_support_left": float(support["side_a"]) if axis == "vertical" else None,
                "free_support_right": float(support["side_b"]) if axis == "vertical" else None,
                "free_support_top": float(support["side_a"]) if axis == "horizontal" else None,
                "free_support_bottom": float(support["side_b"]) if axis == "horizontal" else None,
            }
        )

    for c in range(candidate.shape[1]):
        for start, end, support, gaps in _merge_runs_with_gaps(_runs_1d(candidate[:, c]), max_gap_cells=max_gap_cells):
            emit("vertical", c, start, end, support, gaps)
    for r in range(candidate.shape[0]):
        for start, end, support, gaps in _merge_runs_with_gaps(_runs_1d(candidate[r, :]), max_gap_cells=max_gap_cells):
            emit("horizontal", r, start, end, support, gaps)
    return out


def _free_support(free: np.ndarray, *, axis: str, r0: int, r1: int, c0: int, c1: int, band: int) -> dict:
    h, w = free.shape
    if axis == "vertical":
        rows = slice(max(0, r0), min(h, r1 + 1))
        left = free[rows, max(0, c0 - band) : max(0, c0)]
        right = free[rows, min(w, c1 + 1) : min(w, c1 + 1 + band)]
        return {"side_a": _ratio(left), "side_b": _ratio(right), "axis": "vertical"}
    cols = slice(max(0, c0), min(w, c1 + 1))
    top = free[max(0, r0 - band) : max(0, r0), cols]
    bottom = free[min(h, r1 + 1) : min(h, r1 + 1 + band), cols]
    return {"side_a": _ratio(top), "side_b": _ratio(bottom), "axis": "horizontal"}


def _ratio(mask: np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr)) / float(arr.size)


def _runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < arr.size:
        if not arr[idx]:
            idx += 1
            continue
        start = idx
        while idx < arr.size and arr[idx]:
            idx += 1
        runs.append((int(start), int(idx - 1)))
    return runs


def _merge_runs_with_gaps(runs: list[tuple[int, int]], *, max_gap_cells: int) -> list[tuple[int, int, int, int]]:
    if not runs:
        return []
    merged: list[tuple[int, int, int, int]] = []
    start, end = runs[0]
    support = end - start + 1
    gaps = 0
    for next_start, next_end in runs[1:]:
        gap = int(next_start) - int(end) - 1
        if gap <= int(max_gap_cells):
            end = int(next_end)
            support += int(next_end) - int(next_start) + 1
            gaps += int(gap > 0)
        else:
            merged.append((int(start), int(end), int(support), int(gaps)))
            start, end = int(next_start), int(next_end)
            support = int(next_end) - int(next_start) + 1
            gaps = 0
    merged.append((int(start), int(end), int(support), int(gaps)))
    return merged


def _dedupe_lines(lines: list[Mapping[str, object]], *, tolerance_cells: int) -> list[dict]:
    out: list[dict] = []
    for line in lines:
        axis = _axis(line)
        p0 = np.asarray(line.get("p0", (0, 0)), dtype=np.float32)
        p1 = np.asarray(line.get("p1", (0, 0)), dtype=np.float32)
        if axis == "vertical":
            coord = float((p0[1] + p1[1]) * 0.5)
            lo, hi = sorted((float(p0[0]), float(p1[0])))
        else:
            coord = float((p0[0] + p1[0]) * 0.5)
            lo, hi = sorted((float(p0[1]), float(p1[1])))
        duplicate_idx = None
        for idx, prev in enumerate(out):
            if _axis(prev) != axis:
                continue
            q0 = np.asarray(prev.get("p0", (0, 0)), dtype=np.float32)
            q1 = np.asarray(prev.get("p1", (0, 0)), dtype=np.float32)
            prev_coord = float((q0[1] + q1[1]) * 0.5) if axis == "vertical" else float((q0[0] + q1[0]) * 0.5)
            p_lo, p_hi = sorted((float(q0[0]), float(q1[0]))) if axis == "vertical" else sorted((float(q0[1]), float(q1[1])))
            overlap = min(hi, p_hi) - max(lo, p_lo)
            if abs(prev_coord - coord) <= tolerance_cells and overlap >= 0:
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            out.append(dict(line))
            continue
        prev = out[duplicate_idx]
        q0 = np.asarray(prev.get("p0", (0, 0)), dtype=np.float32)
        q1 = np.asarray(prev.get("p1", (0, 0)), dtype=np.float32)
        if axis == "vertical":
            coord_i = int(round((float(q0[1]) + float(q1[1]) + coord * 2.0) / 4.0))
            out[duplicate_idx]["p0"] = [int(round(min(lo, q0[0], q1[0]))), coord_i]
            out[duplicate_idx]["p1"] = [int(round(max(hi, q0[0], q1[0]))), coord_i]
        else:
            coord_i = int(round((float(q0[0]) + float(q1[0]) + coord * 2.0) / 4.0))
            out[duplicate_idx]["p0"] = [coord_i, int(round(min(lo, q0[1], q1[1])))]
            out[duplicate_idx]["p1"] = [coord_i, int(round(max(hi, q0[1], q1[1])))]
        out[duplicate_idx]["source"] = str(prev.get("source", "thin_wall_from_vertical_free_nonfree"))
        out[duplicate_idx]["thin_wall_gap_count"] = max(
            int(prev.get("thin_wall_gap_count", 0) or 0),
            int(line.get("thin_wall_gap_count", 0) or 0),
        )
        out[duplicate_idx]["thin_wall_axis_merged"] = bool(prev.get("thin_wall_axis_merged", False) or line.get("thin_wall_axis_merged", False))
    return out


def _axis(line: Mapping[str, object]) -> str:
    p0 = np.asarray(line.get("p0", (0, 0)), dtype=np.float32)
    p1 = np.asarray(line.get("p1", (0, 0)), dtype=np.float32)
    return "vertical" if abs(float(p1[0] - p0[0])) >= abs(float(p1[1] - p0[1])) else "horizontal"
