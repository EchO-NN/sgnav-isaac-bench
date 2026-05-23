from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorCandidate
from .utils import dilate, label_components, rasterize_line, relabel_compact


@dataclass
class CorridorAxisV4Result:
    distance_m: np.ndarray
    skeleton: np.ndarray
    narrow_axis: np.ndarray
    corridor_axis: np.ndarray
    corridor_junctions: np.ndarray
    side_branch_points: np.ndarray
    corridor_neck_candidates: list[SeparatorCandidate]
    debug: dict = field(default_factory=dict)


def detect_corridor_axis_v4(
    free_clean: np.ndarray,
    structural_wall_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    config: Mapping[str, object] | object | None = None,
    *,
    start_id: int = 1,
) -> CorridorAxisV4Result:
    cfg = _config(config)
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(structural_wall_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    if free.shape != wall.shape or free.shape != unknown.shape:
        raise ValueError("corridor axis inputs must have the same HxW shape")
    resolution = float(resolution_m)
    distance_cells = ndimage.distance_transform_edt(free)
    distance_m = distance_cells.astype(np.float32) * float(resolution)
    skeleton = _skeletonize(free)
    min_axis_width_m = float(cfg.get("min_axis_width_m", 0.45))
    max_axis_width_m = float(cfg.get("max_axis_width_m", 1.80))
    width_m = 2.0 * distance_m
    narrow_axis = skeleton & (width_m >= float(min_axis_width_m)) & (width_m <= float(max_axis_width_m))
    min_length_m = float(cfg.get("min_length_m", 0.90))
    min_component_cells = max(2, int(round(float(min_length_m) / max(float(resolution), 1e-6))))
    max_mean_width_m = float(cfg.get("max_mean_width_m", max_axis_width_m))
    corridor_axis = np.zeros_like(free, dtype=bool)
    labels, count = label_components(narrow_axis, 8)
    component_debug: list[dict] = []
    for label in range(1, int(count) + 1):
        comp = labels == label
        cells = int(np.count_nonzero(comp))
        mean_width = float(np.mean(width_m[comp])) if cells else 0.0
        keep = bool(cells >= int(min_component_cells) and mean_width <= float(max_mean_width_m))
        if keep:
            corridor_axis |= comp
        component_debug.append(
            {
                "label": int(label),
                "cells": int(cells),
                "length_m": float(cells * resolution),
                "mean_width_m": float(mean_width),
                "accepted": bool(keep),
            }
        )

    neighbor_count = _skeleton_neighbor_count(skeleton)
    branch = corridor_axis & (neighbor_count >= 3)
    endpoints = corridor_axis & (neighbor_count <= 1)
    wide_transition = corridor_axis & (dilate(free & (width_m > float(max_axis_width_m)), 1))
    side_branch_points = branch | wide_transition
    corridor_junctions = branch | endpoints | wide_transition
    corridor_junctions = _thin_points(corridor_junctions, radius_cells=max(1, int(cfg.get("junction_nms_cells", 4))))
    side_branch_points = side_branch_points & dilate(corridor_junctions, max(1, int(cfg.get("junction_nms_cells", 4))))

    candidates = _neck_candidates_from_points(
        points=corridor_junctions,
        corridor_axis=corridor_axis,
        free=free,
        wall=wall,
        unknown=unknown,
        distance_m=distance_m,
        resolution_m=float(resolution),
        cfg=cfg,
        start_id=int(start_id),
    )
    debug = {
        "enabled": bool(cfg.get("enabled", True)),
        "skeleton_cells": int(np.count_nonzero(skeleton)),
        "narrow_axis_cells": int(np.count_nonzero(narrow_axis)),
        "corridor_axis_cells": int(np.count_nonzero(corridor_axis)),
        "corridor_junction_cells": int(np.count_nonzero(corridor_junctions)),
        "side_branch_point_cells": int(np.count_nonzero(side_branch_points)),
        "corridor_neck_candidate_count": int(len(candidates)),
        "min_axis_width_m": float(min_axis_width_m),
        "max_axis_width_m": float(max_axis_width_m),
        "min_length_m": float(min_length_m),
        "components": component_debug[:128],
    }
    return CorridorAxisV4Result(
        distance_m=distance_m.astype(np.float32),
        skeleton=skeleton.astype(bool),
        narrow_axis=narrow_axis.astype(bool),
        corridor_axis=corridor_axis.astype(bool),
        corridor_junctions=corridor_junctions.astype(bool),
        side_branch_points=side_branch_points.astype(bool),
        corridor_neck_candidates=candidates,
        debug=debug,
    )


def _neck_candidates_from_points(
    *,
    points: np.ndarray,
    corridor_axis: np.ndarray,
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    distance_m: np.ndarray,
    resolution_m: float,
    cfg: Mapping[str, object],
    start_id: int,
) -> list[SeparatorCandidate]:
    rows, cols = np.nonzero(np.asarray(points, dtype=bool))
    max_candidates = max(0, int(cfg.get("max_candidates", 96)))
    max_width_m = float(cfg.get("max_neck_width_m", 1.80))
    min_width_m = float(cfg.get("min_neck_width_m", 0.35))
    out: list[SeparatorCandidate] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        if len(out) >= int(max_candidates):
            break
        center = (int(row), int(col))
        tangent = _local_axis_tangent(corridor_axis, center, radius=max(2, int(round(0.35 / max(resolution_m, 1e-6)))))
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        if float(np.linalg.norm(normal)) <= 1e-6:
            continue
        section = _cross_section(center, normal, free, max_width_m=max_width_m, resolution_m=float(resolution_m))
        if section is None:
            continue
        p0, p1, cells = section
        length_m = float(max(1, len(cells)) * resolution_m)
        if length_m < float(min_width_m) or length_m > float(max_width_m) * 1.25:
            continue
        mask = np.zeros_like(free, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        wall_support = float(np.any(dilate(mask, 1) & wall)) + float(np.any(dilate(mask, 1) & unknown)) * 0.25
        wall_support = float(np.clip(wall_support, 0.0, 1.0))
        confidence = float(np.clip(0.55 + 0.25 * wall_support + 0.20 * (distance_m[center] <= max_width_m / 2.0), 0.0, 1.0))
        candidate = SeparatorCandidate(
            candidate_id=int(start_id + len(out)),
            kind="corridor_room_neck_cut",
            p0_rc=np.asarray(p0, dtype=np.float32),
            p1_rc=np.asarray(p1, dtype=np.float32),
            theta=float(np.arctan2(float(p1[0] - p0[0]), float(p1[1] - p0[1]))),
            length_m=float(length_m),
            confidence=float(confidence),
            source_segment_ids=[],
            wall_support_score=float(wall_support),
            free_gap_score=1.0,
            doorway_score=float(confidence),
            visibility_drop_score=0.60,
            anchor_score=max(0.75, float(wall_support)),
            p0_anchor_score=max(0.75, float(wall_support)),
            p1_anchor_score=max(0.75, float(wall_support)),
            p0_anchor_type="corridor_axis_neck",
            p1_anchor_type="corridor_axis_neck",
            debug={
                "candidate_source": "corridor_axis_v4",
                "junction_rc": [int(center[0]), int(center[1])],
                "axis_tangent_rc": [float(tangent[0]), float(tangent[1])],
                "mask_cells_rc": [[int(r), int(c)] for r, c in cells.tolist()],
                "corridor_room_neck_min_room_side_area_m2": float(cfg.get("min_room_side_area_m2", 0.30)),
                "wall_endpoint_support_score": float(wall_support),
            },
        )
        out.append(candidate)
    return out


def _cross_section(
    center: tuple[int, int],
    normal: np.ndarray,
    free: np.ndarray,
    *,
    max_width_m: float,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    max_steps = max(1, int(np.ceil(float(max_width_m) / (2.0 * max(float(resolution_m), 1e-6)))) + 2)
    normal = np.asarray(normal, dtype=np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    sides: list[np.ndarray] = []
    for sign in (-1.0, 1.0):
        last = np.asarray(center, dtype=np.int32)
        for step in range(1, max_steps + 1):
            point = np.rint(np.asarray(center, dtype=np.float32) + sign * float(step) * normal).astype(np.int32)
            row, col = int(point[0]), int(point[1])
            if row < 0 or row >= free.shape[0] or col < 0 or col >= free.shape[1] or not bool(free[row, col]):
                break
            last = point
        sides.append(last.astype(np.float32))
    p0, p1 = sides[0], sides[1]
    mask = rasterize_line(p0, p1, free.shape) & free
    cells = np.asarray(np.argwhere(mask), dtype=np.int32)
    if len(cells) < 2:
        return None
    ordered = _order_cells_along_line(cells, p0, p1)
    return ordered[0].astype(np.float32), ordered[-1].astype(np.float32), ordered


def _order_cells_along_line(cells: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    direction = np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32)
    if float(np.linalg.norm(direction)) <= 1e-6:
        return cells
    scores = (cells.astype(np.float32) - p0[None, :]) @ direction
    return cells[np.argsort(scores)].astype(np.int32)


def _local_axis_tangent(axis: np.ndarray, center: tuple[int, int], *, radius: int) -> np.ndarray:
    row, col = int(center[0]), int(center[1])
    rr0, rr1 = max(0, row - radius), min(axis.shape[0], row + radius + 1)
    cc0, cc1 = max(0, col - radius), min(axis.shape[1], col + radius + 1)
    coords = np.argwhere(axis[rr0:rr1, cc0:cc1])
    if len(coords) < 2:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    coords = coords.astype(np.float32)
    coords[:, 0] += float(rr0)
    coords[:, 1] += float(cc0)
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        tangent = vh[0].astype(np.float32)
    except Exception:
        tangent = np.asarray([1.0, 0.0], dtype=np.float32)
    return tangent / max(float(np.linalg.norm(tangent)), 1e-6)


def _skeleton_neighbor_count(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    count = ndimage.convolve(src, kernel, mode="constant", cval=0)
    return (count - src).astype(np.uint8)


def _thin_points(mask: np.ndarray, *, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if not np.any(src):
        return src
    labels, count = label_components(src, 8)
    out = np.zeros_like(src, dtype=bool)
    for label in range(1, int(count) + 1):
        coords = np.argwhere(labels == label)
        if len(coords) == 0:
            continue
        center = np.mean(coords, axis=0)
        idx = int(np.argmin(np.sum((coords.astype(np.float32) - center[None, :]) ** 2, axis=1)))
        out[tuple(int(v) for v in coords[idx])] = True
    if int(radius_cells) <= 1:
        return out
    labels, count = label_components(out, 8)
    if count <= 1:
        return out
    coords = np.argwhere(out)
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not keep[i]:
            continue
        d2 = np.sum((coords[i + 1 :] - coords[i]) ** 2, axis=1)
        close = np.flatnonzero(d2 <= int(radius_cells) * int(radius_cells))
        keep[i + 1 + close] = False
    final = np.zeros_like(out, dtype=bool)
    kept = coords[keep]
    if len(kept):
        final[kept[:, 0], kept[:, 1]] = True
    return final


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if not np.any(src):
        return src.copy()
    try:
        from skimage.morphology import skeletonize

        return skeletonize(src).astype(bool)
    except Exception:
        distance = ndimage.distance_transform_edt(src)
        local_max = src.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.roll(np.roll(distance, dr, axis=0), dc, axis=1)
                local_max &= distance >= shifted
        labels, count = label_components(src, 8)
        out = np.zeros_like(src, dtype=bool)
        for label in range(1, int(count) + 1):
            comp = labels == label
            comp_max = local_max & comp
            if np.any(comp_max):
                out |= comp_max
            else:
                coords = np.argwhere(comp)
                if len(coords):
                    out[tuple(int(v) for v in coords[len(coords) // 2])] = True
        return out.astype(bool)


def _config(config: Mapping[str, object] | object | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "corridor_axis_v4"):
        return dict(getattr(config, "corridor_axis_v4") or {})
    return {}
