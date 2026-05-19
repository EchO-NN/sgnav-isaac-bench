from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorCandidate
from .utils import component_metrics, label_components, rasterize_line


@dataclass
class CorridorConfig:
    enabled: bool = True
    min_length_m: float = 1.5
    max_width_m: float = 1.8
    min_aspect_ratio: float = 2.5
    skeleton_prune_length_m: float = 0.4

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "CorridorConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class CorridorRoomNeckCutConfig:
    enabled: bool = True
    max_neck_width_m: float = 1.8
    min_room_side_area_m2: float = 1.5
    reject_if_splits_corridor_axis: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "CorridorRoomNeckCutConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


def build_corridor_debug(
    free_clean: np.ndarray,
    *,
    resolution_m: float,
    config: CorridorConfig | Mapping[str, object] | None = None,
) -> dict:
    cfg = config if isinstance(config, CorridorConfig) else CorridorConfig.from_mapping(config)
    free = np.asarray(free_clean, dtype=bool)
    distance = ndimage.distance_transform_edt(free)
    skeleton = _skeletonize_free(free, distance)
    labels, count = label_components(free, 4)
    corridor_map = np.zeros_like(free, dtype=bool)
    components: list[dict] = []
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        metrics = component_metrics(comp, float(resolution_m), distance)
        corridor_like = bool(
            metrics["aspect_ratio"] >= float(cfg.min_aspect_ratio)
            and metrics["length_m"] >= float(cfg.min_length_m)
            and metrics["median_width_m"] <= float(cfg.max_width_m)
        )
        if corridor_like:
            corridor_map |= comp
        components.append({"component": int(idx), **metrics, "corridor_like": corridor_like})
    return {
        "enabled": bool(cfg.enabled),
        "distance_cells": distance.astype(np.float32),
        "corridor_skeleton": skeleton.astype(bool),
        "corridor_candidate_map": corridor_map.astype(bool),
        "corridor_components": components[:256],
    }


def generate_corridor_room_neck_candidates(
    free_clean: np.ndarray,
    *,
    resolution_m: float,
    corridor_config: CorridorConfig | Mapping[str, object] | None = None,
    neck_config: CorridorRoomNeckCutConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    corridor_cfg = corridor_config if isinstance(corridor_config, CorridorConfig) else CorridorConfig.from_mapping(corridor_config)
    neck_cfg = neck_config if isinstance(neck_config, CorridorRoomNeckCutConfig) else CorridorRoomNeckCutConfig.from_mapping(neck_config)
    corridor_debug = build_corridor_debug(free_clean, resolution_m=resolution_m, config=corridor_cfg)
    if not bool(corridor_cfg.enabled) or not bool(neck_cfg.enabled):
        return [], {**corridor_debug, "corridor_room_neck_candidate_count": 0}
    free = np.asarray(free_clean, dtype=bool)
    max_cells = max(2, int(round(float(neck_cfg.max_neck_width_m) / max(float(resolution_m), 1e-9))))
    candidates: list[SeparatorCandidate] = []
    cid = int(start_id)
    seen: set[tuple[str, int, int, int]] = set()
    cid = _add_corridor_axis_probes(candidates, corridor_debug, free.shape, cid, resolution_m)
    # Boundary-overlap candidates catch a side room connected to a corridor where the
    # opening is only visible between two adjacent scan rows/columns.
    for row in range(free.shape[0] - 1):
        overlap = free[row, :] & free[row + 1, :]
        for c0, c1 in _runs(overlap):
            width = c1 - c0
            if 2 <= width <= max_cells and (_row_run_width(free[row, :], c0) > width * 2 or _row_run_width(free[row + 1, :], c0) > width * 2):
                key = ("row_overlap", int(row), int(c0), int(c1))
                if key not in seen:
                    seen.add(key)
                    confidence = 0.8 + 0.19 * min(1.0, float(width) / float(max_cells))
                    candidates.append(_neck_candidate(cid, "row_overlap", np.asarray([row + 1, c0]), np.asarray([row + 1, c1 - 1]), width, resolution_m, confidence=confidence))
                    cid += 1
    for col in range(free.shape[1] - 1):
        overlap = free[:, col] & free[:, col + 1]
        for r0, r1 in _runs(overlap):
            width = r1 - r0
            if 2 <= width <= max_cells and (_row_run_width(free[:, col], r0) > width * 2 or _row_run_width(free[:, col + 1], r0) > width * 2):
                key = ("col_overlap", int(col), int(r0), int(r1))
                if key not in seen:
                    seen.add(key)
                    confidence = 0.8 + 0.19 * min(1.0, float(width) / float(max_cells))
                    candidates.append(_neck_candidate(cid, "col_overlap", np.asarray([r0, col + 1]), np.asarray([r1 - 1, col + 1]), width, resolution_m, confidence=confidence))
                    cid += 1
    return candidates[:2048], {**corridor_debug, "corridor_room_neck_candidate_count": int(min(len(candidates), 2048))}


def _add_corridor_axis_probes(
    candidates: list[SeparatorCandidate],
    corridor_debug: dict,
    shape: tuple[int, int],
    next_id: int,
    resolution_m: float,
) -> int:
    cid = int(next_id)
    for component in corridor_debug.get("corridor_components", []):
        if not bool(component.get("corridor_like", False)):
            continue
        bbox = component.get("bbox") or [0, 0, 0, 0]
        r0, c0, r1, c1 = [int(v) for v in bbox]
        if r1 <= r0 or c1 <= c0:
            continue
        height = int(r1 - r0)
        width = int(c1 - c0)
        bbox_fill = float(component.get("area_cells", 0)) / float(max(1, height * width))
        if bbox_fill < 0.6:
            continue
        if width >= height:
            col = int((c0 + c1 - 1) // 2)
            p0 = np.asarray([r0, col], dtype=np.float32)
            p1 = np.asarray([r1 - 1, col], dtype=np.float32)
            width_cells = height
        else:
            row = int((r0 + r1 - 1) // 2)
            p0 = np.asarray([row, c0], dtype=np.float32)
            p1 = np.asarray([row, c1 - 1], dtype=np.float32)
            width_cells = width
        probe = _neck_candidate(cid, "corridor_axis_probe", p0, p1, width_cells, resolution_m, confidence=0.3)
        if int(np.count_nonzero(rasterize_line(probe.p0_rc, probe.p1_rc, shape))) > 0:
            candidates.append(probe)
            cid += 1
    return cid


def _neck_candidate(
    candidate_id: int,
    source: str,
    p0: np.ndarray,
    p1: np.ndarray,
    width_cells: int,
    resolution_m: float,
    *,
    confidence: float = 0.55,
) -> SeparatorCandidate:
    theta = 0.0 if int(p0[0]) == int(p1[0]) else float(np.pi / 2.0)
    return SeparatorCandidate(
        candidate_id=int(candidate_id),
        kind="corridor_room_neck_cut",
        p0_rc=p0.astype(np.float32),
        p1_rc=p1.astype(np.float32),
        theta=theta,
        length_m=float(max(1, int(width_cells)) * float(resolution_m)),
        confidence=float(confidence),
        source_segment_ids=[],
        corridor_preservation_score=0.5,
        debug={"source": source, "width_cells": int(width_cells)},
    )


def _skeletonize_free(free: np.ndarray, distance: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize

        return np.asarray(skeletonize(free), dtype=bool)
    except Exception:
        dist = np.asarray(distance, dtype=np.float32)
        if dist.size == 0:
            return np.zeros_like(free, dtype=bool)
        maxed = dist == ndimage.maximum_filter(dist, size=3, mode="constant", cval=0)
        return (maxed & free & (dist > 0)).astype(bool)


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(values, dtype=bool)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr.tolist() + [False]):
        if value and start is None:
            start = int(idx)
        elif not value and start is not None:
            out.append((int(start), int(idx)))
            start = None
    return out


def _row_run_width(values: np.ndarray, index: int) -> int:
    arr = np.asarray(values, dtype=bool)
    if index < 0 or index >= arr.size or not arr[int(index)]:
        return 0
    left = int(index)
    while left > 0 and arr[left - 1]:
        left -= 1
    right = int(index)
    while right + 1 < arr.size and arr[right + 1]:
        right += 1
    return int(right - left + 1)
