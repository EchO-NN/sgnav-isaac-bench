from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import conn, rasterize_line


@dataclass
class WallProjectionConfig:
    enabled: bool = True
    projection_band_cells: int = 2
    max_projection_band_cells: int = 3
    min_projected_line_length_m: float = 0.35
    min_projected_support_ratio: float = 0.25
    anchor_min_projected_line_length_m: float = 0.15
    anchor_min_projected_support_ratio: float = 0.15
    max_fill_gap_m: float = 0.25
    max_free_gap_ratio: float = 0.20
    separate_parallel_wall_min_cells: int = 4
    parallel_peak_min_support_cells: int = 3
    max_lateral_std_cells: float = 1.75
    use_axis_aligned_first: bool = True
    allow_diagonal_pca_projection: bool = True
    max_pca_angle_to_axis_deg: float = 20.0
    forbid_projection_across_free: bool = True
    forbid_projection_across_door_seed: bool = True
    forbid_merge_across_unknown_gap: bool = False
    side_validation_enabled: bool = True
    side_band_cells: int = 3
    side_min_free_ratio: float = 0.15
    side_min_nonfree_ratio: float = 0.15
    reject_if_both_sides_free_ratio_gt: float = 0.55
    reject_if_both_sides_unknown_ratio_gt: float = 0.85
    reject_if_no_free_side: bool = True
    reject_if_no_nonfree_side: bool = True
    min_line_observed_support_cells: int = 3
    keep_accepted_line_cells_even_if_unknown_dominant: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WallProjectionConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class ProjectedWallLine:
    line_id: int
    axis: str
    line: int
    start: int
    end: int
    support_cell_count: int
    projected_cell_count: int
    support_ratio: float
    lateral_std_cells: float
    source: str
    reject_reason: str | None = None
    side_free_ratio_a: float = 0.0
    side_free_ratio_b: float = 0.0
    side_unknown_ratio_a: float = 0.0
    side_unknown_ratio_b: float = 0.0
    side_nonfree_ratio_a: float = 0.0
    side_nonfree_ratio_b: float = 0.0
    structural_side_score: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": int(self.line_id),
            "axis": str(self.axis),
            "line": int(self.line),
            "start": int(self.start),
            "end": int(self.end),
            "support_cell_count": int(self.support_cell_count),
            "projected_cell_count": int(self.projected_cell_count),
            "support_ratio": float(self.support_ratio),
            "lateral_std_cells": float(self.lateral_std_cells),
            "source": str(self.source),
            "reject_reason": self.reject_reason,
            "side_free_ratio_a": float(self.side_free_ratio_a),
            "side_free_ratio_b": float(self.side_free_ratio_b),
            "side_unknown_ratio_a": float(self.side_unknown_ratio_a),
            "side_unknown_ratio_b": float(self.side_unknown_ratio_b),
            "side_nonfree_ratio_a": float(self.side_nonfree_ratio_a),
            "side_nonfree_ratio_b": float(self.side_nonfree_ratio_b),
            "structural_side_score": float(self.structural_side_score),
            "debug": _jsonable(self.debug),
        }


@dataclass
class ProjectedWallResult:
    projected_wall_map: np.ndarray
    raw_wall_map: np.ndarray
    support_map: np.ndarray
    rejected_support_map: np.ndarray
    projected_lines: list[ProjectedWallLine]
    debug: dict[str, Any]


def project_wall_evidence_to_lines(
    *,
    wall_raw: np.ndarray,
    free_map: np.ndarray,
    occupied_ratio: np.ndarray | None = None,
    door_forbidden_mask: np.ndarray | None = None,
    unknown_forbidden_mask: np.ndarray | None = None,
    vertical_free_map: np.ndarray | None = None,
    unknown_map: np.ndarray | None = None,
    resolution_m: float,
    config: WallProjectionConfig | Mapping[str, object] | None = None,
) -> ProjectedWallResult:
    started_at = time.perf_counter()
    cfg = config if isinstance(config, WallProjectionConfig) else WallProjectionConfig.from_mapping(config)
    raw = np.asarray(wall_raw, dtype=bool)
    free = np.asarray(free_map, dtype=bool)
    if raw.shape != free.shape:
        raise ValueError("wall_raw and free_map must share HxW shape")
    if occupied_ratio is None:
        ratio = raw.astype(np.float32)
    else:
        ratio = np.asarray(occupied_ratio, dtype=np.float32)
        if ratio.shape != raw.shape:
            raise ValueError("occupied_ratio must match wall_raw shape")
    if door_forbidden_mask is None or not bool(cfg.forbid_projection_across_door_seed):
        door_forbidden = np.zeros(raw.shape, dtype=bool)
    else:
        door_forbidden = np.asarray(door_forbidden_mask, dtype=bool)
        if door_forbidden.shape != raw.shape:
            raise ValueError("door_forbidden_mask must match wall_raw shape")
    if unknown_forbidden_mask is None:
        unknown_forbidden = np.zeros(raw.shape, dtype=bool)
    else:
        unknown_forbidden = np.asarray(unknown_forbidden_mask, dtype=bool)
        if unknown_forbidden.shape != raw.shape:
            raise ValueError("unknown_forbidden_mask must match wall_raw shape")
    side_free = np.asarray(vertical_free_map if vertical_free_map is not None else free, dtype=bool)
    side_unknown = np.asarray(unknown_map if unknown_map is not None else unknown_forbidden, dtype=bool)
    if side_free.shape != raw.shape or side_unknown.shape != raw.shape:
        raise ValueError("vertical_free_map and unknown_map must match wall_raw shape")
    forbidden = door_forbidden | unknown_forbidden

    if not bool(cfg.enabled):
        debug = _debug_dict(started_at, cfg, raw, raw, raw, np.zeros_like(raw), unknown_forbidden, [], Counter({"disabled": 1}))
        return ProjectedWallResult(raw.copy(), raw.copy(), raw.copy(), np.zeros_like(raw), [], debug)
    support = raw & ~forbidden
    projected = np.zeros_like(raw, dtype=bool)
    accepted_support = np.zeros_like(raw, dtype=bool)
    rejected_support = np.zeros_like(raw, dtype=bool)
    lines: list[ProjectedWallLine] = []
    reject_counts: Counter[str] = Counter()
    if np.any(support):
        labels, count = ndimage.label(support, structure=conn(8))
        next_id = 1
        for comp_id in range(1, int(count) + 1):
            cells = np.column_stack(np.nonzero(labels == int(comp_id))).astype(np.int32)
            if cells.size == 0:
                continue
            produced = _project_component_axis_aligned(
                cells,
                free=free,
                ratio=ratio,
                door_forbidden=forbidden,
                side_free=side_free,
                side_unknown=side_unknown,
                resolution_m=float(resolution_m),
                cfg=cfg,
                next_id=next_id,
                shape=raw.shape,
            )
            if not produced and bool(cfg.allow_diagonal_pca_projection):
                produced = _project_component_pca(
                    cells,
                    free=free | forbidden,
                    side_free=side_free,
                    side_unknown=side_unknown,
                    ratio=ratio,
                    resolution_m=float(resolution_m),
                    cfg=cfg,
                    next_id=next_id,
                    shape=raw.shape,
                )
            if not produced:
                rejected_support[cells[:, 0], cells[:, 1]] = True
                reject_counts["projected_wall_too_short"] += 1
                continue
            comp_accepted = False
            for line, mask, support_cells in produced:
                if line.reject_reason is not None:
                    rejected_support[support_cells[:, 0], support_cells[:, 1]] = True
                    reject_counts[str(line.reject_reason)] += 1
                    continue
                projected |= mask
                accepted_support[support_cells[:, 0], support_cells[:, 1]] = True
                lines.append(line)
                next_id += 1
                comp_accepted = True
            if not comp_accepted:
                rejected_support[cells[:, 0], cells[:, 1]] = True
    projected = _fill_axis_gaps_in_projected_map(projected, free, forbidden, float(resolution_m), cfg)
    rejected_support |= support & ~accepted_support
    debug = _debug_dict(started_at, cfg, raw, projected, accepted_support, rejected_support, unknown_forbidden, lines, reject_counts)
    return ProjectedWallResult(projected, raw.copy(), accepted_support, rejected_support, lines, debug)


def _project_component_axis_aligned(
    cells: np.ndarray,
    *,
    free: np.ndarray,
    ratio: np.ndarray,
    door_forbidden: np.ndarray,
    side_free: np.ndarray,
    side_unknown: np.ndarray,
    resolution_m: float,
    cfg: WallProjectionConfig,
    next_id: int,
    shape: tuple[int, int],
) -> list[tuple[ProjectedWallLine, np.ndarray, np.ndarray]]:
    rows = cells[:, 0]
    cols = cells[:, 1]
    row_span = int(rows.max() - rows.min() + 1)
    col_span = int(cols.max() - cols.min() + 1)
    axis = "h" if col_span >= row_span else "v"
    lateral = rows if axis == "h" else cols
    longitudinal = cols if axis == "h" else rows
    hist_values, hist_counts = np.unique(lateral, return_counts=True)
    peaks = hist_values[hist_counts >= int(cfg.parallel_peak_min_support_cells)]
    split = False
    if peaks.size >= 2 and int(peaks.max() - peaks.min()) >= int(cfg.separate_parallel_wall_min_cells):
        split = True
        peak_values = np.sort(peaks)
        groups = []
        for peak in peak_values:
            distance = np.abs(lateral[:, None] - peak_values[None, :])
            nearest = peak_values[np.argmin(distance, axis=1)]
            groups.append(cells[nearest == peak])
    else:
        groups = [cells]

    out: list[tuple[ProjectedWallLine, np.ndarray, np.ndarray]] = []
    line_id = int(next_id)
    for group in groups:
        if group.size == 0:
            continue
        line_tuple = _project_axis_group(
            group,
            axis=axis,
            free=free,
            ratio=ratio,
            door_forbidden=door_forbidden,
            side_free=side_free,
            side_unknown=side_unknown,
            resolution_m=float(resolution_m),
            cfg=cfg,
            line_id=line_id,
            shape=shape,
            source="axis_aligned_split" if split else "axis_aligned",
            split_reason="parallel_wall_peaks_split" if split else None,
        )
        if line_tuple is not None:
            out.append(line_tuple)
            line_id += 1
    return out


def _project_axis_group(
    cells: np.ndarray,
    *,
    axis: str,
    free: np.ndarray,
    ratio: np.ndarray,
    door_forbidden: np.ndarray,
    side_free: np.ndarray,
    side_unknown: np.ndarray,
    resolution_m: float,
    cfg: WallProjectionConfig,
    line_id: int,
    shape: tuple[int, int],
    source: str,
    split_reason: str | None,
) -> tuple[ProjectedWallLine, np.ndarray, np.ndarray] | None:
    rows = cells[:, 0].astype(np.int32)
    cols = cells[:, 1].astype(np.int32)
    lateral = rows if axis == "h" else cols
    longitudinal = cols if axis == "h" else rows
    weights = np.maximum(ratio[rows, cols].astype(np.float64), 1e-3)
    projected_line = int(round(_weighted_median(lateral.astype(np.float64), weights)))
    coords = np.unique(longitudinal.astype(np.int32))
    if coords.size == 0:
        return None
    coords = _fill_projected_gaps(
        coords,
        projected_line=projected_line,
        axis=axis,
        free=free,
        door_forbidden=door_forbidden,
        resolution_m=float(resolution_m),
        cfg=cfg,
    )
    if coords.size == 0:
        return None
    start, end = int(coords.min()), int(coords.max())
    length_m = float((int(coords.size)) * float(resolution_m))
    support_ratio = float(cells.shape[0] / max(1, int(coords.size)))
    lateral_std = float(np.std(lateral.astype(np.float32))) if lateral.size else 0.0
    reject_reason = None
    if length_m + 1e-9 < float(cfg.min_projected_line_length_m):
        reject_reason = "projected_wall_too_short"
    elif support_ratio + 1e-9 < float(cfg.min_projected_support_ratio):
        reject_reason = "projected_support_ratio_too_low"
    elif lateral_std > float(cfg.max_lateral_std_cells) + 1e-9:
        reject_reason = "projected_wall_too_thick"
    mask = np.zeros(shape, dtype=bool)
    if axis == "h":
        valid = (coords >= 0) & (coords < shape[1]) & (projected_line >= 0) & (projected_line < shape[0])
        if np.any(valid):
            mask[int(projected_line), coords[valid]] = True
    else:
        valid = (coords >= 0) & (coords < shape[0]) & (projected_line >= 0) & (projected_line < shape[1])
        if np.any(valid):
            mask[coords[valid], int(projected_line)] = True
    side_metrics = _validate_projected_wall_sides(mask, axis, side_free, side_unknown, cfg)
    if reject_reason is None and bool(side_metrics.get("reject_reason")):
        reject_reason = str(side_metrics["reject_reason"])
    line = ProjectedWallLine(
        line_id=int(line_id),
        axis=str(axis),
        line=int(projected_line),
        start=int(start),
        end=int(end),
        support_cell_count=int(cells.shape[0]),
        projected_cell_count=int(np.count_nonzero(mask)),
        support_ratio=float(support_ratio),
        lateral_std_cells=float(lateral_std),
        source=str(source),
        reject_reason=reject_reason,
        side_free_ratio_a=float(side_metrics.get("side_free_ratio_a", 0.0)),
        side_free_ratio_b=float(side_metrics.get("side_free_ratio_b", 0.0)),
        side_unknown_ratio_a=float(side_metrics.get("side_unknown_ratio_a", 0.0)),
        side_unknown_ratio_b=float(side_metrics.get("side_unknown_ratio_b", 0.0)),
        side_nonfree_ratio_a=float(side_metrics.get("side_nonfree_ratio_a", 0.0)),
        side_nonfree_ratio_b=float(side_metrics.get("side_nonfree_ratio_b", 0.0)),
        structural_side_score=float(side_metrics.get("structural_side_score", 0.0)),
        debug={"split_reason": split_reason, **side_metrics},
    )
    return line, mask, cells


def _fill_projected_gaps(
    coords: np.ndarray,
    *,
    projected_line: int,
    axis: str,
    free: np.ndarray,
    door_forbidden: np.ndarray,
    resolution_m: float,
    cfg: WallProjectionConfig,
) -> np.ndarray:
    values = np.unique(np.asarray(coords, dtype=np.int32))
    if values.size <= 1:
        return values
    max_gap_cells = max(0, int(round(float(cfg.max_fill_gap_m) / max(float(resolution_m), 1e-9))))
    out: list[int] = [int(values[0])]
    for left, right in zip(values[:-1], values[1:]):
        gap = int(right - left - 1)
        if gap > 0 and gap <= max_gap_cells:
            fill = np.arange(int(left) + 1, int(right), dtype=np.int32)
            if axis == "h":
                rr = np.full(fill.shape, int(projected_line), dtype=np.int32)
                cc = fill
                valid = (rr >= 0) & (rr < free.shape[0]) & (cc >= 0) & (cc < free.shape[1])
                free_ratio = float(np.count_nonzero(free[rr[valid], cc[valid]]) / max(1, int(np.count_nonzero(valid))))
                blocked = bool(np.any(door_forbidden[rr[valid], cc[valid]]))
            else:
                rr = fill
                cc = np.full(fill.shape, int(projected_line), dtype=np.int32)
                valid = (rr >= 0) & (rr < free.shape[0]) & (cc >= 0) & (cc < free.shape[1])
                free_ratio = float(np.count_nonzero(free[rr[valid], cc[valid]]) / max(1, int(np.count_nonzero(valid))))
                blocked = bool(np.any(door_forbidden[rr[valid], cc[valid]]))
            if free_ratio <= float(cfg.max_free_gap_ratio) and not blocked:
                out.extend(int(v) for v in fill.tolist())
        out.append(int(right))
    return np.asarray(out, dtype=np.int32)


def _fill_axis_gaps_in_projected_map(
    projected: np.ndarray,
    free: np.ndarray,
    door_forbidden: np.ndarray,
    resolution_m: float,
    cfg: WallProjectionConfig,
) -> np.ndarray:
    out = np.asarray(projected, dtype=bool).copy()
    max_gap_cells = max(0, int(round(float(cfg.max_fill_gap_m) / max(float(resolution_m), 1e-9))))
    if max_gap_cells <= 0:
        return out
    for r in range(out.shape[0]):
        cols = np.flatnonzero(out[r])
        _fill_1d_line_gaps(out, free, door_forbidden, r, cols, axis="h", max_gap_cells=max_gap_cells, cfg=cfg)
    for c in range(out.shape[1]):
        rows = np.flatnonzero(out[:, c])
        _fill_1d_line_gaps(out, free, door_forbidden, c, rows, axis="v", max_gap_cells=max_gap_cells, cfg=cfg)
    return out


def _fill_1d_line_gaps(
    out: np.ndarray,
    free: np.ndarray,
    door_forbidden: np.ndarray,
    fixed: int,
    coords: np.ndarray,
    *,
    axis: str,
    max_gap_cells: int,
    cfg: WallProjectionConfig,
) -> None:
    values = np.asarray(coords, dtype=np.int32)
    if values.size <= 1:
        return
    for left, right in zip(values[:-1], values[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > int(max_gap_cells):
            continue
        fill = np.arange(int(left) + 1, int(right), dtype=np.int32)
        if axis == "h":
            rr = np.full(fill.shape, int(fixed), dtype=np.int32)
            cc = fill
        else:
            rr = fill
            cc = np.full(fill.shape, int(fixed), dtype=np.int32)
        valid = (rr >= 0) & (rr < out.shape[0]) & (cc >= 0) & (cc < out.shape[1])
        if not np.any(valid):
            continue
        free_ratio = float(np.count_nonzero(free[rr[valid], cc[valid]]) / max(1, int(np.count_nonzero(valid))))
        if free_ratio > float(cfg.max_free_gap_ratio) or bool(np.any(door_forbidden[rr[valid], cc[valid]])):
            continue
        out[rr[valid], cc[valid]] = True


def _validate_projected_wall_sides(
    mask: np.ndarray,
    axis: str,
    free: np.ndarray,
    unknown: np.ndarray,
    cfg: WallProjectionConfig,
) -> dict[str, Any]:
    if not bool(cfg.side_validation_enabled):
        return {"structural_side_score": 1.0, "reject_reason": None}
    line = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(line)) < int(cfg.min_line_observed_support_cells):
        return {"structural_side_score": 0.0, "reject_reason": "projected_wall_side_support_too_weak"}
    side_a, side_b = _line_side_bands(line, str(axis), int(cfg.side_band_cells))

    def ratios(side: np.ndarray) -> tuple[float, float, float]:
        count = int(np.count_nonzero(side))
        if count <= 0:
            return 0.0, 1.0, 1.0
        free_ratio = float(np.count_nonzero(side & free)) / float(count)
        unknown_ratio = float(np.count_nonzero(side & unknown)) / float(count)
        nonfree_ratio = float(np.count_nonzero(side & ~free)) / float(count)
        return free_ratio, unknown_ratio, nonfree_ratio

    free_a, unknown_a, nonfree_a = ratios(side_a)
    free_b, unknown_b, nonfree_b = ratios(side_b)
    has_free_side = max(free_a, free_b) >= float(cfg.side_min_free_ratio)
    has_nonfree_side = max(nonfree_a, nonfree_b) >= float(cfg.side_min_nonfree_ratio)
    reject_reason = None
    if free_a > float(cfg.reject_if_both_sides_free_ratio_gt) and free_b > float(cfg.reject_if_both_sides_free_ratio_gt):
        reject_reason = "projected_wall_both_sides_free_furniture_like"
    elif unknown_a > float(cfg.reject_if_both_sides_unknown_ratio_gt) and unknown_b > float(cfg.reject_if_both_sides_unknown_ratio_gt):
        reject_reason = "projected_wall_both_sides_unknown_frontier_like"
    elif bool(cfg.reject_if_no_free_side) and not has_free_side:
        reject_reason = "projected_wall_no_free_side"
    elif bool(cfg.reject_if_no_nonfree_side) and not has_nonfree_side:
        reject_reason = "projected_wall_no_nonfree_side"
    structural_score = max(free_a, free_b) * max(nonfree_a, nonfree_b) * (1.0 - min(unknown_a, unknown_b))
    return {
        "side_free_ratio_a": float(free_a),
        "side_free_ratio_b": float(free_b),
        "side_unknown_ratio_a": float(unknown_a),
        "side_unknown_ratio_b": float(unknown_b),
        "side_nonfree_ratio_a": float(nonfree_a),
        "side_nonfree_ratio_b": float(nonfree_b),
        "structural_side_score": float(structural_score),
        "reject_reason": reject_reason,
    }


def _line_side_bands(mask: np.ndarray, axis: str, band_cells: int) -> tuple[np.ndarray, np.ndarray]:
    line = np.asarray(mask, dtype=bool)
    band = max(1, int(band_cells))
    rows, cols = np.nonzero(line)
    side_a = np.zeros_like(line, dtype=bool)
    side_b = np.zeros_like(line, dtype=bool)
    if rows.size == 0:
        return side_a, side_b
    if str(axis) == "h":
        for offset in range(1, band + 1):
            ra = rows - offset
            rb = rows + offset
            valid_a = (ra >= 0) & (ra < line.shape[0])
            valid_b = (rb >= 0) & (rb < line.shape[0])
            side_a[ra[valid_a], cols[valid_a]] = True
            side_b[rb[valid_b], cols[valid_b]] = True
    elif str(axis) == "v":
        for offset in range(1, band + 1):
            ca = cols - offset
            cb = cols + offset
            valid_a = (ca >= 0) & (ca < line.shape[1])
            valid_b = (cb >= 0) & (cb < line.shape[1])
            side_a[rows[valid_a], ca[valid_a]] = True
            side_b[rows[valid_b], cb[valid_b]] = True
    else:
        dilated = ndimage.binary_dilation(line, structure=conn(8), iterations=band).astype(bool) & ~line
        side_a = dilated
        side_b = dilated
    return side_a & ~line, side_b & ~line


def _project_component_pca(
    cells: np.ndarray,
    *,
    free: np.ndarray,
    side_free: np.ndarray,
    side_unknown: np.ndarray,
    ratio: np.ndarray,
    resolution_m: float,
    cfg: WallProjectionConfig,
    next_id: int,
    shape: tuple[int, int],
) -> list[tuple[ProjectedWallLine, np.ndarray, np.ndarray]]:
    if cells.shape[0] < 2:
        return []
    pts = cells.astype(np.float32)
    center = np.average(pts, axis=0, weights=np.maximum(ratio[cells[:, 0], cells[:, 1]], 1e-3))
    centered = pts - center[None, :]
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0].astype(np.float32)
    norm = float(np.linalg.norm(major))
    if norm <= 1e-6:
        return []
    major = major / norm
    t = np.dot(centered, major)
    t_min = int(math.floor(float(t.min())))
    t_max = int(math.ceil(float(t.max())))
    if (t_max - t_min + 1) * float(resolution_m) < float(cfg.min_projected_line_length_m):
        return []
    p0 = center + major * float(t_min)
    p1 = center + major * float(t_max)
    mask = rasterize_line(p0, p1, shape)
    if bool(cfg.forbid_projection_across_free) and np.any(mask & free):
        line = ProjectedWallLine(int(next_id), "pca", int(round(center[0])), 0, 0, int(cells.shape[0]), int(np.count_nonzero(mask)), 0.0, 0.0, "pca", "projection_crosses_free")
        return [(line, np.zeros(shape, dtype=bool), cells)]
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []
    length_m = float(rows.size * float(resolution_m))
    support_ratio = float(cells.shape[0] / max(1, rows.size))
    side_metrics = _validate_projected_wall_sides(mask, "pca", side_free, side_unknown, cfg)
    reject_reason = None if length_m >= float(cfg.min_projected_line_length_m) else "projected_wall_too_short"
    if reject_reason is None and bool(side_metrics.get("reject_reason")):
        reject_reason = str(side_metrics["reject_reason"])
    line = ProjectedWallLine(
        line_id=int(next_id),
        axis="pca",
        line=int(round(float(center[0]))),
        start=0,
        end=int(rows.size - 1),
        support_cell_count=int(cells.shape[0]),
        projected_cell_count=int(rows.size),
        support_ratio=float(support_ratio),
        lateral_std_cells=0.0,
        source="pca",
        reject_reason=reject_reason,
        side_free_ratio_a=float(side_metrics.get("side_free_ratio_a", 0.0)),
        side_free_ratio_b=float(side_metrics.get("side_free_ratio_b", 0.0)),
        side_unknown_ratio_a=float(side_metrics.get("side_unknown_ratio_a", 0.0)),
        side_unknown_ratio_b=float(side_metrics.get("side_unknown_ratio_b", 0.0)),
        side_nonfree_ratio_a=float(side_metrics.get("side_nonfree_ratio_a", 0.0)),
        side_nonfree_ratio_b=float(side_metrics.get("side_nonfree_ratio_b", 0.0)),
        structural_side_score=float(side_metrics.get("structural_side_score", 0.0)),
        debug=side_metrics,
    )
    return [(line, mask, cells)]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return 0.0
    order = np.argsort(v)
    v = v[order]
    w = np.maximum(w[order], 0.0)
    total = float(np.sum(w))
    if total <= 1e-9:
        return float(np.median(v))
    cutoff = 0.5 * total
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def _debug_dict(
    started_at,
    cfg: WallProjectionConfig,
    raw: np.ndarray,
    projected: np.ndarray,
    support: np.ndarray,
    rejected: np.ndarray,
    unknown_forbidden: np.ndarray,
    lines: list[ProjectedWallLine],
    reject_counts: Counter[str],
) -> dict[str, Any]:
    side_reject_counts = Counter({key: value for key, value in reject_counts.items() if str(key).startswith("projected_wall_")})
    furniture_like = rejected if reject_counts.get("projected_wall_both_sides_free_furniture_like", 0) > 0 else np.zeros_like(rejected, dtype=bool)
    frontier_like = rejected if reject_counts.get("projected_wall_both_sides_unknown_frontier_like", 0) > 0 else np.zeros_like(rejected, dtype=bool)
    return {
        "voxel_wall_projection_enabled": bool(cfg.enabled),
        "voxel_wall_projection_ms": float((time.perf_counter() - started_at) * 1000.0),
        "voxel_wall_raw_xy": raw.astype(bool),
        "voxel_wall_projected_xy": projected.astype(bool),
        "voxel_wall_projection_support_map": support.astype(bool),
        "voxel_wall_projection_rejected_support_map": rejected.astype(bool),
        "voxel_wall_projection_forbidden_unknown_map": np.asarray(unknown_forbidden, dtype=bool),
        "voxel_wall_projection_accepted_structural_map": projected.astype(bool),
        "voxel_wall_projection_raw_cells": int(np.count_nonzero(raw)),
        "voxel_wall_projected_cells": int(np.count_nonzero(projected)),
        "voxel_wall_projection_support_cells": int(np.count_nonzero(support)),
        "voxel_wall_projection_rejected_support_cells": int(np.count_nonzero(rejected)),
        "voxel_wall_projection_accepted_structural_cells": int(np.count_nonzero(projected)),
        "voxel_wall_projection_furniture_like_rejected_cells": int(np.count_nonzero(furniture_like)),
        "voxel_wall_projection_frontier_like_rejected_cells": int(np.count_nonzero(frontier_like)),
        "voxel_wall_projection_forbidden_unknown_cells": int(np.count_nonzero(unknown_forbidden)),
        "voxel_wall_projection_line_count": int(len(lines)),
        "voxel_wall_projection_lines": [line.to_dict() for line in lines[:1024]],
        "voxel_wall_projection_reject_reason_counts": dict(reject_counts),
        "voxel_wall_projection_side_reject_reason_counts": dict(side_reject_counts),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
