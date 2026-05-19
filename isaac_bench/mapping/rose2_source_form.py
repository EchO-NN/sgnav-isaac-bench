from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isaac_bench.mapping.structure_extraction import (
    StructureExtractionConfig,
    auto_threshold_structure_map,
    bresenham_line,
    cluster_wall_segments,
    detect_line_segments,
    directional_structural_score,
    dominant_directions_from_fft,
    face_adjacency_edges,
    faces_from_labels,
    rasterize_representative_lines,
    remove_isolated_clutter,
    representative_lines_from_clusters,
    extend_representative_lines_to_free_boundary,
)


SOURCE_FORM_BACKEND = "rose2_source_form"
SOURCE_EXTERNAL_BACKEND = "rose2_source_external"
LEGACY_STYLE_BACKEND = "legacy_rose2_style_debug"


@dataclass
class ROSE2SourceFormConfig:
    resolution_m: float = 0.05
    min_room_area_m2: float = 1.5
    min_cell_area_m2: float = 0.35
    min_cell_free_ratio: float = 0.12
    min_cut_spacing_m: float = 0.45
    min_wall_line_length_m: float = 1.0
    axis_snap_angle_rad: float = math.radians(28.0)
    max_cuts_per_axis: int = 96
    wall_raster_radius_cells: int = 1
    keep_boundary_pixels_unlabeled: bool = True
    debug_dump: bool = False
    debug_dir: str = "debug/rose2_source_form"

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]] = None, **overrides) -> "ROSE2SourceFormConfig":
        raw = dict(data or {})
        raw.update({key: value for key, value in overrides.items() if value is not None})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class ROSE2SourceResult:
    backend: str
    room_label_map: np.ndarray
    source_room_label_map: np.ndarray
    clean_structure_map: np.ndarray
    structural_score: np.ndarray
    boundary_map: np.ndarray
    dominant_directions_rad: list[float]
    hough_segments: list[dict]
    wall_clusters: list[dict]
    representative_lines: list[dict]
    extended_lines: list[dict]
    faces: list[dict]
    face_adjacency_edges: list[dict]
    cell_edges: list[dict]
    cell_polygons: list[dict]
    timing_ms: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


def run_rose2_source_form(
    *,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    structure_config: StructureExtractionConfig,
    source_config: ROSE2SourceFormConfig | None = None,
    object_memory: Optional[Sequence[object]] = None,
) -> ROSE2SourceResult:
    """Run the strict no-ROS ROSE2 source-form room segmentation pipeline.

    The important distinction from the older local helper is that room labels
    are not the connected components of `free & ~boundary_map`. The source-form
    path derives a compact planar arrangement from ROSE2 wall hypotheses:
    dominant directions, line/cluster extraction, representative wall-line
    extension, intersections expressed as axis cuts, rectangular planar cells,
    inside/outside classification, and then rasterized room labels.
    """

    cfg = source_config or ROSE2SourceFormConfig(
        resolution_m=float(structure_config.resolution_m),
        min_room_area_m2=float(structure_config.min_room_area_m2),
        min_wall_line_length_m=float(structure_config.hough_min_line_length_m),
    )
    t0 = time.perf_counter()
    occupied = np.asarray(observed_occupied, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    if occupied.shape != free.shape or occupied.shape != unknown_arr.shape:
        raise ValueError("ROSE2 source-form inputs must have the same HxW shape")

    cleaned = remove_isolated_clutter(occupied, free, structure_config, object_memory=object_memory)
    t_clean = time.perf_counter()
    dominant = dominant_directions_from_fft(
        cleaned,
        int(structure_config.dominant_direction_count),
        float(structure_config.dominant_direction_min_separation_rad),
    )
    structural_score = directional_structural_score(cleaned, dominant, float(structure_config.directional_filter_width_rad))
    clean_structure_map, threshold_debug = auto_threshold_structure_map(structural_score, cleaned, free, structure_config)
    if not np.any(clean_structure_map) and np.any(cleaned):
        clean_structure_map = cleaned.copy()
    t_score = time.perf_counter()
    segments = detect_line_segments(clean_structure_map, structure_config)
    if not segments:
        segments = detect_line_segments(cleaned, structure_config)
    clusters = cluster_wall_segments(segments, dominant, structure_config)
    representative = representative_lines_from_clusters(clusters, clean_structure_map.shape, structure_config)
    extended = extend_representative_lines_to_free_boundary(representative, free, clean_structure_map.shape, structure_config)
    axis_support = _axis_support_lines(
        cleaned,
        free,
        resolution_m=float(structure_config.resolution_m),
        min_length_m=float(structure_config.hough_min_line_length_m),
        line_gap_m=float(structure_config.hough_line_gap_m),
    )
    extended = _merge_line_hypotheses(extended, axis_support, tolerance_cells=2)
    boundary = rasterize_representative_lines(extended, clean_structure_map.shape, int(cfg.wall_raster_radius_cells))
    t_lines = time.perf_counter()
    labels, cells, cell_edges = _labels_from_source_form_cells(
        free=free,
        boundary_map=boundary,
        lines=extended,
        config=cfg,
    )
    faces = faces_from_labels(labels, structure_config)
    adjacency = face_adjacency_edges(labels, boundary)
    t_cells = time.perf_counter()
    room_count = _label_count(labels)
    debug = {
        "source_backend": SOURCE_FORM_BACKEND,
        "source_form_used": True,
        "source_exact_used": False,
        "legacy_style_used": False,
        "legacy_connected_component_rooms_used": False,
        "source_form_pipeline": [
            "dominant_direction_fft",
            "directional_structure_score",
            "hough_line_segments",
            "angular_clustering",
            "spatial_clustering",
            "representative_line_extension",
            "line_intersections_as_axis_cuts",
            "planar_cell_extraction",
            "inside_outside_classification",
            "rasterized_source_form_labels",
        ],
        "source_room_count": int(room_count),
        "proposal_room_count": int(room_count),
        "final_room_count_before_policy_merge": int(room_count),
        "wall_segment_count": int(len(segments)),
        "wall_cluster_count": int(len(clusters)),
        "extended_wall_line_count": int(len(extended)),
        "axis_support_line_count": int(len(axis_support)),
        "cell_edge_count": int(len(cell_edges)),
        "source_cell_count": int(len(cells)),
        "inside_source_cell_count": int(sum(1 for cell in cells if bool(cell.get("inside", False)))),
        "labels_outside_vertical_free_cells": int(np.count_nonzero((labels > 0) & ~free)),
        "threshold": threshold_debug,
        "cell_polygons": cells,
        "source_cell_edges": cell_edges,
        "source_room_label_map": labels.astype(np.int32),
        "source_form_boundary_map": boundary.astype(bool),
        "failure_mode": "" if room_count > 0 else "no_inside_source_cells",
    }
    timing = {
        "cleaning": (t_clean - t0) * 1000.0,
        "dft_structure": (t_score - t_clean) * 1000.0,
        "wall_lines": (t_lines - t_score) * 1000.0,
        "source_form_cells": (t_cells - t_lines) * 1000.0,
        "total": (t_cells - t0) * 1000.0,
    }
    debug["timing_ms"] = dict(timing)
    return ROSE2SourceResult(
        backend=SOURCE_FORM_BACKEND,
        room_label_map=labels.astype(np.int32),
        source_room_label_map=labels.astype(np.int32),
        clean_structure_map=clean_structure_map.astype(bool),
        structural_score=structural_score.astype(np.float32),
        boundary_map=boundary.astype(bool),
        dominant_directions_rad=[float(v) for v in dominant],
        hough_segments=[dict(item) for item in segments],
        wall_clusters=[dict(item) for item in clusters],
        representative_lines=[dict(item) for item in representative],
        extended_lines=[dict(item) for item in extended],
        faces=faces,
        face_adjacency_edges=adjacency,
        cell_edges=cell_edges,
        cell_polygons=cells,
        timing_ms=timing,
        debug=debug,
    )


def save_rose2_source_debug(
    *,
    out_dir: str | Path,
    step: int,
    result: ROSE2SourceResult,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "rose2_source_step_%06d" % int(step)
    occupied = np.asarray(observed_occupied, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    paths = {
        "npz": str(out / ("%s.npz" % stem)),
        "summary_json": str(out / ("%s.summary.json" % stem)),
        "metric_input_png": str(out / ("%s.metric_input.png" % stem)),
        "vertical_free_png": str(out / ("%s.vertical_free.png" % stem)),
        "vertical_occupied_png": str(out / ("%s.vertical_occupied.png" % stem)),
        "unknown_png": str(out / ("%s.unknown.png" % stem)),
        "source_labels_png": str(out / ("%s.source_labels.png" % stem)),
        "overlay_png": str(out / ("%s.overlay.png" % stem)),
        "lines_png": str(out / ("%s.lines.png" % stem)),
        "cells_png": str(out / ("%s.cells.png" % stem)),
    }
    np.savez_compressed(
        paths["npz"],
        observed_occupied=occupied.astype(np.uint8),
        observed_free=free.astype(np.uint8),
        unknown=unknown_arr.astype(np.uint8),
        clean_structure_map=result.clean_structure_map.astype(np.uint8),
        boundary_map=result.boundary_map.astype(np.uint8),
        source_room_label_map=result.source_room_label_map.astype(np.int32),
        room_label_map=result.room_label_map.astype(np.int32),
        structural_score=result.structural_score.astype(np.float32),
    )
    summary = source_result_summary(result)
    Path(paths["summary_json"]).write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _metric_input_image(occupied, free, unknown_arr).save(paths["metric_input_png"])
    _bool_image(free, true_color=(240, 240, 240), false_color=(0, 0, 0)).save(paths["vertical_free_png"])
    _bool_image(occupied, true_color=(240, 240, 240), false_color=(0, 0, 0)).save(paths["vertical_occupied_png"])
    _bool_image(unknown_arr, true_color=(150, 150, 150), false_color=(0, 0, 0)).save(paths["unknown_png"])
    _label_image(result.source_room_label_map).save(paths["source_labels_png"])
    _overlay_image(occupied, free, unknown_arr, result).save(paths["overlay_png"])
    _line_image(free, result.extended_lines, result.hough_segments).save(paths["lines_png"])
    _cell_image(free, result.cell_polygons, result.room_label_map).save(paths["cells_png"])
    return {"paths": paths, "summary": summary}


def source_result_summary(result: ROSE2SourceResult) -> dict:
    debug = dict(result.debug or {})
    return {
        "backend": str(result.backend),
        "source_backend": str(debug.get("source_backend", result.backend)),
        "source_room_count": int(debug.get("source_room_count", _label_count(result.room_label_map)) or 0),
        "proposal_room_count": int(debug.get("proposal_room_count", _label_count(result.room_label_map)) or 0),
        "wall_segment_count": int(debug.get("wall_segment_count", len(result.hough_segments)) or 0),
        "wall_cluster_count": int(debug.get("wall_cluster_count", len(result.wall_clusters)) or 0),
        "extended_wall_line_count": int(debug.get("extended_wall_line_count", len(result.extended_lines)) or 0),
        "cell_edge_count": int(debug.get("cell_edge_count", len(result.cell_edges)) or 0),
        "source_cell_count": int(debug.get("source_cell_count", len(result.cell_polygons)) or 0),
        "inside_source_cell_count": int(debug.get("inside_source_cell_count", 0) or 0),
        "labels_outside_vertical_free_cells": int(debug.get("labels_outside_vertical_free_cells", 0) or 0),
        "legacy_connected_component_rooms_used": bool(debug.get("legacy_connected_component_rooms_used", False)),
        "failure_mode": str(debug.get("failure_mode", "")),
        "timing_ms": dict(result.timing_ms),
    }


def export_rose2_source_input(
    *,
    out_dir: str | Path,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    stem: str = "rose2_source_input",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    occupied = np.asarray(observed_occupied, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    image_path = out / ("%s.metric_map.png" % stem)
    npz_path = out / ("%s.npz" % stem)
    _metric_input_image(occupied, free, unknown_arr).save(image_path)
    np.savez_compressed(npz_path, occupied=occupied.astype(np.uint8), free=free.astype(np.uint8), unknown=unknown_arr.astype(np.uint8))
    return {"metric_map_png": str(image_path), "npz": str(npz_path)}


def parse_rose2_source_label_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    flat = arr.reshape((-1, 3))
    labels = np.zeros(flat.shape[0], dtype=np.int32)
    color_to_label: dict[tuple[int, int, int], int] = {}
    next_label = 1
    for idx, color_arr in enumerate(flat):
        color = tuple(int(v) for v in color_arr)
        if color in {(0, 0, 0), (255, 255, 255), (127, 127, 127)}:
            continue
        if color not in color_to_label:
            color_to_label[color] = next_label
            next_label += 1
        labels[idx] = color_to_label[color]
    return labels.reshape(arr.shape[:2]).astype(np.int32)


def run_rose2_source_external(
    *,
    source_root: str | Path | None,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    work_dir: str | Path,
) -> ROSE2SourceResult:
    """Export an external-source work bundle and fail clearly.

    The original declutter-reconstruct repository is a script pipeline, not a
    stable importable library. The strict default is therefore `rose2_source_form`.
    This backend exists for exact-source experiments and never falls back to the
    source-form implementation silently.
    """

    export_paths = export_rose2_source_input(
        out_dir=work_dir,
        observed_occupied=observed_occupied,
        observed_free=observed_free,
        unknown=unknown,
        stem="external_source_input",
    )
    root = Path(source_root).expanduser() if source_root else None
    if root is None or not root.exists():
        raise RuntimeError("rose2_source_external requires a valid source_root; exported inputs: %s" % export_paths)
    raise RuntimeError(
        "rose2_source_external is configured but the upstream declutter-reconstruct script API is not callable in-process; "
        "use rose2_source_form for strict no-ROS source-form rooms or provide an external runner wrapper. Exported inputs: %s"
        % export_paths
    )


def _labels_from_source_form_cells(
    *,
    free: np.ndarray,
    boundary_map: np.ndarray,
    lines: Sequence[Mapping[str, object]],
    config: ROSE2SourceFormConfig,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    free_arr = np.asarray(free, dtype=bool)
    boundary = np.asarray(boundary_map, dtype=bool)
    h, w = free_arr.shape
    if not np.any(free_arr):
        return np.zeros_like(free_arr, dtype=np.int32), [], []

    rmin, cmin, rmax, cmax = _bbox(free_arr)
    row_cuts = [int(rmin), int(rmax) + 1]
    col_cuts = [int(cmin), int(cmax) + 1]
    internal_lines: list[dict] = []
    for idx, line in enumerate(lines):
        axis = _line_axis(line, float(config.axis_snap_angle_rad))
        if axis is None:
            continue
        p0 = np.asarray(line.get("p0", (0, 0)), dtype=np.float32)
        p1 = np.asarray(line.get("p1", (0, 0)), dtype=np.float32)
        length_m = float(line.get("length_m", 0.0) or (np.linalg.norm(p1 - p0) * float(config.resolution_m)))
        if length_m < float(config.min_wall_line_length_m):
            continue
        row = dict(line)
        row["source_form_axis"] = axis
        row["source_form_line_id"] = int(idx)
        if axis == "vertical":
            cut = int(round(float((p0[1] + p1[1]) * 0.5)))
            if cmin + 1 <= cut <= cmax - 1:
                col_cuts.append(cut)
                internal_lines.append(row)
        else:
            cut = int(round(float((p0[0] + p1[0]) * 0.5)))
            if rmin + 1 <= cut <= rmax - 1:
                row_cuts.append(cut)
                internal_lines.append(row)

    row_cuts = _prune_cuts(row_cuts, min_spacing=max(1, int(round(float(config.min_cut_spacing_m) / max(float(config.resolution_m), 1e-6)))), max_count=int(config.max_cuts_per_axis))
    col_cuts = _prune_cuts(col_cuts, min_spacing=max(1, int(round(float(config.min_cut_spacing_m) / max(float(config.resolution_m), 1e-6)))), max_count=int(config.max_cuts_per_axis))
    labels = np.zeros_like(free_arr, dtype=np.int32)
    cells: list[dict] = []
    min_area_cells = max(1, int(round(float(config.min_cell_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    next_label = 1
    for ri in range(len(row_cuts) - 1):
        for ci in range(len(col_cuts) - 1):
            r0, r1 = int(row_cuts[ri]), int(row_cuts[ri + 1])
            c0, c1 = int(col_cuts[ci]), int(col_cuts[ci + 1])
            if r1 <= r0 or c1 <= c0:
                continue
            region = np.zeros_like(free_arr, dtype=bool)
            region[r0:r1, c0:c1] = True
            source_free = region & free_arr
            label_free = source_free & ~boundary if bool(config.keep_boundary_pixels_unlabeled) else source_free
            free_cells = int(np.count_nonzero(label_free))
            region_cells = max(1, int(np.count_nonzero(region)))
            free_ratio = float(np.count_nonzero(source_free)) / float(region_cells)
            inside = bool(free_cells >= min_area_cells and free_ratio >= float(config.min_cell_free_ratio))
            cell = {
                "cell_id": len(cells) + 1,
                "row_cut_index": int(ri),
                "col_cut_index": int(ci),
                "bbox_rc": [int(r0), int(c0), int(r1 - 1), int(c1 - 1)],
                "polygon_rc": [[int(r0), int(c0)], [int(r0), int(c1 - 1)], [int(r1 - 1), int(c1 - 1)], [int(r1 - 1), int(c0)]],
                "free_cells": int(free_cells),
                "free_ratio": float(free_ratio),
                "inside": inside,
                "label_id": int(next_label) if inside else 0,
            }
            cells.append(cell)
            if inside:
                labels[label_free] = next_label
                next_label += 1

    edges = _cell_edges(cells, row_cuts=row_cuts, col_cuts=col_cuts, boundary=boundary)
    return labels.astype(np.int32), cells, edges


def _cell_edges(cells: Sequence[Mapping[str, object]], *, row_cuts: Sequence[int], col_cuts: Sequence[int], boundary: np.ndarray) -> list[dict]:
    by_index = {(int(cell["row_cut_index"]), int(cell["col_cut_index"])): cell for cell in cells}
    edges: list[dict] = []
    for (ri, ci), cell in by_index.items():
        label_a = int(cell.get("label_id", 0) or 0)
        if label_a <= 0:
            continue
        for dri, dci, axis in ((1, 0, "horizontal"), (0, 1, "vertical")):
            other = by_index.get((ri + dri, ci + dci))
            if not other:
                continue
            label_b = int(other.get("label_id", 0) or 0)
            if label_b <= 0:
                continue
            if axis == "vertical":
                c = int(col_cuts[ci + 1])
                r0, r1 = int(row_cuts[ri]), int(row_cuts[ri + 1])
                band = boundary[max(0, r0) : min(boundary.shape[0], r1), max(0, c - 1) : min(boundary.shape[1], c + 2)]
            else:
                r = int(row_cuts[ri + 1])
                c0, c1 = int(col_cuts[ci]), int(col_cuts[ci + 1])
                band = boundary[max(0, r - 1) : min(boundary.shape[0], r + 2), max(0, c0) : min(boundary.shape[1], c1)]
            edges.append(
                {
                    "cell_a": int(cell.get("cell_id", 0)),
                    "cell_b": int(other.get("cell_id", 0)),
                    "face_a": int(label_a),
                    "face_b": int(label_b),
                    "axis": axis,
                    "boundary_support_cells": int(np.count_nonzero(band)),
                    "edge_wall_weight": float(np.count_nonzero(band)) / float(max(1, band.size)),
                }
            )
    return edges


def _line_axis(line: Mapping[str, object], tolerance_rad: float) -> str | None:
    angle = float(line.get("orientation_rad", line.get("angle_rad", 0.0)) or 0.0) % math.pi
    horizontal = min(abs(angle), abs(math.pi - angle))
    vertical = abs(angle - math.pi / 2.0)
    if horizontal <= tolerance_rad:
        return "horizontal"
    if vertical <= tolerance_rad:
        return "vertical"
    p0 = np.asarray(line.get("p0", (0, 0)), dtype=np.float32)
    p1 = np.asarray(line.get("p1", (0, 0)), dtype=np.float32)
    dr = abs(float(p1[0] - p0[0]))
    dc = abs(float(p1[1] - p0[1]))
    if dr >= 2.0 * max(dc, 1.0):
        return "vertical"
    if dc >= 2.0 * max(dr, 1.0):
        return "horizontal"
    return None


def _axis_support_lines(
    occupied: np.ndarray,
    free: np.ndarray,
    *,
    resolution_m: float,
    min_length_m: float,
    line_gap_m: float,
) -> list[dict]:
    occ = np.asarray(occupied, dtype=bool)
    free_arr = np.asarray(free, dtype=bool)
    min_len = max(2, int(round(float(min_length_m) / max(float(resolution_m), 1e-6))))
    max_gap = max(0, int(round(float(line_gap_m) / max(float(resolution_m), 1e-6))))
    out: list[dict] = []

    def add_line(axis: str, index: int, spans: list[tuple[int, int]], support: int) -> None:
        if not spans:
            return
        start = int(min(a for a, _b in spans))
        end = int(max(b for _a, b in spans))
        span = max(1, end - start + 1)
        if support < min_len or span < min_len:
            return
        if axis == "horizontal":
            p0 = [int(index), int(start)]
            p1 = [int(index), int(end)]
            orientation = 0.0
        else:
            p0 = [int(start), int(index)]
            p1 = [int(end), int(index)]
            orientation = math.pi / 2.0
        out.append(
            {
                "line_id": int(len(out)),
                "p0": p0,
                "p1": p1,
                "orientation_rad": float(orientation),
                "support_ratio": float(support) / float(span),
                "length_m": float(span * float(resolution_m)),
                "source": "axis_support_from_structural_occupancy",
                "axis_support_cells": int(support),
                "axis_support_span_cells": int(span),
            }
        )

    for r in range(occ.shape[0]):
        spans = _merge_runs(_runs_1d(occ[r, :]), max_gap=max_gap)
        support = sum(int(b - a + 1) for a, b in spans)
        add_line("horizontal", int(r), spans, support)
    for c in range(occ.shape[1]):
        spans = _merge_runs(_runs_1d(occ[:, c]), max_gap=max_gap)
        support = sum(int(b - a + 1) for a, b in spans)
        add_line("vertical", int(c), spans, support)
    return out


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


def _merge_runs(runs: Sequence[tuple[int, int]], *, max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [tuple(runs[0])]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if int(start) - int(prev_end) - 1 <= int(max_gap):
            merged[-1] = (int(prev_start), int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def _merge_line_hypotheses(lines: Sequence[Mapping[str, object]], extra: Sequence[Mapping[str, object]], *, tolerance_cells: int) -> list[dict]:
    out = [dict(line) for line in lines]
    for candidate in extra:
        axis = _line_axis(candidate, math.radians(20.0))
        p0 = np.asarray(candidate.get("p0", (0, 0)), dtype=np.float32)
        p1 = np.asarray(candidate.get("p1", (0, 0)), dtype=np.float32)
        if axis == "vertical":
            coord = float((p0[1] + p1[1]) * 0.5)
            duplicate = any(
                _line_axis(line, math.radians(20.0)) == "vertical"
                and abs(float((np.asarray(line.get("p0", (0, 0)), dtype=np.float32)[1] + np.asarray(line.get("p1", (0, 0)), dtype=np.float32)[1]) * 0.5) - coord) <= tolerance_cells
                for line in out
            )
        elif axis == "horizontal":
            coord = float((p0[0] + p1[0]) * 0.5)
            duplicate = any(
                _line_axis(line, math.radians(20.0)) == "horizontal"
                and abs(float((np.asarray(line.get("p0", (0, 0)), dtype=np.float32)[0] + np.asarray(line.get("p1", (0, 0)), dtype=np.float32)[0]) * 0.5) - coord) <= tolerance_cells
                for line in out
            )
        else:
            duplicate = False
        if not duplicate:
            row = dict(candidate)
            row["line_id"] = int(len(out))
            out.append(row)
    return out


def _prune_cuts(cuts: Sequence[int], *, min_spacing: int, max_count: int) -> list[int]:
    raw = sorted(set(int(v) for v in cuts))
    if len(raw) <= 2:
        return raw
    keep = [raw[0]]
    for cut in raw[1:-1]:
        if cut - keep[-1] >= int(min_spacing):
            keep.append(cut)
    if raw[-1] != keep[-1]:
        keep.append(raw[-1])
    if len(keep) > max(2, int(max_count)):
        first, last = keep[0], keep[-1]
        internal = keep[1:-1]
        stride = max(1, int(math.ceil(len(internal) / max(1, int(max_count) - 2))))
        keep = [first] + internal[::stride] + [last]
    return sorted(set(keep))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
    if rr.size == 0:
        return 0, 0, mask.shape[0] - 1, mask.shape[1] - 1
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


def _metric_input_image(occupied: np.ndarray, free: np.ndarray, unknown: np.ndarray) -> Image.Image:
    canvas = np.zeros((*occupied.shape, 3), dtype=np.uint8)
    canvas[:, :] = (0, 0, 0)
    canvas[np.asarray(free, dtype=bool)] = (240, 240, 240)
    canvas[np.asarray(unknown, dtype=bool)] = (90, 90, 90)
    canvas[np.asarray(occupied, dtype=bool)] = (20, 20, 20)
    return Image.fromarray(canvas)


def _bool_image(mask: np.ndarray, *, true_color: tuple[int, int, int], false_color: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((*np.asarray(mask).shape[:2], 3), dtype=np.uint8)
    arr[:, :] = false_color
    arr[np.asarray(mask, dtype=bool)] = true_color
    return Image.fromarray(arr)


def _label_image(labels: np.ndarray) -> Image.Image:
    arr = np.asarray(labels, dtype=np.int32)
    canvas = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        canvas[arr == label_id] = _palette(label_id)
    return Image.fromarray(canvas)


def _overlay_image(occupied: np.ndarray, free: np.ndarray, unknown: np.ndarray, result: ROSE2SourceResult) -> Image.Image:
    base = np.array(_metric_input_image(occupied, free, unknown), dtype=np.uint8, copy=True)
    labels = np.asarray(result.room_label_map, dtype=np.int32)
    for label_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        color = np.asarray(_palette(label_id), dtype=np.float32)
        mask = labels == label_id
        base[mask] = np.clip(base[mask].astype(np.float32) * 0.45 + color[None, :] * 0.55, 0, 255).astype(np.uint8)
    base[np.asarray(result.boundary_map, dtype=bool)] = (255, 70, 70)
    return Image.fromarray(base)


def _line_image(free: np.ndarray, lines: Sequence[Mapping[str, object]], segments: Sequence[Mapping[str, object]]) -> Image.Image:
    scale = max(1, min(6, int(720 / max(free.shape))))
    img = _bool_image(free, true_color=(40, 40, 45), false_color=(0, 0, 0)).resize((free.shape[1] * scale, free.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)

    def xy(cell):
        return int((float(cell[1]) + 0.5) * scale), int((float(cell[0]) + 0.5) * scale)

    for seg in list(segments)[:1024]:
        draw.line([xy(seg.get("p0", (0, 0))), xy(seg.get("p1", (0, 0)))], fill=(255, 180, 70), width=max(1, scale))
    for line in list(lines)[:512]:
        draw.line([xy(line.get("p0", (0, 0))), xy(line.get("p1", (0, 0)))], fill=(80, 190, 255), width=max(2, scale + 1))
    return img


def _cell_image(free: np.ndarray, cells: Sequence[Mapping[str, object]], labels: np.ndarray) -> Image.Image:
    scale = max(1, min(6, int(720 / max(free.shape))))
    img = _label_image(labels).resize((free.shape[1] * scale, free.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for cell in list(cells)[:512]:
        r0, c0, r1, c1 = [int(v) for v in cell.get("bbox_rc", (0, 0, 0, 0))]
        color = (255, 255, 255) if bool(cell.get("inside", False)) else (120, 120, 120)
        draw.rectangle((c0 * scale, r0 * scale, (c1 + 1) * scale, (r1 + 1) * scale), outline=color, width=max(1, scale // 2))
        if bool(cell.get("inside", False)):
            draw.text((c0 * scale + 2, r0 * scale + 2), str(cell.get("label_id", "")), fill=(255, 255, 255), font=font)
    return img


def _palette(label_id: int) -> tuple[int, int, int]:
    palette = [
        (126, 174, 255),
        (255, 156, 102),
        (130, 222, 150),
        (214, 148, 255),
        (250, 216, 95),
        (95, 224, 224),
        (255, 126, 184),
        (178, 210, 120),
        (180, 165, 255),
    ]
    return palette[(int(label_id) - 1) % len(palette)]


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value
