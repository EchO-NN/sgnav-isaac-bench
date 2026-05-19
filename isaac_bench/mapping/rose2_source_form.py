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
from isaac_bench.mapping.rose2_partition_graph import (
    absorb_partition_boundary_pixels_without_merging,
    generate_doorway_partition_cuts,
    labels_from_partition_boundary_v2,
    select_topology_effective_separators,
)
from isaac_bench.mapping.rose2_separator_detection import detect_thin_wall_separator_candidates


SOURCE_FORM_V1_BACKEND = "rose2_source_form"
SOURCE_FORM_BACKEND = "rose2_source_form_v2"
SOURCE_EXTERNAL_BACKEND = "rose2_source_external"
SOURCE_EXTERNAL_RUNNER_BACKEND = "rose2_source_external_runner"
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
    thin_wall_separator_enabled: bool = True
    thin_wall_min_length_m: float = 0.45
    thin_wall_max_width_m: float = 0.25
    thin_wall_min_aspect_ratio: float = 3.0
    thin_wall_free_support_band_m: float = 0.30
    thin_wall_min_free_support_ratio: float = 0.25
    topology_effective_separator_enabled: bool = True
    topology_effective_min_largest_component_drop: float = 0.08
    topology_effective_max_candidates: int = 128
    doorway_partition_enabled: bool = True
    doorway_width_min_m: float = 0.45
    doorway_width_max_m: float = 1.60
    doorway_min_wall_support_on_sides_m: float = 0.35
    merge_guard_enabled: bool = True
    compare_legacy: bool = False
    compare_external_if_available: bool = False

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

    occupied_for_vertical = np.asarray(observed_occupied, dtype=bool)
    free_for_vertical = np.asarray(observed_free, dtype=bool)
    unknown_for_vertical = np.asarray(unknown, dtype=bool)
    return run_rose2_source_form_v2(
        observed_occupied=occupied_for_vertical,
        observed_free=free_for_vertical,
        unknown=unknown_for_vertical,
        vertical_observed=occupied_for_vertical | free_for_vertical,
        vertical_free=free_for_vertical,
        wall_confidence_map=occupied_for_vertical.astype(np.float32),
        structure_config=structure_config,
        source_config=source_config,
        object_memory=object_memory,
    )

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
        "source_backend": SOURCE_FORM_V1_BACKEND,
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
        backend=SOURCE_FORM_V1_BACKEND,
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


def run_rose2_source_form_v2(
    *,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    vertical_observed: np.ndarray | None,
    vertical_free: np.ndarray | None,
    wall_confidence_map: np.ndarray | None,
    structure_config: StructureExtractionConfig,
    source_config: ROSE2SourceFormConfig | None = None,
    object_memory: Optional[Sequence[object]] = None,
) -> ROSE2SourceResult:
    """Run ROSE2 source-form v2 with topology-effective separators.

    v2 keeps the same strict 0.2-2.0 m vertical-free input semantics, but it
    additionally promotes slender non-free vertical-profile structures into
    candidate room separators and accepts only separators that actually split
    the observed free-space topology into room-sized regions.
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
        raise ValueError("ROSE2 source-form-v2 inputs must have the same HxW shape")
    v_observed = np.asarray(vertical_observed, dtype=bool) if vertical_observed is not None else (free | occupied | ~unknown_arr)
    v_free = np.asarray(vertical_free, dtype=bool) if vertical_free is not None else free
    if v_observed.shape != free.shape or v_free.shape != free.shape:
        raise ValueError("ROSE2 source-form-v2 vertical masks must match the input shape")

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
    for row in representative:
        row.setdefault("source", "rose2_representative_wall")
    extended = extend_representative_lines_to_free_boundary(representative, free, clean_structure_map.shape, structure_config)
    for row in extended:
        row.setdefault("source", "rose2_representative_wall")
    axis_support = _axis_support_lines(
        cleaned,
        free,
        resolution_m=float(structure_config.resolution_m),
        min_length_m=float(structure_config.hough_min_line_length_m),
        line_gap_m=float(structure_config.hough_line_gap_m),
    )
    t_lines = time.perf_counter()

    thin_lines: list[dict] = []
    thin_mask = np.zeros_like(free, dtype=bool)
    thin_debug: dict = {
        "thin_wall_separator_count": 0,
        "vertical_free_separator_count": 0,
        "separator_mask_cells": 0,
    }
    if bool(cfg.thin_wall_separator_enabled):
        thin_lines, thin_mask, thin_debug = detect_thin_wall_separator_candidates(
            vertical_free=v_free,
            vertical_observed=v_observed,
            occupied=occupied,
            unknown=unknown_arr,
            wall_confidence_map=wall_confidence_map,
            resolution_m=float(structure_config.resolution_m),
            min_length_m=float(cfg.thin_wall_min_length_m),
            max_width_m=float(cfg.thin_wall_max_width_m),
            free_support_band_m=float(cfg.thin_wall_free_support_band_m),
            min_free_support_ratio=float(cfg.thin_wall_min_free_support_ratio),
            min_aspect_ratio=float(cfg.thin_wall_min_aspect_ratio),
            doorway_width_max_m=float(cfg.doorway_width_max_m),
        )
        thin_debug["vertical_free_separator_count"] = int(thin_debug.get("thin_wall_separator_count", len(thin_lines)) or 0)
        thin_lines = _extend_lines_to_free_bbox(thin_lines, free, resolution_m=float(cfg.resolution_m))

    raw_lines = _merge_line_hypotheses(thin_lines, extended + axis_support, tolerance_cells=2)
    raw_boundary = rasterize_representative_lines(raw_lines, clean_structure_map.shape, int(cfg.wall_raster_radius_cells))
    doorway_cuts: list[dict] = []
    doorway_debug: dict = {"doorway_partition_cut_count": 0, "doorway_partition_cuts": []}
    if bool(cfg.doorway_partition_enabled):
        doorway_cuts, doorway_debug = generate_doorway_partition_cuts(
            free=free,
            wall_boundary=raw_boundary,
            selected_separator_boundary=raw_boundary,
            candidate_lines=raw_lines,
            resolution_m=float(cfg.resolution_m),
            doorway_width_min_m=float(cfg.doorway_width_min_m),
            doorway_width_max_m=float(cfg.doorway_width_max_m),
            min_wall_support_on_sides_m=float(cfg.doorway_min_wall_support_on_sides_m),
        )
    candidates = raw_lines + doorway_cuts
    if bool(cfg.topology_effective_separator_enabled):
        partition_boundary, accepted_lines, partition_debug = select_topology_effective_separators(
            free=free,
            unknown=unknown_arr,
            base_boundary=np.zeros_like(free, dtype=bool),
            candidate_lines=candidates,
            resolution_m=float(cfg.resolution_m),
            min_room_area_m2=float(cfg.min_room_area_m2),
            wall_raster_radius_cells=int(cfg.wall_raster_radius_cells),
            max_candidates=int(cfg.topology_effective_max_candidates),
            min_largest_component_drop=float(cfg.topology_effective_min_largest_component_drop),
        )
    else:
        partition_boundary = raw_boundary
        accepted_lines = list(candidates)
        partition_debug = {
            "topology_effective_separator_selection": False,
            "accepted_topology_separator_count": int(len(accepted_lines)),
            "rejected_separator_count": 0,
            "accepted_separators": accepted_lines,
            "rejected_separators": [],
        }
    t_partition = time.perf_counter()
    labels_before_absorb, faces_before = labels_from_partition_boundary_v2(
        free=free,
        partition_boundary=partition_boundary,
        unknown=unknown_arr,
        resolution_m=float(cfg.resolution_m),
        min_room_area_m2=float(cfg.min_room_area_m2),
    )
    labels_after_absorb, absorption_debug = absorb_partition_boundary_pixels_without_merging(
        labels_before_absorb,
        free=free,
        partition_boundary=partition_boundary,
        unknown=unknown_arr,
    )
    faces = faces_from_labels(labels_after_absorb, structure_config)
    adjacency = _dedupe_edges(
        face_adjacency_edges(labels_after_absorb, partition_boundary)
        + _partition_boundary_adjacency_edges(labels_after_absorb, partition_boundary)
    )
    t_labels = time.perf_counter()
    room_count = _label_count(labels_after_absorb)
    accepted_mask = rasterize_representative_lines(accepted_lines, free.shape, int(cfg.wall_raster_radius_cells))
    rejected_lines = list(partition_debug.get("rejected_separators") or [])
    rejected_mask = rasterize_representative_lines(rejected_lines, free.shape, int(cfg.wall_raster_radius_cells)) & ~accepted_mask
    doorway_mask = rasterize_representative_lines(doorway_cuts, free.shape, int(cfg.wall_raster_radius_cells))
    raw_candidate_mask = rasterize_representative_lines(candidates, free.shape, int(cfg.wall_raster_radius_cells))
    likely_cause = _likely_cause_if_room_count_1(
        room_count=room_count,
        thin_count=int(thin_debug.get("thin_wall_separator_count", 0) or 0),
        accepted_count=int(partition_debug.get("accepted_topology_separator_count", 0) or 0),
        raw_count=int(len(candidates)),
    )
    debug = {
        "source_backend": SOURCE_FORM_BACKEND,
        "source_form_used": True,
        "source_form_v2_used": True,
        "source_exact_used": False,
        "legacy_style_used": False,
        "legacy_connected_component_rooms_used": False,
        "source_form_pipeline": [
            "dominant_direction_fft",
            "directional_structure_score",
            "hough_line_segments",
            "angular_clustering",
            "representative_line_extension",
            "thin_wall_separator_promotion_from_vertical_free_complement",
            "doorway_partition_cut_detection",
            "topology_effective_separator_selection",
            "partition_boundary_connected_room_labels",
            "boundary_pixel_absorption_without_merging",
        ],
        "vertical_free_source": "vertical_profile_0p2_2p0",
        "vertical_free_cells": int(np.count_nonzero(v_free)),
        "vertical_observed_cells": int(np.count_nonzero(v_observed)),
        "observed_not_vertical_free_cells": int(np.count_nonzero(v_observed & ~v_free)),
        "source_room_count": int(room_count),
        "proposal_room_count": int(room_count),
        "final_room_count_before_policy_merge": int(room_count),
        "wall_segment_count": int(len(segments)),
        "wall_cluster_count": int(len(clusters)),
        "extended_wall_line_count": int(len(extended)),
        "axis_support_line_count": int(len(axis_support)),
        "raw_candidate_separator_count": int(len(candidates)),
        "thin_wall_separator_count": int(thin_debug.get("thin_wall_separator_count", len(thin_lines)) or 0),
        "vertical_free_separator_count": int(thin_debug.get("vertical_free_separator_count", len(thin_lines)) or 0),
        "doorway_partition_cut_count": int(doorway_debug.get("doorway_partition_cut_count", len(doorway_cuts)) or 0),
        "accepted_topology_separator_count": int(partition_debug.get("accepted_topology_separator_count", len(accepted_lines)) or 0),
        "rejected_separator_count": int(partition_debug.get("rejected_separator_count", len(rejected_lines)) or 0),
        "cell_edge_count": int(len(adjacency)),
        "source_cell_count": int(len(faces_before)),
        "inside_source_cell_count": int(len(faces_before)),
        "labels_outside_vertical_free_cells": int(np.count_nonzero((labels_after_absorb > 0) & ~free)),
        "threshold": threshold_debug,
        "thin_wall_debug": thin_debug,
        "doorway_partition_debug": doorway_debug,
        "partition_debug": partition_debug,
        "boundary_absorption_debug": absorption_debug,
        "accepted_separators": list(accepted_lines),
        "rejected_separators": rejected_lines[:128],
        "doorway_partition_cuts": list(doorway_cuts),
        "cell_polygons": faces,
        "source_cell_edges": adjacency,
        "source_room_label_map": labels_after_absorb.astype(np.int32),
        "room_label_map_before_absorb": labels_before_absorb.astype(np.int32),
        "room_label_map_after_absorb": labels_after_absorb.astype(np.int32),
        "source_form_boundary_map": partition_boundary.astype(bool),
        "partition_boundary_map": partition_boundary.astype(bool),
        "raw_candidate_separator_mask": raw_candidate_mask.astype(bool),
        "thin_wall_separator_mask": thin_mask.astype(bool),
        "doorway_partition_cut_mask": doorway_mask.astype(bool),
        "accepted_separator_mask": accepted_mask.astype(bool),
        "rejected_separator_mask": rejected_mask.astype(bool),
        "hough_segments_raster": rasterize_representative_lines(segments, free.shape, int(cfg.wall_raster_radius_cells)),
        "representative_lines_raster": rasterize_representative_lines(extended, free.shape, int(cfg.wall_raster_radius_cells)),
        "axis_support_lines_raster": rasterize_representative_lines(axis_support, free.shape, int(cfg.wall_raster_radius_cells)),
        "failure_mode": "" if room_count > 0 else "no_partition_rooms",
        "likely_cause_if_room_count_1": likely_cause,
        "merge_guard_enabled": bool(cfg.merge_guard_enabled),
    }
    timing = {
        "cleaning": (t_clean - t0) * 1000.0,
        "dft_structure": (t_score - t_clean) * 1000.0,
        "wall_lines": (t_lines - t_score) * 1000.0,
        "partition_selection": (t_partition - t_lines) * 1000.0,
        "room_labels": (t_labels - t_partition) * 1000.0,
        "total": (t_labels - t0) * 1000.0,
    }
    debug["timing_ms"] = dict(timing)
    return ROSE2SourceResult(
        backend=SOURCE_FORM_BACKEND,
        room_label_map=labels_after_absorb.astype(np.int32),
        source_room_label_map=labels_after_absorb.astype(np.int32),
        clean_structure_map=clean_structure_map.astype(bool),
        structural_score=structural_score.astype(np.float32),
        boundary_map=partition_boundary.astype(bool),
        dominant_directions_rad=[float(v) for v in dominant],
        hough_segments=[dict(item) for item in segments],
        wall_clusters=[dict(item) for item in clusters],
        representative_lines=[dict(item) for item in representative],
        extended_lines=[dict(item) for item in accepted_lines],
        faces=faces,
        face_adjacency_edges=adjacency,
        cell_edges=adjacency,
        cell_polygons=faces,
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
        "input_masks_npz": str(out / ("%s.input_masks.npz" % stem)),
        "summary_json": str(out / ("%s.summary.json" % stem)),
        "metric_input_png": str(out / ("%s.metric_input.png" % stem)),
        "vertical_free_png": str(out / ("%s.vertical_free.png" % stem)),
        "vertical_observed_png": str(out / ("%s.vertical_observed.png" % stem)),
        "observed_not_vertical_free_png": str(out / ("%s.observed_not_vertical_free.png" % stem)),
        "repaired_occupied_png": str(out / ("%s.repaired_occupied.png" % stem)),
        "repaired_free_png": str(out / ("%s.repaired_free.png" % stem)),
        "vertical_occupied_png": str(out / ("%s.vertical_occupied.png" % stem)),
        "unknown_png": str(out / ("%s.unknown.png" % stem)),
        "thin_wall_candidates_png": str(out / ("%s.thin_wall_candidates.png" % stem)),
        "accepted_separators_png": str(out / ("%s.accepted_separators.png" % stem)),
        "partition_boundary_png": str(out / ("%s.partition_boundary.png" % stem)),
        "room_labels_png": str(out / ("%s.room_labels.png" % stem)),
        "source_labels_png": str(out / ("%s.source_labels.png" % stem)),
        "overlay_png": str(out / ("%s.overlay.png" % stem)),
        "lines_png": str(out / ("%s.lines.png" % stem)),
        "cells_png": str(out / ("%s.cells.png" % stem)),
    }
    debug = dict(result.debug or {})
    vertical_observed = _debug_mask(debug, "vertical_observed", occupied | free | ~unknown_arr, occupied.shape)
    observed_not_vertical_free = _debug_mask(debug, "observed_not_vertical_free", vertical_observed & ~free, occupied.shape)
    repaired_occupied = _debug_mask(debug, "repaired_roomseg_occupied", occupied, occupied.shape)
    repaired_free = _debug_mask(debug, "repaired_roomseg_free", free, occupied.shape)
    thin_wall_mask = _debug_mask(debug, "thin_wall_separator_mask", np.zeros_like(occupied, dtype=bool), occupied.shape)
    doorway_mask = _debug_mask(debug, "doorway_partition_cut_mask", np.zeros_like(occupied, dtype=bool), occupied.shape)
    raw_candidate_mask = _debug_mask(debug, "raw_candidate_separator_mask", np.zeros_like(occupied, dtype=bool), occupied.shape)
    accepted_mask = _debug_mask(debug, "accepted_separator_mask", result.boundary_map, occupied.shape)
    rejected_mask = _debug_mask(debug, "rejected_separator_mask", np.zeros_like(occupied, dtype=bool), occupied.shape)
    labels_before = np.asarray(debug.get("room_label_map_before_absorb", result.source_room_label_map), dtype=np.int32)
    labels_after = np.asarray(debug.get("room_label_map_after_absorb", result.room_label_map), dtype=np.int32)
    np.savez_compressed(
        paths["npz"],
        observed_occupied=occupied.astype(np.uint8),
        observed_free=free.astype(np.uint8),
        unknown=unknown_arr.astype(np.uint8),
        vertical_free_room_domain=free.astype(np.uint8),
        vertical_observed=vertical_observed.astype(np.uint8),
        observed_not_vertical_free=observed_not_vertical_free.astype(np.uint8),
        repaired_roomseg_free=repaired_free.astype(np.uint8),
        repaired_roomseg_occupied=repaired_occupied.astype(np.uint8),
        repaired_roomseg_unknown=unknown_arr.astype(np.uint8),
        wall_confidence_map=np.asarray(debug.get("wall_confidence_map", np.zeros_like(occupied, dtype=np.float32)), dtype=np.float32),
        clean_structure_map=result.clean_structure_map.astype(np.uint8),
        boundary_map=result.boundary_map.astype(np.uint8),
        partition_boundary_map=result.boundary_map.astype(np.uint8),
        hough_segments_raster=_debug_mask(debug, "hough_segments_raster", np.zeros_like(occupied, dtype=bool), occupied.shape).astype(np.uint8),
        representative_lines_raster=_debug_mask(debug, "representative_lines_raster", np.zeros_like(occupied, dtype=bool), occupied.shape).astype(np.uint8),
        axis_support_lines_raster=_debug_mask(debug, "axis_support_lines_raster", np.zeros_like(occupied, dtype=bool), occupied.shape).astype(np.uint8),
        thin_wall_separator_mask=thin_wall_mask.astype(np.uint8),
        doorway_partition_cut_mask=doorway_mask.astype(np.uint8),
        raw_candidate_separator_mask=raw_candidate_mask.astype(np.uint8),
        accepted_separator_mask=accepted_mask.astype(np.uint8),
        rejected_separator_mask=rejected_mask.astype(np.uint8),
        room_label_map_before_absorb=labels_before.astype(np.int32),
        room_label_map_after_absorb=labels_after.astype(np.int32),
        source_room_label_map=result.source_room_label_map.astype(np.int32),
        room_label_map=result.room_label_map.astype(np.int32),
        structural_score=result.structural_score.astype(np.float32),
    )
    np.savez_compressed(
        paths["input_masks_npz"],
        observed_occupied=occupied.astype(np.uint8),
        observed_free=free.astype(np.uint8),
        unknown=unknown_arr.astype(np.uint8),
        vertical_free_room_domain=free.astype(np.uint8),
        vertical_observed=vertical_observed.astype(np.uint8),
        repaired_roomseg_occupied=repaired_occupied.astype(np.uint8),
        repaired_roomseg_free=repaired_free.astype(np.uint8),
        repaired_roomseg_unknown=unknown_arr.astype(np.uint8),
        wall_confidence_map=np.asarray(debug.get("wall_confidence_map", np.zeros_like(occupied, dtype=np.float32)), dtype=np.float32),
    )
    summary = source_result_summary(result)
    Path(paths["summary_json"]).write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _metric_input_image(occupied, free, unknown_arr).save(paths["metric_input_png"])
    _bool_image(free, true_color=(240, 240, 240), false_color=(0, 0, 0)).save(paths["vertical_free_png"])
    _bool_image(vertical_observed, true_color=(180, 180, 255), false_color=(0, 0, 0)).save(paths["vertical_observed_png"])
    _bool_image(observed_not_vertical_free, true_color=(255, 255, 255), false_color=(0, 0, 0)).save(paths["observed_not_vertical_free_png"])
    _bool_image(repaired_occupied, true_color=(240, 240, 240), false_color=(0, 0, 0)).save(paths["repaired_occupied_png"])
    _bool_image(repaired_free, true_color=(240, 240, 240), false_color=(0, 0, 0)).save(paths["repaired_free_png"])
    _bool_image(occupied, true_color=(240, 240, 240), false_color=(0, 0, 0)).save(paths["vertical_occupied_png"])
    _bool_image(unknown_arr, true_color=(150, 150, 150), false_color=(0, 0, 0)).save(paths["unknown_png"])
    _bool_image(thin_wall_mask, true_color=(255, 40, 40), false_color=(0, 0, 0)).save(paths["thin_wall_candidates_png"])
    _bool_image(accepted_mask, true_color=(255, 0, 255), false_color=(0, 0, 0)).save(paths["accepted_separators_png"])
    _bool_image(result.boundary_map, true_color=(255, 0, 255), false_color=(0, 0, 0)).save(paths["partition_boundary_png"])
    _label_image(labels_after).save(paths["room_labels_png"])
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
        "vertical_free_separator_count": int(debug.get("vertical_free_separator_count", 0) or 0),
        "thin_wall_separator_count": int(debug.get("thin_wall_separator_count", 0) or 0),
        "doorway_partition_cut_count": int(debug.get("doorway_partition_cut_count", 0) or 0),
        "accepted_topology_separator_count": int(debug.get("accepted_topology_separator_count", 0) or 0),
        "rejected_separator_count": int(debug.get("rejected_separator_count", 0) or 0),
        "labels_outside_vertical_free_cells": int(debug.get("labels_outside_vertical_free_cells", 0) or 0),
        "legacy_connected_component_rooms_used": bool(debug.get("legacy_connected_component_rooms_used", False)),
        "failure_mode": str(debug.get("failure_mode", "")),
        "likely_cause_if_room_count_1": str(debug.get("likely_cause_if_room_count_1", "")),
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
    """Run the external declutter-reconstruct wrapper without fallback."""

    from isaac_bench.mapping.rose2_source_external_runner import run_rose2_source_external_runner

    return run_rose2_source_external_runner(
        source_root=source_root,
        observed_occupied=observed_occupied,
        observed_free=observed_free,
        unknown=unknown,
        work_dir=work_dir,
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


def _extend_lines_to_free_bbox(lines: Sequence[Mapping[str, object]], free: np.ndarray, *, resolution_m: float) -> list[dict]:
    if not lines or not np.any(free):
        return [dict(line) for line in lines]
    rmin, cmin, rmax, cmax = _bbox(np.asarray(free, dtype=bool))
    out: list[dict] = []
    for line in lines:
        row = dict(line)
        axis = _line_axis(row, math.radians(20.0))
        p0 = np.asarray(row.get("p0", (0, 0)), dtype=np.float32)
        p1 = np.asarray(row.get("p1", (0, 0)), dtype=np.float32)
        if axis == "vertical":
            c = int(round(float((p0[1] + p1[1]) * 0.5)))
            row["p0"] = [int(rmin), int(c)]
            row["p1"] = [int(rmax), int(c)]
        elif axis == "horizontal":
            r = int(round(float((p0[0] + p1[0]) * 0.5)))
            row["p0"] = [int(r), int(cmin)]
            row["p1"] = [int(r), int(cmax)]
        else:
            out.append(row)
            continue
        row["extended_to_vertical_free_bbox"] = True
        row["extended_to_observed_free_boundary"] = True
        row["pre_bbox_extension_p0"] = [int(round(float(v))) for v in p0]
        row["pre_bbox_extension_p1"] = [int(round(float(v))) for v in p1]
        row["length_m"] = float(np.linalg.norm(np.asarray(row["p1"], dtype=np.float32) - np.asarray(row["p0"], dtype=np.float32)) * float(resolution_m))
        out.append(row)
    return out


def _partition_boundary_adjacency_edges(labels: np.ndarray, boundary: np.ndarray) -> list[dict]:
    label_arr = np.asarray(labels, dtype=np.int32)
    boundary_arr = np.asarray(boundary, dtype=bool)
    edges: dict[tuple[int, int], dict] = {}
    for r, c in zip(*np.nonzero(boundary_arr)):
        neigh = sorted(
            {
                int(label_arr[nr, nc])
                for nr in range(max(0, int(r) - 1), min(label_arr.shape[0], int(r) + 2))
                for nc in range(max(0, int(c) - 1), min(label_arr.shape[1], int(c) + 2))
                if int(label_arr[nr, nc]) > 0
            }
        )
        if len(neigh) < 2:
            continue
        for idx, a in enumerate(neigh):
            for b in neigh[idx + 1 :]:
                key = tuple(sorted((int(a), int(b))))
                row = edges.setdefault(
                    key,
                    {
                        "face_a": key[0],
                        "face_b": key[1],
                        "shared_cells": 0,
                        "edge_wall_weight": 1.0,
                        "source": "partition_boundary_adjacency",
                    },
                )
                row["shared_cells"] += 1
    return list(edges.values())


def _dedupe_edges(edges: Sequence[Mapping[str, object]]) -> list[dict]:
    out: dict[tuple[int, int], dict] = {}
    for edge in edges:
        a = int(edge.get("face_a", edge.get("room_a", 0)) or 0)
        b = int(edge.get("face_b", edge.get("room_b", 0)) or 0)
        if a <= 0 or b <= 0 or a == b:
            continue
        key = tuple(sorted((a, b)))
        row = out.setdefault(key, {"face_a": key[0], "face_b": key[1], "shared_cells": 0, "edge_wall_weight": 0.0})
        row["shared_cells"] = int(row.get("shared_cells", 0)) + int(edge.get("shared_cells", 0) or 1)
        row["edge_wall_weight"] = max(float(row.get("edge_wall_weight", 0.0) or 0.0), float(edge.get("edge_wall_weight", 0.0) or 0.0))
        if "source" in edge:
            row["source"] = str(edge.get("source"))
    return list(out.values())


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


def _debug_mask(debug: Mapping[str, object], key: str, default: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = debug.get(key, default)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != tuple(shape):
        return np.asarray(default, dtype=bool)
    return arr


def _likely_cause_if_room_count_1(*, room_count: int, thin_count: int, accepted_count: int, raw_count: int) -> str:
    if int(room_count) != 1:
        return ""
    if int(thin_count) <= 0:
        return "thin_wall_not_promoted"
    if int(raw_count) > 0 and int(accepted_count) <= 0:
        return "line_detected_but_not_topology_effective"
    return "unclear"


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
