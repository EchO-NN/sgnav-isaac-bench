from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


GridCell = Tuple[int, int]


@dataclass
class StructureExtractionConfig:
    resolution_m: float = 0.05
    dominant_direction_count: int = 2
    dominant_direction_min_separation_rad: float = math.radians(25.0)
    directional_filter_width_rad: float = math.radians(14.0)
    clean_threshold_min_percentile: float = 70.0
    clean_threshold_max_percentile: float = 92.0
    hough_min_line_length_m: float = 1.0
    hough_line_gap_m: float = 0.25
    wall_cluster_angle_rad: float = math.radians(8.0)
    wall_cluster_distance_m: float = 0.35
    wall_cluster_gap_m: float = 0.50
    wall_min_support_ratio: float = 0.25
    wall_raster_radius_cells: int = 1
    clutter_component_max_area_m2: float = 1.2
    min_room_area_m2: float = 1.5
    topology_split_enabled: bool = True

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]] = None, **overrides) -> "StructureExtractionConfig":
        raw = dict(data or {})
        raw.update({key: value for key, value in overrides.items() if value is not None})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class StructureExtractionResult:
    dominant_directions_rad: List[float]
    structural_score: np.ndarray
    clean_structure_map: np.ndarray
    hough_segments: List[dict]
    wall_clusters: List[dict]
    representative_lines: List[dict]
    boundary_map: np.ndarray
    face_labels: np.ndarray
    faces: List[dict]
    face_adjacency_edges: List[dict]
    topology_debug: Optional[dict]
    timing_ms: dict = field(default_factory=dict)


def extract_rose2_structure(
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    config: StructureExtractionConfig,
    object_memory: Optional[Iterable[object]] = None,
) -> StructureExtractionResult:
    """Practical ROSE2-style structure extraction from a 2D occupancy grid.

    This is deliberately CPU-only and deterministic. It follows the ROSE2
    shape of the computation: dominant wall directions from the DFT spectrum,
    directional frequency filtering, inverse-DFT structural scoring, line
    segment extraction, wall clustering, and face extraction from a rasterized
    representative wall map.
    """

    t0 = time.perf_counter()
    occupied = np.asarray(observed_occupied, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    if occupied.shape != free.shape or occupied.shape != unknown_arr.shape:
        raise ValueError("ROSE2 structure inputs must have the same HxW shape")

    cleaned = remove_isolated_clutter(occupied, free, config, object_memory=object_memory)
    t_clean = time.perf_counter()
    dominant = dominant_directions_from_fft(cleaned, int(config.dominant_direction_count), float(config.dominant_direction_min_separation_rad))
    structural_score = directional_structural_score(cleaned, dominant, float(config.directional_filter_width_rad))
    clean_structure_map, threshold_debug = auto_threshold_structure_map(structural_score, cleaned, free, config)
    t_score = time.perf_counter()
    segments = detect_line_segments(clean_structure_map, config)
    if not segments:
        fallback_segments = detect_line_segments(cleaned, config)
        if fallback_segments:
            clean_structure_map = cleaned.copy()
            segments = fallback_segments
    clusters = cluster_wall_segments(segments, dominant, config)
    representative_lines = representative_lines_from_clusters(clusters, clean_structure_map.shape, config)
    boundary_map = rasterize_representative_lines(representative_lines, clean_structure_map.shape, int(config.wall_raster_radius_cells))
    face_labels, faces = faces_from_boundary_map(free, boundary_map, config)
    topology_debug = topology_split_debug(face_labels, free, config) if bool(config.topology_split_enabled) else None
    if bool(config.topology_split_enabled) and topology_debug and topology_debug.get("splits"):
        face_labels = split_faces_by_topology(face_labels, free, config)
        faces = faces_from_labels(face_labels, config)
        topology_debug = topology_split_debug(face_labels, free, config)
        topology_debug["applied"] = True
    face_edges = face_adjacency_edges(face_labels, boundary_map)
    t_end = time.perf_counter()
    return StructureExtractionResult(
        dominant_directions_rad=dominant,
        structural_score=structural_score.astype(np.float32),
        clean_structure_map=clean_structure_map.astype(bool),
        hough_segments=segments,
        wall_clusters=clusters,
        representative_lines=representative_lines,
        boundary_map=boundary_map.astype(bool),
        face_labels=face_labels.astype(np.int32),
        faces=faces,
        face_adjacency_edges=face_edges,
        topology_debug=topology_debug,
        timing_ms={
            "cleaning": (t_clean - t0) * 1000.0,
            "dft_structure": (t_score - t_clean) * 1000.0,
            "wall_faces": (t_end - t_score) * 1000.0,
            "total": (t_end - t0) * 1000.0,
            "threshold": threshold_debug,
        },
    )


def remove_isolated_clutter(
    occupied: np.ndarray,
    free: np.ndarray,
    config: StructureExtractionConfig,
    object_memory: Optional[Iterable[object]] = None,
) -> np.ndarray:
    occ = np.asarray(occupied, dtype=bool).copy()
    if not np.any(occ):
        return occ
    neighbor_count = _neighbor_count_8(occ)
    occ &= neighbor_count >= 2
    max_cells = max(1, int(round(float(config.clutter_component_max_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    out = occ.copy()
    for component in connected_components(occ):
        if len(component) > max_cells:
            continue
        rows = [cell[0] for cell in component]
        cols = [cell[1] for cell in component]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        aspect = max(height, width) / max(1, min(height, width))
        if aspect < 3.0:
            for r, c in component:
                out[r, c] = False
    if object_memory is not None:
        for node in object_memory:
            footprint = getattr(node, "center_grid", None)
            if footprint is None:
                continue
            try:
                rr, cc = int(footprint[0]), int(footprint[1])
            except Exception:
                continue
            radius = max(1, int(round(0.20 / max(float(config.resolution_m), 1e-6))))
            r0, r1 = max(0, rr - radius), min(out.shape[0], rr + radius + 1)
            c0, c1 = max(0, cc - radius), min(out.shape[1], cc + radius + 1)
            if np.count_nonzero(out[r0:r1, c0:c1]) < (r1 - r0) * (c1 - c0):
                out[r0:r1, c0:c1] = False
    return out


def dominant_directions_from_fft(occupied: np.ndarray, count: int = 2, min_separation_rad: float = math.radians(25.0)) -> List[float]:
    occ = np.asarray(occupied, dtype=np.float32)
    if not np.any(occ):
        return [0.0, math.pi / 2.0][: max(1, int(count))]
    spectrum = np.fft.fftshift(np.fft.fft2(occ - float(np.mean(occ))))
    amp = np.abs(spectrum)
    h, w = occ.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    valid = radius > max(1.0 / max(h, w), 1e-6)
    # Frequency orientation is perpendicular to line orientation.
    line_theta = _canonical_angle(np.arctan2(yy, xx) + math.pi / 2.0)
    bins = np.linspace(0.0, math.pi, 181)
    hist = np.zeros(len(bins) - 1, dtype=np.float64)
    flat_theta = line_theta[valid].ravel()
    flat_amp = amp[valid].ravel()
    indices = np.clip(np.searchsorted(bins, flat_theta, side="right") - 1, 0, len(hist) - 1)
    np.add.at(hist, indices, flat_amp)
    candidates = np.argsort(hist)[::-1]
    selected: List[float] = []
    for idx in candidates:
        angle = float((bins[idx] + bins[idx + 1]) * 0.5)
        if all(_angular_distance_pi(angle, prev) >= float(min_separation_rad) for prev in selected):
            selected.append(angle)
        if len(selected) >= max(1, int(count)):
            break
    if not selected:
        selected = [0.0, math.pi / 2.0][: max(1, int(count))]
    return sorted(float(_canonical_angle(v)) for v in selected)


def directional_structural_score(occupied: np.ndarray, dominant_directions: Sequence[float], filter_width_rad: float) -> np.ndarray:
    occ = np.asarray(occupied, dtype=np.float32)
    if not np.any(occ):
        return np.zeros_like(occ, dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(occ - float(np.mean(occ))))
    h, w = occ.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    line_theta = _canonical_angle(np.arctan2(yy, xx) + math.pi / 2.0)
    keep = np.zeros_like(occ, dtype=bool)
    for direction in dominant_directions:
        keep |= _angular_distance_pi_array(line_theta, float(direction)) <= float(filter_width_rad)
    filtered = np.where(keep, spectrum, 0.0)
    inv = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    inv = np.abs(inv)
    if float(np.max(inv)) > 1e-9:
        inv = inv / float(np.max(inv))
    return inv.astype(np.float32)


def auto_threshold_structure_map(
    score: np.ndarray,
    occupied: np.ndarray,
    free: np.ndarray,
    config: StructureExtractionConfig,
) -> Tuple[np.ndarray, dict]:
    values = np.asarray(score, dtype=np.float32)[np.asarray(occupied, dtype=bool)]
    if values.size == 0:
        return np.zeros_like(score, dtype=bool), {"selected_percentile": None, "coverage": 0.0}
    best_mask = None
    best_cost = float("inf")
    best_info = {}
    total_free = max(1, int(np.count_nonzero(free)))
    for percentile in np.linspace(float(config.clean_threshold_min_percentile), float(config.clean_threshold_max_percentile), 8):
        threshold = float(np.percentile(values, percentile))
        mask = (score >= threshold) & occupied
        coverage = float(np.count_nonzero(mask)) / float(total_free)
        segment_count = len(detect_line_segments(mask, config, fast=True))
        cost = abs(coverage - 0.035) + 0.015 * max(0, 3 - segment_count)
        if cost < best_cost:
            best_cost = cost
            best_mask = mask
            best_info = {
                "selected_percentile": float(percentile),
                "threshold": threshold,
                "coverage": coverage,
                "segment_count": int(segment_count),
            }
    return np.asarray(best_mask, dtype=bool), best_info


def detect_line_segments(clean_structure_map: np.ndarray, config: StructureExtractionConfig, fast: bool = False) -> List[dict]:
    clean = np.asarray(clean_structure_map, dtype=bool)
    if not np.any(clean):
        return []
    min_len_px = max(2, int(round(float(config.hough_min_line_length_m) / max(float(config.resolution_m), 1e-6))))
    gap_px = max(1, int(round(float(config.hough_line_gap_m) / max(float(config.resolution_m), 1e-6))))
    try:
        from skimage.transform import probabilistic_hough_line

        lines = probabilistic_hough_line(clean.astype(np.uint8), threshold=5, line_length=min_len_px, line_gap=gap_px)
        out = []
        for idx, ((x0, y0), (x1, y1)) in enumerate(lines):
            out.append(_segment_dict(idx, (int(y0), int(x0)), (int(y1), int(x1)), clean))
        if out or not fast:
            return out
    except Exception:
        pass
    return _axis_aligned_segments(clean, min_len_px=min_len_px)


def cluster_wall_segments(segments: Sequence[Mapping[str, object]], dominant_directions: Sequence[float], config: StructureExtractionConfig) -> List[dict]:
    if not segments:
        return []
    parent = list(range(len(segments)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(segments):
        for j, b in enumerate(segments[i + 1 :], start=i + 1):
            if _segments_clusterable(a, b, dominant_directions, config):
                union(i, j)
    groups: dict[int, list[Mapping[str, object]]] = {}
    for idx, seg in enumerate(segments):
        groups.setdefault(find(idx), []).append(seg)
    clusters = []
    for cid, items in enumerate(groups.values()):
        length = float(sum(float(item.get("length_px", 0.0)) for item in items))
        angle = _mean_angle([float(item.get("angle_rad", 0.0)) for item in items])
        clusters.append(
            {
                "cluster_id": int(cid),
                "segment_ids": [int(item.get("segment_id", idx)) for idx, item in enumerate(items)],
                "orientation_rad": float(angle),
                "support_length_px": length,
                "support_length_m": length * float(config.resolution_m),
                "segments": [dict(item) for item in items],
            }
        )
    return clusters


def representative_lines_from_clusters(clusters: Sequence[Mapping[str, object]], shape: Tuple[int, int], config: StructureExtractionConfig) -> List[dict]:
    reps = []
    for cluster in clusters:
        segments = list(cluster.get("segments") or [])
        if not segments:
            continue
        points = []
        for seg in segments:
            points.append(tuple(seg["p0"]))
            points.append(tuple(seg["p1"]))
        arr = np.asarray(points, dtype=np.float32)
        angle = float(cluster.get("orientation_rad", 0.0))
        direction = np.asarray([math.sin(angle), math.cos(angle)], dtype=np.float32)
        normal = np.asarray([math.cos(angle), -math.sin(angle)], dtype=np.float32)
        proj = arr @ direction
        offset = float(np.median(arr @ normal))
        p0 = direction * float(np.min(proj)) + normal * offset
        p1 = direction * float(np.max(proj)) + normal * offset
        p0 = _clip_point_to_shape(p0, shape)
        p1 = _clip_point_to_shape(p1, shape)
        length_px = float(np.linalg.norm(p1 - p0))
        support_ratio = min(1.0, float(cluster.get("support_length_px", 0.0)) / max(length_px, 1.0))
        if support_ratio < float(config.wall_min_support_ratio):
            continue
        reps.append(
            {
                "line_id": len(reps),
                "p0": [int(round(float(p0[0]))), int(round(float(p0[1])))],
                "p1": [int(round(float(p1[0]))), int(round(float(p1[1])))],
                "orientation_rad": float(_canonical_angle(angle)),
                "support_ratio": float(support_ratio),
                "length_m": float(length_px * float(config.resolution_m)),
                "cluster_id": int(cluster.get("cluster_id", len(reps))),
            }
        )
    return reps


def rasterize_representative_lines(lines: Sequence[Mapping[str, object]], shape: Tuple[int, int], radius_cells: int = 1) -> np.ndarray:
    out = np.zeros(tuple(shape), dtype=bool)
    for line in lines:
        p0 = tuple(int(v) for v in line.get("p0", (0, 0)))
        p1 = tuple(int(v) for v in line.get("p1", (0, 0)))
        for r, c in bresenham_line(p0, p1):
            for rr in range(r - radius_cells, r + radius_cells + 1):
                for cc in range(c - radius_cells, c + radius_cells + 1):
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                        out[rr, cc] = True
    return out


def faces_from_boundary_map(free: np.ndarray, boundary_map: np.ndarray, config: StructureExtractionConfig) -> Tuple[np.ndarray, List[dict]]:
    traversible = np.asarray(free, dtype=bool) & ~np.asarray(boundary_map, dtype=bool)
    labels = np.zeros_like(traversible, dtype=np.int32)
    min_cells = max(1, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    face_rows = []
    next_label = 1
    small_cells: List[GridCell] = []
    for comp in connected_components(traversible):
        if len(comp) < min_cells and np.any(labels > 0):
            small_cells.extend(comp)
            continue
        for r, c in comp:
            labels[r, c] = next_label
        face_rows.append(_face_row(next_label, comp, config))
        next_label += 1
    if not face_rows and np.any(traversible):
        comp = [(int(r), int(c)) for r, c in zip(*np.nonzero(traversible))]
        for r, c in comp:
            labels[r, c] = 1
        face_rows.append(_face_row(1, comp, config))
    for r, c in small_cells:
        nearest = _nearest_label(labels, r, c)
        if nearest:
            labels[r, c] = nearest
    return labels, face_rows


def split_faces_by_topology(face_labels: np.ndarray, free: np.ndarray, config: StructureExtractionConfig) -> np.ndarray:
    labels = np.asarray(face_labels, dtype=np.int32).copy()
    min_cells = max(1, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    next_label = int(np.max(labels)) + 1
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        eroded = erode4(mask)
        comps = [comp for comp in connected_components(eroded) if len(comp) >= min_cells]
        if len(comps) <= 1:
            continue
        seeds = []
        for comp in comps:
            arr = np.asarray(comp, dtype=np.float32)
            seeds.append(np.mean(arr, axis=0))
        rr, cc = np.nonzero(mask)
        new_ids = [label] + [next_label + idx for idx in range(len(seeds) - 1)]
        next_label += max(0, len(seeds) - 1)
        seed_arr = np.asarray(seeds, dtype=np.float32)
        for r, c in zip(rr, cc):
            d2 = np.sum((seed_arr - np.asarray([[float(r), float(c)]], dtype=np.float32)) ** 2, axis=1)
            labels[int(r), int(c)] = int(new_ids[int(np.argmin(d2))])
    return labels


def faces_from_labels(face_labels: np.ndarray, config: StructureExtractionConfig) -> List[dict]:
    labels = np.asarray(face_labels, dtype=np.int32)
    faces: List[dict] = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        comp = [(int(r), int(c)) for r, c in zip(*np.nonzero(labels == label))]
        if comp:
            faces.append(_face_row(label, comp, config))
    return faces


def face_adjacency_edges(face_labels: np.ndarray, boundary_map: np.ndarray) -> List[dict]:
    labels = np.asarray(face_labels, dtype=np.int32)
    boundary = np.asarray(boundary_map, dtype=bool)
    edges: dict[tuple[int, int], dict] = {}
    for r in range(labels.shape[0]):
        for c in range(labels.shape[1]):
            a = int(labels[r, c])
            if a <= 0:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr >= labels.shape[0] or nc >= labels.shape[1]:
                    continue
                b = int(labels[nr, nc])
                if b <= 0 or a == b:
                    continue
                key = tuple(sorted((a, b)))
                row = edges.setdefault(key, {"face_a": key[0], "face_b": key[1], "shared_cells": 0, "edge_wall_weight": 0.0})
                row["shared_cells"] += 1
                if boundary[r, c] or boundary[nr, nc]:
                    row["edge_wall_weight"] += 1.0
    out = []
    for row in edges.values():
        shared = max(1, int(row["shared_cells"]))
        row["edge_wall_weight"] = float(row["edge_wall_weight"]) / float(shared)
        out.append(row)
    return out


def topology_split_debug(face_labels: np.ndarray, free: np.ndarray, config: StructureExtractionConfig) -> dict:
    labels = np.asarray(face_labels, dtype=np.int32)
    out = {"checked": True, "splits": []}
    min_cells = max(1, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        eroded = erode4(mask)
        comps = [comp for comp in connected_components(eroded) if len(comp) >= min_cells]
        if len(comps) > 1:
            out["splits"].append({"face_label": int(label), "eroded_components": int(len(comps)), "reason": "free_space_topology_disconnected_after_erosion"})
    return out


def connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    arr = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(arr, dtype=bool)
    comps: List[List[GridCell]] = []
    for start in zip(*np.nonzero(arr & ~visited)):
        sr, sc = int(start[0]), int(start[1])
        if visited[sr, sc]:
            continue
        queue = deque([(sr, sc)])
        visited[sr, sc] = True
        comp: List[GridCell] = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < arr.shape[0] and 0 <= nc < arr.shape[1] and arr[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def bresenham_line(p0: Tuple[int, int], p1: Tuple[int, int]) -> List[GridCell]:
    r0, c0 = int(p0[0]), int(p0[1])
    r1, c1 = int(p1[0]), int(p1[1])
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    cells = []
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


def erode4(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _neighbor_count_8(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.int32)
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    out = np.zeros_like(arr)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out += padded[1 + dr : 1 + dr + arr.shape[0], 1 + dc : 1 + dc + arr.shape[1]]
    return out


def _axis_aligned_segments(clean: np.ndarray, min_len_px: int) -> List[dict]:
    out: List[dict] = []
    sid = 0
    for r in range(clean.shape[0]):
        c = 0
        while c < clean.shape[1]:
            if not clean[r, c]:
                c += 1
                continue
            start = c
            while c < clean.shape[1] and clean[r, c]:
                c += 1
            if c - start >= min_len_px:
                out.append(_segment_dict(sid, (r, start), (r, c - 1), clean))
                sid += 1
    for c in range(clean.shape[1]):
        r = 0
        while r < clean.shape[0]:
            if not clean[r, c]:
                r += 1
                continue
            start = r
            while r < clean.shape[0] and clean[r, c]:
                r += 1
            if r - start >= min_len_px:
                out.append(_segment_dict(sid, (start, c), (r - 1, c), clean))
                sid += 1
    return out


def _segment_dict(segment_id: int, p0: GridCell, p1: GridCell, clean: np.ndarray) -> dict:
    dr = float(p1[0] - p0[0])
    dc = float(p1[1] - p0[1])
    angle = _canonical_angle(math.atan2(dr, dc))
    length = math.hypot(dr, dc)
    cells = bresenham_line(p0, p1)
    support = float(sum(1 for r, c in cells if 0 <= r < clean.shape[0] and 0 <= c < clean.shape[1] and clean[r, c])) / max(1.0, float(len(cells)))
    return {
        "segment_id": int(segment_id),
        "p0": [int(p0[0]), int(p0[1])],
        "p1": [int(p1[0]), int(p1[1])],
        "angle_rad": float(angle),
        "length_px": float(length),
        "support_ratio": float(support),
    }


def _segments_clusterable(a: Mapping[str, object], b: Mapping[str, object], dominant: Sequence[float], config: StructureExtractionConfig) -> bool:
    angle_a = _align_angle(float(a.get("angle_rad", 0.0)), dominant)
    angle_b = _align_angle(float(b.get("angle_rad", 0.0)), dominant)
    if _angular_distance_pi(angle_a, angle_b) > float(config.wall_cluster_angle_rad):
        return False
    pa = np.asarray([a["p0"], a["p1"]], dtype=np.float32)
    pb = np.asarray([b["p0"], b["p1"]], dtype=np.float32)
    direction = np.asarray([math.sin(angle_a), math.cos(angle_a)], dtype=np.float32)
    normal = np.asarray([math.cos(angle_a), -math.sin(angle_a)], dtype=np.float32)
    offset_dist = abs(float(np.mean(pa @ normal) - np.mean(pb @ normal))) * float(config.resolution_m)
    if offset_dist > float(config.wall_cluster_distance_m):
        return False
    a_proj = sorted(float(v) for v in pa @ direction)
    b_proj = sorted(float(v) for v in pb @ direction)
    gap_cells = max(0.0, max(a_proj[0], b_proj[0]) - min(a_proj[1], b_proj[1]))
    return gap_cells * float(config.resolution_m) <= float(config.wall_cluster_gap_m)


def _align_angle(angle: float, dominant: Sequence[float]) -> float:
    if not dominant:
        return _canonical_angle(angle)
    return min((float(v) for v in dominant), key=lambda item: _angular_distance_pi(angle, item))


def _mean_angle(angles: Sequence[float]) -> float:
    if not angles:
        return 0.0
    doubled = np.asarray(angles, dtype=np.float64) * 2.0
    return float(_canonical_angle(0.5 * math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled))))))


def _canonical_angle(angle) -> np.ndarray | float:
    return np.mod(angle, math.pi)


def _angular_distance_pi(a: float, b: float) -> float:
    diff = abs(float(_canonical_angle(a)) - float(_canonical_angle(b)))
    return min(diff, math.pi - diff)


def _angular_distance_pi_array(a: np.ndarray, b: float) -> np.ndarray:
    diff = np.abs(_canonical_angle(a) - float(_canonical_angle(b)))
    return np.minimum(diff, math.pi - diff)


def _clip_point_to_shape(point: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    out = np.asarray(point, dtype=np.float32).copy()
    out[0] = np.clip(out[0], 0, shape[0] - 1)
    out[1] = np.clip(out[1], 0, shape[1] - 1)
    return out


def _nearest_label(labels: np.ndarray, row: int, col: int) -> int:
    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return 0
    d2 = np.sum((coords - np.asarray([[row, col]])) ** 2, axis=1)
    nearest = coords[int(np.argmin(d2))]
    return int(labels[int(nearest[0]), int(nearest[1])])


def _face_row(label: int, comp: Sequence[GridCell], config: StructureExtractionConfig) -> dict:
    rows = np.asarray([cell[0] for cell in comp], dtype=np.float32)
    cols = np.asarray([cell[1] for cell in comp], dtype=np.float32)
    return {
        "face_id": int(label),
        "cell_count": int(len(comp)),
        "area_m2": float(len(comp) * float(config.resolution_m) ** 2),
        "centroid_rc": [float(np.mean(rows)), float(np.mean(cols))],
        "bbox_rc": [int(np.min(rows)), int(np.min(cols)), int(np.max(rows)), int(np.max(cols))],
    }
