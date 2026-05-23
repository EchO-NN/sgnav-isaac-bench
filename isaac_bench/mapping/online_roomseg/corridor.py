from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorCandidate
from .utils import component_metrics, label_components, rasterize_line


DOOR_NECK_KINDS = {"line_extension_door_neck", "extension_intersection_cut", "doorway_virtual_cut", "corridor_room_neck_cut"}


@dataclass
class CorridorConfig:
    enabled: bool = True
    min_length_m: float = 1.5
    max_width_m: float = 1.8
    max_median_width_m: float = 1.8
    max_p90_width_m: float = 2.25
    min_aspect_ratio: float = 2.5
    max_width_std_m: float = 0.45
    min_parallel_wall_support: float = 0.45
    min_skeleton_density_inv_m: float = 0.45
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


@dataclass
class CorridorMergeConfig:
    enabled: bool = True
    parallel_door_angle_deg: float = 12.0
    parallel_door_pair_max_distance_m: float = 2.20
    parallel_door_min_overlap_m: float = 0.25
    parallel_edge_length_tolerance_ratio: float = 0.15
    parallel_edge_coverage_min_ratio: float = 0.95
    door_neck_edge_touch_search_cells: int = 2
    shared_mask_edge_length_match_enabled: bool = False
    isolated_chunk_max_length_m: float = 2.20
    isolated_chunk_max_area_m2: float = 4.00
    width_similarity_tol_m: float = 0.35
    axis_angle_similarity_deg: float = 15.0
    merge_three_if_both_neighbors_similar: bool = True
    reject_both_doors_if_middle_corridor_like: bool = True
    min_region_area_m2: float = 0.40
    post_corridor_small_region_merge_enabled: bool = True
    post_corridor_small_region_max_area_m2: float = 2.50
    post_corridor_small_region_max_unknown_ratio: float = 0.20
    post_corridor_small_region_require_single_neighbor: bool = True
    post_corridor_small_region_single_neighbor_search_m: float = 0.50

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "CorridorMergeConfig":
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
        widths = 2.0 * distance[comp] * float(resolution_m)
        p90_width = float(np.percentile(widths, 90)) if widths.size else 0.0
        width_std = float(np.std(widths)) if widths.size else 0.0
        corridor_like = bool(
            metrics["aspect_ratio"] >= float(cfg.min_aspect_ratio)
            and metrics["length_m"] >= float(cfg.min_length_m)
            and metrics["median_width_m"] <= min(float(cfg.max_width_m), float(cfg.max_median_width_m))
            and p90_width <= float(cfg.max_p90_width_m)
            and width_std <= max(1e-6, float(cfg.max_width_std_m)) * 4.0
        )
        if corridor_like:
            corridor_map |= comp
        components.append({"component": int(idx), **metrics, "p90_width_m": float(p90_width), "width_std_m": float(width_std), "corridor_like": corridor_like})
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
    corridor_debug: Mapping[str, object] | None = None,
    wall_candidate_clean: np.ndarray | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    corridor_cfg = corridor_config if isinstance(corridor_config, CorridorConfig) else CorridorConfig.from_mapping(corridor_config)
    neck_cfg = neck_config if isinstance(neck_config, CorridorRoomNeckCutConfig) else CorridorRoomNeckCutConfig.from_mapping(neck_config)
    corridor_debug = (
        dict(corridor_debug)
        if corridor_debug is not None
        else build_corridor_debug(free_clean, resolution_m=resolution_m, config=corridor_cfg)
    )
    if not bool(corridor_cfg.enabled) or not bool(neck_cfg.enabled):
        return [], {**corridor_debug, "corridor_room_neck_candidate_count": 0}
    free = np.asarray(free_clean, dtype=bool)
    wall = None if wall_candidate_clean is None else np.asarray(wall_candidate_clean, dtype=bool)
    max_cells = max(2, int(round(float(neck_cfg.max_neck_width_m) / max(float(resolution_m), 1e-9))))
    candidates: list[SeparatorCandidate] = []
    cid = int(start_id)
    seen: set[tuple[str, int, int, int]] = set()
    rejected_by_reason: dict[str, int] = {}

    min_room_side_area_m2 = float(neck_cfg.min_room_side_area_m2)

    def add_candidate(source: str, p0: np.ndarray, p1: np.ndarray, width: int, confidence: float) -> None:
        nonlocal cid
        support_debug = _wall_endpoint_support_debug(
            wall,
            p0,
            p1,
            radius_cells=max(1, int(round(0.20 / max(float(resolution_m), 1e-9)))),
        )
        if wall is not None and not bool(support_debug.get("wall_endpoint_support_both", False)):
            reason = "reject_no_wall_endpoint_support"
            rejected_by_reason[reason] = int(rejected_by_reason.get(reason, 0)) + 1
            return
        candidate = _neck_candidate(cid, source, p0, p1, width, resolution_m, confidence=confidence)
        candidate.debug.update(support_debug)
        candidate.debug["corridor_room_neck_min_room_side_area_m2"] = float(min_room_side_area_m2)
        candidates.append(candidate)
        cid += 1

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
                    add_candidate("row_overlap", np.asarray([row + 1, c0]), np.asarray([row + 1, c1 - 1]), width, confidence)
    for col in range(free.shape[1] - 1):
        overlap = free[:, col] & free[:, col + 1]
        for r0, r1 in _runs(overlap):
            width = r1 - r0
            if 2 <= width <= max_cells and (_row_run_width(free[:, col], r0) > width * 2 or _row_run_width(free[:, col + 1], r0) > width * 2):
                key = ("col_overlap", int(col), int(r0), int(r1))
                if key not in seen:
                    seen.add(key)
                    confidence = 0.8 + 0.19 * min(1.0, float(width) / float(max_cells))
                    add_candidate("col_overlap", np.asarray([r0, col + 1]), np.asarray([r1 - 1, col + 1]), width, confidence)
    return candidates[:2048], {
        **corridor_debug,
        "corridor_room_neck_candidate_count": int(min(len(candidates), 2048)),
        "corridor_room_neck_rejected_by_reason": rejected_by_reason,
    }


def merge_false_parallel_door_corridor_regions(
    raw_labels: np.ndarray,
    *,
    accepted_candidates: Sequence[SeparatorCandidate],
    free_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    filtered_lines: Sequence[object],
    resolution_m: float,
    config: CorridorMergeConfig | Mapping[str, object] | None = None,
    unknown_clean: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    cfg = config if isinstance(config, CorridorMergeConfig) else CorridorMergeConfig.from_mapping(config)
    labels = np.asarray(raw_labels, dtype=np.int32)
    free = np.asarray(free_clean, dtype=bool)
    _unknown = np.zeros_like(free, dtype=bool) if unknown_clean is None else np.asarray(unknown_clean, dtype=bool)
    if _unknown.shape != free.shape:
        _unknown = np.zeros_like(free, dtype=bool)
    if not bool(cfg.enabled) or labels.size == 0:
        return labels.copy(), {"enabled": bool(cfg.enabled), "merge_events": [], "final_region_count": int(len([v for v in np.unique(labels) if int(v) > 0]))}
    distance = ndimage.distance_transform_edt(free)
    infos = _label_region_infos(labels, free, distance, float(resolution_m), wall_candidate_clean, filtered_lines, cfg)
    merge_events: list[dict] = []
    pairs = detect_parallel_door_pairs(
        accepted_candidates,
        resolution_m=float(resolution_m),
        config=cfg,
    )
    edges = _door_neck_region_edges(labels, accepted_candidates, free.shape, cfg)
    wall_total_mask = np.asarray(wall_candidate_clean, dtype=bool)
    if wall_total_mask.shape != free.shape:
        wall_total_mask = np.zeros_like(free, dtype=bool)
    filled_edges = [
        _filled_edge_record(
            edge,
            wall_total_mask,
            float(resolution_m),
            labels=labels,
            label_touch_search_cells=int(cfg.door_neck_edge_touch_search_cells),
        )
        for edge in edges
    ]
    corridor_like_raw_labels, corridor_like_edge_pairs = _corridor_like_labels_from_filled_edges(
        filled_edges,
        config=cfg,
        resolution_m=float(resolution_m),
    )
    for label, info in infos.items():
        info["corridor_like"] = bool(int(label) in corridor_like_raw_labels)
        info["corridor_like_source"] = "filled_parallel_same_length_edges"

    stage1_parent = {int(label): int(label) for label in infos}
    for edge in filled_edges:
        left = int(edge["label_a"])
        right = int(edge["label_b"])
        if left not in corridor_like_raw_labels or right not in corridor_like_raw_labels:
            continue
        _union(stage1_parent, left, right)
        _mark_edge_candidate_rejected(edge, "adjacent_corridor_like_merge")
        merge_events.append(
            {
                "reason": "adjacent_corridor_like_merge",
                "merged_regions": [left, right],
                "shared_edge_candidate": int(edge["candidate_id"]),
                "shared_edge_total_length_m": float(edge["total_length_m"]),
                "shared_edge_total_p0_rc": edge.get("total_p0_rc"),
                "shared_edge_total_p1_rc": edge.get("total_p1_rc"),
                "length_tolerance_ratio": float(cfg.parallel_edge_length_tolerance_ratio),
            }
        )

    changed = True
    while changed:
        changed = False
        for label in sorted(corridor_like_raw_labels):
            root = _find(stage1_parent, int(label))
            for neighbor in _neighbor_labels(labels, int(label), iterations=2):
                if int(neighbor) not in corridor_like_raw_labels:
                    continue
                other = _find(stage1_parent, int(neighbor))
                if root == other:
                    continue
                _union(stage1_parent, root, other)
                merge_events.append(
                    {
                        "reason": "adjacent_corridor_like_touch_merge",
                        "merged_regions": [int(root), int(other)],
                        "source_label": int(label),
                        "neighbor_label": int(neighbor),
                    }
                )
                changed = True

    strict_out, strict_label_remap = _remap_labels_with_parent(labels, stage1_parent, free)
    strict_infos = _label_region_infos(strict_out, free, distance, float(resolution_m), wall_candidate_clean, filtered_lines, cfg)
    corridor_like_stage_labels = {
        int(strict_label_remap[int(label)])
        for label in corridor_like_raw_labels
        if int(label) in strict_label_remap
    }
    for label, info in strict_infos.items():
        info["corridor_like"] = bool(int(label) in corridor_like_stage_labels)
        info["corridor_like_source"] = "filled_parallel_same_length_edges"
    stage_edges = _remap_filled_edges(filled_edges, strict_label_remap)
    stage2_parent = {int(label): int(label) for label in strict_infos}
    corridor_roots = {int(label) for label in corridor_like_stage_labels}
    rejected_merge_events: list[dict] = []
    changed = True
    while changed:
        changed = False
        for shared_edge in stage_edges:
            left = _find(stage2_parent, int(shared_edge["label_a"]))
            right = _find(stage2_parent, int(shared_edge["label_b"]))
            if left == right:
                continue
            left_corridor = left in corridor_roots
            right_corridor = right in corridor_roots
            if left_corridor and right_corridor:
                _union(stage2_parent, left, right)
                corridor_roots.add(_find(stage2_parent, left))
                changed = True
                continue
            if left_corridor == right_corridor:
                continue
            corridor_label = left if left_corridor else right
            adjacent_label = right if left_corridor else left
            matching_edge = _best_parallel_length_matched_adjacent_edge(
                shared_edge,
                corridor_label,
                adjacent_label,
                stage_edges,
                strict_out,
                stage2_parent,
                config=cfg,
                resolution_m=float(resolution_m),
            )
            if matching_edge is None:
                rejected_merge_events.append(
                    {
                        "reason": "reject_no_parallel_same_length_adjacent_edge",
                        "shared_edge_candidate": int(shared_edge["candidate_id"]),
                        "corridor_region": int(corridor_label),
                        "adjacent_region": int(adjacent_label),
                        "shared_edge_total_length_m": float(shared_edge["total_length_m"]),
                        "shared_edge_total_p0_rc": shared_edge.get("total_p0_rc"),
                        "shared_edge_total_p1_rc": shared_edge.get("total_p1_rc"),
                        "shared_edge_corridor_side": _edge_debug(_edge_view_for_label(shared_edge, int(corridor_label))),
                        "shared_edge_adjacent_side": _edge_debug(_edge_view_for_label(shared_edge, int(adjacent_label))),
                    }
                )
                continue
            _union(stage2_parent, corridor_label, adjacent_label)
            corridor_roots.add(_find(stage2_parent, corridor_label))
            _mark_edge_candidate_rejected(shared_edge, "corridor_parallel_same_length_neighbor_merge")
            merge_events.append(
                {
                    "reason": "corridor_parallel_same_length_neighbor_merge",
                    "merged_regions": [int(corridor_label), int(adjacent_label)],
                    "shared_edge_candidate": int(shared_edge["candidate_id"]),
                    "adjacent_matching_edge_candidate": int(matching_edge["candidate_id"]),
                    "adjacent_matching_edge_kind": str(matching_edge.get("kind", "")),
                    "shared_edge_total_length_m": float(shared_edge["total_length_m"]),
                    "adjacent_matching_edge_total_length_m": float(matching_edge["total_length_m"]),
                    "shared_edge_total_p0_rc": shared_edge.get("total_p0_rc"),
                    "shared_edge_total_p1_rc": shared_edge.get("total_p1_rc"),
                    "adjacent_matching_edge_total_p0_rc": matching_edge.get("total_p0_rc"),
                    "adjacent_matching_edge_total_p1_rc": matching_edge.get("total_p1_rc"),
                    "length_tolerance_ratio": float(cfg.parallel_edge_length_tolerance_ratio),
                }
            )
            changed = True

    sliver_merge_events: list[dict] = []
    out, _final_label_remap = _remap_labels_with_parent(strict_out, stage2_parent, free)
    final_infos = _label_region_infos(out, free, distance, float(resolution_m), wall_candidate_clean, filtered_lines, cfg)
    final_corridor_labels = {
        int(_final_label_remap[int(label)])
        for label in corridor_like_stage_labels
        if int(label) in _final_label_remap
    }
    for label, info in final_infos.items():
        info["corridor_like"] = bool(int(label) in final_corridor_labels)
        info["corridor_like_source"] = "filled_parallel_same_length_edges"
    debug = {
        "enabled": True,
        "parallel_door_pairs": pairs,
        "strict_parallel_door_neck_edges": [_edge_debug(edge) for edge in edges],
        "filled_door_neck_edges": [_edge_debug(edge) for edge in filled_edges],
        "corridor_like_edge_pairs": corridor_like_edge_pairs,
        "corridor_like_labels": sorted(int(v) for v in corridor_like_raw_labels),
        "corridor_like_labels_after_adjacent_merge": sorted(int(v) for v in corridor_like_stage_labels),
        "corridor_like_labels_after_merge": sorted(int(v) for v in final_corridor_labels),
        "merge_events": merge_events,
        "rejected_merge_events": rejected_merge_events[:512],
        "sliver_merge_events": sliver_merge_events,
        "protected_post_corridor_labels": [],
        "post_corridor_small_region_merge_max_area_m2": float(cfg.post_corridor_small_region_max_area_m2),
        "post_corridor_small_region_merge_max_unknown_ratio": float(cfg.post_corridor_small_region_max_unknown_ratio),
        "post_corridor_small_region_require_single_neighbor": bool(cfg.post_corridor_small_region_require_single_neighbor),
        "post_corridor_small_region_single_neighbor_search_m": float(cfg.post_corridor_small_region_single_neighbor_search_m),
        "region_infos_before_merge": list(infos.values()),
        "region_infos_after_strict_corridor_merge": list(strict_infos.values()),
        "region_infos_after_merge": list(final_infos.values()),
        "final_region_count": int(len([v for v in np.unique(out) if int(v) > 0])),
    }
    return out.astype(np.int32), debug


def detect_parallel_door_pairs(
    candidates: Sequence[SeparatorCandidate],
    *,
    resolution_m: float,
    config: CorridorMergeConfig | Mapping[str, object] | None = None,
) -> list[dict]:
    cfg = config if isinstance(config, CorridorMergeConfig) else CorridorMergeConfig.from_mapping(config)
    doors = [item for item in candidates if str(item.kind) in DOOR_NECK_KINDS]
    out: list[dict] = []
    for idx, a in enumerate(doors):
        for b in doors[idx + 1 :]:
            angle = _angle_diff(float(a.theta), float(b.theta))
            center_delta = _candidate_center(b) - _candidate_center(a)
            distance_m = float(np.linalg.norm(center_delta) * float(resolution_m))
            tangent_overlap_m = _candidate_tangent_overlap_m(a, b, float(resolution_m))
            if (
                angle <= np.deg2rad(float(cfg.parallel_door_angle_deg))
                and distance_m <= float(cfg.parallel_door_pair_max_distance_m)
                and tangent_overlap_m >= float(cfg.parallel_door_min_overlap_m)
            ):
                out.append(
                    {
                        "a": int(a.candidate_id),
                        "b": int(b.candidate_id),
                        "angle_diff_deg": float(np.rad2deg(angle)),
                        "center_distance_m": float(distance_m),
                        "tangent_overlap_m": float(tangent_overlap_m),
                    }
                )
    return out


def _door_neck_region_edges(
    labels: np.ndarray,
    candidates: Sequence[SeparatorCandidate],
    shape: tuple[int, int],
    config: CorridorMergeConfig,
) -> list[dict]:
    out: list[dict] = []
    for candidate in candidates:
        if str(candidate.kind) not in DOOR_NECK_KINDS:
            continue
        touch_search = max(1, int(config.door_neck_edge_touch_search_cells))
        touching = _labels_touching_candidate(labels, candidate, shape, iterations=touch_search)
        if len(touching) < 2:
            continue
        for idx, label_a in enumerate(touching):
            for label_b in touching[idx + 1 :]:
                coverage = _candidate_label_pair_coverage(
                    labels,
                    candidate,
                    int(label_a),
                    int(label_b),
                    shape,
                    iterations=touch_search,
                )
                if coverage + 1e-9 < float(config.parallel_edge_coverage_min_ratio):
                    continue
                out.append(
                    {
                        "candidate": candidate,
                        "candidate_id": int(candidate.candidate_id),
                        "kind": str(candidate.kind),
                        "label_a": int(label_a),
                        "label_b": int(label_b),
                        "theta": float(candidate.theta),
                        "length_m": float(candidate.length_m),
                        "coverage_ratio": float(coverage),
                    }
                )
    return out


def _candidate_label_pair_coverage(
    labels: np.ndarray,
    candidate: SeparatorCandidate,
    label_a: int,
    label_b: int,
    shape: tuple[int, int],
    *,
    iterations: int = 1,
) -> float:
    line = np.asarray(candidate.mask(shape), dtype=bool)
    denom = int(np.count_nonzero(line))
    if denom <= 0:
        return 0.0
    labels_arr = np.asarray(labels, dtype=np.int32)
    search = max(1, int(iterations))
    near_a = ndimage.binary_dilation(labels_arr == int(label_a), iterations=search)
    near_b = ndimage.binary_dilation(labels_arr == int(label_b), iterations=search)
    covered = line & near_a & near_b
    return float(np.count_nonzero(covered)) / float(denom)


def _best_other_parallel_edge(
    shared_edge: Mapping[str, object],
    label: int,
    *,
    forbidden_label: int,
    edges: Sequence[Mapping[str, object]],
    config: CorridorMergeConfig,
    resolution_m: float,
) -> Mapping[str, object] | None:
    shared_id = int(shared_edge.get("candidate_id", -1))
    label_i = int(label)
    forbidden_i = int(forbidden_label)
    options: list[Mapping[str, object]] = []
    for edge in edges:
        if int(edge.get("candidate_id", -2)) == shared_id:
            continue
        edge_labels = {int(edge.get("label_a", 0)), int(edge.get("label_b", 0))}
        if label_i not in edge_labels:
            continue
        if forbidden_i in edge_labels:
            continue
        if not _edge_pair_geometry_ok(shared_edge, edge, config, float(resolution_m)):
            continue
        options.append(edge)
    if not options:
        return None
    shared_length = float(shared_edge.get("length_m", 0.0))
    return min(options, key=lambda edge: abs(float(edge.get("length_m", 0.0)) - shared_length))


def _edges_parallel(a: Mapping[str, object], b: Mapping[str, object], config: CorridorMergeConfig) -> bool:
    return bool(_angle_diff(float(a.get("theta", 0.0)), float(b.get("theta", 0.0))) <= np.deg2rad(float(config.parallel_door_angle_deg)))


def _edge_pair_geometry_ok(
    a: Mapping[str, object],
    b: Mapping[str, object],
    config: CorridorMergeConfig,
    resolution_m: float,
) -> bool:
    if not _edges_parallel(a, b, config):
        return False
    ca = a.get("candidate")
    cb = b.get("candidate")
    if not isinstance(ca, SeparatorCandidate) or not isinstance(cb, SeparatorCandidate):
        return False
    center_distance_m = float(np.linalg.norm(_candidate_center(cb) - _candidate_center(ca)) * float(resolution_m))
    tangent_overlap_m = _candidate_tangent_overlap_m(ca, cb, float(resolution_m))
    return bool(
        center_distance_m <= float(config.parallel_door_pair_max_distance_m)
        and tangent_overlap_m >= float(config.parallel_door_min_overlap_m)
    )


def _lengths_close(a_m: float, b_m: float, config: CorridorMergeConfig) -> bool:
    denom = max(abs(float(a_m)), abs(float(b_m)), 1e-6)
    return bool(abs(float(a_m) - float(b_m)) / denom <= float(config.parallel_edge_length_tolerance_ratio))


def _edge_debug(edge: Mapping[str, object]) -> dict:
    out = {
        "candidate_id": int(edge.get("candidate_id", 0)),
        "kind": str(edge.get("kind", "")),
        "labels": [int(edge.get("label_a", 0)), int(edge.get("label_b", 0))],
        "theta": float(edge.get("theta", 0.0)),
        "length_m": float(edge.get("length_m", 0.0)),
        "coverage_ratio": float(edge.get("coverage_ratio", 0.0)),
        "p0_rc": _edge_point(edge, "p0"),
        "p1_rc": _edge_point(edge, "p1"),
    }
    if "total_length_m" in edge:
        out.update(
            {
                "total_length_m": float(edge.get("total_length_m", 0.0)),
                "total_p0_rc": edge.get("total_p0_rc"),
                "total_p1_rc": edge.get("total_p1_rc"),
                "length_source": str(edge.get("length_source", "")),
            }
        )
    return out


def _filled_edge_record(
    edge: Mapping[str, object],
    wall_mask: np.ndarray,
    resolution_m: float,
    *,
    labels: np.ndarray | None = None,
    label_touch_search_cells: int = 2,
) -> dict:
    total = _edge_total_length_after_noise_fill(edge, wall_mask, float(resolution_m))
    p0 = total.get("p0_rc") or _edge_point(edge, "p0")
    p1 = total.get("p1_rc") or _edge_point(edge, "p1")
    theta = _line_theta_from_points(p0, p1)
    label_spans: dict[int, dict] = {}
    if labels is not None:
        for label in (int(edge.get("label_a", 0)), int(edge.get("label_b", 0))):
            span = _edge_label_side_span(
                labels,
                label,
                p0,
                p1,
                resolution_m=float(resolution_m),
                iterations=max(1, int(label_touch_search_cells)),
            )
            if span is not None:
                label_spans[int(label)] = span
    out = dict(edge)
    out.update(
        {
            "theta": float(theta),
            "candidate_theta": float(edge.get("theta", 0.0)),
            "total_length_m": float(total.get("length_m", edge.get("length_m", 0.0))),
            "total_p0_rc": p0,
            "total_p1_rc": p1,
            "length_source": str(total.get("source", "candidate_fallback")),
            "label_spans": label_spans,
        }
    )
    return out


def _corridor_like_labels_from_filled_edges(
    edges: Sequence[Mapping[str, object]],
    *,
    config: CorridorMergeConfig,
    resolution_m: float,
) -> tuple[set[int], list[dict]]:
    by_label: dict[int, list[Mapping[str, object]]] = {}
    for edge in edges:
        by_label.setdefault(int(edge.get("label_a", 0)), []).append(edge)
        by_label.setdefault(int(edge.get("label_b", 0)), []).append(edge)
    labels: set[int] = set()
    pairs: list[dict] = []
    for label, incident in by_label.items():
        if label <= 0:
            continue
        for idx, first in enumerate(incident):
            for second in incident[idx + 1 :]:
                if int(first.get("candidate_id", -1)) == int(second.get("candidate_id", -2)):
                    continue
                first_view = _edge_view_for_label(first, int(label))
                second_view = _edge_view_for_label(second, int(label))
                if not _filled_edges_parallel_and_length_close(first_view, second_view, config=config, resolution_m=float(resolution_m)):
                    continue
                labels.add(int(label))
                pairs.append(
                    {
                        "label": int(label),
                        "edge_a_candidate": int(first.get("candidate_id", 0)),
                        "edge_b_candidate": int(second.get("candidate_id", 0)),
                        "edge_a_labels": [int(first.get("label_a", 0)), int(first.get("label_b", 0))],
                        "edge_b_labels": [int(second.get("label_a", 0)), int(second.get("label_b", 0))],
                        "edge_a_total_length_m": float(first_view.get("total_length_m", 0.0)),
                        "edge_b_total_length_m": float(second_view.get("total_length_m", 0.0)),
                        "edge_a_total_p0_rc": first_view.get("total_p0_rc"),
                        "edge_a_total_p1_rc": first_view.get("total_p1_rc"),
                        "edge_b_total_p0_rc": second_view.get("total_p0_rc"),
                        "edge_b_total_p1_rc": second_view.get("total_p1_rc"),
                        "length_tolerance_ratio": float(config.parallel_edge_length_tolerance_ratio),
                    }
                )
                break
            if int(label) in labels:
                break
    return labels, pairs


def _remap_filled_edges(edges: Sequence[Mapping[str, object]], label_remap: Mapping[int, int]) -> list[dict]:
    out: list[dict] = []
    for edge in edges:
        left_raw = int(edge.get("label_a", 0))
        right_raw = int(edge.get("label_b", 0))
        if left_raw not in label_remap or right_raw not in label_remap:
            continue
        left = int(label_remap[left_raw])
        right = int(label_remap[right_raw])
        if left == right:
            continue
        item = dict(edge)
        item["label_a"] = left
        item["label_b"] = right
        item["raw_label_a"] = left_raw
        item["raw_label_b"] = right_raw
        raw_spans = dict(edge.get("label_spans") or {})
        remapped_spans = {}
        if left_raw in raw_spans:
            remapped_spans[left] = raw_spans[left_raw]
        if right_raw in raw_spans:
            remapped_spans[right] = raw_spans[right_raw]
        item["label_spans"] = remapped_spans
        out.append(item)
    return out


def _best_parallel_length_matched_adjacent_edge(
    shared_edge: Mapping[str, object],
    corridor_label: int,
    adjacent_label: int,
    edges: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    parent: dict[int, int],
    *,
    config: CorridorMergeConfig,
    resolution_m: float,
) -> Mapping[str, object] | None:
    adjacent_root = _find(parent, int(adjacent_label))
    corridor_root = _find(parent, int(corridor_label))
    options: list[Mapping[str, object]] = []
    shared_id = int(shared_edge.get("candidate_id", -1))
    shared_left = _find(parent, int(shared_edge.get("label_a", 0)))
    shared_right = _find(parent, int(shared_edge.get("label_b", 0)))
    shared_side = _edge_side_label_for_root(shared_edge, corridor_root, parent)
    shared_view = _edge_view_for_label(shared_edge, shared_side)
    for edge in edges:
        if int(edge.get("candidate_id", -2)) == shared_id:
            continue
        left = _find(parent, int(edge.get("label_a", 0)))
        right = _find(parent, int(edge.get("label_b", 0)))
        if left == right:
            continue
        if adjacent_root not in {left, right}:
            continue
        other = right if left == adjacent_root else left
        if other in {shared_left, shared_right}:
            continue
        edge_side = _edge_side_label_for_root(edge, adjacent_root, parent)
        edge_view = _edge_view_for_label(edge, edge_side)
        if _filled_edges_parallel_and_length_close(shared_view, edge_view, config=config, resolution_m=float(resolution_m)):
            options.append(edge)
    if not options:
        if not bool(config.shared_mask_edge_length_match_enabled):
            return None
        return _best_parallel_length_matched_mask_boundary_edge(
            shared_edge,
            corridor_root,
            adjacent_root,
            labels,
            parent,
            config=config,
            resolution_m=float(resolution_m),
        )
    shared_length = float(shared_view.get("total_length_m", shared_view.get("length_m", 0.0)))
    return min(
        options,
        key=lambda item: abs(
            float(
                _edge_view_for_label(
                    item,
                    _edge_side_label_for_root(item, adjacent_root, parent),
                ).get("total_length_m", item.get("length_m", 0.0))
            )
            - shared_length
        ),
    )


def _best_parallel_length_matched_mask_boundary_edge(
    shared_edge: Mapping[str, object],
    corridor_root: int,
    adjacent_root: int,
    labels: np.ndarray,
    parent: dict[int, int],
    *,
    config: CorridorMergeConfig,
    resolution_m: float,
) -> Mapping[str, object] | None:
    corridor_side = _edge_side_label_for_root(shared_edge, int(corridor_root), parent)
    shared_view = _edge_view_for_label(shared_edge, corridor_side)
    if not str(shared_view.get("length_source", "")).startswith("label_side_"):
        return None
    mask = _component_mask_for_root(labels, int(adjacent_root), parent)
    if not np.any(mask):
        return None
    options = [
        edge
        for edge in _mask_boundary_edges(
            mask,
            resolution_m=float(resolution_m),
            reference_edge=shared_view,
            exclude_near_reference_cells=max(1, int(config.door_neck_edge_touch_search_cells)),
            min_length_m=max(float(config.parallel_door_min_overlap_m), float(resolution_m)),
        )
        if _filled_edges_parallel_and_length_close(shared_view, edge, config=config, resolution_m=float(resolution_m))
    ]
    if not options:
        return None
    shared_length = float(shared_view.get("total_length_m", shared_view.get("length_m", 0.0)))
    best = min(options, key=lambda item: abs(float(item.get("total_length_m", item.get("length_m", 0.0))) - shared_length))
    out = dict(best)
    out["candidate_id"] = -int(abs(int(shared_edge.get("candidate_id", 0))) or 1)
    out["kind"] = "region_mask_parallel_edge"
    out["source_candidate_id"] = int(shared_edge.get("candidate_id", 0))
    out["adjacent_region"] = int(adjacent_root)
    out["shared_edge_label_side_length_m"] = float(shared_length)
    out["shared_edge_label_side_p0_rc"] = shared_view.get("total_p0_rc")
    out["shared_edge_label_side_p1_rc"] = shared_view.get("total_p1_rc")
    return out


def _edge_side_label_for_root(edge: Mapping[str, object], root: int, parent: dict[int, int]) -> int:
    root_i = int(root)
    for label in (int(edge.get("label_a", 0)), int(edge.get("label_b", 0))):
        if label > 0 and _find(parent, label) == root_i:
            return int(label)
    return root_i


def _component_mask_for_root(labels: np.ndarray, root: int, parent: dict[int, int]) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=bool)
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        if _find(parent, int(label)) == int(root):
            out |= arr == int(label)
    return out


def _mask_boundary_edges(
    mask: np.ndarray,
    *,
    resolution_m: float,
    reference_edge: Mapping[str, object],
    exclude_near_reference_cells: int,
    min_length_m: float,
) -> list[dict]:
    comp = np.asarray(mask, dtype=bool)
    if comp.ndim != 2 or not np.any(comp):
        return []
    horizontal_ref = _edge_is_horizontal(reference_edge)
    padded = np.pad(comp, 1, mode="constant", constant_values=False)
    out: list[dict] = []
    if horizontal_ref:
        for side, boundary in (
            ("up", comp & ~padded[:-2, 1:-1]),
            ("down", comp & ~padded[2:, 1:-1]),
        ):
            for row in np.where(np.any(boundary, axis=1))[0]:
                for start, end_exclusive in _runs(boundary[int(row), :]):
                    end = int(end_exclusive) - 1
                    edge = _boundary_edge_record(side, [int(row), int(start)], [int(row), int(end)], float(resolution_m))
                    if float(edge["total_length_m"]) + 1e-9 < float(min_length_m):
                        continue
                    if _edge_near_and_overlapping(edge, reference_edge, int(exclude_near_reference_cells), float(resolution_m)):
                        continue
                    out.append(edge)
        return out
    for side, boundary in (
        ("left", comp & ~padded[1:-1, :-2]),
        ("right", comp & ~padded[1:-1, 2:]),
    ):
        for col in np.where(np.any(boundary, axis=0))[0]:
            for start, end_exclusive in _runs(boundary[:, int(col)]):
                end = int(end_exclusive) - 1
                edge = _boundary_edge_record(side, [int(start), int(col)], [int(end), int(col)], float(resolution_m))
                if float(edge["total_length_m"]) + 1e-9 < float(min_length_m):
                    continue
                if _edge_near_and_overlapping(edge, reference_edge, int(exclude_near_reference_cells), float(resolution_m)):
                    continue
                out.append(edge)
    return out


def _boundary_edge_record(side: str, p0: list[int], p1: list[int], resolution_m: float) -> dict:
    return {
        "candidate_id": 0,
        "kind": "region_mask_boundary_edge",
        "boundary_side": str(side),
        "theta": _line_theta_from_points(p0, p1),
        "length_m": float((abs(int(p1[0]) - int(p0[0])) + abs(int(p1[1]) - int(p0[1])) + 1) * float(resolution_m)),
        "total_length_m": float((abs(int(p1[0]) - int(p0[0])) + abs(int(p1[1]) - int(p0[1])) + 1) * float(resolution_m)),
        "total_p0_rc": [int(p0[0]), int(p0[1])],
        "total_p1_rc": [int(p1[0]), int(p1[1])],
        "length_source": "region_mask_boundary",
    }


def _edge_is_horizontal(edge: Mapping[str, object]) -> bool:
    p0 = np.asarray(edge.get("total_p0_rc") or _edge_point(edge, "p0") or [0, 0], dtype=np.float32)
    p1 = np.asarray(edge.get("total_p1_rc") or _edge_point(edge, "p1") or [0, 0], dtype=np.float32)
    return bool(abs(float(p1[1] - p0[1])) >= abs(float(p1[0] - p0[0])))


def _edge_near_and_overlapping(a: Mapping[str, object], b: Mapping[str, object], max_normal_distance_cells: int, resolution_m: float) -> bool:
    if not _edge_is_horizontal(a) == _edge_is_horizontal(b):
        return False
    ap0 = np.asarray(a.get("total_p0_rc") or [0, 0], dtype=np.float32)
    ap1 = np.asarray(a.get("total_p1_rc") or [0, 0], dtype=np.float32)
    bp0 = np.asarray(b.get("total_p0_rc") or _edge_point(b, "p0") or [0, 0], dtype=np.float32)
    bp1 = np.asarray(b.get("total_p1_rc") or _edge_point(b, "p1") or [0, 0], dtype=np.float32)
    if _edge_is_horizontal(a):
        normal = abs(float(0.5 * (ap0[0] + ap1[0]) - 0.5 * (bp0[0] + bp1[0])))
    else:
        normal = abs(float(0.5 * (ap0[1] + ap1[1]) - 0.5 * (bp0[1] + bp1[1])))
    return bool(normal <= float(max(1, int(max_normal_distance_cells))) and _edge_tangent_overlap_m(a, b, float(resolution_m)) > 0.0)


def _filled_edges_parallel_and_length_close(
    a: Mapping[str, object],
    b: Mapping[str, object],
    *,
    config: CorridorMergeConfig,
    resolution_m: float,
) -> bool:
    if not _edges_parallel(a, b, config):
        return False
    if not _lengths_close(float(a.get("total_length_m", a.get("length_m", 0.0))), float(b.get("total_length_m", b.get("length_m", 0.0))), config):
        return False
    return bool(_edge_tangent_overlap_m(a, b, float(resolution_m)) >= float(config.parallel_door_min_overlap_m))


def _edge_view_for_label(edge: Mapping[str, object], label: int) -> dict:
    out = dict(edge)
    spans = edge.get("label_spans")
    span = None
    if isinstance(spans, Mapping):
        span = spans.get(int(label))
        if span is None:
            span = spans.get(str(int(label)))
    if isinstance(span, Mapping):
        out["total_length_m"] = float(span.get("length_m", out.get("total_length_m", out.get("length_m", 0.0))))
        out["total_p0_rc"] = span.get("p0_rc", out.get("total_p0_rc"))
        out["total_p1_rc"] = span.get("p1_rc", out.get("total_p1_rc"))
        out["theta"] = _line_theta_from_points(out.get("total_p0_rc"), out.get("total_p1_rc"))
        out["length_source"] = "label_side_" + str(out.get("length_source", "candidate_fallback"))
    return out


def _edge_label_side_span(
    labels: np.ndarray,
    label: int,
    p0: object,
    p1: object,
    *,
    resolution_m: float,
    iterations: int,
) -> dict | None:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.ndim != 2 or label <= 0:
        return None
    mask = arr == int(label)
    if not np.any(mask):
        return None
    near = ndimage.binary_dilation(mask, iterations=max(1, int(iterations)))
    a = np.asarray(p0, dtype=np.float32).reshape(-1)
    b = np.asarray(p1, dtype=np.float32).reshape(-1)
    if a.size < 2 or b.size < 2:
        return None
    horizontal = abs(float(b[1] - a[1])) >= abs(float(b[0] - a[0]))
    if horizontal:
        line = int(round(float(0.5 * (a[0] + b[0]))))
        span0 = int(round(float(min(a[1], b[1]))))
        span1 = int(round(float(max(a[1], b[1]))))
        lo = max(0, line - max(1, int(iterations)))
        hi = min(arr.shape[0], line + max(1, int(iterations)) + 1)
        profile = np.any(near[lo:hi, :], axis=0)
        run = _overlapping_profile_run(profile, span0, span1)
        if run is None:
            return None
        start, end = run
        start = max(0, int(start))
        end = min(arr.shape[1] - 1, int(end))
        if end < start:
            return None
        return {
            "length_m": float((int(end) - int(start) + 1) * float(resolution_m)),
            "p0_rc": [int(line), int(start)],
            "p1_rc": [int(line), int(end)],
        }
    line = int(round(float(0.5 * (a[1] + b[1]))))
    span0 = int(round(float(min(a[0], b[0]))))
    span1 = int(round(float(max(a[0], b[0]))))
    lo = max(0, line - max(1, int(iterations)))
    hi = min(arr.shape[1], line + max(1, int(iterations)) + 1)
    profile = np.any(near[:, lo:hi], axis=1)
    run = _overlapping_profile_run(profile, span0, span1)
    if run is None:
        return None
    start, end = run
    start = max(0, int(start))
    end = min(arr.shape[0] - 1, int(end))
    if end < start:
        return None
    return {
        "length_m": float((int(end) - int(start) + 1) * float(resolution_m)),
        "p0_rc": [int(start), int(line)],
        "p1_rc": [int(end), int(line)],
    }


def _line_theta_from_points(p0: object, p1: object) -> float:
    try:
        a = np.asarray(p0, dtype=np.float32).reshape(-1)
        b = np.asarray(p1, dtype=np.float32).reshape(-1)
        if a.size < 2 or b.size < 2:
            return 0.0
        delta = b[:2] - a[:2]
        if float(np.linalg.norm(delta)) <= 1e-6:
            return 0.0
        return float(np.arctan2(float(delta[0]), float(delta[1])) % float(np.pi))
    except Exception:
        return 0.0


def _edge_center_from_total_points(edge: Mapping[str, object]) -> np.ndarray:
    p0 = edge.get("total_p0_rc") or _edge_point(edge, "p0") or [0, 0]
    p1 = edge.get("total_p1_rc") or _edge_point(edge, "p1") or [0, 0]
    return 0.5 * (np.asarray(p0, dtype=np.float32) + np.asarray(p1, dtype=np.float32))


def _edge_tangent_overlap_m(a: Mapping[str, object], b: Mapping[str, object], resolution_m: float) -> float:
    theta = 0.5 * (float(a.get("theta", 0.0)) + float(b.get("theta", 0.0)))
    tangent = np.asarray([np.sin(theta), np.cos(theta)], dtype=np.float32)

    def vals(edge: Mapping[str, object]) -> list[float]:
        p0 = edge.get("total_p0_rc") or _edge_point(edge, "p0") or [0, 0]
        p1 = edge.get("total_p1_rc") or _edge_point(edge, "p1") or [0, 0]
        return [
            float(np.dot(np.asarray(p0, dtype=np.float32), tangent)),
            float(np.dot(np.asarray(p1, dtype=np.float32), tangent)),
        ]

    a_vals = vals(a)
    b_vals = vals(b)
    overlap = min(max(a_vals), max(b_vals)) - max(min(a_vals), min(b_vals))
    return float(max(0.0, overlap) * float(resolution_m))


def _mark_edge_candidate_rejected(edge: Mapping[str, object], reason: str) -> None:
    candidate = edge.get("candidate")
    if not isinstance(candidate, SeparatorCandidate):
        return
    if bool(candidate.debug.get("rejected_after_corridor_merge", False)):
        return
    candidate.debug["rejected_after_corridor_merge"] = True
    candidate.debug["corridor_merge_reject_reason"] = str(reason)


def _edge_total_length_after_noise_fill(edge: Mapping[str, object], wall_mask: np.ndarray, resolution_m: float) -> dict:
    candidate = edge.get("candidate")
    if not isinstance(candidate, SeparatorCandidate):
        fallback = float(edge.get("length_m", 0.0))
        return {"length_m": fallback, "source": "candidate_fallback", "p0_rc": None, "p1_rc": None}
    shape = tuple(np.asarray(wall_mask, dtype=bool).shape[:2])
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        fallback = float(candidate.length_m)
        return {"length_m": fallback, "source": "candidate_fallback", "p0_rc": _edge_point(edge, "p0"), "p1_rc": _edge_point(edge, "p1")}
    p0 = np.rint(np.asarray(candidate.p0_rc, dtype=np.float32)).astype(np.int32).reshape(-1)
    p1 = np.rint(np.asarray(candidate.p1_rc, dtype=np.float32)).astype(np.int32).reshape(-1)
    if p0.size < 2 or p1.size < 2:
        fallback = float(candidate.length_m)
        return {"length_m": fallback, "source": "candidate_fallback", "p0_rc": _edge_point(edge, "p0"), "p1_rc": _edge_point(edge, "p1")}
    merged = np.asarray(wall_mask, dtype=bool).copy()
    merged |= candidate.mask(shape)
    horizontal = abs(int(p1[1]) - int(p0[1])) >= abs(int(p1[0]) - int(p0[0]))
    if horizontal:
        line = int(round(float(p0[0] + p1[0]) * 0.5))
        span0 = int(min(p0[1], p1[1]))
        span1 = int(max(p0[1], p1[1]))
        lo = max(0, line - 1)
        hi = min(shape[0], line + 2)
        profile = np.any(merged[lo:hi, :], axis=0)
        run = _overlapping_profile_run(profile, span0, span1)
        if run is None:
            length_m = float(candidate.length_m)
            return {"length_m": length_m, "source": "candidate_fallback", "p0_rc": _edge_point(edge, "p0"), "p1_rc": _edge_point(edge, "p1")}
        start, end = run
        return {
            "length_m": float((int(end) - int(start) + 1) * float(resolution_m)),
            "source": "noise_gap_filled_wall_target_plus_candidate",
            "p0_rc": [int(line), int(start)],
            "p1_rc": [int(line), int(end)],
        }
    line = int(round(float(p0[1] + p1[1]) * 0.5))
    span0 = int(min(p0[0], p1[0]))
    span1 = int(max(p0[0], p1[0]))
    lo = max(0, line - 1)
    hi = min(shape[1], line + 2)
    profile = np.any(merged[:, lo:hi], axis=1)
    run = _overlapping_profile_run(profile, span0, span1)
    if run is None:
        length_m = float(candidate.length_m)
        return {"length_m": length_m, "source": "candidate_fallback", "p0_rc": _edge_point(edge, "p0"), "p1_rc": _edge_point(edge, "p1")}
    start, end = run
    return {
        "length_m": float((int(end) - int(start) + 1) * float(resolution_m)),
        "source": "noise_gap_filled_wall_target_plus_candidate",
        "p0_rc": [int(start), int(line)],
        "p1_rc": [int(end), int(line)],
    }


def _overlapping_profile_run(profile: np.ndarray, span0: int, span1: int) -> tuple[int, int] | None:
    arr = np.asarray(profile, dtype=bool)
    if arr.size == 0:
        return None
    lo = max(0, int(min(span0, span1)))
    hi = min(arr.size - 1, int(max(span0, span1)))
    best: tuple[int, int] | None = None
    best_overlap = -1
    best_len = -1
    for start, end_exclusive in _runs(arr):
        end = int(end_exclusive) - 1
        overlap = max(0, min(end, hi) - max(int(start), lo) + 1)
        length = int(end) - int(start) + 1
        if overlap <= 0:
            continue
        if overlap > best_overlap or (overlap == best_overlap and length > best_len):
            best = (int(start), int(end))
            best_overlap = int(overlap)
            best_len = int(length)
    return best


def _edge_point(edge: Mapping[str, object], key: str) -> list[int] | None:
    candidate = edge.get("candidate")
    if not isinstance(candidate, SeparatorCandidate):
        return None
    raw = candidate.p0_rc if key == "p0" else candidate.p1_rc
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    return [int(round(float(arr[0]))), int(round(float(arr[1])))]


def _label_centroid_rc(labels: np.ndarray, label: int) -> list[int] | None:
    coords = np.argwhere(np.asarray(labels, dtype=np.int32) == int(label))
    if coords.size == 0:
        return None
    return [int(round(float(coords[:, 0].mean()))), int(round(float(coords[:, 1].mean())))]


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


def _wall_endpoint_support_debug(
    wall_candidate_clean: np.ndarray | None,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius_cells: int,
) -> dict:
    if wall_candidate_clean is None:
        return {
            "wall_endpoint_support_required": False,
            "wall_endpoint_support_both": True,
            "wall_endpoint_support_score": 1.0,
            "wall_endpoint_support_counts": [],
        }
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    radius = max(0, int(radius_cells))
    counts: list[int] = []
    for point in (p0, p1):
        rc = np.rint(np.asarray(point, dtype=np.float32)).astype(np.int32)
        r = int(rc[0])
        c = int(rc[1])
        patch = wall[
            max(0, r - radius) : min(int(wall.shape[0]), r + radius + 1),
            max(0, c - radius) : min(int(wall.shape[1]), c + radius + 1),
        ]
        counts.append(int(np.count_nonzero(patch)))
    support_hits = [count > 0 for count in counts]
    return {
        "wall_endpoint_support_required": True,
        "wall_endpoint_support_radius_cells": int(radius),
        "wall_endpoint_support_counts": [int(v) for v in counts],
        "wall_endpoint_support_both": bool(all(support_hits)),
        "wall_endpoint_support_score": float(sum(1 for hit in support_hits if hit) / max(1, len(support_hits))),
    }


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


def _candidate_center(candidate: SeparatorCandidate) -> np.ndarray:
    return 0.5 * (np.asarray(candidate.p0_rc, dtype=np.float32) + np.asarray(candidate.p1_rc, dtype=np.float32))


def _angle_diff(a: float, b: float) -> float:
    diff = abs((float(a) - float(b)) % float(np.pi))
    return float(min(diff, float(np.pi) - diff))


def _candidate_tangent_overlap_m(a: SeparatorCandidate, b: SeparatorCandidate, resolution_m: float) -> float:
    theta = 0.5 * (float(a.theta) + float(b.theta))
    tangent = np.asarray([np.sin(theta), np.cos(theta)], dtype=np.float32)
    a_vals = [float(np.dot(np.asarray(a.p0_rc, dtype=np.float32), tangent)), float(np.dot(np.asarray(a.p1_rc, dtype=np.float32), tangent))]
    b_vals = [float(np.dot(np.asarray(b.p0_rc, dtype=np.float32), tangent)), float(np.dot(np.asarray(b.p1_rc, dtype=np.float32), tangent))]
    overlap = min(max(a_vals), max(b_vals)) - max(min(a_vals), min(b_vals))
    return float(max(0.0, overlap) * float(resolution_m))


def _candidate_by_id(candidates: Sequence[SeparatorCandidate], candidate_id: int) -> SeparatorCandidate | None:
    for candidate in candidates:
        if int(candidate.candidate_id) == int(candidate_id):
            return candidate
    return None


def _labels_touching_candidate(labels: np.ndarray, candidate: SeparatorCandidate, shape: tuple[int, int], *, iterations: int = 1) -> list[int]:
    mask = ndimage.binary_dilation(candidate.mask(shape), iterations=max(1, int(iterations)))
    return sorted(int(v) for v in np.unique(labels[mask]) if int(v) > 0)


def _label_region_infos(
    labels: np.ndarray,
    free: np.ndarray,
    distance: np.ndarray,
    resolution_m: float,
    wall_candidate_clean: np.ndarray,
    filtered_lines: Sequence[object],
    config: CorridorMergeConfig,
) -> dict[int, dict]:
    _ = wall_candidate_clean, filtered_lines
    out: dict[int, dict] = {}
    width_distance = np.asarray(distance, dtype=np.float32)
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = (labels == int(label)) & free
        metrics = component_metrics(mask, float(resolution_m), width_distance)
        widths = 2.0 * width_distance[mask] * float(resolution_m)
        p90_width = float(np.percentile(widths, 90)) if widths.size else 0.0
        width_std = float(np.std(widths)) if widths.size else 0.0
        corridor_like = bool(
            metrics["aspect_ratio"] >= float(config.parallel_door_min_overlap_m / max(0.01, config.parallel_door_min_overlap_m)) * 2.5
            and float(metrics.get("median_width_m", 0.0)) <= 1.80
            and p90_width <= 2.25
            and float(metrics.get("length_m", 0.0)) >= 1.0
        )
        open_living_room_like = bool(
            float(metrics.get("median_width_m", 0.0)) >= 1.80
            and float(metrics.get("area_m2", 0.0)) >= 2.0
            and not corridor_like
        )
        out[int(label)] = {
            "label": int(label),
            **metrics,
            "p90_width_m": float(p90_width),
            "width_std_m": float(width_std),
            "corridor_like": bool(corridor_like),
            "open_living_room_like": bool(open_living_room_like),
        }
    return out


def _width_axis_similar(a: Mapping[str, object], b: Mapping[str, object], config: CorridorMergeConfig) -> bool:
    if not a or not b:
        return False
    return bool(
        abs(float(a.get("median_width_m", 0.0)) - float(b.get("median_width_m", 0.0))) <= float(config.width_similarity_tol_m)
        and abs(float(a.get("aspect_ratio", 1.0)) - float(b.get("aspect_ratio", 1.0))) <= 4.0
    )


def _is_protected_room_region(info: Mapping[str, object], config: CorridorMergeConfig) -> bool:
    if not info:
        return False
    area_m2 = float(info.get("area_m2", 0.0))
    return bool(
        area_m2 >= max(float(config.isolated_chunk_max_area_m2), float(config.min_region_area_m2) * 4.0)
        and not bool(info.get("corridor_like", False))
    )


def _best_single_merge_neighbor(
    middle_info: Mapping[str, object],
    neighbors: Sequence[int],
    infos: Mapping[int, Mapping[str, object]],
) -> int:
    if not neighbors:
        return 0
    middle_width = float(middle_info.get("median_width_m", 0.0))
    middle_aspect = float(middle_info.get("aspect_ratio", 1.0))

    def score(label: int) -> tuple[float, float]:
        info = infos.get(int(label), {})
        width_delta = abs(float(info.get("median_width_m", 0.0)) - middle_width)
        aspect_delta = abs(float(info.get("aspect_ratio", 1.0)) - middle_aspect)
        return (float(width_delta), float(aspect_delta))

    return int(min([int(n) for n in neighbors], key=score))


def _candidates_rejected_by_merge(
    *,
    mid_label: int,
    merge_labels: Sequence[int],
    candidate_a: SeparatorCandidate,
    candidate_b: SeparatorCandidate,
    labels_a: Sequence[int],
    labels_b: Sequence[int],
) -> list[SeparatorCandidate]:
    merged = {int(v) for v in merge_labels}
    if len(merged) >= 3:
        return [candidate_a, candidate_b]
    neighbors = sorted(v for v in merged if int(v) != int(mid_label))
    if not neighbors:
        return []
    neighbor = int(neighbors[0])
    out: list[SeparatorCandidate] = []
    if int(mid_label) in {int(v) for v in labels_a} and neighbor in {int(v) for v in labels_a}:
        out.append(candidate_a)
    if int(mid_label) in {int(v) for v in labels_b} and neighbor in {int(v) for v in labels_b}:
        out.append(candidate_b)
    return out


def _neighbor_labels(labels: np.ndarray, label: int, *, iterations: int = 1) -> list[int]:
    mask = labels == int(label)
    border = ndimage.binary_dilation(mask, iterations=max(1, int(iterations))) & ~mask
    return sorted(int(v) for v in np.unique(labels[border]) if int(v) > 0)


def _free_reachable_neighbor_labels(labels: np.ndarray, free: np.ndarray, label: int, *, max_iterations: int = 10) -> list[int]:
    arr = np.asarray(labels, dtype=np.int32)
    free_arr = np.asarray(free, dtype=bool)
    if arr.shape != free_arr.shape:
        return []
    seed = (arr == int(label)) & free_arr
    if not np.any(seed):
        return []
    wave = seed.copy()
    passable_unlabeled = free_arr & (arr == 0)
    found: set[int] = set()
    for _ in range(max(1, int(max_iterations))):
        border = ndimage.binary_dilation(wave, iterations=1) & ~wave
        found.update(int(v) for v in np.unique(arr[border & free_arr]) if int(v) > 0 and int(v) != int(label))
        next_wave = border & passable_unlabeled
        if not np.any(next_wave):
            break
        wave |= next_wave
    final_border = ndimage.binary_dilation(wave, iterations=1) & ~wave
    found.update(int(v) for v in np.unique(arr[final_border & free_arr]) if int(v) > 0 and int(v) != int(label))
    return sorted(found)


def _region_surrounding_unknown_ratio(
    labels: np.ndarray,
    label: int,
    unknown: np.ndarray,
    *,
    iterations: int = 2,
) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    unk = np.asarray(unknown, dtype=bool)
    if arr.shape != unk.shape:
        return 0.0
    mask = arr == int(label)
    if not np.any(mask):
        return 1.0
    ring = ndimage.binary_dilation(mask, iterations=max(1, int(iterations))) & ~mask
    if not np.any(ring):
        return 0.0
    return float(np.count_nonzero(unk & ring)) / float(np.count_nonzero(ring))


def _remap_labels_with_parent(labels: np.ndarray, parent: dict[int, int], free: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    arr = np.asarray(labels, dtype=np.int32)
    out = arr.copy()
    remap: dict[int, int] = {}
    label_to_new: dict[int, int] = {}
    next_label = 1
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        root = _find(parent, int(label))
        if root not in remap:
            remap[root] = next_label
            next_label += 1
        new_label = int(remap[root])
        out[arr == label] = new_label
        label_to_new[int(label)] = new_label
    out[~np.asarray(free, dtype=bool)] = 0
    return out.astype(np.int32), label_to_new


def _find(parent: dict[int, int], label: int) -> int:
    parent.setdefault(int(label), int(label))
    if parent[int(label)] != int(label):
        parent[int(label)] = _find(parent, parent[int(label)])
    return parent[int(label)]


def _union(parent: dict[int, int], a: int, b: int) -> None:
    ra = _find(parent, int(a))
    rb = _find(parent, int(b))
    if ra != rb:
        parent[rb] = ra
