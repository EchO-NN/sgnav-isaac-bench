from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from isaac_bench.mapping.door_pattern_detector import DoorPatternConfig, detect_pre_extension_doors
from isaac_bench.mapping.rose2_source_form import ROSE2SourceResult


VERTICAL_FREE_GAP_CLOSURE_BACKEND = "vertical_free_gap_closure_v1"
VERTICAL_FREE_GAP_CLOSURE_ALGORITHM = "vertical_free_gap_closure_v1"
VERTICAL_FREE_GAP_CLOSURE_CONTEXT = "vertical_free_gap_closure_v1_vlm"


@dataclass
class VFGCConfig:
    resolution_m: float = 0.05
    close_max_gap_m: float = 1.50
    close_min_gap_m: float = 0.15
    virtual_boundary_radius_m: float = 0.06
    virtual_boundary_connect_to_wall_radius_m: float = 0.08
    wall_min_component_cells: int = 3
    free_min_component_cells: int = 8
    wall_micro_close_radius_cells: int = 1
    use_skeleton_endpoints: bool = True
    use_curvature_endpoints: bool = True
    endpoint_min_wall_support_m: float = 0.25
    endpoint_tangent_window_m: float = 0.35
    max_endpoints_per_component: int = 128
    tangent_parallel_cos_min: float = 0.70
    tangent_line_cos_min: float = 0.65
    throat_tangent_parallel_cos_min: float = 0.45
    corner_guard_min_tangent_parallel_cos: float = 0.35
    same_component_min_wall_path_m: float = 0.50
    line_free_ratio_min: float = 0.70
    line_unknown_ratio_max: float = 0.10
    line_mid_wall_ratio_max: float = 0.20
    side_support_band_m: float = 0.35
    side_support_min_free_cells: int = 15
    side_support_min_ratio: float = 0.15
    side_support_balance_min: float = 0.25
    max_candidates: int = 2048
    nms_line_distance_m: float = 0.25
    nms_endpoint_distance_m: float = 0.20
    candidate_score_min: float = 0.45
    topology_verify_enabled: bool = True
    min_room_area_m2: float = 1.0
    small_component_area_m2: float = 0.6
    keep_closure_if_between_two_meaningful_rooms: bool = True
    pre_extension_door_detection_enabled: bool = True
    pre_extension_door_pattern: DoorPatternConfig = field(default_factory=DoorPatternConfig)
    debug_dump: bool = False
    debug_dir: str = "debug/vertical_free_gap_closure"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VFGCConfig":
        raw = dict(data or {})
        debug_layers = dict(raw.get("debug_layers", {}) or {})
        if "enabled" in debug_layers:
            raw.setdefault("debug_dump", bool(debug_layers.get("enabled")))
        if "output_dir" in debug_layers:
            raw.setdefault("debug_dir", debug_layers.get("output_dir"))
        raw.update({key: value for key, value in overrides.items() if value is not None})
        nested = {
            "pre_extension_door_pattern": DoorPatternConfig.from_mapping(
                raw.get("pre_extension_door_pattern") or raw.get("pre_extension_door_detection")
            )
        }
        fields = {name for name in cls.__dataclass_fields__}
        values = {key: raw[key] for key in raw if key in fields and key not in nested}
        values.update(nested)
        return cls(**values)


@dataclass
class WallEndpoint:
    id: int
    rc: tuple[int, int]
    component_id: int
    tangent: np.ndarray
    normal: np.ndarray
    confidence: float
    source: str


@dataclass
class GapClosureCandidate:
    id: int
    endpoint_a: WallEndpoint
    endpoint_b: WallEndpoint
    p0: tuple[int, int]
    p1: tuple[int, int]
    length_m: float
    line_rcs: np.ndarray
    closure_type: str
    score: float = 0.0
    accepted: bool = False
    reject_reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict:
        return {
            "id": int(self.id),
            "p0": [int(self.p0[0]), int(self.p0[1])],
            "p1": [int(self.p1[0]), int(self.p1[1])],
            "endpoint_a": int(self.endpoint_a.id),
            "endpoint_b": int(self.endpoint_b.id),
            "component_a": int(self.endpoint_a.component_id),
            "component_b": int(self.endpoint_b.component_id),
            "length_m": float(self.length_m),
            "type": str(self.closure_type),
            "score": float(self.score),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            **_json_ready(self.debug),
        }


@dataclass
class VerticalFreeGapClosureResult:
    room_label_map: np.ndarray
    virtual_boundary_map: np.ndarray
    wall_boundary_map: np.ndarray
    wall_skeleton_map: np.ndarray
    endpoint_map: np.ndarray
    candidate_closure_map: np.ndarray
    accepted_closure_map: np.ndarray
    rejected_closure_map: np.ndarray
    room_label_map_visual: np.ndarray
    room_stats: list[dict]
    closure_candidates: list[dict]
    debug: dict[str, Any] = field(default_factory=dict)


def run_vertical_free_gap_closure_roomseg(
    *,
    free_mask: np.ndarray,
    wall_mask: np.ndarray,
    unknown_mask: np.ndarray,
    resolution_m: float,
    config: VFGCConfig | None = None,
    debug_dir: str | Path | None = None,
    step_id: int = 0,
) -> VerticalFreeGapClosureResult:
    t0 = time.perf_counter()
    cfg = config or VFGCConfig(resolution_m=float(resolution_m))
    cfg.resolution_m = float(resolution_m)
    free, wall, unknown, clean_debug = clean_vfgc_inputs(free_mask, wall_mask, unknown_mask, cfg)
    door_cfg = cfg.pre_extension_door_pattern
    if not bool(cfg.pre_extension_door_detection_enabled):
        door_cfg = DoorPatternConfig(enabled=False)
    door_result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=wall,
        unknown_mask=unknown,
        config=door_cfg,
        observed_mask=None,
        roi=None,
    )
    pre_extension_door_detected_map = np.asarray(door_result.detected_door_mask, dtype=bool)
    pre_extension_door_cut_mask = np.asarray(door_result.door_cut_mask, dtype=bool) & free
    pre_extension_door_pattern_type_map = np.asarray(door_result.pattern_type_map, dtype=np.uint8)
    pre_extension_partition_free = free & ~pre_extension_door_cut_mask
    pre_extension_room_label_map, pre_extension_room_count_before_merge = ndimage.label(
        pre_extension_partition_free,
        structure=_conn(8),
    )
    pre_extension_room_label_map = merge_small_components(
        pre_extension_room_label_map.astype(np.int32),
        free,
        pre_extension_door_cut_mask,
        cfg,
    )
    pre_extension_room_label_map = _relabel_compact(pre_extension_room_label_map)
    pre_extension_room_label_map[unknown] = 0
    pre_extension_room_label_map[~free] = 0
    pre_extension_room_count = int(_label_count(pre_extension_room_label_map))
    distance_m = ndimage.distance_transform_edt(free) * float(cfg.resolution_m)
    wall_boundary = compute_wall_boundary(wall, free)
    wall_skeleton = skeletonize_wall(wall)
    wall_labels, num_wall_components = ndimage.label(wall, structure=_conn(8))
    endpoints, endpoint_debug = extract_wall_endpoints(wall, wall_boundary, wall_skeleton, wall_labels, cfg)
    raw_candidates, gen_debug = generate_endpoint_gap_candidates(endpoints, free, wall, unknown, wall_labels, cfg)

    provisional: list[GapClosureCandidate] = []
    rejected: list[GapClosureCandidate] = []
    for cand in raw_candidates:
        ok, reason = validate_candidate_geometry(cand, free, wall, unknown, wall_labels, cfg)
        if ok:
            cand.accepted = True
            provisional.append(cand)
        else:
            cand.accepted = False
            cand.reject_reason = reason
            rejected.append(cand)

    provisional, nms_rejected = nms_closure_candidates(provisional, cfg)
    rejected.extend(nms_rejected)
    provisional_boundary = rasterize_closures(provisional, free.shape, cfg, include_rejected=False)
    provisional_boundary_with_pre_doors = provisional_boundary | pre_extension_door_cut_mask
    labels0, _ = ndimage.label(free & ~provisional_boundary_with_pre_doors, structure=_conn(8))
    accepted, topology_rejected, topology_debug = topology_prune_closures(provisional, labels0, free, cfg)
    rejected.extend(topology_rejected)
    original_step1_step2_virtual_boundary = rasterize_closures(accepted, free.shape, cfg, include_rejected=False)
    virtual_boundary = (pre_extension_door_cut_mask | original_step1_step2_virtual_boundary) & free
    partition_free = free & ~virtual_boundary
    labels, num_before_small_merge = ndimage.label(partition_free, structure=_conn(8))
    labels = merge_small_components(labels.astype(np.int32), free, virtual_boundary, cfg)
    labels = _relabel_compact(labels)
    labels[unknown] = 0
    labels[~free] = 0
    visual_labels = absorb_virtual_boundary_for_visual(labels, virtual_boundary, free)
    room_stats = compute_vfgc_room_stats(labels, distance_m, cfg)

    candidate_map = rasterize_closures(raw_candidates, free.shape, cfg, include_rejected=True)
    rejected_map = rasterize_closures(rejected, free.shape, cfg, include_rejected=True)
    endpoint_map = np.zeros_like(free, dtype=np.int32)
    for ep in endpoints:
        endpoint_map[int(ep.rc[0]), int(ep.rc[1])] = int(ep.id)
    accepted_count = int(len(accepted))
    rejected_reason_counts = dict(Counter(str(c.reject_reason or "accepted") for c in rejected))
    labels_outside_free = int(np.count_nonzero((labels > 0) & ~np.asarray(free_mask, dtype=bool)))
    labels_in_unknown = int(np.count_nonzero((labels > 0) & np.asarray(unknown_mask, dtype=bool)))
    debug = {
        "backend": VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        "actual_backend": VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        "source_backend": VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        "roomseg_backend": VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        "algorithm": VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        "source": VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        "context_source": VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
        "room_map_mode": VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
        "source_repository": None,
        "source_root_required": False,
        "source_provenance": {"available": False, "required_files": [], "files": []},
        "source_form_used": False,
        "source_form_used_for_final": False,
        "legacy_style_used": False,
        "legacy_style_used_for_final": False,
        "silent_fallback_used": False,
        "legacy_connected_component_rooms_used": False,
        "strict_fallback_used": False,
        "step_id": int(step_id),
        "resolution_m": float(cfg.resolution_m),
        "free_cells": int(np.count_nonzero(free)),
        "wall_cells": int(np.count_nonzero(wall)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "pre_extension_door_detected_map": pre_extension_door_detected_map,
        "pre_extension_door_cut_mask": pre_extension_door_cut_mask,
        "pre_extension_door_pattern_type_map": pre_extension_door_pattern_type_map,
        "pre_extension_partition_free": pre_extension_partition_free,
        "pre_extension_room_label_map": pre_extension_room_label_map,
        "pre_extension_room_count_before_small_merge": int(pre_extension_room_count_before_merge),
        "pre_extension_room_count": int(pre_extension_room_count),
        "pre_extension_door_detection_inserted_before_wall_extension": True,
        "pre_extension_door_debug_summary": {
            "enabled": bool(door_result.debug.get("pre_extension_door_detection_enabled", False)),
            "accepted": int(door_result.debug.get("pre_extension_door_num_accepted", 0)),
            "rule_a": int(door_result.debug.get("pre_extension_door_rule_a_count", 0)),
            "rule_b": int(door_result.debug.get("pre_extension_door_rule_b_count", 0)),
            "cut_cells": int(np.count_nonzero(pre_extension_door_cut_mask)),
            "pre_room_count": int(pre_extension_room_count),
        },
        **door_result.debug,
        "num_wall_components": int(num_wall_components),
        "num_endpoints": int(len(endpoints)),
        "num_candidate_pairs": int(len(raw_candidates)),
        "num_provisional_closures": int(len(provisional)),
        "num_accepted_closures": accepted_count,
        "num_rejected_closures": int(len(rejected)),
        "num_rooms_before_small_merge": int(num_before_small_merge),
        "num_rooms_final": int(_label_count(labels)),
        "room_count": int(_label_count(labels)),
        "final_room_count": int(_label_count(labels)),
        "accepted_closure_count": accepted_count,
        "rejection_reasons": rejected_reason_counts,
        "labels_outside_vertical_free_cells": labels_outside_free,
        "labels_in_unknown_cells": labels_in_unknown,
        "room_stats": room_stats,
        "closure_candidates": [c.serialize() for c in [*accepted, *rejected]],
        "timing_ms": {"total": float((time.perf_counter() - t0) * 1000.0)},
        "input_free": np.asarray(free_mask, dtype=bool),
        "input_wall": np.asarray(wall_mask, dtype=bool),
        "input_unknown": np.asarray(unknown_mask, dtype=bool),
        "clean_free": free,
        "clean_wall": wall,
        "wall_boundary_map": wall_boundary,
        "wall_skeleton_map": wall_skeleton,
        "endpoint_map": endpoint_map,
        "candidate_closure_map": candidate_map,
        "accepted_closure_map": virtual_boundary,
        "step1_step2_accepted_closure_map": original_step1_step2_virtual_boundary,
        "rejected_closure_map": rejected_map,
        "virtual_boundary_map": virtual_boundary,
        "partition_free": partition_free,
        "final_room_label_map": labels,
        "room_label_map_visual": visual_labels,
    }
    debug.update(clean_debug)
    debug.update(endpoint_debug)
    debug.update(gen_debug)
    debug.update(topology_debug)
    result = VerticalFreeGapClosureResult(
        room_label_map=labels.astype(np.int32),
        virtual_boundary_map=virtual_boundary.astype(bool),
        wall_boundary_map=wall_boundary.astype(bool),
        wall_skeleton_map=wall_skeleton.astype(bool),
        endpoint_map=endpoint_map.astype(np.int32),
        candidate_closure_map=candidate_map.astype(bool),
        accepted_closure_map=virtual_boundary.astype(bool),
        rejected_closure_map=rejected_map.astype(bool),
        room_label_map_visual=visual_labels.astype(np.int32),
        room_stats=room_stats,
        closure_candidates=list(debug["closure_candidates"]),
        debug=debug,
    )
    if labels_outside_free or labels_in_unknown:
        raise AssertionError("vertical-free gap closure labels escaped free/unknown constraints")
    if cfg.debug_dump or debug_dir is not None:
        save_vertical_free_gap_closure_debug(result=result, out_dir=debug_dir or cfg.debug_dir, step=int(step_id))
    return result


def clean_vfgc_inputs(
    free_mask: np.ndarray,
    wall_mask: np.ndarray,
    unknown_mask: np.ndarray,
    cfg: VFGCConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    free = np.asarray(free_mask, dtype=bool).copy()
    wall = np.asarray(wall_mask, dtype=bool).copy()
    unknown = np.asarray(unknown_mask, dtype=bool).copy()
    if free.shape != wall.shape or free.shape != unknown.shape:
        raise ValueError("vertical-free gap closure inputs must have same HxW shape")
    free = free & ~unknown
    wall = wall & ~unknown & ~free
    removed_wall = _remove_small_components_inplace(wall, int(cfg.wall_min_component_cells))
    removed_free = _remove_small_components_inplace(free, int(cfg.free_min_component_cells))
    if int(cfg.wall_micro_close_radius_cells) > 0 and np.any(wall):
        closed = ndimage.binary_closing(wall, structure=_disk(int(cfg.wall_micro_close_radius_cells)))
        wall = wall | (closed & ~free & ~unknown)
    return free, wall, unknown, {
        "removed_small_wall_cells": int(removed_wall),
        "removed_small_free_cells": int(removed_free),
        "wall_micro_close_radius_cells": int(cfg.wall_micro_close_radius_cells),
    }


def compute_wall_boundary(wall: np.ndarray, free: np.ndarray) -> np.ndarray:
    return np.asarray(wall, dtype=bool) & ndimage.binary_dilation(np.asarray(free, dtype=bool), structure=_conn(8))


def skeletonize_wall(wall: np.ndarray) -> np.ndarray:
    wall_arr = np.asarray(wall, dtype=bool)
    try:
        from skimage.morphology import skeletonize

        return np.asarray(skeletonize(wall_arr), dtype=bool)
    except Exception:
        dist = ndimage.distance_transform_edt(wall_arr)
        local_max = dist == ndimage.maximum_filter(dist, footprint=_conn(8))
        skel = wall_arr & local_max
        if not np.any(skel):
            skel = wall_arr.copy()
        return skel.astype(bool)


def extract_wall_endpoints(
    wall: np.ndarray,
    wall_boundary: np.ndarray,
    wall_skeleton: np.ndarray,
    wall_labels: np.ndarray,
    cfg: VFGCConfig,
) -> tuple[list[WallEndpoint], dict]:
    endpoints: list[WallEndpoint] = []
    taken: set[tuple[int, int, str]] = set()
    next_id = 1
    if bool(cfg.use_skeleton_endpoints):
        nbr = ndimage.convolve(wall_skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0) - wall_skeleton.astype(np.uint8)
        for r, c in np.argwhere(wall_skeleton & (nbr <= 1)):
            comp = int(wall_labels[int(r), int(c)])
            if comp <= 0:
                comp = _nearest_component_id(wall_labels, int(r), int(c))
            ep = _make_endpoint(next_id, int(r), int(c), comp, wall, cfg, source="skeleton_endpoint")
            endpoints.append(ep)
            taken.add((int(r), int(c), "skeleton_endpoint"))
            next_id += 1
    curvature_added = 0
    if bool(cfg.use_curvature_endpoints):
        for comp_id in sorted(int(v) for v in np.unique(wall_labels) if int(v) > 0):
            comp_boundary = wall_boundary & (wall_labels == comp_id)
            coords = np.argwhere(comp_boundary)
            if coords.size == 0:
                continue
            scores = []
            for r, c in coords:
                wall_neighbors = sum(1 for nr, nc in _neighbors(int(r), int(c), wall.shape, 8) if wall[nr, nc])
                free_neighbors = sum(1 for nr, nc in _neighbors(int(r), int(c), wall.shape, 8) if not wall[nr, nc])
                if wall_neighbors <= 3 and free_neighbors >= 2:
                    scores.append((wall_neighbors, -free_neighbors, int(r), int(c)))
            scores.sort()
            limit = max(0, int(cfg.max_endpoints_per_component) - len([e for e in endpoints if e.component_id == comp_id]))
            for _score in scores[:limit]:
                _wn, _fn, r, c = _score
                if any(abs(r - er) <= 1 and abs(c - ec) <= 1 for er, ec, _src in taken):
                    continue
                ep = _make_endpoint(next_id, int(r), int(c), comp_id, wall, cfg, source="curvature_endpoint")
                endpoints.append(ep)
                taken.add((int(r), int(c), "curvature_endpoint"))
                next_id += 1
                curvature_added += 1
    return endpoints, {
        "skeleton_endpoint_count": int(len([e for e in endpoints if e.source == "skeleton_endpoint"])),
        "curvature_endpoint_count": int(curvature_added),
    }


def generate_endpoint_gap_candidates(
    endpoints: Sequence[WallEndpoint],
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    wall_labels: np.ndarray,
    cfg: VFGCConfig,
) -> tuple[list[GapClosureCandidate], dict]:
    del wall, unknown, wall_labels
    if len(endpoints) < 2:
        return [], {"candidate_pair_query_count": 0}
    coords = np.asarray([ep.rc for ep in endpoints], dtype=np.float32)
    tree = cKDTree(coords)
    max_gap_cells = float(cfg.close_max_gap_m) / max(float(cfg.resolution_m), 1e-9)
    min_gap_cells = float(cfg.close_min_gap_m) / max(float(cfg.resolution_m), 1e-9)
    pairs = sorted(tree.query_pairs(r=max_gap_cells))
    candidates: list[GapClosureCandidate] = []
    cid = 1
    for i, j in pairs:
        if len(candidates) >= int(cfg.max_candidates):
            break
        a, b = endpoints[int(i)], endpoints[int(j)]
        dist_cells = float(np.linalg.norm(np.asarray(a.rc, dtype=np.float32) - np.asarray(b.rc, dtype=np.float32)))
        if dist_cells < min_gap_cells:
            continue
        line = np.asarray(_bresenham_line(a.rc, b.rc), dtype=np.int32)
        if line.size == 0:
            continue
        u = _unit(np.asarray(b.rc, dtype=np.float32) - np.asarray(a.rc, dtype=np.float32))
        tangent_parallel = abs(float(np.dot(a.tangent, b.tangent)))
        line_a = abs(float(np.dot(u, a.tangent)))
        line_b = abs(float(np.dot(u, b.tangent)))
        if tangent_parallel >= float(cfg.tangent_parallel_cos_min) and line_a >= float(cfg.tangent_line_cos_min) and line_b >= float(cfg.tangent_line_cos_min):
            closure_type = "same_wall_gap"
        elif tangent_parallel >= float(cfg.throat_tangent_parallel_cos_min):
            closure_type = "throat_portal"
        else:
            closure_type = "candidate_unclassified"
        candidates.append(
            GapClosureCandidate(
                id=cid,
                endpoint_a=a,
                endpoint_b=b,
                p0=a.rc,
                p1=b.rc,
                length_m=dist_cells * float(cfg.resolution_m),
                line_rcs=line,
                closure_type=closure_type,
                debug={
                    "tangent_parallel": tangent_parallel,
                    "line_tangent_a": line_a,
                    "line_tangent_b": line_b,
                },
            )
        )
        cid += 1
    return candidates, {"candidate_pair_query_count": int(len(pairs))}


def validate_candidate_geometry(
    cand: GapClosureCandidate,
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    wall_labels: np.ndarray,
    cfg: VFGCConfig,
) -> tuple[bool, str | None]:
    if cand.length_m > float(cfg.close_max_gap_m):
        return _reject(cand, "gap_too_wide_open_plan")
    if cand.length_m < float(cfg.close_min_gap_m):
        return _reject(cand, "too_close")
    if abs(float(np.dot(cand.endpoint_a.tangent, cand.endpoint_b.tangent))) < float(cfg.corner_guard_min_tangent_parallel_cos):
        return _reject(cand, "corner_tangent_not_parallel")
    if cand.closure_type == "candidate_unclassified":
        return _reject(cand, "endpoint_alignment_failed")
    if cand.endpoint_a.component_id == cand.endpoint_b.component_id:
        path_m = _component_path_distance_m(wall_labels == cand.endpoint_a.component_id, cand.endpoint_a.rc, cand.endpoint_b.rc, cfg)
        cand.debug["same_component_path_distance_m"] = float(path_m)
        if path_m < float(cfg.same_component_min_wall_path_m):
            return _reject(cand, "same_component_local_corner")
    line = np.asarray(cand.line_rcs, dtype=np.int32)
    mid = _line_without_endpoint_margin(line, margin=max(1, _radius_cells(float(cfg.virtual_boundary_connect_to_wall_radius_m), cfg)))
    if mid.size == 0:
        return _reject(cand, "line_too_short_after_endpoint_margin")
    rr, cc = mid[:, 0], mid[:, 1]
    free_ratio = float(np.mean(free[rr, cc])) if rr.size else 0.0
    unknown_ratio = float(np.mean(unknown[rr, cc])) if rr.size else 1.0
    wall_ratio = float(np.mean(wall[rr, cc])) if rr.size else 1.0
    cand.debug.update({"free_ratio": free_ratio, "unknown_ratio": unknown_ratio, "mid_wall_ratio": wall_ratio})
    if unknown_ratio > float(cfg.line_unknown_ratio_max):
        return _reject(cand, "line_crosses_unknown")
    if wall_ratio > float(cfg.line_mid_wall_ratio_max):
        return _reject(cand, "line_crosses_existing_wall")
    if free_ratio < float(cfg.line_free_ratio_min):
        return _reject(cand, "line_not_free_enough")
    left_ratio, right_ratio, left_cells, right_cells = _side_free_support(cand, free, unknown, cfg)
    balance = min(left_ratio, right_ratio) / max(max(left_ratio, right_ratio), 1e-6)
    cand.debug.update(
        {
            "side_support_left": float(left_ratio),
            "side_support_right": float(right_ratio),
            "side_support_left_cells": int(left_cells),
            "side_support_right_cells": int(right_cells),
            "side_support_balance": float(balance),
        }
    )
    if (
        left_cells < int(cfg.side_support_min_free_cells)
        or right_cells < int(cfg.side_support_min_free_cells)
        or left_ratio < float(cfg.side_support_min_ratio)
        or right_ratio < float(cfg.side_support_min_ratio)
        or balance < float(cfg.side_support_balance_min)
    ):
        return _reject(cand, "one_sided_free_support_corner_like")
    support_a = _wall_support_cells(wall, cand.endpoint_a.rc, cfg)
    support_b = _wall_support_cells(wall, cand.endpoint_b.rc, cfg)
    cand.debug.update({"wall_support_a": int(support_a), "wall_support_b": int(support_b)})
    if support_a < 2 or support_b < 2:
        return _reject(cand, "weak_wall_support_near_endpoint")
    align = float(cand.debug.get("tangent_parallel", 0.0))
    cand.score = float(
        0.25 * min(1.0, (cand.endpoint_a.confidence + cand.endpoint_b.confidence) / 2.0)
        + 0.20 * align
        + 0.20 * free_ratio
        + 0.20 * min(left_ratio, right_ratio)
        + 0.15 * balance
        - 0.25 * unknown_ratio
    )
    if cand.score < float(cfg.candidate_score_min):
        return _reject(cand, "score_below_threshold")
    return True, None


def nms_closure_candidates(candidates: Sequence[GapClosureCandidate], cfg: VFGCConfig) -> tuple[list[GapClosureCandidate], list[GapClosureCandidate]]:
    kept: list[GapClosureCandidate] = []
    rejected: list[GapClosureCandidate] = []
    endpoint_thresh = float(cfg.nms_endpoint_distance_m)
    line_thresh = float(cfg.nms_line_distance_m)
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = False
        for old in kept:
            endpoint_dist = min(
                _point_dist_m(cand.p0, old.p0, cfg) + _point_dist_m(cand.p1, old.p1, cfg),
                _point_dist_m(cand.p0, old.p1, cfg) + _point_dist_m(cand.p1, old.p0, cfg),
            )
            line_dist = _line_mean_min_dist_m(cand.line_rcs, old.line_rcs, cfg)
            if endpoint_dist <= 2.0 * endpoint_thresh or line_dist <= line_thresh:
                duplicate = True
                break
        if duplicate:
            cand.accepted = False
            cand.reject_reason = "nms_duplicate"
            rejected.append(cand)
        else:
            kept.append(cand)
    return kept, rejected


def topology_prune_closures(
    provisional: Sequence[GapClosureCandidate],
    labels: np.ndarray,
    free: np.ndarray,
    cfg: VFGCConfig,
) -> tuple[list[GapClosureCandidate], list[GapClosureCandidate], dict]:
    if not bool(cfg.topology_verify_enabled):
        return list(provisional), [], {"topology_verify_enabled": False}
    min_cells = max(int(cfg.side_support_min_free_cells), int(round(float(cfg.min_room_area_m2) / max(cfg.resolution_m**2, 1e-9))))
    label_sizes = {int(v): int(np.count_nonzero(labels == int(v))) for v in np.unique(labels) if int(v) > 0}
    accepted: list[GapClosureCandidate] = []
    rejected: list[GapClosureCandidate] = []
    for cand in provisional:
        adjacent_radius = max(
            2,
            _radius_cells(float(cfg.virtual_boundary_radius_m) + 0.5 * float(cfg.side_support_band_m), cfg),
        )
        adjacent = _labels_near_line(cand.line_rcs, labels, radius_cells=adjacent_radius)
        meaningful = sorted(label for label in adjacent if label_sizes.get(int(label), 0) >= min_cells)
        cand.debug["topology_adjacent_labels"] = [int(v) for v in meaningful]
        if len(set(meaningful)) >= 2:
            cand.accepted = True
            cand.reject_reason = None
            accepted.append(cand)
        else:
            cand.accepted = False
            cand.reject_reason = "not_between_two_meaningful_rooms"
            rejected.append(cand)
    return accepted, rejected, {"topology_verify_enabled": True, "topology_meaningful_min_cells": int(min_cells)}


def rasterize_closures(
    candidates: Sequence[GapClosureCandidate],
    shape: tuple[int, int],
    cfg: VFGCConfig,
    *,
    include_rejected: bool,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    radius = max(0, _radius_cells(float(cfg.virtual_boundary_radius_m), cfg))
    for cand in candidates:
        if not include_rejected and not bool(cand.accepted):
            continue
        for r, c in np.asarray(cand.line_rcs, dtype=np.int32):
            if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
                out[int(r), int(c)] = True
    if radius > 0 and np.any(out):
        out = ndimage.binary_dilation(out, structure=_disk(radius))
    return out


def merge_small_components(labels: np.ndarray, free: np.ndarray, virtual_boundary: np.ndarray, cfg: VFGCConfig) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    min_cells = max(1, int(round(float(cfg.small_component_area_m2) / max(cfg.resolution_m**2, 1e-9))))
    for label in sorted(int(v) for v in np.unique(out) if int(v) > 0):
        mask = out == label
        if int(np.count_nonzero(mask)) >= min_cells:
            continue
        target = _best_adjacent_label_no_cross(out, mask, virtual_boundary)
        if target > 0:
            out[mask] = target
        else:
            out[mask] = 0
    out = _relabel_compact(out)
    out[~np.asarray(free, dtype=bool)] = 0
    out[np.asarray(virtual_boundary, dtype=bool)] = 0
    return out


def absorb_virtual_boundary_for_visual(labels: np.ndarray, virtual_boundary: np.ndarray, free: np.ndarray) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    target = np.asarray(virtual_boundary, dtype=bool) & np.asarray(free, dtype=bool)
    if not np.any(target) or not np.any(out > 0):
        return out
    _, inds = ndimage.distance_transform_edt(out <= 0, return_indices=True)
    out[target] = out[inds[0][target], inds[1][target]]
    return out


def compute_vfgc_room_stats(labels: np.ndarray, distance_m: np.ndarray, cfg: VFGCConfig) -> list[dict]:
    rows = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        rr, cc = np.nonzero(mask)
        if rr.size == 0:
            continue
        rows.append(
            {
                "label_id": int(label),
                "area_m2": float(rr.size) * float(cfg.resolution_m) ** 2,
                "centroid_rc": [float(np.mean(rr)), float(np.mean(cc))],
                "bbox_rc": [int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1],
                "mean_clearance_m": float(np.mean(distance_m[mask])) if np.any(mask) else 0.0,
                "max_clearance_m": float(np.max(distance_m[mask])) if np.any(mask) else 0.0,
                "reliability": 1.0,
            }
        )
    return rows


def vfgc_result_to_source_result(result: VerticalFreeGapClosureResult) -> ROSE2SourceResult:
    labels = np.asarray(result.room_label_map, dtype=np.int32)
    debug = dict(result.debug)
    return ROSE2SourceResult(
        backend=VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        room_label_map=labels,
        source_room_label_map=labels.copy(),
        clean_structure_map=np.asarray(result.virtual_boundary_map, dtype=bool),
        structural_score=np.asarray(labels > 0, dtype=np.float32),
        boundary_map=np.asarray(result.virtual_boundary_map, dtype=bool),
        dominant_directions_rad=[],
        hough_segments=[],
        wall_clusters=[],
        representative_lines=[],
        extended_lines=[],
        faces=_faces_from_labels(labels, float(debug.get("resolution_m", 0.05) or 0.05)),
        face_adjacency_edges=_adjacency_from_labels(labels),
        cell_edges=_adjacency_from_labels(labels),
        cell_polygons=_room_polygons_debug(labels),
        timing_ms=dict(debug.get("timing_ms", {})),
        debug=debug,
    )


def save_vertical_free_gap_closure_debug(
    *,
    result: VerticalFreeGapClosureResult,
    out_dir: str | Path,
    step: int = 0,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "vfgc_step_%06d" % int(step)
    npz_path = out / ("%s.npz" % stem)
    summary_path = out / ("%s.summary.json" % stem)
    overlay_path = out / ("%s.overlay.png" % stem)
    layers_path = out / ("%s.layers.png" % stem)
    d = result.debug
    _zero_bool = np.zeros_like(result.room_label_map, dtype=np.uint8)
    _zero_i32 = np.zeros_like(result.room_label_map, dtype=np.int32)
    np.savez_compressed(
        npz_path,
        free=np.asarray(d.get("clean_free"), dtype=np.uint8),
        wall=np.asarray(d.get("clean_wall"), dtype=np.uint8),
        unknown=np.asarray(d.get("input_unknown"), dtype=np.uint8),
        wall_boundary=np.asarray(result.wall_boundary_map, dtype=np.uint8),
        wall_skeleton=np.asarray(result.wall_skeleton_map, dtype=np.uint8),
        endpoint_map=np.asarray(result.endpoint_map, dtype=np.int32),
        candidate_closure_map=np.asarray(result.candidate_closure_map, dtype=np.uint8),
        accepted_closure_map=np.asarray(result.accepted_closure_map, dtype=np.uint8),
        step1_step2_accepted_closure_map=np.asarray(
            d.get("step1_step2_accepted_closure_map", _zero_bool),
            dtype=np.uint8,
        ),
        rejected_closure_map=np.asarray(result.rejected_closure_map, dtype=np.uint8),
        virtual_boundary_map=np.asarray(result.virtual_boundary_map, dtype=np.uint8),
        pre_extension_door_detected_map=np.asarray(
            d.get("pre_extension_door_detected_map", _zero_bool),
            dtype=np.uint8,
        ),
        pre_extension_door_cut_mask=np.asarray(
            d.get("pre_extension_door_cut_mask", _zero_bool),
            dtype=np.uint8,
        ),
        pre_extension_door_pattern_type_map=np.asarray(
            d.get("pre_extension_door_pattern_type_map", _zero_bool),
            dtype=np.uint8,
        ),
        pre_extension_partition_free=np.asarray(
            d.get("pre_extension_partition_free", _zero_bool),
            dtype=np.uint8,
        ),
        pre_extension_room_label_map=np.asarray(
            d.get("pre_extension_room_label_map", _zero_i32),
            dtype=np.int32,
        ),
        partition_free=np.asarray(d.get("partition_free"), dtype=np.uint8),
        room_label_map=np.asarray(result.room_label_map, dtype=np.int32),
        room_label_map_visual=np.asarray(result.room_label_map_visual, dtype=np.int32),
    )
    summary = _summary_from_result(result)
    summary["debug_npz"] = str(npz_path)
    summary["overlay_png"] = str(overlay_path)
    summary["layers_png"] = str(layers_path)
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        from PIL import Image, ImageDraw

        Image.fromarray(_vfgc_overlay_rgb(result), mode="RGB").save(overlay_path)
        panels = [
            _bool_rgb(np.asarray(d.get("clean_free"), dtype=bool), (220, 220, 220)),
            _bool_rgb(np.asarray(d.get("clean_wall"), dtype=bool), (20, 20, 20)),
            _bool_rgb(result.wall_boundary_map, (255, 150, 0)),
            _bool_rgb(result.wall_skeleton_map, (255, 190, 0)),
            _label_rgb(result.endpoint_map),
            _bool_rgb(result.candidate_closure_map, (0, 220, 255)),
            _bool_rgb(result.rejected_closure_map, (255, 0, 220)),
            _bool_rgb(result.accepted_closure_map, (255, 40, 40)),
            _label_rgb(result.room_label_map),
            _vfgc_overlay_rgb(result),
        ]
        h, w = panels[0].shape[:2]
        canvas = Image.new("RGB", (w * 5, h * 2), (0, 0, 0))
        for idx, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel, mode="RGB"), ((idx % 5) * w, (idx // 5) * h))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), "VFGC RoomSeg layers", fill=(255, 255, 255))
        canvas.save(layers_path)
    except Exception:
        pass
    result.debug["vertical_free_gap_closure_layers"] = {
        "backend": VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        "debug_npz": str(npz_path),
        "overlay_png": str(overlay_path),
        "layers_png": str(layers_path),
        "room_count": int(_label_count(result.room_label_map)),
        "accepted_closure_count": int(result.debug.get("num_accepted_closures", 0) or 0),
        "rejected_closure_count": int(result.debug.get("num_rejected_closures", 0) or 0),
    }
    return {"paths": result.debug["vertical_free_gap_closure_layers"], "summary": summary}


def _summary_from_result(result: VerticalFreeGapClosureResult) -> dict:
    keys = [
        "algorithm",
        "step_id",
        "resolution_m",
        "free_cells",
        "wall_cells",
        "unknown_cells",
        "num_wall_components",
        "num_endpoints",
        "num_candidate_pairs",
        "num_provisional_closures",
        "num_accepted_closures",
        "num_rejected_closures",
        "num_rooms_before_small_merge",
        "num_rooms_final",
        "rejection_reasons",
        "labels_outside_vertical_free_cells",
    ]
    out = {key: result.debug.get(key) for key in keys}
    out["closure_candidates"] = list(result.closure_candidates)
    out["room_stats"] = list(result.room_stats)
    return out


def _make_endpoint(endpoint_id: int, r: int, c: int, component_id: int, wall: np.ndarray, cfg: VFGCConfig, *, source: str) -> WallEndpoint:
    tangent = _estimate_tangent(wall, (r, c), cfg)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    support = _wall_support_cells(wall, (r, c), cfg)
    confidence = float(min(1.0, max(0.2, support / 12.0)))
    return WallEndpoint(
        id=int(endpoint_id),
        rc=(int(r), int(c)),
        component_id=int(component_id),
        tangent=tangent,
        normal=_unit(normal),
        confidence=confidence,
        source=source,
    )


def _estimate_tangent(wall: np.ndarray, rc: tuple[int, int], cfg: VFGCConfig) -> np.ndarray:
    radius = max(1, _radius_cells(float(cfg.endpoint_tangent_window_m), cfg))
    r, c = rc
    r0, r1 = max(0, r - radius), min(wall.shape[0], r + radius + 1)
    c0, c1 = max(0, c - radius), min(wall.shape[1], c + radius + 1)
    coords = np.argwhere(wall[r0:r1, c0:c1])
    if coords.shape[0] >= 2:
        coords = coords.astype(np.float32)
        coords[:, 0] += float(r0)
        coords[:, 1] += float(c0)
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        return _unit(vt[0].astype(np.float32))
    for nr, nc in _neighbors(r, c, wall.shape, 8):
        if wall[nr, nc]:
            return _unit(np.asarray([nr - r, nc - c], dtype=np.float32))
    return np.asarray([1.0, 0.0], dtype=np.float32)


def _reject(cand: GapClosureCandidate, reason: str) -> tuple[bool, str]:
    cand.accepted = False
    cand.reject_reason = reason
    return False, reason


def _line_without_endpoint_margin(line: np.ndarray, margin: int) -> np.ndarray:
    arr = np.asarray(line, dtype=np.int32)
    if arr.shape[0] <= 2 * margin:
        return arr[0:0]
    return arr[int(margin) : arr.shape[0] - int(margin)]


def _side_free_support(cand: GapClosureCandidate, free: np.ndarray, unknown: np.ndarray, cfg: VFGCConfig) -> tuple[float, float, int, int]:
    line = np.asarray(cand.line_rcs, dtype=np.int32)
    if line.shape[0] < 2:
        return 0.0, 0.0, 0, 0
    u = _unit(np.asarray(cand.p1, dtype=np.float32) - np.asarray(cand.p0, dtype=np.float32))
    normal = _unit(np.asarray([-u[1], u[0]], dtype=np.float32))
    max_off = max(1, _radius_cells(float(cfg.side_support_band_m), cfg))
    left_total = right_total = left_free = right_free = 0
    for sign in (-1.0, 1.0):
        for r, c in line:
            for off in range(1, max_off + 1):
                rr = int(round(float(r) + sign * normal[0] * off))
                cc = int(round(float(c) + sign * normal[1] * off))
                if not (0 <= rr < free.shape[0] and 0 <= cc < free.shape[1]) or unknown[rr, cc]:
                    continue
                if sign < 0:
                    left_total += 1
                    left_free += int(bool(free[rr, cc]))
                else:
                    right_total += 1
                    right_free += int(bool(free[rr, cc]))
    left_ratio = float(left_free) / float(max(1, left_total))
    right_ratio = float(right_free) / float(max(1, right_total))
    return left_ratio, right_ratio, int(left_free), int(right_free)


def _wall_support_cells(wall: np.ndarray, rc: tuple[int, int], cfg: VFGCConfig) -> int:
    radius = max(1, _radius_cells(float(cfg.endpoint_min_wall_support_m), cfg))
    r, c = rc
    r0, r1 = max(0, r - radius), min(wall.shape[0], r + radius + 1)
    c0, c1 = max(0, c - radius), min(wall.shape[1], c + radius + 1)
    return int(np.count_nonzero(wall[r0:r1, c0:c1]))


def _component_path_distance_m(comp: np.ndarray, start: tuple[int, int], goal: tuple[int, int], cfg: VFGCConfig) -> float:
    if start == goal:
        return 0.0
    queue = deque([(int(start[0]), int(start[1]), 0)])
    seen = {(int(start[0]), int(start[1]))}
    while queue:
        r, c, dist = queue.popleft()
        for nr, nc in _neighbors(r, c, comp.shape, 8):
            if (nr, nc) in seen or not bool(comp[nr, nc]):
                continue
            if (nr, nc) == (int(goal[0]), int(goal[1])):
                return float(dist + 1) * float(cfg.resolution_m)
            seen.add((nr, nc))
            queue.append((nr, nc, dist + 1))
    return float("inf")


def _labels_near_line(line: np.ndarray, labels: np.ndarray, radius_cells: int) -> list[int]:
    mask = np.zeros(labels.shape, dtype=bool)
    for r, c in np.asarray(line, dtype=np.int32):
        if 0 <= int(r) < labels.shape[0] and 0 <= int(c) < labels.shape[1]:
            mask[int(r), int(c)] = True
    dil = ndimage.binary_dilation(mask, structure=_disk(int(radius_cells)))
    return sorted(int(v) for v in np.unique(labels[dil]) if int(v) > 0)


def _best_adjacent_label_no_cross(labels: np.ndarray, mask: np.ndarray, virtual_boundary: np.ndarray) -> int:
    counts: Counter[int] = Counter()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors(int(r), int(c), labels.shape, 8):
            if virtual_boundary[nr, nc]:
                continue
            lab = int(labels[nr, nc])
            if lab > 0 and not mask[nr, nc]:
                counts[lab] += 1
    return int(counts.most_common(1)[0][0]) if counts else 0


def _remove_small_components_inplace(mask: np.ndarray, min_cells: int) -> int:
    removed = 0
    labels, n = ndimage.label(mask, structure=_conn(8))
    for label in range(1, int(n) + 1):
        comp = labels == label
        count = int(np.count_nonzero(comp))
        if count < int(min_cells):
            mask[comp] = False
            removed += count
    return int(removed)


def _nearest_component_id(labels: np.ndarray, r: int, c: int) -> int:
    for radius in range(1, 4):
        r0, r1 = max(0, r - radius), min(labels.shape[0], r + radius + 1)
        c0, c1 = max(0, c - radius), min(labels.shape[1], c + radius + 1)
        vals = [int(v) for v in np.unique(labels[r0:r1, c0:c1]) if int(v) > 0]
        if vals:
            return vals[0]
    return 0


def _point_dist_m(a: tuple[int, int], b: tuple[int, int], cfg: VFGCConfig) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))) * float(cfg.resolution_m)


def _line_mean_min_dist_m(a: np.ndarray, b: np.ndarray, cfg: VFGCConfig) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    tree = cKDTree(np.asarray(b, dtype=np.float32))
    dists, _ = tree.query(np.asarray(a, dtype=np.float32), k=1)
    return float(np.mean(dists)) * float(cfg.resolution_m)


def _bresenham_line(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = int(p0[0]), int(p0[1])
    r1, c1 = int(p1[0]), int(p1[1])
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


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


def _radius_cells(radius_m: float, cfg: VFGCConfig) -> int:
    return max(0, int(round(float(radius_m) / max(float(cfg.resolution_m), 1e-9))))


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


def _relabel_compact(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    for new_id, old_id in enumerate(sorted(int(v) for v in np.unique(arr) if int(v) > 0), start=1):
        out[arr == old_id] = int(new_id)
    return out


def _faces_from_labels(labels: np.ndarray, resolution_m: float) -> list[dict]:
    rows = []
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = labels == label
        rr, cc = np.nonzero(mask)
        if rr.size:
            rows.append(
                {
                    "face_id": int(label),
                    "label_id": int(label),
                    "area_m2": float(rr.size) * float(resolution_m) ** 2,
                    "bbox_rc": [int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1],
                }
            )
    return rows


def _adjacency_from_labels(labels: np.ndarray) -> list[dict]:
    edges: set[tuple[int, int]] = set()
    arr = np.asarray(labels, dtype=np.int32)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            a = int(arr[r, c])
            if a <= 0:
                continue
            for nr, nc in _neighbors(r, c, arr.shape, 4):
                b = int(arr[nr, nc])
                if b > 0 and b != a:
                    edges.add((min(a, b), max(a, b)))
    return [{"label_a": int(a), "label_b": int(b)} for a, b in sorted(edges)]


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
        out[arr == label] = np.asarray(_color(label), dtype=np.uint8)
    return out


def _vfgc_overlay_rgb(result: VerticalFreeGapClosureResult) -> np.ndarray:
    free = np.asarray(result.debug.get("clean_free"), dtype=bool)
    wall = np.asarray(result.debug.get("clean_wall"), dtype=bool)
    unknown = np.asarray(result.debug.get("input_unknown"), dtype=bool)
    out = np.zeros((*result.room_label_map.shape, 3), dtype=np.uint8)
    out[unknown] = (55, 55, 55)
    out[free] = (210, 210, 210)
    out[wall] = (0, 0, 0)
    colors = _label_rgb(result.room_label_map_visual)
    mask = result.room_label_map_visual > 0
    out[mask] = (0.45 * out[mask] + 0.55 * colors[mask]).astype(np.uint8)
    out[result.candidate_closure_map] = (0, 220, 255)
    out[result.rejected_closure_map] = (255, 0, 220)
    out[result.accepted_closure_map] = (255, 40, 40)
    out[result.wall_skeleton_map] = (255, 170, 0)
    out[result.endpoint_map > 0] = (255, 255, 0)
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
