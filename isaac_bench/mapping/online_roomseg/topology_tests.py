from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorAnchorConfig, SeparatorCandidate, extend_candidate_to_anchors, separator_mask_for_candidate
from .utils import component_metrics, conn, dilate, label_components, relabel_compact


@dataclass
class TopologyTestConfig:
    enabled: bool = True
    min_split_area_m2: float = 1.0
    per_kind_min_split_area_m2: dict[str, float] | None = None
    min_new_component_width_m: float = 0.15
    min_new_component_area_cells: int = 4
    allow_partial_room_split: bool = True
    max_tiny_fragment_count: int = 3
    reject_corridor_split: bool = True
    open_room_width_min_m: float = 1.80
    open_room_min_side_area_m2: float = 2.00
    narrow_neck_width_max_m: float = 1.60
    reject_open_living_room_internal_split: bool = True
    separator: SeparatorAnchorConfig | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "TopologyTestConfig":
        raw = dict(data or {})
        separator_raw = dict(raw.pop("separator", {}) or {})
        fields = {name for name in cls.__dataclass_fields__}
        values = {key: raw[key] for key in raw if key in fields}
        values["separator"] = SeparatorAnchorConfig.from_mapping(separator_raw or raw.get("separator_config"))
        return cls(**values)

    def min_area_for_kind(self, kind: str) -> float:
        mapping = dict(self.per_kind_min_split_area_m2 or {})
        return float(mapping.get(str(kind), self.min_split_area_m2))

    def thickness_for_kind(self, kind: str) -> int:
        sep = self.separator or SeparatorAnchorConfig()
        if str(kind) == "doorway_virtual_cut":
            return int(sep.doorway_thickness_cells)
        if str(kind) in {"physical_wall_completion", "missed_scan_gap_closure", "short_unknown_gap_closure", "single_sided_wall_extension"}:
            return int(sep.wall_completion_thickness_cells)
        if str(kind) == "corridor_room_neck_cut":
            return int(sep.corridor_neck_thickness_cells)
        if str(kind) == "line_extension_door_neck":
            return int(sep.doorway_thickness_cells)
        return int(sep.thickness_cells)


def greedily_select_separators(
    candidates: Sequence[SeparatorCandidate],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    wall_candidate_clean: np.ndarray | None = None,
    corridor_skeleton: np.ndarray | None = None,
    resolution_m: float,
    config: TopologyTestConfig | Mapping[str, object] | None = None,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], np.ndarray, np.ndarray, dict]:
    cfg = config if isinstance(config, TopologyTestConfig) else TopologyTestConfig.from_mapping(config)
    free = np.asarray(free_clean, dtype=bool)
    accepted_map = np.zeros_like(free, dtype=bool)
    accepted: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    ordered = sorted(list(candidates), key=_candidate_sort_key)
    for candidate in ordered:
        ok, reason, candidate_map, metrics = evaluate_candidate(
            candidate,
            free_clean=free,
            unknown_clean=unknown_clean,
            wall_candidate_clean=wall_candidate_clean,
            corridor_skeleton=corridor_skeleton,
            current_separator_map=accepted_map,
            resolution_m=float(resolution_m),
            config=cfg,
        )
        candidate.debug.update(metrics)
        if ok:
            candidate.accepted = True
            candidate.reject_reason = ""
            accepted_map |= candidate_map
            accepted.append(candidate)
        else:
            candidate.accepted = False
            candidate.reject_reason = reason
            rejected.append(candidate)
    final_labels, _ = label_components(free & ~accepted_map, 4)
    final_labels = relabel_compact(final_labels)
    debug = {
        "topology_test_enabled": bool(cfg.enabled),
        "accepted_separator_count": int(len(accepted)),
        "rejected_separator_count": int(len(rejected)),
        "accepted_separator_kinds": _kind_counts(accepted),
        "rejected_separator_reasons": _reason_counts(rejected),
        "corridor_to_corridor_rejected_count": int(sum(1 for item in rejected if item.reject_reason in {"reject_split_main_corridor_axis", "reject_corridor_to_corridor_split"})),
        "corridor_to_room_accepted_count": int(sum(1 for item in accepted if (item.debug.get("corridor_side_classification") or {}).get("one_corridor_one_room"))),
    }
    return accepted, rejected, accepted_map.astype(bool), final_labels.astype(np.int32), debug


def evaluate_candidate(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray | None = None,
    wall_candidate_clean: np.ndarray | None = None,
    corridor_skeleton: np.ndarray | None = None,
    current_separator_map: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[bool, str, np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    wall = np.zeros_like(free, dtype=bool) if wall_candidate_clean is None else np.asarray(wall_candidate_clean, dtype=bool)
    unknown = np.zeros_like(free, dtype=bool) if unknown_clean is None else np.asarray(unknown_clean, dtype=bool)
    skeleton = np.zeros_like(free, dtype=bool) if corridor_skeleton is None else np.asarray(corridor_skeleton, dtype=bool)
    sep_cfg = config.separator or SeparatorAnchorConfig()
    candidate, anchor_debug = extend_candidate_to_anchors(
        candidate,
        free_clean=free,
        wall_candidate_clean=wall,
        unknown_clean=unknown,
        existing_separator_map=current,
        resolution_m=float(resolution_m),
        max_anchor_extension_m=float(sep_cfg.max_anchor_extension_m),
    )
    if bool(sep_cfg.require_two_anchors) and float(anchor_debug.get("anchor_score", 0.0)) < 1.0:
        reason = "reject_unanchored_separator" if float(anchor_debug.get("anchor_score", 0.0)) <= 0.0 else "reject_one_sided_unanchored"
        return False, reason, np.zeros_like(free, dtype=bool), anchor_debug
    thickness = config.thickness_for_kind(str(candidate.kind))
    candidate_map, cut_debug = _candidate_separator_map_for_topology(
        candidate,
        free_clean=free,
        current_separator_map=current,
        resolution_m=float(resolution_m),
        thickness_cells=thickness,
    )
    anchor_debug = {**anchor_debug, **cut_debug}
    if int(np.count_nonzero(candidate_map)) <= 0:
        return False, "reject_candidate_not_on_free_space", candidate_map, {**anchor_debug, "candidate_mask_cell_count": 0}
    if not bool(config.enabled):
        candidate.topology_gain_score = 1.0
        return True, "", candidate_map, {**anchor_debug, "topology_test_skipped": True}
    before = free & ~current
    after = before & ~candidate_map
    before_labels, before_count = label_components(before, 4)
    after_labels, after_count = label_components(after, 4)
    if int(after_count) <= int(before_count):
        return False, "reject_no_topology_gain", candidate_map, {**anchor_debug, "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)), "before_components": int(before_count), "after_components": int(after_count)}
    adjacent = dilate(candidate_map, 1) & after
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    if len(touched) < 2:
        return False, "reject_no_two_sides", candidate_map, {**anchor_debug, "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)), "touched_components": touched}
    component_info = []
    min_area_m2 = float(config.min_area_for_kind(str(candidate.kind)))
    min_cells = max(int(config.min_new_component_area_cells), int(round(min_area_m2 / max(float(resolution_m) ** 2, 1e-9))))
    distance_cells = ndimage.distance_transform_edt(before)
    for label in touched:
        mask = after_labels == int(label)
        info = component_metrics(mask, float(resolution_m), distance_cells)
        info["label"] = int(label)
        component_info.append(info)
    component_info.sort(key=lambda item: int(item["area_cells"]), reverse=True)
    main = component_info[:2]
    min_width = min(float(item.get("thickness_m", 0.0)) for item in main) if len(main) >= 2 else 0.0
    if len(main) < 2 or min(int(item["area_cells"]) for item in main) < min_cells or min_width < float(config.min_new_component_width_m):
        return False, "reject_tiny_split", candidate_map, {
            **anchor_debug,
            "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
            "components": component_info[:8],
            "min_split_cells": int(min_cells),
            "topology_min_area_m2": float(min_area_m2),
        }
    tiny_count = int(sum(1 for label in range(1, int(after_count) + 1) if int(np.count_nonzero(after_labels == label)) < min_cells))
    if tiny_count > int(config.max_tiny_fragment_count):
        candidate.fragmentation_penalty = float(tiny_count)
        return False, "reject_too_many_tiny_fragments", candidate_map, {**anchor_debug, "tiny_fragment_count": tiny_count, "components": component_info[:8]}
    side_class = classify_cut_sides_local(
        candidate_map,
        free_clean=before,
        after_labels=after_labels,
        touched_labels=touched,
        component_info=main,
        distance_transform=distance_cells,
        corridor_skeleton=skeleton,
        resolution_m=float(resolution_m),
    )
    both_corridor = bool(side_class.get("side_a_corridor_like") and side_class.get("side_b_corridor_like"))
    one_corridor = bool(side_class.get("side_a_corridor_like") or side_class.get("side_b_corridor_like"))
    one_room = bool(side_class.get("side_a_room_like") or side_class.get("side_b_room_like"))
    if bool(config.reject_corridor_split) and both_corridor:
        candidate.corridor_split_penalty = 1.0
        return False, "reject_split_main_corridor_axis", candidate_map, {**anchor_debug, "components": main, "corridor_side_classification": side_class, "both_sides_corridor_like": True}
    if bool(config.reject_open_living_room_internal_split) and bool(side_class.get("both_open_living_room_like", False)):
        return False, "reject_open_living_room_internal_split", candidate_map, {**anchor_debug, "components": main, "corridor_side_classification": side_class}
    if candidate.kind == "corridor_room_neck_cut" and not (one_corridor and one_room):
        return False, "reject_not_corridor_room_neck", candidate_map, {**anchor_debug, "components": main, "corridor_side_classification": side_class, "one_corridor": one_corridor, "one_room": one_room}
    candidate.topology_gain_score = float(max(0, int(after_count) - int(before_count)))
    candidate.corridor_preservation_score = 1.0 if not both_corridor else 0.0
    return True, "", candidate_map, {
        **anchor_debug,
        "before_components": int(before_count),
        "after_components": int(after_count),
        "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
        "touched_components": touched,
        "components": main,
        "corridor_side_classification": side_class,
        "topology_min_area_m2": float(min_area_m2),
        "topology_gain_score": float(candidate.topology_gain_score),
        "corridor_preservation_score": float(candidate.corridor_preservation_score),
    }


def classify_cut_sides_local(
    candidate_map: np.ndarray,
    *,
    free_clean: np.ndarray,
    after_labels: np.ndarray,
    touched_labels: Sequence[int],
    component_info: Sequence[dict],
    distance_transform: np.ndarray,
    corridor_skeleton: np.ndarray,
    resolution_m: float,
) -> dict:
    _ = free_clean, after_labels, touched_labels, distance_transform, resolution_m
    first = dict(component_info[0]) if len(component_info) > 0 else {}
    second = dict(component_info[1]) if len(component_info) > 1 else {}
    skel = np.asarray(corridor_skeleton, dtype=bool)
    around_cut = dilate(np.asarray(candidate_map, dtype=bool), 2)
    axis_crossed = bool(np.any(around_cut & skel))
    side_a_corridor = bool(first.get("corridor_like", False))
    side_b_corridor = bool(second.get("corridor_like", False))
    side_a_room = bool(first.get("room_like", False)) or (float(first.get("area_m2", 0.0)) >= 0.30 and not side_a_corridor)
    side_b_room = bool(second.get("room_like", False)) or (float(second.get("area_m2", 0.0)) >= 0.30 and not side_b_corridor)
    side_a_open = bool(float(first.get("median_width_m", 0.0)) >= 1.80 and float(first.get("area_m2", 0.0)) >= 2.00 and not side_a_corridor)
    side_b_open = bool(float(second.get("median_width_m", 0.0)) >= 1.80 and float(second.get("area_m2", 0.0)) >= 2.00 and not side_b_corridor)
    return {
        "side_a_corridor_like": bool(side_a_corridor),
        "side_b_corridor_like": bool(side_b_corridor),
        "side_a_room_like": bool(side_a_room),
        "side_b_room_like": bool(side_b_room),
        "one_corridor_one_room": bool((side_a_corridor and side_b_room) or (side_b_corridor and side_a_room)),
        "side_a_open_living_room_like": bool(side_a_open),
        "side_b_open_living_room_like": bool(side_b_open),
        "both_open_living_room_like": bool(side_a_open and side_b_open),
        "corridor_axis_crossed": bool(axis_crossed),
        "corridor_axis_preserved": bool(not (side_a_corridor and side_b_corridor and axis_crossed)),
    }


def _candidate_separator_map_for_topology(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    current_separator_map: np.ndarray,
    resolution_m: float,
    thickness_cells: int,
) -> tuple[np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    if str(candidate.kind) != "line_extension_door_neck":
        mask = separator_mask_for_candidate(candidate, free.shape, thickness_cells) & free & ~current
        return mask.astype(bool), {"centered_extension_cut_enabled": False}
    mask, debug = _centered_line_extension_free_run_mask(
        candidate,
        free_clean=free,
        current_separator_map=current,
        resolution_m=float(resolution_m),
        thickness_cells=int(thickness_cells),
    )
    return mask.astype(bool), debug


def _centered_line_extension_free_run_mask(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    current_separator_map: np.ndarray,
    resolution_m: float,
    thickness_cells: int,
) -> tuple[np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    cells = _ordered_line_cells(candidate.p0_rc, candidate.p1_rc, free.shape)
    valid = np.asarray([bool(free[int(r), int(c)] and not current[int(r), int(c)]) for r, c in cells], dtype=bool)
    if not np.any(valid):
        mask = separator_mask_for_candidate(candidate, free.shape, thickness_cells) & free & ~current
        return mask.astype(bool), {
            "centered_extension_cut_enabled": True,
            "centered_extension_cut_applied": False,
            "centered_extension_cut_reason": "no_free_cells_on_extension",
            "candidate_mask_cell_count": int(np.count_nonzero(mask)),
        }

    midpoint = 0.5 * (np.asarray(candidate.p0_rc, dtype=np.float32) + np.asarray(candidate.p1_rc, dtype=np.float32))
    valid_indices = np.flatnonzero(valid)
    distances = np.sum((cells[valid_indices].astype(np.float32) - midpoint[None, :]) ** 2, axis=1)
    center_idx = int(valid_indices[int(np.argmin(distances))])
    lo = center_idx
    while lo - 1 >= 0 and bool(valid[lo - 1]):
        lo -= 1
    hi = center_idx
    while hi + 1 < len(valid) and bool(valid[hi + 1]):
        hi += 1

    cut_cells = cells[lo : hi + 1]
    base = np.zeros_like(free, dtype=bool)
    base[cut_cells[:, 0], cut_cells[:, 1]] = True
    mask = dilate(base, int(thickness_cells)) if int(thickness_cells) > 0 else base
    mask = mask & free & ~current
    p0_cut = cut_cells[0].astype(np.float32)
    p1_cut = cut_cells[-1].astype(np.float32)
    original_p0 = np.asarray(candidate.p0_rc, dtype=np.float32).copy()
    original_p1 = np.asarray(candidate.p1_rc, dtype=np.float32).copy()
    candidate.p0_rc = p0_cut
    candidate.p1_rc = p1_cut
    candidate.length_m = float(max(1, int(np.max(np.abs(p1_cut - p0_cut)) + 1)) * float(resolution_m))
    debug = {
        "centered_extension_cut_enabled": True,
        "centered_extension_cut_applied": True,
        "centered_extension_cut_mode": "middle_contiguous_free_run",
        "centered_extension_cut_original_p0": [int(round(float(v))) for v in original_p0.tolist()],
        "centered_extension_cut_original_p1": [int(round(float(v))) for v in original_p1.tolist()],
        "centered_extension_cut_p0": [int(v) for v in p0_cut.astype(np.int32).tolist()],
        "centered_extension_cut_p1": [int(v) for v in p1_cut.astype(np.int32).tolist()],
        "centered_extension_cut_cell_count": int(len(cut_cells)),
        "centered_extension_cut_mask_cell_count": int(np.count_nonzero(mask)),
        "candidate_mask_cell_count": int(np.count_nonzero(mask)),
        "centered_extension_cut_trimmed": bool(
            not np.array_equal(np.rint(original_p0).astype(np.int32), p0_cut.astype(np.int32))
            or not np.array_equal(np.rint(original_p1).astype(np.int32), p1_cut.astype(np.int32))
        ),
    }
    candidate.debug.update(debug)
    return mask.astype(bool), debug


def _ordered_line_cells(p0_rc: np.ndarray, p1_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p0 = np.rint(np.asarray(p0_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(p1_rc, dtype=np.float32)).astype(np.int32)
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    steps = max(abs(dr), abs(dc))
    if steps <= 0:
        rows = np.asarray([int(p0[0])], dtype=np.int32)
        cols = np.asarray([int(p0[1])], dtype=np.int32)
    else:
        rows = np.rint(np.linspace(int(p0[0]), int(p1[0]), steps + 1)).astype(np.int32)
        cols = np.rint(np.linspace(int(p0[1]), int(p1[1]), steps + 1)).astype(np.int32)
    inside = (rows >= 0) & (rows < int(shape[0])) & (cols >= 0) & (cols < int(shape[1]))
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    if len(coords) <= 1:
        return coords.astype(np.int32)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _kind_order(kind: str) -> int:
    return {
        "physical_wall_completion": 0,
        "missed_scan_gap_closure": 1,
        "short_unknown_gap_closure": 2,
        "doorway_virtual_cut": 3,
        "line_extension_door_neck": 3,
        "single_sided_wall_extension": 4,
        "corridor_room_neck_cut": 5,
    }.get(str(kind), 99)


def _candidate_sort_key(item: SeparatorCandidate) -> tuple:
    kind = str(item.kind)
    if kind == "corridor_room_neck_cut":
        return (
            _kind_order(kind),
            -float(item.confidence),
            -float(item.length_m),
            int(item.candidate_id),
        )
    return (
        _kind_order(kind),
        -float(item.confidence),
        -float(item.debug.get("anchor_score", 0.0)),
        float(item.length_m),
        int(item.candidate_id),
    )


def _kind_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        out[str(candidate.kind)] = out.get(str(candidate.kind), 0) + 1
    return out


def _reason_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        reason = str(candidate.reject_reason or "")
        out[reason] = out.get(reason, 0) + 1
    return out
