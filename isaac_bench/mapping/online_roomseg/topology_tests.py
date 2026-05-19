from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorCandidate
from .utils import component_metrics, conn, dilate, label_components, relabel_compact


@dataclass
class TopologyTestConfig:
    enabled: bool = True
    min_split_area_m2: float = 1.0
    max_tiny_fragment_count: int = 3
    reject_corridor_split: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "TopologyTestConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


def greedily_select_separators(
    candidates: Sequence[SeparatorCandidate],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
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
    }
    return accepted, rejected, accepted_map.astype(bool), final_labels.astype(np.int32), debug


def evaluate_candidate(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    current_separator_map: np.ndarray,
    resolution_m: float,
    config: TopologyTestConfig,
) -> tuple[bool, str, np.ndarray, dict]:
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    candidate_map = candidate.mask(free.shape) & free & ~current
    if int(np.count_nonzero(candidate_map)) <= 0:
        return False, "reject_candidate_not_on_free_space", candidate_map, {}
    if not bool(config.enabled):
        candidate.topology_gain_score = 1.0
        return True, "", candidate_map, {"topology_test_skipped": True}
    before = free & ~current
    after = before & ~candidate_map
    before_labels, before_count = label_components(before, 4)
    after_labels, after_count = label_components(after, 4)
    if int(after_count) <= int(before_count):
        return False, "reject_no_topology_gain", candidate_map, {"before_components": int(before_count), "after_components": int(after_count)}
    adjacent = dilate(candidate_map, 1) & after
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    if len(touched) < 2:
        return False, "reject_no_two_sides", candidate_map, {"touched_components": touched}
    component_info = []
    min_cells = max(1, int(round(float(config.min_split_area_m2) / max(float(resolution_m) ** 2, 1e-9))))
    distance_cells = ndimage.distance_transform_edt(before)
    for label in touched:
        mask = after_labels == int(label)
        info = component_metrics(mask, float(resolution_m), distance_cells)
        info["label"] = int(label)
        component_info.append(info)
    component_info.sort(key=lambda item: int(item["area_cells"]), reverse=True)
    main = component_info[:2]
    if len(main) < 2 or min(int(item["area_cells"]) for item in main) < min_cells:
        return False, "reject_tiny_split", candidate_map, {"components": component_info[:8], "min_split_cells": int(min_cells)}
    tiny_count = int(sum(1 for label in range(1, int(after_count) + 1) if int(np.count_nonzero(after_labels == label)) < min_cells))
    if tiny_count > int(config.max_tiny_fragment_count):
        candidate.fragmentation_penalty = float(tiny_count)
        return False, "reject_too_many_tiny_fragments", candidate_map, {"tiny_fragment_count": tiny_count, "components": component_info[:8]}
    both_corridor = bool(main[0].get("corridor_like") and main[1].get("corridor_like"))
    one_corridor = bool(main[0].get("corridor_like") or main[1].get("corridor_like"))
    one_room = bool(main[0].get("room_like") or main[1].get("room_like"))
    if bool(config.reject_corridor_split) and both_corridor:
        candidate.corridor_split_penalty = 1.0
        return False, "reject_split_main_corridor_axis", candidate_map, {"components": main, "both_sides_corridor_like": True}
    if candidate.kind == "corridor_room_neck_cut" and not (one_corridor and one_room):
        return False, "reject_not_corridor_room_neck", candidate_map, {"components": main, "one_corridor": one_corridor, "one_room": one_room}
    candidate.topology_gain_score = float(max(0, int(after_count) - int(before_count)))
    candidate.corridor_preservation_score = 1.0 if not both_corridor else 0.0
    return True, "", candidate_map, {
        "before_components": int(before_count),
        "after_components": int(after_count),
        "touched_components": touched,
        "components": main,
        "topology_gain_score": float(candidate.topology_gain_score),
        "corridor_preservation_score": float(candidate.corridor_preservation_score),
    }


def _kind_order(kind: str) -> int:
    return {
        "physical_wall_completion": 0,
        "doorway_virtual_cut": 1,
        "corridor_room_neck_cut": 2,
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
