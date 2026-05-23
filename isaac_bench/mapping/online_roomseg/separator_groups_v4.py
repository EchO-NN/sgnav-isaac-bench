from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .separator_candidates import SeparatorCandidate, score_separator_candidate
from .topology_tests import TopologyTestConfig, _candidate_sort_key, _kind_counts, _reason_counts, evaluate_candidate
from .utils import component_metrics, dilate, label_components, relabel_compact


@dataclass
class SeparatorGroup:
    group_id: int
    candidate_ids: list[int]
    mask: np.ndarray
    score: float
    reason: str = ""
    accepted: bool = False
    debug: dict = field(default_factory=dict)


@dataclass
class SeparatorGroupV4Config:
    enabled: bool = True
    pending_score_min: float = 0.35
    accept_score_min: float = 0.55
    max_group_size: int = 4
    max_groups_per_component: int = 64
    max_candidate_distance_m: float = 1.20
    min_group_topology_gain: int = 1
    min_side_area_m2: float = 0.30
    accept_no_single_topology_gain_only: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "SeparatorGroupV4Config":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


def greedily_select_separator_groups(
    candidates: Sequence[SeparatorCandidate],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    wall_candidate_clean: np.ndarray | None = None,
    corridor_skeleton: np.ndarray | None = None,
    corridor_axis: np.ndarray | None = None,
    resolution_m: float,
    topology_config: TopologyTestConfig | Mapping[str, object] | None = None,
    group_config: SeparatorGroupV4Config | Mapping[str, object] | None = None,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], np.ndarray, np.ndarray, dict]:
    topo_cfg = topology_config if isinstance(topology_config, TopologyTestConfig) else TopologyTestConfig.from_mapping(topology_config)
    group_cfg = group_config if isinstance(group_config, SeparatorGroupV4Config) else SeparatorGroupV4Config.from_mapping(group_config)
    free = np.asarray(free_clean, dtype=bool)
    accepted_map = np.zeros_like(free, dtype=bool)
    accepted: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    pending: list[SeparatorCandidate] = []
    pending_maps: dict[int, np.ndarray] = {}
    pending_metrics: dict[int, dict] = {}
    skeleton = _merged_corridor_axis(corridor_skeleton, corridor_axis, free.shape)

    for candidate in sorted(list(candidates), key=_candidate_sort_key):
        ok, reason, candidate_map, metrics = evaluate_candidate(
            candidate,
            free_clean=free,
            unknown_clean=unknown_clean,
            wall_candidate_clean=wall_candidate_clean,
            corridor_skeleton=skeleton,
            current_separator_map=accepted_map,
            resolution_m=float(resolution_m),
            config=topo_cfg,
        )
        candidate.debug.update(metrics)
        if ok:
            candidate.accepted = True
            candidate.reject_reason = ""
            candidate.debug["accepted_as"] = "single"
            accepted_map |= candidate_map
            accepted.append(candidate)
            continue
        if bool(group_cfg.enabled) and _candidate_can_pending(candidate, reason, metrics, group_cfg):
            candidate.accepted = False
            candidate.reject_reason = "pending_no_single_topology_gain"
            candidate.debug["pending_group_reason"] = str(reason)
            pending.append(candidate)
            pending_maps[int(candidate.candidate_id)] = candidate_map.astype(bool)
            pending_metrics[int(candidate.candidate_id)] = dict(metrics)
        else:
            candidate.accepted = False
            candidate.reject_reason = reason
            rejected.append(candidate)

    accepted_groups: list[SeparatorGroup] = []
    rejected_groups: list[SeparatorGroup] = []
    if bool(group_cfg.enabled) and pending:
        group_candidates = _candidate_groups(
            pending,
            free.shape,
            resolution_m=float(resolution_m),
            max_size=int(group_cfg.max_group_size),
            max_groups=int(group_cfg.max_groups_per_component),
            max_distance_m=float(group_cfg.max_candidate_distance_m),
        )
        used_ids: set[int] = set()
        for group_id, group_items in enumerate(group_candidates, start=1):
            ids = [int(item.candidate_id) for item in group_items]
            if any(item_id in used_ids for item_id in ids):
                continue
            group_map = np.zeros_like(free, dtype=bool)
            for item in group_items:
                item_map = pending_maps.get(int(item.candidate_id), item.mask(free.shape))
                group_map |= np.asarray(item_map, dtype=bool)
            group_map &= free & ~accepted_map
            group_ok, group_reason, group_debug = _evaluate_group_map(
                group_items,
                group_map=group_map,
                free_clean=free,
                current_separator_map=accepted_map,
                corridor_axis=skeleton,
                resolution_m=float(resolution_m),
                topology_config=topo_cfg,
                group_config=group_cfg,
            )
            group = SeparatorGroup(
                group_id=int(group_id),
                candidate_ids=ids,
                mask=group_map.astype(bool),
                score=float(group_debug.get("group_score", 0.0)),
                reason=str(group_reason),
                accepted=bool(group_ok),
                debug=group_debug,
            )
            if group_ok:
                accepted_groups.append(group)
                accepted_map |= group_map
                used_ids.update(ids)
                for item in group_items:
                    item.accepted = True
                    item.reject_reason = ""
                    item.debug.update(
                        {
                            "accepted_as": "separator_group_v4",
                            "separator_group_id": int(group_id),
                            "separator_group_size": int(len(group_items)),
                            "separator_group_score": float(group.score),
                        }
                    )
                    accepted.append(item)
            else:
                rejected_groups.append(group)
        for item in pending:
            if int(item.candidate_id) in used_ids:
                continue
            item.accepted = False
            item.reject_reason = "reject_no_topology_gain_group_failed"
            item.debug["separator_group_failed"] = True
            rejected.append(item)

    final_labels, _ = label_components(free & ~accepted_map, int(topo_cfg.connectivity))
    final_labels = relabel_compact(final_labels)
    debug = {
        "topology_test_enabled": bool(topo_cfg.enabled),
        "separator_group_v4_enabled": bool(group_cfg.enabled),
        "accepted_separator_count": int(len(accepted)),
        "rejected_separator_count": int(len(rejected)),
        "pending_no_single_topology_gain_count": int(len(pending)),
        "accepted_group_count": int(len(accepted_groups)),
        "rejected_group_count": int(len(rejected_groups)),
        "accepted_separator_kinds": _kind_counts(accepted),
        "rejected_separator_reasons": _reason_counts(rejected),
        "accepted_groups": [_group_to_debug(group) for group in accepted_groups[:128]],
        "rejected_groups": [_group_to_debug(group) for group in rejected_groups[:128]],
        "corridor_to_corridor_rejected_count": int(sum(1 for item in rejected if item.reject_reason in {"reject_split_main_corridor_axis", "reject_corridor_to_corridor_split"})),
        "corridor_to_room_accepted_count": int(sum(1 for item in accepted if (item.debug.get("corridor_side_classification") or {}).get("one_corridor_one_room"))),
    }
    return accepted, rejected, accepted_map.astype(bool), final_labels.astype(np.int32), debug


def _candidate_can_pending(
    candidate: SeparatorCandidate,
    reason: str,
    metrics: Mapping[str, object],
    cfg: SeparatorGroupV4Config,
) -> bool:
    if str(reason) not in {"reject_no_topology_gain", "reject_no_two_sides", "reject_low_separator_final_score"}:
        return False
    mask_cells = int(metrics.get("candidate_mask_cell_count", 0) or 0)
    if mask_cells <= 0:
        return False
    proxy_score = max(
        float(candidate.final_score),
        float(candidate.confidence) * 0.35
        + float(candidate.anchor_score) * 0.35
        + float(candidate.wall_support_score) * 0.20
        + float(candidate.doorway_score) * 0.10,
    )
    if str(reason) == "reject_low_separator_final_score":
        proxy_score = max(proxy_score, float(metrics.get("final_score", 0.0) or 0.0))
    candidate.debug["pending_proxy_score"] = float(proxy_score)
    return bool(proxy_score >= float(cfg.pending_score_min))


def _candidate_groups(
    pending: Sequence[SeparatorCandidate],
    shape: tuple[int, int],
    *,
    resolution_m: float,
    max_size: int,
    max_groups: int,
    max_distance_m: float,
) -> list[list[SeparatorCandidate]]:
    items = list(pending)
    if not items:
        return []
    max_size = max(2, int(max_size))
    max_distance_cells = float(max_distance_m) / max(float(resolution_m), 1e-6)
    centers = {int(item.candidate_id): _candidate_center(item, shape) for item in items}
    groups: list[list[SeparatorCandidate]] = []
    for size in range(2, min(max_size, len(items)) + 1):
        for combo in combinations(items, size):
            if len(groups) >= int(max_groups):
                return groups
            ids = [int(item.candidate_id) for item in combo]
            close = True
            for i, left_id in enumerate(ids):
                for right_id in ids[i + 1 :]:
                    if float(np.linalg.norm(centers[left_id] - centers[right_id])) > float(max_distance_cells):
                        close = False
                        break
                if not close:
                    break
            if not close:
                continue
            groups.append(list(combo))
    groups.sort(key=lambda group: (-float(np.mean([item.confidence for item in group])), len(group), [int(item.candidate_id) for item in group]))
    return groups


def _evaluate_group_map(
    candidates: Sequence[SeparatorCandidate],
    *,
    group_map: np.ndarray,
    free_clean: np.ndarray,
    current_separator_map: np.ndarray,
    corridor_axis: np.ndarray,
    resolution_m: float,
    topology_config: TopologyTestConfig,
    group_config: SeparatorGroupV4Config,
) -> tuple[bool, str, dict]:
    free = np.asarray(free_clean, dtype=bool)
    current = np.asarray(current_separator_map, dtype=bool)
    candidate_map = np.asarray(group_map, dtype=bool) & free & ~current
    if int(np.count_nonzero(candidate_map)) <= 0:
        return False, "reject_group_empty", {"candidate_mask_cell_count": 0}
    before = free & ~current
    after = before & ~candidate_map
    before_labels, before_count = label_components(before, int(topology_config.connectivity))
    after_labels, after_count = label_components(after, int(topology_config.connectivity))
    topology_gain = int(after_count) - int(before_count)
    if topology_gain < int(group_config.min_group_topology_gain):
        return False, "reject_group_no_topology_gain", {
            "before_components": int(before_count),
            "after_components": int(after_count),
            "topology_gain": int(topology_gain),
            "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
        }
    adjacent = dilate(candidate_map, 1) & after
    touched = sorted({int(v) for v in np.unique(after_labels[adjacent]) if int(v) > 0})
    if len(touched) < 2:
        return False, "reject_group_no_two_sides", {
            "before_components": int(before_count),
            "after_components": int(after_count),
            "touched_components": touched,
        }
    distance_cells = ndimage.distance_transform_edt(before)
    components = []
    min_cells = max(
        int(topology_config.min_new_component_area_cells),
        int(round(float(group_config.min_side_area_m2) / max(float(resolution_m) ** 2, 1e-9))),
    )
    for label in touched:
        info = component_metrics(after_labels == int(label), float(resolution_m), distance_cells)
        info["label"] = int(label)
        components.append(info)
    components.sort(key=lambda item: int(item["area_cells"]), reverse=True)
    main = components[:2]
    if len(main) < 2 or min(int(item.get("area_cells", 0)) for item in main) < int(min_cells):
        return False, "reject_group_tiny_split", {
            "before_components": int(before_count),
            "after_components": int(after_count),
            "components": main,
            "min_split_cells": int(min_cells),
        }
    corridor_cross = bool(np.any(dilate(candidate_map, 1) & np.asarray(corridor_axis, dtype=bool)))
    both_corridor_like = bool(main[0].get("corridor_like", False) and main[1].get("corridor_like", False))
    if bool(topology_config.reject_corridor_split) and both_corridor_like and corridor_cross:
        return False, "reject_group_split_main_corridor_axis", {
            "components": main,
            "corridor_axis_crossed": bool(corridor_cross),
        }
    avg_score = float(np.mean([max(float(item.final_score), score_separator_candidate(item, topology_config.separator_scoring)) for item in candidates]))
    topology_score = float(np.clip(topology_gain / 2.0, 0.0, 1.0))
    group_score = float(np.clip(0.60 * avg_score + 0.40 * topology_score, 0.0, 1.0))
    if group_score < float(group_config.accept_score_min):
        return False, "reject_group_low_score", {
            "group_score": float(group_score),
            "accept_score_min": float(group_config.accept_score_min),
            "avg_candidate_score": float(avg_score),
            "topology_gain_score": float(topology_score),
            "components": main,
        }
    return True, "", {
        "before_components": int(before_count),
        "after_components": int(after_count),
        "topology_gain": int(topology_gain),
        "candidate_mask_cell_count": int(np.count_nonzero(candidate_map)),
        "touched_components": touched,
        "components": main,
        "corridor_axis_crossed": bool(corridor_cross),
        "avg_candidate_score": float(avg_score),
        "topology_gain_score": float(topology_score),
        "group_score": float(group_score),
    }


def _candidate_center(candidate: SeparatorCandidate, shape: tuple[int, int]) -> np.ndarray:
    mask = candidate.mask(shape)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0.5 * (np.asarray(candidate.p0_rc, dtype=np.float32) + np.asarray(candidate.p1_rc, dtype=np.float32))
    return np.mean(coords.astype(np.float32), axis=0)


def _merged_corridor_axis(corridor_skeleton: np.ndarray | None, corridor_axis: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if corridor_skeleton is not None:
        arr = np.asarray(corridor_skeleton, dtype=bool)
        if arr.shape == shape:
            out |= arr
    if corridor_axis is not None:
        arr = np.asarray(corridor_axis, dtype=bool)
        if arr.shape == shape:
            out |= arr
    return out.astype(bool)


def _group_to_debug(group: SeparatorGroup) -> dict:
    return {
        "group_id": int(group.group_id),
        "candidate_ids": [int(v) for v in group.candidate_ids],
        "accepted": bool(group.accepted),
        "score": float(group.score),
        "reason": str(group.reason),
        **{key: value for key, value in dict(group.debug).items() if str(key) != "mask"},
    }
