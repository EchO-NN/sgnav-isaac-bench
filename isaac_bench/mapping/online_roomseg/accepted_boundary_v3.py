from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .separator_candidates import SeparatorCandidate
from .topology_tests import TopologyTestConfig, greedily_select_separators
from .utils import component_metrics, label_components, rasterize_line


@dataclass
class AcceptedBoundaryV3Result:
    accepted_virtual_boundary_map: np.ndarray
    candidate_separator_map: np.ndarray
    rejected_separator_map: np.ndarray
    topology_reject_reason_map: np.ndarray
    accepted_candidates: list[SeparatorCandidate]
    rejected_candidates: list[SeparatorCandidate]
    debug: dict = field(default_factory=dict)


def build_accepted_boundary_v3(
    candidates: Sequence[SeparatorCandidate],
    *,
    roomseg_free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    structural_wall_clean: np.ndarray,
    corridor_skeleton: np.ndarray | None,
    resolution_m: float,
    config: TopologyTestConfig | Mapping[str, object] | None,
) -> AcceptedBoundaryV3Result:
    accepted, rejected, accepted_map, _labels, topology_debug = greedily_select_separators(
        list(candidates),
        free_clean=roomseg_free_clean,
        unknown_clean=unknown_clean,
        wall_candidate_clean=structural_wall_clean,
        corridor_skeleton=corridor_skeleton,
        resolution_m=float(resolution_m),
        config=config,
    )
    candidate_map = _rasterize_many(candidates, np.asarray(roomseg_free_clean, dtype=bool).shape)
    rejected_map = _rasterize_many(rejected, np.asarray(roomseg_free_clean, dtype=bool).shape)
    reason_map = _reject_reason_map(rejected, np.asarray(roomseg_free_clean, dtype=bool).shape)
    return AcceptedBoundaryV3Result(
        accepted_virtual_boundary_map=accepted_map.astype(bool),
        candidate_separator_map=candidate_map.astype(bool),
        rejected_separator_map=rejected_map.astype(bool),
        topology_reject_reason_map=reason_map.astype(np.int32),
        accepted_candidates=list(accepted),
        rejected_candidates=list(rejected),
        debug={
            "algorithm": "roomseg_evidence_line_closure_v3",
            "candidate_count": int(len(candidates)),
            "accepted_closure_count": int(len(accepted)),
            "rejected_closure_count": int(len(rejected)),
            **dict(topology_debug),
        },
    )


def generate_mandatory_rescue_candidates(
    *,
    roomseg_free_clean: np.ndarray,
    structural_wall_clean: np.ndarray,
    filtered_lines: Sequence[object] | None,
    wall_runs: Sequence[object] | None,
    accepted_closure_count: int,
    resolution_m: float,
    config: Mapping[str, object] | object | None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    root = _root_config(config)
    cfg = _section(root, "separator_v3")
    free = np.asarray(roomseg_free_clean, dtype=bool)
    wall = np.asarray(structural_wall_clean, dtype=bool)
    if not bool(cfg.get("mandatory_rescue_enabled", True)):
        return [], {"mandatory_rescue_triggered": False, "reason": "disabled"}
    largest_area_m2 = _largest_free_area_m2(free, float(resolution_m))
    wall_run_count = int(len(filtered_lines or [])) + int(len(wall_runs or []))
    trigger = bool(
        largest_area_m2 >= float(cfg.get("rescue_largest_free_area_m2", 8.0))
        and int(accepted_closure_count) == 0
        and wall_run_count >= int(cfg.get("rescue_min_wall_run_count", 3))
    )
    if not trigger:
        return [], {
            "mandatory_rescue_triggered": False,
            "largest_free_area_m2": float(largest_area_m2),
            "accepted_closure_count": int(accepted_closure_count),
            "wall_run_count": int(wall_run_count),
        }

    max_candidates = max(1, int(cfg.get("rescue_max_candidates", 80)))
    candidates: list[SeparatorCandidate] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    cid = int(start_id)
    line_items = list(filtered_lines or [])
    for line in line_items:
        for endpoint_name, point in (("p0", getattr(line, "p0_rc", None)), ("p1", getattr(line, "p1_rc", None))):
            if point is None:
                continue
            for p0, p1, cells, axis in _rescue_cuts_from_endpoint(
                point,
                line_theta=float(getattr(line, "theta", 0.0)),
                free=free,
                wall=wall,
                resolution_m=float(resolution_m),
                max_length_m=float(cfg.get("rescue_max_length_m", 2.40)),
                min_length_m=float(cfg.get("rescue_min_length_m", 0.45)),
            ):
                key = _candidate_key(p0, p1)
                if key in seen:
                    continue
                seen.add(key)
                confidence = float(np.clip(0.55 + 0.20 * float(getattr(line, "confidence", 0.0)), 0.0, 1.0))
                candidate = SeparatorCandidate(
                    candidate_id=cid,
                    kind="mandatory_rescue_wall_endpoint_cut",
                    p0_rc=np.asarray(p0, dtype=np.float32),
                    p1_rc=np.asarray(p1, dtype=np.float32),
                    theta=float(0.0 if axis == "horizontal" else np.pi / 2.0),
                    length_m=float(max(1, len(cells)) * float(resolution_m)),
                    confidence=confidence,
                    source_segment_ids=[int(getattr(line, "line_id", 0))],
                    wall_support_score=float(np.clip(getattr(line, "confidence", 0.5), 0.0, 1.0)),
                    free_gap_score=1.0,
                    doorway_score=0.65,
                    axis_alignment_score=1.0,
                    debug={
                        "candidate_source": "mandatory_rescue_wall_endpoint_cut",
                        "source_endpoint": str(endpoint_name),
                        "axis": str(axis),
                        "mask_cells_rc": [[int(r), int(c)] for r, c in cells.tolist()],
                        "mandatory_rescue": True,
                    },
                )
                candidates.append(candidate)
                cid += 1
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if len(candidates) < max_candidates:
        cid = _add_component_corner_rescue_candidates(
            candidates,
            seen=seen,
            wall=wall,
            free=free,
            resolution_m=float(resolution_m),
            start_id=cid,
            max_candidates=max_candidates,
        )
        _ = cid
    return candidates, {
        "mandatory_rescue_triggered": True,
        "largest_free_area_m2": float(largest_area_m2),
        "accepted_closure_count": int(accepted_closure_count),
        "wall_run_count": int(wall_run_count),
        "rescue_candidate_count": int(len(candidates)),
        "rescue_max_candidates": int(max_candidates),
    }


def _rescue_cuts_from_endpoint(
    point: object,
    *,
    line_theta: float,
    free: np.ndarray,
    wall: np.ndarray,
    resolution_m: float,
    max_length_m: float,
    min_length_m: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    p = np.rint(np.asarray(point, dtype=np.float32)).astype(np.int32)
    if p.shape[0] < 2:
        return []
    horizontal_wall = abs(np.cos(float(line_theta))) >= abs(np.sin(float(line_theta)))
    axis = "vertical" if horizontal_wall else "horizontal"
    max_steps = max(1, int(round(float(max_length_m) / max(float(resolution_m), 1e-9))))
    min_cells = max(2, int(round(float(min_length_m) / max(float(resolution_m), 1e-9))))
    direction = np.asarray([1, 0], dtype=np.int32) if axis == "vertical" else np.asarray([0, 1], dtype=np.int32)
    cells = _contiguous_free_run_around(p, direction, free, wall, max_steps=max_steps)
    if len(cells) < min_cells:
        return []
    p0 = cells[0].astype(np.float32)
    p1 = cells[-1].astype(np.float32)
    return [(p0, p1, cells, axis)]


def _contiguous_free_run_around(
    point: np.ndarray,
    direction: np.ndarray,
    free: np.ndarray,
    wall: np.ndarray,
    *,
    max_steps: int,
) -> np.ndarray:
    start = np.asarray(point, dtype=np.int32)
    candidates: list[np.ndarray] = [start]
    for sign in (-1, 1):
        for step in range(1, int(max_steps) + 1):
            rc = start + np.asarray(direction, dtype=np.int32) * int(sign * step)
            if not _inside(rc, free.shape):
                break
            r, c = int(rc[0]), int(rc[1])
            if bool(wall[r, c]):
                break
            if not bool(free[r, c]):
                if step == 1:
                    continue
                break
            if sign < 0:
                candidates.insert(0, rc.copy())
            else:
                candidates.append(rc.copy())
    rows = []
    for rc in candidates:
        if _inside(rc, free.shape) and bool(free[int(rc[0]), int(rc[1])]):
            rows.append(rc.copy())
    if not rows:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def _add_component_corner_rescue_candidates(
    out: list[SeparatorCandidate],
    *,
    seen: set[tuple[tuple[int, int], tuple[int, int]]],
    wall: np.ndarray,
    free: np.ndarray,
    resolution_m: float,
    start_id: int,
    max_candidates: int,
) -> int:
    labels, count = label_components(wall, 8)
    cid = int(start_id)
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        metrics = component_metrics(comp, float(resolution_m))
        if float(metrics.get("length_m", 0.0)) < 0.50:
            continue
        bbox = metrics.get("bbox") or []
        if len(bbox) != 4:
            continue
        r0, c0, r1, c1 = [int(v) for v in bbox]
        points = [(r0, c0), (r0, c1 - 1), (r1 - 1, c0), (r1 - 1, c1 - 1)]
        for point in points:
            for axis in ("horizontal", "vertical"):
                direction = np.asarray([0, 1], dtype=np.int32) if axis == "horizontal" else np.asarray([1, 0], dtype=np.int32)
                cells = _contiguous_free_run_around(np.asarray(point, dtype=np.int32), direction, free, wall, max_steps=48)
                if len(cells) < max(2, int(round(0.45 / max(float(resolution_m), 1e-9)))):
                    continue
                p0 = cells[0].astype(np.float32)
                p1 = cells[-1].astype(np.float32)
                key = _candidate_key(p0, p1)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    SeparatorCandidate(
                        candidate_id=cid,
                        kind="mandatory_rescue_wall_endpoint_cut",
                        p0_rc=p0,
                        p1_rc=p1,
                        theta=float(0.0 if axis == "horizontal" else np.pi / 2.0),
                        length_m=float(max(1, len(cells)) * float(resolution_m)),
                        confidence=0.58,
                        source_segment_ids=[int(idx)],
                        wall_support_score=0.65,
                        free_gap_score=1.0,
                        doorway_score=0.60,
                        axis_alignment_score=1.0,
                        debug={
                            "candidate_source": "mandatory_rescue_wall_corner_cut",
                            "axis": str(axis),
                            "mask_cells_rc": [[int(r), int(c)] for r, c in cells.tolist()],
                            "mandatory_rescue": True,
                        },
                    )
                )
                cid += 1
                if len(out) >= int(max_candidates):
                    return cid
    return cid


def _largest_free_area_m2(free: np.ndarray, resolution_m: float) -> float:
    labels, count = label_components(np.asarray(free, dtype=bool), 4)
    largest = 0
    for idx in range(1, int(count) + 1):
        largest = max(largest, int(np.count_nonzero(labels == idx)))
    return float(largest) * float(resolution_m) ** 2


def _rasterize_many(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= candidate.mask(shape)
    return out.astype(bool)


def _reject_reason_map(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int32)
    code_by_reason: dict[str, int] = {}
    next_code = 1
    for candidate in candidates:
        reason = str(candidate.reject_reason or "rejected")
        if reason not in code_by_reason:
            code_by_reason[reason] = next_code
            next_code += 1
        out[candidate.mask(shape)] = int(code_by_reason[reason])
    return out.astype(np.int32)


def _candidate_key(p0: np.ndarray, p1: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    a = tuple(int(v) for v in np.rint(p0).astype(np.int32).tolist())
    b = tuple(int(v) for v in np.rint(p1).astype(np.int32).tolist())
    return (a, b) if a <= b else (b, a)


def _root_config(config: Mapping[str, object] | object | None) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    out: dict[str, object] = {}
    if hasattr(config, "separator_v3"):
        out["separator_v3"] = getattr(config, "separator_v3")
    return out


def _section(config: Mapping[str, object], name: str) -> dict[str, object]:
    section = dict(config).get(name)
    return dict(section) if isinstance(section, Mapping) else {}


def _inside(point: np.ndarray, shape: tuple[int, int]) -> bool:
    return 0 <= int(point[0]) < int(shape[0]) and 0 <= int(point[1]) < int(shape[1])
