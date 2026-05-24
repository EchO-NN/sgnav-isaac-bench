from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass
class DoorPatternConfig:
    enabled: bool = True
    min_free_run_cells: int = 3
    min_occupied_run_cells: int = 3
    min_unknown_run_cells: int = 1
    scan_dr: int = -1
    scan_dc: int = 0
    turn_dirs: tuple[tuple[int, int], ...] = ((0, -1), (0, 1))
    allow_diagonal_turns: bool = False
    rule_a_turn_probe_max_cells: int | None = None
    unknown_tail_padding_cells: int = 8
    reject_unknown_below: bool = True
    allow_start_lift: bool = False
    max_start_lift_cells: int = 0
    door_cut_thickness_cells: int = 1
    door_cut_lateral_radius_cells: int = 1
    merge_nearby_door_cells: int = 2
    max_candidates: int = 4096

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorPatternConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        if "turn_dirs" in raw:
            raw["turn_dirs"] = tuple((int(item[0]), int(item[1])) for item in list(raw["turn_dirs"] or []))
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorPatternCandidate:
    candidate_id: int
    pattern_type: str
    start_rc: tuple[int, int]
    free_cells: list[tuple[int, int]] = field(default_factory=list)
    occupied_cells: list[tuple[int, int]] = field(default_factory=list)
    unknown_cells: list[tuple[int, int]] = field(default_factory=list)
    turn_rc: tuple[int, int] | None = None
    door_cut_cells: list[tuple[int, int]] = field(default_factory=list)
    accepted: bool = False
    reject_reason: str | None = None

    def serialize(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "pattern_type": str(self.pattern_type),
            "start_rc": [int(self.start_rc[0]), int(self.start_rc[1])],
            "turn_rc": None if self.turn_rc is None else [int(self.turn_rc[0]), int(self.turn_rc[1])],
            "free_len": int(len(self.free_cells)),
            "occupied_len": int(len(self.occupied_cells)),
            "unknown_len": int(len(self.unknown_cells)),
            "door_cut_len": int(len(self.door_cut_cells)),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
        }


@dataclass
class DoorPatternDetectionResult:
    detected_door_mask: np.ndarray
    door_cut_mask: np.ndarray
    pattern_type_map: np.ndarray
    candidates: list[DoorPatternCandidate]
    debug: dict[str, Any]


def detect_pre_extension_doors(
    *,
    free_mask: np.ndarray,
    occupied_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: DoorPatternConfig | Mapping[str, object] | None = None,
    observed_mask: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> DoorPatternDetectionResult:
    """Detect strict occupancy-pattern doors before roomseg wall extension."""

    cfg = config if isinstance(config, DoorPatternConfig) else DoorPatternConfig.from_mapping(config)
    free = np.asarray(free_mask, dtype=bool).copy()
    occ = np.asarray(occupied_mask, dtype=bool).copy()
    unk = np.asarray(unknown_mask, dtype=bool).copy()

    if free.shape != occ.shape or free.shape != unk.shape:
        raise ValueError("door pattern detector masks must have same HxW shape")

    detected_door_mask = np.zeros_like(free, dtype=bool)
    door_cut_mask = np.zeros_like(free, dtype=bool)
    pattern_type_map = np.zeros(free.shape, dtype=np.uint8)
    candidates: list[DoorPatternCandidate] = []

    if not bool(cfg.enabled):
        return DoorPatternDetectionResult(
            detected_door_mask=detected_door_mask,
            door_cut_mask=door_cut_mask,
            pattern_type_map=pattern_type_map,
            candidates=[],
            debug={
                "pre_extension_door_detection_enabled": False,
                "pre_extension_door_num_candidates": 0,
                "pre_extension_door_num_accepted": 0,
            },
        )

    rois = resolve_detection_rois(free=free, occ=occ, unk=unk, observed_mask=observed_mask, roi=roi, cfg=cfg)
    next_id = 1

    for current_roi in rois:
        min_r, min_c, max_r, max_c = current_roi
        _ = min_r
        bottom_r = max_r
        for c in range(min_c, max_c + 1):
            start = resolve_bottom_start(
                free=free,
                occ=occ,
                unk=unk,
                bottom_r=bottom_r,
                col=c,
                roi=current_roi,
                cfg=cfg,
            )
            if start is None:
                continue

            cand_b = try_match_rule_b_free_occupied_unknown(
                candidate_id=next_id,
                start=start,
                free=free,
                occ=occ,
                unk=unk,
                roi=current_roi,
                cfg=cfg,
            )
            if cand_b.accepted:
                candidates.append(cand_b)
                mark_candidate(cand_b, detected_door_mask, door_cut_mask, pattern_type_map, 2, free)
                next_id += 1
                if next_id > int(cfg.max_candidates):
                    break
                continue

            cand_a = try_match_rule_a_free_turn_occupied(
                candidate_id=next_id,
                start=start,
                free=free,
                occ=occ,
                unk=unk,
                roi=current_roi,
                cfg=cfg,
            )
            if cand_a.accepted:
                candidates.append(cand_a)
                mark_candidate(cand_a, detected_door_mask, door_cut_mask, pattern_type_map, 1, free)
                next_id += 1
                if next_id > int(cfg.max_candidates):
                    break

    door_cut_mask = merge_nearby_door_cuts(door_cut_mask, cfg) & free
    debug = {
        "pre_extension_door_detection_enabled": True,
        "pre_extension_door_num_candidates": int(len(candidates)),
        "pre_extension_door_num_accepted": int(sum(c.accepted for c in candidates)),
        "pre_extension_door_rule_a_count": int(
            sum(c.accepted and c.pattern_type == "free_turn_occupied" for c in candidates)
        ),
        "pre_extension_door_rule_b_count": int(
            sum(c.accepted and c.pattern_type == "free_occupied_unknown" for c in candidates)
        ),
        "pre_extension_door_cut_cells": int(np.count_nonzero(door_cut_mask)),
        "pre_extension_door_detected_cells": int(np.count_nonzero(detected_door_mask)),
        "pre_extension_door_rois": [list(map(int, r)) for r in rois],
        "pre_extension_door_candidates": [c.serialize() for c in candidates],
    }
    return DoorPatternDetectionResult(
        detected_door_mask=detected_door_mask,
        door_cut_mask=door_cut_mask,
        pattern_type_map=pattern_type_map,
        candidates=candidates,
        debug=debug,
    )


def resolve_detection_rois(
    *,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    observed_mask: np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
    cfg: DoorPatternConfig,
) -> list[tuple[int, int, int, int]]:
    _ = unk
    h, w = free.shape
    if roi is not None:
        clipped = clip_roi(roi, h, w)
        return [] if clipped is None else [clipped]
    if observed_mask is not None:
        obs = np.asarray(observed_mask, dtype=bool)
        if obs.shape != free.shape:
            raise ValueError("observed_mask must have same HxW shape")
        bbox = bbox_from_mask(obs)
        return [] if bbox is None else [bbox]
    bbox = bbox_from_mask(free | occ)
    if bbox is None:
        return []
    min_r, min_c, max_r, max_c = bbox
    min_r = max(0, int(min_r) - max(0, int(cfg.unknown_tail_padding_cells)))
    return [(min_r, int(min_c), int(max_r), int(max_c))]


def clip_roi(roi: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int] | None:
    min_r, min_c, max_r, max_c = (int(v) for v in roi)
    min_r = max(0, min(min_r, h - 1))
    max_r = max(0, min(max_r, h - 1))
    min_c = max(0, min(min_c, w - 1))
    max_c = max(0, min(max_c, w - 1))
    if min_r > max_r or min_c > max_c:
        return None
    return min_r, min_c, max_r, max_c


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size <= 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def resolve_bottom_start(
    *,
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    bottom_r: int,
    col: int,
    roi: tuple[int, int, int, int],
    cfg: DoorPatternConfig,
) -> tuple[int, int] | None:
    min_r, _min_c, _max_r, _max_c = roi
    start = (int(bottom_r), int(col))
    if cell_state(start, free, occ, unk) == "F":
        return start
    if not bool(cfg.allow_start_lift):
        return None
    for lift in range(1, max(0, int(cfg.max_start_lift_cells)) + 1):
        row = int(bottom_r) + int(cfg.scan_dr) * lift
        if row < min_r:
            break
        lifted = (row, int(col))
        if cell_state(lifted, free, occ, unk) == "F":
            return lifted
    return None


def try_match_rule_b_free_occupied_unknown(
    *,
    candidate_id: int,
    start: tuple[int, int],
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    roi: tuple[int, int, int, int],
    cfg: DoorPatternConfig,
) -> DoorPatternCandidate:
    cells = trace_line(start, int(cfg.scan_dr), int(cfg.scan_dc), roi)
    i = 0
    free_cells: list[tuple[int, int]] = []
    occupied_cells: list[tuple[int, int]] = []
    unknown_cells: list[tuple[int, int]] = []

    while i < len(cells) and is_strict_free(cells[i], free, occ, unk):
        free_cells.append(cells[i])
        i += 1
    if len(free_cells) < int(cfg.min_free_run_cells):
        return reject(candidate_id, "free_occupied_unknown", start, "free_run_too_short")
    if bool(cfg.reject_unknown_below) and i < len(cells) and is_unknown(cells[i], free, occ, unk):
        return reject(candidate_id, "free_occupied_unknown", start, "unknown_below_occupied")

    while i < len(cells) and is_strict_occupied(cells[i], free, occ, unk):
        occupied_cells.append(cells[i])
        i += 1
    if len(occupied_cells) < int(cfg.min_occupied_run_cells):
        return reject(candidate_id, "free_occupied_unknown", start, "occupied_run_too_short")

    while i < len(cells) and is_unknown(cells[i], free, occ, unk):
        unknown_cells.append(cells[i])
        i += 1
    if len(unknown_cells) < int(cfg.min_unknown_run_cells):
        return reject(candidate_id, "free_occupied_unknown", start, "unknown_tail_missing")
    if i != len(cells):
        return reject(candidate_id, "free_occupied_unknown", start, "non_unknown_after_unknown_tail")

    return DoorPatternCandidate(
        candidate_id=int(candidate_id),
        pattern_type="free_occupied_unknown",
        start_rc=start,
        free_cells=free_cells,
        occupied_cells=occupied_cells,
        unknown_cells=unknown_cells,
        turn_rc=None,
        door_cut_cells=select_cut_cells(free_cells, cfg, free.shape),
        accepted=True,
    )


def try_match_rule_a_free_turn_occupied(
    *,
    candidate_id: int,
    start: tuple[int, int],
    free: np.ndarray,
    occ: np.ndarray,
    unk: np.ndarray,
    roi: tuple[int, int, int, int],
    cfg: DoorPatternConfig,
) -> DoorPatternCandidate:
    main_cells = trace_line(start, int(cfg.scan_dr), int(cfg.scan_dc), roi)
    free_cells: list[tuple[int, int]] = []
    for rc in main_cells:
        if is_unknown(rc, free, occ, unk):
            return reject(candidate_id, "free_turn_occupied", start, "unknown_in_lower_free_run")
        if is_strict_free(rc, free, occ, unk):
            free_cells.append(rc)
            continue
        break
    if len(free_cells) < int(cfg.min_free_run_cells):
        return reject(candidate_id, "free_turn_occupied", start, "free_run_too_short")

    turn_base = free_cells[-1]
    turn_dirs = list(cfg.turn_dirs)
    if bool(cfg.allow_diagonal_turns):
        turn_dirs.extend([(-1, -1), (-1, 1)])
    for dr, dc in turn_dirs:
        turn_path = trace_line(
            turn_base,
            int(dr),
            int(dc),
            roi,
            skip_start=True,
            max_cells=cfg.rule_a_turn_probe_max_cells,
        )
        occupied_cells: list[tuple[int, int]] = []
        failed = False
        for rc in turn_path:
            if is_strict_occupied(rc, free, occ, unk):
                occupied_cells.append(rc)
                continue
            failed = True
            break
        if failed or len(occupied_cells) < int(cfg.min_occupied_run_cells):
            continue
        return DoorPatternCandidate(
            candidate_id=int(candidate_id),
            pattern_type="free_turn_occupied",
            start_rc=start,
            free_cells=free_cells,
            occupied_cells=occupied_cells,
            unknown_cells=[],
            turn_rc=turn_base,
            door_cut_cells=select_cut_cells(free_cells, cfg, free.shape),
            accepted=True,
        )
    return reject(candidate_id, "free_turn_occupied", start, "no_valid_turn_with_all_occupied_after")


def trace_line(
    start: tuple[int, int],
    dr: int,
    dc: int,
    roi: tuple[int, int, int, int],
    *,
    skip_start: bool = False,
    max_cells: int | None = None,
) -> list[tuple[int, int]]:
    min_r, min_c, max_r, max_c = roi
    r, c = int(start[0]), int(start[1])
    if skip_start:
        r += int(dr)
        c += int(dc)
    out: list[tuple[int, int]] = []
    while int(min_r) <= r <= int(max_r) and int(min_c) <= c <= int(max_c):
        out.append((int(r), int(c)))
        if max_cells is not None and len(out) >= int(max_cells):
            break
        r += int(dr)
        c += int(dc)
    return out


def cell_state(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> str:
    r, c = int(rc[0]), int(rc[1])
    if bool(unk[r, c]):
        return "U"
    if bool(occ[r, c]):
        return "O"
    if bool(free[r, c]):
        return "F"
    return "U"


def is_unknown(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> bool:
    return cell_state(rc, free, occ, unk) == "U"


def is_strict_free(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> bool:
    r, c = int(rc[0]), int(rc[1])
    return bool(free[r, c]) and not bool(occ[r, c]) and not bool(unk[r, c])


def is_strict_occupied(rc: tuple[int, int], free: np.ndarray, occ: np.ndarray, unk: np.ndarray) -> bool:
    r, c = int(rc[0]), int(rc[1])
    return bool(occ[r, c]) and not bool(free[r, c]) and not bool(unk[r, c])


def select_cut_cells(
    free_cells: Sequence[tuple[int, int]],
    cfg: DoorPatternConfig,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    if not free_cells:
        return []
    thickness = max(1, int(cfg.door_cut_thickness_cells))
    lateral = max(0, int(cfg.door_cut_lateral_radius_cells))
    base_cells = list(free_cells)[-thickness:]
    lat_dirs = _lateral_dirs(int(cfg.scan_dr), int(cfg.scan_dc))
    out: list[tuple[int, int]] = []
    for r, c in base_cells:
        out.append((int(r), int(c)))
        for ldr, ldc in lat_dirs:
            for dist in range(1, lateral + 1):
                out.append((int(r) + int(ldr) * dist, int(c) + int(ldc) * dist))
    return unique_in_bounds(out, shape)


def _lateral_dirs(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if int(dr) == 0 and int(dc) == 0:
        return (0, -1), (0, 1)
    return (-int(dc), int(dr)), (int(dc), -int(dr))


def unique_in_bounds(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = int(shape[0]), int(shape[1])
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for r, c in cells:
        rc = (int(r), int(c))
        if rc in seen:
            continue
        if 0 <= rc[0] < h and 0 <= rc[1] < w:
            seen.add(rc)
            out.append(rc)
    return out


def mark_candidate(
    cand: DoorPatternCandidate,
    detected_door_mask: np.ndarray,
    door_cut_mask: np.ndarray,
    pattern_type_map: np.ndarray,
    pattern_value: int,
    free: np.ndarray,
) -> None:
    for rc in [*cand.free_cells, *cand.occupied_cells, *cand.unknown_cells]:
        detected_door_mask[rc] = True
        pattern_type_map[rc] = np.uint8(pattern_value)
    for rc in cand.door_cut_cells:
        if bool(free[rc]):
            door_cut_mask[rc] = True


def merge_nearby_door_cuts(door_cut_mask: np.ndarray, cfg: DoorPatternConfig) -> np.ndarray:
    cut = np.asarray(door_cut_mask, dtype=bool)
    radius = int(cfg.merge_nearby_door_cells)
    if radius <= 0 or not np.any(cut):
        return cut
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(cut, structure=structure).astype(bool)


def reject(candidate_id: int, pattern_type: str, start: tuple[int, int], reason: str) -> DoorPatternCandidate:
    return DoorPatternCandidate(
        candidate_id=int(candidate_id),
        pattern_type=str(pattern_type),
        start_rc=(int(start[0]), int(start[1])),
        accepted=False,
        reject_reason=str(reason),
    )
