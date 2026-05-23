from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .utils import dilate, label_components, rasterize_line


@dataclass
class CorridorSeparatorV62:
    start_rc: tuple[int, int]
    end_rc: tuple[int, int]
    direction_rc: tuple[int, int]
    length_m: float
    free_ratio: float
    accepted: bool
    reject_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "start_rc": [int(self.start_rc[0]), int(self.start_rc[1])],
            "end_rc": [int(self.end_rc[0]), int(self.end_rc[1])],
            "direction_rc": [int(self.direction_rc[0]), int(self.direction_rc[1])],
            "length_m": float(self.length_m),
            "free_ratio": float(self.free_ratio),
            "accepted": bool(self.accepted),
            "reject_reason": str(self.reject_reason),
        }


@dataclass
class CorridorLCornerV62Result:
    corridor_l_corner_candidates: np.ndarray
    accepted_separator_mask: np.ndarray
    rejected_separator_mask: np.ndarray
    accepted: list[CorridorSeparatorV62] = field(default_factory=list)
    rejected: list[CorridorSeparatorV62] = field(default_factory=list)
    debug: dict[str, object] = field(default_factory=dict)


def detect_corridor_l_corners_v6_2(
    *,
    wall_line_support_mask: np.ndarray,
    stable_free_mask: np.ndarray,
    door_mask: np.ndarray,
    resolution_m: float,
    config: Mapping[str, object] | None = None,
) -> CorridorLCornerV62Result:
    cfg = dict(config or {})
    corridor_cfg = dict(cfg.get("corridor", cfg) or {})
    shape = np.asarray(stable_free_mask, dtype=bool).shape
    enabled = bool(corridor_cfg.get("enabled", True))
    if not enabled:
        zeros = np.zeros(shape, dtype=bool)
        return CorridorLCornerV62Result(zeros, zeros.copy(), zeros.copy(), debug={"enabled": False})

    wall_line = np.asarray(wall_line_support_mask, dtype=bool)
    free = np.asarray(stable_free_mask, dtype=bool)
    door = np.asarray(door_mask, dtype=bool)
    min_leg_m = float(corridor_cfg.get("l_corner_min_leg_m", 0.15))
    min_leg_cells = max(2, int(np.ceil(min_leg_m / max(1e-6, float(resolution_m)))))
    corner_tol_cells = max(1, int(np.ceil(min_leg_m / max(1e-6, float(resolution_m)))))
    horizontal = _axis_runs(wall_line, axis="h", min_len_cells=min_leg_cells)
    vertical = _axis_runs(wall_line, axis="v", min_len_cells=min_leg_cells)
    corners = _l_corners(horizontal, vertical, corner_tol_cells=corner_tol_cells)

    candidate_mask = np.zeros(shape, dtype=bool)
    accepted_mask = np.zeros(shape, dtype=bool)
    rejected_mask = np.zeros(shape, dtype=bool)
    accepted: list[CorridorSeparatorV62] = []
    rejected: list[CorridorSeparatorV62] = []
    max_cells = max(1, int(np.ceil(float(corridor_cfg.get("extension_max_m", 2.0)) / max(1e-6, float(resolution_m)))))
    free_ratio_min = float(corridor_cfg.get("extension_free_ratio_min", 0.85))
    min_split_area_m2 = float(corridor_cfg.get("min_split_area_m2", 0.25))
    thickness_radius = max(0, int(round(float(corridor_cfg.get("extension_thickness_m", 0.07)) / max(1e-6, float(resolution_m)))))
    start_skip_cells = max(2 * min_leg_cells, int(np.ceil(0.25 / max(1e-6, float(resolution_m)))))
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    wall_labels, _wall_component_count = label_components(wall_line, 8)

    seen: set[tuple[int, int, int, int]] = set()
    for corner in corners:
        for direction in directions:
            candidate = _trace_extension(
                start=corner,
                direction=direction,
                wall_line=wall_line,
                free=free,
                door=door,
                wall_labels=wall_labels,
                max_cells=max_cells,
                start_skip_cells=start_skip_cells,
                resolution_m=float(resolution_m),
                free_ratio_min=free_ratio_min,
                min_split_area_m2=min_split_area_m2,
                thickness_radius=thickness_radius,
            )
            if candidate is None:
                continue
            key = (*candidate.start_rc, *candidate.end_rc)
            if key in seen:
                continue
            seen.add(key)
            mask = _separator_mask(candidate.start_rc, candidate.end_rc, shape, thickness_radius)
            candidate_mask |= mask
            if candidate.accepted:
                accepted.append(candidate)
                accepted_mask |= mask
            else:
                rejected.append(candidate)
                rejected_mask |= mask

    debug = {
        "enabled": True,
        "source": "corridor_l_corner_v6_2",
        "horizontal_run_count": int(len(horizontal)),
        "vertical_run_count": int(len(vertical)),
        "l_corner_count": int(len(corners)),
        "candidate_count": int(len(accepted) + len(rejected)),
        "accepted_corridor_separator_count": int(len(accepted)),
        "rejected_corridor_separator_count": int(len(rejected)),
        "extension_max_m": float(corridor_cfg.get("extension_max_m", 2.0)),
        "extension_free_ratio_min": float(free_ratio_min),
        "min_split_area_m2": float(min_split_area_m2),
        "accepted": [item.to_dict() for item in accepted[:256]],
        "rejected": [item.to_dict() for item in rejected[:256]],
    }
    return CorridorLCornerV62Result(
        corridor_l_corner_candidates=candidate_mask.astype(bool),
        accepted_separator_mask=accepted_mask.astype(bool),
        rejected_separator_mask=rejected_mask.astype(bool),
        accepted=accepted,
        rejected=rejected,
        debug=debug,
    )


def _axis_runs(mask: np.ndarray, *, axis: str, min_len_cells: int) -> list[dict[str, object]]:
    arr = np.asarray(mask, dtype=bool)
    h, w = arr.shape
    runs: list[dict[str, object]] = []
    if axis == "h":
        for r in range(h):
            c = 0
            while c < w:
                if not arr[r, c]:
                    c += 1
                    continue
                start = c
                while c + 1 < w and arr[r, c + 1]:
                    c += 1
                end = c
                if end - start + 1 >= int(min_len_cells):
                    runs.append({"axis": "h", "row": int(r), "start": int(start), "end": int(end), "endpoints": [(int(r), int(start)), (int(r), int(end))]})
                c += 1
    else:
        for c in range(w):
            r = 0
            while r < h:
                if not arr[r, c]:
                    r += 1
                    continue
                start = r
                while r + 1 < h and arr[r + 1, c]:
                    r += 1
                end = r
                if end - start + 1 >= int(min_len_cells):
                    runs.append({"axis": "v", "col": int(c), "start": int(start), "end": int(end), "endpoints": [(int(start), int(c)), (int(end), int(c))]})
                r += 1
    return runs


def _l_corners(horizontal: list[dict[str, object]], vertical: list[dict[str, object]], *, corner_tol_cells: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for h_run in horizontal:
        for h_endpoint in h_run["endpoints"]:  # type: ignore[index]
            hr, hc = int(h_endpoint[0]), int(h_endpoint[1])
            for v_run in vertical:
                for v_endpoint in v_run["endpoints"]:  # type: ignore[index]
                    vr, vc = int(v_endpoint[0]), int(v_endpoint[1])
                    if max(abs(hr - vr), abs(hc - vc)) <= int(corner_tol_cells):
                        corner = (int(round((hr + vr) * 0.5)), int(round((hc + vc) * 0.5)))
                        if corner not in seen:
                            seen.add(corner)
                            out.append(corner)
    return out


def _trace_extension(
    *,
    start: tuple[int, int],
    direction: tuple[int, int],
    wall_line: np.ndarray,
    free: np.ndarray,
    door: np.ndarray,
    wall_labels: np.ndarray,
    max_cells: int,
    start_skip_cells: int,
    resolution_m: float,
    free_ratio_min: float,
    min_split_area_m2: float,
    thickness_radius: int,
) -> CorridorSeparatorV62 | None:
    h, w = wall_line.shape
    dr, dc = int(direction[0]), int(direction[1])
    sr, sc = int(start[0]), int(start[1])
    start_wall_label = int(wall_labels[sr, sc]) if 0 <= sr < h and 0 <= sc < w else 0
    last_door_only: tuple[int, int] | None = None
    for step in range(1, int(max_cells) + 1):
        rr, cc = sr + dr * step, sc + dc * step
        if rr < 0 or rr >= h or cc < 0 or cc >= w:
            break
        if step <= int(start_skip_cells):
            continue
        if wall_line[rr, cc]:
            if door[rr, cc]:
                last_door_only = (int(rr), int(cc))
                continue
            if start_wall_label > 0 and int(wall_labels[rr, cc]) == start_wall_label:
                continue
            thin_mask = _separator_mask(start, (int(rr), int(cc)), wall_line.shape, 0)
            split_mask = _separator_mask(start, (int(rr), int(cc)), wall_line.shape, thickness_radius)
            free_ratio = _free_ratio(thin_mask, free, wall_line)
            accepted = bool(free_ratio >= float(free_ratio_min) and _split_creates_valid_regions(split_mask, free, min_split_area_m2, resolution_m))
            return CorridorSeparatorV62(
                start_rc=(int(sr), int(sc)),
                end_rc=(int(rr), int(cc)),
                direction_rc=(int(dr), int(dc)),
                length_m=float(step) * float(resolution_m),
                free_ratio=float(free_ratio),
                accepted=accepted,
                reject_reason="" if accepted else "reject_free_ratio_or_topology",
            )
    if last_door_only is not None:
        return CorridorSeparatorV62(
            start_rc=(int(sr), int(sc)),
            end_rc=last_door_only,
            direction_rc=(int(dr), int(dc)),
            length_m=float(max(abs(last_door_only[0] - sr), abs(last_door_only[1] - sc))) * float(resolution_m),
            free_ratio=0.0,
            accepted=False,
            reject_reason="reject_only_intersects_door",
        )
    return None


def _separator_mask(start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int], thickness_radius: int) -> np.ndarray:
    mask = rasterize_line(start, end, shape)
    if int(thickness_radius) > 0:
        mask = dilate(mask, int(thickness_radius))
    return mask.astype(bool)


def _free_ratio(mask: np.ndarray, free: np.ndarray, wall_line: np.ndarray) -> float:
    path = np.asarray(mask, dtype=bool) & ~np.asarray(wall_line, dtype=bool)
    total = int(np.count_nonzero(path))
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(path & np.asarray(free, dtype=bool)) / total)


def _split_creates_valid_regions(mask: np.ndarray, free: np.ndarray, min_split_area_m2: float, resolution_m: float) -> bool:
    core = np.asarray(free, dtype=bool) & ~np.asarray(mask, dtype=bool)
    labels, count = label_components(core, 4)
    min_cells = max(1, int(np.ceil(float(min_split_area_m2) / max(1e-9, float(resolution_m) ** 2))))
    valid = 0
    for idx in range(1, int(count) + 1):
        if int(np.count_nonzero(labels == idx)) >= min_cells:
            valid += 1
    return bool(valid >= 2)
