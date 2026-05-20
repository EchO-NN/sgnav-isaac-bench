from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.utils import dilate, label_components


@dataclass
class WatershedSeedConfig:
    confirmed_min_clearance_m: float = 0.55
    confirmed_min_distance_m: float = 1.20
    confirmed_min_component_area_m2: float = 1.00
    confirmed_seed_radius_m: float = 0.06
    provisional_enabled: bool = True
    provisional_search_radius_m: float = 0.80
    provisional_min_clearance_m: float = 0.25
    provisional_min_distance_from_confirmed_m: float = 0.70
    provisional_seed_radius_m: float = 0.06

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "WatershedSeedConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WatershedSeedResult:
    marker_labels: np.ndarray
    seed_type_by_label: dict[int, str]
    confirmed_room_seeds: np.ndarray
    frontier_room_seeds: np.ndarray
    corridor_seeds: np.ndarray
    report: dict


def build_watershed_markers(
    *,
    free_clean: np.ndarray,
    dist_struct_m: np.ndarray,
    corridor_seed_mask: np.ndarray,
    frontier_clusters: Sequence[object] | None,
    resolution_m: float,
    config: WatershedSeedConfig,
) -> WatershedSeedResult:
    free = np.asarray(free_clean, dtype=bool)
    dist = np.asarray(dist_struct_m, dtype=np.float32)
    corridor = np.asarray(corridor_seed_mask, dtype=bool) & free
    markers = np.zeros(free.shape, dtype=np.int32)
    seed_types: dict[int, str] = {}
    next_label = 1

    corridor_labels, corridor_count = label_components(corridor, 8)
    for label in range(1, int(corridor_count) + 1):
        mask = corridor_labels == label
        if not np.any(mask):
            continue
        markers[mask] = next_label
        seed_types[next_label] = "corridor"
        next_label += 1

    confirmed = np.zeros_like(free, dtype=bool)
    confirmed_points, confirmed_report = _select_confirmed_seed_points(
        free=free,
        excluded=corridor,
        dist_struct_m=dist,
        resolution_m=float(resolution_m),
        config=config,
    )
    seed_radius = max(0, int(round(float(config.confirmed_seed_radius_m) / max(1e-6, float(resolution_m)))))
    for r, c, score in confirmed_points:
        mask = np.zeros_like(free, dtype=bool)
        mask[int(r), int(c)] = True
        mask = dilate(mask, seed_radius) & free & ~corridor & (markers == 0)
        if not np.any(mask):
            continue
        markers[mask] = next_label
        seed_types[next_label] = "confirmed_room"
        confirmed |= mask
        confirmed_report.append({"row": int(r), "col": int(c), "score_m": float(score), "label": int(next_label), "active": True})
        next_label += 1

    frontier_seeds = np.zeros_like(free, dtype=bool)
    frontier_report: list[dict] = []
    if bool(config.provisional_enabled):
        confirmed_distance = ndimage.distance_transform_edt(~confirmed).astype(np.float32) * float(resolution_m)
        for idx, cluster in enumerate(frontier_clusters or []):
            cells = _frontier_cells(cluster)
            candidate = _frontier_seed_cell(
                cells,
                free=free,
                dist_struct_m=dist,
                corridor=corridor,
                confirmed_distance_m=confirmed_distance,
                resolution_m=float(resolution_m),
                config=config,
            )
            report = {"frontier_index": int(idx), "frontier_id": _frontier_id(cluster, idx), "cell_count": int(len(cells))}
            if candidate is None:
                frontier_report.append({**report, "state": "rejected", "reason": "no_clear_non_corridor_free_seed"})
                continue
            r, c, score = candidate
            seed = np.zeros_like(free, dtype=bool)
            seed[int(r), int(c)] = True
            radius = max(0, int(round(float(config.provisional_seed_radius_m) / max(1e-6, float(resolution_m)))))
            seed = dilate(seed, radius) & free & ~corridor & (markers == 0)
            if not np.any(seed):
                frontier_report.append({**report, "state": "pending", "reason": "overlaps_higher_priority_seed", "row": int(r), "col": int(c)})
                continue
            markers[seed] = next_label
            seed_types[next_label] = "provisional_room"
            frontier_seeds |= seed
            frontier_report.append({**report, "state": "active", "row": int(r), "col": int(c), "score_m": float(score), "label": int(next_label)})
            next_label += 1

    if next_label == 1 and np.any(free):
        r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
        markers[int(r), int(c)] = next_label
        seed_types[next_label] = "confirmed_room"
        confirmed[int(r), int(c)] = True
        confirmed_report.append({"row": int(r), "col": int(c), "score_m": float(dist[int(r), int(c)]), "label": int(next_label), "active": True, "forced": True})

    return WatershedSeedResult(
        marker_labels=markers,
        seed_type_by_label=seed_types,
        confirmed_room_seeds=confirmed.astype(bool),
        frontier_room_seeds=frontier_seeds.astype(bool),
        corridor_seeds=corridor.astype(bool),
        report={
            "marker_count": int(len(seed_types)),
            "seed_type_by_label": {str(k): v for k, v in seed_types.items()},
            "confirmed_room_seeds": confirmed_report,
            "frontier_room_seeds": frontier_report,
            "corridor_seed_components": int(corridor_count),
            "priority": ["corridor", "confirmed_room", "provisional_room"],
        },
    )


def _select_confirmed_seed_points(
    *,
    free: np.ndarray,
    excluded: np.ndarray,
    dist_struct_m: np.ndarray,
    resolution_m: float,
    config: WatershedSeedConfig,
) -> tuple[list[tuple[int, int, float]], list[dict]]:
    labels, count = label_components(free, 4)
    min_area_cells = max(1, int(round(float(config.confirmed_min_component_area_m2) / max(1e-6, float(resolution_m) ** 2))))
    min_distance_cells = max(1, int(round(float(config.confirmed_min_distance_m) / max(1e-6, float(resolution_m)))))
    points: list[tuple[int, int, float]] = []
    report: list[dict] = []
    for comp_id in range(1, int(count) + 1):
        comp = (labels == comp_id) & ~np.asarray(excluded, dtype=bool)
        if int(np.count_nonzero(comp)) < min_area_cells:
            report.append({"component": int(comp_id), "active": False, "reason": "small_component", "area_cells": int(np.count_nonzero(comp))})
            continue
        local_max = ndimage.maximum_filter(dist_struct_m, size=max(3, min_distance_cells | 1), mode="constant", cval=0.0)
        candidates = comp & (dist_struct_m >= float(config.confirmed_min_clearance_m)) & (dist_struct_m >= local_max - 1e-6)
        candidate_components: list[tuple[int, int, float]] = []
        peak_labels, peak_count = label_components(candidates, 8)
        for peak_id in range(1, int(peak_count) + 1):
            peak = peak_labels == peak_id
            if not np.any(peak):
                continue
            score_grid = np.where(peak, dist_struct_m, -1.0)
            row, col = np.unravel_index(int(np.argmax(score_grid)), score_grid.shape)
            candidate_components.append((int(row), int(col), float(score_grid[int(row), int(col)])))
        if not candidate_components:
            row, col = np.unravel_index(int(np.argmax(np.where(comp, dist_struct_m, -1.0))), dist_struct_m.shape)
            if dist_struct_m[int(row), int(col)] <= 0:
                report.append({"component": int(comp_id), "active": False, "reason": "no_clearance"})
                continue
            candidate_components.append((int(row), int(col), float(dist_struct_m[int(row), int(col)])))
        ordered = sorted(candidate_components, key=lambda item: -item[2])
        accepted: list[tuple[int, int, float]] = []
        for item in ordered:
            if all((item[0] - prev[0]) ** 2 + (item[1] - prev[1]) ** 2 >= min_distance_cells**2 for prev in accepted):
                accepted.append(item)
        points.extend(accepted)
        report.append({"component": int(comp_id), "active": bool(accepted), "seed_count": int(len(accepted)), "area_cells": int(np.count_nonzero(comp))})
    return points, report


def _frontier_seed_cell(
    cells: list[tuple[int, int]],
    *,
    free: np.ndarray,
    dist_struct_m: np.ndarray,
    corridor: np.ndarray,
    confirmed_distance_m: np.ndarray,
    resolution_m: float,
    config: WatershedSeedConfig,
) -> tuple[int, int, float] | None:
    if not cells:
        return None
    search = max(1, int(round(float(config.provisional_search_radius_m) / max(1e-6, float(resolution_m)))))
    mask = np.zeros_like(free, dtype=bool)
    h, w = free.shape
    for r, c in cells:
        if 0 <= int(r) < h and 0 <= int(c) < w:
            mask[int(r), int(c)] = True
    near = dilate(mask, search) & free & ~corridor
    near &= dist_struct_m >= float(config.provisional_min_clearance_m)
    near &= confirmed_distance_m >= float(config.provisional_min_distance_from_confirmed_m)
    if not np.any(near):
        return None
    scores = np.where(near, dist_struct_m, -1.0)
    row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(row), int(col), float(scores[int(row), int(col)])


def _frontier_cells(frontier: object) -> list[tuple[int, int]]:
    if isinstance(frontier, Mapping):
        raw = frontier.get("cells", frontier.get("members", []))
    else:
        raw = getattr(frontier, "cells", getattr(frontier, "members", []))
    out: list[tuple[int, int]] = []
    for cell in raw or []:
        if len(cell) >= 2:
            out.append((int(cell[0]), int(cell[1])))
    return out


def _frontier_id(frontier: object, idx: int) -> str:
    if isinstance(frontier, Mapping):
        return str(frontier.get("frontier_id", frontier.get("id", idx)))
    return str(getattr(frontier, "frontier_id", getattr(frontier, "id", idx)))
