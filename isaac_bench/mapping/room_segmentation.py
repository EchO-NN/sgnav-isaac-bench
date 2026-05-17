from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid


GridCell = Tuple[int, int]


@dataclass
class RoomSegmentationConfig:
    enabled: bool = True
    source_grid: str = "online_depth_observed"
    update_every_steps: int = 5
    min_observed_free_cells: int = 50
    min_room_area_m2: float = 1.5
    max_clutter_component_area_m2: float = 1.0
    morphology_close_radius_m: float = 0.20
    morphology_open_radius_m: float = 0.10
    distance_smooth_sigma_cells: float = 1.0
    seed_min_clearance_m: float = 0.45
    seed_min_distance_m: float = 1.2
    doorway_width_min_m: float = 0.55
    doorway_width_max_m: float = 1.45
    doorway_clearance_max_m: float = 0.85
    merge_wide_openings: bool = True
    small_segment_merge_area_m2: float = 1.2
    id_iou_threshold: float = 0.35
    unknown_boundary_confidence_penalty: bool = True
    stale_ttl_steps: int = 2
    debug_dump: bool = False
    debug_dir: str = "debug/room_segmentation"
    resolution_m: float = 0.05
    map_info: Optional[MapInfo] = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]] = None, **overrides) -> "RoomSegmentationConfig":
        raw = dict(data or {})
        raw.update({key: value for key, value in overrides.items() if value is not None})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class RoomMask:
    room_id: str
    mask: np.ndarray
    centroid_xy: Tuple[float, float]
    area_m2: float
    boundary_unknown_fraction: float
    doorway_edges: List[dict]
    confidence: float
    source: str = "online_geometry_watershed"
    observed_free_cells: int = 0
    mask_confidence: float = 0.0
    is_partial: bool = False
    step: int = 0
    stale: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ObjectRoomAssignment:
    object_id: str
    room_id: Optional[str]
    edge_type: str = "contains"
    source: str = "online_geometry_mask_overlap"
    centroid_inside: bool = False
    mask_overlap_ratio: float = 0.0
    assignment_confidence: float = 0.0
    ambiguous: bool = False

    def to_edge_metadata(self) -> dict:
        return {
            "edge_type": self.edge_type,
            "source": self.source,
            "centroid_inside": bool(self.centroid_inside),
            "mask_overlap_ratio": float(self.mask_overlap_ratio),
            "assignment_confidence": float(self.assignment_confidence),
            "ambiguous": bool(self.ambiguous),
        }


class OnlineRoomSegmenter:
    def __init__(self, config: Optional[RoomSegmentationConfig | Mapping[str, object]] = None):
        if isinstance(config, RoomSegmentationConfig):
            self.config = config
        else:
            self.config = RoomSegmentationConfig.from_mapping(config or {})
        self._previous: Dict[str, RoomMask] = {}
        self._last_live_ids: set[str] = set()
        self._next_room_index = 1
        self.last_debug: Dict[str, object] = {}

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
    ) -> List[RoomMask]:
        if not self.config.enabled:
            self.last_debug = {"enabled": False, "room_count": 0}
            return []
        free = np.asarray(observed_free_mask, dtype=bool)
        occ = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool)
        unknown = np.asarray(unknown_mask, dtype=bool)
        if free.shape != occ.shape or free.shape != unknown.shape:
            raise ValueError("room segmentation masks must have the same HxW shape")

        structural = build_structural_free_mask(free, occ, unknown, self.config)
        room_masks = segment_room_masks(structural, unknown, self.config, step=int(step))
        room_masks = self._assign_stable_ids(room_masks, int(step))
        self._previous = {room.room_id: room for room in room_masks}
        self._last_live_ids = {room.room_id for room in room_masks if not room.stale}
        self.last_debug = room_segmentation_debug(room_masks, structural)
        return room_masks

    def _assign_stable_ids(self, rooms: Sequence[RoomMask], step: int) -> List[RoomMask]:
        previous_items = [(room_id, room) for room_id, room in self._previous.items() if not room.stale]
        matches: Dict[int, str] = {}
        used_prev: set[str] = set()
        candidates = []
        for idx, room in enumerate(rooms):
            for room_id, prev in previous_items:
                iou = _mask_iou(room.mask, prev.mask)
                dist = _centroid_distance(room.centroid_xy, prev.centroid_xy)
                if iou >= self.config.id_iou_threshold or (iou > 0.0 and dist <= max(2.0, 2.0 * self.config.seed_min_distance_m)):
                    candidates.append((iou, -dist, idx, room_id))
        candidates.sort(reverse=True)
        for _iou, _neg_dist, idx, room_id in candidates:
            if idx in matches or room_id in used_prev:
                continue
            matches[idx] = room_id
            used_prev.add(room_id)
        out: List[RoomMask] = []
        for idx, room in enumerate(rooms):
            room_id = matches.get(idx)
            if room_id is None:
                room_id = "room_%04d" % self._next_room_index
                self._next_room_index += 1
            room.room_id = room_id
            room.step = int(step)
            out.append(room)
        label_to_room = {
            int(room.metadata["label_id"]): room.room_id
            for room in out
            if isinstance(room.metadata, dict) and room.metadata.get("label_id") is not None and not room.stale
        }
        for room in out:
            enriched_edges = []
            for edge in room.doorway_edges:
                item = dict(edge)
                if item.get("room_a_label") is not None:
                    item["room_a"] = label_to_room.get(int(item["room_a_label"]))
                if item.get("room_b_label") is not None:
                    item["room_b"] = label_to_room.get(int(item["room_b_label"]))
                if item.get("room_a") and item.get("room_b") and item.get("room_a") != item.get("room_b"):
                    enriched_edges.append(item)
            room.doorway_edges = enriched_edges
        stale_ttl = max(0, int(self.config.stale_ttl_steps))
        if stale_ttl:
            for room_id, prev in self._previous.items():
                if room_id in used_prev:
                    continue
                if int(step) - int(prev.step) <= stale_ttl:
                    stale = RoomMask(
                        room_id=room_id,
                        mask=np.asarray(prev.mask, dtype=bool).copy(),
                        centroid_xy=tuple(float(v) for v in prev.centroid_xy),
                        area_m2=float(prev.area_m2),
                        boundary_unknown_fraction=float(prev.boundary_unknown_fraction),
                        doorway_edges=list(prev.doorway_edges),
                        confidence=float(prev.confidence),
                        source=prev.source,
                        observed_free_cells=int(prev.observed_free_cells),
                        mask_confidence=float(prev.mask_confidence),
                        is_partial=bool(prev.is_partial),
                        step=int(prev.step),
                        stale=True,
                        metadata={**dict(prev.metadata), "stale_until_step": int(prev.step) + stale_ttl},
                    )
                    out.append(stale)
        return out


def build_structural_free_mask(
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: RoomSegmentationConfig,
) -> np.ndarray:
    free = np.asarray(observed_free_mask, dtype=bool).copy()
    obstacles = np.asarray(obstacle_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    cell_area = float(config.resolution_m) ** 2
    max_clutter_cells = max(0, int(round(float(config.max_clutter_component_area_m2) / max(cell_area, 1e-9))))
    if max_clutter_cells > 0 and np.any(obstacles):
        for component in _connected_components(obstacles):
            if len(component) <= max_clutter_cells:
                for row, col in component:
                    if not unknown[row, col]:
                        free[row, col] = True
    free &= ~unknown
    close_radius = _radius_cells(config.morphology_close_radius_m, config.resolution_m)
    open_radius = _radius_cells(config.morphology_open_radius_m, config.resolution_m)
    if close_radius:
        free = _binary_close(free, close_radius)
    if open_radius:
        free = _binary_open(free, open_radius)
    free &= ~unknown
    return free.astype(bool)


def segment_room_masks(
    structural_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
    config: RoomSegmentationConfig,
    step: int = 0,
) -> List[RoomMask]:
    free = np.asarray(structural_free_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if not np.any(free):
        return []
    min_cells = max(1, int(config.min_observed_free_cells))
    min_area_cells = max(1, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9))))
    components = _connected_components(free)
    if not components:
        return []
    if len(components) == 1:
        component_list = components
    else:
        component_list = [comp for comp in components if len(comp) >= min_cells and len(comp) >= min_area_cells]
        if not component_list:
            component_list = [max(components, key=len)]

    all_rooms: List[RoomMask] = []
    for component in component_list:
        comp_mask = np.zeros_like(free, dtype=bool)
        for row, col in component:
            comp_mask[row, col] = True
        labels, distance_m = _watershed_component(comp_mask, config)
        labels, doorway_edges = _refine_labels_by_doorways(labels, distance_m, config)
        for label_id in sorted(v for v in np.unique(labels) if int(v) > 0):
            mask = labels == label_id
            if not np.any(mask):
                continue
            if int(np.count_nonzero(mask)) < min_cells and len(all_rooms) > 0:
                continue
            room = _room_from_mask("pending", mask, unknown, doorway_edges, config, step)
            room.metadata["label_id"] = int(label_id)
            all_rooms.append(room)
    if not all_rooms and np.any(free):
        all_rooms.append(_room_from_mask("pending", free, unknown, [], config, step))
    return all_rooms


def assign_objects_to_room_masks(
    object_nodes: Iterable[object],
    room_masks: Sequence[RoomMask],
    map_info: MapInfo,
    overlap_ambiguity_margin: float = 0.10,
) -> Dict[str, ObjectRoomAssignment]:
    rooms = [room for room in room_masks if not room.stale]
    assignments: Dict[str, ObjectRoomAssignment] = {}
    if not rooms:
        return assignments
    for obj in object_nodes:
        object_id = str(getattr(obj, "id", getattr(obj, "node_id", "")))
        center = np.asarray(getattr(obj, "center_world", [math.nan, math.nan, 0.0]), dtype=np.float32)
        centroid_inside = False
        centroid_room: Optional[str] = None
        if len(center) >= 2 and np.all(np.isfinite(center[:2])):
            row, col = world_xy_to_grid(float(center[0]), float(center[1]), map_info)
            if is_inside_grid(row, col, map_info):
                for room in rooms:
                    if bool(room.mask[row, col]):
                        centroid_inside = True
                        centroid_room = room.room_id
                        break
        object_cells = _object_cells(obj, map_info)
        overlap_scores: List[Tuple[float, str]] = []
        if object_cells:
            denom = max(1, len(object_cells))
            for room in rooms:
                hits = sum(1 for row, col in object_cells if 0 <= row < room.mask.shape[0] and 0 <= col < room.mask.shape[1] and room.mask[row, col])
                overlap_scores.append((float(hits) / float(denom), room.room_id))
        elif centroid_room is not None:
            overlap_scores.append((1.0, centroid_room))
        overlap_scores.sort(reverse=True)
        if centroid_room is not None:
            best_score = next((score for score, rid in overlap_scores if rid == centroid_room), 1.0)
            best_room = centroid_room
        elif overlap_scores:
            best_score, best_room = overlap_scores[0]
        else:
            assignments[object_id] = ObjectRoomAssignment(object_id=object_id, room_id=None)
            continue
        if best_score <= 0.0 and not centroid_inside:
            assignments[object_id] = ObjectRoomAssignment(object_id=object_id, room_id=None)
            continue
        second = overlap_scores[1][0] if len(overlap_scores) > 1 else 0.0
        ambiguous = bool((best_score - second) < float(overlap_ambiguity_margin) and second > 0.0)
        confidence = 1.0 if centroid_inside else float(best_score)
        if ambiguous:
            confidence = min(confidence, max(0.0, best_score - second + 0.5))
        assignments[object_id] = ObjectRoomAssignment(
            object_id=object_id,
            room_id=best_room,
            centroid_inside=centroid_inside,
            mask_overlap_ratio=float(best_score),
            assignment_confidence=float(np.clip(confidence, 0.0, 1.0)),
            ambiguous=ambiguous,
        )
    return assignments


def room_segmentation_debug(room_masks: Sequence[RoomMask], structural_free_mask: Optional[np.ndarray] = None) -> dict:
    return {
        "source": "online_geometry_watershed",
        "room_count": int(len([room for room in room_masks if not room.stale])),
        "structural_free_cells": int(np.count_nonzero(structural_free_mask)) if structural_free_mask is not None else None,
        "rooms": [room_mask_to_dict(room, include_mask=False) for room in room_masks],
    }


def room_mask_to_dict(room: RoomMask, include_mask: bool = False) -> dict:
    payload = {
        "room_id": room.room_id,
        "area_m2": float(room.area_m2),
        "centroid_xy": [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
        "boundary_unknown_fraction": float(room.boundary_unknown_fraction),
        "observed_free_cells": int(room.observed_free_cells),
        "mask_confidence": float(room.mask_confidence),
        "confidence": float(room.confidence),
        "is_partial": bool(room.is_partial),
        "source": room.source,
        "stale": bool(room.stale),
        "doorway_edges": list(room.doorway_edges),
        "metadata": dict(room.metadata),
    }
    if include_mask:
        payload["mask"] = np.asarray(room.mask, dtype=bool).astype(np.uint8).tolist()
    return payload


def _watershed_component(component_mask: np.ndarray, config: RoomSegmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy import ndimage as ndi

        distance_cells = ndi.distance_transform_edt(component_mask)
        if float(config.distance_smooth_sigma_cells) > 0:
            distance_cells = ndi.gaussian_filter(distance_cells, sigma=float(config.distance_smooth_sigma_cells))
    except Exception:
        distance_cells = _distance_transform_fallback(component_mask)
    distance_m = np.asarray(distance_cells, dtype=np.float32) * float(config.resolution_m)
    seeds = _seed_points(component_mask, distance_m, config)
    markers = np.zeros_like(component_mask, dtype=np.int32)
    for idx, (row, col) in enumerate(seeds, start=1):
        markers[row, col] = idx
    if not seeds:
        return component_mask.astype(np.int32), distance_m
    try:
        from skimage.segmentation import watershed

        labels = watershed(-distance_m, markers=markers, mask=component_mask).astype(np.int32)
    except Exception:
        labels = _seeded_region_grow(component_mask, distance_m, seeds)
    if len([v for v in np.unique(labels) if int(v) > 0]) <= 1:
        bottleneck = _split_by_structural_bottleneck(component_mask, config)
        if bottleneck is not None:
            labels = bottleneck
    return labels, distance_m


def _split_by_structural_bottleneck(mask: np.ndarray, config: RoomSegmentationConfig) -> Optional[np.ndarray]:
    """Fallback split for classic wall-with-doorway geometry.

    Watershed can collapse two rooms connected by a narrow doorway into a single
    basin when the distance field has one broad maximum. This deterministic
    refinement detects a low-free-width column/row inside a large component and
    splits across that bottleneck while keeping doorway cells adjacent.
    """
    src = np.asarray(mask, dtype=bool)
    rr, cc = np.nonzero(src)
    if rr.size == 0:
        return None
    r0, r1 = int(np.min(rr)), int(np.max(rr)) + 1
    c0, c1 = int(np.min(cc)), int(np.max(cc)) + 1
    sub = src[r0:r1, c0:c1]
    min_side_cells = max(4, int(round(float(config.min_room_area_m2) / max(float(config.resolution_m) ** 2, 1e-9) * 0.25)))

    def split_vertical() -> Optional[np.ndarray]:
        sums = np.sum(sub, axis=0)
        if sums.size < 8 or float(np.max(sums)) <= 0.0:
            return None
        candidates = np.where((sums > 0) & (sums <= 0.45 * float(np.max(sums))))[0]
        best = None
        for col in candidates:
            left = int(np.count_nonzero(sub[:, : col + 1]))
            right = int(np.count_nonzero(sub[:, col + 1 :]))
            if left >= min_side_cells and right >= min_side_cells:
                score = float(sums[col]) + abs(left - right) * 1e-4
                if best is None or score < best[0]:
                    best = (score, int(col))
        if best is None:
            return None
        col = best[1]
        out = np.zeros_like(src, dtype=np.int32)
        left_mask = np.zeros_like(sub, dtype=bool)
        right_mask = np.zeros_like(sub, dtype=bool)
        left_mask[:, : col + 1] = sub[:, : col + 1]
        right_mask[:, col + 1 :] = sub[:, col + 1 :]
        if not np.any(right_mask):
            return None
        view = out[r0:r1, c0:c1]
        view[left_mask] = 1
        view[right_mask] = 2
        return out

    def split_horizontal() -> Optional[np.ndarray]:
        sums = np.sum(sub, axis=1)
        if sums.size < 8 or float(np.max(sums)) <= 0.0:
            return None
        candidates = np.where((sums > 0) & (sums <= 0.45 * float(np.max(sums))))[0]
        best = None
        for row in candidates:
            top = int(np.count_nonzero(sub[: row + 1, :]))
            bottom = int(np.count_nonzero(sub[row + 1 :, :]))
            if top >= min_side_cells and bottom >= min_side_cells:
                score = float(sums[row]) + abs(top - bottom) * 1e-4
                if best is None or score < best[0]:
                    best = (score, int(row))
        if best is None:
            return None
        row = best[1]
        out = np.zeros_like(src, dtype=np.int32)
        top_mask = np.zeros_like(sub, dtype=bool)
        bottom_mask = np.zeros_like(sub, dtype=bool)
        top_mask[: row + 1, :] = sub[: row + 1, :]
        bottom_mask[row + 1 :, :] = sub[row + 1 :, :]
        if not np.any(bottom_mask):
            return None
        view = out[r0:r1, c0:c1]
        view[top_mask] = 1
        view[bottom_mask] = 2
        return out

    vertical = split_vertical()
    horizontal = split_horizontal()
    if vertical is None:
        return horizontal
    if horizontal is None:
        return vertical
    # Prefer the split whose two sides are more balanced.
    v_counts = [np.count_nonzero(vertical == 1), np.count_nonzero(vertical == 2)]
    h_counts = [np.count_nonzero(horizontal == 1), np.count_nonzero(horizontal == 2)]
    return vertical if abs(v_counts[0] - v_counts[1]) <= abs(h_counts[0] - h_counts[1]) else horizontal


def _seed_points(mask: np.ndarray, distance_m: np.ndarray, config: RoomSegmentationConfig) -> List[GridCell]:
    candidates: List[GridCell] = []
    try:
        from skimage.feature import peak_local_max

        min_distance_cells = max(1, _radius_cells(config.seed_min_distance_m, config.resolution_m))
        coords = peak_local_max(
            distance_m,
            min_distance=min_distance_cells,
            threshold_abs=float(config.seed_min_clearance_m),
            labels=mask.astype(np.uint8),
            exclude_border=False,
        )
        candidates = [(int(row), int(col)) for row, col in coords]
    except Exception:
        candidates = _local_maxima(mask, distance_m, float(config.seed_min_clearance_m))
    if not candidates:
        rr, cc = np.nonzero(mask)
        if rr.size:
            center_idx = int(np.argmin((rr - float(np.mean(rr))) ** 2 + (cc - float(np.mean(cc))) ** 2))
            candidates = [(int(rr[center_idx]), int(cc[center_idx]))]
    candidates.sort(key=lambda cell: float(distance_m[cell]), reverse=True)
    seeds: List[GridCell] = []
    min_cells = max(1.0, float(config.seed_min_distance_m) / max(float(config.resolution_m), 1e-9))
    for cell in candidates:
        if float(distance_m[cell]) < float(config.seed_min_clearance_m) and seeds:
            continue
        if all(math.hypot(cell[0] - other[0], cell[1] - other[1]) >= min_cells for other in seeds):
            seeds.append(cell)
    return seeds


def _refine_labels_by_doorways(labels: np.ndarray, distance_m: np.ndarray, config: RoomSegmentationConfig) -> Tuple[np.ndarray, List[dict]]:
    labels = np.asarray(labels, dtype=np.int32).copy()
    positive = [int(v) for v in np.unique(labels) if int(v) > 0]
    if len(positive) <= 1:
        return labels, []
    adjacency: Dict[Tuple[int, int], List[GridCell]] = {}
    h, w = labels.shape
    for row in range(h):
        for col in range(w):
            a = int(labels[row, col])
            if a <= 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = row + dr, col + dc
                if rr >= h or cc >= w:
                    continue
                b = int(labels[rr, cc])
                if b <= 0 or b == a:
                    continue
                key = tuple(sorted((a, b)))
                adjacency.setdefault(key, []).append((row, col))
                adjacency.setdefault(key, []).append((rr, cc))
    parent = {label: label for label in positive}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    areas = {label: int(np.count_nonzero(labels == label)) * float(config.resolution_m) ** 2 for label in positive}
    doorway_edges: List[dict] = []
    for (a, b), cells in sorted(adjacency.items()):
        unique_cells = sorted(set(cells))
        width_m = max(float(config.resolution_m), math.sqrt(float(len(unique_cells))) * float(config.resolution_m))
        clearances = [float(distance_m[cell]) for cell in unique_cells if np.isfinite(float(distance_m[cell]))]
        mean_clearance = float(np.mean(clearances)) if clearances else 0.0
        likely_doorway = (
            float(config.doorway_width_min_m) <= width_m <= float(config.doorway_width_max_m)
            and mean_clearance <= float(config.doorway_clearance_max_m)
        )
        tiny = min(float(areas.get(a, 0.0)), float(areas.get(b, 0.0))) < float(config.small_segment_merge_area_m2)
        wide_open = bool(config.merge_wide_openings and width_m > float(config.doorway_width_max_m))
        if likely_doorway and not tiny:
            doorway_edges.append(
                {
                    "room_a_label": int(a),
                    "room_b_label": int(b),
                    "edge_type": "adjacent_via_doorway",
                    "doorway_width_m": float(width_m),
                    "boundary_mean_clearance_m": float(mean_clearance),
                    "source": "online_geometry_watershed_boundary",
                }
            )
        else:
            if tiny or wide_open:
                union(a, b)
    remap: Dict[int, int] = {}
    next_label = 1
    out = np.zeros_like(labels, dtype=np.int32)
    for label in positive:
        root = find(label)
        if root not in remap:
            remap[root] = next_label
            next_label += 1
        out[labels == label] = remap[root]
    remapped_edges = []
    for edge in doorway_edges:
        a = remap.get(find(int(edge["room_a_label"])))
        b = remap.get(find(int(edge["room_b_label"])))
        if a is None or b is None or a == b:
            continue
        remapped_edges.append({**edge, "room_a_label": int(a), "room_b_label": int(b)})
    return out, remapped_edges


def _room_from_mask(room_id: str, mask: np.ndarray, unknown: np.ndarray, doorway_edges: Sequence[dict], config: RoomSegmentationConfig, step: int) -> RoomMask:
    mask_bool = np.asarray(mask, dtype=bool)
    rr, cc = np.nonzero(mask_bool)
    centroid_grid = (float(np.mean(rr)), float(np.mean(cc))) if rr.size else (0.0, 0.0)
    if config.map_info is not None and rr.size:
        cx, cy = grid_to_world_xy(int(round(centroid_grid[0])), int(round(centroid_grid[1])), config.map_info)
        centroid_xy = (float(cx), float(cy))
    else:
        centroid_xy = (float(centroid_grid[1]) * float(config.resolution_m), float(centroid_grid[0]) * float(config.resolution_m))
    cells = int(np.count_nonzero(mask_bool))
    boundary = _boundary(mask_bool)
    boundary_count = int(np.count_nonzero(boundary))
    unknown_fraction = float(np.count_nonzero(boundary & unknown)) / float(max(1, boundary_count))
    area_m2 = float(cells) * float(config.resolution_m) ** 2
    partial = bool(unknown_fraction >= 0.35 or cells < int(config.min_observed_free_cells))
    confidence = 1.0
    if bool(config.unknown_boundary_confidence_penalty):
        confidence -= 0.5 * unknown_fraction
    if partial:
        confidence -= 0.15
    confidence = float(np.clip(confidence, 0.05, 1.0))
    mask_confidence = confidence
    return RoomMask(
        room_id=room_id,
        mask=mask_bool,
        centroid_xy=centroid_xy,
        area_m2=area_m2,
        boundary_unknown_fraction=unknown_fraction,
        doorway_edges=[dict(edge) for edge in doorway_edges],
        confidence=confidence,
        observed_free_cells=cells,
        mask_confidence=mask_confidence,
        is_partial=partial,
        step=int(step),
        metadata={"centroid_grid": [float(centroid_grid[0]), float(centroid_grid[1])]},
    )


def _object_cells(obj: object, map_info: MapInfo) -> List[GridCell]:
    cells = set()
    center_grid = getattr(obj, "center_grid", None)
    if center_grid is not None:
        cells.add((int(center_grid[0]), int(center_grid[1])))
    points = getattr(obj, "point_cloud_world", None)
    if points is not None:
        arr = np.asarray(points, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            for x, y in arr[:, :2]:
                row, col = world_xy_to_grid(float(x), float(y), map_info)
                if is_inside_grid(row, col, map_info):
                    cells.add((int(row), int(col)))
    bbox = getattr(obj, "bbox_world", None)
    if bbox is not None:
        arr = np.asarray(bbox, dtype=np.float32)
        if arr.shape == (2, 3):
            xs = np.linspace(float(arr[0, 0]), float(arr[1, 0]), 3)
            ys = np.linspace(float(arr[0, 1]), float(arr[1, 1]), 3)
            for x in xs:
                for y in ys:
                    row, col = world_xy_to_grid(float(x), float(y), map_info)
                    if is_inside_grid(row, col, map_info):
                        cells.add((int(row), int(col)))
    return sorted(cells)


def _connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    src = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    components: List[List[GridCell]] = []
    for row, col in zip(*np.nonzero(src)):
        start = (int(row), int(col))
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[GridCell] = deque([start])
        comp: List[GridCell] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = cur[0] + dr, cur[1] + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or visited[rr, cc] or not src[rr, cc]:
                    continue
                visited[rr, cc] = True
                queue.append((rr, cc))
        components.append(comp)
    return components


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    try:
        from skimage.morphology import binary_closing, disk

        return binary_closing(mask, disk(radius)).astype(bool)
    except Exception:
        return _erode(_dilate(mask, radius), radius)


def _binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    try:
        from skimage.morphology import binary_opening, disk

        return binary_opening(mask, disk(radius)).astype(bool)
    except Exception:
        return _dilate(_erode(mask, radius), radius)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    src = np.asarray(mask, dtype=bool)
    out = np.zeros_like(src, dtype=bool)
    rows, cols = np.nonzero(src)
    h, w = src.shape
    offsets = _disk_offsets(radius)
    for row, col in zip(rows, cols):
        for dr, dc in offsets:
            rr, cc = int(row + dr), int(col + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return src.copy()
    return ~_dilate(~src, radius)


def _disk_offsets(radius: int) -> List[GridCell]:
    return [
        (dr, dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr * dr + dc * dc <= radius * radius
    ]


def _radius_cells(radius_m: float, resolution_m: float) -> int:
    return max(0, int(round(float(radius_m) / max(float(resolution_m), 1e-9))))


def _local_maxima(mask: np.ndarray, distance_m: np.ndarray, threshold: float) -> List[GridCell]:
    out: List[GridCell] = []
    h, w = mask.shape
    for row, col in zip(*np.nonzero(mask)):
        value = float(distance_m[row, col])
        if value < threshold:
            continue
        local = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < h and 0 <= cc < w and float(distance_m[rr, cc]) > value:
                    local = False
                    break
            if not local:
                break
        if local:
            out.append((int(row), int(col)))
    return out


def _seeded_region_grow(mask: np.ndarray, distance_m: np.ndarray, seeds: Sequence[GridCell]) -> np.ndarray:
    labels = np.zeros_like(mask, dtype=np.int32)
    heap: List[Tuple[float, int, int, int]] = []
    for idx, (row, col) in enumerate(seeds, start=1):
        if not mask[row, col]:
            continue
        labels[row, col] = idx
        heapq.heappush(heap, (-float(distance_m[row, col]), idx, int(row), int(col)))
    h, w = mask.shape
    while heap:
        _neg_dist, label, row, col = heapq.heappop(heap)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + dr, col + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or not mask[rr, cc] or labels[rr, cc] > 0:
                continue
            labels[rr, cc] = label
            heapq.heappush(heap, (-float(distance_m[rr, cc]), label, rr, cc))
    return labels


def _distance_transform_fallback(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    obstacles = np.asarray(~src, dtype=bool)
    rows, cols = np.nonzero(obstacles)
    if rows.size == 0:
        return np.ones_like(src, dtype=np.float32) * max(h, w)
    out = np.zeros_like(src, dtype=np.float32)
    obstacle_points = np.stack([rows, cols], axis=1).astype(np.float32)
    for row, col in zip(*np.nonzero(src)):
        d2 = np.min(np.sum((obstacle_points - np.asarray([row, col], dtype=np.float32)) ** 2, axis=1))
        out[row, col] = math.sqrt(float(d2))
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    eroded = _erode(src, 1)
    return src & ~eroded


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    return 0.0 if union <= 0 else float(inter) / float(union)


def _centroid_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:2], dtype=np.float32) - np.asarray(b[:2], dtype=np.float32)))
