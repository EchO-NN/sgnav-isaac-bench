from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.mapping.room_segmentation import (
    RoomMask,
    RoomProposalState,
    RoomSegmentationConfig,
    merge_open_plan_proposals,
    room_segmentation_debug,
)
from isaac_bench.mapping.structure_extraction import (
    StructureExtractionConfig,
    StructureExtractionResult,
    connected_components,
    extract_rose2_structure,
)


@dataclass
class ROSE2Input:
    occupancy: np.ndarray
    observed_free: np.ndarray
    observed_occupied: np.ndarray
    unknown: np.ndarray
    resolution_m: float
    origin_xy: Tuple[float, float]
    frame_id: int
    step_index: int


@dataclass
class ROSE2RoomMask:
    room_id: str
    mask: np.ndarray
    polygon_xy: List[Tuple[float, float]]
    centroid_xy: Tuple[float, float]
    area_m2: float
    boundary_confidence: float
    segmentation_source: str = "rose2_structure"
    debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class ROSE2Debug:
    dominant_directions_rad: List[float]
    structural_score: np.ndarray
    clean_structure_map: np.ndarray
    hough_segments: List[dict]
    wall_clusters: List[dict]
    representative_lines: List[dict]
    faces: List[dict]
    face_adjacency_edges: List[dict]
    room_polygons: List[dict]
    voronoi_topology_debug: Optional[dict]
    timing_ms: dict


class OnlineROSE2RoomSegmenter:
    source = "rose2_structure"
    context_source = "online_rose2_structure_vlm"

    def __init__(self, config: Optional[RoomSegmentationConfig | Mapping[str, object]] = None):
        rose2_cfg: Mapping[str, object] = {}
        if isinstance(config, RoomSegmentationConfig):
            self.config = config
            rose2_cfg = dict(getattr(config, "rose2", {}) or {})
        else:
            raw_cfg = dict(config or {})
            self.config = RoomSegmentationConfig.from_mapping(raw_cfg)
            rose2_cfg = dict(raw_cfg.get("rose2", {}) or {})
        self.structure_config = StructureExtractionConfig.from_mapping(
            rose2_cfg,
            resolution_m=float(self.config.resolution_m),
            min_room_area_m2=float(self.config.min_room_area_m2),
            clutter_component_max_area_m2=float(self.config.max_clutter_component_area_m2),
            hough_min_line_length_m=float(self.config.min_wall_line_length_m),
        )
        self._previous: Dict[str, RoomMask] = {}
        self._last_live_ids: set[str] = set()
        self._next_room_index = 1
        self.last_debug: Dict[str, object] = {}
        self.last_structure_result: Optional[StructureExtractionResult] = None

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Optional[Iterable[object]] = None,
    ) -> List[RoomMask]:
        proposals, state = self.build_proposals(
            occupancy_map,
            observed_free_mask,
            obstacle_mask,
            unknown_mask,
            step=step,
            object_memory=object_memory,
        )
        _ = proposals
        return self.finalize_proposals(state, proposal_semantic_labels=None)

    def build_proposals(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Optional[Iterable[object]] = None,
    ) -> Tuple[List[RoomMask], RoomProposalState]:
        free = np.asarray(observed_free_mask, dtype=bool)
        occupied = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool)
        unknown = np.asarray(unknown_mask, dtype=bool)
        if free.shape != occupied.shape or free.shape != unknown.shape:
            raise ValueError("ROSE2 room segmentation masks must have the same HxW shape")
        structure = extract_rose2_structure(
            observed_occupied=occupied,
            observed_free=free,
            unknown=unknown,
            config=self.structure_config,
            object_memory=object_memory,
        )
        self.last_structure_result = structure
        physical_labels = np.asarray(structure.face_labels, dtype=np.int32)
        proposal_labels, proposal_rooms = self._build_premerge_zone_proposals(physical_labels, free, unknown, step)
        state = RoomProposalState(
            proposal_labels=proposal_labels,
            structural_free_mask=free,
            structural_obstacle_mask=np.asarray(structure.boundary_map, dtype=bool),
            unknown_mask=unknown,
            distance_m=np.asarray(structure.structural_score, dtype=np.float32),
            step=int(step),
            debug=self._debug_from_structure(structure, proposal_labels, physical_labels),
        )
        return proposal_rooms, state

    def finalize_proposals(
        self,
        proposal_state: RoomProposalState,
        proposal_semantic_labels: Optional[Mapping[int, object]] = None,
    ) -> List[RoomMask]:
        labels = np.asarray(proposal_state.proposal_labels, dtype=np.int32)
        if not np.any(labels > 0):
            self.last_debug = self._empty_debug()
            return []
        final_labels, merge_debug, doorway_edges = merge_open_plan_proposals(
            proposal_labels=labels,
            structural_free_mask=proposal_state.structural_free_mask,
            structural_obstacle_mask=proposal_state.structural_obstacle_mask,
            unknown_mask=proposal_state.unknown_mask,
            distance_m=proposal_state.distance_m,
            config=self.config,
            proposal_semantic_labels=proposal_semantic_labels,
        )
        rooms: List[RoomMask] = []
        for label_id in sorted(int(v) for v in np.unique(final_labels) if int(v) > 0):
            mask = final_labels == label_id
            room = self._room_from_mask("pending", mask, proposal_state.unknown_mask, doorway_edges, int(proposal_state.step), label_id)
            room.metadata["proposal_labels"] = sorted(int(v) for v in np.unique(labels[mask]) if int(v) > 0)
            room.metadata["algorithm"] = "rose2_structure"
            room.metadata["segmentation_source"] = "rose2_structure"
            rooms.append(room)
        rooms = self._assign_stable_ids(rooms, int(proposal_state.step))
        debug = dict(proposal_state.debug or {})
        debug.update(
            {
                "algorithm": "rose2_structure",
                "source": "rose2_structure",
                "step": int(proposal_state.step),
                "proposal_room_count": int(len([v for v in np.unique(labels) if int(v) > 0])),
                "final_room_count": int(len(rooms)),
                "room_count": int(len(rooms)),
                "rooms": [room_mask_to_debug(room) for room in rooms],
                "merge_split_decisions": list(merge_debug.get("adjacency_decisions") or []),
                "merge_operations": list(merge_debug.get("merge_operations") or []),
                "functional_split_edges": list(merge_debug.get("functional_split_edges") or []),
                "adjacency_evidence": list(merge_debug.get("adjacency_evidence") or []),
                "adjacency_decisions": list(merge_debug.get("adjacency_decisions") or []),
                "doorway_edges": list(merge_debug.get("doorway_edges") or []),
                "num_final_rooms": int(len(rooms)),
            }
        )
        self.last_debug = debug
        self._previous = {room.room_id: room for room in rooms}
        self._last_live_ids = {room.room_id for room in rooms if not room.stale}
        if bool(self.config.debug_dump):
            self._write_debug_dump(int(proposal_state.step), proposal_state, rooms, debug)
        return rooms

    def _build_premerge_zone_proposals(
        self,
        physical_labels: np.ndarray,
        free: np.ndarray,
        unknown: np.ndarray,
        step: int,
    ) -> Tuple[np.ndarray, List[RoomMask]]:
        labels = np.asarray(physical_labels, dtype=np.int32)
        proposal_labels = np.zeros_like(labels, dtype=np.int32)
        proposal_rooms: List[RoomMask] = []
        next_label = 1
        for physical_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            mask = labels == physical_id
            submasks = self._functional_zone_candidates(mask)
            for submask in submasks:
                if not np.any(submask):
                    continue
                proposal_labels[submask] = next_label
                room = self._room_from_mask("proposal_%04d" % next_label, submask, unknown, [], int(step), next_label)
                room.metadata["label_id"] = int(next_label)
                room.metadata["physical_room_label"] = int(physical_id)
                room.metadata["is_premerge_proposal"] = True
                room.metadata["proposal_labels"] = [int(next_label)]
                room.metadata["algorithm"] = "rose2_structure"
                proposal_rooms.append(room)
                next_label += 1
        if not proposal_rooms and np.any(free):
            proposal_labels[free] = 1
            room = self._room_from_mask("proposal_0001", free, unknown, [], int(step), 1)
            room.metadata["label_id"] = 1
            room.metadata["physical_room_label"] = 1
            room.metadata["is_premerge_proposal"] = True
            room.metadata["algorithm"] = "rose2_structure"
            proposal_rooms.append(room)
        return proposal_labels, proposal_rooms

    def _functional_zone_candidates(self, mask: np.ndarray) -> List[np.ndarray]:
        arr = np.asarray(mask, dtype=bool)
        area_m2 = float(np.count_nonzero(arr)) * float(self.config.resolution_m) ** 2
        min_split_area = max(float(self.config.min_room_area_m2) * 2.5, 8.0)
        if area_m2 < min_split_area:
            return [arr]
        rr, cc = np.nonzero(arr)
        if rr.size == 0:
            return [arr]
        height = int(np.max(rr) - np.min(rr) + 1)
        width = int(np.max(cc) - np.min(cc) + 1)
        left = np.zeros_like(arr)
        right = np.zeros_like(arr)
        if width >= height:
            split = int(round(float(np.median(cc))))
            left = arr & (np.indices(arr.shape)[1] <= split)
            right = arr & (np.indices(arr.shape)[1] > split)
        else:
            split = int(round(float(np.median(rr))))
            left = arr & (np.indices(arr.shape)[0] <= split)
            right = arr & (np.indices(arr.shape)[0] > split)
        min_cells = max(1, int(round(float(self.config.min_room_area_m2) / max(float(self.config.resolution_m) ** 2, 1e-9))))
        if np.count_nonzero(left) < min_cells or np.count_nonzero(right) < min_cells:
            return [arr]
        return [left, right]

    def _room_from_mask(
        self,
        room_id: str,
        mask: np.ndarray,
        unknown: np.ndarray,
        doorway_edges: Sequence[Mapping[str, object]],
        step: int,
        label_id: int,
    ) -> RoomMask:
        arr = np.asarray(mask, dtype=bool)
        rr, cc = np.nonzero(arr)
        if rr.size:
            centroid_grid = (float(np.mean(rr)), float(np.mean(cc)))
            centroid_xy = self._grid_to_world(float(centroid_grid[0]), float(centroid_grid[1]))
        else:
            centroid_grid = (0.0, 0.0)
            centroid_xy = (0.0, 0.0)
        boundary = _mask_boundary(arr)
        unknown_fraction = (
            float(np.count_nonzero(np.asarray(unknown, dtype=bool) & _dilate(boundary))) / max(1.0, float(np.count_nonzero(boundary)))
            if np.any(boundary)
            else 0.0
        )
        confidence = float(np.clip(1.0 - 0.5 * unknown_fraction, 0.0, 1.0))
        room = RoomMask(
            room_id=str(room_id),
            mask=arr.copy(),
            centroid_xy=centroid_xy,
            area_m2=float(np.count_nonzero(arr)) * float(self.config.resolution_m) ** 2,
            boundary_unknown_fraction=unknown_fraction,
            doorway_edges=[dict(edge) for edge in doorway_edges],
            confidence=confidence,
            source="rose2_structure",
            observed_free_cells=int(np.count_nonzero(arr)),
            mask_confidence=confidence,
            is_partial=unknown_fraction > 0.35,
            step=int(step),
            metadata={
                "label_id": int(label_id),
                "centroid_grid": [float(centroid_grid[0]), float(centroid_grid[1])],
                "segmentation_source": "rose2_structure",
                "polygon_xy": _polygon_from_mask(arr, self._grid_to_world),
            },
        )
        return room

    def _assign_stable_ids(self, rooms: Sequence[RoomMask], step: int) -> List[RoomMask]:
        assigned: List[RoomMask] = []
        used_prev: set[str] = set()
        for room in rooms:
            best_id = None
            best_iou = 0.0
            for prev_id, prev in self._previous.items():
                if prev_id in used_prev:
                    continue
                iou = _mask_iou(room.mask, prev.mask)
                if iou > best_iou:
                    best_id = prev_id
                    best_iou = iou
            if best_id is not None and best_iou >= float(self.config.id_iou_threshold):
                room.room_id = best_id
                used_prev.add(best_id)
            else:
                room.room_id = "room_%04d" % self._next_room_index
                self._next_room_index += 1
            room.step = int(step)
            assigned.append(room)
        return assigned

    def _grid_to_world(self, row: float, col: float) -> Tuple[float, float]:
        info = self.config.map_info
        if info is None:
            return (float(col) * float(self.config.resolution_m), float(row) * float(self.config.resolution_m))
        return grid_to_world_xy(int(round(row)), int(round(col)), info)

    def _debug_from_structure(self, structure: StructureExtractionResult, proposal_labels: np.ndarray, physical_labels: np.ndarray) -> dict:
        room_polygons = []
        for label_id in sorted(int(v) for v in np.unique(physical_labels) if int(v) > 0):
            mask = physical_labels == label_id
            room_polygons.append({"room_label": int(label_id), "polygon_xy": _polygon_from_mask(mask, self._grid_to_world)})
        return {
            "algorithm": "rose2_structure",
            "source": "rose2_structure",
            "segmentation_source": "rose2_structure",
            "dominant_directions_rad": [float(v) for v in structure.dominant_directions_rad],
            "structural_score": structure.structural_score,
            "clean_structure_map": structure.clean_structure_map,
            "hough_segments": list(structure.hough_segments),
            "wall_clusters": list(structure.wall_clusters),
            "representative_lines": list(structure.representative_lines),
            "faces": list(structure.faces),
            "face_adjacency_edges": list(structure.face_adjacency_edges),
            "room_polygons": room_polygons,
            "voronoi_topology_debug": dict(structure.topology_debug or {}),
            "timing_ms": dict(structure.timing_ms),
            "proposal_room_count": int(len([v for v in np.unique(proposal_labels) if int(v) > 0])),
            "num_hough_segments": int(len(structure.hough_segments)),
            "num_wall_clusters": int(len(structure.wall_clusters)),
            "num_representative_lines": int(len(structure.representative_lines)),
            "num_physical_rooms": int(len([v for v in np.unique(physical_labels) if int(v) > 0])),
            "num_final_rooms": 0,
            "proposal_room_masks": _proposal_masks_debug(proposal_labels),
            "merge_split_decisions": [],
        }

    def _empty_debug(self) -> dict:
        return {
            "algorithm": "rose2_structure",
            "source": "rose2_structure",
            "room_count": 0,
            "step": None,
            "rooms": [],
            "dominant_directions_rad": [],
            "num_hough_segments": 0,
            "num_wall_clusters": 0,
            "num_representative_lines": 0,
            "num_physical_rooms": 0,
            "num_final_rooms": 0,
            "merge_split_decisions": [],
        }

    def _write_debug_dump(self, step: int, proposal_state: RoomProposalState, rooms: Sequence[RoomMask], debug: Mapping[str, object]) -> None:
        from isaac_bench.mapping.room_segmentation_debug import save_rose2_roomseg_debug

        out_dir = Path(str(self.config.debug_dir or "debug/roomseg_rose2"))
        save_rose2_roomseg_debug(
            out_dir=out_dir,
            episode_id="episode",
            step=step,
            occupancy=np.asarray(proposal_state.structural_obstacle_mask, dtype=bool),
            room_masks=rooms,
            debug=debug,
            room_labels={},
        )


def room_mask_to_debug(room: RoomMask) -> dict:
    return {
        "room_id": room.room_id,
        "area_m2": float(room.area_m2),
        "centroid_xy": [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
        "source": "rose2_structure",
        "boundary_confidence": float(room.mask_confidence),
        "functional_split": bool((room.metadata or {}).get("functional_split", False)),
        "polygon_xy": list((room.metadata or {}).get("polygon_xy", [])),
    }


def _proposal_masks_debug(labels: np.ndarray) -> List[dict]:
    out = []
    arr = np.asarray(labels, dtype=np.int32)
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        mask = arr == label_id
        out.append({"label_id": int(label_id), "mask": mask.astype(np.uint8).tolist(), "cell_count": int(np.count_nonzero(mask))})
    return out


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    neighbors = padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]
    return arr & ~neighbors


def _dilate(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    out = np.zeros_like(arr)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr : 1 + dr + arr.shape[0], 1 + dc : 1 + dc + arr.shape[1]]
    return out


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape:
        return 0.0
    inter = float(np.count_nonzero(aa & bb))
    union = float(np.count_nonzero(aa | bb))
    return 0.0 if union <= 0 else inter / union


def _polygon_from_mask(mask: np.ndarray, grid_to_world) -> List[Tuple[float, float]]:
    arr = np.asarray(mask, dtype=bool)
    if not np.any(arr):
        return []
    rr, cc = np.nonzero(arr)
    r0, r1 = int(np.min(rr)), int(np.max(rr))
    c0, c1 = int(np.min(cc)), int(np.max(cc))
    corners = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    return [tuple(float(v) for v in grid_to_world(float(r), float(c))) for r, c in corners]


def rose2_debug_json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): rose2_debug_json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return rose2_debug_json_ready(value.astype(float).tolist() if value.dtype.kind == "f" else value.astype(int).tolist())
    if isinstance(value, (list, tuple)):
        return [rose2_debug_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return rose2_debug_json_ready(value.item())
    if isinstance(value, float):
        return None if not np.isfinite(value) else float(value)
    return value


def write_rose2_layers_json(path: str | Path, payload: Mapping[str, object]) -> None:
    Path(path).write_text(json.dumps(rose2_debug_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
