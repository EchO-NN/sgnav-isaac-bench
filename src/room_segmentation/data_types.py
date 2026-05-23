from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


GridCell = Tuple[int, int]


@dataclass
class GridSpec:
    resolution_m: float
    origin_xy: Tuple[float, float]
    width: int
    height: int


@dataclass
class VerticalMapState:
    """3D vertical evidence compressed into 2D probability layers."""

    p_free: np.ndarray
    p_occupied: np.ndarray
    p_unknown: np.ndarray
    p_endpoint: np.ndarray
    p_structural_wall: np.ndarray
    observation_count: np.ndarray
    last_observed_frame: np.ndarray


@dataclass
class StructuralMap:
    """Clean but probabilistic structural representation."""

    p_wall: np.ndarray
    p_free: np.ndarray
    p_unknown: np.ndarray
    hard_wall_mask: np.ndarray
    observed_free_mask: np.ndarray
    frontier_mask: np.ndarray
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class FreeSpaceState:
    observed_free_mask: np.ndarray
    free_eroded_mask: np.ndarray
    frontier_mask: np.ndarray
    distance_transform_m: np.ndarray
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SkeletonNode:
    node_id: int
    xy: Tuple[float, float]
    uv: Tuple[int, int]
    clearance_m: float
    degree: int = 0


@dataclass
class SkeletonEdge:
    src: int
    dst: int
    length_m: float
    mean_clearance_m: float


@dataclass
class SkeletonGraph:
    nodes: List[SkeletonNode]
    edges: List[SkeletonEdge]
    adjacency: Dict[int, List[int]]
    node_id_map: np.ndarray
    skeleton_mask: np.ndarray
    distance_transform_m: np.ndarray


@dataclass
class CutCandidate:
    candidate_id: int
    center_uv: Tuple[int, int]
    center_xy: Tuple[float, float]
    normal_xy: Tuple[float, float]
    tangent_xy: Tuple[float, float]
    width_m: float
    cut_cells: List[Tuple[int, int]]
    left_seed_cells: List[Tuple[int, int]]
    right_seed_cells: List[Tuple[int, int]]
    kind: str = "skeleton_local_minimum"

    width_score: float = 0.0
    wall_support_score: float = 0.0
    visibility_drop_score: float = 0.0
    graph_conductance_score: float = 0.0
    temporal_score: float = 0.0
    frontier_penalty: float = 0.0
    alcove_penalty: float = 0.0
    area_balance_score: float = 0.0
    corridor_consistency_score: float = 0.0

    final_score: float = 0.0
    is_hard_separator: bool = False
    is_soft_separator: bool = False
    temporal_observations: int = 0
    debug: Dict[str, object] = field(default_factory=dict)

    def mask(self, shape: Tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        for row, col in self.cut_cells:
            if 0 <= int(row) < shape[0] and 0 <= int(col) < shape[1]:
                out[int(row), int(col)] = True
        return out

    def to_dict(self) -> dict:
        return {
            "candidate_id": int(self.candidate_id),
            "kind": str(self.kind),
            "center_uv": [int(self.center_uv[0]), int(self.center_uv[1])],
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])],
            "normal_xy": [float(self.normal_xy[0]), float(self.normal_xy[1])],
            "tangent_xy": [float(self.tangent_xy[0]), float(self.tangent_xy[1])],
            "width_m": float(self.width_m),
            "width_score": float(self.width_score),
            "wall_support_score": float(self.wall_support_score),
            "visibility_drop_score": float(self.visibility_drop_score),
            "graph_conductance_score": float(self.graph_conductance_score),
            "temporal_score": float(self.temporal_score),
            "frontier_penalty": float(self.frontier_penalty),
            "alcove_penalty": float(self.alcove_penalty),
            "area_balance_score": float(self.area_balance_score),
            "corridor_consistency_score": float(self.corridor_consistency_score),
            "final_score": float(self.final_score),
            "is_hard_separator": bool(self.is_hard_separator),
            "is_soft_separator": bool(self.is_soft_separator),
            "temporal_observations": int(self.temporal_observations),
            "cut_cell_count": int(len(self.cut_cells)),
            "debug": dict(self.debug),
        }


@dataclass
class PlaceNode:
    node_id: int
    node_type: str
    xy: Tuple[float, float]
    uv: Tuple[int, int]
    room_label: int = -1
    visibility_signature: Optional[object] = None
    semantic_embedding: Optional[np.ndarray] = None


@dataclass
class PlaceEdge:
    src: int
    dst: int
    weight: float
    geodesic_m: float
    visibility_jaccard: float
    line_of_sight: float
    bottleneck_penalty: float


@dataclass
class PlaceGraph:
    nodes: List[PlaceNode]
    edges: List[PlaceEdge]
    adjacency: Dict[int, List[int]]


@dataclass
class RoomInstance:
    room_id: int
    room_type: str
    functional_zone_label: str
    area_m2: float
    centroid_xy: Tuple[float, float]
    confidence: float
    is_corridor: bool
    is_open_space: bool
    boundary_cells: List[Tuple[int, int]]
    connected_room_ids: List[int]
    object_ids: List[int] = field(default_factory=list)


@dataclass
class RoomSegmentationOutput:
    room_id_map: np.ndarray
    room_confidence_map: np.ndarray
    room_soft_masks: Dict[int, np.ndarray]
    corridor_mask: np.ndarray
    open_space_mask: np.ndarray
    functional_zone_map: np.ndarray
    room_instances: List[RoomInstance]
    room_graph_edges: List[Tuple[int, int, float]]
    cut_candidates: List[CutCandidate]
    debug_layers: Dict[str, np.ndarray]


@dataclass
class PartitionResult:
    node_labels: Dict[int, int]
    separator_map: np.ndarray
    seed_cells_by_label: Dict[int, List[Tuple[int, int]]]
    debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class MaskAssignmentResult:
    room_id_map: np.ndarray
    room_confidence_map: np.ndarray
    room_soft_masks: Dict[int, np.ndarray]
    corridor_mask: np.ndarray
    open_space_mask: np.ndarray
    functional_zone_map: np.ndarray
    room_instances: List[RoomInstance]
    room_graph_edges: List[Tuple[int, int, float]]
    debug_layers: Dict[str, np.ndarray] = field(default_factory=dict)

