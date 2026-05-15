from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.perception.detection_types import FusedInstance
from isaac_bench.perception.object_memory import ObjectMemory

PAPER_RELATED_CATEGORY_PAIRS = {
    tuple(sorted((normalize_category(a), normalize_category(b))))
    for a, b in [
        ("bed", "nightstand"),
        ("wardrobe", "dresser"),
        ("bookshelf", "chair"),
        ("counter", "stove"),
        ("table", "chair"),
        ("bathroom sink", "mirror"),
        ("sink", "mirror"),
        ("shower", "bathtub"),
        ("refrigerator", "freezer"),
        ("oven", "microwave"),
        ("washing machine", "dryer"),
        ("sofa", "table"),
        ("desk", "office chair"),
        ("desk", "chair"),
        ("computer", "monitor"),
        ("computer", "tv"),
        ("piano", "bench"),
        ("fireplace", "mantel"),
        ("table", "mirror"),
        ("window", "curtains"),
        ("closet", "hangers"),
        ("bathroom cabinet", "toiletries"),
        ("living room rug", "coffee table"),
        ("kitchen cabinet", "dishes"),
        ("dining room chandelier", "dining table"),
        ("clock", "wall"),
        ("floor lamp", "reading chair"),
        ("couch", "throw pillows"),
        ("bookcase", "books"),
        ("tv", "sofa"),
        ("tv", "table"),
        ("vase", "table"),
        ("lamp", "table"),
    ]
}


@dataclass
class ObjectNode:
    id: str
    category: str
    confidence: float
    point_cloud_world: np.ndarray
    bbox_world: np.ndarray
    center_world: np.ndarray
    observed_count: int
    last_seen_step: int
    room_id: Optional[str] = None


@dataclass
class RoomNode:
    id: str
    room_type: str
    confidence: float
    region_polygon_world: Optional[np.ndarray]
    point_cloud_world: Optional[np.ndarray]
    contained_object_ids: List[str] = field(default_factory=list)


@dataclass
class GroupNode:
    id: str
    category_summary: str
    object_ids: List[str]
    center_world: np.ndarray
    room_id: Optional[str]


@dataclass
class ObjectEdge:
    src_id: str
    dst_id: str
    relation: str
    confidence: float = 1.0
    is_short_edge: bool = False
    source: str = "paper_scene_graph"
    pruning_debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class AffiliationEdge:
    src_id: str
    dst_id: str
    relation: str
    confidence: float = 1.0


class PaperSceneGraph:
    def __init__(self, related_category_pairs: Optional[Iterable[Tuple[str, str]]] = None):
        self.related_category_pairs = {
            tuple(sorted((normalize_category(a), normalize_category(b)))) for a, b in (related_category_pairs or PAPER_RELATED_CATEGORY_PAIRS)
        }
        self.object_nodes: Dict[str, ObjectNode] = {}
        self.room_nodes: Dict[str, RoomNode] = {}
        self.group_nodes: Dict[str, GroupNode] = {}
        self.object_edges: List[ObjectEdge] = []
        self.affiliation_edges: List[AffiliationEdge] = []
        self.version = 0

    def reset(self) -> None:
        self.object_nodes = {}
        self.room_nodes = {}
        self.group_nodes = {}
        self.object_edges = []
        self.affiliation_edges = []
        self.version = 0

    def update_object_and_room_nodes(self, fused_instances: Sequence[FusedInstance]) -> List[ObjectNode]:
        new_objects: List[ObjectNode] = []
        for instance in fused_instances:
            if instance.node_type == "room":
                self._upsert_room(instance)
            else:
                node = self._upsert_object(instance)
                if int(node.observed_count) <= int(instance.observed_count):
                    new_objects.append(node)
        if not self.room_nodes:
            self._ensure_unknown_room()
        self.update_affiliation_edges()
        self.version += 1
        return new_objects

    def update_from_object_memory(self, object_memory: ObjectMemory) -> None:
        for mem_node in object_memory.nodes:
            node_id = "object:%s" % int(mem_node.node_id)
            points = _points_or_center(mem_node.point_cloud_world, mem_node.center_world)
            bbox = _bbox_or_points(mem_node.bbox_world, points)
            self.object_nodes[node_id] = ObjectNode(
                id=node_id,
                category=normalize_category(mem_node.category),
                confidence=float(mem_node.confidence),
                point_cloud_world=points,
                bbox_world=bbox,
                center_world=np.asarray(mem_node.center_world, dtype=np.float32),
                observed_count=int(mem_node.observed_count),
                last_seen_step=int(mem_node.last_seen_step),
            )
        if not self.room_nodes:
            self._ensure_unknown_room()
        self.update_affiliation_edges()
        self.version += 1

    def update_affiliation_edges(self) -> None:
        self.affiliation_edges = []
        for room in self.room_nodes.values():
            room.contained_object_ids = []
        for obj in self.object_nodes.values():
            room = self._find_room_for_object(obj)
            obj.room_id = room.id if room is not None else None
            if room is None:
                continue
            room.contained_object_ids.append(obj.id)
            self.affiliation_edges.append(AffiliationEdge(src_id=obj.id, dst_id=room.id, relation="belongs_to"))
        for group in self.group_nodes.values():
            room_ids = {self.object_nodes[obj_id].room_id for obj_id in group.object_ids if obj_id in self.object_nodes}
            room_ids.discard(None)
            if len(room_ids) == 1:
                group.room_id = next(iter(room_ids))
                self.affiliation_edges.append(AffiliationEdge(src_id=group.id, dst_id=group.room_id, relation="belongs_to"))

    def update_group_nodes(self) -> None:
        adjacency: Dict[str, set] = {obj_id: set() for obj_id in self.object_nodes}
        for edge in self.object_edges:
            src = self.object_nodes.get(edge.src_id)
            dst = self.object_nodes.get(edge.dst_id)
            if src is None or dst is None:
                continue
            pair = tuple(sorted((normalize_category(src.category), normalize_category(dst.category))))
            if pair not in self.related_category_pairs:
                continue
            adjacency[edge.src_id].add(edge.dst_id)
            adjacency[edge.dst_id].add(edge.src_id)
        self.group_nodes = {}
        visited = set()
        for obj_id in sorted(adjacency):
            if obj_id in visited or not adjacency[obj_id]:
                continue
            stack = [obj_id]
            visited.add(obj_id)
            members: List[str] = []
            while stack:
                cur = stack.pop()
                members.append(cur)
                for nxt in adjacency[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            if len(members) < 2:
                continue
            centers = [self.object_nodes[mid].center_world for mid in members]
            categories = sorted({self.object_nodes[mid].category for mid in members})
            rooms = {self.object_nodes[mid].room_id for mid in members}
            rooms.discard(None)
            group_id = "group:%s" % "-".join(mid.split(":", 1)[-1] for mid in sorted(members))
            self.group_nodes[group_id] = GroupNode(
                id=group_id,
                category_summary=", ".join(categories),
                object_ids=sorted(members),
                center_world=np.mean(np.asarray(centers, dtype=np.float32), axis=0),
                room_id=next(iter(rooms)) if len(rooms) == 1 else None,
            )
        self.update_affiliation_edges()
        self.version += 1

    def find_goal_candidates(self, goal_category: str) -> List[ObjectNode]:
        goal = normalize_category(goal_category)
        return [node for node in self.object_nodes.values() if normalize_category(node.category) == goal]

    def _upsert_object(self, instance: FusedInstance) -> ObjectNode:
        node_id = "object:%s" % instance.instance_id
        node = ObjectNode(
            id=node_id,
            category=normalize_category(instance.category),
            confidence=float(instance.confidence),
            point_cloud_world=np.asarray(instance.point_cloud_world, dtype=np.float32).copy(),
            bbox_world=np.asarray(instance.bbox_world, dtype=np.float32).copy(),
            center_world=np.asarray(instance.center_world, dtype=np.float32).copy(),
            observed_count=int(instance.observed_count),
            last_seen_step=int(instance.last_seen_step),
        )
        self.object_nodes[node_id] = node
        return node

    def _upsert_room(self, instance: FusedInstance) -> RoomNode:
        room_type = normalize_category(instance.category)
        room_id = "room:%s" % instance.instance_id
        room = RoomNode(
            id=room_id,
            room_type=room_type,
            confidence=float(instance.confidence),
            region_polygon_world=None,
            point_cloud_world=np.asarray(instance.point_cloud_world, dtype=np.float32).copy(),
            contained_object_ids=[],
        )
        self.room_nodes[room_id] = room
        return room

    def _ensure_unknown_room(self) -> None:
        self.room_nodes.setdefault(
            "room:unknown_room",
            RoomNode(
                id="room:unknown_room",
                room_type="unknown_room",
                confidence=0.0,
                region_polygon_world=None,
                point_cloud_world=None,
                contained_object_ids=[],
            ),
        )

    def _find_room_for_object(self, obj: ObjectNode) -> Optional[RoomNode]:
        concrete_rooms = [room for room in self.room_nodes.values() if room.point_cloud_world is not None and len(room.point_cloud_world) > 0]
        for room in concrete_rooms:
            room_bbox = _bbox_from_points(room.point_cloud_world)
            if _object_inside_room(obj, room_bbox):
                return room
        return self.room_nodes.get("room:unknown_room")


def _points_or_center(points: Optional[np.ndarray], center_world: Sequence[float]) -> np.ndarray:
    if points is not None and len(points) > 0:
        return np.asarray(points, dtype=np.float32).copy()
    return np.asarray([center_world], dtype=np.float32)


def _bbox_or_points(bbox: Optional[np.ndarray], points: np.ndarray) -> np.ndarray:
    if bbox is not None:
        arr = np.asarray(bbox, dtype=np.float32)
        if arr.shape == (2, 3):
            return arr.copy()
    return _bbox_from_points(points)


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    return np.stack([np.min(arr, axis=0), np.max(arr, axis=0)], axis=0)


def _object_inside_room(obj: ObjectNode, room_bbox: np.ndarray) -> bool:
    center = np.asarray(obj.center_world, dtype=np.float32)
    if np.all(center >= room_bbox[0]) and np.all(center <= room_bbox[1]):
        return True
    points = np.asarray(obj.point_cloud_world, dtype=np.float32)
    if len(points) == 0:
        return False
    inside = np.all((points >= room_bbox[0]) & (points <= room_bbox[1]), axis=1)
    return float(np.count_nonzero(inside)) / float(len(points)) >= 0.5
