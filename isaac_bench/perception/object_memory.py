from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.mapping.coordinate_transform import MapInfo, world_xy_to_grid
from isaac_bench.perception.detection_types import Detection3D


@dataclass
class ObjectNode:
    node_id: int
    category: str
    center_world: Tuple[float, float, float]
    center_grid: Tuple[int, int]
    confidence: float
    observed_count: int
    last_seen_step: int
    raw_label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ObjectMemory:
    def __init__(self, merge_radius_m: float = 0.5):
        self.merge_radius_m = float(merge_radius_m)
        self.nodes: List[ObjectNode] = []
        self._next_id = 0

    def reset(self) -> None:
        self.nodes = []
        self._next_id = 0

    @staticmethod
    def _category_key(category: str) -> str:
        return normalize_category(category)

    @staticmethod
    def _xy_distance(a: Sequence[float], b: Sequence[float]) -> float:
        aa = np.asarray(a[:2], dtype=np.float32)
        bb = np.asarray(b[:2], dtype=np.float32)
        return float(np.linalg.norm(aa - bb))

    def find_match(self, category: str, center_world: Sequence[float]) -> Optional[ObjectNode]:
        category_key = self._category_key(category)
        best = None
        best_dist = float("inf")
        for node in self.nodes:
            if self._category_key(node.category) != category_key:
                continue
            dist = self._xy_distance(center_world, node.center_world)
            if dist < self.merge_radius_m and dist < best_dist:
                best = node
                best_dist = dist
        return best

    def update(self, detections: List[Detection3D], step_id: int, map_info: Optional[MapInfo] = None) -> List[ObjectNode]:
        changed: List[ObjectNode] = []
        for det in detections:
            center = tuple(float(v) for v in det.center_world)
            category = self._category_key(det.category)
            matched = self.find_match(category, center)
            if map_info is not None:
                center_grid = world_xy_to_grid(center[0], center[1], map_info)
            else:
                center_grid = (0, 0)
            if matched is None:
                node = ObjectNode(
                    node_id=self._next_id,
                    category=category,
                    center_world=center,
                    center_grid=center_grid,
                    confidence=float(det.confidence),
                    observed_count=1,
                    last_seen_step=int(step_id),
                    raw_label=det.raw_label,
                )
                self._next_id += 1
                self.nodes.append(node)
                changed.append(node)
                continue
            old_count = matched.observed_count
            old_center = np.asarray(matched.center_world, dtype=np.float32)
            new_center = np.asarray(center, dtype=np.float32)
            merged = (old_center * old_count + new_center) / float(old_count + 1)
            matched.center_world = tuple(float(v) for v in merged)
            if map_info is not None:
                matched.center_grid = world_xy_to_grid(float(merged[0]), float(merged[1]), map_info)
            else:
                matched.center_grid = center_grid
            matched.confidence = max(float(matched.confidence), float(det.confidence))
            matched.observed_count += 1
            matched.last_seen_step = int(step_id)
            changed.append(matched)
        self.dedupe(map_info=map_info)
        return changed

    def dedupe(self, map_info: Optional[MapInfo] = None) -> None:
        if len(self.nodes) < 2:
            return
        merged: List[ObjectNode] = []
        for node in sorted(self.nodes, key=lambda item: int(item.node_id)):
            node.category = self._category_key(node.category)
            match = None
            best_dist = float("inf")
            for existing in merged:
                if self._category_key(existing.category) != node.category:
                    continue
                dist = self._xy_distance(existing.center_world, node.center_world)
                if dist < self.merge_radius_m and dist < best_dist:
                    match = existing
                    best_dist = dist
            if match is None:
                merged.append(node)
                continue
            self._merge_node_into(match, node, map_info=map_info)
        self.nodes = merged

    def _merge_node_into(self, target: ObjectNode, duplicate: ObjectNode, map_info: Optional[MapInfo] = None) -> None:
        target_count = max(1, int(target.observed_count))
        dup_count = max(1, int(duplicate.observed_count))
        total = float(target_count + dup_count)
        target_center = np.asarray(target.center_world, dtype=np.float32)
        duplicate_center = np.asarray(duplicate.center_world, dtype=np.float32)
        merged_center = (target_center * float(target_count) + duplicate_center * float(dup_count)) / total
        target.center_world = tuple(float(v) for v in merged_center)
        if map_info is not None:
            target.center_grid = world_xy_to_grid(float(merged_center[0]), float(merged_center[1]), map_info)
        target.confidence = max(float(target.confidence), float(duplicate.confidence))
        target.observed_count = int(target_count + dup_count)
        target.last_seen_step = max(int(target.last_seen_step), int(duplicate.last_seen_step))
        if not target.raw_label and duplicate.raw_label:
            target.raw_label = duplicate.raw_label

    def to_dicts(self) -> List[dict]:
        return [node.to_dict() for node in self.nodes]
