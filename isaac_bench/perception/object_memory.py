from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.mapping.coordinate_transform import MapInfo, world_xy_to_grid
from isaac_bench.perception.detection_types import Detection3D, FusedInstance


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
    point_cloud_world: Optional[np.ndarray] = None
    bbox_world: Optional[np.ndarray] = None
    last_mask: Optional[np.ndarray] = None
    source_instance_id: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": int(self.node_id),
            "category": self.category,
            "center_world": tuple(float(v) for v in self.center_world),
            "center_grid": tuple(int(v) for v in self.center_grid),
            "confidence": float(self.confidence),
            "observed_count": int(self.observed_count),
            "last_seen_step": int(self.last_seen_step),
            "raw_label": self.raw_label,
            "point_cloud_world": _array_to_list(self.point_cloud_world),
            "bbox_world": _array_to_list(self.bbox_world),
            "source_instance_id": self.source_instance_id,
            "source": self.source,
        }


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
                    point_cloud_world=_copy_array(det.point_cloud_world),
                    bbox_world=_copy_array(det.bbox_world),
                    last_mask=_copy_array(det.mask),
                    source=getattr(det, "source", ""),
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
            matched.point_cloud_world = _merge_point_clouds(matched.point_cloud_world, det.point_cloud_world)
            matched.bbox_world = _copy_array(det.bbox_world) if det.bbox_world is not None else _bbox_from_points(matched.point_cloud_world)
            matched.last_mask = _copy_array(det.mask)
            matched.observed_count += 1
            matched.last_seen_step = int(step_id)
            changed.append(matched)
        self.dedupe(map_info=map_info)
        return changed

    def update_fused_instances(
        self,
        instances: Sequence[FusedInstance],
        step_id: int,
        map_info: Optional[MapInfo] = None,
    ) -> List[ObjectNode]:
        detections: List[Detection3D] = []
        instance_ids: List[str] = []
        sources: List[str] = []
        for instance in instances:
            if instance.node_type != "object":
                continue
            detections.append(
                Detection3D(
                    category=instance.category,
                    raw_label=instance.category,
                    confidence=float(instance.confidence),
                    center_world=tuple(float(v) for v in instance.center_world),
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    point_cloud_world=instance.point_cloud_world,
                    bbox_world=instance.bbox_world,
                    mask=instance.last_mask,
                )
            )
            instance_ids.append(instance.instance_id)
            sources.append(instance.source)
        changed = self.update(detections, step_id=step_id, map_info=map_info)
        for node, instance_id, source in zip(changed, instance_ids, sources):
            node.source_instance_id = instance_id
            node.source = source
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
        target.point_cloud_world = _merge_point_clouds(target.point_cloud_world, duplicate.point_cloud_world)
        target.bbox_world = _bbox_from_points(target.point_cloud_world) if target.point_cloud_world is not None else target.bbox_world
        if duplicate.last_mask is not None:
            target.last_mask = _copy_array(duplicate.last_mask)
        target.observed_count = int(target_count + dup_count)
        target.last_seen_step = max(int(target.last_seen_step), int(duplicate.last_seen_step))
        if not target.raw_label and duplicate.raw_label:
            target.raw_label = duplicate.raw_label
        if not target.source_instance_id and duplicate.source_instance_id:
            target.source_instance_id = duplicate.source_instance_id
        if not target.source and duplicate.source:
            target.source = duplicate.source

    def to_dicts(self) -> List[dict]:
        return [node.to_dict() for node in self.nodes]


def _copy_array(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value).copy()


def _array_to_list(value: Optional[np.ndarray]):
    if value is None:
        return None
    return np.asarray(value).tolist()


def _merge_point_clouds(a: Optional[np.ndarray], b: Optional[np.ndarray], max_points: int = 4096) -> Optional[np.ndarray]:
    if a is None and b is None:
        return None
    parts = [np.asarray(item, dtype=np.float32) for item in (a, b) if item is not None and len(item) > 0]
    if not parts:
        return None
    merged = np.concatenate(parts, axis=0)
    if len(merged) <= int(max_points):
        return merged.copy()
    indices = np.linspace(0, len(merged) - 1, int(max_points)).astype(np.int64)
    return merged[indices].copy()


def _bbox_from_points(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or len(points) == 0:
        return None
    arr = np.asarray(points, dtype=np.float32)
    return np.stack([np.min(arr, axis=0), np.max(arr, axis=0)], axis=0)
