from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

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

    def find_match(self, category: str, center_world: Sequence[float]) -> Optional[ObjectNode]:
        center = np.asarray(center_world, dtype=np.float32)
        best = None
        best_dist = float("inf")
        for node in self.nodes:
            if node.category != category:
                continue
            dist = float(np.linalg.norm(center - np.asarray(node.center_world, dtype=np.float32)))
            if dist < self.merge_radius_m and dist < best_dist:
                best = node
                best_dist = dist
        return best

    def update(self, detections: List[Detection3D], step_id: int, map_info: Optional[MapInfo] = None) -> List[ObjectNode]:
        changed: List[ObjectNode] = []
        for det in detections:
            center = tuple(float(v) for v in det.center_world)
            matched = self.find_match(det.category, center)
            if map_info is not None:
                center_grid = world_xy_to_grid(center[0], center[1], map_info)
            else:
                center_grid = (0, 0)
            if matched is None:
                node = ObjectNode(
                    node_id=self._next_id,
                    category=det.category,
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
        return changed

    def to_dicts(self) -> List[dict]:
        return [node.to_dict() for node in self.nodes]
