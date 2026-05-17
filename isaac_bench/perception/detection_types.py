from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np


MIN_VALID_DETECTION_CONFIDENCE = 0.65


def detection_confidence_is_valid(confidence: float, min_confidence: float = MIN_VALID_DETECTION_CONFIDENCE) -> bool:
    return float(confidence) > float(min_confidence)


def bbox_touches_image_edge(
    bbox_xyxy: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin_px: float = 2,
    margin_ratio: float = 0.0,
) -> bool:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    margin = max(float(margin_px), float(margin_ratio) * float(max(int(image_width), int(image_height))))
    return bool(
        x1 <= margin
        or y1 <= margin
        or x2 >= float(int(image_width) - 1) - margin
        or y2 >= float(int(image_height) - 1) - margin
    )


@dataclass
class Detection2D:
    category: str
    raw_label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    class_id: Optional[int] = None
    mask: Optional[np.ndarray] = None
    bbox_touches_edge: bool = False
    used_for_object_track: bool = True
    reject_reason: Optional[str] = None


@dataclass
class Detection3D:
    category: str
    raw_label: str
    confidence: float
    center_world: Tuple[float, float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    point_cloud_world: Optional[np.ndarray] = None
    bbox_world: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


@dataclass
class FusedInstance:
    instance_id: str
    category: str
    node_type: Literal["object", "room"]
    confidence: float
    point_cloud_world: np.ndarray
    bbox_world: np.ndarray
    center_world: np.ndarray
    last_mask: Optional[np.ndarray]
    last_seen_step: int
    observed_count: int
    source: str = "yolo_world_sam2_depth_fusion"
    class_conf_sums: Dict[str, float] = field(default_factory=dict)
    class_hits: Dict[str, int] = field(default_factory=dict)
    valid_detection_count: int = 0
    total_conf_sum: float = 0.0
    edge_rejected_count: int = 0

    @property
    def stable_category(self) -> str:
        if not self.class_conf_sums:
            return self.category
        return max(sorted(self.class_conf_sums), key=lambda key: float(self.class_conf_sums[key]))

    @property
    def winner_detection_count(self) -> int:
        return int(self.class_hits.get(self.stable_category, self.observed_count))

    @property
    def mean_confidence(self) -> float:
        hits = max(1, self.winner_detection_count)
        return float(self.class_conf_sums.get(self.stable_category, float(self.confidence) * hits)) / float(hits)
