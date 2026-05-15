from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class Detection2D:
    category: str
    raw_label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    class_id: Optional[int] = None
    mask: Optional[np.ndarray] = None


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
