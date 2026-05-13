from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
