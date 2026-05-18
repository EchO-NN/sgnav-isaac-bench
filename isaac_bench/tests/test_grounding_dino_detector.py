from __future__ import annotations

import numpy as np

from isaac_bench.perception.grounding_dino_detector import grounding_dino_caption, _nms_numpy
from isaac_bench.perception.yolo_world_detector import build_detector
from isaac_bench.perception.detector_base import DryRunDetector


def test_grounding_dino_caption_uses_period_separated_vocabulary():
    caption = grounding_dino_caption(["ceiling_light", "sofa", "sofa", "unknown"])

    assert caption == "ceiling light. sofa."


def test_grounding_dino_nms_keeps_highest_overlapping_box():
    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [30.0, 30.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    assert _nms_numpy(boxes, scores, 0.5) == [0, 2]


def test_detector_factory_keeps_dry_run_without_heavy_imports():
    detector = build_detector("dry_run", "", conf=0.45)

    assert isinstance(detector, DryRunDetector)
