from __future__ import annotations

import numpy as np

from isaac_bench.perception.grounding_dino_detector import (
    _match_phrase_to_vocab,
    _nms_numpy,
    grounding_dino_caption,
    grounding_dino_vocabulary,
)
from isaac_bench.perception.yolo_world_detector import build_detector
from isaac_bench.perception.detector_base import DryRunDetector


def test_grounding_dino_caption_uses_period_separated_vocabulary():
    caption = grounding_dino_caption(["ceiling_light", "sofa", "sofa", "unknown"])

    assert caption == "ceiling light. sofa."


def test_grounding_dino_vocabulary_always_includes_door_and_doorframe():
    vocab = grounding_dino_vocabulary(["sofa", "door", "unknown", "sofa"])

    assert vocab == ["sofa", "door", "doorframe"]


def test_grounding_dino_caption_prompts_doorframe_as_natural_text():
    caption = grounding_dino_caption(grounding_dino_vocabulary(["sofa"]))

    assert caption == "sofa. door. door frame."


def test_grounding_dino_phrase_door_frame_matches_fixed_doorframe_category():
    class_id, label = _match_phrase_to_vocab("door frame", grounding_dino_vocabulary(["sofa"]))

    assert class_id == 2
    assert label == "doorframe"


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
