import pytest

from isaac_bench.graph.room_semantics import ObjectEvidence, VLMRoomLabeler
from isaac_bench.mapping.room_segmentation import RoomMask


class FakeRoomLLM:
    def __init__(self, response):
        self.response = response

    def complete_json(self, prompt: str):
        assert "allowed_categories" in prompt
        assert "choose unknown" in prompt.lower()
        return self.response


def _room(partial=False):
    import numpy as np

    return RoomMask(
        room_id="room_0001",
        mask=np.ones((4, 4), dtype=bool),
        centroid_xy=(0.0, 0.0),
        area_m2=4.0,
        boundary_unknown_fraction=0.1 if not partial else 0.7,
        doorway_edges=[],
        confidence=0.8,
        is_partial=partial,
    )


def test_empty_or_weak_evidence_is_unknown_in_debug_labeler():
    label = VLMRoomLabeler(client=None, require_backend=False).label_room(_room(), [], None)

    assert label.category == "unknown"
    assert label.backend == "deterministic_debug"
    assert label.unknown_reason == "insufficient_or_ambiguous_evidence"


def test_confident_bathroom_evidence_can_be_bathroom():
    label = VLMRoomLabeler(client=None, require_backend=False).label_room(
        _room(),
        [
            ObjectEvidence("toilet", mean_confidence=0.9, hits=3),
            ObjectEvidence("sink", mean_confidence=0.8, hits=2),
            ObjectEvidence("bathtub", mean_confidence=0.8, hits=2),
        ],
        None,
    )

    assert label.category == "bathroom"


def test_confident_bedroom_evidence_can_be_bedroom():
    label = VLMRoomLabeler(client=None, require_backend=False).label_room(
        _room(),
        [
            ObjectEvidence("bed", mean_confidence=0.9, hits=3),
            ObjectEvidence("nightstand", mean_confidence=0.8, hits=2),
            ObjectEvidence("wardrobe", mean_confidence=0.8, hits=2),
        ],
        None,
    )

    assert label.category == "bedroom"


def test_invalid_category_from_vlm_is_forced_unknown():
    labeler = VLMRoomLabeler(client=FakeRoomLLM({"room_id": "room_0001", "category": "garage", "confidence": 0.99}))

    label = labeler.label_room(_room(), [ObjectEvidence("chair", hits=2), ObjectEvidence("table", hits=2)], None)

    assert label.category == "unknown"
    assert label.unknown_reason == "invalid_category"


def test_low_confidence_vlm_label_is_forced_unknown():
    labeler = VLMRoomLabeler(client=FakeRoomLLM({"room_id": "room_0001", "category": "kitchen", "confidence": 0.3}))

    label = labeler.label_room(_room(), [ObjectEvidence("stove", hits=2), ObjectEvidence("sink", hits=2)], None)

    assert label.category == "unknown"
    assert label.unknown_reason == "low_confidence"


def test_ambiguous_ranked_vlm_label_is_forced_unknown():
    labeler = VLMRoomLabeler(
        client=FakeRoomLLM(
            {
                "room_id": "room_0001",
                "category": "kitchen",
                "confidence": 0.8,
                "ranked_categories": [
                    {"category": "kitchen", "confidence": 0.80},
                    {"category": "dining_room", "confidence": 0.74},
                ],
            }
        )
    )

    label = labeler.label_room(_room(), [ObjectEvidence("table", hits=2), ObjectEvidence("chair", hits=2)], None)

    assert label.category == "unknown"
    assert label.unknown_reason == "ambiguous_ranked_categories"


def test_strict_missing_vlm_backend_raises():
    with pytest.raises(RuntimeError, match="requires an enabled room VLM backend"):
        VLMRoomLabeler(client=None, require_backend=True).label_room(_room(), [], None)
