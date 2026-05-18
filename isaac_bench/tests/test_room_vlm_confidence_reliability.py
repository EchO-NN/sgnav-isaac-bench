import pytest

from isaac_bench.graph.room_semantics import ObjectEvidence, VLMRoomLabeler
from isaac_bench.mapping.room_segmentation import RoomMask


class FakeRoomLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, prompt):
        return self.payload


def _room(partial=False):
    return RoomMask(
        room_id="room_0001",
        mask=None,
        centroid_xy=(0.0, 0.0),
        area_m2=10.0,
        boundary_unknown_fraction=0.6 if partial else 0.1,
        doorway_edges=[],
        confidence=0.8,
        mask_confidence=0.8,
        is_partial=partial,
    )


def test_high_vlm_self_confidence_without_evidence_keeps_category_with_low_reliability():
    labeler = VLMRoomLabeler(
        client=FakeRoomLLM(
            {
                "room_id": "room_0001",
                "category": "living_room",
                "confidence": 0.99,
                "ranked_categories": [{"category": "living_room", "confidence": 0.99}],
            }
        )
    )

    label = labeler.label_room(_room(), [ObjectEvidence("chair", hits=1, mean_confidence=0.4)], None)

    assert label.category == "living_room"
    assert label.vlm_self_confidence == pytest.approx(0.99)
    assert 0.0 < label.label_reliability < 0.65
    assert label.unknown_reason is None
    assert label.reliability_factors["label_quality_warning"] in {
        "insufficient_or_ambiguous_evidence",
        "no_diagnostic_evidence",
    }


def test_low_self_confidence_keeps_category_even_with_warning():
    labeler = VLMRoomLabeler(
        client=FakeRoomLLM(
            {
                "room_id": "room_0001",
                "category": "bathroom",
                "confidence": 0.2,
                "ranked_categories": [{"category": "bathroom", "confidence": 0.2}],
            }
        )
    )

    label = labeler.label_room(_room(), [ObjectEvidence("toilet", hits=3), ObjectEvidence("sink", hits=3)], None)

    assert label.category == "bathroom"
    assert label.unknown_reason is None
    assert label.reliability_factors["label_quality_warning"] == "low_confidence"


def test_accepted_room_label_uses_evidence_reliability_not_raw_confidence():
    labeler = VLMRoomLabeler(
        client=FakeRoomLLM(
            {
                "room_id": "room_0001",
                "category": "bathroom",
                "confidence": 0.91,
                "ranked_categories": [
                    {"category": "bathroom", "confidence": 0.91},
                    {"category": "unknown", "confidence": 0.05},
                ],
                "supporting_objects": ["toilet", "sink", "bathtub"],
            }
        )
    )

    label = labeler.label_room(
        _room(),
        [ObjectEvidence("toilet", hits=3), ObjectEvidence("sink", hits=3), ObjectEvidence("bathtub", hits=2)],
        None,
    )

    assert label.category == "bathroom"
    assert label.vlm_self_confidence == pytest.approx(0.91)
    assert label.confidence == pytest.approx(label.label_reliability)
    assert label.label_reliability != pytest.approx(label.vlm_self_confidence)
    assert label.reliability_factors["diagnostic_object_hits"] >= 2


def test_evidence_based_unknown_is_not_backend_fallback():
    labeler = VLMRoomLabeler(client=FakeRoomLLM({"room_id": "room_0001", "category": "unknown", "confidence": 0.3}))

    label = labeler.label_room(_room(), [], None)

    assert label.category == "unknown"
    assert label.backend == "vlm"
    assert label.unknown_reason == "insufficient_or_ambiguous_evidence"


def test_strict_missing_room_vlm_backend_fails_clearly():
    labeler = VLMRoomLabeler(client=None, require_backend=True)

    with pytest.raises(RuntimeError, match="strict benchmark requires an enabled room VLM backend"):
        labeler.label_room(_room(), [], None)
