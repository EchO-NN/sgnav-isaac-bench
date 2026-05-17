import numpy as np

from isaac_bench.graph.room_context import RoomContextCache, prepare_room_context_for_frontier_scoring
from isaac_bench.graph.room_semantics import RoomSemanticLabel
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import (
    RoomMask,
    RoomProposalState,
    RoomSegmentationConfig,
    merge_open_plan_proposals,
)
from isaac_bench.perception.object_memory import ObjectMemory


def _cfg() -> RoomSegmentationConfig:
    return RoomSegmentationConfig(
        resolution_m=0.1,
        min_observed_free_cells=4,
        min_room_area_m2=0.1,
        morphology_close_radius_m=0.0,
        morphology_open_radius_m=0.0,
        doorway_width_min_m=0.4,
        doorway_width_max_m=1.4,
        doorway_clearance_max_m=2.5,
        doorway_endpoint_wall_distance_m=0.35,
        small_segment_merge_area_m2=0.2,
        min_label_reliability_for_functional_split=0.65,
    )


def _open_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros((30, 50), dtype=np.int32)
    labels[5:25, 5:25] = 1
    labels[5:25, 25:45] = 2
    free = labels > 0
    obstacle = np.zeros_like(free)
    unknown = ~free
    distance = np.ones_like(labels, dtype=np.float32) * 0.5
    return labels, free, obstacle, unknown, distance


def _label(category: str, reliability: float) -> dict:
    return {"category": category, "label_reliability": reliability, "is_reliable": reliability >= 0.65}


def _final_count(labels: np.ndarray) -> int:
    return len([v for v in np.unique(labels) if int(v) > 0])


def test_same_reliable_room_type_merges_under_open_boundary():
    labels, free, obstacle, unknown, distance = _open_labels()

    out, debug, edges = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance,
        _cfg(),
        proposal_semantic_labels={1: _label("living_room", 0.9), 2: _label("living_room", 0.8)},
    )

    assert _final_count(out) == 1
    assert edges == []
    assert debug["adjacency_decisions"][0]["decision"] == "merge"
    assert "same_semantic_living_room" in debug["merge_operations"][0]["reason"]


def test_reliable_different_room_types_keep_open_plan_functional_split():
    labels, free, obstacle, unknown, distance = _open_labels()

    out, debug, edges = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance,
        _cfg(),
        proposal_semantic_labels={1: _label("kitchen", 0.9), 2: _label("living_room", 0.8)},
    )

    assert _final_count(out) == 2
    assert edges == []
    assert debug["merge_operations"] == []
    assert debug["functional_split_edges"][0]["reason"] == "reliable_different_room_types"
    assert debug["adjacency_decisions"][0]["decision"] == "keep_split"


def test_unknown_or_unreliable_labels_merge_under_open_boundary():
    labels, free, obstacle, unknown, distance = _open_labels()

    unknown_out, unknown_debug, _ = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance,
        _cfg(),
        proposal_semantic_labels={1: _label("kitchen", 0.9), 2: _label("unknown", 0.0)},
    )
    unreliable_out, unreliable_debug, _ = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance,
        _cfg(),
        proposal_semantic_labels={1: _label("kitchen", 0.4), 2: _label("living_room", 0.9)},
    )

    assert _final_count(unknown_out) == 1
    assert unknown_debug["adjacency_decisions"][0]["right_premerge_category"] == "unknown"
    assert _final_count(unreliable_out) == 1
    assert unreliable_debug["adjacency_decisions"][0]["left_reliable"] is False


def test_structural_boundary_preserves_split_even_when_labels_unknown():
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[10:70, 8:39] = 1
    labels[10:70, 41:72] = 2
    labels[34:45, 39] = 1
    labels[34:45, 40] = 2
    free = labels > 0
    obstacle = np.zeros_like(free)
    obstacle[:, 39:41] = True
    obstacle[34:45, 39:41] = False
    unknown = ~free & ~obstacle
    distance = np.ones_like(labels, dtype=np.float32) * 0.45

    out, debug, edges = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance,
        _cfg(),
        proposal_semantic_labels={1: _label("unknown", 0.0), 2: _label("unknown", 0.0)},
    )

    assert _final_count(out) == 2
    assert edges
    assert debug["adjacency_decisions"][0]["verified_structural_boundary"] is True
    assert debug["adjacency_decisions"][0]["decision"] == "keep_split"


def test_object_evidence_alone_does_not_create_functional_split():
    labels, free, obstacle, unknown, distance = _open_labels()

    out, debug, _ = merge_open_plan_proposals(labels, free, obstacle, unknown, distance, _cfg())

    assert _final_count(out) == 1
    assert debug["functional_split_edges"] == []
    assert debug["adjacency_decisions"][0]["decision"] == "merge"


class TwoStageSegmenter:
    def __init__(self):
        self.last_debug = {}
        self.finalize_labels = None

    def build_proposals(self, occupancy_map, observed_free_mask, obstacle_mask, unknown_mask, step, object_memory=None):
        _ = occupancy_map, obstacle_mask, object_memory
        left = np.zeros_like(observed_free_mask, dtype=bool)
        right = np.zeros_like(observed_free_mask, dtype=bool)
        left[1:4, 1:3] = True
        right[1:4, 3:5] = True
        labels = np.zeros_like(observed_free_mask, dtype=np.int32)
        labels[left] = 1
        labels[right] = 2
        rooms = [
            _room("proposal_0001", left, 1),
            _room("proposal_0002", right, 2),
        ]
        state = RoomProposalState(labels, observed_free_mask, np.zeros_like(observed_free_mask), unknown_mask, np.ones_like(labels), step=step)
        return rooms, state

    def finalize_proposals(self, proposal_state, proposal_semantic_labels=None):
        self.finalize_labels = dict(proposal_semantic_labels or {})
        mask = proposal_state.proposal_labels > 0
        room = _room("room_0001", mask, 1)
        self.last_debug = {
            "proposal_room_count": 2,
            "final_room_count": 1,
            "adjacency_decisions": [{"decision": "merge", "reason": "same semantic"}],
        }
        return [room]


class OrderLabeler:
    backend = "vlm"

    def __init__(self):
        self.request_count = 0
        self.failure_count = 0
        self.calls = []

    def label_room(self, room_mask, object_evidence, visual_evidence):
        _ = object_evidence, visual_evidence
        self.request_count += 1
        self.calls.append(room_mask.room_id)
        category = "kitchen" if room_mask.room_id.startswith("proposal") else "living_room"
        return RoomSemanticLabel(
            room_id=room_mask.room_id,
            category=category,
            confidence=0.9,
            supporting_objects=[],
            conflicting_evidence=[],
            rationale="test",
            backend="vlm",
            vlm_self_confidence=0.9,
            label_reliability=0.9,
            reliability_factors={},
        )


def _room(room_id: str, mask: np.ndarray, label_id: int) -> RoomMask:
    return RoomMask(
        room_id=room_id,
        mask=mask,
        centroid_xy=(2.0, 2.0),
        area_m2=float(np.count_nonzero(mask)),
        boundary_unknown_fraction=0.0,
        doorway_edges=[],
        confidence=0.9,
        observed_free_cells=int(np.count_nonzero(mask)),
        mask_confidence=0.9,
        metadata={"label_id": int(label_id)},
    )


def test_premerge_labels_used_only_for_merge_and_final_label_is_recomputed():
    free = np.zeros((6, 6), dtype=bool)
    free[1:4, 1:5] = True
    unknown = ~free
    labeler = OrderLabeler()
    segmenter = TwoStageSegmenter()

    result = prepare_room_context_for_frontier_scoring(
        step_idx=0,
        object_memory=ObjectMemory(),
        room_segmenter=segmenter,
        room_labeler=labeler,
        map_info=MapInfo(resolution_m=1.0, min_x=0.0, max_x=6.0, min_y=0.0, max_y=6.0, width=6, height=6),
        previous_room_context=RoomContextCache(),
        strict_benchmark=True,
        occupancy=~free,
        observed_free_mask=free,
        obstacle_mask=~free,
        unknown_mask=unknown,
    )

    assert labeler.calls[:2] == ["proposal_0001", "proposal_0002"]
    assert labeler.calls[-1] == "room_0001"
    assert set(segmenter.finalize_labels) == {1, 2}
    assert list(result.room_semantic_labels.values())[0].category == "living_room"
    assert result.room_segmentation_debug["premerge_room_semantics"]["final_labels_recomputed_after_merge"] is True
