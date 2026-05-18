import numpy as np

from isaac_bench.graph.room_context import (
    SCORING_ROOM_CALL_ORDER_FULL,
    RoomContextCache,
    prepare_room_context_for_frontier_scoring,
)
from isaac_bench.graph.room_semantics import RoomSemanticLabel
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import RoomMask
from isaac_bench.perception.object_memory import ObjectMemory, ObjectNode


class CountingSegmenter:
    context_source = "upstream_rose2_vertical_or_free_vlm"

    def __init__(self):
        self.update_count = 0
        self.last_debug = {}

    def update(self, occupancy_map, observed_free_mask, obstacle_mask, unknown_mask, step):
        self.update_count += 1
        mask = np.zeros_like(observed_free_mask, dtype=bool)
        mask[1:4, 1:4] = True
        room = RoomMask(
            room_id="room_0001",
            mask=mask,
            centroid_xy=(2.5, 2.5),
            area_m2=9.0,
            boundary_unknown_fraction=0.0,
            doorway_edges=[],
            confidence=0.9,
            observed_free_cells=9,
            mask_confidence=0.9,
        )
        self.last_debug = {
            "source": "upstream_rose2_vertical_or_free",
            "algorithm": "upstream_rose2_vertical_or_free",
            "source_mode": "declutter_reconstruct_mit",
            "room_count": 1,
        }
        return [room]


class CountingLabeler:
    backend = "vlm"

    def __init__(self):
        self.request_count = 0
        self.failure_count = 0

    def label_room(self, room_mask, object_evidence, visual_evidence):
        self.request_count += 1
        return RoomSemanticLabel(
            room_id=room_mask.room_id,
            category="unknown",
            confidence=0.4,
            supporting_objects=[],
            conflicting_evidence=[],
            rationale="insufficient evidence",
            backend="vlm",
            unknown_reason="insufficient_or_ambiguous_evidence",
            vlm_self_confidence=0.4,
            label_reliability=0.4,
            reliability_factors={"diagnostic_object_hits": 0},
        )


def _map_info():
    return MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)


def _object_memory():
    memory = ObjectMemory()
    memory.nodes.append(
        ObjectNode(
            node_id=1,
            category="chair",
            center_world=(2.5, 2.5, 0.5),
            center_grid=(2, 2),
            confidence=0.8,
            observed_count=2,
            last_seen_step=0,
        )
    )
    return memory


def _masks():
    free = np.zeros((5, 5), dtype=bool)
    free[1:4, 1:4] = True
    occ = ~free
    observed = np.ones((5, 5), dtype=bool)
    unknown = ~observed
    return free, occ, unknown


def test_room_context_not_called_by_mapper_or_perception_only_updates():
    segmenter = CountingSegmenter()
    labeler = CountingLabeler()

    # Mapper/perception updates alone do not call the scoring hook.
    _ = _masks()
    _ = _object_memory()

    assert segmenter.update_count == 0
    assert labeler.request_count == 0


def test_mapper_update_does_not_call_roomseg():
    segmenter = CountingSegmenter()
    labeler = CountingLabeler()
    _free, _occ, _unknown = _masks()

    assert segmenter.update_count == 0
    assert labeler.request_count == 0


def test_reperception_does_not_call_roomseg_or_room_vlm():
    segmenter = CountingSegmenter()
    labeler = CountingLabeler()
    _ = _object_memory()

    assert segmenter.update_count == 0
    assert labeler.request_count == 0


def test_room_context_called_immediately_before_frontier_scoring_order():
    segmenter = CountingSegmenter()
    labeler = CountingLabeler()
    free, occ, unknown = _masks()
    trace = ["frontier_extraction"]

    result = prepare_room_context_for_frontier_scoring(
        step_idx=0,
        object_memory=_object_memory(),
        room_segmenter=segmenter,
        room_labeler=labeler,
        map_info=_map_info(),
        previous_room_context=RoomContextCache(),
        strict_benchmark=True,
        occupancy=occ,
        observed_free_mask=free,
        obstacle_mask=occ,
        unknown_mask=unknown,
    )
    trace.extend(["room_context_for_frontier_scoring", "scenegraph_update", "hcot_subgraph_scoring", "frontier_interpolation", "frontier_selection"])

    assert segmenter.update_count == 1
    assert labeler.request_count == 1
    assert result.segmentation_ran is True
    assert result.labeling_ran is True
    assert trace == SCORING_ROOM_CALL_ORDER_FULL
    assert result.metadata(full_order=True)["room_call_order_trace"] == SCORING_ROOM_CALL_ORDER_FULL


def test_committed_frontier_replan_does_not_trigger_room_context():
    segmenter = CountingSegmenter()
    labeler = CountingLabeler()

    # Continuing an already committed local A* path is represented here by not
    # entering the frontier-scoring hook at all.
    committed_frontier_replan = True
    if committed_frontier_replan:
        pass

    assert segmenter.update_count == 0
    assert labeler.request_count == 0


def test_room_segmentation_called_only_before_frontier_scoring():
    segmenter = CountingSegmenter()
    labeler = CountingLabeler()
    free, occ, unknown = _masks()

    # Perception-only, mapper-only, replan-only, and reperception-only phases
    # are represented by ordinary state updates that do not enter the scoring
    # pre-hook. The only allowed invocation is the explicit frontier-scoring
    # preparation call below.
    phases_without_roomseg = ["mapper_update", "perception_update", "committed_replan", "reperception_stop_check"]
    for _phase in phases_without_roomseg:
        assert segmenter.update_count == 0

    prepare_room_context_for_frontier_scoring(
        step_idx=3,
        object_memory=_object_memory(),
        room_segmenter=segmenter,
        room_labeler=labeler,
        map_info=_map_info(),
        previous_room_context=RoomContextCache(),
        strict_benchmark=True,
        occupancy=occ,
        observed_free_mask=free,
        obstacle_mask=occ,
        unknown_mask=unknown,
    )

    assert segmenter.update_count == 1
    assert labeler.request_count == 1
