import numpy as np

from isaac_bench.graph.room_context import RoomContextCache, prepare_room_context_for_frontier_scoring
from isaac_bench.graph.room_semantics import RoomSemanticLabel
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import RoomMask
from isaac_bench.perception.object_memory import ObjectMemory, ObjectNode


class StableSegmenter:
    def __init__(self):
        self.update_count = 0
        self.last_debug = {}

    def update(self, occupancy_map, observed_free_mask, obstacle_mask, unknown_mask, step):
        self.update_count += 1
        mask = np.zeros_like(observed_free_mask, dtype=bool)
        mask[1:4, 1:4] = True
        if self.update_count > 1:
            mask[0, 0] = True
        room = RoomMask(
            room_id="room_0001",
            mask=mask,
            centroid_xy=(2.5, 2.5),
            area_m2=float(np.count_nonzero(mask)),
            boundary_unknown_fraction=0.0,
            doorway_edges=[],
            confidence=0.9,
            observed_free_cells=int(np.count_nonzero(mask)),
            mask_confidence=0.9,
        )
        self.last_debug = {"source": "online_geometry_watershed", "room_count": 1}
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
            confidence=0.5,
            supporting_objects=[],
            conflicting_evidence=[],
            rationale="ambiguous",
            backend="vlm",
            unknown_reason="insufficient_or_ambiguous_evidence",
            vlm_self_confidence=0.5,
            label_reliability=0.5,
            reliability_factors={},
        )


def _memory():
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


def _map_info():
    return MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)


def _arrays(extra_free=False):
    free = np.zeros((5, 5), dtype=bool)
    free[1:4, 1:4] = True
    if extra_free:
        free[0, 0] = True
    occ = ~free
    observed = np.ones((5, 5), dtype=bool)
    return free, occ, ~observed


def _prepare(cache, segmenter, labeler, extra_free=False):
    free, occ, unknown = _arrays(extra_free=extra_free)
    return prepare_room_context_for_frontier_scoring(
        step_idx=0,
        object_memory=_memory(),
        room_segmenter=segmenter,
        room_labeler=labeler,
        map_info=_map_info(),
        previous_room_context=cache,
        strict_benchmark=True,
        occupancy=occ,
        observed_free_mask=free,
        obstacle_mask=occ,
        unknown_mask=unknown,
    )


def test_room_context_cache_reuses_full_context_when_map_and_objects_unchanged():
    cache = RoomContextCache()
    segmenter = StableSegmenter()
    labeler = CountingLabeler()

    first = _prepare(cache, segmenter, labeler)
    second = _prepare(cache, segmenter, labeler)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert segmenter.update_count == 1
    assert labeler.request_count == 1
    assert second.label_cache_hits == 1


def test_room_context_reruns_segmentation_but_reuses_label_for_unchanged_object_evidence():
    cache = RoomContextCache()
    segmenter = StableSegmenter()
    labeler = CountingLabeler()

    first = _prepare(cache, segmenter, labeler)
    second = _prepare(cache, segmenter, labeler, extra_free=True)

    assert first.segmentation_ran is True
    assert second.segmentation_ran is True
    assert second.cache_hit is False
    assert segmenter.update_count == 2
    assert labeler.request_count == 1
    assert second.label_cache_hits == 1
