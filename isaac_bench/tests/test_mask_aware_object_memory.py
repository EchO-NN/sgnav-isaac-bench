import numpy as np
import pytest

from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.fused_instance_registry import FusedInstanceRegistry
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.sensors.camera_geometry import CameraIntrinsics


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(width=12, height=12, fx=8.0, fy=8.0, cx=6.0, cy=6.0)


def _depth() -> np.ndarray:
    return np.full((12, 12), 2.0, dtype=np.float32)


def _mask(r0, r1, c0, c1):
    out = np.zeros((12, 12), dtype=bool)
    out[int(r0) : int(r1), int(c0) : int(c1)] = True
    return out


def _registry(**kwargs) -> FusedInstanceRegistry:
    return FusedInstanceRegistry(
        merge_distance_m=0.75,
        merge_iou_3d=0.0,
        min_valid_confidence=0.0,
        reject_edge_touching_bboxes=True,
        **kwargs,
    )


def test_category_confidence_accumulation_not_last_frame_override():
    registry = _registry()
    observations = [
        Detection2D("pillow", "pillow", 0.60, (4, 4, 8, 8), mask=_mask(4, 8, 4, 8)),
        Detection2D("pillow", "pillow", 0.65, (4, 4, 8, 8), mask=_mask(4, 8, 4, 8)),
        Detection2D("pillow", "pillow", 0.63, (4, 4, 8, 8), mask=_mask(4, 8, 4, 8)),
        Detection2D("other", "other", 0.64, (4, 4, 8, 8), mask=_mask(4, 8, 4, 8)),
    ]

    for step, det in enumerate(observations, start=1):
        registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=step, min_points=1, stride=1)

    track = registry.instances[0]
    assert track.category == "pillow"
    assert track.stable_category == "pillow"
    assert track.class_conf_sums["pillow"] == pytest.approx(1.88)
    assert track.class_conf_sums["other"] == pytest.approx(0.64)
    assert track.mean_confidence == pytest.approx(1.88 / 3.0)
    assert track.valid_detection_count == 4
    assert track.winner_detection_count == 3


def test_edge_partial_overlaps_existing_full_mask_inherits_track():
    registry = _registry()
    full = Detection2D("curtain", "curtain", 0.90, (4, 2, 9, 10), mask=_mask(2, 10, 4, 9))
    partial = Detection2D("curtain", "curtain", 0.88, (0, 2, 7, 10), mask=_mask(2, 10, 4, 7))

    registry.update([full], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    stable_center = registry.instances[0].center_world.copy()
    registry.update([partial], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    track = registry.instances[0]
    assert len(registry.instances) == 1
    assert track.visibility_status_counts["partial_edge"] == 1
    assert track.center_estimation_mode == "inherited_full_mask"
    assert np.allclose(track.center_world, stable_center)
    assert track.used_for_policy_graph is True


def test_edge_partial_without_overlap_creates_tentative_track():
    registry = _registry()
    edge = Detection2D("curtain", "curtain", 0.92, (0, 2, 4, 9), mask=_mask(2, 9, 0, 4))

    registry.update([edge], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    track = registry.instances[0]
    assert registry.raw_detection_log[0]["used_for_object_track"] is True
    assert registry.raw_detection_log[0]["visibility_status"] == "partial_edge"
    assert track.is_stable is False
    assert track.used_for_policy_graph is False


def test_large_curtain_edge_detection_not_discarded():
    registry = _registry()
    det = Detection2D("curtain", "curtain", 0.93, (0, 0, 11, 11), mask=_mask(0, 12, 0, 5))

    registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    assert len(registry.instances) == 1
    assert registry.raw_detection_log[0]["reject_reason"] is None
    assert registry.raw_detection_log[0]["visibility_status"] == "partial_edge"


def test_partial_does_not_update_stable_center():
    registry = _registry()
    full = Detection2D("cabinet", "cabinet", 0.91, (4, 4, 9, 9), mask=_mask(4, 9, 4, 9))
    partial = Detection2D("cabinet", "cabinet", 0.91, (0, 4, 8, 9), mask=_mask(4, 9, 4, 8))

    registry.update([full], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    stable_center = registry.instances[0].center_world.copy()
    registry.update([partial], _depth(), _intr(), (0.5, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    assert np.allclose(registry.instances[0].center_world, stable_center)


def test_door_occluded_detection_not_policy_object_without_history():
    registry = _registry()
    occluded = Detection2D("mirror", "mirror", 0.91, (5, 5, 7, 7), mask=_mask(5, 7, 5, 7))

    registry.update([occluded], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    track = registry.instances[0]
    assert track.visibility_status_counts["partial_occluded"] == 1
    assert track.used_for_policy_graph is False


def test_sofa_pillow_contained_masks_create_parent_child_tracks():
    registry = _registry()
    sofa = Detection2D("sofa", "sofa", 0.94, (2, 2, 10, 10), mask=_mask(2, 10, 2, 10))
    pillow = Detection2D("pillow", "pillow", 0.88, (5, 5, 7, 7), mask=_mask(5, 7, 5, 7))

    registry.update([sofa, pillow], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    assert len(registry.instances) == 2
    sofa_track = next(track for track in registry.instances if track.category == "sofa")
    pillow_track = next(track for track in registry.instances if track.category == "pillow")
    assert pillow_track.parent_track_id == sofa_track.instance_id
    assert pillow_track.instance_id in sofa_track.child_track_ids


def test_raw_detections_saved_for_gnn_snapshot():
    registry = _registry()
    registry.update(
        [Detection2D("chair", "chair", 0.9, (4, 4, 8, 8), mask=_mask(4, 8, 4, 8))],
        _depth(),
        _intr(),
        (0.0, 0.0, 1.0, 0.0),
        step_id=1,
        min_points=1,
        stride=1,
    )

    snapshot = registry.gnn_snapshot()
    assert snapshot["raw_detection_count"] == 1
    assert snapshot["stable_track_count"] == 1
    assert snapshot["object_tracks"][0]["class_conf_sums"]["chair"] == pytest.approx(0.9)
    assert "visibility_status_counts" in snapshot["object_tracks"][0]


def test_only_stable_tracks_enter_policy_graph():
    registry = _registry()
    full = Detection2D("chair", "chair", 0.90, (4, 4, 8, 8), mask=_mask(4, 8, 4, 8))
    edge = Detection2D("table", "table", 0.92, (0, 2, 4, 9), mask=_mask(2, 9, 0, 4))
    registry.update([full, edge], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    memory = ObjectMemory(merge_radius_m=0.75, min_valid_confidence=0.0)

    memory.update_fused_instances(registry.instances, step_id=1)

    assert [node.category for node in memory.nodes] == ["chair"]


def test_tentative_tracks_do_not_trigger_goal_candidate_or_stop():
    registry = _registry()
    edge_goal = Detection2D("mirror", "mirror", 0.92, (0, 2, 4, 9), mask=_mask(2, 9, 0, 4))
    registry.update([edge_goal], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    memory = ObjectMemory(merge_radius_m=0.75, min_valid_confidence=0.0)
    memory.update_fused_instances(registry.instances, step_id=1)
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("mirror")
    scenegraph.update(memory)
    planner = GridAStarPlanner(np.ones((5, 5), dtype=bool), resolution_m=1.0, allow_diagonal=True)
    decision = SGNavDecision(scenegraph, candidate_start_min_hits=1)

    nav = decision.choose_navigation_target(
        memory,
        "mirror",
        (2, 2),
        [],
        planner,
        MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5),
        (2.0, 2.0, 0.0, 0.0),
    )

    assert memory.nodes == []
    assert nav.stop is False
    assert nav.selected_candidate is None

