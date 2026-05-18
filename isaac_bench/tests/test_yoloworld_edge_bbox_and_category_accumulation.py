import pytest
import numpy as np

from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.graph.paper_scene_graph import PaperSceneGraph
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.detection_types import Detection2D, Detection3D, bbox_touches_image_edge
from isaac_bench.perception.fused_instance_registry import FusedInstanceRegistry
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.scripts.run_one_episode import filter_edge_touching_detections
from isaac_bench.sensors.camera_geometry import CameraIntrinsics


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(width=8, height=8, fx=4.0, fy=4.0, cx=4.0, cy=4.0)


def _depth() -> np.ndarray:
    return np.full((8, 8), 2.0, dtype=np.float32)


def _mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[3:6, 3:6] = True
    return mask


def test_bbox_touches_image_edge_helper():
    assert bbox_touches_image_edge((0.0, 3.0, 5.0, 6.0), 8, 8, margin_px=2)
    assert not bbox_touches_image_edge((3.0, 3.0, 4.0, 4.0), 8, 8, margin_px=2)


def test_edge_bbox_kept_as_partial_tentative_track_and_logged_raw():
    registry = FusedInstanceRegistry(
        merge_distance_m=0.75,
        merge_iou_3d=0.0,
        min_valid_confidence=0.0,
        reject_edge_touching_bboxes=True,
    )
    edge = Detection2D("chair", "chair", 0.9, (0.0, 3.0, 5.0, 6.0), mask=_mask())

    instances = registry.update([edge], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    assert len(instances) == 1
    assert registry.raw_detection_log[0]["bbox_touches_edge"] is True
    assert registry.raw_detection_log[0]["used_for_object_track"] is True
    assert registry.raw_detection_log[0]["visibility_status"] == "partial_edge"
    assert registry.raw_detection_log[0]["reject_reason"] is None
    assert instances[0].used_for_policy_graph is False


def test_low_confidence_detection_is_logged_but_not_forwarded_to_sam_or_memory():
    raw_log = []
    low = Detection2D("pillow", "pillow", 0.45, (3.0, 3.0, 4.0, 4.0), mask=_mask())

    kept = filter_edge_touching_detections(
        [low],
        image_width=8,
        image_height=8,
        step_idx=3,
        reject_edge_touching_bboxes=True,
        margin_px=2,
        margin_ratio=0.0,
        min_confidence=0.45,
        raw_log=raw_log,
    )

    assert kept == []
    assert raw_log[0]["confidence"] == pytest.approx(0.45)
    assert raw_log[0]["used_for_object_track"] is False
    assert raw_log[0]["reject_reason"] == "low_confidence"


def test_non_edge_bbox_updates_category_sums_hits_and_mean_confidence():
    memory = ObjectMemory(merge_radius_m=0.5, min_valid_confidence=0.0)

    memory.update([Detection3D("pillow", "pillow", 0.6, (1.0, 1.0, 0.5), (3, 3, 5, 5))], step_id=1)

    node = memory.nodes[0]
    assert node.class_conf_sums["pillow"] == pytest.approx(0.6)
    assert node.class_hits["pillow"] == 1
    assert node.valid_detection_count == 1
    assert node.mean_confidence == pytest.approx(0.6)
    assert node.to_dict()["detection_count"] == 1


def test_confidence_accumulation_beats_one_frame_class_switch():
    memory = ObjectMemory(merge_radius_m=0.5, min_valid_confidence=0.0)
    detections = [
        Detection3D("pillow", "pillow", 0.60, (1.0, 1.0, 0.5), (3, 3, 5, 5)),
        Detection3D("pillow", "pillow", 0.65, (1.05, 1.0, 0.5), (3, 3, 5, 5)),
        Detection3D("pillow", "pillow", 0.63, (1.02, 1.0, 0.5), (3, 3, 5, 5)),
        Detection3D("other", "other", 0.64, (1.03, 1.0, 0.5), (3, 3, 5, 5)),
    ]

    for idx, det in enumerate(detections, start=1):
        memory.update([det], step_id=idx)

    node = memory.nodes[0]
    assert node.category == "pillow"
    assert node.class_conf_sums["pillow"] == pytest.approx(1.88)
    assert node.class_conf_sums["other"] == pytest.approx(0.64)
    assert node.mean_confidence == pytest.approx(1.88 / 3.0)
    assert node.valid_detection_count == 4
    assert node.winner_detection_count == 3


def test_edge_touching_high_confidence_wrong_label_does_not_overwrite_track():
    registry = FusedInstanceRegistry(
        merge_distance_m=0.75,
        merge_iou_3d=0.0,
        min_valid_confidence=0.0,
        reject_edge_touching_bboxes=True,
    )
    good = Detection2D("pillow", "pillow", 0.8, (3.0, 3.0, 4.0, 4.0), mask=_mask())
    edge_wrong = Detection2D("chair", "chair", 0.99, (0.0, 3.0, 5.0, 6.0), mask=_mask())

    registry.update([good], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    instances = registry.update([edge_wrong], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    assert len(instances) == 1
    assert instances[0].category == "pillow"
    assert instances[0].class_conf_sums["pillow"] == pytest.approx(0.8)
    assert instances[0].class_conf_sums["chair"] == pytest.approx(0.99 * registry.partial_class_weight)
    assert instances[0].visibility_status_counts["partial_edge"] == 1


def test_edge_bbox_does_not_create_goal_candidate_or_stop():
    raw_log = []
    edge = Detection2D("mirror", "mirror", 0.99, (0.0, 3.0, 5.0, 6.0), mask=_mask())
    kept = filter_edge_touching_detections(
        [edge],
        image_width=8,
        image_height=8,
        step_idx=1,
        reject_edge_touching_bboxes=True,
        margin_px=2,
        margin_ratio=0.0,
        raw_log=raw_log,
    )
    memory = ObjectMemory(merge_radius_m=0.5, min_valid_confidence=0.0)
    memory.update([], step_id=1)
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

    assert kept == [edge]
    assert raw_log[0]["reject_reason"] is None
    assert raw_log[0]["visibility_status"] == "partial_edge"
    assert memory.nodes == []
    assert nav.stop is False
    assert nav.selected_candidate is None


def test_object_memory_and_graph_dump_simple_gnn_fields():
    memory = ObjectMemory(merge_radius_m=0.5, min_valid_confidence=0.0)
    memory.update([Detection3D("pillow", "pillow", 0.7, (1.0, 1.0, 0.5), (3, 3, 5, 5))], step_id=1)
    memory.update([Detection3D("pillow", "pillow", 0.9, (1.1, 1.0, 0.5), (3, 3, 5, 5))], step_id=2)
    graph = PaperSceneGraph()
    graph.update_from_object_memory(memory)
    node = next(iter(graph.object_nodes.values()))
    payload = memory.nodes[0].to_dict()

    assert payload["category"] == "pillow"
    assert payload["mean_confidence"] == pytest.approx(0.8)
    assert payload["detection_count"] == 2
    assert payload["winner_detection_count"] == 2
    assert node.mean_confidence == pytest.approx(0.8)
    assert node.detection_count == 2


def test_fused_track_accumulators_are_preserved_in_object_memory():
    registry = FusedInstanceRegistry(
        merge_distance_m=0.75,
        merge_iou_3d=0.0,
        min_valid_confidence=0.0,
        reject_edge_touching_bboxes=True,
    )
    observations = [
        Detection2D("pillow", "pillow", 0.60, (3.0, 3.0, 4.0, 4.0), mask=_mask()),
        Detection2D("pillow", "pillow", 0.65, (3.0, 3.0, 4.0, 4.0), mask=_mask()),
        Detection2D("pillow", "pillow", 0.63, (3.0, 3.0, 4.0, 4.0), mask=_mask()),
        Detection2D("other", "other", 0.64, (3.0, 3.0, 4.0, 4.0), mask=_mask()),
    ]
    for step, det in enumerate(observations, start=1):
        registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=step, min_points=1, stride=1)
    memory = ObjectMemory(merge_radius_m=0.75, min_valid_confidence=0.0)

    changed = memory.update_fused_instances(registry.instances, step_id=4)

    assert len(changed) == 1
    node = memory.nodes[0]
    assert node.category == "pillow"
    assert node.class_conf_sums["pillow"] == pytest.approx(1.88)
    assert node.class_conf_sums["other"] == pytest.approx(0.64)
    assert node.valid_detection_count == 4
    assert node.winner_detection_count == 3
