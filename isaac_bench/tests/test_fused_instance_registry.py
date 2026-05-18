import numpy as np

from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.fused_instance_registry import FusedInstanceRegistry, bbox_iou_3d
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.sensors.camera_geometry import CameraIntrinsics
from isaac_bench.sensors.depth_backproject import detection_to_world_points, detections_to_3d


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(width=6, height=6, fx=4.0, fy=4.0, cx=3.0, cy=3.0)


def _depth() -> np.ndarray:
    return np.full((6, 6), 2.0, dtype=np.float32)


def _mask() -> np.ndarray:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:5, 2:5] = True
    return mask


def _intr20() -> CameraIntrinsics:
    return CameraIntrinsics(width=20, height=20, fx=14.0, fy=14.0, cx=10.0, cy=10.0)


def _depth20() -> np.ndarray:
    return np.full((20, 20), 2.0, dtype=np.float32)


def _mask20(r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    mask = np.zeros((20, 20), dtype=bool)
    mask[int(r0) : int(r1), int(c0) : int(c1)] = True
    return mask


def test_mask_projection_returns_point_cloud_and_detection3d_payload():
    det = Detection2D("chair", "chair", 0.9, (2.0, 2.0, 4.0, 4.0), mask=_mask())

    points = detection_to_world_points(det, _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), min_points=1, stride=1)
    dets3d = detections_to_3d([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), min_points=1)

    assert points is not None
    assert points.shape[1] == 3
    assert len(dets3d) == 1
    assert dets3d[0].point_cloud_world is not None
    assert dets3d[0].bbox_world is not None
    assert dets3d[0].mask is not None


def test_fused_instance_registry_registers_and_merges_close_instances():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det1 = Detection2D("chair", "chair", 0.7, (2.0, 2.0, 4.0, 4.0), mask=_mask())
    det2 = Detection2D("chair", "chair", 0.9, (2.0, 2.0, 4.0, 4.0), mask=_mask())

    registry.update([det1], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    instances = registry.update([det2], _depth(), _intr(), (0.1, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    assert len(instances) == 1
    assert instances[0].instance_id == "object_0"
    assert instances[0].observed_count == 2
    assert instances[0].confidence == 0.9
    assert instances[0].last_seen_step == 2
    assert instances[0].last_mask is not None


def test_fused_instance_registry_ignores_detections_beyond_depth_max():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("chair", "chair", 0.9, (2.0, 2.0, 4.0, 4.0), mask=_mask())
    far_depth = np.full((6, 6), 4.0, dtype=np.float32)

    instances = registry.update([det], far_depth, _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, depth_max_m=3.0, min_points=1, stride=1)

    assert instances == []


def test_fused_instance_registry_keeps_far_instances_separate_and_exports_objects():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("chair", "chair", 0.8, (2.0, 2.0, 4.0, 4.0), mask=_mask())

    registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    instances = registry.update([det], _depth(), _intr(), (3.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)
    dets3d = registry.to_detections_3d()

    assert len(instances) == 2
    assert [item.instance_id for item in instances] == ["object_0", "object_1"]
    assert len(dets3d) == 2
    assert all(det3d.point_cloud_world is not None for det3d in dets3d)


def test_room_prompt_registers_room_instance():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("living room", "living room", 0.7, (2.0, 2.0, 4.0, 4.0), mask=_mask())

    instances = registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    assert len(instances) == 1
    assert instances[0].node_type == "room"


def test_fused_instance_registry_rejects_low_confidence_detections():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    low = Detection2D("chair", "chair", 0.45, (2.0, 2.0, 4.0, 4.0), mask=_mask())
    high = Detection2D("chair", "chair", 0.46, (2.0, 2.0, 4.0, 4.0), mask=_mask())

    assert registry.update([low], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1) == []
    instances = registry.update([high], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    assert len(instances) == 1
    assert instances[0].confidence == 0.46


def test_object_memory_can_register_fused_instances_with_geometry():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("chair", "chair", 0.8, (2.0, 2.0, 4.0, 4.0), mask=_mask())
    instances = registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    memory = ObjectMemory(merge_radius_m=0.75)

    changed = memory.update_fused_instances(instances, step_id=1)

    assert len(changed) == 1
    assert len(memory.nodes) == 1
    assert memory.nodes[0].source_instance_id == "object_0"
    assert memory.nodes[0].point_cloud_world is not None
    assert memory.nodes[0].bbox_world is not None


def test_bbox_iou_3d_handles_flat_projected_surfaces():
    a = np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    b = np.asarray([[0.5, 0.5, 1.0], [1.5, 1.5, 1.0]], dtype=np.float32)

    assert bbox_iou_3d(a, b) > 0.0


def test_class_confidence_accumulation_pillow_case():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05, min_valid_confidence=0.55)
    detections = [
        Detection2D("pillow", "pillow", 0.60, (6, 6, 14, 14), mask=_mask20(6, 6, 14, 14)),
        Detection2D("pillow", "pillow", 0.65, (6, 6, 14, 14), mask=_mask20(6, 6, 14, 14)),
        Detection2D("pillow", "pillow", 0.63, (6, 6, 14, 14), mask=_mask20(6, 6, 14, 14)),
        Detection2D("other", "other", 0.64, (6, 6, 14, 14), mask=_mask20(6, 6, 14, 14)),
    ]

    for step, det in enumerate(detections):
        registry.update([det], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=step, min_points=1, stride=1)

    track = registry.instances[0]
    assert track.stable_category == "pillow"
    assert abs(track.mean_confidence - ((0.60 + 0.65 + 0.63) / 3.0)) < 1e-6
    assert track.winner_detection_count == 3
    assert track.valid_detection_count == 4


def test_edge_partial_overlaps_existing_track_inherits_full_geometry():
    registry = FusedInstanceRegistry(
        merge_distance_m=0.75,
        merge_iou_3d=0.05,
        mask_iou_association_threshold=0.25,
        mask_containment_track_match_threshold=0.50,
    )
    full = Detection2D("cabinet", "cabinet", 0.90, (5, 5, 15, 15), mask=_mask20(5, 5, 15, 15))
    registry.update([full], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    stable_center = registry.instances[0].center_world.copy()
    partial = Detection2D("wardrobe", "wardrobe", 0.80, (0, 5, 10, 15), mask=_mask20(5, 0, 15, 10))

    registry.update([partial], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)
    track = registry.instances[0]

    assert len(registry.instances) == 1
    assert track.stable_category == "cabinet"
    assert np.allclose(track.center_world, stable_center)
    assert track.center_estimation_mode == "inherited_full_mask"
    assert track.used_for_policy_graph is True
    assert registry.raw_detection_log[-1]["visibility_status"] == "partial_edge"


def test_edge_partial_without_overlap_is_tentative_not_policy():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    partial = Detection2D("curtain", "curtain", 0.72, (0, 2, 5, 18), mask=_mask20(2, 0, 18, 5))

    registry.update([partial], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    track = registry.instances[0]
    snapshot = registry.gnn_snapshot()

    assert track.is_stable is False
    assert track.used_for_policy_graph is False
    assert snapshot["raw_detection_count"] == 1
    assert snapshot["tentative_track_count"] == 1
    assert snapshot["raw_detections"][0]["raw_category"] == "curtain"
    assert snapshot["raw_detections"][0]["visibility_status"] == "partial_edge"


def test_large_curtain_edge_detection_not_discarded():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05, partial_stability_min_observations=2)
    det = Detection2D("curtain", "curtain", 0.78, (0, 1, 8, 19), mask=_mask20(1, 0, 19, 8))

    registry.update([det], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    registry.update([det], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)
    snapshot = registry.gnn_snapshot()

    assert len(registry.instances) == 1
    assert registry.instances[0].category == "curtain"
    assert snapshot["raw_detection_count"] == 2
    assert snapshot["partial_edge_count"] == 2
    assert snapshot["object_tracks"][0]["used_for_policy_graph"] is False
    assert snapshot["object_tracks"][0]["used_for_stop"] is False


def test_edge_touch_large_object_not_discarded():
    test_large_curtain_edge_detection_not_discarded()


def test_partial_mask_overlap_inherits_existing_track_center_and_category():
    test_edge_partial_overlaps_existing_track_inherits_full_geometry()


def test_partial_no_overlap_tentative_not_policy_graph():
    test_edge_partial_without_overlap_is_tentative_not_policy()


def test_class_confidence_accumulation_keeps_pillow_winner():
    test_class_confidence_accumulation_pillow_case()


def test_occluded_center_patch_not_stable_from_single_view():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    occluded = Detection2D("plant", "plant", 0.82, (9, 9, 11, 11), mask=_mask20(9, 9, 11, 11))

    registry.update([occluded], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    track = registry.instances[0]

    assert track.visibility_status_counts["partial_occluded"] == 1
    assert track.used_for_policy_graph is False
    assert track.center_estimation_mode == "visible_extent_low_conf"


def test_sofa_pillow_containment_creates_child_relation():
    registry = FusedInstanceRegistry(
        merge_distance_m=0.75,
        merge_iou_3d=0.05,
        child_containment_threshold=0.70,
        child_area_ratio_threshold=0.35,
    )
    sofa = Detection2D("sofa", "sofa", 0.91, (3, 3, 17, 17), mask=_mask20(3, 3, 17, 17))
    pillow = Detection2D("pillow", "pillow", 0.88, (7, 7, 10, 10), mask=_mask20(7, 7, 10, 10))

    registry.update([sofa], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    registry.update([pillow], _depth20(), _intr20(), (0.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    sofa_track, pillow_track = registry.instances
    assert sofa_track.stable_category == "sofa"
    assert pillow_track.stable_category == "pillow"
    assert pillow_track.parent_track_id == sofa_track.instance_id
    assert pillow_track.instance_id in sofa_track.child_track_ids


def test_sofa_pillow_containment_creates_child_relation_alias():
    test_sofa_pillow_containment_creates_child_relation()
