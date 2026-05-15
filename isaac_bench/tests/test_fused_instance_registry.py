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


def test_mask_projection_returns_point_cloud_and_detection3d_payload():
    det = Detection2D("chair", "chair", 0.9, (2.0, 2.0, 5.0, 5.0), mask=_mask())

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
    det1 = Detection2D("chair", "chair", 0.7, (2.0, 2.0, 5.0, 5.0), mask=_mask())
    det2 = Detection2D("chair", "chair", 0.9, (2.0, 2.0, 5.0, 5.0), mask=_mask())

    registry.update([det1], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    instances = registry.update([det2], _depth(), _intr(), (0.1, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)

    assert len(instances) == 1
    assert instances[0].instance_id == "object_0"
    assert instances[0].observed_count == 2
    assert instances[0].confidence == 0.9
    assert instances[0].last_seen_step == 2
    assert instances[0].last_mask is not None


def test_fused_instance_registry_keeps_far_instances_separate_and_exports_objects():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("chair", "chair", 0.8, (2.0, 2.0, 5.0, 5.0), mask=_mask())

    registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)
    instances = registry.update([det], _depth(), _intr(), (3.0, 0.0, 1.0, 0.0), step_id=2, min_points=1, stride=1)
    dets3d = registry.to_detections_3d()

    assert len(instances) == 2
    assert [item.instance_id for item in instances] == ["object_0", "object_1"]
    assert len(dets3d) == 2
    assert all(det3d.point_cloud_world is not None for det3d in dets3d)


def test_room_prompt_registers_room_instance():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("living room", "living room", 0.6, (2.0, 2.0, 5.0, 5.0), mask=_mask())

    instances = registry.update([det], _depth(), _intr(), (0.0, 0.0, 1.0, 0.0), step_id=1, min_points=1, stride=1)

    assert len(instances) == 1
    assert instances[0].node_type == "room"


def test_object_memory_can_register_fused_instances_with_geometry():
    registry = FusedInstanceRegistry(merge_distance_m=0.75, merge_iou_3d=0.05)
    det = Detection2D("chair", "chair", 0.8, (2.0, 2.0, 5.0, 5.0), mask=_mask())
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
