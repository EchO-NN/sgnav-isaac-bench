import math

import numpy as np

from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.sensors.camera_geometry import CameraIntrinsics
from isaac_bench.sensors.depth_backproject import detection_to_world_points, detections_to_3d


def test_bbox_center_depth_projects_forward_and_yaw_to_map_axes():
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[2, 2] = 2.0
    det = Detection2D("chair", "chair", 0.9, (2.0, 2.0, 3.0, 3.0))

    yaw0 = detections_to_3d([det], depth, intr, (0.0, 0.0, 1.0, 0.0), depth_max_m=6.0, min_points=1)
    yaw90 = detections_to_3d([det], depth, intr, (0.0, 0.0, 1.0, math.pi / 2.0), depth_max_m=6.0, min_points=1)

    assert len(yaw0) == 1
    assert np.allclose(yaw0[0].center_world, (2.0, 0.0, 1.0), atol=1e-5)
    assert len(yaw90) == 1
    assert np.allclose(yaw90[0].center_world[:2], (0.0, 2.0), atol=1e-5)


def test_bbox_right_side_projects_to_robot_right():
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[2, 4] = 2.0
    det = Detection2D("chair", "chair", 0.9, (4.0, 2.0, 5.0, 3.0))

    out = detections_to_3d([det], depth, intr, (0.0, 0.0, 1.0, 0.0), depth_max_m=6.0, min_points=1)

    assert len(out) == 1
    assert np.allclose(out[0].center_world, (2.0, -2.0, 1.0), atol=1e-5)


def test_bbox_only_projection_prefers_foreground_depth_over_background():
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), 5.0, dtype=np.float32)
    depth[1:4, 1:3] = 2.0
    det = Detection2D("chair", "chair", 0.9, (1.0, 1.0, 5.0, 4.0))

    points = detection_to_world_points(
        det,
        depth,
        intr,
        (0.0, 0.0, 1.0, 0.0),
        depth_max_m=6.0,
        min_points=2,
        stride=1,
    )

    assert points is not None
    assert float(np.median(points[:, 0])) < 3.0
