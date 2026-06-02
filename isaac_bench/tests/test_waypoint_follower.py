import math

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.navigation.waypoint_follower import HolonomicWaypointFollower
from isaac_bench.scripts.run_one_episode import guard_kinematic_cmd, pose_is_grid_safe, pose_swept_is_grid_safe, predict_kinematic_pose


def test_follower_preserves_world_direction_when_clamped():
    follower = HolonomicWaypointFollower(max_vx=0.15, max_vy=0.15, max_wz=0.35, lookahead_m=0.50)
    pose = (-6.5736, 0.6383, 0.05, -0.2707)
    path_world = [(-6.525, 0.625), (-6.475, 0.625), (-6.425, 0.625), (-6.375, 0.625), (-6.325, 0.625), (-6.275, 0.625), (-6.225, 0.625), (-6.175, 0.625), (-6.125, 0.625), (-6.075, 0.625), (-6.025, 0.625)]

    cmd = follower.compute_cmd(pose, path_world)
    predicted = predict_kinematic_pose(pose, cmd, dt=0.2)

    assert predicted[0] > pose[0]
    assert predicted[1] < pose[1]
    assert math.hypot(predicted[0] - pose[0], predicted[1] - pose[1]) <= 0.031


def test_kinematic_guard_checks_swept_path_not_only_endpoint():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=3.0, width=5, height=3)
    navigable = np.ones((3, 5), dtype=bool)
    navigable[1, 2] = False
    pose = (0.5, 1.5, 0.05, 0.0)
    cmd = (4.0, 0.0, 0.0)
    end = predict_kinematic_pose(pose, cmd, dt=1.0)

    assert pose_is_grid_safe(end, navigable, info)
    assert not pose_swept_is_grid_safe(pose, cmd, 1.0, navigable, info)

    guarded, blocked = guard_kinematic_cmd(pose, cmd, 1.0, navigable, info, camera_forward_offset_m=0.0)
    assert blocked is False
    assert guarded[0] == pytest.approx(1.0)
    assert pose_swept_is_grid_safe(pose, guarded, 1.0, navigable, info)


def test_kinematic_guard_reports_blocked_when_translation_is_suppressed_to_rotation_only():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=3.0, min_y=0.0, max_y=3.0, width=3, height=3)
    navigable = np.zeros((3, 3), dtype=bool)
    navigable[1, 1] = True
    pose = (1.99, 1.5, 0.05, 0.0)
    cmd = (1.0, 0.0, 0.25)

    guarded, blocked = guard_kinematic_cmd(pose, cmd, 1.0, navigable, info, camera_forward_offset_m=0.0)

    assert blocked is True
    assert guarded == pytest.approx((0.0, 0.0, 0.25))
