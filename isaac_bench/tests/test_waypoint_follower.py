import math

from isaac_bench.navigation.waypoint_follower import HolonomicWaypointFollower
from isaac_bench.scripts.run_one_episode import predict_kinematic_pose


def test_follower_preserves_world_direction_when_clamped():
    follower = HolonomicWaypointFollower(max_vx=0.15, max_vy=0.15, max_wz=0.35, lookahead_m=0.50)
    pose = (-6.5736, 0.6383, 0.05, -0.2707)
    path_world = [(-6.525, 0.625), (-6.475, 0.625), (-6.425, 0.625), (-6.375, 0.625), (-6.325, 0.625), (-6.275, 0.625), (-6.225, 0.625), (-6.175, 0.625), (-6.125, 0.625), (-6.075, 0.625), (-6.025, 0.625)]

    cmd = follower.compute_cmd(pose, path_world)
    predicted = predict_kinematic_pose(pose, cmd, dt=0.2)

    assert predicted[0] > pose[0]
    assert predicted[1] < pose[1]
    assert math.hypot(predicted[0] - pose[0], predicted[1] - pose[1]) <= 0.031
