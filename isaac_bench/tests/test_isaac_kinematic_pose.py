from __future__ import annotations

import math

import pytest

from isaac_bench.env.isaac_process import IsaacSimServer


class _DummyApp:
    def __init__(self) -> None:
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


class _DummyPosePrim:
    def __init__(self) -> None:
        self.calls = []

    def set_world_pose(self, *, position, orientation) -> None:
        self.calls.append((tuple(float(v) for v in position), tuple(float(v) for v in orientation)))


def _server_for_kinematic_step() -> IsaacSimServer:
    server = IsaacSimServer(verbose=False)
    server.app = _DummyApp()
    server.robot = _DummyPosePrim()
    server.camera = _DummyPosePrim()
    server.nearfield_camera = _DummyPosePrim()
    server.camera_orientation = lambda yaw: (1.0, 0.0, 0.0, 0.0)
    server.nearfield_camera_orientation = lambda pose: (1.0, 0.0, 0.0, 0.0)
    server._camera_frame_token = lambda: 0
    server._wait_for_fresh_camera_frame = lambda *args, **kwargs: None
    server.get_observation = lambda **kwargs: {"pose_world": server.get_pose_world()}
    return server


def test_kinematic_step_moves_robot_and_cameras_together():
    server = _server_for_kinematic_step()
    server.set_pose_world((1.0, 2.0, 0.05, 0.0), sync_robot=True)

    assert len(server.robot.calls) == 1
    assert len(server.camera.calls) == 1

    obs = server.step_kinematic_velocity(0.1, 0.0, 0.2, dt=1.0, render_updates=3)

    assert obs["pose_world"][0] == pytest.approx(1.1)
    assert obs["pose_world"][3] == pytest.approx(0.2)
    assert server.app.updates == 3
    assert len(server.robot.calls) == 2
    assert len(server.camera.calls) == 2
    assert len(server.nearfield_camera.calls) == 2


def test_kinematic_step_rejects_nonfinite_command_before_isaac_update():
    server = _server_for_kinematic_step()
    server.set_pose_world((1.0, 2.0, 0.05, 0.0), sync_robot=False)

    with pytest.raises(ValueError, match="non-finite"):
        server.step_kinematic_velocity(math.nan, 0.0, 0.0)

    assert server.app.updates == 0
