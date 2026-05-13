import numpy as np

from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.sensors.camera_geometry import CameraIntrinsics


def test_online_mapper_marks_depth_obstacle_and_free_ray():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=4.0,
        depth_stride_px=1,
        obstacle_min_height_m=0.05,
        obstacle_max_height_m=1.5,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=10.0, fy=10.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), 2.0, dtype=np.float32)

    grid = mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    obstacle_cell = grid.world_to_grid(2.0, 0.0)
    free_cell = grid.world_to_grid(1.0, 0.0)

    assert grid.occupied[obstacle_cell] == 1
    assert grid.free[free_cell] == 1
    assert not mapper.traversible(unknown_is_obstacle=True)[obstacle_cell]
