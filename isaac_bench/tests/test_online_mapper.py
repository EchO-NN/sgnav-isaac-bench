import numpy as np
import math

from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.sensors.camera_geometry import CameraIntrinsics


def test_online_mapper_ray_casts_depth_obstacle_and_floor_free_cells():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=4.0,
        depth_stride_px=1,
        obstacle_min_height_m=0.05,
        obstacle_max_height_m=1.5,
        free_min_height_m=-1.5,
        free_max_height_m=0.1,
        splat_point_threshold=1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[2, 2] = 2.0
    depth[4, 2] = 1.0

    grid = mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    obstacle_cell = grid.world_to_grid(2.0, 0.0)
    free_cell = grid.world_to_grid(1.0, 0.0)

    assert grid.occupied[obstacle_cell] == 1
    assert grid.free[free_cell] == 1
    assert not mapper.traversible(unknown_is_obstacle=True)[obstacle_cell]
    assert mapper.last_debug_stats["mapping_mode"] == "depth_ray_cast"
    assert mapper.last_debug_stats["ray_count"] == 2
    assert mapper.last_debug_stats["free_ray_cells"] >= 1
    assert mapper.last_debug_stats["occupied_endpoint_cells"] >= 1


def test_online_mapper_ignores_ceiling_height_points_for_occupancy():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=4.0,
        depth_stride_px=1,
        obstacle_min_height_m=0.12,
        obstacle_max_height_m=0.90,
        splat_point_threshold=1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[2, 2] = 2.0

    grid = mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.2, 0.0))
    ceiling_like_cell = grid.world_to_grid(2.0, 0.0)

    assert grid.occupied[ceiling_like_cell] == 0
    assert grid.free[ceiling_like_cell] == 0
    assert mapper.traversible(unknown_is_obstacle=False)[ceiling_like_cell]
    assert mapper.last_debug_stats["above_obstacle_max_points"] == 1
    assert mapper.last_debug_stats["obstacle_band_points"] == 0
    assert mapper.last_debug_stats["ray_count"] == 0
    assert mapper.last_debug_stats["skipped_height_rays"] == 1
    assert np.isclose(mapper.last_debug_stats["rel_z_m_percentiles"]["p50"], 1.2)


def test_online_mapper_traversal_inflates_by_robot_radius_only():
    base = OnlineMapper(size_m=4.0, resolution_m=0.1, robot_radius_m=0.11, inflation_radius_m=0.0)
    extra = OnlineMapper(size_m=4.0, resolution_m=0.1, robot_radius_m=0.11, inflation_radius_m=0.2)
    base.reset((0.0, 0.0))
    extra.reset((0.0, 0.0))

    center = base.grid.world_to_grid(0.0, 0.0)
    base.grid.mark_occupied(*center)
    extra.grid.mark_occupied(*center)

    assert int(base.inflated_occupied().sum()) == 13
    assert int(extra.inflated_occupied().sum()) == 13


def test_online_mapper_free_ray_ignores_legacy_splat_thresholds():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=4.0,
        depth_stride_px=1,
        obstacle_min_height_m=0.20,
        obstacle_max_height_m=0.90,
        free_min_height_m=-1.5,
        free_max_height_m=0.1,
        splat_point_threshold=4,
        free_splat_point_threshold=1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[4, 2] = 1.0

    mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    floor_cell = mapper.grid.world_to_grid(1.0, 0.0)
    assert mapper.grid.free[floor_cell] == 1
    assert mapper.last_debug_stats["mapping_mode"] == "depth_ray_cast"
    assert mapper.last_debug_stats["splat_thresholds"]["free"] == 1
    assert mapper.last_debug_stats["splat_thresholds"]["obstacle"] == 4
    assert mapper.last_debug_stats["splat_thresholds"]["unused_for_mapping_mode"] == "depth_ray_cast"


def test_online_mapper_nearfield_topdown_fills_blind_spot_from_depth():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=4.0,
        depth_stride_px=1,
        splat_point_threshold=1,
        free_splat_point_threshold=1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[4, 2] = 1.0
    depth[1, 2] = 0.6

    stats = mapper.update_nearfield_topdown(
        depth,
        intr,
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        radius_m=1.2,
        ignore_radius_m=0.0,
        depth_stride_px=1,
        floor_tolerance_m=0.12,
        obstacle_min_height_m=0.18,
        obstacle_max_height_m=0.90,
        splat_point_threshold=1,
        free_splat_point_threshold=1,
    )

    free_cell = mapper.grid.world_to_grid(-1.0, 0.0)
    assert mapper.grid.free[free_cell] == 1
    assert int(mapper.grid.occupied.sum()) >= 1
    assert stats["free_splat_cells"] >= 1
    assert stats["obstacle_splat_cells"] >= 1


def test_online_mapper_static_nearfield_overrides_local_blind_spot_from_map():
    mapper = OnlineMapper(
        size_m=4.0,
        resolution_m=0.1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    static_info = mapper.grid.map_info
    static_occupancy = np.zeros_like(mapper.grid.occupied, dtype=np.uint8)
    static_navigable = np.zeros_like(mapper.grid.free, dtype=np.uint8)
    static_openings = np.zeros_like(mapper.grid.free, dtype=np.uint8)
    free_cell = mapper.grid.world_to_grid(0.3, 0.0)
    blocked_cell = mapper.grid.world_to_grid(0.0, 0.3)
    door_cell = mapper.grid.world_to_grid(-0.3, 0.0)
    static_navigable[free_cell] = 1
    static_occupancy[blocked_cell] = 1
    static_occupancy[door_cell] = 1
    static_openings[door_cell] = 1
    mapper.grid.mark_occupied(*free_cell)
    mapper.grid.mark_free(*blocked_cell)
    mapper.grid.mark_occupied(*door_cell)

    stats = mapper.update_static_nearfield(
        static_occupancy,
        static_navigable,
        static_info,
        (0.0, 0.0, 0.0, 0.0),
        radius_m=0.5,
        static_openings=static_openings,
    )

    assert mapper.grid.free[free_cell] == 1
    assert mapper.grid.occupied[free_cell] == 0
    assert mapper.grid.free[blocked_cell] == 0
    assert mapper.grid.occupied[blocked_cell] == 0
    assert not mapper.traversible(unknown_is_obstacle=True)[blocked_cell]
    assert mapper.grid.free[door_cell] == 1
    assert mapper.grid.occupied[door_cell] == 0
    assert stats["source"] == "preprocessed_static_map"
    assert stats["free_cells"] >= 1
    assert stats["occupied_cells"] >= 1
    assert stats["opening_cells"] >= 1


def test_online_mapper_floor_ray_follows_camera_yaw():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=2.0,
        depth_min_m=0.1,
        depth_stride_px=1,
        free_min_height_m=-1.5,
        free_max_height_m=0.1,
        splat_point_threshold=1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[4, 2] = 1.0

    mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    assert mapper.grid.free[mapper.grid.world_to_grid(1.0, 0.0)] == 1

    mapper.update(depth, intr, (0.0, 0.0, 0.0, math.pi / 2.0), (0.0, 0.0, 1.0, math.pi / 2.0))
    assert mapper.grid.free[mapper.grid.world_to_grid(0.0, 1.0)] == 1


def test_online_mapper_free_ray_clears_stale_obstacle_cells():
    mapper = OnlineMapper(
        size_m=8.0,
        resolution_m=0.1,
        depth_max_m=2.0,
        depth_stride_px=1,
        free_min_height_m=-1.5,
        free_max_height_m=0.1,
        splat_point_threshold=1,
        robot_radius_m=0.1,
    )
    mapper.reset((0.0, 0.0))
    shared_cell = mapper.grid.world_to_grid(1.0, 0.0)
    mapper.grid.mark_occupied(*shared_cell)
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)
    depth[4, 2] = 1.0

    mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    assert mapper.grid.free[shared_cell] == 1
    assert mapper.grid.occupied[shared_cell] == 0
    assert mapper.traversible(unknown_is_obstacle=True)[shared_cell]


def test_online_mapper_invalid_depth_only_marks_robot_footprint():
    mapper = OnlineMapper(size_m=8.0, resolution_m=0.1, depth_max_m=2.0, depth_stride_px=1, robot_radius_m=0.1)
    mapper.reset((0.0, 0.0))
    intr = CameraIntrinsics(width=5, height=5, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    depth = np.full((5, 5), np.inf, dtype=np.float32)

    mapper.update(depth, intr, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

    assert mapper.grid.free[mapper.grid.world_to_grid(0.0, 0.0)] == 1
    assert mapper.grid.free[mapper.grid.world_to_grid(1.0, 0.0)] == 0
