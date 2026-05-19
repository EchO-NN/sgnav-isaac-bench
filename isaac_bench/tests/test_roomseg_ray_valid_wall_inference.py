import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.online_mapper import OnlineMapper, _append_vertical_profile_free_ray_cells
from isaac_bench.mapping.roomseg_ray_valid_wall import build_ray_valid_wall_inference
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import (
    UpstreamROSE2Config,
    _vertical_profile_structural_maps,
)
from isaac_bench.mapping.room_segmentation import RoomSegmentationConfig
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


def test_valid_depth_terminal_wall_is_not_unknown():
    shape = (1, 5)
    vertical_free = np.zeros(shape, dtype=bool)
    vertical_free[0, 1:3] = True
    terminal_count = np.zeros(shape, dtype=np.uint16)
    terminal_count[0, 3] = 1

    result = build_ray_valid_wall_inference(
        vertical_free=vertical_free,
        vertical_occupied=np.zeros(shape, dtype=bool),
        vertical_observed=vertical_free,
        terminal_wall_count=terminal_count,
        config={"enabled": True, "terminal_wall_splat_radius_cells": 0},
    )

    assert result["initial_roomseg_free"][0, 1]
    assert result["initial_roomseg_free"][0, 2]
    assert result["initial_roomseg_occupied"][0, 3]
    assert not result["initial_roomseg_unknown"][0, 3]


def test_vertical_free_cannot_be_overwritten_by_terminal_wall_splat():
    shape = (3, 3)
    vertical_free = np.zeros(shape, dtype=bool)
    vertical_free[1, 1] = True
    terminal_count = np.zeros(shape, dtype=np.uint16)
    terminal_count[1, 1] = 3

    result = build_ray_valid_wall_inference(
        vertical_free=vertical_free,
        vertical_occupied=np.zeros(shape, dtype=bool),
        vertical_observed=vertical_free,
        terminal_wall_count=terminal_count,
        config={"enabled": True, "terminal_wall_splat_radius_cells": 1, "require_no_vertical_free": True},
    )

    assert result["initial_roomseg_free"][1, 1]
    assert not result["initial_roomseg_occupied"][1, 1]
    assert result["debug"]["vertical_free_overridden_by_wall_cells"] == 0


def test_endpoint_beyond_depth_max_produces_no_terminal_wall():
    mapper = OnlineMapper(size_m=2.0, resolution_m=0.1, depth_max_m=3.0)
    mapper.reset((0.0, 0.0))
    row, col = mapper.grid.world_to_grid(0.4, 0.0)

    mapper._mark_roomseg_terminal_wall_cell(row, col, endpoint_depth_m=3.5, endpoint_rel_z_m=1.2)

    assert mapper.roomseg_terminal_wall_count[row, col] == 0
    result = build_ray_valid_wall_inference(
        vertical_free=np.zeros_like(mapper.grid.free, dtype=bool),
        vertical_occupied=np.zeros_like(mapper.grid.free, dtype=bool),
        terminal_wall_count=mapper.roomseg_terminal_wall_count,
        config={"enabled": True, "terminal_wall_splat_radius_cells": 0},
    )
    assert not result["initial_roomseg_occupied"][row, col]
    assert result["initial_roomseg_unknown"][row, col]


def test_high_wall_endpoint_writes_roomseg_evidence_without_navigation_free():
    mapper = OnlineMapper(
        size_m=2.0,
        resolution_m=0.1,
        depth_max_m=3.0,
        obstacle_max_height_m=0.9,
        vertical_profile_free_min_height_m=0.2,
        vertical_profile_free_max_height_m=2.0,
    )
    mapper.reset((0.0, 0.0))
    map_width = int(mapper.grid.map_info.width)
    line = [(10, 10), (10, 11), (10, 12), (10, 13)]
    free_by_band = [[] for _ in mapper.vertical_profile.band_names]
    added = _append_vertical_profile_free_ray_cells(
        free_by_band,
        line[:-1],
        map_width=map_width,
        origin_rel_z_m=1.2,
        endpoint_rel_z_m=1.6,
        z_min_m=0.2,
        z_max_m=2.0,
        vertical_profile=mapper.vertical_profile,
    )
    mapper._mark_vertical_profile_free_flat(free_by_band)
    mapper._mark_roomseg_ray_covered_flat([r * map_width + c for r, c in line[:-1]])
    mapper._mark_roomseg_terminal_wall_cell(10, 13, endpoint_depth_m=1.8, endpoint_rel_z_m=1.6)

    assert added > 0
    assert not np.any(mapper.grid.free.astype(bool))
    assert mapper.roomseg_ray_covered_count[10, 11] > 0
    assert mapper.roomseg_terminal_wall_count[10, 13] == 1
    assert mapper.vertical_profile.reliable_free_mask(band_names=("mid", "upper"))[10, 11]


def test_adapter_does_not_use_navigation_obstacle_overlay_as_wall():
    shape = (4, 4)
    nav_obstacle = np.zeros(shape, dtype=bool)
    nav_obstacle[2, 2] = True
    free = np.zeros(shape, dtype=bool)
    unknown = np.ones(shape, dtype=bool)
    vp = VerticalProfileMap.zeros(shape)
    cfg = UpstreamROSE2Config(
        source_root=None,
        backend="vertical_free_gap_closure_v1",
        fail_on_missing_source=False,
        roomseg_use_navigation_obstacle_for_vertical_unknown=True,
        ray_valid_wall_strict_no_navigation_obstacle_overlay=True,
    )
    room_cfg = RoomSegmentationConfig(
        algorithm="vertical_free_gap_closure_v1",
        source_grid="online_depth_observed",
        resolution_m=0.05,
        map_info=MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.2, min_y=0.0, max_y=0.2, width=4, height=4),
    )

    result = _vertical_profile_structural_maps(
        occupied=nav_obstacle,
        free=free,
        unknown=unknown,
        vertical_profile=vp,
        vertical_profile_provided=True,
        roomseg_static_structural_occupied=None,
        roomseg_ray_evidence={},
        object_memory=[],
        map_info=room_cfg.map_info,
        config=cfg,
        room_config=room_cfg,
    )

    assert not result["initial_roomseg_occupied"][2, 2]
    assert result["initial_roomseg_unknown"][2, 2]
    assert result["navigation_free_added_to_strict_roomseg_cells"] == 0
    assert result["evidence_fusion"]["navigation_obstacle_overlay_for_roomseg_occupied"] is False
