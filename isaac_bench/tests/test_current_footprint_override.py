from __future__ import annotations

from isaac_bench.mapping.online_mapper import OnlineMapper


def test_current_footprint_override_clears_dynamic_occupied_for_start_cell() -> None:
    mapper = OnlineMapper(
        size_m=2.0,
        resolution_m=0.10,
        voxel_navigation_blind_zone_config={
            "enabled": True,
            "force_current_footprint_free": True,
            "current_footprint_radius_m": 0.10,
            "current_footprint_extra_margin_m": 0.0,
            "current_footprint_clear_dynamic_occupied": True,
            "write_to_grid": True,
        },
    )
    mapper.reset((0.0, 0.0))
    row, col = mapper.grid.world_to_grid(0.0, 0.0)
    mapper.grid.free[row, col] = 0
    mapper.grid.occupied[row, col] = 1
    mapper.grid.observed[row, col] = 1

    debug = mapper._apply_voxel_navigation_blind_zone((0.0, 0.0, 0.0, 0.0), write_to_voxel=False, write_to_grid=True)

    assert mapper.grid.free[row, col] == 1
    assert mapper.grid.occupied[row, col] == 0
    assert mapper.current_pose_navigation_override_mask[row, col]
    assert mapper.traversible(unknown_is_obstacle=True)[row, col]
    assert int(debug["current_pose_dynamic_occupied_cleared_cells"]) >= 1


def test_current_footprint_override_preserves_static_wall() -> None:
    mapper = OnlineMapper(
        size_m=2.0,
        resolution_m=0.10,
        voxel_navigation_blind_zone_config={
            "enabled": True,
            "force_current_footprint_free": True,
            "current_footprint_radius_m": 0.10,
            "current_footprint_extra_margin_m": 0.0,
            "current_footprint_clear_dynamic_occupied": True,
            "current_footprint_preserve_static_structural": True,
            "write_to_grid": True,
        },
    )
    mapper.reset((0.0, 0.0))
    row, col = mapper.grid.world_to_grid(0.0, 0.0)
    mapper.grid.free[row, col] = 0
    mapper.grid.occupied[row, col] = 1
    mapper.grid.observed[row, col] = 1
    mapper.roomseg_static_structural_occupied[row, col] = 1

    debug = mapper._apply_voxel_navigation_blind_zone((0.0, 0.0, 0.0, 0.0), write_to_voxel=False, write_to_grid=True)

    assert mapper.grid.free[row, col] == 0
    assert mapper.grid.occupied[row, col] == 1
    assert not mapper.current_pose_navigation_override_mask[row, col]
    assert int(debug["current_pose_static_wall_preserved_cells"]) >= 1
