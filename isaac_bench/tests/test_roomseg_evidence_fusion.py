import numpy as np

from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.mapping.roomseg_evidence_fusion import (
    EVIDENCE_FUSION_MODE,
    fuse_vertical_profile_with_navigation_obstacles,
)


def _single_cell(**values):
    return {key: np.asarray([[bool(value)]], dtype=bool) for key, value in values.items()}


def _fuse(**values):
    masks = _single_cell(
        vertical_free=values.get("vertical_free", False),
        vertical_observed=values.get("vertical_observed", False),
        nav_raw_obstacle=values.get("nav_raw_obstacle", False),
        static_structural_occupied=values.get("static_structural_occupied", False),
        navigation_free=values.get("navigation_free", False),
        inflated_obstacle=values.get("inflated_obstacle", False),
    )
    return fuse_vertical_profile_with_navigation_obstacles(
        vertical_free=masks["vertical_free"],
        vertical_observed=masks["vertical_observed"],
        nav_raw_obstacle=masks["nav_raw_obstacle"],
        static_structural_occupied=masks["static_structural_occupied"],
        navigation_free=masks["navigation_free"],
        inflated_obstacle=masks["inflated_obstacle"],
        config={
            "enabled": True,
            "mode": EVIDENCE_FUSION_MODE,
            "vertical_free_priority": True,
            "use_navigation_obstacle_for_vertical_unknown": True,
            "use_navigation_free_for_roomseg_free": False,
            "use_inflated_obstacle": False,
            "use_depth_valid_as_wall": False,
            "use_static_structural_occupied": True,
        },
    )


def _cell(result, key):
    return bool(np.asarray(result[key], dtype=bool)[0, 0])


def test_vertical_free_wins_over_navigation_obstacle():
    result = _fuse(vertical_free=True, vertical_observed=True, nav_raw_obstacle=True)

    assert _cell(result, "initial_roomseg_free")
    assert not _cell(result, "initial_roomseg_occupied")
    assert not _cell(result, "initial_roomseg_unknown")
    assert _cell(result, "vertical_free_over_nav_obstacle")


def test_vertical_unknown_plus_nav_obstacle_becomes_wall():
    result = _fuse(vertical_free=False, vertical_observed=False, nav_raw_obstacle=True)

    assert not _cell(result, "initial_roomseg_free")
    assert _cell(result, "initial_roomseg_occupied")
    assert not _cell(result, "initial_roomseg_unknown")
    assert _cell(result, "nav_obstacle_overlay_accepted")
    assert _cell(result, "walls_rescued_from_unknown")


def test_vertical_unknown_plus_nav_non_obstacle_stays_unknown():
    result = _fuse(vertical_free=False, vertical_observed=False, nav_raw_obstacle=False)

    assert not _cell(result, "initial_roomseg_free")
    assert not _cell(result, "initial_roomseg_occupied")
    assert _cell(result, "initial_roomseg_unknown")


def test_vertical_observed_non_free_remains_occupied():
    result = _fuse(vertical_free=False, vertical_observed=True, nav_raw_obstacle=False)

    assert not _cell(result, "initial_roomseg_free")
    assert _cell(result, "initial_roomseg_occupied")
    assert not _cell(result, "initial_roomseg_unknown")


def test_navigation_free_does_not_create_roomseg_free():
    result = _fuse(
        vertical_free=False,
        vertical_observed=False,
        nav_raw_obstacle=False,
        navigation_free=True,
    )

    assert not _cell(result, "initial_roomseg_free")
    assert not _cell(result, "initial_roomseg_occupied")
    assert _cell(result, "initial_roomseg_unknown")
    assert result["debug"]["navigation_free_cells_seen_but_not_used_as_roomseg_free"] == 1


def test_static_structural_occupied_source_rescues_wall():
    result = _fuse(
        vertical_free=False,
        vertical_observed=False,
        nav_raw_obstacle=False,
        static_structural_occupied=True,
    )

    assert _cell(result, "initial_roomseg_occupied")
    assert _cell(result, "nav_obstacle_overlay_accepted")
    assert result["debug"]["static_structural_occupied_cells"] == 1


def test_static_opening_is_not_roomseg_static_wall():
    mapper = OnlineMapper(size_m=1.0, resolution_m=0.1)
    mapper.reset((0.0, 0.0))
    shape = mapper.grid.free.shape
    static_occupancy = np.ones(shape, dtype=bool)
    static_navigable = np.zeros(shape, dtype=bool)
    static_openings = np.ones(shape, dtype=bool)

    stats = mapper.update_static_nearfield(
        static_occupancy,
        static_navigable,
        mapper.grid.map_info,
        base_pose_world=(0.0, 0.0, 0.0, 0.0),
        radius_m=0.2,
        static_openings=static_openings,
    )

    assert stats["opening_cells"] > 0
    assert stats["roomseg_static_structural_occupied_cells"] == 0
    assert not np.any(mapper.roomseg_static_structural_occupied.astype(bool))


def test_static_nonopening_structural_wall_is_roomseg_only_source():
    mapper = OnlineMapper(size_m=1.0, resolution_m=0.1)
    mapper.reset((0.0, 0.0))
    shape = mapper.grid.free.shape
    static_occupancy = np.ones(shape, dtype=bool)
    static_navigable = np.zeros(shape, dtype=bool)
    static_openings = np.zeros(shape, dtype=bool)

    stats = mapper.update_static_nearfield(
        static_occupancy,
        static_navigable,
        mapper.grid.map_info,
        base_pose_world=(0.0, 0.0, 0.0, 0.0),
        radius_m=0.2,
        static_openings=static_openings,
    )

    assert stats["roomseg_static_structural_occupied_cells"] > 0
    assert np.any(mapper.roomseg_static_structural_occupied.astype(bool))
    assert not np.any(mapper.grid.occupied.astype(bool))


def test_inflated_obstacle_is_not_used_as_wall_source():
    result = _fuse(
        vertical_free=False,
        vertical_observed=False,
        nav_raw_obstacle=False,
        inflated_obstacle=True,
    )

    assert not _cell(result, "initial_roomseg_occupied")
    assert _cell(result, "initial_roomseg_unknown")
    assert result["debug"]["inflated_obstacle_cells_seen_but_ignored"] == 1
