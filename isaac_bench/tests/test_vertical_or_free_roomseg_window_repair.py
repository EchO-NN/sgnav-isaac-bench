import sys

import numpy as np

from isaac_bench.mapping.vertical_profile import VerticalProfileMap, band_index
from isaac_bench.perception.object_memory import ObjectNode
from isaac_bench.tests.test_vertical_profile_rose2_room_segmentation import _segmenter, _two_room_wall


def _profile(shape):
    return VerticalProfileMap.zeros(shape)


def _mark(profile, mask, bands, *, free=False, occupied=False, count=5):
    for name in bands:
        idx = band_index(name)
        if free:
            profile.free_ray_count[idx, mask] = count
        if occupied:
            profile.occupied_count[idx, mask] = count
        profile.observed_count[idx, mask] = count
        profile.unknown_count[idx, mask] = 0


def test_vertical_or_free_marks_furniture_space_free(tmp_path):
    shape = (50, 60)
    occupied = np.zeros(shape, dtype=bool)
    free = np.zeros(shape, dtype=bool)
    free[6:44, 6:54] = True
    furniture = np.zeros(shape, dtype=bool)
    furniture[22:29, 25:36] = True
    occupied[furniture] = True
    free[furniture] = False
    unknown = ~(free | occupied)
    profile = _profile(shape)
    _mark(profile, free, ("low", "robot_body", "mid", "upper"), free=True)
    _mark(profile, furniture, ("low",), occupied=True)
    _mark(profile, furniture, ("mid",), free=True)

    segmenter = _segmenter(tmp_path, shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert np.count_nonzero(segmenter.last_debug["vertical_or_free_map"][furniture]) > 0
    assert np.count_nonzero(segmenter.last_debug["initial_roomseg_occupied"][furniture]) == 0
    assert np.all(segmenter.last_debug["repaired_roomseg_free"][furniture])
    assert segmenter.last_debug["vertical_free_overrides_occupied"] is True
    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][furniture]) == 0


def test_vertical_or_free_any_band_free_overrides_roomseg_occupied(tmp_path):
    shape = (20, 20)
    occupied = np.zeros(shape, dtype=bool)
    occupied[10, 10] = True
    free = np.ones(shape, dtype=bool)
    free[10, 10] = False
    unknown = ~(free | occupied)
    profile = _profile(shape)
    _mark(profile, free, ("low", "robot_body"), free=True)
    upper_only = np.zeros(shape, dtype=bool)
    upper_only[10, 10] = True
    _mark(profile, upper_only, ("upper",), free=True)

    segmenter = _segmenter(tmp_path, shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert segmenter.last_debug["vertical_or_free_map"][10, 10]
    assert not segmenter.last_debug["initial_roomseg_occupied"][10, 10]
    assert segmenter.last_debug["repaired_roomseg_free"][10, 10]
    assert not segmenter.last_debug["structural_wall_mask"][10, 10]


def test_rose_roomseg_input_does_not_overlay_2d_occupied_without_vertical_observation(tmp_path):
    shape = (24, 24)
    occupied = np.zeros(shape, dtype=bool)
    occupied[12, 12] = True
    free = np.zeros(shape, dtype=bool)
    unknown = ~(free | occupied)
    profile = _profile(shape)

    segmenter = _segmenter(tmp_path, shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert segmenter.last_debug["roomseg_input_source"] == "vertical_profile_only"
    assert not segmenter.last_debug["vertical_observed_map"][12, 12]
    assert not segmenter.last_debug["initial_roomseg_occupied"][12, 12]
    assert not segmenter.last_debug["initial_roomseg_free"][12, 12]


def test_rose_roomseg_input_marks_observed_column_without_free_as_wall(tmp_path):
    shape = (24, 24)
    occupied = np.zeros(shape, dtype=bool)
    free = np.zeros(shape, dtype=bool)
    unknown = ~(free | occupied)
    profile = _profile(shape)
    wall_like = np.zeros(shape, dtype=bool)
    wall_like[12, 12] = True
    _mark(profile, wall_like, ("mid",), occupied=True)

    segmenter = _segmenter(tmp_path, shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert segmenter.last_debug["vertical_observed_map"][12, 12]
    assert not segmenter.last_debug["vertical_or_free_map"][12, 12]
    assert segmenter.last_debug["initial_roomseg_occupied"][12, 12]
    assert not segmenter.last_debug["initial_roomseg_free"][12, 12]


def test_line_supported_wall_is_carved_when_roomseg_vertical_free_exists(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall()
    wall = np.zeros_like(occupied, dtype=bool)
    wall[:, col : col + 2] = occupied[:, col : col + 2]
    _mark(profile, wall, ("mid", "upper"), free=True)

    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert np.count_nonzero(segmenter.last_debug["vertical_or_free_map"][:, col : col + 2]) > 0
    assert np.count_nonzero(segmenter.last_debug["initial_roomseg_occupied"][:, col : col + 2]) == 0
    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][:, col : col + 2]) == 0
    assert segmenter.last_debug["vertical_free_overridden_occupied_cells"] > 0


def test_vertical_or_free_does_not_modify_navigation_planner_map(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall()
    original_occupied = occupied.copy()
    carved = np.zeros_like(occupied, dtype=bool)
    carved[20:25, 20:25] = True
    occupied[carved] = True
    free[carved] = False
    _mark(profile, carved, ("mid",), free=True)

    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert np.array_equal(occupied, original_occupied | carved)
    assert np.count_nonzero(segmenter.last_debug["initial_roomseg_occupied"][carved]) == 0
    assert np.all(segmenter.last_debug["repaired_roomseg_free"][carved])
    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][carved]) == 0


def test_navigation_free_path_cells_are_not_added_to_vertical_only_roomseg_input(tmp_path):
    shape = (48, 64)
    occupied = np.zeros(shape, dtype=bool)
    free = np.zeros(shape, dtype=bool)
    free[10:34, 10:34] = True
    free[22:25, 34:54] = True
    unknown = ~(free | occupied)
    profile = _profile(shape)
    room_core = np.zeros(shape, dtype=bool)
    room_core[10:34, 10:34] = True
    _mark(profile, room_core, ("low", "robot_body", "mid", "upper"), free=True)
    path_only_nav_free = np.zeros(shape, dtype=bool)
    path_only_nav_free[22:25, 38:52] = True

    segmenter = _segmenter(tmp_path, shape)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    room_union = np.zeros(shape, dtype=bool)
    for room in rooms:
        room_union |= np.asarray(room.mask, dtype=bool)
    assert segmenter.last_debug["roomseg_input_source"] == "vertical_profile_only"
    assert segmenter.last_debug["navigation_free_added_to_roomseg_cells"] == 0
    assert segmenter.last_debug["navigation_free_not_added_to_roomseg_cells"] > 0
    assert not np.any(segmenter.last_debug["repaired_roomseg_free"][path_only_nav_free])
    assert not np.any(room_union[path_only_nav_free])


def test_window_high_gap_floor_blocked_is_closed_as_wall(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall(window=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert not segmenter.last_debug["repaired_window_gaps"]
    assert segmenter.last_debug["vertical_free_overrides_occupied"] is True


def test_curtain_or_window_detection_without_floor_traversability_is_closed(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True)
    object_memory = [ObjectNode(1, "curtain", (4.4, 3.4, 1.0), (34, col), 0.9, 2, 1)]
    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert not segmenter.last_debug["repaired_window_gaps"]
    assert np.count_nonzero(segmenter.last_debug["furniture_suppression_mask"]) == 0


def test_exterior_window_gap_does_not_merge_rooms(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True, exterior=True)
    object_memory = [ObjectNode(2, "window", (1.0, 3.4, 1.0), (34, col), 0.9, 2, 1)]
    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert len([room for room in rooms if not room.stale]) >= 1
    assert not segmenter.last_debug["repaired_window_gaps"]
    assert not segmenter.last_debug["verified_doorway_gaps"]


def test_vertical_free_window_gap_is_repaired_before_rose2(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    gap_rows = slice(30, 38)
    assert np.count_nonzero(segmenter.last_debug["vertical_or_free_map"][gap_rows, col : col + 2]) > 0
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_occupied"][gap_rows, col : col + 2]) == 0
    assert np.all(segmenter.last_debug["repaired_roomseg_free"][gap_rows, col : col + 2])


def test_window_object_node_does_not_create_room_adjacency(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True)
    object_memory = [ObjectNode(3, "window", (4.4, 3.4, 1.0), (34, col), 0.9, 2, 1)]
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert not segmenter.last_debug["repaired_window_gaps"]
    assert not segmenter.last_debug["verified_doorway_gaps"]


def test_floor_traversable_doorway_remains_open(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall(doorway=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert not segmenter.last_debug["verified_doorway_gaps"]
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_free"][:, _col : _col + 2]) > 0


def test_lintel_high_occupied_but_floor_free_preserves_doorway(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(doorway=True)
    lintel = np.zeros_like(occupied, dtype=bool)
    lintel[32:42, col : col + 2] = True
    _mark(profile, lintel, ("upper",), occupied=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert not segmenter.last_debug["verified_doorway_gaps"]
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_free"][:, col : col + 2]) > 0


def test_gap_without_wall_support_is_not_doorway(tmp_path):
    shape = (40, 40)
    occupied = np.zeros(shape, dtype=bool)
    free = np.zeros(shape, dtype=bool)
    free[5:35, 5:35] = True
    unknown = ~(free | occupied)
    profile = _profile(shape)
    _mark(profile, free, ("low", "robot_body"), free=True)
    segmenter = _segmenter(tmp_path, shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert not segmenter.last_debug.get("verified_doorway_gaps")


def test_rose2_consumes_repaired_vertical_carved_map(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert "repaired_roomseg_occupied" in segmenter.last_debug
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_occupied"][:, col : col + 2]) >= np.count_nonzero(
        segmenter.last_debug["pre_repair_structural_wall_mask"][:, col : col + 2]
    )


def test_strict_mode_does_not_use_watershed_fallback(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall()
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert segmenter.last_debug["algorithm"] == "rose2_source_form_v2"
    assert segmenter.last_debug["strict_fallback_used"] is False


def test_no_ros_dependency_in_strict_roomseg():
    for name in ("rospy", "nav_msgs", "jsk_recognition_msgs", "roslaunch"):
        assert name not in sys.modules
