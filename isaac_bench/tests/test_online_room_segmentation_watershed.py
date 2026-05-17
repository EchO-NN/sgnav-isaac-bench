import numpy as np

from isaac_bench.mapping.room_segmentation import OnlineRoomSegmenter, RoomSegmentationConfig


def _cfg():
    return RoomSegmentationConfig(
        resolution_m=0.1,
        min_observed_free_cells=20,
        min_room_area_m2=0.2,
        morphology_close_radius_m=0.0,
        morphology_open_radius_m=0.0,
        seed_min_clearance_m=0.2,
        seed_min_distance_m=2.0,
        doorway_width_min_m=0.4,
        doorway_width_max_m=1.4,
        doorway_clearance_max_m=2.5,
        small_segment_merge_area_m2=0.2,
        max_clutter_component_area_m2=0.4,
    )


def test_two_rooms_with_narrow_doorway_produce_masks_and_doorway_edge():
    free = np.zeros((80, 80), dtype=bool)
    free[10:70, 8:72] = True
    obstacle = np.zeros_like(free)
    obstacle[:, 39:41] = True
    free[:, 39:41] = False
    free[34:45, 39:41] = True
    obstacle[34:45, 39:41] = False
    unknown = ~free & ~obstacle

    rooms = OnlineRoomSegmenter(_cfg()).update(obstacle, free, obstacle, unknown, step=0)

    live = [room for room in rooms if not room.stale]
    assert len(live) == 2
    assert any(edge["edge_type"] == "adjacent_via_doorway" for room in live for edge in room.doorway_edges)


def test_open_plan_room_is_not_over_split_into_tiny_rooms():
    free = np.zeros((60, 80), dtype=bool)
    free[8:52, 8:72] = True
    obstacle = np.zeros_like(free)
    unknown = ~free

    rooms = OnlineRoomSegmenter(_cfg()).update(obstacle, free, obstacle, unknown, step=0)

    assert len([room for room in rooms if not room.stale]) == 1


def test_small_clutter_obstacle_does_not_split_room():
    free = np.zeros((50, 50), dtype=bool)
    free[5:45, 5:45] = True
    obstacle = np.zeros_like(free)
    obstacle[22:24, 22:24] = True
    free[22:24, 22:24] = False
    unknown = ~free & ~obstacle

    rooms = OnlineRoomSegmenter(_cfg()).update(obstacle, free, obstacle, unknown, step=0)

    assert len([room for room in rooms if not room.stale]) == 1


def test_unknown_cells_are_not_included_in_room_masks():
    free = np.zeros((30, 30), dtype=bool)
    free[5:25, 5:25] = True
    obstacle = np.zeros_like(free)
    unknown = np.zeros_like(free)
    unknown[10:15, 10:15] = True
    free[10:15, 10:15] = False

    rooms = OnlineRoomSegmenter(_cfg()).update(obstacle, free, obstacle, unknown, step=0)

    assert rooms
    assert all(int(np.count_nonzero(room.mask & unknown)) == 0 for room in rooms)
