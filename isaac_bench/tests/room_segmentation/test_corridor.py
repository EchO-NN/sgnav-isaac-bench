import numpy as np

from .test_helpers import corridor_with_two_rooms, open_room, run_masks


def test_long_narrow_region_is_corridor():
    free, wall, unknown = corridor_with_two_rooms()
    out = run_masks(free, wall, unknown)
    assert np.count_nonzero(out.corridor_mask) > 0
    assert any(room.is_corridor for room in out.room_instances)


def test_ordinary_open_room_is_not_corridor():
    free, wall, unknown = open_room()
    out = run_masks(free, wall, unknown)
    assert not any(room.is_corridor for room in out.room_instances)

