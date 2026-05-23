import numpy as np

from .test_helpers import corridor_with_two_rooms, open_room, run_masks, two_rooms_one_door


def _assert_basic(out, free, unknown):
    assert out.room_id_map.shape == free.shape
    assert out.room_confidence_map.shape == free.shape
    assert out.room_instances
    assert np.all(out.room_id_map[unknown] == -1)
    assigned = np.count_nonzero((out.room_id_map > 0) & free) / max(1, np.count_nonzero(free))
    assert assigned >= 0.85
    assert not np.isnan(out.room_confidence_map).any()
    assert all(room.area_m2 > 0 for room in out.room_instances)
    ids = {room.room_id for room in out.room_instances}
    assert all(a in ids and b in ids for a, b, _w in out.room_graph_edges)


def test_two_rooms_one_door_integration():
    free, wall, unknown = two_rooms_one_door()
    out = run_masks(free, wall, unknown)
    _assert_basic(out, free, unknown)
    assert len(out.room_instances) == 2


def test_corridor_with_two_rooms_integration():
    free, wall, unknown = corridor_with_two_rooms()
    out = run_masks(free, wall, unknown)
    _assert_basic(out, free, unknown)
    assert np.count_nonzero(out.corridor_mask) > 0


def test_open_kitchen_living_is_not_shredded():
    free, wall, unknown = open_room()
    out = run_masks(free, wall, unknown)
    _assert_basic(out, free, unknown)
    assert len(out.room_instances) <= 2
    assert np.count_nonzero(out.open_space_mask) > 0


def test_furniture_clutter_room_stays_conservative():
    free, wall, unknown = open_room()
    furniture = np.zeros_like(free)
    furniture[25:34, 40:43] = True
    free[furniture] = False
    wall = wall | furniture
    unknown = ~(free | wall)
    out = run_masks(free, wall, unknown)
    _assert_basic(out, free, unknown)
    assert len(out.room_instances) <= 2


def test_partial_frontier_confidence_is_lower():
    free, wall, unknown = two_rooms_one_door()
    unknown[:, 70:] = True
    free[:, 70:] = False
    wall[:, 70:] = False
    out = run_masks(free, wall, unknown)
    _assert_basic(out, free, unknown)
    frontier = out.debug_layers["frontier_mask"].astype(bool)
    if np.any(frontier & (out.room_id_map > 0)):
        assert float(np.mean(out.room_confidence_map[frontier & (out.room_id_map > 0)])) <= 0.70

