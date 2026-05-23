from .test_helpers import corridor_with_two_rooms, open_room, run_masks


def test_geometry_fallback_outputs_unknown_zone():
    free, wall, unknown = open_room()
    out = run_masks(free, wall, unknown)
    assert out.room_instances
    assert all(room.functional_zone_label in {"unknown_functional_zone", "storage_zone"} for room in out.room_instances)


def test_corridor_outputs_corridor_zone():
    free, wall, unknown = corridor_with_two_rooms()
    out = run_masks(free, wall, unknown)
    assert any(room.functional_zone_label == "corridor_zone" for room in out.room_instances)


def test_kitchen_and_living_objects_vote_zones():
    free, wall, unknown = open_room()
    objects = [
        {"object_id": 1, "label": "stove", "uv": [30, 30], "confidence": 1.0},
        {"object_id": 2, "label": "refrigerator", "uv": [32, 32], "confidence": 1.0},
    ]
    out = run_masks(free, wall, unknown, objects=objects)
    assert any(room.functional_zone_label == "kitchen_zone" for room in out.room_instances)

