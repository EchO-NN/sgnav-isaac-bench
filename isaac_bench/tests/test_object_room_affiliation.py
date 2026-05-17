from types import SimpleNamespace

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import RoomMask, assign_objects_to_room_masks


def _map_info():
    return MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)


def _room(room_id, rows, cols):
    mask = np.zeros((10, 10), dtype=bool)
    mask[rows, cols] = True
    return RoomMask(
        room_id=room_id,
        mask=mask,
        centroid_xy=(0.0, 0.0),
        area_m2=float(np.count_nonzero(mask)),
        boundary_unknown_fraction=0.0,
        doorway_edges=[],
        confidence=1.0,
    )


def test_object_centroid_inside_room_creates_contains_assignment():
    rooms = [_room("room_0001", slice(0, 5), slice(0, 5))]
    obj = SimpleNamespace(id="object:1", center_world=np.asarray([2.5, 7.5, 0.5]), point_cloud_world=None, bbox_world=None)

    out = assign_objects_to_room_masks([obj], rooms, _map_info())

    assignment = out["object:1"]
    assert assignment.room_id == "room_0001"
    assert assignment.centroid_inside is True
    assert assignment.to_edge_metadata()["edge_type"] == "contains"


def test_footprint_overlap_resolves_centroid_outside_room():
    rooms = [_room("room_0001", slice(0, 5), slice(0, 5))]
    points = np.asarray([[2.0, 7.0, 0.5], [3.0, 7.0, 0.5]], dtype=np.float32)
    obj = SimpleNamespace(id="object:1", center_world=np.asarray([8.5, 1.5, 0.5]), point_cloud_world=points, bbox_world=None)

    out = assign_objects_to_room_masks([obj], rooms, _map_info())

    assert out["object:1"].room_id == "room_0001"
    assert out["object:1"].centroid_inside is False
    assert out["object:1"].mask_overlap_ratio > 0.0


def test_ambiguous_overlap_is_recorded():
    rooms = [_room("room_a", slice(0, 5), slice(0, 5)), _room("room_b", slice(0, 5), slice(5, 10))]
    points = np.asarray([[2.0, 7.0, 0.5], [7.0, 7.0, 0.5]], dtype=np.float32)
    obj = SimpleNamespace(id="object:1", center_world=np.asarray([5.0, 7.0, 0.5]), point_cloud_world=points, bbox_world=None)

    out = assign_objects_to_room_masks([obj], rooms, _map_info(), overlap_ambiguity_margin=0.2)

    assert out["object:1"].ambiguous is True
