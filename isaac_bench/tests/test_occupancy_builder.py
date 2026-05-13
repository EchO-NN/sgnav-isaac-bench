import numpy as np

from isaac_bench.dataset.episode_generator import filter_start_clearance_objects
from isaac_bench.dataset.occupancy_builder import build_navigable, carve_passable_openings, rasterize_bbox_occupancy
from isaac_bench.mapping.coordinate_transform import MapInfo


def test_doors_are_passable_for_occupancy_and_clearance():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    door = {
        "category": "door",
        "bbox_min_world": [1.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 3.0, 2.0],
    }
    chair = {
        "category": "chair",
        "bbox_min_world": [3.0, 3.0, 0.0],
        "bbox_max_world": [4.0, 4.0, 1.0],
    }

    occupancy = rasterize_bbox_occupancy([door, chair], info, min_z=0.05, max_z=1.5)

    assert occupancy.sum() > 0
    assert not occupancy[3, 1]
    assert occupancy[1, 3]
    assert filter_start_clearance_objects([door, chair]) == [chair]


def test_navigable_inflation_uses_robot_radius_plus_extra_only():
    occupancy = np.zeros((9, 9), dtype=np.uint8)
    occupancy[4, 4] = 1

    navigable_robot_only, inflated_robot_only = build_navigable(
        occupancy,
        resolution_m=0.1,
        robot_radius_m=0.11,
        inflation_radius_m=0.0,
    )
    navigable_extra, inflated_extra = build_navigable(
        occupancy,
        resolution_m=0.1,
        robot_radius_m=0.11,
        inflation_radius_m=0.2,
    )

    assert inflated_robot_only.sum() < inflated_extra.sum()
    assert navigable_robot_only.sum() > navigable_extra.sum()


def test_door_bbox_carves_wall_occupancy_for_passage():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    wall = {
        "category": "wall",
        "bbox_min_world": [1.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 3.0, 2.0],
    }
    door = {
        "category": "door",
        "bbox_min_world": [1.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 3.0, 2.0],
    }

    occupancy = rasterize_bbox_occupancy([wall, door], info, min_z=0.05, max_z=1.5)
    carved = carve_passable_openings(occupancy, [wall, door], info, min_z=0.05, max_z=1.5)

    assert occupancy.sum() > 0
    assert carved.sum() == 0
