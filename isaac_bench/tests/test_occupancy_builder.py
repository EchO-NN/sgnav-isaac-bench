import numpy as np

from isaac_bench.dataset.episode_generator import filter_start_clearance_objects
from isaac_bench.dataset.occupancy_builder import (
    bbox_grid_bounds,
    build_navigable,
    carve_passable_openings,
    rasterize_bbox_occupancy,
    rasterize_passable_openings,
)
from isaac_bench.mapping.coordinate_transform import MapInfo


def test_closed_doors_are_occupancy_and_clearance_obstacles():
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
    assert occupancy[3, 1]
    assert occupancy[1, 3]
    assert filter_start_clearance_objects([door, chair]) == [door, chair]


def test_bbox_rasterization_does_not_include_next_cell_on_max_boundary():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    box = {
        "category": "chair",
        "bbox_min_world": [3.0, 3.0, 0.0],
        "bbox_max_world": [4.0, 4.0, 1.0],
    }

    occupancy = rasterize_bbox_occupancy([box], info, min_z=0.05, max_z=1.5)

    assert bbox_grid_bounds(box["bbox_min_world"], box["bbox_max_world"], info) == (1, 1, 3, 3)
    assert int(occupancy.sum()) == 1
    assert occupancy[1, 3]
    assert not occupancy[1, 4]
    assert not occupancy[2, 3]


def test_navigable_inflation_uses_robot_radius_only():
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

    assert inflated_robot_only.sum() == inflated_extra.sum()
    assert navigable_robot_only.sum() == navigable_extra.sum()
    assert inflated_robot_only.sum() == 13


def test_closed_door_over_sill_keeps_passage_blocked():
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
    sill = {
        "category": "doorsill",
        "bbox_min_world": [1.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 3.0, 0.01],
    }

    occupancy = rasterize_bbox_occupancy([wall, door, sill], info, min_z=0.05, max_z=1.5)
    opening_mask = rasterize_passable_openings([wall, door, sill], info, min_z=0.0, max_z=1.5)
    carved = carve_passable_openings(occupancy, [wall, door, sill], info, min_z=0.0, max_z=1.5)

    assert occupancy.sum() > 0
    assert opening_mask.sum() == 0
    assert carved.sum() > 0


def test_open_door_near_sill_keeps_passage_open():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    wall = {
        "category": "wall",
        "bbox_min_world": [1.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 4.0, 2.0],
    }
    sill = {
        "category": "doorsill",
        "bbox_min_world": [2.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 4.0, 0.01],
    }
    open_door = {
        "category": "door",
        "bbox_min_world": [3.0, 1.0, 0.0],
        "bbox_max_world": [4.0, 1.3, 2.0],
    }

    occupancy = rasterize_bbox_occupancy([wall, sill, open_door], info, min_z=0.05, max_z=1.5)
    opening_mask = rasterize_passable_openings([wall, sill, open_door], info, min_z=0.0, max_z=1.5)
    carved = carve_passable_openings(occupancy, [wall, sill, open_door], info, min_z=0.0, max_z=1.5)

    assert opening_mask.sum() == 3
    assert carved[:, 2].sum() == 0
    assert not occupancy[3, 3]


def test_door_aliases_are_passable_openings_even_with_instance_suffixes():
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    wall = {
        "category": "wall",
        "bbox_min_world": [1.0, 1.0, 0.0],
        "bbox_max_world": [4.0, 4.0, 2.0],
    }
    doorway = {
        "category": "doorway_0003",
        "bbox_min_world": [2.0, 1.0, 0.0],
        "bbox_max_world": [3.0, 4.0, 2.0],
    }

    occupancy = rasterize_bbox_occupancy([wall, doorway], info, min_z=0.05, max_z=1.5)
    opening_mask = rasterize_passable_openings([wall, doorway], info, min_z=0.05, max_z=1.5)
    carved = carve_passable_openings(occupancy, [wall, doorway], info, min_z=0.05, max_z=1.5)

    assert opening_mask.sum() == 3
    assert occupancy[:, 2].sum() > 0
    assert carved[:, 2].sum() == 0
