from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.debug_viz_v6_2 import V6_2_COLORS, V6_2_LEGEND_ITEMS, render_v6_2_debug_panel
from isaac_bench.mapping.online_roomseg.door_wall_repair_v6_2 import repair_door_wall_v6_2
from isaac_bench.mapping.online_roomseg.height_profile_v6_2 import FloorCeilingEstimate, HeightProfileState
from isaac_bench.mapping.online_roomseg.online_room_segmenter import HEIGHT_PROFILE_DOOR_WALL_V6_2_BACKEND, OnlineRoseStyleConfig, run_online_rose_style_roomseg
from isaac_bench.mapping.online_roomseg.profile_classifier_v6_2 import classify_height_profile_v6_2
from isaac_bench.mapping.online_roomseg.room_labeler_v6_2 import label_rooms_v6_2
from isaac_bench.mapping.online_roomseg.debug_viz_v6_2 import build_v6_2_debug_layers


def test_v6_2_pipeline_keeps_wall_door_layers_and_labels_rooms():
    wall_cols = [(10, c) for c in range(30)]
    door_cols = [(15, c) for c in range(30)]
    profile = _profile_with_wall_and_door((30, 30), wall_cols=wall_cols, door_cols=door_cols)

    classification = classify_height_profile_v6_2(profile)
    repair = repair_door_wall_v6_2(
        wall_mask_raw=classification.wall_mask_raw,
        door_mask_raw=classification.door_mask_raw,
        unknown_mask=classification.unknown_mask,
        resolution_m=0.10,
        config={"repair": {"remove_island_max_area_m2": 0.0}},
    )
    free_for_navigation = classification.free_mask_raw | repair.door_mask
    labels = label_rooms_v6_2(
        free_for_navigation=free_for_navigation,
        wall_mask=repair.wall_mask,
        door_mask=repair.door_mask,
        corridor_separator_mask=np.zeros((30, 30), dtype=bool),
        unknown_mask=classification.unknown_mask,
        resolution_m=0.10,
        config={"labels": {"min_room_area_m2": 0.25}},
        door_confidence_map=np.minimum(classification.best_door_lower_free_ratio, classification.best_door_upper_occupied_ratio),
    )

    room_ids = [int(v) for v in np.unique(labels.room_label_map) if int(v) > 0]
    assert len(room_ids) >= 3
    assert int(np.count_nonzero(repair.wall_mask)) > 0
    assert int(np.count_nonzero(repair.door_mask)) > 0
    assert np.all(labels.room_label_map[repair.door_mask] == 0)
    assert np.all(labels.room_label_map[classification.unknown_mask] == -1)

    layers = build_v6_2_debug_layers(
        wall_endpoint_ratio=classification.wall_endpoint_ratio,
        best_door_lower_free_ratio=classification.best_door_lower_free_ratio,
        best_door_upper_occ_ratio=classification.best_door_upper_occupied_ratio,
        best_door_transition_z=classification.best_door_transition_z,
        wall_mask_raw=classification.wall_mask_raw,
        door_mask_raw=classification.door_mask_raw,
        wall_mask_repaired=repair.wall_mask,
        door_mask_repaired=repair.door_mask,
        wall_line_support_mask=repair.wall_line_support_mask,
        corridor_l_corner_candidates=np.zeros((30, 30), dtype=bool),
        accepted_corridor_separator_mask=np.zeros((30, 30), dtype=bool),
        room_cut_mask=labels.room_cut_mask,
        final_room_labels=labels.room_label_map,
        free_mask=free_for_navigation,
        unknown_mask=classification.unknown_mask,
    )
    panel = render_v6_2_debug_panel(layers)
    assert panel.size[0] > 0 and panel.size[1] > 0
    assert int(np.count_nonzero(layers["wall_mask_repaired"])) > 0
    assert int(np.count_nonzero(layers["door_mask_repaired"])) > 0
    assert V6_2_COLORS["wall"] != V6_2_COLORS["door"]
    assert "wall" in V6_2_LEGEND_ITEMS
    assert "door/header" in V6_2_LEGEND_ITEMS


def test_online_roomseg_v6_2_backend_smoke():
    free = np.ones((20, 20), dtype=bool)
    wall = np.zeros_like(free)
    wall[10, :] = True
    free[wall] = False
    unknown = np.zeros_like(free)

    result = run_online_rose_style_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=unknown,
        vertical_profile=None,
        config=OnlineRoseStyleConfig(resolution_m=0.10, min_observed_free_cells=1),
        step=7,
    )

    assert result.debug["algorithm"] == HEIGHT_PROFILE_DOOR_WALL_V6_2_BACKEND
    assert result.debug["navigation_obstacle_written"] is False
    assert result.debug["door_not_planner_obstacle"] is True
    assert int(np.count_nonzero(result.layers["wall_mask_repaired"])) > 0
    assert len([v for v in np.unique(result.room_label_map) if int(v) > 0]) == 2


def _profile_with_wall_and_door(
    shape_2d: tuple[int, int],
    *,
    wall_cols: list[tuple[int, int]],
    door_cols: list[tuple[int, int]],
) -> HeightProfileState:
    k = 60
    z_edges = np.linspace(0.0, 3.0, k + 1, dtype=np.float32)
    z_centers = (z_edges[:-1] + z_edges[1:]) * 0.5
    shape = (k, *shape_2d)
    free_supported = np.ones(shape, dtype=bool)
    endpoint_supported = np.zeros(shape, dtype=bool)
    occupied_supported = np.zeros(shape, dtype=bool)
    unknown_supported = np.zeros(shape, dtype=bool)

    for r, c in wall_cols:
        free_supported[:, r, c] = False
        endpoint_supported[:, r, c] = True
        occupied_supported[:, r, c] = True
    for r, c in door_cols:
        free_supported[:, r, c] = False
        free_supported[:39, r, c] = True
        endpoint_supported[40:59, r, c] = True
        occupied_supported[40:59, r, c] = True

    unknown_supported[:, :3, :] = True
    free_supported[:, :3, :] = False
    fc = FloorCeilingEstimate(0.0, 3.0, 3.0, 0.0, 3.0, 1.0)
    zeros = np.zeros(shape, dtype=np.uint16)
    return HeightProfileState(
        z_edges_m=z_edges,
        z_centers_m=z_centers,
        free_count=zeros.copy(),
        endpoint_count=zeros.copy(),
        observed_count=zeros.copy(),
        unknown_count=zeros.copy(),
        free_supported=free_supported,
        endpoint_supported=endpoint_supported,
        occupied_supported=occupied_supported,
        unknown_supported=unknown_supported,
        floor_ceiling=fc,
    )
