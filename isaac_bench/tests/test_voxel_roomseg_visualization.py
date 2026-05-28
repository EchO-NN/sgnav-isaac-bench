from __future__ import annotations

import numpy as np

from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def test_voxel_roomseg_panel_uses_voxel_layers_and_records_overlay_metadata() -> None:
    shape = (12, 12)
    debug = {
        "backend": "voxel_occupancy_door_wall_v9",
        "voxel_vertical_free_xy": np.zeros(shape, dtype=bool),
        "voxel_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_display_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_raw_occupied_wall_support_xy": np.zeros(shape, dtype=bool),
        "voxel_strict_raw_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_wall_suppressed_by_free_xy": np.zeros(shape, dtype=bool),
        "voxel_ratio_wall_debug_xy": np.zeros(shape, dtype=bool),
        "voxel_projected_wall_map": np.zeros(shape, dtype=bool),
        "voxel_anchor_projected_wall_map": np.zeros(shape, dtype=bool),
        "voxel_step1_completed_wall_map": np.zeros(shape, dtype=bool),
        "voxel_unknown_xy": np.ones(shape, dtype=bool),
        "voxel_final_room_label_map": np.zeros(shape, dtype=np.int32),
        "voxel_door_centerline_visual_mask": np.zeros(shape, dtype=bool),
        "voxel_door_cut_mask": np.zeros(shape, dtype=bool),
        "voxel_step2_extension_separator_map": np.zeros(shape, dtype=bool),
        "voxel_step1_wall_gap_fill_map": np.zeros(shape, dtype=bool),
        "voxel_filtered_wall_line_mask": np.zeros(shape, dtype=bool),
        "voxel_door_seed_mask": np.zeros(shape, dtype=bool),
        "voxel_active_observed_xy": np.ones(shape, dtype=bool),
        "voxel_vertical_observed_xy": np.ones(shape, dtype=bool),
        "voxel_ceiling_height_m": 2.8,
        "voxel_active_z_max_m": 2.52,
    }
    debug["voxel_vertical_free_xy"][2:10, 2:10] = True
    debug["voxel_unknown_xy"][2:10, 2:10] = False
    debug["voxel_wall_xy"][2:10, 6] = True
    debug["voxel_display_wall_xy"][2:10, 6] = True
    debug["voxel_raw_occupied_wall_support_xy"][2:10, 6] = True
    debug["voxel_strict_raw_wall_xy"][2:10, 6] = True
    debug["voxel_projected_wall_map"][2:10, 6] = True
    debug["voxel_anchor_projected_wall_map"][6, 3:9] = True
    debug["voxel_step1_completed_wall_map"][2:10, 6] = True
    debug["voxel_wall_suppressed_by_free_xy"][4, 4] = True
    debug["voxel_ratio_wall_debug_xy"][5, 5] = True
    debug["voxel_door_centerline_visual_mask"][6, 3:9] = True
    debug["voxel_accepted_door_centerline_mask"] = np.zeros(shape, dtype=bool)
    debug["voxel_accepted_door_centerline_mask"][6, 3:9] = True
    debug["voxel_door_cut_mask"][6, 4:8] = True
    debug["voxel_step2_extension_separator_map"][3:7, 8] = True

    viz = SGNavPopupVisualizer(enabled=False, panel_size=(320, 240), debug_overlay_layers=True)
    viz.set_room_context([], {}, debug)
    image, layers = viz._render_rose_occupancy_panel(
        occupancy=np.zeros(shape, dtype=bool),
        navigable=np.ones(shape, dtype=bool),
        observed=np.ones(shape, dtype=bool),
        size=(160, 120),
        crop_bounds=(0, shape[0], 0, shape[1]),
    )

    assert image.size == (160, 120)
    names = {layer["name"] for layer in layers}
    assert "voxel_vertical_panel" in names
    assert "voxel_raw_occupied_wall_support" in names
    assert "voxel_strict_raw_wall" in names
    assert "voxel_wall_projected" in names
    assert "voxel_step1_completed_wall" in names
    assert "voxel_door_centerline" in names
    assert "voxel_door_cut" in names
    assert "voxel_step2_extension" in names
    panel = next(layer for layer in layers if layer["name"] == "voxel_vertical_panel")
    assert panel["missing_keys"] == []


def test_voxel_roomseg_panel_reports_missing_key_without_height_profile_fallback() -> None:
    shape = (8, 8)
    debug = {
        "backend": "voxel_occupancy_door_wall_v9",
        "height_profile_vertical_free_xy": np.ones(shape, dtype=bool),
        "height_profile_wall_xy": np.zeros(shape, dtype=bool),
        "height_profile_unknown_xy": np.zeros(shape, dtype=bool),
        "height_profile_final_room_label_map": np.zeros(shape, dtype=np.int32),
    }
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(320, 240), debug_overlay_layers=True)
    viz.set_room_context([], {}, debug)
    _image, layers = viz._render_rose_occupancy_panel(
        occupancy=np.zeros(shape, dtype=bool),
        navigable=np.ones(shape, dtype=bool),
        observed=np.ones(shape, dtype=bool),
        size=(160, 120),
        crop_bounds=(0, shape[0], 0, shape[1]),
    )

    panel = next(layer for layer in layers if layer["name"] == "voxel_vertical_panel")
    assert "voxel_vertical_free_xy" in panel["missing_keys"]


def test_voxel_frontier_source_uses_voxel_panel_even_without_backend_key() -> None:
    shape = (8, 8)
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(320, 240), debug_overlay_layers=True)
    viz.set_room_context([], {}, {"frontier_source": "voxel_vertical_free"})
    _image, layers = viz._render_rose_occupancy_panel(
        occupancy=np.zeros(shape, dtype=bool),
        navigable=np.ones(shape, dtype=bool),
        observed=np.ones(shape, dtype=bool),
        size=(160, 120),
        crop_bounds=(0, shape[0], 0, shape[1]),
    )

    names = {layer["name"] for layer in layers}
    assert "voxel_vertical_panel" in names
    assert "rose_occupancy_map" not in names
