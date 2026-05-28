from __future__ import annotations

import numpy as np

from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def _render_layers(debug: dict[str, object], shape: tuple[int, int] = (5, 5)) -> dict[str, dict]:
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(140, 140), debug_overlay_layers=True)
    viz.set_room_context([], {}, debug)
    _image, layers = viz._render_rose_occupancy_panel(
        occupancy=np.zeros(shape, dtype=bool),
        navigable=np.ones(shape, dtype=bool),
        observed=np.ones(shape, dtype=bool),
        size=(100, 100),
        crop_bounds=(0, shape[0], 0, shape[1]),
    )
    return {str(item["name"]): item for item in layers}


def _base_debug(shape: tuple[int, int]) -> dict[str, object]:
    return {
        "backend": "voxel_occupancy_door_wall_v9",
        "voxel_vertical_free_xy": np.ones(shape, dtype=bool),
        "voxel_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_display_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_unknown_xy": np.zeros(shape, dtype=bool),
        "voxel_final_room_label_map": np.zeros(shape, dtype=np.int32),
        "voxel_door_seed_mask": np.zeros(shape, dtype=bool),
        "voxel_door_centerline_visual_mask": np.zeros(shape, dtype=bool),
        "voxel_accepted_door_centerline_mask": np.zeros(shape, dtype=bool),
        "voxel_door_visual_only_mask": np.zeros(shape, dtype=bool),
        "voxel_door_cut_mask": np.zeros(shape, dtype=bool),
        "voxel_step2_extension_separator_map": np.zeros(shape, dtype=bool),
        "voxel_step1_wall_gap_fill_map": np.zeros(shape, dtype=bool),
        "voxel_filtered_wall_line_mask": np.zeros(shape, dtype=bool),
        "voxel_free_wall_conflict_xy": np.zeros(shape, dtype=bool),
        "voxel_wall_projection_rejected_support_map": np.zeros(shape, dtype=bool),
        "voxel_anchor_projected_wall_map": np.zeros(shape, dtype=bool),
    }


def test_v15_voxel_overlay_colors_for_seed_door_and_step2() -> None:
    shape = (5, 5)
    debug = _base_debug(shape)
    debug["voxel_door_seed_mask"][1, 1] = True
    debug["voxel_accepted_door_centerline_mask"][1, 2] = True
    debug["voxel_door_cut_mask"][1, 2] = True
    debug["voxel_step2_extension_separator_map"][2, 3] = True

    layers = _render_layers(debug, shape)

    assert layers["voxel_door_seed"]["color"] == [0, 80, 255]
    assert layers["voxel_door_centerline"]["color"] == [0, 255, 70]
    assert layers["voxel_step2_extension"]["color"] == [220, 60, 255]


def test_v15_visual_only_door_is_diagnostic_not_fake_cut() -> None:
    shape = (5, 5)
    debug = _base_debug(shape)
    debug["voxel_door_centerline_visual_mask"][1, 2] = True

    layers = _render_layers(debug, shape)

    assert layers["voxel_door_centerline"]["enabled"] is False
    assert layers["voxel_door_cut"]["primitive_count"] == 0

    debug["voxel_show_wall_diagnostics"] = True
    debug["voxel_door_visual_only_mask"][1, 2] = True
    layers = _render_layers(debug, shape)
    assert layers["voxel_door_visual_only"]["enabled"] is True
