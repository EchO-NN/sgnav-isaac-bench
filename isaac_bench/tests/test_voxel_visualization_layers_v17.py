from __future__ import annotations

import numpy as np

from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def _render_layers(debug: dict[str, object], shape: tuple[int, int] = (6, 6)) -> dict[str, dict]:
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(160, 160), debug_overlay_layers=True)
    viz.set_room_context([], {}, debug)
    _image, layers = viz._render_rose_occupancy_panel(
        occupancy=np.zeros(shape, dtype=bool),
        navigable=np.ones(shape, dtype=bool),
        observed=np.ones(shape, dtype=bool),
        size=(120, 120),
        crop_bounds=(0, shape[0], 0, shape[1]),
    )
    return {str(item["name"]): item for item in layers}


def test_v17_voxel_visualization_exposes_unknown_gate_stable_and_warning_layers() -> None:
    shape = (6, 6)
    zero = np.zeros(shape, dtype=bool)
    debug: dict[str, object] = {
        "backend": "voxel_occupancy_door_wall_v9",
        "voxel_show_wall_diagnostics": True,
        "voxel_show_wall_support_rejected_unknown": True,
        "voxel_show_door_visual_only_candidates": True,
        "voxel_vertical_free_xy": np.ones(shape, dtype=bool),
        "voxel_wall_xy": zero.copy(),
        "voxel_display_wall_xy": zero.copy(),
        "voxel_unknown_xy": zero.copy(),
        "voxel_final_room_label_map": np.zeros(shape, dtype=np.int32),
        "voxel_door_cut_mask": zero.copy(),
        "voxel_step2_extension_separator_map": zero.copy(),
        "voxel_unknown_dominant_xy": zero.copy(),
        "voxel_wall_support_unknown_gated_xy": zero.copy(),
        "voxel_wall_support_rejected_unknown_xy": zero.copy(),
        "voxel_stable_door_cut_mask": zero.copy(),
        "voxel_stable_door_visual_mask": zero.copy(),
        "voxel_door_topology_warning_cut_mask": zero.copy(),
    }
    debug["voxel_unknown_dominant_xy"][1, 1] = True
    debug["voxel_wall_support_unknown_gated_xy"][1, 2] = True
    debug["voxel_wall_support_rejected_unknown_xy"][1, 3] = True
    debug["voxel_stable_door_cut_mask"][3, 3] = True
    debug["voxel_stable_door_visual_mask"][3, 2] = True
    debug["voxel_door_topology_warning_cut_mask"][4, 2] = True
    debug["voxel_door_cut_mask"][3, 3] = True

    layers = _render_layers(debug, shape)

    assert layers["voxel_unknown_dominant"]["enabled"] is True
    assert layers["voxel_wall_support_unknown_gated"]["enabled"] is True
    assert layers["voxel_wall_support_rejected_unknown"]["enabled"] is True
    assert layers["voxel_stable_door_cut"]["enabled"] is True
    assert layers["voxel_stable_door_visual"]["enabled"] is True
    assert layers["voxel_door_topology_warning_cut"]["enabled"] is True
    assert layers["voxel_vertical_panel"]["stable_door_cut_cells"] == 1
