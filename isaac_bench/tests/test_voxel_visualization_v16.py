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


def test_v16_voxel_visualization_exposes_door_and_step2_partition_layers() -> None:
    shape = (6, 6)
    debug: dict[str, object] = {
        "backend": "voxel_occupancy_door_wall_v9",
        "voxel_show_wall_diagnostics": True,
        "voxel_show_door_visual_only_candidates": True,
        "voxel_show_step2_source_candidates": True,
        "voxel_vertical_free_xy": np.ones(shape, dtype=bool),
        "voxel_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_display_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_unknown_xy": np.zeros(shape, dtype=bool),
        "voxel_final_room_label_map": np.zeros(shape, dtype=np.int32),
        "voxel_door_seed_mask": np.zeros(shape, dtype=bool),
        "voxel_door_extension_attempt_all_mask": np.zeros(shape, dtype=bool),
        "voxel_door_extension_attempt_rejected_mask": np.zeros(shape, dtype=bool),
        "voxel_door_centerline_visual_mask": np.zeros(shape, dtype=bool),
        "voxel_accepted_door_centerline_mask": np.zeros(shape, dtype=bool),
        "voxel_door_visual_only_mask": np.zeros(shape, dtype=bool),
        "voxel_door_partition_cut_candidate_mask": np.zeros(shape, dtype=bool),
        "voxel_door_partition_cut_accepted_mask": np.zeros(shape, dtype=bool),
        "voxel_door_partition_cut_rejected_mask": np.zeros(shape, dtype=bool),
        "voxel_door_cut_mask": np.zeros(shape, dtype=bool),
        "voxel_step2_extension_hits_all_map": np.zeros(shape, dtype=bool),
        "voxel_step2_separator_candidates_pre_topology_map": np.zeros(shape, dtype=bool),
        "voxel_step2_partition_cut_candidate_map": np.zeros(shape, dtype=bool),
        "voxel_step2_partition_cut_accepted_map": np.zeros(shape, dtype=bool),
        "voxel_step2_extension_separator_map": np.zeros(shape, dtype=bool),
        "voxel_step2_topology_rejected_separator_map": np.zeros(shape, dtype=bool),
        "voxel_step1_wall_gap_fill_map": np.zeros(shape, dtype=bool),
        "voxel_filtered_wall_line_mask": np.zeros(shape, dtype=bool),
        "voxel_free_wall_conflict_xy": np.zeros(shape, dtype=bool),
        "voxel_wall_projection_rejected_support_map": np.zeros(shape, dtype=bool),
        "voxel_anchor_projected_wall_map": np.zeros(shape, dtype=bool),
    }
    debug["voxel_door_seed_mask"][1, 1] = True
    debug["voxel_door_extension_attempt_all_mask"][1, 2] = True
    debug["voxel_door_centerline_visual_mask"][1, 3] = True
    debug["voxel_accepted_door_centerline_mask"][1, 3] = True
    debug["voxel_door_visual_only_mask"][1, 3] = True
    debug["voxel_door_partition_cut_candidate_mask"][2, 3] = True
    debug["voxel_door_partition_cut_rejected_mask"][2, 3] = True
    debug["voxel_door_cut_mask"][3, 3] = True
    debug["voxel_step2_extension_hits_all_map"][4, 2] = True
    debug["voxel_step2_partition_cut_candidate_map"][4, 3] = True
    debug["voxel_step2_partition_cut_accepted_map"][4, 4] = True
    debug["voxel_step2_extension_separator_map"][4, 4] = True

    layers = _render_layers(debug, shape)

    assert layers["voxel_vertical_free"]["color"] == [170, 220, 245]
    assert layers["voxel_door_seed"]["color"] == [0, 80, 255]
    assert layers["voxel_door_extension_attempt"]["color"] == [0, 120, 40]
    assert layers["voxel_door_centerline"]["color"] == [80, 255, 80]
    assert layers["voxel_door_cut"]["color"] == [80, 255, 80]
    assert layers["voxel_door_partition_cut_candidate"]["enabled"] is True
    assert layers["voxel_door_partition_cut_rejected"]["enabled"] is True
    assert layers["voxel_step2_extension_hits"]["color"] == [160, 80, 255]
    assert layers["voxel_step2_partition_cut_candidate"]["enabled"] is True
    assert layers["voxel_step2_extension"]["color"] == [220, 60, 255]
