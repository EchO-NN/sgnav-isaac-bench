from __future__ import annotations

import numpy as np

from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def _base_debug(shape: tuple[int, int]) -> dict[str, object]:
    debug: dict[str, object] = {
        "backend": "voxel_occupancy_door_wall_v9",
        "voxel_vertical_free_xy": np.ones(shape, dtype=bool),
        "voxel_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_display_wall_xy": np.zeros(shape, dtype=bool),
        "voxel_unknown_xy": np.zeros(shape, dtype=bool),
        "voxel_final_room_label_map": np.zeros(shape, dtype=np.int32),
        "voxel_door_cut_mask": np.zeros(shape, dtype=bool),
        "voxel_step2_extension_separator_map": np.zeros(shape, dtype=bool),
        "voxel_step1_wall_gap_fill_map": np.zeros(shape, dtype=bool),
        "voxel_filtered_wall_line_mask": np.zeros(shape, dtype=bool),
        "voxel_door_seed_mask": np.zeros(shape, dtype=bool),
        "voxel_door_centerline_visual_mask": np.zeros(shape, dtype=bool),
        "voxel_accepted_door_centerline_mask": np.zeros(shape, dtype=bool),
        "voxel_door_visual_only_mask": np.zeros(shape, dtype=bool),
        "voxel_free_wall_conflict_xy": np.zeros(shape, dtype=bool),
        "voxel_wall_projection_rejected_support_map": np.zeros(shape, dtype=bool),
        "voxel_anchor_projected_wall_map": np.zeros(shape, dtype=bool),
    }
    return debug


def _render(debug: dict[str, object], shape: tuple[int, int] = (4, 4)):
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(120, 120), debug_overlay_layers=True)
    viz.set_room_context([], {}, debug)
    image, _layers = viz._render_rose_occupancy_panel(
        occupancy=np.zeros(shape, dtype=bool),
        navigable=np.ones(shape, dtype=bool),
        observed=np.ones(shape, dtype=bool),
        size=(80, 80),
        crop_bounds=(0, shape[0], 0, shape[1]),
    )
    return np.asarray(image)


def _cell_pixel(arr: np.ndarray, row: int, col: int, shape: tuple[int, int] = (4, 4)) -> tuple[int, int, int]:
    label_h = 36
    margin = 8
    available_h = max(1, int(arr.shape[0]) - label_h - margin)
    scale = min((arr.shape[1] - 2 * margin) / shape[1], available_h / shape[0])
    cell = int(scale)
    ox = (arr.shape[1] - int(shape[1] * scale)) // 2
    oy = label_h + max(0, (available_h - int(shape[0] * scale)) // 2)
    y = oy + row * cell + cell // 2
    x = ox + col * cell + cell // 2
    return tuple(int(v) for v in arr[y, x].tolist())


def test_voxel_panel_uses_blue_seed_and_green_only_for_door_cut() -> None:
    shape = (4, 4)
    debug = _base_debug(shape)
    debug["voxel_door_seed_mask"][1, 1] = True
    debug["voxel_door_centerline_visual_mask"][1, 2] = True
    debug["voxel_accepted_door_centerline_mask"][1, 2] = True
    debug["voxel_door_cut_mask"][2, 2] = True

    arr = _render(debug, shape)

    assert _cell_pixel(arr, 1, 1, shape) == (0, 80, 255)
    assert _cell_pixel(arr, 1, 2, shape) != (0, 255, 70)
    assert _cell_pixel(arr, 2, 2, shape) == (80, 255, 80)


def test_voxel_panel_draws_visual_line_without_faking_bright_cut() -> None:
    shape = (4, 4)
    debug = _base_debug(shape)
    debug["voxel_door_centerline_visual_mask"][1, 2] = True

    arr = _render(debug, shape)

    assert _cell_pixel(arr, 1, 2, shape) != (0, 255, 70)
    assert _cell_pixel(arr, 1, 2, shape) != (80, 255, 80)


def test_voxel_panel_uses_purple_for_accepted_step2() -> None:
    shape = (4, 4)
    debug = _base_debug(shape)
    debug["voxel_step2_extension_separator_map"][2, 1] = True

    arr = _render(debug, shape)

    assert _cell_pixel(arr, 2, 1, shape) == (220, 60, 255)


def test_voxel_panel_hides_orange_diagnostics_by_default_and_shows_when_enabled() -> None:
    shape = (4, 4)
    debug = _base_debug(shape)
    debug["voxel_free_wall_conflict_xy"][1, 1] = True
    debug["voxel_wall_projection_rejected_support_map"][1, 2] = True

    arr = _render(debug, shape)
    assert _cell_pixel(arr, 1, 1, shape) != (255, 126, 45)
    assert _cell_pixel(arr, 1, 2, shape) != (255, 142, 45)

    debug["voxel_show_wall_diagnostics"] = True
    debug["voxel_show_wall_support_rejected_unknown"] = True
    arr = _render(debug, shape)
    assert _cell_pixel(arr, 1, 1, shape) == (255, 126, 45)
    assert _cell_pixel(arr, 1, 2, shape) == (255, 142, 45)
