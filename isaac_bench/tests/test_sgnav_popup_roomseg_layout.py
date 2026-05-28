import numpy as np

from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def _render_with_height_debug():
    shape = (24, 32)
    occupancy = np.zeros(shape, dtype=bool)
    navigable = np.zeros(shape, dtype=bool)
    navigable[4:20, 5:26] = True
    observed = navigable.copy()
    vfree = navigable.copy()
    wall = np.zeros(shape, dtype=bool)
    wall[10, 8:24] = True
    conflict = np.zeros(shape, dtype=bool)
    conflict[9, 8:10] = True
    unknown = ~vfree
    labels = np.zeros(shape, dtype=np.int32)
    labels[4:10, 5:26] = 1
    labels[11:20, 5:26] = 2
    door = np.zeros(shape, dtype=bool)
    door[10, 15:18] = True
    step2 = np.zeros(shape, dtype=bool)
    step2[14:18, 20] = True
    step1 = np.zeros(shape, dtype=bool)
    step1[10, 12:14] = True
    debug = {
        "height_profile_vertical_free_xy": vfree,
        "height_profile_vertical_observed_xy": observed,
        "height_profile_wall_xy": wall,
        "height_profile_free_wall_conflict_xy": conflict,
        "height_profile_unknown_xy": unknown,
        "height_profile_filtered_wall_line_mask": wall,
        "height_profile_door_seed_mask": door,
        "height_profile_door_cut_mask": door,
        "height_profile_accepted_door_centerline_mask": door,
        "height_profile_step1_wall_gap_fill_map": step1,
        "height_profile_step2_extension_separator_map": step2,
        "height_profile_final_room_label_map": labels,
        "height_profile_boundary_source_map": wall.astype(np.uint8),
        "ceiling_height_estimate_m": 2.60,
        "height_profile_active_z_max_m": 2.34,
    }
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360), show_rose_occupancy_map=True)
    viz.set_room_context([], {}, debug)
    panel = viz.render(
        step=1,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        detections_2d=[],
        occupancy=occupancy,
        navigable=navigable,
        observed=observed,
        goal_cells=[],
        current_grid=(12, 14),
        pose=(0.0, 0.0, 0.0, 0.0),
        frontiers=[],
        nav_decision=None,
        current_path=[],
        full_path=[],
        object_memory=ObjectMemory(),
        goal_category="chair",
        distance_to_goal=1.0,
        path_length=0.0,
        scenegraph_backend="fallback",
    )
    return viz, panel


def test_popup_renders_rgb_left_nav_top_right_vertical_bottom_right():
    viz, panel = _render_with_height_debug()
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert panel.shape == (360, 640, 3)
    assert "navigation_free" in layers
    assert "vertical_free" in layers
    assert "height_profile_wall_red" in layers
    assert "height_profile_conflict" in layers
    assert layers["vertical_free"]["primitive_count"] > 0


def test_nav_and_vertical_panels_have_equal_size_and_shared_crop():
    viz, _panel = _render_with_height_debug()
    right_h = viz.panel_size[1]
    divider_h = 2
    top_h = (right_h - divider_h) // 2
    bottom_h = right_h - divider_h - top_h

    assert top_h == bottom_h
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}
    assert layers["navigation_free"]["primitive_count"] == layers["vertical_free"]["primitive_count"]


def test_door_and_step2_lines_drawn_with_distinct_overlay_names():
    viz, _panel = _render_with_height_debug()
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["height_profile_door_centerline"]["primitive_count"] > 0
    assert layers["height_profile_step2_extension"]["primitive_count"] > 0
    assert layers["height_profile_door_centerline"]["color"] != layers["height_profile_step2_extension"]["color"]


def test_door_line_not_overpainted_by_wall_layer():
    _viz, panel = _render_with_height_debug()
    magenta = np.asarray([255, 95, 220], dtype=np.uint8)
    red = np.asarray([238, 42, 42], dtype=np.uint8)
    blue = np.asarray([40, 175, 255], dtype=np.uint8)

    assert int(np.count_nonzero(np.all(panel == magenta, axis=-1))) > 0
    assert int(np.count_nonzero(np.all(panel == red, axis=-1))) > 0
    assert int(np.count_nonzero(np.all(panel == blue, axis=-1))) > 0


def test_height_profile_overlay_metadata_has_strict_counts():
    viz, _panel = _render_with_height_debug()
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}
    panel_layer = layers["height_profile_vertical_panel"]

    assert panel_layer["strict_wall_cells"] > 0
    assert panel_layer["conflict_cells"] > 0
    assert panel_layer["door_cut_cells"] > 0
    assert panel_layer["step2_separator_cells"] > 0
    assert panel_layer["height_profile_active_z_max_m"] == "2.34"
