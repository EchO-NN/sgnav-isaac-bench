import json

import numpy as np

from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.graph.room_semantics import RoomSemanticLabel
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.mapping.room_segmentation import RoomMask
from isaac_bench.perception.detection_types import Detection3D
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def _scene():
    occupancy = np.zeros((20, 30), dtype=bool)
    navigable = np.ones_like(occupancy, dtype=bool)
    observed = np.ones_like(occupancy, dtype=bool)
    memory = ObjectMemory()
    memory.update([Detection3D("table", "table", 0.9, (0.0, 0.0, 0.5), (0, 0, 10, 10))], step_id=1)
    memory.nodes[0].center_grid = (11, 14)
    frontiers = [FrontierCluster((8, 18), (0.0, 0.0), [(8, 18), (8, 19), (9, 19)], 3, 3.0)]
    room_mask = np.zeros_like(occupancy, dtype=bool)
    room_mask[5:16, 7:23] = True
    room = RoomMask(
        room_id="room_0001",
        mask=room_mask,
        centroid_xy=(0.0, 0.0),
        area_m2=12.0,
        boundary_unknown_fraction=0.1,
        doorway_edges=[],
        confidence=0.87,
        observed_free_cells=int(np.count_nonzero(room_mask)),
        mask_confidence=0.87,
        metadata={"centroid_grid": [10, 15]},
    )
    label = RoomSemanticLabel(
        room_id="room_0001",
        category="living_room",
        confidence=0.8,
        supporting_objects=["table"],
        conflicting_evidence=[],
        rationale="synthetic room label",
        backend="vlm",
        vlm_self_confidence=0.8,
        label_reliability=0.74,
        reliability_factors={"synthetic": True},
    )
    return occupancy, navigable, observed, memory, frontiers, [room], {"room_0001": label}


def _update(viz, tmp_path=None, room_segmentation_debug=None):
    occupancy, navigable, observed, memory, frontiers, room_masks, room_labels = _scene()
    viz.set_room_context(room_masks, room_labels, room_segmentation_debug)
    decision = NavigationDecision(
        mode="frontier",
        target_cells=[(8, 18)],
        stop=False,
        selected_candidate=None,
        frontier_decision=None,
        reason="selected_frontier",
    )
    return viz.update(
        step=0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        detections_2d=[],
        occupancy=occupancy,
        navigable=navigable,
        observed=observed,
        goal_cells=[(11, 14)],
        current_grid=(10, 12),
        pose=(0.0, 0.0, 0.0, 0.0),
        frontiers=frontiers,
        nav_decision=decision,
        current_path=[(10, 12), (11, 13)],
        full_path=[(10, 12), (11, 13)],
        object_memory=memory,
        goal_category="table",
        distance_to_goal=2.0,
        path_length=1.0,
        scenegraph_backend="fallback",
    )


def test_gt_goal_cells_not_drawn_by_default_and_sidecar_counts(tmp_path):
    viz = SGNavPopupVisualizer(enabled=False, save_dir=str(tmp_path), panel_size=(640, 360))
    _update(viz)

    sidecar = tmp_path / "sgnav_step_000000.layers.json"
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    layers = {layer["name"]: layer for layer in meta["layers"]}
    assert layers["gt_goal_cells"]["enabled"] is False
    assert layers["gt_goal_cells"]["primitive_count"] == 0
    assert layers["room_masks"]["enabled"] is True
    assert layers["room_masks"]["primitive_count"] > 0
    assert layers["room_boundaries"]["primitive_count"] > 0
    assert layers["room_labels"]["primitive_count"] == 1
    assert layers["accepted_candidate"]["primitive_count"] == 0


def test_room_masks_are_actually_rendered_into_map_panel():
    enabled_viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    disabled_viz = SGNavPopupVisualizer(
        enabled=False,
        show_room_masks=False,
        show_room_labels=False,
        panel_size=(640, 360),
    )

    with_rooms = _update(enabled_viz)
    without_rooms = _update(disabled_viz)
    enabled_layers = {layer["name"]: layer for layer in enabled_viz.overlay_layer_metadata()["layers"]}
    disabled_layers = {layer["name"]: layer for layer in disabled_viz.overlay_layer_metadata()["layers"]}

    assert enabled_layers["room_masks"]["primitive_count"] > 0
    assert enabled_layers["room_boundaries"]["primitive_count"] > 0
    assert enabled_layers["room_labels"]["primitive_count"] == 1
    assert disabled_layers["room_masks"]["enabled"] is False
    assert disabled_layers["room_masks"]["primitive_count"] == 0
    assert disabled_layers["room_labels"]["enabled"] is False
    assert disabled_layers["room_labels"]["primitive_count"] == 0
    assert np.count_nonzero(with_rooms != without_rooms) > 0


def test_adjacent_room_masks_draw_explicit_boundary():
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    base = np.zeros((8, 10, 3), dtype=np.uint8)
    left = np.zeros((8, 10), dtype=bool)
    right = np.zeros((8, 10), dtype=bool)
    left[2:6, 2:5] = True
    right[2:6, 5:8] = True
    rooms = [
        RoomMask(
            room_id="left",
            mask=left,
            centroid_xy=(0.0, 0.0),
            area_m2=1.0,
            boundary_unknown_fraction=0.0,
            doorway_edges=[],
            confidence=1.0,
            observed_free_cells=int(np.count_nonzero(left)),
            mask_confidence=1.0,
        ),
        RoomMask(
            room_id="right",
            mask=right,
            centroid_xy=(0.0, 0.0),
            area_m2=1.0,
            boundary_unknown_fraction=0.0,
            doorway_edges=[],
            confidence=1.0,
            observed_free_cells=int(np.count_nonzero(right)),
            mask_confidence=1.0,
        ),
    ]

    rendered, _mask_cells, boundary_cells = viz._apply_room_mask_overlay(base, rooms)

    assert boundary_cells > 0
    assert np.any(np.all(rendered[:, 4:6] == np.asarray((245, 250, 255), dtype=np.uint8), axis=-1))


def test_disabling_frontier_member_cells_removes_raw_frontier_primitives():
    viz = SGNavPopupVisualizer(enabled=False, show_frontier_member_cells=False, panel_size=(640, 360))
    _update(viz)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["frontier_member_cells"]["enabled"] is False
    assert layers["frontier_member_cells"]["primitive_count"] == 0
    assert layers["frontier_centers"]["primitive_count"] == 1


def test_disabling_object_nodes_removes_object_dot_primitives():
    viz = SGNavPopupVisualizer(enabled=False, show_object_nodes=False, panel_size=(640, 360))
    _update(viz)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["object_nodes"]["enabled"] is False
    assert layers["object_nodes"]["primitive_count"] == 0


def test_corridor_merge_debug_draws_bright_red_dashed_markers():
    debug = {
        "corridor_merge_report": {
            "merge_events": [
                {
                    "reason": "strict_parallel_door_neck_edge_merge",
                    "shared_edge_p0_rc": [8, 10],
                    "shared_edge_p1_rc": [8, 20],
                }
            ],
            "sliver_merge_events": [
                {
                    "reason": "merge_post_corridor_small_region_to_larger_neighbor",
                    "source_centroid_rc": [12, 16],
                }
            ],
        }
    }
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel = _update(viz, room_segmentation_debug=debug)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["corridor_merge_edges"]["primitive_count"] == 1
    assert layers["corridor_merge_edges"]["color"] == [255, 24, 24]
    assert layers["post_corridor_small_region_merges"]["primitive_count"] == 1
    assert layers["post_corridor_small_region_merges"]["color"] == [255, 190, 24]
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 24, 24], dtype=np.uint8), axis=-1))) > 0
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 190, 24], dtype=np.uint8), axis=-1))) > 0


def test_pre_extension_door_debug_layers_are_reported_and_drawn():
    detected = np.zeros((20, 30), dtype=bool)
    detected[8, 14:18] = True
    cut = np.zeros_like(detected)
    cut[9, 15:18] = True
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[5:12, 8:15] = 1
    labels[5:12, 15:22] = 2
    debug = {
        "pre_extension_door_detected_map": detected,
        "pre_extension_door_cut_mask": cut,
        "pre_extension_room_label_map": labels,
    }

    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel = _update(viz, room_segmentation_debug=debug)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["pre_extension_doors"]["primitive_count"] == int(np.count_nonzero(detected))
    assert layers["pre_extension_door_cuts"]["primitive_count"] == int(np.count_nonzero(cut))
    assert layers["pre_extension_room_labels"]["primitive_count"] == 2
    assert layers["pre_extension_room_labels"]["boundary_cell_count"] > 0
    assert int(np.count_nonzero(np.all(panel == np.asarray([0, 210, 255], dtype=np.uint8), axis=-1))) > 0
    assert layers["door_completion_boundaries"]["primitive_count"] == int(np.count_nonzero(cut))
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 120, 40], dtype=np.uint8), axis=-1))) > 0


def test_roomseg_sanitizer_and_partial_door_layers_are_reported():
    clipped = np.zeros((20, 30), dtype=bool)
    clipped[4, 4:7] = True
    conflict = np.zeros_like(clipped)
    conflict[5, 6:9] = True
    sanitized_free = np.zeros_like(clipped)
    sanitized_free[6:10, 8:14] = True
    sanitized_wall = np.zeros_like(clipped)
    sanitized_wall[6:10, 15] = True
    seed = np.zeros_like(clipped)
    seed[8, 16:18] = True
    line = np.zeros_like(clipped)
    line[8, 12:21] = True
    cut = np.zeros_like(clipped)
    cut[8, 13:16] = True
    rejected = np.zeros_like(clipped)
    rejected[9, 13:18] = True
    wall_extension = np.zeros_like(clipped)
    wall_extension[7, 10:14] = True
    door_completion = np.zeros_like(clipped)
    door_completion[8, 13:16] = True
    debug = {
        "vertical_free_clipped_outside_navigation_map": clipped,
        "free_wall_conflict_map_before_sanitize": conflict,
        "roomseg_sanitized_free": sanitized_free,
        "roomseg_sanitized_wall": sanitized_wall,
        "wall_extension_boundary_mask": wall_extension,
        "door_completion_boundary_mask": door_completion,
        "partial_door_seed_mask": seed,
        "partial_door_line_mask": line,
        "partial_door_extension_cut_mask": cut,
        "rejected_door_extension_mask": rejected,
        "partial_door_line_reject_reason_counts": {"extension_endpoint_is_other_door": 1},
        "segmentation_degenerate_one_room": True,
    }

    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel = _update(viz, room_segmentation_debug=debug)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["roomseg_vertical_free_outside_navigation"]["primitive_count"] == int(np.count_nonzero(clipped))
    assert layers["roomseg_free_wall_conflict"]["primitive_count"] == int(np.count_nonzero(conflict))
    assert layers["roomseg_sanitized_free"]["primitive_count"] == int(np.count_nonzero(sanitized_free))
    assert layers["roomseg_sanitized_wall"]["primitive_count"] == int(np.count_nonzero(sanitized_wall))
    assert layers["wall_extension_boundaries"]["primitive_count"] == int(np.count_nonzero(wall_extension))
    assert layers["door_completion_boundaries"]["primitive_count"] == int(np.count_nonzero(door_completion))
    assert layers["partial_door_seed_points"]["primitive_count"] == int(np.count_nonzero(seed))
    assert layers["accepted_partial_door_extension_lines"]["primitive_count"] == int(np.count_nonzero(line))
    assert layers["partial_door_extension_cuts"]["primitive_count"] == int(np.count_nonzero(cut))
    assert layers["rejected_partial_door_extension_lines"]["primitive_count"] == int(np.count_nonzero(rejected))
    assert layers["segmentation_degenerate_warning"]["primitive_count"] == 1
    assert layers["wall_extension_boundaries"]["color"] == [80, 170, 255]
    assert layers["door_completion_boundaries"]["color"] == [255, 120, 40]
    assert layers["wall_extension_boundaries"]["color"] != layers["door_completion_boundaries"]["color"]
    assert int(np.count_nonzero(np.all(panel == np.asarray([80, 170, 255], dtype=np.uint8), axis=-1))) > 0
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 120, 40], dtype=np.uint8), axis=-1))) > 0
    assert int(np.count_nonzero(np.all(panel == np.asarray([60, 250, 180], dtype=np.uint8), axis=-1))) > 0
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 60, 180], dtype=np.uint8), axis=-1))) > 0


def test_roomseg_wall_lines_and_extensions_draw_red_even_without_room_split():
    debug = {
        "filtered_wall_lines_report": {
            "filtered_wall_lines": [
                {
                    "line_id": 1,
                    "p0_rc": [6, 8],
                    "p1_rc": [6, 20],
                    "length_m": 1.2,
                }
            ]
        },
        "line_extension_report": {
            "pass1": {
                "extensions": [
                    {
                        "extension_id": 1,
                        "p_start_rc": [6, 20],
                        "p_hit_rc": [10, 24],
                        "reject_reason": "reject_extension_no_wall_or_virtual_door_hit",
                    }
                ]
            },
            "pass2": {"extensions": []},
        },
    }
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel = _update(viz, room_segmentation_debug=debug)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["roomseg_wall_lines_red"]["primitive_count"] == 1
    assert layers["roomseg_wall_lines_red"]["color"] == [255, 35, 35]
    assert layers["roomseg_wall_extensions_red_dashed"]["primitive_count"] >= 2
    assert layers["roomseg_wall_extensions_red_dashed"]["color"] == [255, 0, 0]
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 0, 0], dtype=np.uint8), axis=-1))) > 0


def test_roomseg_wall_endpoint_probe_draws_when_extension_report_is_missing():
    debug = {
        "resolution_m": 0.05,
        "filtered_wall_lines_report": {
            "filtered_wall_lines": [
                {
                    "line_id": 1,
                    "p0_rc": [8, 8],
                    "p1_rc": [8, 20],
                    "length_m": 0.6,
                }
            ]
        },
        "line_extension_report": {"pass1": {"extensions": []}, "pass2": {"extensions": []}},
    }
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel = _update(viz, room_segmentation_debug=debug)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert layers["roomseg_wall_lines_red"]["primitive_count"] == 1
    assert layers["roomseg_wall_extensions_red_dashed"]["primitive_count"] == 2
    assert int(np.count_nonzero(np.all(panel == np.asarray([255, 0, 0], dtype=np.uint8), axis=-1))) > 0


def test_rose_input_occupancy_map_is_rendered_below_runtime_map(tmp_path):
    occupancy, _, _, _, _, _, _ = _scene()
    structural = np.zeros_like(occupancy, dtype=bool)
    structural[4:16, 8] = True
    vertical_carved = np.zeros_like(occupancy, dtype=bool)
    vertical_carved[10:13, 14:17] = True
    wall_conf = np.zeros_like(occupancy, dtype=np.float32)
    wall_conf[structural] = 0.92
    rejected = np.zeros_like(occupancy, dtype=bool)
    rejected[12:14, 18:21] = True
    debug = {
        "algorithm": "upstream_rose2_vertical_or_free",
        "structural_wall_mask": structural,
        "structural_component_rejected_mask": rejected,
        "vertical_carved_map": vertical_carved,
        "wall_confidence_map": wall_conf,
        "wall_confidence_threshold": 0.55,
        "repaired_window_gaps": [{"axis": "vertical", "index": 8, "start": 6, "end": 7, "kind": "window"}],
        "verified_doorway_gaps": [{"axis": "vertical", "index": 8, "start": 12, "end": 13, "kind": "doorway"}],
    }
    viz = SGNavPopupVisualizer(enabled=False, save_dir=str(tmp_path), panel_size=(640, 360), save_every_steps=1)
    panel = _update(viz, room_segmentation_debug=debug)
    layers = {layer["name"]: layer for layer in viz.overlay_layer_metadata()["layers"]}

    assert panel.shape == (360, 640, 3)
    assert layers["rose_occupancy_map"]["primitive_count"] == int(np.count_nonzero(structural))
    assert layers["rose_wall_confidence_map"]["primitive_count"] == int(np.count_nonzero(wall_conf >= 0.55))
    assert layers["rose_vertical_carved_map"]["primitive_count"] == int(np.count_nonzero(vertical_carved))
    assert layers["rose_structural_rejected_clutter"]["primitive_count"] == int(np.count_nonzero(rejected))
    assert layers["rose_repaired_window_gaps"]["primitive_count"] == 1
    assert layers["rose_verified_doorway_gaps"]["primitive_count"] == 1

    sidecar = tmp_path / "sgnav_step_000000.layers.json"
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    saved_layers = {layer["name"]: layer for layer in saved["layers"]}
    assert saved_layers["rose_occupancy_map"]["has_rose_input"] is True


def test_rose_input_panel_waiting_state_is_not_black():
    occupancy = np.zeros((20, 30), dtype=bool)
    occupancy[4:16, 8] = True
    navigable = np.zeros_like(occupancy, dtype=bool)
    navigable[6:14, 10:22] = True
    observed = np.zeros_like(occupancy, dtype=bool)
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel, layers = viz._render_rose_occupancy_panel(
        occupancy=occupancy,
        navigable=navigable,
        observed=observed,
        size=(360, 110),
        crop_bounds=(0, 20, 0, 30),
    )
    layer_map = {layer["name"]: layer for layer in layers}
    arr = np.asarray(panel)

    assert layer_map["rose_occupancy_map"]["has_rose_input"] is False
    assert layer_map["rose_occupancy_map"]["current_occupancy_underlay_cells"] == int(np.count_nonzero(occupancy))
    assert int(arr.max()) > 180
    assert len(np.unique(arr.reshape(-1, 3), axis=0)) > 3
