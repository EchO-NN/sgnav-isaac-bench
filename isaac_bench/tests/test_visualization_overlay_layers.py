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


def _update(viz, tmp_path=None):
    occupancy, navigable, observed, memory, frontiers, room_masks, room_labels = _scene()
    viz.set_room_context(room_masks, room_labels)
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
