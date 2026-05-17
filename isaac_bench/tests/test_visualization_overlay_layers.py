import json

import numpy as np

from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.mapping.frontier import FrontierCluster
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
    return occupancy, navigable, observed, memory, frontiers


def _update(viz, tmp_path=None):
    occupancy, navigable, observed, memory, frontiers = _scene()
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
    assert layers["accepted_candidate"]["primitive_count"] == 0


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
