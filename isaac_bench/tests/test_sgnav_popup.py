import numpy as np

from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.perception.detection_types import Detection2D, Detection3D
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def test_sgnav_popup_render_panel():
    occupancy = np.zeros((20, 30), dtype=bool)
    navigable = np.ones_like(occupancy, dtype=bool)
    observed = np.zeros_like(occupancy, dtype=bool)
    observed[8:14, 10:18] = True
    memory = ObjectMemory()
    memory.update([Detection3D("table", "table", 0.9, (0.0, 0.0, 0.5), (0, 0, 10, 10))], step_id=1)
    memory.nodes[0].center_grid = (11, 14)
    decision = NavigationDecision(
        mode="candidate",
        target_cells=[(11, 16), (12, 16)],
        stop=False,
        selected_candidate=memory.nodes[0],
        frontier_decision=None,
        reason="navigate_to_goal_candidate",
    )
    frontiers = [FrontierCluster((8, 18), (0.0, 0.0), [(8, 18), (8, 19)], 2, 3.0)]
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(640, 360))
    panel = viz.render(
        step=3,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        detections_2d=[Detection2D("table", "table", 0.8, (4, 4, 20, 20), 0)],
        occupancy=occupancy,
        navigable=navigable,
        observed=observed,
        goal_cells=[(11, 14)],
        current_grid=(10, 12),
        pose=(0.0, 0.0, 0.0, 0.0),
        frontiers=frontiers,
        nav_decision=decision,
        current_path=[(10, 12), (11, 13), (11, 14)],
        full_path=[(10, 12), (11, 13), (11, 14)],
        object_memory=memory,
        goal_category="table",
        distance_to_goal=2.0,
        path_length=1.0,
        scenegraph_backend="fallback",
        score_debug={"mode": "fallback"},
    )
    assert panel.shape == (360, 640, 3)
    assert panel.dtype == np.uint8
    assert int(panel.sum()) > 0


def test_sgnav_popup_bbox_colors_goal_red_normal_green():
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(320, 240))
    image = viz._render_rgb(
        np.zeros((80, 100, 3), dtype=np.uint8),
        [
            Detection2D("chair", "chair", 0.8, (20, 20, 40, 40), 0),
            Detection2D("mirror", "mirror", 0.9, (60, 20, 80, 40), 1),
        ],
        (100, 80),
        goal_category="mirror",
        nav_decision=None,
    )
    arr = np.asarray(image)
    assert tuple(arr[40, 30]) == (80, 230, 120)
    assert tuple(arr[40, 70]) == (255, 60, 60)


def test_sgnav_popup_does_not_draw_low_confidence_bbox():
    viz = SGNavPopupVisualizer(enabled=False, panel_size=(320, 240))
    image = viz._render_rgb(
        np.zeros((80, 100, 3), dtype=np.uint8),
        [
            Detection2D("chair", "chair", 0.65, (20, 20, 40, 40), 0),
            Detection2D("mirror", "mirror", 0.66, (60, 20, 80, 40), 1),
        ],
        (100, 80),
        goal_category="mirror",
        nav_decision=None,
    )
    arr = np.asarray(image)

    assert tuple(arr[40, 30]) == (0, 0, 0)
    assert tuple(arr[40, 70]) == (255, 60, 60)


def test_sgnav_popup_map_nodes_only_goal_or_selected_candidate():
    memory = ObjectMemory()
    memory.update(
        [
            Detection3D("chair", "chair", 0.9, (0.0, 0.0, 0.5), (0, 0, 10, 10)),
            Detection3D("mirror", "mirror", 0.9, (1.0, 0.0, 0.5), (0, 0, 10, 10)),
            Detection3D("table", "table", 0.9, (2.0, 0.0, 0.5), (0, 0, 10, 10)),
        ],
        step_id=1,
    )
    for idx, node in enumerate(memory.nodes):
        node.center_grid = (10, 10 + idx)

    decision = NavigationDecision(
        mode="candidate",
        target_cells=[],
        stop=False,
        selected_candidate=memory.nodes[2],
        frontier_decision=None,
        reason="navigate_to_goal_candidate",
    )
    visible = SGNavPopupVisualizer._visible_map_nodes(memory, "mirror", decision)
    assert [node.category for node in visible] == ["mirror", "table"]
