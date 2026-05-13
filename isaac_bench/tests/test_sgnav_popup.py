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
