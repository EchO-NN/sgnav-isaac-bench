import json

import numpy as np

from isaac_bench.debug.graph_debug_dump import save_graph_debug_dump
from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.detection_types import Detection3D
from isaac_bench.perception.object_memory import ObjectMemory


def test_graph_debug_dump_contains_runtime_nodes_edges_and_scores(tmp_path):
    map_info = MapInfo(resolution_m=0.5, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=10, height=10)
    memory = ObjectMemory(merge_radius_m=0.2)
    memory.update(
        [
            Detection3D("table", "table", 0.9, (2.0, 2.0, 0.5), (0, 0, 10, 10)),
            Detection3D("chair", "chair", 0.8, (2.5, 2.0, 0.5), (0, 0, 10, 10)),
        ],
        step_id=1,
        map_info=map_info,
    )
    room_map = np.zeros((1, len(SGNavSceneGraphAdapter().room_names), 10, 10), dtype=np.float32)
    room_map[0, 1, :, :] = 1.0
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("table")
    scenegraph.update_from_frame(memory, room_map=room_map, map_info=map_info)
    decision = SGNavDecision(scenegraph, frontier_distance_weight=0.7)
    frontiers = [FrontierCluster((3, 3), (1.5, 1.5), [(3, 3), (3, 4)], 2, 1.0)]
    nav = decision.choose_navigation_target(
        memory,
        "unknown_goal",
        (1, 1),
        frontiers,
        GridAStarPlanner(np.ones((10, 10), dtype=bool), resolution_m=0.5),
        map_info,
        (0.5, 0.5, 0.0, 0.0),
    )

    out = save_graph_debug_dump(
        str(tmp_path),
        step=7,
        goal="table",
        scenegraph=scenegraph,
        frontiers=frontiers,
        frontier_decision=nav.frontier_decision,
        nav_decision=nav,
        commitment_metadata={"active_frontier_id": 3},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["objects"]
    assert payload["rooms"]
    assert payload["groups"]
    assert any(edge["rel"] in {"belongs to", "related near", "near"} for edge in payload["edges"])
    assert payload["frontiers"][0]["total"] is not None
    assert payload["commitment"]["active_frontier_id"] == 3


def test_scenegraph_loads_semantic_priors_yaml_aliases(tmp_path):
    priors = tmp_path / "priors.yaml"
    priors.write_text(
        """
room_names: [living room]
related_category_pairs:
  - [tv, sofa]
goal_room_priors:
  tv: [living room]
aliases:
  television: tv
""",
        encoding="utf-8",
    )
    scenegraph = SGNavSceneGraphAdapter(use_original=False, semantic_priors_path=str(priors))
    scenegraph.reset("television")

    assert scenegraph.obj_goal_sg == "tv"
    assert ("sofa", "tv") in scenegraph.related_category_pairs
