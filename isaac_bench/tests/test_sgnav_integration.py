import json

import numpy as np

from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.mapping.room_map_from_rooms_json import load_rooms
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.detection_types import Detection2D, Detection3D
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.scripts.run_one_episode import detections_to_3d_static_map_ray, filter_detections_by_confidence


def test_rooms_json_accepts_polygon_xy(tmp_path):
    rooms_path = tmp_path / "rooms.json"
    rooms_path.write_text(
        json.dumps(
            [
                {
                    "room_id": "0",
                    "room_type": "living_room",
                    "polygon_xy": [[0, 0], [2, 0], [2, 2], [0, 2]],
                }
            ]
        ),
        encoding="utf-8",
    )
    rooms = load_rooms(str(rooms_path))
    assert rooms[0].room_type == "living_room"
    assert rooms[0].area_m2 == 4.0


def test_sgnav_decision_prefers_detected_goal_candidate():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    traversible = np.ones((10, 10), dtype=bool)
    memory = ObjectMemory(merge_radius_m=0.5)
    memory.update([Detection3D("table", "table", 0.9, (5.0, 5.0, 0.5), (0, 0, 10, 10))], step_id=1, map_info=map_info)

    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("table")
    scenegraph.update(memory)
    decision = SGNavDecision(scenegraph, candidate_min_hits=1)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    target = decision.choose_navigation_target(memory, "table", (8, 1), [], planner, map_info, (1.0, 1.0, 0.0, 0.0))

    assert target.mode == "candidate"
    assert target.selected_candidate is memory.nodes[0]
    assert target.target_cells


def test_scenegraph_fallback_scores_goal_near_frontier_higher():
    memory = ObjectMemory()
    memory.nodes.append(
        memory_node := type(
            "Node",
            (),
            {
                "category": "table",
                "center_grid": (5, 5),
                "center_world": (5.0, 5.0, 0.0),
                "confidence": 1.0,
                "observed_count": 2,
            },
        )()
    )
    _ = memory_node
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("table")
    scenegraph.update(memory)
    frontiers = np.asarray([[5, 6], [9, 9]], dtype=np.int32)
    scores = scenegraph.score(frontiers, 2)
    assert scores[0] > scores[1]


def test_low_confidence_2d_detections_are_filtered():
    detections = [
        Detection2D("table", "table", 0.49, (0, 0, 10, 10)),
        Detection2D("chair", "chair", 0.50, (0, 0, 10, 10)),
    ]

    kept = filter_detections_by_confidence(detections, 0.5)

    assert [det.category for det in kept] == ["chair"]


def test_static_map_ray_localizes_rgb_detection_to_first_occupied_cell():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=5, height=5)
    occupancy = np.zeros((5, 5), dtype=bool)
    navigable = np.ones((5, 5), dtype=bool)
    occupancy[2, 3] = True
    navigable[2, 3] = False
    detections = [Detection2D("table", "table", 0.9, (45.0, 10.0, 55.0, 30.0))]

    localized = detections_to_3d_static_map_ray(
        detections,
        camera_pose_world=(1.5, 2.5, 1.35, 0.0),
        image_width=100,
        camera_hfov_deg=90.0,
        map_info=map_info,
        occupancy=occupancy,
        navigable=navigable,
        max_range_m=4.0,
    )

    assert len(localized) == 1
    assert localized[0].category == "table"
    assert localized[0].center_world[:2] == (3.5, 2.5)
