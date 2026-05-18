import json

import numpy as np

from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter, VLLMFrontierScorer
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.mapping.room_map_from_rooms_json import load_rooms
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.detection_types import Detection2D, Detection3D
from isaac_bench.perception.object_memory import ObjectMemory, ObjectNode
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
    for step in range(1, 4):
        memory.update([Detection3D("table", "table", 0.9, (5.0, 5.0, 0.5), (0, 0, 10, 10))], step_id=step, map_info=map_info)

    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("table")
    scenegraph.update(memory)
    decision = SGNavDecision(scenegraph, candidate_min_hits=1)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    target = decision.choose_navigation_target(memory, "table", (8, 1), [], planner, map_info, (1.0, 1.0, 0.0, 0.0), current_step=3)

    assert target.mode == "candidate"
    assert target.selected_candidate is memory.nodes[0]
    assert target.target_cells


def test_sgnav_can_score_frontiers_before_candidate():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    traversible = np.ones((10, 10), dtype=bool)
    memory = ObjectMemory(merge_radius_m=0.5)
    for step in range(1, 4):
        memory.update([Detection3D("table", "table", 0.9, (5.0, 5.0, 0.5), (0, 0, 10, 10))], step_id=step, map_info=map_info)

    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("table")
    scenegraph.update(memory)
    calls = []

    def score(frontier_locations, num_frontiers):
        calls.append((frontier_locations.copy(), num_frontiers))
        return np.ones((num_frontiers,), dtype=np.float32)

    scenegraph.score = score
    decision = SGNavDecision(scenegraph, candidate_min_hits=1, score_frontiers_before_candidate=True)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    frontiers = [FrontierCluster((2, 2), (2.0, 2.0), [(2, 2), (2, 3), (3, 2)], 3, 2.0)]

    target = decision.choose_navigation_target(memory, "table", (8, 1), frontiers, planner, map_info, (1.0, 1.0, 0.0, 0.0), current_step=3)

    assert target.mode == "candidate"
    assert calls
    assert target.frontier_decision is not None


def test_sgnav_candidate_uses_nearest_reachable_cell_when_standoff_unknown():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    traversible = np.zeros((10, 10), dtype=bool)
    traversible[1:7, 1:7] = True
    memory = ObjectMemory(merge_radius_m=0.5)
    memory.update([Detection3D("mirror", "mirror", 0.9, (8.0, 8.0, 0.5), (0, 0, 10, 10))], step_id=1, map_info=map_info)

    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("mirror")
    scenegraph.update(memory)
    decision = SGNavDecision(
        scenegraph,
        candidate_min_hits=1,
        candidate_start_min_hits=1,
        candidate_accept_requires_reperception=False,
        candidate_standoff_max_m=1.0,
        frontier_allow_near_fallback=True,
    )
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    target = decision.choose_navigation_target(memory, "mirror", (1, 1), [], planner, map_info, (1.0, 1.0, 0.0, 0.0))

    assert target.mode == "candidate"
    assert target.selected_candidate is memory.nodes[0]
    assert target.target_cells
    assert target.target_cells[0] != (1, 1)


def test_sgnav_tiny_candidate_progress_falls_back_to_frontier():
    map_info = MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.25, min_y=0.0, max_y=0.25, width=5, height=5)
    traversible = np.zeros((5, 5), dtype=bool)
    traversible[2, 2:4] = True
    memory = ObjectMemory(merge_radius_m=0.5)
    memory.update([Detection3D("mirror", "mirror", 0.9, (4.0, 4.0, 0.5), (0, 0, 10, 10))], step_id=1, map_info=map_info)

    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("mirror")
    scenegraph.update(memory)
    decision = SGNavDecision(
        scenegraph,
        candidate_min_hits=1,
        candidate_start_min_hits=1,
        candidate_accept_requires_reperception=False,
        candidate_standoff_max_m=1.0,
        frontier_allow_near_fallback=True,
    )
    planner = GridAStarPlanner(traversible, resolution_m=0.05, allow_diagonal=True)
    frontiers = [FrontierCluster((2, 3), (0.15, 0.10), [(2, 3)], 1, 0.05)]
    target = decision.choose_navigation_target(memory, "mirror", (2, 2), frontiers, planner, map_info, (0.1, 0.1, 0.0, 0.0))

    assert target.mode == "frontier"
    assert target.selected_candidate is None
    assert target.target_cells == [(2, 3)]


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


def test_scenegraph_builds_runtime_object_group_and_edges():
    map_info = MapInfo(resolution_m=0.5, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=10, height=10)
    memory = ObjectMemory(merge_radius_m=0.2)
    memory.update(
        [
            Detection3D("table", "table", 0.9, (2.0, 2.0, 0.5), (0, 0, 10, 10)),
            Detection3D("chair", "chair", 0.8, (2.6, 2.0, 0.5), (0, 0, 10, 10)),
        ],
        step_id=1,
        map_info=map_info,
    )
    room_map = np.zeros((1, len(SGNavSceneGraphAdapter().room_names), 10, 10), dtype=np.float32)
    room_map[0, 1, :, :] = 1.0
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("table")
    scenegraph.update_from_frame(memory, room_map=room_map, map_info=map_info, observed=np.ones((10, 10), dtype=bool))

    assert len(scenegraph.runtime_nodes) == 2
    assert len(scenegraph.runtime_groups) == 1
    assert any(edge.relation in {"related near", "near"} for edge in scenegraph.runtime_edges)
    assert "groups:" in scenegraph.graph_summary()


def test_original_scenegraph_receives_fbe_free_map_not_navigable():
    class FakeOriginalSceneGraph:
        def __init__(self):
            self.full_map = None
            self.fbe_free_map = None
            self.room_nodes = []
            self.nodes = []
            self._subgraph_score_cache_by_key = {}
            self._subgraph_score_cache = []
            self._subgraph_score_cache_key = None

        def set_room_map(self, room_map):
            self.room_map = room_map

        def set_full_map(self, full_map):
            self.full_map = full_map

        def set_fbe_free_map(self, fbe_free_map):
            self.fbe_free_map = fbe_free_map

        def update_group(self):
            pass

    memory = ObjectMemory()
    occupancy = np.zeros((3, 3), dtype=bool)
    free = np.ones((3, 3), dtype=bool)
    navigable = np.zeros((3, 3), dtype=bool)
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    fake = FakeOriginalSceneGraph()
    scenegraph.scenegraph = fake

    scenegraph.update_from_frame(memory, occupancy=occupancy, free=free, navigable=navigable)

    assert np.array_equal(fake.full_map[0, 0], occupancy.astype(np.float32))
    assert np.array_equal(fake.fbe_free_map[0, 0], free.astype(np.float32))


def test_scenegraph_dedupes_memory_nodes_before_runtime_graph():
    map_info = MapInfo(resolution_m=0.5, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0, width=10, height=10)
    memory = ObjectMemory(merge_radius_m=0.5)
    memory.nodes = [
        ObjectNode(0, "sofa", (1.0, 1.0, 0.5), (2, 2), 0.8, 1, 1, "sofa"),
        ObjectNode(1, "couch", (1.2, 1.1, 0.7), (2, 2), 0.9, 2, 2, "couch"),
    ]
    memory._next_id = 2
    scenegraph = SGNavSceneGraphAdapter(use_original=False)

    scenegraph.update_from_frame(memory, map_info=map_info)

    assert len(memory.nodes) == 1
    assert len(scenegraph.runtime_nodes) == 1
    node = next(iter(scenegraph.runtime_nodes.values()))
    assert node.caption == "sofa"
    assert node.observed_count == 3


def test_sgnav_decision_reperception_then_verified_stop():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    traversible = np.ones((10, 10), dtype=bool)
    memory = ObjectMemory(merge_radius_m=0.5)
    memory.update([Detection3D("mirror", "mirror", 0.9, (1.2, 1.0, 0.5), (0, 0, 10, 10))], step_id=1, map_info=map_info)
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("mirror")
    scenegraph.update(memory)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    decision = SGNavDecision(
        scenegraph,
        candidate_min_hits=2,
        candidate_start_min_hits=1,
        reperception_min_observations=3,
        stop_verification_steps=1,
        found_goal_stop_distance_m=0.5,
    )

    first = decision.choose_navigation_target(memory, "mirror", (1, 1), [], planner, map_info, (1.0, 1.0, 0.0, 0.0))
    assert first.mode == "reperception"

    memory.nodes[0].observed_count = 3
    second = decision.choose_navigation_target(memory, "mirror", (1, 1), [], planner, map_info, (1.0, 1.0, 0.0, 0.0))
    assert second.mode == "stop"
    assert second.reason == "stop_verification_confirmed"


def test_one_hit_candidate_does_not_override_frontier():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    traversible = np.ones((10, 10), dtype=bool)
    memory = ObjectMemory(merge_radius_m=0.5)
    memory.nodes = [
        ObjectNode(1, "mirror", (5.0, 5.0, 0.5), (5, 5), 0.79, 1, 35, "mirror"),
    ]
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    scenegraph.reset("mirror")
    scenegraph.update(memory)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)
    frontiers = [FrontierCluster((2, 3), (3.0, 2.0), [(2, 3)], 1, 2.0)]
    decision = SGNavDecision(scenegraph)

    target = decision.choose_navigation_target(
        memory,
        "mirror",
        (2, 2),
        frontiers,
        planner,
        map_info,
        (2.0, 2.0, 0.0, 0.0),
        current_step=35,
    )

    assert target.mode == "frontier"
    assert target.selected_candidate is None
    assert target.target_cells == [(2, 3)]


def test_candidate_standoff_cells_are_capped():
    map_info = MapInfo(resolution_m=0.25, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=40, height=40)
    traversible = np.ones((40, 40), dtype=bool)
    candidate = ObjectNode(3, "mirror", (5.0, 5.0, 0.5), (20, 20), 0.9, 3, 3, "mirror")
    scenegraph = SGNavSceneGraphAdapter(use_original=False)
    decision = SGNavDecision(scenegraph, candidate_standoff_max_cells=5)

    cells = decision.candidate_standoff_cells(candidate, traversible, map_info, current_grid=(4, 4))

    assert cells
    assert len(cells) <= 5


def test_vllm_frontier_scorer_encodes_cpu_rgb_image():
    scorer = VLLMFrontierScorer({"enabled": True, "image_max_width": 16, "image_jpeg_quality": 50})
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    data_url = scorer._image_to_data_url(image)

    assert data_url is not None
    assert data_url.startswith("data:image/jpeg;base64,")


def test_low_confidence_2d_detections_are_filtered():
    detections = [
        Detection2D("table", "table", 0.55, (0, 0, 10, 10)),
        Detection2D("chair", "chair", 0.56, (0, 0, 10, 10)),
    ]

    kept = filter_detections_by_confidence(detections, 0.55)

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
