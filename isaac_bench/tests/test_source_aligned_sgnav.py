import numpy as np

from isaac_bench.graph.decision import NavigationDecision, SGNavDecision
from isaac_bench.graph.paper_scene_graph import PaperSceneGraph
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier_debug import save_frontier_debug_snapshot
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.detection_types import Detection3D
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.scripts.run_one_episode import (
    LongTermGoalState,
    candidate_center_payload,
    goal_candidate_pair_distances,
    trim_path_to_nearest,
)


def test_long_term_goal_state_reuses_locked_navigation_decision():
    decision = NavigationDecision("frontier", [(4, 5), (4, 6)], False, None, None, "selected_frontier")
    state = LongTermGoalState()

    state.set_from(decision, step=7)
    locked = state.to_navigation_decision()

    assert state.exists()
    assert locked.target_cells == [(4, 5), (4, 6)]
    assert locked.reason == "locked_long_term_goal"
    assert locked.metadata["long_term_goal_locked"] is True


def test_trim_path_to_nearest_tolerates_small_tracking_error():
    path = [(0, 0), (1, 1), (2, 2), (3, 3)]

    assert trim_path_to_nearest(path, (2, 3), max_dist_cells=2) == [(2, 2), (3, 3)]
    assert trim_path_to_nearest(path, (10, 10), max_dist_cells=2) == []


def test_goal_candidate_relaxed_dedupe_and_duplicate_metadata():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    memory = ObjectMemory(merge_radius_m=0.2)
    memory.update(
        [
            Detection3D("chair", "chair", 0.8, (1.0, 1.0, 0.5), (0, 0, 10, 10)),
            Detection3D("chair", "chair", 0.9, (1.5, 1.0, 0.5), (0, 0, 10, 10)),
            Detection3D("table", "table", 0.9, (2.0, 2.0, 0.5), (0, 0, 10, 10)),
        ],
        step_id=1,
        map_info=map_info,
    )

    before = goal_candidate_pair_distances(memory, "chair")
    memory.dedupe_goal_candidates("chair", merge_radius_m=0.75, map_info=map_info)

    assert before and before[0]["dist"] < 0.75
    assert len([node for node in memory.nodes if node.category == "chair"]) == 1


def test_candidate_center_payload_marks_selected_candidate():
    memory = ObjectMemory()
    memory.update([Detection3D("chair", "chair", 0.8, (1.0, 1.0, 0.5), (0, 0, 10, 10))], step_id=1)
    memory.update([Detection3D("chair", "chair", 0.8, (2.0, 2.0, 0.5), (0, 0, 10, 10))], step_id=1)

    payload = candidate_center_payload(memory, "chair", selected_candidate_id=memory.nodes[1].node_id)

    assert payload[0]["status"] == "candidate"
    assert payload[1]["status"] == "selected"


def test_frontier_debug_accepts_candidate_status_payload(tmp_path):
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    npz_path, png_path = save_frontier_debug_snapshot(
        tmp_path,
        1,
        free=mask,
        occupied=np.zeros_like(mask),
        observed=mask,
        unknown=~mask,
        unknown_dilated=~mask,
        frontier=mask,
        traversible=mask,
        dist_map=np.zeros((5, 5), dtype=np.float32),
        agent_grid=(2, 2),
        clusters=[],
        candidate_centers=[{"center_grid": (2, 2), "status": "selected"}],
    )

    assert npz_path.exists()
    assert png_path is None or png_path.exists()
    loaded = np.load(npz_path)
    assert loaded["candidate_center_status"].tolist() == ["selected"]


def test_paper_scene_graph_tracks_new_objects_for_incremental_edges():
    memory = ObjectMemory()
    graph = PaperSceneGraph()

    memory.update([Detection3D("table", "table", 0.9, (1.0, 1.0, 0.5), (0, 0, 10, 10))], step_id=1)
    graph.update_from_object_memory(memory)
    first_new = list(graph.new_object_ids)
    graph.update_from_object_memory(memory)

    assert first_new == ["object:0"]
    assert graph.new_object_ids == []


def test_paper_adapter_only_proposes_edges_for_new_objects(monkeypatch):
    memory = ObjectMemory()
    scenegraph = SGNavSceneGraphAdapter(use_original=False, sgnav_mode="paper")
    calls = []

    def fake_propose(new_objects, all_objects, llm_client=None):
        calls.append(([node.id for node in new_objects], [node.id for node in all_objects], llm_client))
        return []

    monkeypatch.setattr("isaac_bench.graph.sgnav_scenegraph_adapter.propose_object_edges_with_llm", fake_propose)
    memory.update([Detection3D("table", "table", 0.9, (1.0, 1.0, 0.5), (0, 0, 10, 10))], step_id=1)
    scenegraph.update_from_frame(memory)
    scenegraph.update_from_frame(memory)

    assert calls[0][0] == ["object:0"]
    assert len(calls) == 1


def test_room_dbscan_like_groups_cluster_by_distance_not_category_relation():
    memory = ObjectMemory(merge_radius_m=0.1)
    graph = PaperSceneGraph()
    memory.update(
        [
            Detection3D("plant", "plant", 0.9, (1.0, 1.0, 0.5), (0, 0, 10, 10)),
            Detection3D("picture", "picture", 0.9, (1.3, 1.0, 0.5), (0, 0, 10, 10)),
            Detection3D("chair", "chair", 0.9, (3.0, 3.0, 0.5), (0, 0, 10, 10)),
        ],
        step_id=1,
    )
    graph.update_from_object_memory(memory)
    graph.update_group_nodes(eps_m=0.5)

    groups = list(graph.group_nodes.values())
    assert any({"object:0", "object:1"} == set(group.object_ids) for group in groups)
    assert any(group.object_ids == ["object:2"] for group in groups)


def test_paper_mode_disables_direct_frontier_vllm_scorer():
    scenegraph = SGNavSceneGraphAdapter(use_original=False, sgnav_mode="paper", vllm_config={"enabled": True})

    assert scenegraph.paper_llm_client is not None
    assert scenegraph.vllm_scorer.enabled is False


def test_paper_reperception_metadata_accumulates_graph_credibility():
    map_info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, width=10, height=10)
    traversible = np.ones((10, 10), dtype=bool)
    memory = ObjectMemory(merge_radius_m=0.5)
    for step in range(1, 4):
        memory.update([Detection3D("chair", "chair", 0.9, (1.0, 1.0, 0.5), (0, 0, 10, 10))], step_id=step, map_info=map_info)
    scenegraph = SGNavSceneGraphAdapter(use_original=False, sgnav_mode="paper")
    scenegraph.reset("chair")
    scenegraph.update_from_frame(memory, map_info=map_info)
    decision = SGNavDecision(scenegraph, candidate_start_min_hits=1, reperception_min_observations=1, candidate_accept_threshold=0.1)
    planner = GridAStarPlanner(traversible, resolution_m=1.0, allow_diagonal=True)

    target = decision.choose_navigation_target(memory, "chair", (1, 1), [], planner, map_info, (1.0, 1.0, 0.0, 0.0), current_step=3)

    assert target.metadata["candidate_credibility_method"] == "graph_based"
    assert target.metadata["reperception"]["accumulated_credibility"] > 0.0
