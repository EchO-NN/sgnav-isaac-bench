import json

import numpy as np

from isaac_bench.graph.room_semantics import RoomSemanticLabel
from isaac_bench.mapping.room_segmentation import (
    OnlineRoomSegmenter,
    RoomMask,
    RoomSegmentationConfig,
    merge_open_plan_proposals,
)
from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def _cfg():
    return RoomSegmentationConfig(
        resolution_m=0.1,
        min_observed_free_cells=20,
        min_room_area_m2=0.2,
        morphology_close_radius_m=0.0,
        morphology_open_radius_m=0.0,
        seed_min_clearance_m=0.2,
        seed_min_distance_m=1.1,
        doorway_width_min_m=0.4,
        doorway_width_max_m=1.4,
        doorway_clearance_max_m=2.5,
        doorway_endpoint_wall_distance_m=0.35,
        small_segment_merge_area_m2=0.2,
        max_clutter_component_area_m2=4.0,
        min_wall_line_length_m=1.5,
        wall_like_aspect_ratio_min=3.0,
    )


def test_open_living_room_with_furniture_merges_to_one_final_room():
    free = np.zeros((90, 110), dtype=bool)
    free[10:80, 10:100] = True
    obstacle = np.zeros_like(free)
    for r0, c0, r1, c1 in [(24, 24, 31, 34), (46, 48, 58, 62), (22, 70, 28, 78), (60, 22, 67, 31)]:
        obstacle[r0:r1, c0:c1] = True
        free[r0:r1, c0:c1] = False
    unknown = ~free & ~obstacle

    segmenter = OnlineRoomSegmenter(_cfg())
    rooms = [room for room in segmenter.update(obstacle, free, obstacle, unknown, step=0) if not room.stale]

    assert len(rooms) == 1
    assert segmenter.last_debug["proposal_room_count"] >= 1
    assert segmenter.last_debug["final_room_count"] == 1
    assert segmenter.last_debug["structural_obstacle_mask"]["suppressed_obstacle_cells"] > 0


def test_two_rooms_with_narrow_doorway_preserves_verified_doorway_split():
    free = np.zeros((80, 80), dtype=bool)
    free[10:70, 8:72] = True
    obstacle = np.zeros_like(free)
    obstacle[:, 39:41] = True
    free[:, 39:41] = False
    free[34:45, 39:41] = True
    obstacle[34:45, 39:41] = False
    unknown = ~free & ~obstacle

    segmenter = OnlineRoomSegmenter(_cfg())
    rooms = [room for room in segmenter.update(obstacle, free, obstacle, unknown, step=0) if not room.stale]

    assert len(rooms) == 2
    assert segmenter.last_debug["final_room_count"] == 2
    assert any(item["verified_doorway"] for item in segmenter.last_debug["adjacency_evidence"])
    assert any(edge["edge_type"] == "adjacent_via_doorway" for room in rooms for edge in room.doorway_edges)


def test_living_room_three_proposals_same_label_merge():
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[5:35, 5:22] = 1
    labels[5:35, 22:40] = 2
    labels[5:35, 40:55] = 3
    free = labels > 0
    obstacle = np.zeros_like(free)
    unknown = ~free
    distance_m = np.ones_like(labels, dtype=np.float32) * 0.5

    out, debug, doorway_edges = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance_m,
        _cfg(),
        proposal_semantic_labels={1: "living_room", 2: "living_room", 3: "living_room"},
    )

    assert len([v for v in np.unique(out) if int(v) > 0]) == 1
    assert doorway_edges == []
    assert any("same_semantic_living_room" in op["reason"] for op in debug["merge_operations"])


def test_unknown_open_adjacent_to_living_room_merges_without_guessing_category():
    labels = np.zeros((30, 50), dtype=np.int32)
    labels[5:25, 5:25] = 1
    labels[5:25, 25:45] = 2
    free = labels > 0
    obstacle = np.zeros_like(free)
    unknown = ~free
    distance_m = np.ones_like(labels, dtype=np.float32) * 0.5

    out, debug, _doorway_edges = merge_open_plan_proposals(
        labels,
        free,
        obstacle,
        unknown,
        distance_m,
        _cfg(),
        proposal_semantic_labels={1: "unknown", 2: "living_room"},
    )

    assert len([v for v in np.unique(out) if int(v) > 0]) == 1
    assert any(op["reason"] == "open_unknown_into_open_labeled_region" for op in debug["merge_operations"])


def test_unknown_boundary_not_wall_support_for_fake_doorway():
    labels = np.zeros((40, 50), dtype=np.int32)
    labels[5:35, 5:25] = 1
    labels[5:35, 25:45] = 2
    free = labels > 0
    obstacle = np.zeros_like(free)
    unknown = np.zeros_like(free)
    unknown[4:36, 23:27] = True
    distance_m = np.ones_like(labels, dtype=np.float32) * 0.45

    out, debug, doorway_edges = merge_open_plan_proposals(labels, free, obstacle, unknown, distance_m, _cfg())

    assert len([v for v in np.unique(out) if int(v) > 0]) == 1
    assert doorway_edges == []
    assert any(item["merge_reason"] != "verified_structural_doorway" for item in debug["adjacency_evidence"])


def test_room_visualization_proposal_vs_final_layers(tmp_path):
    final_mask = np.zeros((30, 50), dtype=bool)
    final_mask[5:25, 5:45] = True
    proposal_a = np.zeros_like(final_mask)
    proposal_b = np.zeros_like(final_mask)
    proposal_a[5:25, 5:25] = True
    proposal_b[5:25, 25:45] = True
    room = RoomMask(
        room_id="room_0001",
        mask=final_mask,
        centroid_xy=(0.0, 0.0),
        area_m2=8.0,
        boundary_unknown_fraction=0.0,
        doorway_edges=[],
        confidence=0.9,
        observed_free_cells=int(np.count_nonzero(final_mask)),
        mask_confidence=0.9,
        metadata={"centroid_grid": [15, 25]},
    )
    label = RoomSemanticLabel(
        room_id="room_0001",
        category="living_room",
        confidence=0.78,
        supporting_objects=[],
        conflicting_evidence=[],
        rationale="test",
        backend="vlm",
        vlm_self_confidence=0.8,
        label_reliability=0.78,
        reliability_factors={},
    )
    debug = {
        "proposal_room_count": 2,
        "final_room_count": 1,
        "proposal_room_masks": [
            {"label_id": 1, "mask": proposal_a.astype(np.uint8).tolist()},
            {"label_id": 2, "mask": proposal_b.astype(np.uint8).tolist()},
        ],
        "adjacency_evidence": [
            {
                "room_a_label": 1,
                "room_b_label": 2,
                "verified_doorway": False,
                "merge_reason": "open_plan_no_verified_doorway_same_semantic_living_room",
                "boundary_cells_sample": [[row, 24] for row in range(5, 25)],
            }
        ],
    }
    viz = SGNavPopupVisualizer(enabled=False, save_dir=str(tmp_path), panel_size=(640, 360), save_every_steps=1)
    viz.set_room_context([room], {"room_0001": label}, debug)
    viz.update(
        step=0,
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        detections_2d=[],
        occupancy=np.zeros_like(final_mask),
        navigable=np.ones_like(final_mask),
        observed=np.ones_like(final_mask),
        goal_cells=[],
        current_grid=(15, 25),
        pose=(0.0, 0.0, 0.0, 0.0),
        frontiers=[],
        nav_decision=None,
        current_path=[],
        full_path=[],
        object_memory=type("Memory", (), {"nodes": []})(),
        goal_category="chair",
        distance_to_goal=1.0,
        path_length=0.0,
        scenegraph_backend="test",
    )

    layers = {
        item["name"]: item
        for item in json.loads((tmp_path / "sgnav_step_000000.layers.json").read_text(encoding="utf-8"))["layers"]
    }
    assert layers["proposal_room_masks"]["primitive_count"] > 0
    assert layers["final_room_masks"]["primitive_count"] > 0
    assert layers["room_merged_boundaries"]["primitive_count"] > 0
    assert layers["room_merged_boundaries"]["adjacency_merge_reasons"][0]["merge_reason"].endswith("living_room")
