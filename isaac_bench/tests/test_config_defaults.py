from isaac_bench.config import load_config


def test_required_sgnav_defaults():
    cfg = load_config("isaac_bench/configs/isaac_bench.yaml")

    assert cfg["mapping"]["frontier_min_distance_m"] == 1.0
    assert cfg["sgnav"]["frontier_distance_weight"] == 0.2
    assert cfg["llm"]["enabled"] is True
    assert cfg["perception"]["confidence_threshold"] == 0.65
    assert cfg["perception"]["min_valid_detection_confidence"] == 0.65
    assert cfg["sgnav"]["candidate_start_min_confidence"] == 0.65
    assert cfg["mapping"]["room_map_mode"] == "online_geometry_watershed"
    assert cfg["mapping"]["room_segmentation"]["proposal_mode"] == "distance_watershed"
    assert cfg["mapping"]["room_segmentation"]["finalization_mode"] == "doorway_constrained_merge"
    assert cfg["mapping"]["room_segmentation"]["use_structural_obstacle_mask"] is True
    assert cfg["mapping"]["room_segmentation"]["max_clutter_component_area_m2"] == 4.0
    assert cfg["mapping"]["room_segmentation"]["use_premerge_labels_for_open_plan_merge"] is True
    assert cfg["room_semantics"]["min_label_reliability_for_functional_split"] == 0.65
    assert cfg["room_semantics"]["unknown_allows_functional_split"] is False
    assert cfg["perception"]["yolo_world"]["reject_edge_touching_bboxes"] is True
    assert cfg["perception"]["yolo_world"]["category_accumulation"] is True
    assert cfg["object_memory"]["category_update_mode"] == "confidence_sum"
    assert cfg["sgnav"]["scene_graph"]["room_nodes"]["source"] == "online_geometry_watershed_vlm"
    assert cfg["visualization"]["show_gt_goal_cells"] is False
    assert cfg["visualization"]["show_room_proposals"] is True
