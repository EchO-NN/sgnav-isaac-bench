from isaac_bench.config import load_config
from isaac_bench.scripts.run_one_episode import effective_perception_every_steps


def test_required_sgnav_defaults():
    cfg = load_config("isaac_bench/configs/isaac_bench.yaml")

    assert cfg["mapping"]["frontier_min_distance_m"] == 1.0
    assert cfg["sgnav"]["frontier_distance_weight"] == 0.2
    assert cfg["llm"]["enabled"] is True
    assert cfg["perception"]["confidence_threshold"] == 0.55
    assert cfg["perception"]["min_valid_detection_confidence"] == 0.55
    assert cfg["sgnav"]["candidate_start_min_confidence"] == 0.55
    assert cfg["mapping"]["room_map_mode"] == "upstream_rose2_vertical_or_free"
    assert cfg["mapping"]["strict_no_oracle_rooms"] is True
    assert cfg["mapping"]["room_segmentation"]["algorithm"] == "upstream_rose2_vertical_or_free"
    assert cfg["mapping"]["room_segmentation"]["source_mode"] == "declutter_reconstruct_mit"
    assert cfg["mapping"]["room_segmentation"]["legacy_watershed_allowed"] == "debug_only"
    assert cfg["mapping"]["room_segmentation"]["local_rose2_lite_allowed"] == "debug_only"
    assert cfg["mapping"]["room_segmentation"]["require_upstream_source_for_strict"] is True
    assert cfg["mapping"]["room_segmentation"]["upstream_repo_env"] == "ROSE2_SOURCE_ROOT"
    assert cfg["mapping"]["room_segmentation"]["run_only_before_frontier_scoring"] is True
    assert cfg["mapping"]["room_segmentation"]["proposal_mode"] == "distance_watershed"
    assert cfg["mapping"]["room_segmentation"]["finalization_mode"] == "doorway_constrained_merge"
    assert cfg["mapping"]["room_segmentation"]["use_structural_obstacle_mask"] is True
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["z_min_m"] == 0.20
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["z_max_m"] == 2.00
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["min_free_rays"] == 1
    assert cfg["mapping"]["room_segmentation"]["vertical_or_free"]["min_observed_rays"] == 1
    assert cfg["mapping"]["room_segmentation"]["max_clutter_component_area_m2"] == 4.0
    assert cfg["mapping"]["room_segmentation"]["open_boundary_merge"] is True
    assert cfg["mapping"]["room_segmentation"]["use_premerge_labels_for_open_plan_merge"] is True
    assert cfg["mapping"]["room_segmentation"]["wall_confidence_threshold"] == 0.55
    assert cfg["mapping"]["room_segmentation"]["vertical_free_suppression_weight"] == 0.35
    assert cfg["mapping"]["room_segmentation"]["object_overlap_suppression_weight"] == 0.45
    assert cfg["mapping"]["room_segmentation"]["furniture_suppression_radius_m"] == 0.90
    assert cfg["mapping"]["room_segmentation"]["wall_like_aspect_ratio_min"] == 4.0
    assert cfg["mapping"]["room_segmentation"]["rose2"]["wall_extension_enabled"] is True
    assert cfg["mapping"]["room_segmentation"]["rose2"]["wall_extension_band_m"] == 0.45
    assert cfg["mapping"]["room_segmentation"]["rose2"]["wall_extension_margin_m"] == 0.15
    assert cfg["room_semantics"]["min_label_reliability_for_functional_split"] == 0.65
    assert cfg["room_semantics"]["unknown_allows_functional_split"] is False
    assert cfg["perception"]["yolo_world"]["reject_edge_touching_bboxes"] is False
    assert cfg["perception"]["yolo_world"]["mask_aware_partial_tracking"] is True
    assert cfg["perception"]["yolo_world"]["category_accumulation"] is True
    assert cfg["object_memory"]["category_update_mode"] == "confidence_sum"
    assert cfg["object_memory"]["partial_class_weight"] == 0.25
    assert cfg["object_memory"]["mask_iou_track_match_threshold"] == 0.25
    assert cfg["object_memory"]["mask_containment_track_match_threshold"] == 0.60
    assert cfg["object_memory"]["footprint_iou_track_match_threshold"] == 0.20
    assert cfg["object_memory"]["child_containment_threshold"] == 0.70
    assert cfg["object_memory"]["child_object_area_ratio_max"] == 0.35
    assert cfg["sgnav"]["scene_graph"]["room_nodes"]["source"] == "upstream_rose2_vertical_or_free_vlm"
    assert cfg["visualization"]["show_gt_goal_cells"] is False
    assert cfg["visualization"]["show_room_proposals"] is True
    assert cfg["isaac"]["perception_every_steps"] == 1


def test_yolo_world_forces_every_frame_perception():
    assert effective_perception_every_steps("yolo_world", 5) == 1
    assert effective_perception_every_steps("yolo_world", 1) == 1
    assert effective_perception_every_steps("dry_run", 5) == 5
    assert effective_perception_every_steps("none", 5) == 5
