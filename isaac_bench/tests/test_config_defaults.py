from isaac_bench.config import load_config


def test_required_sgnav_defaults():
    cfg = load_config("isaac_bench/configs/isaac_bench.yaml")

    assert cfg["mapping"]["frontier_min_distance_m"] == 1.0
    assert cfg["sgnav"]["frontier_distance_weight"] == 0.2
    assert cfg["llm"]["enabled"] is True
    assert cfg["mapping"]["room_map_mode"] == "online_geometry_watershed"
    assert cfg["sgnav"]["scene_graph"]["room_nodes"]["source"] == "online_geometry_watershed_vlm"
    assert cfg["visualization"]["show_gt_goal_cells"] is False
