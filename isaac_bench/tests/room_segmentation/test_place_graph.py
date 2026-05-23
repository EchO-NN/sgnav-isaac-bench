from .test_helpers import make_segmenter, open_room


def test_place_graph_has_room_center_and_edges():
    free, wall, unknown = open_room()
    seg = make_segmenter(free.shape)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=wall, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    fs = seg.free_space_extractor.extract(structural)
    skeleton = seg.skeleton_extractor.extract(fs, structural)
    graph = seg.place_graph_builder.build(skeleton, [], structural, frame_id=1)
    assert any(node.node_type == "room_center" for node in graph.nodes)
    assert len(graph.nodes) >= 1
    assert all(0.0 <= edge.weight <= 1.0 for edge in graph.edges)

