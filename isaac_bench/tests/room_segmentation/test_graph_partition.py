from .test_helpers import make_segmenter, two_rooms_one_door


def test_graph_partition_respects_soft_separator_map():
    free, wall, unknown = two_rooms_one_door()
    seg = make_segmenter(free.shape)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=wall, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    fs = seg.free_space_extractor.extract(structural)
    skeleton = seg.skeleton_extractor.extract(fs, structural)
    candidates = seg.separator_scorer.score_all(seg.bottleneck_detector.generate(skeleton, structural, fs), skeleton, structural, fs, {})
    graph = seg.place_graph_builder.build(skeleton, candidates, structural, frame_id=1)
    partition = seg.graph_partitioner.partition(graph, candidates, structural)
    assert partition.separator_map.shape == free.shape
    assert partition.seed_cells_by_label

