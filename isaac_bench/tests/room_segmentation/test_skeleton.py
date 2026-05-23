import numpy as np

from .test_helpers import make_segmenter, corridor_with_two_rooms, open_room, two_rooms_one_door


def _skeleton(free, wall, unknown):
    seg = make_segmenter(free.shape)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=wall, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    fs = seg.free_space_extractor.extract(structural)
    return seg.skeleton_extractor.extract(fs, structural), fs


def test_rectangular_room_skeleton_nonempty_and_center_clearance_large():
    free, wall, unknown = open_room()
    graph, fs = _skeleton(free, wall, unknown)
    assert len(graph.nodes) > 0
    assert max(node.clearance_m for node in graph.nodes) >= 0.5 * np.max(fs.distance_transform_m)


def test_doorway_or_corridor_has_low_clearance_skeleton_node():
    free, wall, unknown = two_rooms_one_door()
    graph, _fs = _skeleton(free, wall, unknown)
    assert len(graph.nodes) > 0
    assert min(2.0 * node.clearance_m for node in graph.nodes) <= 1.8


def test_corridor_skeleton_is_chain_like():
    free, wall, unknown = corridor_with_two_rooms()
    graph, _fs = _skeleton(free, wall, unknown)
    assert len(graph.nodes) > 0
    degree2 = sum(1 for node in graph.nodes if node.degree == 2)
    assert degree2 / max(1, len(graph.nodes)) >= 0.25

