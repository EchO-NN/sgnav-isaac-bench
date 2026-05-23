import numpy as np

from .test_helpers import make_segmenter, open_room


def test_free_space_outputs_frontier_and_distance_without_unknown_walls():
    free, wall, unknown = open_room()
    unknown[8:14, 30:50] = True
    free[8:14, 30:50] = False
    seg = make_segmenter(free.shape)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=wall, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    fs = seg.free_space_extractor.extract(structural)
    assert fs.observed_free_mask.shape == free.shape
    assert np.count_nonzero(fs.frontier_mask) > 0
    assert float(np.max(fs.distance_transform_m)) > 0.0
    assert not np.any(fs.observed_free_mask & unknown)

