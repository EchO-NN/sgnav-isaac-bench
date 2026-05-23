import numpy as np

from .test_helpers import make_segmenter


def test_long_wall_high_short_object_low_and_unknown_not_wall():
    seg = make_segmenter((60, 80))
    free = np.zeros((60, 80), dtype=bool)
    free[8:52, 8:72] = True
    wall = np.zeros_like(free)
    wall[12:48, 39:41] = True
    free[wall] = False
    wall[20:22, 20:22] = True
    free[20:22, 20:22] = False
    unknown = ~(free | wall)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=wall, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    assert float(np.max(structural.p_wall[12:48, 39:41])) > 0.8
    assert float(np.max(structural.p_wall[20:22, 20:22])) < 0.75
    assert not np.any(structural.hard_wall_mask[unknown])


def test_two_side_free_furniture_is_suppressed_relative_to_line_wall():
    seg = make_segmenter((50, 70))
    free = np.zeros((50, 70), dtype=bool)
    free[5:45, 5:65] = True
    wall = np.zeros_like(free)
    wall[8:42, 10:12] = True
    furniture = np.zeros_like(free)
    furniture[20:28, 34:36] = True
    obstacle = wall | furniture
    free[obstacle] = False
    unknown = ~(free | obstacle)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=obstacle, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    assert float(np.mean(structural.p_wall[wall])) > float(np.mean(structural.p_wall[furniture]))

