import numpy as np

from src.room_segmentation import GridSpec
from src.room_segmentation.config import OVBConfig
from src.room_segmentation.vertical_map_builder import VerticalMapBuilder


def _builder():
    cfg = OVBConfig()
    return VerticalMapBuilder(cfg.map, cfg.depth, cfg.vertical_projection, GridSpec(0.05, (0.0, 0.0), 40, 40))


def test_mask_projection_keeps_unknown_without_free_as_uncertain():
    b = _builder()
    free = np.zeros((40, 40), dtype=bool)
    occ = np.zeros_like(free)
    unknown = np.ones_like(free)
    occ[10, 10] = True
    unknown[10, 10] = False
    state = b.from_masks(observed_free_mask=free, obstacle_mask=occ, unknown_mask=unknown, frame_id=1)
    assert state.p_occupied[10, 10] > 0.8
    assert state.p_unknown[0, 0] > 0.8
    assert state.p_unknown[10, 10] < 0.2


def test_depth_ray_updates_endpoint_and_clips_log_odds():
    b = _builder()
    depth = np.full((8, 8), 1.0, dtype=np.float32)
    k = np.asarray([[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]])
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 1.0
    pose[1, 3] = 1.0
    for frame in range(4):
        state = b.update_from_depth(depth, k, pose, frame + 1)
    assert np.max(state.p_endpoint) > 0.5
    assert np.max(np.abs(b.occupied_log_odds)) <= b.projection_config.max_log_odds

