from __future__ import annotations

import numpy as np

from src.room_segmentation import GridSpec, OVBConfig, OnlineRoomSegmenter


def make_segmenter(shape=(80, 100), overrides=None):
    cfg = OVBConfig.from_mapping(overrides or {})
    grid = GridSpec(float(cfg.map.resolution_m), (0.0, 0.0), int(shape[1]), int(shape[0]))
    return OnlineRoomSegmenter(cfg, grid)


def two_rooms_one_door(shape=(80, 100), gap_rows=(35, 45)):
    free = np.zeros(shape, dtype=bool)
    free[10 : shape[0] - 10, 10 : shape[1] - 10] = True
    wall = np.zeros(shape, dtype=bool)
    mid = shape[1] // 2
    wall[10 : shape[0] - 10, mid - 1 : mid + 1] = True
    free[:, mid - 1 : mid + 1] = False
    free[gap_rows[0] : gap_rows[1], mid - 1 : mid + 1] = True
    wall[gap_rows[0] : gap_rows[1], mid - 1 : mid + 1] = False
    unknown = ~(free | wall)
    return free, wall, unknown


def open_room(shape=(70, 90)):
    free = np.zeros(shape, dtype=bool)
    free[8:-8, 8:-8] = True
    wall = np.zeros(shape, dtype=bool)
    unknown = ~(free | wall)
    return free, wall, unknown


def corridor_with_two_rooms(shape=(90, 120)):
    free = np.zeros(shape, dtype=bool)
    wall = np.zeros(shape, dtype=bool)
    free[38:52, 10:110] = True
    free[15:38, 18:42] = True
    free[52:75, 78:102] = True
    wall[14:76, 17:18] = True
    wall[14:76, 42:43] = True
    wall[14:76, 77:78] = True
    wall[14:76, 102:103] = True
    free[35:45, 17:18] = True
    wall[35:45, 17:18] = False
    free[45:55, 102:103] = True
    wall[45:55, 102:103] = False
    unknown = ~(free | wall)
    return free, wall, unknown


def run_masks(free, wall, unknown, frames=1, overrides=None, objects=None):
    seg = make_segmenter(free.shape, overrides=overrides)
    out = None
    for frame in range(frames):
        out = seg.update_from_masks(free, wall, unknown, frame_id=frame + 1, semantic_objects=objects or [])
    assert out is not None
    return out

