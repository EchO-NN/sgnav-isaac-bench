from __future__ import annotations

from typing import Dict

import numpy as np

from .data_types import RoomSegmentationOutput


PALETTE = np.asarray(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ],
    dtype=np.uint8,
)


def render_room_segmentation_debug(output: RoomSegmentationOutput) -> Dict[str, np.ndarray]:
    labels = np.asarray(output.room_id_map, dtype=np.int32)
    room_rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    room_rgb[labels < 0] = np.asarray([24, 24, 24], dtype=np.uint8)
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        room_rgb[labels == label] = PALETTE[(label - 1) % len(PALETTE)]
    confidence = np.clip(np.asarray(output.room_confidence_map, dtype=np.float32), 0.0, 1.0)
    confidence_rgb = np.stack(
        [
            (255.0 * (1.0 - confidence)).astype(np.uint8),
            (255.0 * confidence).astype(np.uint8),
            np.zeros_like(confidence, dtype=np.uint8),
        ],
        axis=-1,
    )
    cut_overlay = room_rgb.copy()
    hard = np.asarray(output.debug_layers.get("hard_separator_map", np.zeros(labels.shape)), dtype=bool)
    soft = np.asarray(output.debug_layers.get("soft_separator_map", np.zeros(labels.shape)), dtype=bool)
    skeleton = np.asarray(output.debug_layers.get("skeleton_mask", np.zeros(labels.shape)), dtype=bool)
    cut_overlay[soft] = np.asarray([255, 214, 10], dtype=np.uint8)
    cut_overlay[hard] = np.asarray([255, 0, 0], dtype=np.uint8)
    cut_overlay[skeleton] = np.asarray([255, 255, 255], dtype=np.uint8)
    corridor_open = room_rgb.copy()
    corridor_open[np.asarray(output.corridor_mask, dtype=bool)] = np.asarray([0, 190, 255], dtype=np.uint8)
    corridor_open[np.asarray(output.open_space_mask, dtype=bool)] = np.asarray([120, 220, 120], dtype=np.uint8)
    zone = np.asarray(output.functional_zone_map, dtype=np.int32)
    zone_rgb = np.zeros((*zone.shape, 3), dtype=np.uint8)
    for label in sorted(int(v) for v in np.unique(zone) if int(v) > 0):
        zone_rgb[zone == label] = PALETTE[(label - 1) % len(PALETTE)]
    return {
        "room_id_rgb": room_rgb,
        "confidence_heatmap": confidence_rgb,
        "skeleton_cut_overlay": cut_overlay,
        "corridor_open_space_overlay": corridor_open,
        "functional_zone_rgb": zone_rgb,
    }

