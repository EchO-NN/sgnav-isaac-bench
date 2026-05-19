from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from isaac_bench.mapping.rose2_upstream_parser import parse_rose2_room_output


def test_parse_thin_wall_two_rooms_does_not_cross_wall(tmp_path):
    free = np.zeros((40, 60), dtype=bool)
    free[5:35, 5:55] = True
    wall = np.zeros_like(free)
    wall[5:35, 30] = True
    free[wall] = False
    image = np.zeros((*free.shape, 3), dtype=np.uint8)
    image[free & (np.indices(free.shape)[1] < 30)] = (80, 140, 255)
    image[free & (np.indices(free.shape)[1] > 30)] = (255, 160, 80)
    path = _save(tmp_path, "two_rooms.png", image)

    labels, debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=~(free | wall),
        min_room_area_m2=0.05,
        resolution_m=0.10,
    )

    assert _count(labels) >= 2
    assert np.all(labels[wall] == 0)
    assert debug["labels_removed_outside_free"] == 0


def test_parse_door_opening_keeps_source_room_labels(tmp_path):
    free = np.zeros((40, 60), dtype=bool)
    free[5:35, 5:55] = True
    wall = np.zeros_like(free)
    wall[5:16, 30] = True
    wall[24:35, 30] = True
    free[wall] = False
    cc = np.indices(free.shape)[1]
    image = np.zeros((*free.shape, 3), dtype=np.uint8)
    image[free & (cc < 30)] = (80, 140, 255)
    image[free & (cc > 30)] = (255, 160, 80)
    image[free & (cc == 30)] = (120, 220, 130)
    path = _save(tmp_path, "door.png", image)

    labels, _debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=~(free | wall),
        min_room_area_m2=0.05,
        resolution_m=0.10,
    )

    assert _count(labels) >= 2
    assert np.all(labels[wall] == 0)


def test_parse_l_shaped_room_not_forced_into_rectangular_cells(tmp_path):
    free = np.zeros((45, 45), dtype=bool)
    free[5:38, 5:18] = True
    free[25:38, 5:38] = True
    image = np.zeros((*free.shape, 3), dtype=np.uint8)
    image[free] = (130, 222, 150)
    path = _save(tmp_path, "l_room.png", image)

    labels, _debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=~free,
        min_room_area_m2=0.05,
        resolution_m=0.10,
    )

    assert _count(labels) == 1
    assert np.array_equal(labels > 0, free)


def test_parse_three_rooms_plus_corridor_reports_multiple_rooms(tmp_path):
    free = np.zeros((70, 90), dtype=bool)
    free[30:40, 10:80] = True
    free[8:28, 8:28] = True
    free[8:28, 35:55] = True
    free[42:62, 60:82] = True
    image = np.zeros((*free.shape, 3), dtype=np.uint8)
    image[8:28, 8:28] = (80, 140, 255)
    image[8:28, 35:55] = (255, 160, 80)
    image[42:62, 60:82] = (130, 222, 150)
    image[30:40, 10:80] = (214, 148, 255)
    path = _save(tmp_path, "three_plus_corridor.png", image)

    labels, _debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=~free,
        min_room_area_m2=0.05,
        resolution_m=0.10,
    )

    assert _count(labels) >= 3


def test_parse_upstream_grayscale_cells_as_room_candidates(tmp_path):
    free = np.zeros((36, 54), dtype=bool)
    free[6:30, 6:24] = True
    free[6:30, 30:48] = True
    wall = np.zeros_like(free)
    wall[6:30, 27] = True
    unknown = ~(free | wall)
    image = np.full((*free.shape, 3), 255, dtype=np.uint8)
    image[wall] = (0, 0, 0)
    image[free & (np.indices(free.shape)[1] < 27)] = (164, 164, 164)
    image[free & (np.indices(free.shape)[1] > 27)] = (205, 205, 205)
    path = _save(tmp_path, "8a_cells_in_out_partial_th1.png", image)

    labels, debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=unknown,
        min_room_area_m2=0.05,
        resolution_m=0.10,
    )

    assert debug["foreground_mode"] == "grayscale_cells"
    assert _count(labels) >= 2
    assert np.all(labels[wall] == 0)


def test_parse_sparse_grayscale_cells_expand_inside_observed_free(tmp_path):
    free = np.zeros((40, 60), dtype=bool)
    free[6:34, 6:27] = True
    free[6:34, 33:54] = True
    wall = np.zeros_like(free)
    wall[6:34, 30] = True
    unknown = ~(free | wall)
    image = np.full((*free.shape, 3), 255, dtype=np.uint8)
    image[wall] = (0, 0, 0)
    image[12:18, 12:18] = (164, 164, 164)
    image[22:28, 42:48] = (205, 205, 205)
    path = _save(tmp_path, "sparse_8a_cells.png", image)

    labels, debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=unknown,
        min_room_area_m2=0.50,
        resolution_m=0.10,
    )

    assert debug["foreground_mode"] == "grayscale_cells"
    assert debug["grayscale_seed_expansion"] == "used"
    assert _count(labels) >= 2
    assert np.count_nonzero(labels > 0) > 1000
    assert np.all(labels[wall] == 0)
    assert np.all(labels[unknown] == 0)


def test_unknown_is_never_expanded_into_room_labels(tmp_path):
    free = np.zeros((30, 30), dtype=bool)
    free[5:20, 5:20] = True
    unknown = ~free
    image = np.zeros((*free.shape, 3), dtype=np.uint8)
    image[5:25, 5:25] = (80, 140, 255)
    path = _save(tmp_path, "unknown_expansion.png", image)

    labels, debug = parse_rose2_room_output(
        output_png=path,
        observed_free=free,
        unknown=unknown,
        min_room_area_m2=0.05,
        resolution_m=0.10,
    )

    assert np.all(labels[unknown] == 0)
    assert debug["labels_removed_outside_free"] > 0


def _save(tmp_path: Path, name: str, image: np.ndarray) -> Path:
    path = tmp_path / name
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)
    return path


def _count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(labels) if int(v) > 0]))
