import numpy as np

from isaac_bench.mapping.room_context_overlay import build_navigation_free_room_context_overlay


def test_navigation_free_overlay_absorbs_free_but_not_unknown():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2:5, 1:3] = 1
    nav = np.zeros_like(labels, dtype=bool)
    nav[2:5, 1:5] = True
    unknown = np.zeros_like(nav)
    unknown[2:5, 5] = True

    context, reliability, debug = build_navigation_free_room_context_overlay(
        labels,
        nav,
        unknown,
        obstacle=None,
        structural_boundary=None,
        resolution_m=0.5,
        max_absorb_distance_m=1.0,
        min_seed_room_area_cells=1,
    )

    assert context[3, 4] == 1
    assert context[3, 5] == 0
    assert reliability[3, 4] > 0.0
    assert debug["absorbed_cells"] > 0


def test_navigation_free_overlay_does_not_cross_structural_boundary():
    labels = np.zeros((5, 7), dtype=np.int32)
    labels[1:4, 1:3] = 1
    nav = np.zeros_like(labels, dtype=bool)
    nav[1:4, 1:6] = True
    unknown = np.zeros_like(nav)
    boundary = np.zeros_like(nav)
    boundary[1:4, 3] = True

    context, _, _ = build_navigation_free_room_context_overlay(
        labels,
        nav,
        unknown,
        obstacle=None,
        structural_boundary=boundary,
        resolution_m=0.5,
        max_absorb_distance_m=2.0,
        min_seed_room_area_cells=1,
        do_not_cross_structural_boundary=True,
    )

    assert context[2, 2] == 1
    assert context[2, 4] == 0

