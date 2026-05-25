import numpy as np

from isaac_bench.mapping.roomseg_input_sanitizer import (
    RoomsegInputSanitizerConfig,
    sanitize_roomseg_inputs,
)


def test_sanitized_vertical_free_subset_of_navigation_free():
    nav = np.zeros((20, 20), dtype=bool)
    nav[5:15, 5:15] = True
    raw_free = nav.copy()
    raw_free[2:5, 5:15] = True
    wall = np.zeros_like(nav)
    unknown = ~nav

    free, _wall_out, _unknown_out, debug = sanitize_roomseg_inputs(
        vertical_free_raw=raw_free,
        wall_raw=wall,
        unknown_raw=unknown,
        navigation_free_mask=nav,
        cfg=RoomsegInputSanitizerConfig(),
    )

    assert np.all(free <= nav)
    assert debug["vertical_free_outside_navigation_cells"] > 0
    assert debug["sanitized_free_subset_navigation_ok"] is True


def test_wall_core_not_erased_by_vertical_free_overlap():
    nav = np.ones((20, 20), dtype=bool)
    raw_free = np.zeros_like(nav)
    raw_free[:, 10] = True
    wall = np.zeros_like(nav)
    wall[:, 10] = True
    unknown = np.zeros_like(nav)

    free, wall_out, _unknown_out, debug = sanitize_roomseg_inputs(
        vertical_free_raw=raw_free,
        wall_raw=wall,
        unknown_raw=unknown,
        navigation_free_mask=nav,
        cfg=RoomsegInputSanitizerConfig(wall_core_min_component_cells=1),
    )

    assert np.count_nonzero(wall_out[:, 10]) > 0
    assert np.count_nonzero(free & wall_out) == 0
    assert debug["free_wall_conflict_cells_before_sanitize"] > 0


def test_terminal_wall_count_enters_roomseg_wall_mask():
    nav = np.ones((20, 20), dtype=bool)
    raw_free = nav.copy()
    wall = np.zeros_like(nav)
    unknown = np.zeros_like(nav)
    terminal = np.zeros_like(nav, dtype=np.uint16)
    terminal[5:15, 10] = 1

    free, wall_out, _unknown_out, debug = sanitize_roomseg_inputs(
        vertical_free_raw=raw_free,
        wall_raw=wall,
        unknown_raw=unknown,
        navigation_free_mask=nav,
        roomseg_ray_evidence={"terminal_wall_count": terminal},
        cfg=RoomsegInputSanitizerConfig(wall_core_min_component_cells=1, terminal_wall_splat_radius_cells=0),
    )

    assert np.count_nonzero(wall_out[:, 10]) > 0
    assert np.count_nonzero(free & wall_out) == 0
    assert debug["terminal_wall_used_as_roomseg_wall"] is True
