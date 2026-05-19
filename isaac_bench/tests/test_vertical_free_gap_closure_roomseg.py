import numpy as np

from isaac_bench.mapping.vertical_free_gap_closure_roomseg import (
    VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
    VERTICAL_FREE_GAP_CLOSURE_BACKEND,
    VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
    VFGCConfig,
    run_vertical_free_gap_closure_roomseg,
    vfgc_result_to_source_result,
)


def _cfg(**overrides):
    base = dict(
        resolution_m=0.1,
        close_max_gap_m=1.5,
        close_min_gap_m=0.1,
        virtual_boundary_radius_m=0.06,
        wall_min_component_cells=1,
        free_min_component_cells=1,
        wall_micro_close_radius_cells=0,
        side_support_min_free_cells=3,
        side_support_min_ratio=0.10,
        side_support_balance_min=0.10,
        line_free_ratio_min=0.50,
        line_unknown_ratio_max=0.05,
        line_mid_wall_ratio_max=0.25,
        candidate_score_min=0.05,
        min_room_area_m2=0.50,
        small_component_area_m2=0.10,
    )
    base.update(overrides)
    return VFGCConfig(**base)


def _two_rooms_with_vertical_gap(gap_rows=(35, 45), *, unknown_gap=False):
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:110] = True
    wall = np.zeros_like(free)
    wall[10:70, 59:61] = True
    free[wall] = False
    gr0, gr1 = gap_rows
    wall[gr0:gr1, 59:61] = False
    free[gr0:gr1, 59:61] = True
    unknown = np.zeros_like(free)
    if unknown_gap:
        unknown[gr0:gr1, 59:61] = True
    return free, wall, unknown


def _room_count(result):
    return int(result.debug["num_rooms_final"])


def test_gap_closure_closes_short_wall_gap_into_two_rooms():
    free, wall, unknown = _two_rooms_with_vertical_gap((35, 45))

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
    )

    assert result.debug["backend"] == VERTICAL_FREE_GAP_CLOSURE_BACKEND
    assert result.debug["algorithm"] == VERTICAL_FREE_GAP_CLOSURE_ALGORITHM
    assert result.debug["context_source"] == VERTICAL_FREE_GAP_CLOSURE_CONTEXT
    assert result.debug["num_accepted_closures"] >= 1
    assert _room_count(result) == 2
    assert np.count_nonzero(result.virtual_boundary_map & ~wall) > 0


def test_gap_closure_does_not_close_wide_open_plan_gap():
    free, wall, unknown = _two_rooms_with_vertical_gap((30, 52))

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
    )

    assert result.debug["num_accepted_closures"] == 0
    assert _room_count(result) == 1


def test_gap_closure_rejects_unknown_window_like_gap():
    free, wall, unknown = _two_rooms_with_vertical_gap((35, 45), unknown_gap=True)

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
    )

    assert result.debug["num_accepted_closures"] == 0
    assert "line_crosses_unknown" in result.debug["rejection_reasons"]
    assert int(np.count_nonzero((result.room_label_map > 0) & unknown)) == 0


def test_gap_closure_ignores_one_sided_exterior_wall_gap():
    free = np.zeros((80, 120), dtype=bool)
    free[15:65, 20:70] = True
    wall = np.zeros_like(free)
    wall[15:65, 70:72] = True
    wall[36:44, 70:72] = False
    free[36:44, 70:72] = True
    unknown = np.zeros_like(free)

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
    )

    assert result.debug["num_accepted_closures"] == 0
    assert set(result.debug["rejection_reasons"]) & {
        "one_sided_free_support_corner_like",
        "not_between_two_meaningful_rooms",
    }


def test_gap_closure_keeps_virtual_boundaries_out_of_planner_wall_map():
    free, wall, unknown = _two_rooms_with_vertical_gap((35, 45))
    wall_before = wall.copy()

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
    )

    assert np.array_equal(wall, wall_before)
    assert np.count_nonzero(result.accepted_closure_map) > 0
    assert np.count_nonzero(result.accepted_closure_map & ~wall_before) > 0


def test_gap_closure_source_result_contract():
    free, wall, unknown = _two_rooms_with_vertical_gap((35, 45))
    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
    )

    source_result = vfgc_result_to_source_result(result)

    assert source_result.backend == VERTICAL_FREE_GAP_CLOSURE_BACKEND
    assert source_result.room_label_map.shape == free.shape
    assert source_result.debug["source_root_required"] is False
    assert source_result.debug["silent_fallback_used"] is False
