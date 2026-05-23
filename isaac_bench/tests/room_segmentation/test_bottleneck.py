import numpy as np

from .test_helpers import run_masks, two_rooms_one_door


def test_door_generates_scored_cut_candidate():
    free, wall, unknown = two_rooms_one_door(gap_rows=(35, 45))
    out = run_masks(free, wall, unknown)
    assert out.cut_candidates
    assert max(c.final_score for c in out.cut_candidates) >= 0.65
    assert np.count_nonzero(out.debug_layers["soft_separator_map"]) > 0


def test_wide_door_is_kept_as_soft_candidate():
    free, wall, unknown = two_rooms_one_door(gap_rows=(30, 50))
    out = run_masks(free, wall, unknown)
    assert out.cut_candidates
    assert any(c.width_m > 0.8 and c.final_score >= 0.55 for c in out.cut_candidates)


def test_frontier_candidate_is_conservative():
    free, wall, unknown = two_rooms_one_door(gap_rows=(35, 45))
    unknown[30:50, 45:55] = True
    free[30:50, 45:55] = False
    wall[30:50, 45:55] = False
    out = run_masks(free, wall, unknown)
    if out.cut_candidates:
        assert max(c.final_score for c in out.cut_candidates) <= 0.75

