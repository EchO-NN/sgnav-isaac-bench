from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorAnchorConfig, SeparatorCandidate
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators


def _candidate(cid: int, p0: tuple[int, int], p1: tuple[int, int]) -> SeparatorCandidate:
    p0_arr = np.asarray(p0, dtype=np.float32)
    p1_arr = np.asarray(p1, dtype=np.float32)
    return SeparatorCandidate(
        candidate_id=cid,
        kind="line_extension_corridor_separator",
        p0_rc=p0_arr,
        p1_rc=p1_arr,
        theta=0.0,
        length_m=float(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1) * 0.10,
        confidence=1.0,
        source_segment_ids=[cid],
    )


def test_step2_corridor_separator_uses_relaxed_corridor_thresholds() -> None:
    shape = (24, 30)
    free = np.zeros(shape, dtype=bool)
    free[4:20, 4:26] = True
    wall = np.zeros(shape, dtype=bool)
    wall[3, 15] = True
    wall[20, 15] = True

    accepted, rejected, accepted_map, _labels, debug = greedily_select_separators(
        [_candidate(1, (4, 15), (19, 15))],
        free_clean=free,
        unknown_clean=np.zeros(shape, dtype=bool),
        wall_candidate_clean=wall,
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=TopologyTestConfig(
            min_split_area_m2=1.0,
            corridor_min_split_area_m2=0.05,
            corridor_min_new_component_width_m=0.10,
            corridor_reject_tiny_side_width_cells_leq=3,
            reject_small_known_side_for_line_extensions=False,
        ),
    )

    assert len(accepted) == 1
    assert not rejected
    assert np.any(accepted_map)
    assert debug["accepted_separator_kinds"]["line_extension_corridor_separator"] == 1


def test_step2_corridor_separator_still_rejects_true_tiny_sliver() -> None:
    shape = (20, 24)
    free = np.zeros(shape, dtype=bool)
    free[5:15, 5:20] = True
    free[8:10, 3:5] = True
    free[8:10, 5] = True

    accepted, rejected, _accepted_map, _labels, _debug = greedily_select_separators(
        [_candidate(1, (8, 5), (9, 5))],
        free_clean=free,
        unknown_clean=np.zeros(shape, dtype=bool),
        wall_candidate_clean=np.zeros(shape, dtype=bool),
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=TopologyTestConfig(
            separator=SeparatorAnchorConfig(require_two_anchors=False),
            corridor_min_split_area_m2=0.01,
            corridor_min_new_component_width_m=0.10,
            corridor_reject_tiny_side_width_cells_leq=3,
            corridor_tiny_side_min_area_m2=0.08,
            corridor_tiny_side_min_length_m=0.25,
            reject_small_known_side_for_line_extensions=False,
        ),
    )

    assert not accepted
    assert len(rejected) == 1
    assert rejected[0].reject_reason == "reject_split_tiny_side_width_1_to_3_cells"
