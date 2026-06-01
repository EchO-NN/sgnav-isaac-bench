from __future__ import annotations

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorAnchorConfig, SeparatorCandidate
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators
from isaac_bench.mapping.online_roomseg.utils import conn
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import build_step2_partition_cut_v16


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


def _cfg() -> TopologyTestConfig:
    return TopologyTestConfig(
        separator=SeparatorAnchorConfig(require_two_anchors=False),
        corridor_min_split_area_m2=0.03,
        corridor_min_new_component_width_m=0.10,
        corridor_reject_tiny_side_width_cells_leq=2,
        corridor_tiny_side_min_area_m2=0.03,
        corridor_tiny_side_min_length_m=0.35,
        corridor_accept_long_narrow_side=True,
        corridor_local_topology_radius_cells=20,
        reject_small_known_side_for_line_extensions=False,
    )


def test_v16_narrow_long_corridor_side_is_not_tiny_sliver_rejected() -> None:
    shape = (24, 34)
    free = np.zeros(shape, dtype=bool)
    free[4:20, 5:22] = True
    free[10:12, 22:32] = True

    accepted, rejected, accepted_map, labels, debug = greedily_select_separators(
        [_candidate(1, (10, 22), (11, 22))],
        free_clean=free,
        unknown_clean=np.zeros(shape, dtype=bool),
        wall_candidate_clean=np.zeros(shape, dtype=bool),
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_cfg(),
    )

    assert len(accepted) == 1
    assert np.any(accepted_map)
    assert int(labels.max()) >= 2
    assert "reject_split_tiny_side_width_1_to_3_cells" not in debug["rejected_separator_reasons"]
    assert accepted[0].debug["corridor_topology_v16"] is True
    assert accepted[0].debug["corridor_long_narrow_side_accepted"] is True
    assert not rejected


def test_v16_short_tiny_corridor_fragment_is_still_rejected() -> None:
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
        config=_cfg(),
    )

    assert not accepted
    assert len(rejected) == 1
    assert rejected[0].reject_reason in {"reject_split_tiny_side_width_1_to_3_cells", "reject_tiny_split"}


def test_v16_step2_accepted_cut_participates_final_partition() -> None:
    shape = (24, 34)
    free = np.zeros(shape, dtype=bool)
    free[4:20, 5:22] = True
    free[10:12, 22:32] = True
    accepted, _rejected, accepted_map, _labels, _debug = greedily_select_separators(
        [_candidate(1, (10, 22), (11, 22))],
        free_clean=free,
        unknown_clean=np.zeros(shape, dtype=bool),
        wall_candidate_clean=np.zeros(shape, dtype=bool),
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=_cfg(),
    )

    cut, candidate_cut, cut_debug = build_step2_partition_cut_v16(
        accepted,
        accepted_topology_map=accepted_map,
        base_partition_free=free,
        partition_unknown=np.zeros(shape, dtype=bool),
        real_wall_barrier=np.zeros(shape, dtype=bool),
        shape=shape,
    )
    labels, count = ndimage.label(free & ~cut, structure=conn(4))

    assert np.any(candidate_cut)
    assert np.any(cut)
    assert cut_debug["step2_partition_cut_v16"] is True
    assert int(count) >= 2
    assert int(labels.max()) >= 2


def test_line_extension_rejects_small_known_side_when_boundary_unknown_low() -> None:
    shape = (32, 32)
    free = np.zeros(shape, dtype=bool)
    free[5:25, 5:25] = True

    cfg = TopologyTestConfig(
        separator=SeparatorAnchorConfig(require_two_anchors=False),
        corridor_min_split_area_m2=0.01,
        corridor_reject_tiny_side_width_cells_leq=0,
        reject_small_known_side_for_line_extensions=True,
        small_known_side_area_m2=2.0,
        small_known_side_unknown_ratio_max=0.20,
    )
    accepted, rejected, _accepted_map, _labels, _debug = greedily_select_separators(
        [_candidate(1, (5, 8), (24, 8))],
        free_clean=free,
        unknown_clean=np.zeros(shape, dtype=bool),
        wall_candidate_clean=np.zeros(shape, dtype=bool),
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
    )

    assert not accepted
    assert len(rejected) == 1
    assert rejected[0].reject_reason == "reject_small_known_side_low_unknown"
    assert rejected[0].debug["small_known_side_rejected"] is True


def test_line_extension_keeps_small_side_when_boundary_unknown_high() -> None:
    shape = (32, 32)
    free = np.zeros(shape, dtype=bool)
    free[5:25, 5:25] = True
    unknown = np.zeros(shape, dtype=bool)
    unknown[4, 4:9] = True
    unknown[25, 4:9] = True
    unknown[5:25, 4] = True

    cfg = TopologyTestConfig(
        separator=SeparatorAnchorConfig(require_two_anchors=False),
        corridor_min_split_area_m2=0.01,
        corridor_reject_tiny_side_width_cells_leq=0,
        reject_small_known_side_for_line_extensions=True,
        small_known_side_area_m2=2.0,
        small_known_side_unknown_ratio_max=0.20,
    )
    accepted, rejected, accepted_map, _labels, _debug = greedily_select_separators(
        [_candidate(1, (5, 8), (24, 8))],
        free_clean=free,
        unknown_clean=unknown,
        wall_candidate_clean=np.zeros(shape, dtype=bool),
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
    )

    assert len(accepted) == 1
    assert not rejected
    assert np.any(accepted_map)
    assert accepted[0].debug["small_known_side_rejected"] is False
