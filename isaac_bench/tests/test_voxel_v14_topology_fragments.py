from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorAnchorConfig, SeparatorCandidate
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators


def _candidate() -> SeparatorCandidate:
    return SeparatorCandidate(
        candidate_id=1,
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray([5, 10], dtype=np.float32),
        p1_rc=np.asarray([24, 10], dtype=np.float32),
        theta=float(np.pi / 2.0),
        length_m=2.0,
        confidence=0.9,
        source_segment_ids=[1],
    )


def _free_with_remote_tiny_islands() -> np.ndarray:
    free = np.zeros((30, 40), dtype=bool)
    free[5:25, 5:16] = True
    for idx, cell in enumerate([(0, 0), (0, 4), (0, 8), (0, 12), (0, 16), (29, 25), (29, 29), (29, 33)]):
        free[cell] = True
        if idx % 2 == 0:
            free[cell[0], min(39, cell[1] + 1)] = True
    return free


def test_topology_tiny_fragments_uses_delta_not_absolute() -> None:
    free = _free_with_remote_tiny_islands()
    cfg = TopologyTestConfig(
        min_split_area_m2=0.05,
        per_kind_min_split_area_m2={"line_extension_corridor_separator": 0.05},
        min_new_component_area_cells=4,
        max_tiny_fragment_count=3,
        tiny_fragment_count_mode="local_delta",
        local_fragment_radius_cells=2,
        reject_corridor_split=False,
        reject_open_living_room_internal_split=False,
        reject_if_side_width_cells_leq=0,
        reject_small_known_side_for_line_extensions=False,
        separator=SeparatorAnchorConfig(require_two_anchors=False),
    )

    accepted, rejected, accepted_map, _labels, debug = greedily_select_separators(
        [_candidate()],
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        corridor_skeleton=np.zeros_like(free),
        resolution_m=0.10,
        config=cfg,
    )

    assert len(accepted) == 1
    assert not rejected
    assert np.any(accepted_map)
    stats = accepted[0].debug["topology_fragment_stats"]
    assert stats["after_tiny_count"] > cfg.max_tiny_fragment_count
    assert stats["local_new_tiny_count"] == 0
    assert debug["tiny_fragment_count_mode"] == "local_delta"


def test_topology_absolute_tiny_fragment_mode_still_rejects_for_compatibility() -> None:
    free = _free_with_remote_tiny_islands()
    cfg = TopologyTestConfig(
        min_split_area_m2=0.05,
        per_kind_min_split_area_m2={"line_extension_corridor_separator": 0.05},
        min_new_component_area_cells=4,
        max_tiny_fragment_count=3,
        tiny_fragment_count_mode="absolute",
        reject_corridor_split=False,
        reject_open_living_room_internal_split=False,
        reject_if_side_width_cells_leq=0,
        reject_small_known_side_for_line_extensions=False,
        separator=SeparatorAnchorConfig(require_two_anchors=False),
    )

    accepted, rejected, _accepted_map, _labels, _debug = greedily_select_separators(
        [_candidate()],
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        corridor_skeleton=np.zeros_like(free),
        resolution_m=0.10,
        config=cfg,
    )

    assert not accepted
    assert rejected[0].reject_reason == "reject_too_many_tiny_fragments"
