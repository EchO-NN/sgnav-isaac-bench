from __future__ import annotations

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorAnchorConfig, SeparatorCandidate
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators
from isaac_bench.mapping.online_roomseg.utils import conn
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    StableSeparatorMemory,
    build_step2_partition_cut_v16,
    validate_stable_separators_against_current_partition,
)


def _candidate(cid: int, p0: tuple[int, int], p1: tuple[int, int]) -> SeparatorCandidate:
    return SeparatorCandidate(
        candidate_id=int(cid),
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray(p0, dtype=np.float32),
        p1_rc=np.asarray(p1, dtype=np.float32),
        theta=0.0,
        length_m=float(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1) * 0.10,
        confidence=1.0,
        source_segment_ids=[int(cid)],
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
    )


def _tiny_side_free_map() -> np.ndarray:
    free = np.zeros((20, 24), dtype=bool)
    free[5:15, 5:20] = True
    free[8:10, 3:5] = True
    free[8:10, 5] = True
    return free


def test_tiny_width_split_rejects_separator_without_deleting_region() -> None:
    free = _tiny_side_free_map()
    before_free_cells = int(np.count_nonzero(free))
    candidate = _candidate(1, (8, 5), (9, 5))

    accepted, rejected, accepted_map, _labels, debug = greedily_select_separators(
        [candidate],
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        corridor_skeleton=np.zeros_like(free),
        resolution_m=0.10,
        config=_cfg(),
    )
    cut, _candidate_cut, _cut_debug = build_step2_partition_cut_v16(
        accepted,
        accepted_topology_map=accepted_map,
        base_partition_free=free,
        partition_unknown=np.zeros_like(free),
        real_wall_barrier=np.zeros_like(free),
        shape=free.shape,
    )
    final_labels, _ = ndimage.label(free & ~cut, structure=conn(4))

    assert not accepted
    assert len(rejected) == 1
    assert rejected[0].reject_reason == "reject_split_tiny_side_width_1_to_3_cells"
    assert debug["rejected_separator_reasons"]["reject_split_tiny_side_width_1_to_3_cells"] == 1
    assert not np.any(accepted_map)
    assert not np.any(cut)
    assert int(np.count_nonzero(free)) == before_free_cells
    assert np.all(final_labels[free] > 0)


def test_stable_separator_that_now_creates_tiny_side_is_pruned_not_applied() -> None:
    free = _tiny_side_free_map()
    memory = StableSeparatorMemory(ttl_updates=30, decay_per_update=0.02, min_confidence_to_keep=0.15)
    candidate = _candidate(1, (8, 5), (9, 5))
    candidate.accepted = True
    stable_raw, _debug = memory.update([candidate], step=1, shape=free.shape)

    stable_valid, validation_debug = validate_stable_separators_against_current_partition(
        memory,
        stable_raw,
        free_after_step1=free,
        accepted_separator_map=np.zeros_like(free),
        resolution_m=0.10,
        topology_config=_cfg(),
    )

    assert np.any(stable_raw)
    assert not np.any(stable_valid)
    assert validation_debug["voxel_stable_separator_tiny_reject_count"] == 1
    assert np.any(validation_debug["voxel_stable_separator_tiny_reject_map"])
    assert len(memory._tracks) == 0
