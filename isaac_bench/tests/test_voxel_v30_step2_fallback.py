from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorCandidate
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import _apply_corridor_local_acceptance


def _candidate(reason: str = "reject_no_two_sides") -> SeparatorCandidate:
    return SeparatorCandidate(
        candidate_id=1,
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray([4, 15], dtype=np.float32),
        p1_rc=np.asarray([19, 15], dtype=np.float32),
        theta=float(np.pi / 2.0),
        length_m=1.60,
        confidence=1.0,
        source_segment_ids=[1],
        accepted=False,
        reject_reason=reason,
    )


def test_v30_step2_local_acceptance_handles_common_topology_reject_reasons() -> None:
    shape = (24, 30)
    free = np.zeros(shape, dtype=bool)
    free[4:20, 4:26] = True
    target_wall = np.zeros(shape, dtype=bool)
    target_wall[3, 15] = True
    target_wall[20, 15] = True
    visual_only_door = np.zeros(shape, dtype=bool)
    visual_only_door[10, 10:14] = True

    accepted, rejected, accepted_map, debug = _apply_corridor_local_acceptance(
        [],
        [_candidate("reject_no_two_sides")],
        np.zeros(shape, dtype=bool),
        free_after_step1=free,
        target_wall_map=target_wall,
        door_block_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
    )

    assert len(accepted) == 1
    assert not rejected
    assert np.any(accepted_map)
    assert debug["voxel_step2_local_neck_v30_accepted_count"] == 1
    assert accepted[0].debug["corridor_local_acceptance_reason"] == "accepted_wall_to_wall_neck_v30"
    assert not np.any(accepted_map & visual_only_door)


def test_v30_step2_local_acceptance_still_rejects_verified_door_intersection() -> None:
    shape = (24, 30)
    free = np.zeros(shape, dtype=bool)
    free[4:20, 4:26] = True
    target_wall = np.zeros(shape, dtype=bool)
    target_wall[3, 15] = True
    target_wall[20, 15] = True
    verified_door_block = _candidate().mask(shape)

    accepted, rejected, accepted_map, debug = _apply_corridor_local_acceptance(
        [],
        [_candidate("reject_side_area_too_small")],
        np.zeros(shape, dtype=bool),
        free_after_step1=free,
        target_wall_map=target_wall,
        door_block_mask=verified_door_block,
        resolution_m=0.10,
    )

    assert not accepted
    assert len(rejected) == 1
    assert not np.any(accepted_map)
    assert debug["voxel_step2_local_neck_v30_reject_counts"]["intersects_verified_door"] == 1
