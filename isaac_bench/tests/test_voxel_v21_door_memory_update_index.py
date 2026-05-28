from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import VoxelDoorDetectorConfig, VoxelDoorLineCandidate, VoxelDoorMemory


def _accepted_candidate() -> VoxelDoorLineCandidate:
    return VoxelDoorLineCandidate(
        candidate_id=1,
        seed_component_id=1,
        seed_cells=[(5, 5)],
        center_rc=(5.0, 5.0),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(-1.0, 0.0),
        seed_projected_centerline_cells=[(5, 5)],
        extended_centerline_cells=[(5, 4), (5, 5), (5, 6)],
        door_cut_cells=[(5, 5)],
        wall_anchor_a=(5, 4),
        wall_anchor_b=(5, 6),
        width_m=0.30,
        accepted=True,
        reject_reason=None,
        debug={"partition_accepted": True},
    )


def test_v21_door_memory_ttl_uses_roomseg_update_index_not_sim_step() -> None:
    shape = (12, 12)
    cfg = VoxelDoorDetectorConfig(door_memory_ttl_updates=20, door_memory_decay_per_update=0.05)
    memory = VoxelDoorMemory(cfg)

    first = memory.update([_accepted_candidate()], step=10, update_index=1, shape=shape)
    second = memory.update([], step=100, update_index=2, shape=shape)

    assert np.any(first.stable_door_cut_mask)
    assert np.array_equal(first.stable_door_cut_mask, second.stable_door_cut_mask)
    assert second.debug["voxel_door_memory_track_count"] == 1
    assert second.debug["voxel_door_memory_update_index"] == 2
    assert second.debug["voxel_door_memory_missed_update_counts"] == [1]


def test_v21_door_memory_prunes_only_after_update_ttl_expires() -> None:
    shape = (12, 12)
    cfg = VoxelDoorDetectorConfig(door_memory_ttl_updates=2, door_memory_decay_per_update=0.0)
    memory = VoxelDoorMemory(cfg)

    memory.update([_accepted_candidate()], step=10, update_index=1, shape=shape)
    kept = memory.update([], step=1000, update_index=3, shape=shape)
    pruned = memory.update([], step=1001, update_index=4, shape=shape)

    assert kept.debug["voxel_door_memory_track_count"] == 1
    assert pruned.debug["voxel_door_memory_track_count"] == 0
    assert pruned.debug["voxel_door_memory_prune_reason_counts"]["ttl_updates_exceeded"] == 1
