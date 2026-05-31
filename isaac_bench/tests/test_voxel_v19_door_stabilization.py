from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    DoorSeedCluster,
    VoxelDoorDetectorConfig,
    VoxelDoorLineCandidate,
    VoxelDoorMemory,
    _door_orientation_candidates,
    validate_door_line_local_neck,
)


def test_v19_default_orientation_does_not_add_unconditional_diagonals() -> None:
    shape = (12, 12)
    mask = np.zeros(shape, dtype=bool)
    mask[6, 6] = True
    cluster = DoorSeedCluster(
        cluster_id=1,
        component_ids=[1],
        seed_cells=[(6, 6)],
        mask=mask,
        bbox_rc=(6, 6, 6, 6),
        center_rc=(6.0, 6.0),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(-1.0, 0.0),
        line_fit_residual_cells=0.0,
        thickness_m=0.10,
        length_m=0.10,
        accepted_for_completion=True,
        reject_reason=None,
    )

    sources = [source for source, _vec in _door_orientation_candidates(cluster, VoxelDoorDetectorConfig())]

    assert sources == ["seed_major"]
    assert "axis_h" not in sources
    assert "axis_v" not in sources
    assert "diag_down" not in sources
    assert "diag_up" not in sources


def test_v19_door_neck_rejects_unsupported_diagonal_flying_line_without_width_gate() -> None:
    shape = (32, 32)
    line = [(i, i) for i in range(4, 25)]
    seed = np.zeros(shape, dtype=bool)
    seed[4, 4] = True
    free = np.ones(shape, dtype=bool)
    wall = np.zeros(shape, dtype=bool)

    ok, reason, debug = validate_door_line_local_neck(
        line,
        seed,
        free,
        wall,
        wall,
        0.10,
        VoxelDoorDetectorConfig(),
        completion_mode="middle_seed_two_wall",
        orientation_source="diag_down",
    )

    assert not ok
    assert reason == "door_diagonal_without_support"
    assert debug["door_line_width_limit_enforced"] is False
    assert debug["door_line_max_width_m"] == 1.80


def test_v19_door_memory_keeps_only_partition_accepted_cuts() -> None:
    shape = (8, 8)
    visual_only = VoxelDoorLineCandidate(
        candidate_id=1,
        seed_component_id=1,
        seed_cells=[(3, 3)],
        center_rc=(3.0, 3.0),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(-1.0, 0.0),
        seed_projected_centerline_cells=[(3, 3)],
        extended_centerline_cells=[(3, 3), (3, 4)],
        door_cut_cells=[],
        wall_anchor_a=None,
        wall_anchor_b=None,
        width_m=0.20,
        accepted=True,
        reject_reason=None,
        debug={"partition_accepted": False},
    )
    accepted = VoxelDoorLineCandidate(
        candidate_id=2,
        seed_component_id=2,
        seed_cells=[(5, 2)],
        center_rc=(5.0, 2.0),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(-1.0, 0.0),
        seed_projected_centerline_cells=[(5, 2)],
        extended_centerline_cells=[(5, 2), (5, 3), (5, 4)],
        door_cut_cells=[(5, 3)],
        wall_anchor_a=(5, 2),
        wall_anchor_b=(5, 4),
        width_m=0.30,
        accepted=True,
        reject_reason=None,
        debug={"partition_accepted": True},
    )

    result = VoxelDoorMemory().update([visual_only, accepted], step=1, shape=shape)

    assert result.debug["voxel_door_memory_current_candidate_count"] == 1
    assert result.debug["voxel_door_memory_rejected_visual_only_count"] == 1
    assert result.stable_door_cut_mask[5, 3]
    assert not result.stable_door_visual_mask[3, 4]
