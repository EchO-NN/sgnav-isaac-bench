from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.wall_lines import filtered_wall_line_mask
from isaac_bench.mapping.voxel_door_detector import VoxelDoorDetectorConfig, VoxelDoorSeedResult, complete_voxel_doors_from_seeds
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    StableSeparatorMemory,
    VoxelOccupancyDoorWallRoomSegConfig,
    accepted_seed_mask_from_candidates,
    build_step2_line_pool,
    projected_wall_line_to_filtered_wall_line,
)
from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorCandidate
from isaac_bench.mapping.wall_projection import ProjectedWallLine
from isaac_bench.scripts.run_one_episode import FrontierRoomsegUpdateGateState, should_update_roomseg_frontiers


def _seed_result(shape: tuple[int, int], cells: list[tuple[int, int]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for r, c in cells:
        seed[int(r), int(c)] = True
        labels[int(r), int(c)] = 1
    return VoxelDoorSeedResult(
        door_seed_mask=seed,
        door_seed_component_map=labels,
        door_seed_reject_reason_map=np.where(seed, 1, 0).astype(np.uint8),
        lower_free_cells_xy=np.zeros(shape, dtype=np.uint16),
        top_occupied_cells_xy=np.zeros(shape, dtype=np.uint16),
        first_occupied_z_xy=np.full(shape, np.nan, dtype=np.float32),
        unknown_tail_cells_xy=np.zeros(shape, dtype=np.uint16),
        seed_evidence=[],
        debug={},
    )


def _door_completion(seed_cells: list[tuple[int, int]]):
    shape = (16, 18)
    free = np.ones(shape, dtype=bool)
    wall = np.zeros(shape, dtype=bool)
    wall[8, 3] = True
    wall[8, 12] = True
    wall[3:13, 10] = True  # nearby vertical distractor must not flip direction
    unknown = np.zeros(shape, dtype=bool)
    return complete_voxel_doors_from_seeds(
        seed_result=_seed_result(shape, seed_cells),
        free_map=free,
        free_map_for_visual_validation=free,
        base_partition_free=free,
        anchor_wall_map=wall,
        unknown_map=unknown,
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            min_seed_cells_for_accepted_extension=3,
            min_seed_line_length_cells_for_accepted_extension=3,
            min_seed_elongation_for_direction=1.4,
            accepted_orientation_mode="seed_major_only",
            local_free_neck_orientation_debug_only=True,
            partition_topology_enabled=False,
        ),
        real_wall_barrier_map=wall,
        seed_cluster_barrier_map=np.zeros(shape, dtype=bool),
    )


def test_v26_door_extension_requires_three_seed_cells() -> None:
    one = _door_completion([(8, 6)])
    two = _door_completion([(8, 6), (8, 7)])
    three = _door_completion([(8, 6), (8, 7), (8, 8)])

    assert np.any(one.door_seed_mask)
    assert np.any(two.door_seed_mask)
    assert not np.any(one.door_cut_mask_for_partition)
    assert not np.any(two.door_cut_mask_for_partition)
    assert np.any(three.door_cut_mask_for_partition)
    assert "seed_group_too_few_cells_for_extension" in one.debug["voxel_door_trial_reject_reason_counts"]
    assert "seed_group_too_few_cells_for_extension" in two.debug["voxel_door_trial_reject_reason_counts"]


def test_v26_door_direction_follows_seed_major_axis() -> None:
    result = _door_completion([(8, 6), (8, 7), (8, 8)])
    accepted = [item for item in result.candidates if item.accepted]
    assert accepted
    for candidate in accepted:
        assert abs(float(candidate.major_dir_rc[1])) >= abs(float(candidate.major_dir_rc[0]))
        assert str(candidate.debug.get("orientation_source")) in {"seed_major", "axis_snapped_from_seed_major", "wall_pair_aligned_with_seed"}


def test_v26_accepted_seed_mask_uses_actual_candidates_not_raw_cluster_geometry() -> None:
    shape = (8, 8)
    cluster_map = np.zeros(shape, dtype=np.int32)
    cluster_map[3, 2:5] = 1
    raw_seed = cluster_map > 0

    assert not np.any(accepted_seed_mask_from_candidates([], cluster_map, shape) & raw_seed)

    result = _door_completion([(8, 6), (8, 7), (8, 8)])
    accepted = accepted_seed_mask_from_candidates(result.candidates, np.where(result.door_seed_mask, 1, 0), result.door_seed_mask.shape)
    assert np.any(accepted)


def test_v26_projected_wall_lines_enter_step2_source_pool() -> None:
    shape = (20, 20)
    projected = ProjectedWallLine(
        line_id=7,
        axis="h",
        line=10,
        start=3,
        end=15,
        support_cell_count=8,
        projected_cell_count=13,
        support_ratio=0.7,
        lateral_std_cells=0.0,
        source="axis_accumulator",
    )
    filtered = projected_wall_line_to_filtered_wall_line(projected, 0.10, line_id=1007)
    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        projected_step2_lines=[filtered],
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        shape=shape,
        resolution_m=0.10,
    )

    assert pool.debug["voxel_step2_projected_source_line_count"] == 1
    assert pool.debug["voxel_step2_source_line_count_by_source"]["projected_step2_source"] >= 1
    assert pool.source_lines[0].debug["step2_source_kind"] == "projected_step2_source"
    assert np.any(pool.source_line_map & filtered_wall_line_mask([filtered], shape))


def test_v26_extension_intersection_fallback_is_disabled_by_default() -> None:
    cfg = VoxelOccupancyDoorWallRoomSegConfig(resolution_m=0.10)
    assert cfg.enable_extension_intersection_fallback is False


def test_v26_stable_separator_memory_keeps_corridor_after_missing_update() -> None:
    shape = (10, 12)
    candidate = SeparatorCandidate(
        candidate_id=1,
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray([5, 3], dtype=np.float32),
        p1_rc=np.asarray([5, 8], dtype=np.float32),
        theta=0.0,
        length_m=0.60,
        confidence=0.9,
        source_segment_ids=[7],
        accepted=True,
    )
    memory = StableSeparatorMemory(ttl_updates=30, decay_per_update=0.02, min_confidence_to_keep=0.15)
    first, first_debug = memory.update([candidate], step=1, shape=shape)
    second, second_debug = memory.update([], step=2, shape=shape)

    assert np.any(first)
    assert np.array_equal(first, second)
    assert first_debug["voxel_separator_memory_track_count"] == 1
    assert second_debug["voxel_separator_memory_track_count"] == 1


def test_v26_frontier_roomseg_gate_does_not_update_on_invalidated_path_by_default() -> None:
    gate = FrontierRoomsegUpdateGateState(initialized=True, last_update_step=0, last_update_reason="initial")
    update, reason = should_update_roomseg_frontiers(
        step=10,
        has_current_path=False,
        gate_state=gate,
        target_invalidated=True,
        no_active_path=True,
    )
    assert update is False
    assert reason == "target_invalidated_cached"
