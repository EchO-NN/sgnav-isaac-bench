from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorCandidate
from isaac_bench.mapping.voxel_door_detector import (
    VoxelDoorDetectorConfig,
    VoxelDoorLineCandidate,
    VoxelDoorMemory,
    VoxelDoorSeedResult,
    complete_voxel_doors_from_seeds,
)
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    build_step2_line_pool,
    reject_step2_candidate_conflicts_v22,
)
from isaac_bench.mapping.wall_projection import ProjectedWallLine


def _projected_line(*, line_id: int, start: int, end: int, support_cells: int, support_ratio: float) -> ProjectedWallLine:
    return ProjectedWallLine(
        line_id=int(line_id),
        axis="h",
        line=6,
        start=int(start),
        end=int(end),
        support_cell_count=int(support_cells),
        projected_cell_count=max(0, int(end) - int(start) + 1),
        support_ratio=float(support_ratio),
        lateral_std_cells=0.0,
        source="axis_accumulator",
        reject_reason=None,
        side_free_ratio_a=0.20,
        side_free_ratio_b=0.80,
        side_unknown_ratio_a=0.0,
        side_unknown_ratio_b=0.0,
        side_nonfree_ratio_a=0.50,
        side_nonfree_ratio_b=0.0,
        structural_side_score=0.50,
    )


def test_v22_short_projected_edge_is_not_step2_source() -> None:
    shape = (14, 14)
    short_line = _projected_line(line_id=1, start=4, end=7, support_cells=4, support_ratio=1.0)
    support = np.zeros(shape, dtype=bool)
    support[6, 4:8] = True

    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        projected_display_lines=[],
        projected_source_lines=[short_line],
        source_support_map=support,
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        projected_source_line_map=support,
        shape=shape,
        resolution_m=0.10,
    )

    assert len(pool.source_lines) == 0
    assert pool.debug["voxel_step2_rejected_source_line_count"] == 1
    assert pool.debug["voxel_step2_source_reject_reason_counts"]["reject_step2_source_too_short"] == 1
    assert not np.any(pool.source_line_map)


def test_v22_frontier_unknown_band_projected_line_is_not_step2_source() -> None:
    shape = (16, 18)
    long_line = _projected_line(line_id=2, start=3, end=12, support_cells=10, support_ratio=1.0)
    support = np.zeros(shape, dtype=bool)
    support[6, 3:13] = True
    frontier_band = np.zeros(shape, dtype=bool)
    frontier_band[6, 8] = True

    pool = build_step2_line_pool(
        filtered_lines=[],
        extension_seed_lines=[],
        projected_display_lines=[],
        projected_source_lines=[long_line],
        source_support_map=support,
        frontier_unknown_band=frontier_band,
        strict_raw_wall=np.zeros(shape, dtype=bool),
        projected_wall_map=np.zeros(shape, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        projected_source_line_map=support,
        shape=shape,
        resolution_m=0.10,
    )

    assert len(pool.source_lines) == 0
    assert pool.debug["voxel_step2_source_reject_reason_counts"]["reject_step2_source_frontier_unknown_edge"] == 1


def test_v22_pairwise_step2_candidate_crossing_is_rejected() -> None:
    shape = (20, 20)
    a = SeparatorCandidate(
        candidate_id=1,
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray([4, 10], dtype=np.float32),
        p1_rc=np.asarray([16, 10], dtype=np.float32),
        theta=0.0,
        length_m=1.2,
        confidence=0.7,
        source_segment_ids=[1],
        wall_support_score=0.5,
        free_gap_score=0.5,
        topology_gain_score=0.5,
        debug={},
    )
    b = SeparatorCandidate(
        candidate_id=2,
        kind="line_extension_corridor_separator",
        p0_rc=np.asarray([10, 4], dtype=np.float32),
        p1_rc=np.asarray([10, 16], dtype=np.float32),
        theta=1.5708,
        length_m=1.2,
        confidence=0.7,
        source_segment_ids=[2],
        wall_support_score=0.5,
        free_gap_score=0.5,
        topology_gain_score=0.5,
        debug={},
    )

    kept, rejected, debug = reject_step2_candidate_conflicts_v22(
        [a, b],
        accepted_door_mask=np.zeros(shape, dtype=bool),
        stable_door_mask=np.zeros(shape, dtype=bool),
        accepted_step2_memory_mask=np.zeros(shape, dtype=bool),
        real_wall_map=np.zeros(shape, dtype=bool),
        shape=shape,
        cfg=VoxelOccupancyDoorWallRoomSegConfig(),
    )

    assert len(kept) == 0
    assert len(rejected) == 2
    assert debug["voxel_step2_conflict_reject_reason_counts"]["reject_step2_pairwise_crossing_ambiguous"] == 2


def _accepted_candidate() -> VoxelDoorLineCandidate:
    return VoxelDoorLineCandidate(
        candidate_id=1,
        seed_component_id=1,
        seed_cells=[(5, 4), (5, 5), (5, 6)],
        center_rc=(5.0, 5.0),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(-1.0, 0.0),
        seed_projected_centerline_cells=[(5, 4), (5, 5), (5, 6)],
        extended_centerline_cells=[(5, 3), (5, 4), (5, 5), (5, 6), (5, 7)],
        door_cut_cells=[(5, 5)],
        wall_anchor_a=(5, 3),
        wall_anchor_b=(5, 7),
        width_m=0.50,
        accepted=True,
        reject_reason=None,
        debug={"partition_accepted": True},
    )


def test_v22_door_memory_ignores_unknown_and_requires_repeated_strict_wall_contradiction() -> None:
    shape = (12, 12)
    cfg = VoxelDoorDetectorConfig(door_memory_decay_per_update=0.0, door_memory_hard_contradiction_min_updates=3)
    memory = VoxelDoorMemory(cfg)
    memory.update([_accepted_candidate()], step=1, update_index=1, shape=shape)

    unknown = np.ones(shape, dtype=bool)
    kept_unknown = memory.update([], step=2, update_index=2, shape=shape, contradiction_unknown_map=unknown)
    assert kept_unknown.debug["voxel_door_memory_track_count"] == 1
    assert np.any(kept_unknown.stable_door_cut_mask)

    strict_wall = np.zeros(shape, dtype=bool)
    strict_wall[5, 5] = True
    kept_once = memory.update([], step=3, update_index=3, shape=shape, contradiction_wall_map=strict_wall)
    kept_twice = memory.update([], step=4, update_index=4, shape=shape, contradiction_wall_map=strict_wall)
    pruned = memory.update([], step=5, update_index=5, shape=shape, contradiction_wall_map=strict_wall)

    assert kept_once.debug["voxel_door_memory_track_count"] == 1
    assert kept_twice.debug["voxel_door_memory_track_count"] == 1
    assert pruned.debug["voxel_door_memory_track_count"] == 0
    assert pruned.debug["voxel_door_memory_prune_reason_counts"]["hard_wall_contradiction"] == 1


def _seed_result(shape: tuple[int, int], cells: list[tuple[int, int]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for r, c in cells:
        seed[int(r), int(c)] = True
        labels[int(r), int(c)] = 1
    return VoxelDoorSeedResult(
        door_seed_mask=seed,
        door_seed_component_map=labels,
        door_seed_reject_reason_map=np.zeros(shape, dtype=np.uint8),
        lower_free_cells_xy=np.zeros(shape, dtype=np.uint16),
        top_occupied_cells_xy=np.zeros(shape, dtype=np.uint16),
        first_occupied_z_xy=np.full(shape, np.nan, dtype=np.float32),
        unknown_tail_cells_xy=np.zeros(shape, dtype=np.uint16),
        seed_evidence=[],
        debug={},
    )


def test_v22_one_seed_one_wall_completion_is_visual_only_by_default() -> None:
    shape = (16, 16)
    free = np.ones(shape, dtype=bool)
    anchors = np.zeros(shape, dtype=bool)
    anchors[8, 4] = True
    seed_result = _seed_result(shape, [(8, 8)])
    cfg = VoxelDoorDetectorConfig(
        wall_anchor_radius_cells=0,
        seed_cluster_morph_close_radius_cells=0,
        min_seed_cells_for_partition_completion=3,
        one_seed_one_wall_partition_enabled=False,
    )

    completion = complete_voxel_doors_from_seeds(
        seed_result=seed_result,
        free_map=free,
        base_partition_free=free & ~anchors,
        anchor_wall_map=anchors,
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
        real_wall_barrier_map=anchors,
    )

    accepted_visual = [candidate for candidate in completion.candidates if bool(candidate.accepted)]
    assert accepted_visual
    assert all(not bool(candidate.debug.get("partition_accepted", False)) for candidate in accepted_visual)
    assert "door_partition_one_seed_one_wall_visual_only" in completion.debug["voxel_door_partition_reject_reason_counts"]
