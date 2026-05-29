from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.online_roomseg.separator_candidates import LineExtensionHit
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import (
    VoxelOccupancyDoorWallRoomSegConfig,
    build_voxel_partition_maps,
    reject_step2_extension_hits_v23,
)
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)
from isaac_bench.mapping.voxel_roomseg_evidence import VoxelRoomsegEvidenceConfig, build_voxel_roomseg_evidence
from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


def _grid(shape: tuple[int, int] = (16, 18), z_bins: int = 6) -> VoxelOccupancyGrid3D:
    info = MapInfo(
        resolution_m=0.10,
        min_x=0.0,
        max_x=float(shape[1]) * 0.10,
        min_y=0.0,
        max_y=float(shape[0]) * 0.10,
        width=shape[1],
        height=shape[0],
    )
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(
            z_min_m=0.0,
            z_max_m=float(z_bins) * 0.10,
            z_resolution_m=0.10,
            active_z_min_m=0.0,
            active_z_max_fallback_m=float(z_bins) * 0.10,
        ),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = float(z_bins) * 0.10
    return grid


def test_v23_both_sides_free_projected_wall_is_preserved() -> None:
    shape = (18, 24)
    seed = np.zeros(shape, dtype=bool)
    seed[9, 4:20] = True
    free = np.ones(shape, dtype=bool)

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=np.zeros(shape, dtype=bool),
        forbidden_frontier_residual_map=np.zeros(shape, dtype=bool),
        support_weight=seed.astype(np.float32),
        vertical_free_map=free,
        unknown_map=np.zeros(shape, dtype=bool),
        structural_side_support_map=seed,
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.20, reject_if_both_sides_free_ratio_gt=0.55),
    )

    assert np.any(result.projected_wall_map)
    assert result.debug["voxel_wall_projection_reject_reason_counts"].get("projected_wall_both_sides_free_furniture_like", 0) == 0
    assert any(bool(line.debug.get("both_sides_free_like", False)) for line in result.projected_lines)


def test_v23_frontier_diagonal_residual_does_not_project_or_enter_step2_target() -> None:
    grid = _grid()
    shape = grid.shape
    nav_free = np.zeros(shape, dtype=bool)
    nav_free[2:12, 1:8] = True
    grid.state[:, :, :] = int(VOXEL_UNKNOWN)
    grid.state[0:3, 2:12, 1:8] = int(VOXEL_FREE)
    diagonal = [(4, 8), (5, 9), (6, 10), (7, 11)]
    for r, c in diagonal:
        grid.state[0, r, c] = int(VOXEL_OCCUPIED)
    nav_unknown = np.zeros(shape, dtype=bool)
    nav_unknown[:, 10:] = True

    evidence = build_voxel_roomseg_evidence(
        voxel_grid=grid,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask_from_navigation=nav_unknown,
        resolution_m=0.10,
        config=VoxelRoomsegEvidenceConfig(active_z_min_m=0.0, wall_line_support_remove_small_area_cells=0),
    )
    residual = np.asarray(evidence.forbidden_frontier_residual_support_xy, dtype=bool)
    diagonal_mask = np.zeros(shape, dtype=bool)
    for r, c in diagonal:
        diagonal_mask[r, c] = True

    projection = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=np.asarray(evidence.support_seed_for_projection_xy, dtype=bool),
        support_bridge_map=np.asarray(evidence.support_bridge_for_projection_xy, dtype=bool),
        forbidden_frontier_residual_map=residual | np.asarray(evidence.forbidden_unknown_boundary_support_xy, dtype=bool),
        support_weight=np.asarray(evidence.wall_support_weight_xy, dtype=np.float32),
        vertical_free_map=evidence.vertical_free_xy,
        unknown_map=evidence.unknown_xy,
        unknown_ratio_map=evidence.unknown_ratio_active_xy,
        navigation_unknown_map=nav_unknown,
        frontier_unknown_band=np.asarray(evidence.frontier_unknown_band_xy, dtype=bool),
        structural_side_support_map=evidence.wall_xy | np.asarray(evidence.support_seed_for_projection_xy, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.20),
    )
    partitions = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_mask=np.zeros(shape, dtype=bool),
        strict_raw_wall=evidence.wall_xy,
        projected_wall_map=np.asarray(projection.projected_wall_map, dtype=bool),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        filtered_line_map=np.zeros(shape, dtype=bool),
        extension_seed_line_map=np.zeros(shape, dtype=bool),
        step1_gap_fill_map=np.zeros(shape, dtype=bool),
        cfg=VoxelOccupancyDoorWallRoomSegConfig(resolution_m=0.10),
    )

    assert np.any(residual & diagonal_mask)
    assert not np.any(np.asarray(projection.projected_wall_map, dtype=bool) & diagonal_mask)
    assert not np.any(partitions.step2_target_wall_map & diagonal_mask)


def test_v23_bridge_only_support_cannot_create_projected_line_but_can_fill_seed_gap() -> None:
    shape = (12, 16)
    bridge_only = np.zeros(shape, dtype=bool)
    bridge_only[6, 3:12] = True
    bridge_result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=np.zeros(shape, dtype=bool),
        support_bridge_map=bridge_only,
        vertical_free_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.20),
    )
    assert not np.any(bridge_result.projected_wall_map)
    assert bridge_result.debug["voxel_wall_projection_reject_reason_counts"]["projected_wall_bridge_only_line"] == 1

    seed = np.zeros(shape, dtype=bool)
    seed[6, 2:5] = True
    seed[6, 9:12] = True
    bridge = np.zeros(shape, dtype=bool)
    bridge[6, 5:9] = True
    filled_result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=seed,
        support_bridge_map=bridge,
        vertical_free_map=bridge,
        unknown_map=np.zeros(shape, dtype=bool),
        structural_side_support_map=seed,
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(
            min_projected_line_length_m=0.20,
            max_fill_gap_m=0.50,
            reject_if_no_free_side=False,
            reject_if_no_nonfree_side=False,
        ),
    )
    assert np.any(filled_result.projected_wall_map[6, 5:9])
    assert filled_result.projected_lines[0].debug["line_seed_support_cells"] >= 3
    assert filled_result.projected_lines[0].debug["line_bridge_support_cells"] >= 1


def test_v23_step2_hit_frontier_residual_wall_is_rejected() -> None:
    shape = (12, 14)
    residual = np.zeros(shape, dtype=bool)
    residual[6, 10] = True
    hit = LineExtensionHit(
        extension_id=1,
        source_line_id=7,
        source_endpoint="p1",
        pass_id=2,
        p_start_rc=np.asarray([6, 4], dtype=np.float32),
        p_hit_rc=np.asarray([6, 10], dtype=np.float32),
        theta=0.0,
        length_m=0.60,
        interior_free_ratio=1.0,
        interior_unknown_ratio=0.0,
        interior_wall_ratio=0.0,
        hit_type="real_wall",
        hit_candidate_id=None,
        confidence=0.8,
        reject_reason=None,
        debug={},
    )

    debug = reject_step2_extension_hits_v23(
        [hit],
        forbidden_frontier_residual_map=residual,
        frontier_unknown_band=np.zeros(shape, dtype=bool),
        target_wall_map=np.zeros(shape, dtype=bool),
        target_source_map=np.zeros(shape, dtype=np.uint8),
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        existing_separator_map=np.zeros(shape, dtype=bool),
        shape=shape,
    )

    assert hit.reject_reason == "reject_step2_hit_frontier_residual_wall"
    assert debug["voxel_step2_v23_hit_reject_reason_counts"]["reject_step2_hit_frontier_residual_wall"] == 1
