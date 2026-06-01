from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_door_detector import (
    DoorSeedGroup,
    VoxelDoorDetectorConfig,
    VoxelDoorSeedResult,
    complete_voxel_doors_from_seeds,
    extract_seed_line_primitives_from_group,
)


def _seed_result(shape: tuple[int, int], components: list[list[tuple[int, int]]]) -> VoxelDoorSeedResult:
    seed = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for cid, cells in enumerate(components, start=1):
        for r, c in cells:
            seed[int(r), int(c)] = True
            labels[int(r), int(c)] = int(cid)
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


def _group(shape: tuple[int, int], cells: list[tuple[int, int]]) -> DoorSeedGroup:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    pts = np.asarray(cells, dtype=np.float32)
    center = np.mean(pts, axis=0)
    return DoorSeedGroup(
        group_id=1,
        group_kind="single_cluster",
        source_cluster_ids=[1],
        component_ids=[1],
        seed_cells=list(cells),
        mask=mask,
        bbox_rc=(min(r for r, _c in cells), max(r for r, _c in cells), min(c for _r, c in cells), max(c for _r, c in cells)),
        center_rc=(float(center[0]), float(center[1])),
        major_dir_rc=(0.0, 1.0),
        minor_dir_rc=(1.0, 0.0),
        line_fit_residual_cells=10.0,
        thickness_m=1.0,
        length_m=1.0,
        accepted_for_completion=True,
        reject_reason=None,
    )


def test_v32_seed_pair_bridge_primitive_feeds_partition_cut() -> None:
    shape = (18, 24)
    seed = _seed_result(shape, [[(9, 10)], [(10, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[9:11, 9:14] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free,
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(
            min_seed_cells_for_accepted_extension=1,
            min_seed_line_length_cells_for_accepted_extension=1,
            min_seed_elongation_for_direction=1.0,
            seed_cluster_morph_close_radius_cells=0,
            seed_cluster_merge_distance_cells=0,
            seed_cluster_max_perpendicular_gap_cells=0,
            partition_topology_enabled=False,
        ),
    )

    assert result.debug["voxel_v32_seed_line_primitive_completion"] is True
    assert result.debug["voxel_door_line_primitive_count"] >= 1
    assert result.debug["voxel_door_extensible_primitive_count"] >= 1
    assert np.any(result.debug["voxel_door_seed_line_primitive_mask"])
    assert np.any(result.debug["voxel_door_partition_cut_mask"])
    assert np.any(result.door_cut_mask_for_partition)


def test_v32_raw_seed_without_verified_cut_stays_out_of_partition_mask() -> None:
    shape = (20, 24)
    seed = _seed_result(shape, [[(10, 10), (10, 11), (10, 12)]])
    free = np.zeros(shape, dtype=bool)
    free[10, 8:15] = True

    result = complete_voxel_doors_from_seeds(
        seed_result=seed,
        free_map=free,
        base_partition_free=free,
        anchor_wall_map=np.zeros(shape, dtype=bool),
        unknown_map=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=VoxelDoorDetectorConfig(partition_topology_enabled=True),
    )

    assert int(result.debug["voxel_door_raw_seed_cells"]) == 3
    assert np.any(result.debug["voxel_door_seed_line_primitive_mask"])
    assert not np.any(result.door_cut_mask_for_partition)
    assert not np.any(result.debug["voxel_door_partition_cut_mask"])


def test_v32_l_shaped_seed_blob_extracts_line_primitives_before_rejecting_blob() -> None:
    shape = (24, 24)
    cells = [(10, c) for c in range(5, 13)] + [(r, 8) for r in range(7, 14)]
    cfg = VoxelDoorDetectorConfig(
        primitive_max_residual_cells=0.2,
        max_seed_line_residual_cells_for_direction=0.2,
        ransac_inlier_residual_cells=0.1,
        ransac_min_inliers=3,
        primitive_max_thickness_cells=2,
        primitive_max_along_gap_cells=1.5,
        max_primitives_per_cluster=3,
    )

    primitives = extract_seed_line_primitives_from_group(
        _group(shape, cells),
        primitive_id_start=1,
        shape=shape,
        resolution_m=0.10,
        cfg=cfg,
    )

    assert primitives[0].extraction_method == "direct_pca"
    assert primitives[0].accepted_for_extension is False
    assert primitives[0].reject_reason in {"primitive_too_thick", "primitive_residual_too_high", "primitive_elongation_too_low"}
    accepted = [item for item in primitives if item.accepted_for_extension]
    assert accepted
    assert all(item.extraction_method == "ransac_blob" for item in accepted)
