from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_roomseg_evidence import VoxelRoomsegEvidenceConfig, classify_voxel_columns_for_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_UNKNOWN
from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


def test_v23_free_conflict_support_is_bridge_only_without_strong_seed() -> None:
    shape = (12, 16)
    state = np.full((6, *shape), int(VOXEL_UNKNOWN), dtype=np.uint8)
    state[0:2, 6, 3:13] = int(VOXEL_OCCUPIED)
    state[2:5, 6, 3:13] = int(VOXEL_FREE)
    state[2:5, 7:10, 3:13] = int(VOXEL_FREE)
    nav_free = np.zeros(shape, dtype=bool)
    nav_free[6:10, 3:13] = True

    cfg = VoxelRoomsegEvidenceConfig(
        nav_assisted_free_enabled=False,
        wall_line_support_remove_small_area_cells=0,
    )
    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        cfg=cfg,
    )

    assert classified["vertical_free"][6, 8]
    assert not classified["wall"][6, 8]
    assert classified["free_conflict_support"][6, 8]
    assert not classified["wall_line_support_conflict"][6, 8]
    assert not classified["wall_line_support"][6, 8]
    assert not classified["support_seed_for_projection"][6, 8]
    assert not classified["support_bridge_for_projection"][6, 8]

    result = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=classified["support_seed_for_projection"],
        support_bridge_map=classified["support_bridge_for_projection"],
        support_weight=classified["wall_line_support_weight"],
        vertical_free_map=classified["vertical_free"],
        unknown_map=classified["unknown"],
        door_forbidden_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.30, min_projected_support_ratio=0.25),
    )

    assert not np.any(result.projected_wall_display_map[6, 3:13])


def test_v21_furniture_island_is_rejected_from_wall_support_boundary() -> None:
    shape = (12, 16)
    state = np.full((6, *shape), int(VOXEL_FREE), dtype=np.uint8)
    state[0:2, 6, 5:10] = int(VOXEL_OCCUPIED)

    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=np.ones(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        cfg=VoxelRoomsegEvidenceConfig(
            nav_assisted_free_enabled=False,
            wall_line_support_remove_small_area_cells=0,
        ),
    )

    assert np.any(classified["wall_line_support_raw"][6, 5:10])
    assert not np.any(classified["wall_line_support"][6, 5:10])
    assert np.any(classified["wall_line_support_rejected_furniture"][6, 5:10])
