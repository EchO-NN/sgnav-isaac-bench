from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_CONFLICT, VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_UNKNOWN
from isaac_bench.mapping.voxel_roomseg_evidence import VoxelRoomsegEvidenceConfig, classify_voxel_columns_for_roomseg


def _state(shape: tuple[int, int], z_bins: int = 6) -> np.ndarray:
    return np.full((z_bins, *shape), int(VOXEL_UNKNOWN), dtype=np.uint8)


def test_v22_navigation_free_has_priority_over_ratio_wall_when_allowed() -> None:
    shape = (4, 4)
    state = _state(shape)
    state[:, 1, 1] = int(VOXEL_OCCUPIED)
    nav_free = np.zeros(shape, dtype=bool)
    nav_free[1, 1] = True

    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        navigation_unknown_mask=np.zeros(shape, dtype=bool),
        cfg=VoxelRoomsegEvidenceConfig(nav_assisted_free_forbidden_on_ratio_wall=False),
    )

    assert classified["vertical_free"][1, 1]
    assert not classified["wall"][1, 1]
    assert not classified["unknown"][1, 1]


def test_v22_unknown_dominant_has_priority_over_occupied_support() -> None:
    shape = (4, 4)
    state = _state(shape)
    state[0:2, 1, 1] = int(VOXEL_OCCUPIED)
    state[2:6, 1, 1] = int(VOXEL_UNKNOWN)

    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=np.zeros(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        navigation_unknown_mask=np.zeros(shape, dtype=bool),
        cfg=VoxelRoomsegEvidenceConfig(wall_line_support_remove_small_area_cells=0),
    )

    assert classified["unknown"][1, 1]
    assert not classified["wall"][1, 1]
    assert not classified["wall_support_for_projection"][1, 1]
    assert classified["wall_support_unknown_rejected"][1, 1]


def test_v22_ratio_wall_survives_when_not_free_or_unknown() -> None:
    shape = (4, 4)
    state = _state(shape)
    state[:, 2, 2] = int(VOXEL_OCCUPIED)

    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=np.zeros(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        navigation_unknown_mask=np.zeros(shape, dtype=bool),
        cfg=VoxelRoomsegEvidenceConfig(wall_line_support_remove_small_area_cells=0),
    )

    assert classified["wall"][2, 2]
    assert not classified["vertical_free"][2, 2]
    assert not classified["unknown"][2, 2]


def test_v22_navigation_unknown_blocks_residual_wall_support() -> None:
    shape = (5, 5)
    state = _state(shape)
    state[0, 2, 2] = int(VOXEL_OCCUPIED)
    state[1:3, 2, 2] = int(VOXEL_CONFLICT)
    nav_unknown = np.zeros(shape, dtype=bool)
    nav_unknown[2, 2] = True

    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=np.zeros(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        navigation_unknown_mask=nav_unknown,
        cfg=VoxelRoomsegEvidenceConfig(wall_line_support_remove_small_area_cells=0),
    )

    assert classified["unknown"][2, 2]
    assert not classified["wall_support_for_projection"][2, 2]
    assert classified["wall_support_nav_unknown_rejected"][2, 2]


def test_v22_frontier_unknown_band_blocks_short_boundary_support() -> None:
    shape = (8, 8)
    state = _state(shape)
    state[0, 3, 4] = int(VOXEL_OCCUPIED)
    state[1:3, 3, 4] = int(VOXEL_FREE)
    state[3, 3, 4] = int(VOXEL_CONFLICT)
    state[0:3, 3, 1:4] = int(VOXEL_FREE)
    nav_free = np.zeros(shape, dtype=bool)
    nav_free[3, 1:4] = True
    nav_unknown = np.zeros(shape, dtype=bool)
    nav_unknown[3, 5] = True

    classified = classify_voxel_columns_for_roomseg(
        state_active=state,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        navigation_unknown_mask=nav_unknown,
        cfg=VoxelRoomsegEvidenceConfig(wall_line_support_remove_small_area_cells=0),
    )

    assert classified["wall_support_raw_occupied"][3, 4]
    assert classified["frontier_unknown_band"][3, 4]
    assert not classified["wall_support_for_projection"][3, 4]
    assert classified["wall_support_frontier_band_rejected"][3, 4]
