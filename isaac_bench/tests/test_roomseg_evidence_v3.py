import numpy as np

from isaac_bench.mapping.roomseg_evidence_v3 import build_roomseg_evidence_v3
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


def test_unknown_ray_does_not_create_occupied_diagonal():
    shape = (12, 12)
    vp = VerticalProfileMap.zeros(shape)
    ray_covered = np.zeros(shape, dtype=np.uint16)
    for i in range(2, 10):
        ray_covered[i, i] = 1

    result = build_roomseg_evidence_v3(
        vertical_profile=vp,
        grid_free=np.zeros(shape, dtype=bool),
        grid_occupied=np.zeros(shape, dtype=bool),
        grid_observed=ray_covered > 0,
        roomseg_ray_covered_count=ray_covered,
        roomseg_terminal_wall_count=np.zeros(shape, dtype=np.uint16),
        roomseg_terminal_wall_splat=np.zeros(shape, dtype=np.uint8),
        roomseg_terminal_wall_height_min=None,
        roomseg_terminal_wall_height_max=None,
        roomseg_terminal_wall_depth_min=None,
        robot_rc=None,
        resolution_m=0.05,
        config={},
    )

    assert int(np.count_nonzero(result.structural_wall_clean)) == 0
    assert int(np.count_nonzero(result.raw_endpoint_occupied)) == 0
    assert not np.any(result.roomseg_free_clean)


def test_single_endpoint_is_not_structural_wall():
    shape = (8, 8)
    vp = VerticalProfileMap.zeros(shape)
    vp.occupied_count[1, 4, 4] = 1
    vp.observed_count[1, 4, 4] = 1
    vp.unknown_count[1, 4, 4] = 0

    result = build_roomseg_evidence_v3(
        vertical_profile=vp,
        grid_free=np.zeros(shape, dtype=bool),
        grid_occupied=np.zeros(shape, dtype=bool),
        grid_observed=np.zeros(shape, dtype=bool),
        roomseg_ray_covered_count=np.zeros(shape, dtype=np.uint16),
        roomseg_terminal_wall_count=np.zeros(shape, dtype=np.uint16),
        roomseg_terminal_wall_splat=np.zeros(shape, dtype=np.uint8),
        roomseg_terminal_wall_height_min=None,
        roomseg_terminal_wall_height_max=None,
        roomseg_terminal_wall_depth_min=None,
        robot_rc=None,
        resolution_m=0.05,
        config={},
    )

    assert result.raw_endpoint_occupied[4, 4]
    assert not result.structural_wall_clean[4, 4]


def test_roomseg_free_removes_random_islands():
    shape = (30, 30)
    vp = VerticalProfileMap.zeros(shape)
    main = np.zeros(shape, dtype=bool)
    main[5:22, 5:22] = True
    island = np.zeros(shape, dtype=bool)
    island[26, 26] = True
    vp.free_ray_count[0, main | island] = 1
    vp.observed_count[0, main | island] = 1
    vp.unknown_count[0, main | island] = 0
    grid_free = main.copy()
    grid_observed = main.copy()

    result = build_roomseg_evidence_v3(
        vertical_profile=vp,
        grid_free=grid_free,
        grid_occupied=np.zeros(shape, dtype=bool),
        grid_observed=grid_observed,
        roomseg_ray_covered_count=np.zeros(shape, dtype=np.uint16),
        roomseg_terminal_wall_count=np.zeros(shape, dtype=np.uint16),
        roomseg_terminal_wall_splat=np.zeros(shape, dtype=np.uint8),
        roomseg_terminal_wall_height_min=None,
        roomseg_terminal_wall_height_max=None,
        roomseg_terminal_wall_depth_min=None,
        robot_rc=(10, 10),
        resolution_m=0.05,
        config={},
    )

    assert result.roomseg_free_raw[26, 26]
    assert not result.roomseg_free_clean[26, 26]
    intersection = int(np.count_nonzero(result.roomseg_free_clean & grid_free))
    union = int(np.count_nonzero(result.roomseg_free_clean | grid_free))
    assert intersection / max(1, union) >= 0.75
