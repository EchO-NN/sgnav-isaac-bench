import numpy as np

from isaac_bench.mapping.online_roomseg.evidence_maps import build_evidence_maps
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


def _empty_vertical_profile(shape: tuple[int, int]) -> VerticalProfileMap:
    return VerticalProfileMap.zeros(shape)


def test_terminal_wall_endpoint_turns_unknown_into_roomseg_occupied():
    shape = (7, 7)
    terminal = np.zeros(shape, dtype=np.uint16)
    terminal[3, 4] = 1

    evidence = build_evidence_maps(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        resolution_m=0.1,
        vertical_profile=_empty_vertical_profile(shape),
        roomseg_ray_evidence={"terminal_wall_count": terminal},
    )

    assert evidence.vertical_occupied_raw[3, 4]
    assert evidence.vertical_observed_raw[3, 4]
    assert not evidence.vertical_unknown_raw[3, 4]
    assert evidence.debug["terminal_wall_added_to_occupied_cells"] == 1
    assert evidence.debug["terminal_wall_added_to_observed_cells"] == 1


def test_terminal_wall_endpoint_does_not_override_vertical_free():
    shape = (7, 7)
    terminal = np.zeros(shape, dtype=np.uint16)
    terminal[3, 4] = 1
    vp = _empty_vertical_profile(shape)
    vp.free_ray_count[:, 3, 4] = 1
    vp.observed_count[:, 3, 4] = 1
    vp.unknown_count[:, 3, 4] = 0

    evidence = build_evidence_maps(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        resolution_m=0.1,
        vertical_profile=vp,
        roomseg_ray_evidence={"terminal_wall_count": terminal},
    )

    assert evidence.vertical_free_raw[3, 4]
    assert not evidence.vertical_occupied_raw[3, 4]
    assert evidence.debug["terminal_wall_suppressed_by_vertical_free_cells"] == 1


def test_vertical_observed_without_free_or_occupied_stays_unknown_not_wall():
    shape = (7, 7)
    vp = _empty_vertical_profile(shape)
    vp.observed_count[:, 3, 4] = 1
    vp.unknown_count[:, 3, 4] = 0

    evidence = build_evidence_maps(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        resolution_m=0.1,
        vertical_profile=vp,
        roomseg_ray_evidence={},
    )

    assert evidence.vertical_observed_raw[3, 4]
    assert not evidence.vertical_free_raw[3, 4]
    assert not evidence.vertical_occupied_raw[3, 4]
    assert evidence.vertical_unknown_raw[3, 4]
    assert not evidence.wall_candidate_clean[3, 4]
    assert evidence.unknown_clean[3, 4]
    assert evidence.debug["wall_candidate_clean"]["observed_nonfree_ambiguous_cells_ignored"] == 1


def test_vertical_occupied_with_remaining_unknown_band_stays_unknown_not_wall():
    shape = (7, 7)
    vp = _empty_vertical_profile(shape)
    vp.occupied_count[0, 3, 4] = 1
    vp.observed_count[0, 3, 4] = 1
    vp.unknown_count[0, 3, 4] = 0

    evidence = build_evidence_maps(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        resolution_m=0.1,
        vertical_profile=vp,
        roomseg_ray_evidence={},
    )

    assert evidence.vertical_observed_raw[3, 4]
    assert not evidence.vertical_free_raw[3, 4]
    assert not evidence.vertical_occupied_raw[3, 4]
    assert evidence.vertical_unknown_raw[3, 4]
    assert not evidence.wall_candidate_clean[3, 4]
    assert evidence.unknown_clean[3, 4]
    assert evidence.debug["vertical_partial_unknown_occupied_suppressed_cells"] == 1
