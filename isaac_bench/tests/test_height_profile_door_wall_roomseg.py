import numpy as np

from isaac_bench.mapping.height_column_profile import HeightColumnProfileConfig, HeightColumnProfileMap
from isaac_bench.mapping.height_profile_door_wall_roomseg import (
    HeightProfileDoorWallRoomSegConfig,
    HeightProfileDoorWallRoomSegmenter,
    extend_step2_wall_lines,
    run_height_profile_door_wall_roomseg,
)
from isaac_bench.mapping.online_roomseg.separator_candidates import LineExtensionConfig, SeparatorAnchorConfig, SeparatorCandidate
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators
from isaac_bench.mapping.online_roomseg.wall_lines import FilteredWallLine


def _profile(shape):
    cfg = HeightColumnProfileConfig(use_navigation_free_gate=False)
    return HeightColumnProfileMap.zeros(shape, cfg)


def _mark_free(profile, cells):
    for r, c in cells:
        profile.free_ray_count[0:3, r, c] = 1
        profile.observed_count[0:3, r, c] = 1


def _mark_wall(profile, cells):
    active = (profile.bin_centers_m >= profile.active_z_min_m) & (profile.bin_centers_m <= float(profile.active_z_max_m))
    active_idxs = np.flatnonzero(active)
    for r, c in cells:
        profile.occupied_count[active_idxs, r, c] = 1
        profile.observed_count[active_idxs, r, c] = 1


def _line(line_id=1):
    return FilteredWallLine(
        line_id=line_id,
        p0_rc=np.asarray([10, 10], dtype=np.float32),
        p1_rc=np.asarray([10, 15], dtype=np.float32),
        theta=0.0,
        normal_theta=np.pi / 2.0,
        length_m=0.30,
        support_ratio=1.0,
        mean_wall_score=1.0,
        thickness_m=0.05,
        source_segment_ids=[1],
        confidence=1.0,
    )


def test_step1_fills_small_collinear_wall_gap_without_filling_unknown_as_wall():
    shape = (24, 32)
    profile = _profile(shape)
    left = [(10, c) for c in range(5, 11)]
    right = [(10, c) for c in range(14, 21)]
    gap = [(10, c) for c in range(11, 14)]
    unknown_gap = [(10, c) for c in range(22, 25)]
    _mark_wall(profile, left + right)
    wall_cells = set(left + right)
    _mark_free(profile, [(r, c) for r in range(6, 15) for c in range(5, 21) if (r, c) not in wall_cells] + gap)
    nav_free = profile.free_ray_count.any(axis=0)
    nav_obstacle = profile.occupied_count.any(axis=0)
    cfg = HeightProfileDoorWallRoomSegConfig.from_mapping(
        {
            "height_profile_evidence": {"free_remove_island_max_area_cells": 0, "wall_remove_island_max_area_cells": 0, "wall_min_component_cells": 1},
            "online_roomseg": {
                "line_walls": {"min_line_length_m": 0.20, "min_support_ratio": 0.1},
                "line_filtering": {"min_filtered_line_length_m": 0.20, "min_filtered_support_ratio": 0.1, "endpoint_min_wall_support_m": 0.05, "min_confidence": 0.1},
                "wall_run_snap": {"min_run_length_m": 0.20, "min_support_ratio": 0.1},
            },
        }
    )

    result = run_height_profile_door_wall_roomseg(
        occupancy_map=nav_obstacle,
        observed_free_mask=nav_free,
        obstacle_mask=nav_obstacle,
        unknown_mask=~(nav_free | nav_obstacle),
        height_profile=profile,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        roomseg_ray_evidence=None,
        resolution_m=0.05,
        config=cfg,
    )

    assert np.any(result.step1_wall_gap_fill_map[10, 11:14])
    assert not np.any(result.step1_wall_gap_fill_map[[r for r, _c in unknown_gap], [c for _r, c in unknown_gap]])


def test_step2_extension_requires_hit_wall_between_0p4_and_1p6m():
    shape = (24, 40)
    free = np.zeros(shape, dtype=bool)
    free[10, 16:25] = True
    wall = np.zeros(shape, dtype=bool)
    wall[10, 25] = True
    unknown = ~(free | wall)

    hits, _debug = extend_step2_wall_lines(
        filtered_lines=[_line()],
        free_after_step1=free,
        step1_wall_mask=wall,
        unknown_after_step1=unknown,
        door_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.05,
        config=LineExtensionConfig(min_extension_m=0.40, max_extension_m=1.60, max_probe_m=1.60),
        reject_if_intersects_door=True,
        door_intersection_dilation_cells=1,
    )

    assert any(hit.reject_reason is None and 0.40 <= hit.length_m <= 1.60 for hit in hits)


def test_step2_extension_rejects_hit_on_door():
    shape = (24, 40)
    free = np.zeros(shape, dtype=bool)
    free[10, 16:25] = True
    wall = np.zeros(shape, dtype=bool)
    wall[10, 25] = True
    door = np.zeros(shape, dtype=bool)
    door[10, 25] = True

    hits, _debug = extend_step2_wall_lines(
        filtered_lines=[_line()],
        free_after_step1=free,
        step1_wall_mask=wall,
        unknown_after_step1=~(free | wall),
        door_mask=door,
        resolution_m=0.05,
        config=LineExtensionConfig(min_extension_m=0.40, max_extension_m=1.60, max_probe_m=1.60),
        reject_if_intersects_door=True,
        door_intersection_dilation_cells=0,
    )

    assert any(hit.reject_reason == "reject_extension_hits_door" for hit in hits)


def test_step2_extension_rejects_intersection_with_door():
    shape = (24, 40)
    free = np.zeros(shape, dtype=bool)
    free[10, 16:25] = True
    wall = np.zeros(shape, dtype=bool)
    wall[10, 25] = True
    door = np.zeros(shape, dtype=bool)
    door[10, 20] = True

    hits, _debug = extend_step2_wall_lines(
        filtered_lines=[_line()],
        free_after_step1=free,
        step1_wall_mask=wall,
        unknown_after_step1=~(free | wall),
        door_mask=door,
        resolution_m=0.05,
        config=LineExtensionConfig(min_extension_m=0.40, max_extension_m=1.60, max_probe_m=1.60),
        reject_if_intersects_door=True,
        door_intersection_dilation_cells=0,
    )

    assert any(hit.reject_reason == "reject_extension_intersects_door" for hit in hits)


def test_step2_extension_rejects_split_with_side_width_1_to_3_cells():
    free = np.ones((10, 16), dtype=bool)
    candidate = SeparatorCandidate(
        candidate_id=1,
        kind="line_extension_door_neck",
        p0_rc=np.asarray([3, 1], dtype=np.float32),
        p1_rc=np.asarray([3, 14], dtype=np.float32),
        theta=0.0,
        length_m=0.70,
        confidence=1.0,
        source_segment_ids=[1],
    )
    cfg = TopologyTestConfig(
        min_split_area_m2=0.01,
        min_new_component_area_cells=1,
        min_new_component_width_m=0.0,
        reject_if_side_width_cells_leq=3,
        separator=SeparatorAnchorConfig(require_two_anchors=False),
    )

    accepted, rejected, _sep, _labels, _debug = greedily_select_separators(
        [candidate],
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        corridor_skeleton=np.zeros_like(free),
        resolution_m=0.05,
        config=cfg,
    )

    assert not accepted
    assert rejected[0].reject_reason == "reject_split_tiny_side_width_1_to_3_cells"


def test_final_partition_uses_4_connectivity_not_diagonal_merge():
    shape = (5, 5)
    profile = _profile(shape)
    _mark_free(profile, [(1, 1), (2, 2)])
    nav_free = profile.free_ray_count.any(axis=0)
    cfg = HeightProfileDoorWallRoomSegConfig.from_mapping(
        {"height_profile_evidence": {"free_remove_island_max_area_cells": 0}, "min_observed_free_cells": 1}
    )

    result = run_height_profile_door_wall_roomseg(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=nav_free,
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=~nav_free,
        height_profile=profile,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        roomseg_ray_evidence=None,
        resolution_m=0.05,
        config=cfg,
    )

    assert len([v for v in np.unique(result.room_label_map) if int(v) > 0]) == 2


def test_small_room_not_merged_by_default():
    shape = (8, 8)
    profile = _profile(shape)
    _mark_free(profile, [(1, 1), (1, 2), (6, 6)])
    nav_free = profile.free_ray_count.any(axis=0)
    cfg = HeightProfileDoorWallRoomSegConfig.from_mapping(
        {"height_profile_evidence": {"free_remove_island_max_area_cells": 0}, "min_observed_free_cells": 1}
    )

    result = run_height_profile_door_wall_roomseg(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=nav_free,
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=~nav_free,
        height_profile=profile,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        roomseg_ray_evidence=None,
        resolution_m=0.05,
        config=cfg,
    )

    assert result.debug["height_profile_room_count"] == 2
    assert result.debug["merge_small_components_enabled"] is False


def test_segmenter_update_handles_debug_separator_mask_without_numpy_truth_value_error():
    shape = (5, 5)
    profile = _profile(shape)
    _mark_free(profile, [(1, 1), (2, 2)])
    nav_free = profile.free_ray_count.any(axis=0)
    cfg = HeightProfileDoorWallRoomSegConfig.from_mapping(
        {"height_profile_evidence": {"free_remove_island_max_area_cells": 0}, "min_observed_free_cells": 1}
    )
    segmenter = HeightProfileDoorWallRoomSegmenter(cfg)

    rooms = segmenter.update(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=nav_free,
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=~nav_free,
        step=3,
        height_profile=profile,
    )

    assert len(rooms) == 2
    assert all(int(room.metadata["accepted_separator_count"]) == 0 for room in rooms)
