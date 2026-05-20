from __future__ import annotations

import json

import numpy as np

from isaac_bench.mapping.online_roomseg import OnlineRoseStyleConfig, run_online_rose_style_roomseg
from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorCandidate
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, evaluate_candidate
from isaac_bench.mapping.online_roomseg.wall_lines import extract_line_supported_walls, snap_wall_segments_to_runs
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


def _vp(free: np.ndarray, occupied: np.ndarray) -> VerticalProfileMap:
    observed = np.asarray(free, dtype=bool) | np.asarray(occupied, dtype=bool)
    vp = VerticalProfileMap(
        band_names=("roomseg_0p2_2p0",),
        band_ranges_m=((0.2, 2.0),),
        occupied_count=np.zeros((1, *free.shape), dtype=np.uint16),
        free_ray_count=np.zeros((1, *free.shape), dtype=np.uint16),
        observed_count=np.zeros((1, *free.shape), dtype=np.uint16),
        unknown_count=np.zeros((1, *free.shape), dtype=np.uint16),
    )
    vp.free_ray_count[0][free] = 1
    vp.occupied_count[0][occupied] = 1
    vp.observed_count[0][observed] = 1
    vp.unknown_count[0][~observed] = 1
    return vp


def _run(free: np.ndarray, occupied: np.ndarray, *, tmp_path=None):
    cfg_data = {
        "online_roomseg": {
            "debug": {"save_layers": bool(tmp_path), "save_candidate_json": bool(tmp_path)},
            "debug_dir": str(tmp_path / "online_roomseg") if tmp_path else "debug/test_online_roomseg",
            "topology_test": {"min_split_area_m2": 0.2},
            "corridor": {"min_length_m": 1.0, "max_width_m": 0.8},
            "corridor_room_neck_cut": {"max_neck_width_m": 1.8, "min_room_side_area_m2": 0.4},
        }
    }
    cfg = OnlineRoseStyleConfig.from_mapping(cfg_data, resolution_m=0.1)
    unknown = ~(free | occupied)
    return run_online_rose_style_roomseg(
        occupancy_map=occupied,
        observed_free_mask=free,
        obstacle_mask=occupied,
        unknown_mask=unknown,
        vertical_profile=_vp(free, occupied),
        config=cfg,
        step=0,
    )


def _room_count(result) -> int:
    return int(len([v for v in np.unique(result.room_label_map) if int(v) > 0]))


def _accepted_kinds(result) -> list[str]:
    return [candidate.kind for candidate in result.accepted_candidates]


def _rejected_reasons(result) -> list[str]:
    return [candidate.reject_reason for candidate in result.rejected_candidates]


def test_short_wall_gap_below_door_range_is_closed_before_door_logic():
    free = np.zeros((50, 60), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:45, 5:55] = True
    occupied[5:45, 30] = True
    free[occupied] = False
    occupied[24:26, 30] = False
    free[24:26, 30] = True

    result = _run(free, occupied)
    report = result.debug["separator_report"]

    assert "line_extension_door_neck" not in _accepted_kinds(result)
    assert report["noise_wall_gap_fill_count"] >= 0
    assert np.any(result.layers["wall_target_after_noise_gap_fill"][24:26, 30])
    assert _room_count(result) == 2


def test_exact_0p4m_wall_gap_is_not_noise_filled():
    free = np.zeros((50, 60), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:45, 5:55] = True
    occupied[5:45, 30] = True
    free[occupied] = False
    occupied[23:27, 30] = False
    free[23:27, 30] = True

    result = _run(free, occupied)
    report = result.debug["separator_report"]

    assert report["noise_wall_gap_fill_count"] == 0
    assert not np.any(result.layers["noise_wall_gap_fill"][23:27, 30])


def test_jittered_wall_gap_completion_snaps_and_splits():
    free = np.zeros((55, 65), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:50, 5:60] = True
    occupied[25, 5:24] = True
    occupied[25, 30:60] = True
    free[occupied] = False
    free[25, 24:30] = True

    result = _run(free, occupied)
    report = result.debug["separator_report"]

    assert report["snapped_wall_run_count"] >= 2
    assert "line_extension_door_neck" in _accepted_kinds(result)
    assert _room_count(result) == 2


def test_wavy_wall_is_snapped_into_run():
    wall = np.zeros((40, 70), dtype=bool)
    for col in range(8, 60):
        row = 20 + ((col // 4) % 3) - 1
        wall[row, col] = True
    segments, _ = extract_line_supported_walls(
        wall,
        resolution_m=0.1,
        config={"hough_enabled": False, "pca_enabled": True, "min_line_length_m": 0.5, "min_support_ratio": 0.2},
    )
    runs, debug = snap_wall_segments_to_runs(
        segments,
        wall,
        resolution_m=0.1,
        max_angle_to_axis_deg=25.0,
        support_band_cells=2,
        min_run_length_m=0.5,
        min_support_ratio=0.2,
    )

    assert segments
    assert runs
    assert debug["snapped_wall_run_count"] >= 1


def test_doorway_virtual_cut_not_navigation_obstacle():
    free = np.zeros((50, 60), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:45, 5:55] = True
    occupied[5:45, 30] = True
    free[occupied] = False
    occupied[20:29, 30] = False
    free[20:29, 30] = True
    navigation_obstacle_before = occupied.copy()

    result = _run(free, occupied)

    assert "line_extension_door_neck" in _accepted_kinds(result)
    assert _room_count(result) == 2
    assert np.array_equal(occupied, navigation_obstacle_before)
    assert result.debug["navigation_obstacle_written"] is False


def test_virtual_separator_cells_are_absorbed_into_room_masks():
    free = np.zeros((50, 60), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:45, 5:55] = True
    occupied[5:45, 30] = True
    free[occupied] = False
    occupied[20:29, 30] = False
    free[20:29, 30] = True

    result = _run(free, occupied)
    structural = result.layers["structural_wall_free_overlap"]
    virtual_separator = result.layers["accepted_separators_after_corridor_merge"] & result.layers["free_clean"] & ~structural

    assert np.count_nonzero(virtual_separator) > 0
    assert not np.any(virtual_separator & (result.room_label_map <= 0))
    assert result.debug["virtual_separator_label_fill"]["filled_cell_count"] > 0


def test_do_not_split_corridor():
    free = np.zeros((36, 70), dtype=bool)
    occupied = np.zeros_like(free)
    free[16:21, 5:65] = True

    result = _run(free, occupied)

    assert _room_count(result) == 1
    assert not result.accepted_candidates


def test_corridor_to_bedroom_without_wall_line_has_no_hard_split():
    free = np.zeros((60, 80), dtype=bool)
    occupied = np.zeros_like(free)
    free[30:35, 5:75] = True
    free[10:30, 28:43] = True

    result = _run(free, occupied)

    assert "corridor_room_neck_cut" not in _accepted_kinds(result)
    assert _room_count(result) == 1


def test_open_plan_no_false_cut():
    free = np.zeros((60, 80), dtype=bool)
    occupied = np.zeros_like(free)
    free[8:52, 8:72] = True

    result = _run(free, occupied)

    assert _room_count(result) == 1
    assert not result.accepted_candidates


def test_unanchored_separator_rejected():
    free = np.zeros((40, 50), dtype=bool)
    free[5:35, 5:45] = True
    candidate = SeparatorCandidate(
        candidate_id=1,
        kind="single_sided_wall_extension",
        p0_rc=np.asarray([20, 20], dtype=np.float32),
        p1_rc=np.asarray([20, 28], dtype=np.float32),
        theta=0.0,
        length_m=0.9,
        confidence=1.0,
        source_segment_ids=[],
    )

    ok, reason, _mask, debug = evaluate_candidate(
        candidate,
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        current_separator_map=np.zeros_like(free),
        resolution_m=0.1,
        config=TopologyTestConfig.from_mapping(
            {
                "min_split_area_m2": 0.1,
                "separator": {"max_anchor_extension_m": 0.3, "require_two_anchors": True},
            }
        ),
    )

    assert not ok
    assert reason == "reject_unanchored_separator"
    assert debug["anchor_score"] == 0.0


def test_line_extension_cut_uses_middle_contiguous_free_run_only():
    free = np.zeros((15, 30), dtype=bool)
    free[7, 2:8] = True
    free[7, 12:21] = True
    candidate = SeparatorCandidate(
        candidate_id=3,
        kind="line_extension_door_neck",
        p0_rc=np.asarray([7, 2], dtype=np.float32),
        p1_rc=np.asarray([7, 20], dtype=np.float32),
        theta=0.0,
        length_m=1.9,
        confidence=1.0,
        source_segment_ids=[],
    )

    ok, reason, mask, debug = evaluate_candidate(
        candidate,
        free_clean=free,
        unknown_clean=np.zeros_like(free),
        wall_candidate_clean=np.zeros_like(free),
        current_separator_map=np.zeros_like(free),
        resolution_m=0.1,
        config=TopologyTestConfig.from_mapping(
            {
                "enabled": False,
                "separator": {"max_anchor_extension_m": 0.0, "require_two_anchors": False, "doorway_thickness_cells": 0},
            }
        ),
    )

    assert ok, reason
    assert debug["centered_extension_cut_mode"] == "middle_contiguous_free_run"
    assert debug["centered_extension_cut_p0"] == [7, 12]
    assert debug["centered_extension_cut_p1"] == [7, 20]
    assert not np.any(mask[7, 2:8])
    assert np.all(mask[7, 12:21])
    assert candidate.to_dict()["p0_rc"] == [7, 12]
    assert candidate.to_dict()["p1_rc"] == [7, 20]


def test_debug_report_written(tmp_path):
    free = np.zeros((50, 60), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:45, 5:55] = True
    occupied[5:45, 30] = True
    free[occupied] = False
    occupied[20:29, 30] = False
    free[20:29, 30] = True

    result = _run(free, occupied, tmp_path=tmp_path)
    paths = result.debug["online_roomseg_debug_paths"]
    report = json.loads(open(paths["separator_report"], "r", encoding="utf-8").read())

    assert report["accepted_count"] >= 1
    assert report["wall_segment_count"] > 0
    assert report["snapped_wall_run_count"] > 0
    assert report["merged_wall_run_count"] > 0
    assert "candidate_count_by_kind" in report
    assert "line_extension_door_neck" in report["accepted_count_by_kind"]
    assert "accepted_count_by_kind" in report
    assert "rejected_count_by_reason" in report
    assert "unanchored_candidate_count" in report
    assert "largest_room_area_ratio" in report
    assert all("accepted" in item and "reject_reason" in item for item in report["candidates"])
