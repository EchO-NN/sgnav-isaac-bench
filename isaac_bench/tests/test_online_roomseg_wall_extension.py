from __future__ import annotations

import json

import numpy as np

from isaac_bench.mapping.online_roomseg import OnlineRoseStyleConfig, run_online_rose_style_roomseg
from isaac_bench.mapping.online_roomseg.separator_candidates import (
    NoiseWallGapFillConfig,
    SeparatorCandidate,
    build_l_corner_door_neck_candidates,
    fill_noise_wall_gaps_from_runs,
)
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, evaluate_candidate
from isaac_bench.mapping.online_roomseg.wall_lines import (
    FilteredWallLine,
    SnappedWallRun,
    extract_line_supported_walls,
    snap_wall_segments_to_runs,
)
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


def _wall_run(run_id: int, axis: str, line: int, start: int, end: int) -> SnappedWallRun:
    return SnappedWallRun(
        run_id=run_id,
        axis=axis,
        line=line,
        start=start,
        end=end,
        support_cells=np.empty((0, 2), dtype=np.int32),
        source_segment_ids=[run_id],
        support_ratio=1.0,
        length_m=float(end - start + 1) * 0.1,
        lateral_std_cells=0.0,
        confidence=1.0,
    )


def _filtered_line(line_id: int, p0: tuple[int, int], p1: tuple[int, int]) -> FilteredWallLine:
    p0_rc = np.asarray(p0, dtype=np.float32)
    p1_rc = np.asarray(p1, dtype=np.float32)
    delta = p1_rc - p0_rc
    theta = 0.0 if abs(float(delta[1])) >= abs(float(delta[0])) else float(np.pi / 2.0)
    return FilteredWallLine(
        line_id=line_id,
        p0_rc=p0_rc,
        p1_rc=p1_rc,
        theta=theta,
        normal_theta=float(theta + np.pi / 2.0),
        length_m=float(max(abs(int(delta[0])), abs(int(delta[1]))) + 1) * 0.1,
        support_ratio=1.0,
        mean_wall_score=1.0,
        thickness_m=0.1,
        source_segment_ids=[line_id],
        confidence=1.0,
    )


def test_l_corner_door_neck_candidate_generated_in_step2():
    free = np.zeros((30, 30), dtype=bool)
    free[10, 9:13] = True
    free[11:14, 12] = True
    unknown = np.zeros_like(free)
    lines = [
        _filtered_line(1, (10, 2), (10, 8)),
        _filtered_line(2, (14, 12), (24, 12)),
    ]

    candidates, debug = build_l_corner_door_neck_candidates(
        lines,
        free_clean=free,
        unknown_clean=unknown,
        resolution_m=0.1,
        start_id=7,
    )

    assert debug["candidate_count"] == 1
    candidate = candidates[0]
    assert candidate.candidate_id == 7
    assert candidate.kind == "line_extension_door_neck"
    assert candidate.debug["candidate_source"] == "l_corner_door_neck"
    assert np.isclose(candidate.length_m, 0.8)
    mask = candidate.mask(free.shape)
    assert mask[10, 9]
    assert mask[10, 12]
    assert mask[13, 12]
    assert not mask[10, 8]
    assert not mask[14, 12]


def test_long_plain_wall_gap_is_not_noise_filled():
    runs = [
        _wall_run(1, "vertical", 10, 5, 8),
        _wall_run(2, "vertical", 10, 15, 20),
    ]

    fill, debug = fill_noise_wall_gaps_from_runs(
        runs,
        shape=(30, 30),
        resolution_m=0.1,
        config=NoiseWallGapFillConfig(
            max_gap_m=0.4,
            corner_gap_enabled=True,
            corner_thickness_cells=0,
        ),
    )

    assert debug["filled_gap_count"] == 0
    assert not np.any(fill[9:15, 10])


def test_l_corner_wall_gap_is_noise_filled():
    runs = [
        _wall_run(1, "horizontal", 10, 4, 8),
        _wall_run(2, "vertical", 10, 11, 18),
    ]

    fill, debug = fill_noise_wall_gaps_from_runs(
        runs,
        shape=(30, 30),
        resolution_m=0.1,
        config=NoiseWallGapFillConfig(
            max_gap_m=0.4,
            corner_gap_enabled=True,
            corner_thickness_cells=0,
        ),
    )

    assert debug["filled_gap_count"] == 1
    assert debug["corner_filled_gap_count"] == 1
    assert debug["events"][0]["kind"] == "l_corner_gap"
    assert np.isclose(debug["events"][0]["gap_m"], 0.3)
    assert fill[10, 9]
    assert fill[10, 10]
    assert not fill[11, 10]


def test_l_corner_wall_gap_uses_corner_thickness():
    runs = [
        _wall_run(1, "horizontal", 10, 4, 8),
        _wall_run(2, "vertical", 10, 11, 18),
    ]

    fill, debug = fill_noise_wall_gaps_from_runs(
        runs,
        shape=(30, 30),
        resolution_m=0.1,
        config=NoiseWallGapFillConfig(
            max_gap_m=0.4,
            corner_gap_enabled=True,
            corner_thickness_cells=1,
        ),
    )

    assert debug["filled_gap_count"] == 1
    assert debug["corner_thickness_cells"] == 1
    assert debug["events"][0]["corner_thickness_cells"] == 1
    assert fill[10, 9]
    assert fill[10, 10]
    assert fill[11, 10]


def test_l_corner_wall_gap_at_0p4m_is_not_noise_filled():
    runs = [
        _wall_run(1, "horizontal", 10, 4, 7),
        _wall_run(2, "vertical", 10, 11, 18),
    ]

    fill, debug = fill_noise_wall_gaps_from_runs(
        runs,
        shape=(30, 30),
        resolution_m=0.1,
        config=NoiseWallGapFillConfig(
            max_gap_m=0.4,
            corner_gap_enabled=True,
            corner_thickness_cells=0,
        ),
    )

    assert debug["filled_gap_count"] == 0
    assert not np.any(fill)


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
    assert set(_accepted_kinds(result)) & {"doorway_virtual_cut", "line_extension_door_neck", "extension_intersection_cut"}
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

    assert set(_accepted_kinds(result)) & {"doorway_virtual_cut", "line_extension_door_neck", "extension_intersection_cut"}
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
    pass2_completion = result.layers["pass2_line_extension_completion"]
    wall_after_extension = result.layers["wall_target_after_line_extension"]

    assert np.count_nonzero(virtual_separator) > 0
    assert result.debug["separator_report"]["accepted_virtual_boundary_cells"] > 0
    assert np.array_equal(wall_after_extension, result.layers["wall_target_after_noise_gap_fill"])
    if np.count_nonzero(pass2_completion) > 0:
        assert np.count_nonzero(pass2_completion & ~wall_after_extension) > 0
    assert result.debug["separator_report"]["wall_target_after_line_extension_source"] == "wall_target_after_noise_gap_fill_only"
    assert np.any(virtual_separator & (result.room_label_map <= 0))
    assert result.debug["separator_report"]["labels_outside_free_cells"] == 0
    assert result.debug["separator_report"]["labels_in_unknown_cells"] == 0
    assert result.debug["virtual_separator_label_fill"]["filled_cell_count"] == 0
    assert result.debug["virtual_separator_label_fill"]["enabled"] is False


def test_do_not_split_corridor():
    free = np.zeros((36, 70), dtype=bool)
    occupied = np.zeros_like(free)
    free[16:21, 5:65] = True

    result = _run(free, occupied)

    assert _room_count(result) == 1
    assert not result.accepted_candidates


def test_wall_supported_corridor_room_neck_splits_side_room():
    free = np.zeros((60, 80), dtype=bool)
    occupied = np.zeros_like(free)
    free[30:35, 5:75] = True
    free[10:30, 28:43] = True
    occupied[10:30, 27] = True
    occupied[10:30, 43] = True
    occupied[9, 27:44] = True

    result = _run(free, occupied)
    report = result.debug["separator_report"]

    assert report["accepted_count_by_kind"].get("corridor_room_neck_cut", 0) == 1
    assert _room_count(result) == 2


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


def test_topology_tiny_fragment_gate_ignores_remote_preexisting_noise():
    free = np.zeros((40, 50), dtype=bool)
    free[8:28, 8:29] = True
    for cell in [(1, 1), (1, 4), (3, 2), (35, 45), (37, 43)]:
        free[cell] = True
    candidate = SeparatorCandidate(
        candidate_id=2,
        kind="doorway_virtual_cut",
        p0_rc=np.asarray([8, 18], dtype=np.float32),
        p1_rc=np.asarray([27, 18], dtype=np.float32),
        theta=float(np.pi / 2.0),
        length_m=2.0,
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
                "max_tiny_fragment_count": 3,
                "separator": {"max_anchor_extension_m": 0.0, "require_two_anchors": False, "doorway_thickness_cells": 0},
            }
        ),
    )

    assert ok, reason
    assert debug["tiny_fragment_count"] == 0
    assert debug["global_tiny_fragment_count"] == 5
    assert debug["tiny_fragment_scope"] == "candidate_adjacent_touched_components"


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
    assert report["line_extension_passes_requested"] == 2
    assert report["pass2_extension_enabled"] is True
    assert report["pass1_extension_count"] > 0
    assert report["pass1_candidate_count"] >= 1
    assert report["pass2_extension_count"] > 0
    assert report["pass2_candidate_count"] >= 1
    assert report["wall_segment_count"] > 0
    assert report["snapped_wall_run_count"] > 0
    assert report["merged_wall_run_count"] > 0
    assert "candidate_count_by_kind" in report
    assert set(report["accepted_count_by_kind"]) & {"doorway_virtual_cut", "line_extension_door_neck", "extension_intersection_cut"}
    assert "accepted_count_by_kind" in report
    assert "rejected_count_by_reason" in report
    assert "unanchored_candidate_count" in report
    assert "largest_room_area_ratio" in report
    assert all("accepted" in item and "reject_reason" in item for item in report["candidates"])
