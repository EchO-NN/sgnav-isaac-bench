from __future__ import annotations

import json

import numpy as np

from isaac_bench.mapping.online_roomseg import OnlineRoseStyleConfig, run_online_rose_style_roomseg
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


def test_short_wall_gap_completion():
    free = np.zeros((50, 60), dtype=bool)
    occupied = np.zeros_like(free)
    free[5:45, 5:55] = True
    occupied[5:45, 30] = True
    free[occupied] = False
    occupied[24:26, 30] = False
    free[24:26, 30] = True

    result = _run(free, occupied)

    assert "physical_wall_completion" in _accepted_kinds(result)
    assert _room_count(result) == 2


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

    assert "doorway_virtual_cut" in _accepted_kinds(result)
    assert _room_count(result) == 2
    assert np.array_equal(occupied, navigation_obstacle_before)
    assert result.debug["navigation_obstacle_written"] is False


def test_do_not_split_corridor():
    free = np.zeros((36, 70), dtype=bool)
    occupied = np.zeros_like(free)
    free[16:21, 5:65] = True

    result = _run(free, occupied)

    assert _room_count(result) == 1
    assert "reject_split_main_corridor_axis" in _rejected_reasons(result)


def test_corridor_to_bedroom_neck_cut():
    free = np.zeros((60, 80), dtype=bool)
    occupied = np.zeros_like(free)
    free[30:35, 5:75] = True
    free[10:30, 28:43] = True

    result = _run(free, occupied)

    assert "corridor_room_neck_cut" in _accepted_kinds(result)
    assert _room_count(result) == 2
    assert "reject_split_main_corridor_axis" in _rejected_reasons(result)


def test_open_plan_no_false_cut():
    free = np.zeros((60, 80), dtype=bool)
    occupied = np.zeros_like(free)
    free[8:52, 8:72] = True

    result = _run(free, occupied)

    assert _room_count(result) == 1
    assert not result.accepted_candidates


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
    assert all("accepted" in item and "reject_reason" in item for item in report["candidates"])
