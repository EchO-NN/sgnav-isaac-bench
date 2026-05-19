from __future__ import annotations

import numpy as np

from isaac_bench.mapping.vertical_free_roomseg import (
    VERTICAL_FREE_ROOMSEG_BACKEND,
    VerticalFreeRoomSegConfig,
    run_vertical_free_roomseg,
    save_vertical_free_roomseg_debug,
)


def _cfg(**overrides) -> VerticalFreeRoomSegConfig:
    data = dict(
        resolution_m=0.10,
        min_room_area_m2=0.20,
        min_room_free_cells=5,
        min_free_component_area_m2=0.01,
        seed_min_clearance_m=0.20,
        seed_min_distance_m=1.00,
        seed_min_area_m2=0.05,
        merge_small_area_m2=0.20,
        merge_thin_sliver_area_m2=0.10,
    )
    data.update(overrides)
    return VerticalFreeRoomSegConfig(**data)


def _run(free: np.ndarray, *, unknown: np.ndarray | None = None, **cfg_overrides):
    unknown_arr = np.zeros_like(free, dtype=bool) if unknown is None else np.asarray(unknown, dtype=bool)
    occupied = ~np.asarray(free, dtype=bool) & ~unknown_arr
    return run_vertical_free_roomseg(
        observed_free=np.asarray(free, dtype=bool),
        observed_occupied=occupied,
        unknown=unknown_arr,
        resolution_m=0.10,
        config=_cfg(**cfg_overrides),
    )


def _room_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(labels) if int(v) > 0]))


def test_single_rectangle_room_segments_as_one_room():
    free = np.zeros((80, 100), dtype=bool)
    free[10:70, 10:90] = True

    result = _run(free)

    assert result.debug["backend"] == VERTICAL_FREE_ROOMSEG_BACKEND
    assert _room_count(result.room_label_map) == 1
    assert np.count_nonzero((result.room_label_map > 0) & ~free) == 0


def test_two_rooms_with_one_meter_doorway_are_split():
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:50] = True
    free[10:70, 70:110] = True
    free[35:45, 50:70] = True

    result = _run(free)

    assert _room_count(result.room_label_map) == 2
    assert result.debug["doorway_boundary_count"] > 0
    assert np.count_nonzero((result.room_label_map > 0) & ~free) == 0


def test_wide_open_rectangle_does_not_oversegment():
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:110] = True

    result = _run(free)

    assert _room_count(result.room_label_map) == 1
    assert result.debug["open_merge_count"] == 0


def test_l_shaped_open_room_merges_after_open_boundary_check():
    free = np.zeros((80, 100), dtype=bool)
    free[10:60, 10:40] = True
    free[40:70, 10:80] = True

    result = _run(
        free,
        seed_min_distance_m=0.80,
        open_merge_min_boundary_width_m=0.80,
        open_merge_min_contact_length_m=0.20,
    )

    assert _room_count(result.room_label_map) == 1


def test_narrow_corridor_between_rooms_is_not_collapsed_to_one_blob():
    free = np.zeros((100, 120), dtype=bool)
    free[10:45, 10:45] = True
    free[55:90, 75:110] = True
    free[35:65, 45:75] = True

    result = _run(free, seed_min_distance_m=0.80)

    assert _room_count(result.room_label_map) >= 2
    assert result.debug["doorway_boundary_count"] > 0


def test_unknown_cells_are_never_labeled():
    free = np.zeros((60, 80), dtype=bool)
    free[10:50, 10:50] = True
    unknown = np.zeros_like(free, dtype=bool)
    unknown[20:40, 45:70] = True

    result = _run(free, unknown=unknown)

    assert np.count_nonzero(result.room_label_map[unknown] > 0) == 0
    assert result.debug["labels_in_unknown_cells"] == 0


def test_boundary_absorption_keeps_two_labels():
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:50] = True
    free[10:70, 70:110] = True
    free[35:45, 50:70] = True

    result = _run(free, keep_boundary_unlabeled=False)

    assert _room_count(result.room_label_map) == 2
    assert result.debug["boundary_absorbed_cells"] > 0


def test_isolated_wall_noise_does_not_explode_open_room():
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:110] = True
    free[30, 30] = False
    free[45, 80] = False
    free[55, 60] = False

    result = _run(free)

    assert _room_count(result.room_label_map) == 1


def test_debug_dump_contains_required_layers(tmp_path):
    free = np.zeros((50, 70), dtype=bool)
    free[10:40, 10:60] = True
    result = _run(free)

    dump = save_vertical_free_roomseg_debug(result=result, out_dir=tmp_path, step=3)

    paths = dump["paths"]
    assert paths["backend"] == VERTICAL_FREE_ROOMSEG_BACKEND
    assert (tmp_path / "vertical_free_step_000003.npz").exists()
    assert (tmp_path / "vertical_free_step_000003.summary.json").exists()
