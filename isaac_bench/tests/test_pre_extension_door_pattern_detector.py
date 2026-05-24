import numpy as np

from isaac_bench.mapping.door_pattern_detector import DoorPatternConfig, detect_pre_extension_doors


def _blank(h=12, w=12):
    free = np.zeros((h, w), dtype=bool)
    occ = np.zeros((h, w), dtype=bool)
    unk = np.zeros((h, w), dtype=bool)
    return free, occ, unk


def test_rule_b_free_occupied_unknown_accepts():
    free, occ, unk = _blank()
    c = 5
    free[8:11, c] = True
    occ[5:8, c] = True
    unk[3:5, c] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(min_free_run_cells=3, min_occupied_run_cells=3, min_unknown_run_cells=1),
        roi=(3, c, 10, c),
    )

    assert result.detected_door_mask.any()
    assert result.door_cut_mask.any()
    assert int(result.pattern_type_map.max()) == 2
    assert result.debug["pre_extension_door_rule_b_count"] == 1
    assert np.count_nonzero(result.door_cut_mask & ~free) == 0


def test_rule_b_rejects_unknown_below_occupied():
    free, occ, unk = _blank()
    c = 5
    free[9:11, c] = True
    unk[8, c] = True
    occ[5:8, c] = True
    unk[3:5, c] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(min_free_run_cells=2, min_occupied_run_cells=2),
        roi=(3, c, 10, c),
    )

    assert not result.detected_door_mask.any()
    assert not result.door_cut_mask.any()


def test_rule_b_rejects_occupied_after_unknown_tail():
    free, occ, unk = _blank()
    c = 5
    free[8:11, c] = True
    occ[6:8, c] = True
    unk[5, c] = True
    occ[4, c] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(min_free_run_cells=3, min_occupied_run_cells=2),
        roi=(4, c, 10, c),
    )

    assert not result.detected_door_mask.any()
    assert not result.door_cut_mask.any()


def test_rule_a_free_turn_occupied_accepts():
    free, occ, unk = _blank()
    free[8:11, 5] = True
    occ[8, 6:9] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(min_free_run_cells=3, min_occupied_run_cells=3, merge_nearby_door_cells=0),
        roi=(8, 5, 10, 8),
    )

    assert result.detected_door_mask.any()
    assert result.door_cut_mask.any()
    assert int(result.pattern_type_map.max()) == 1
    assert result.debug["pre_extension_door_rule_a_count"] == 1


def test_rule_a_rejects_unknown_on_turned_probe():
    free, occ, unk = _blank()
    free[8:11, 5] = True
    occ[8, 6:8] = True
    unk[8, 8] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(min_free_run_cells=3, min_occupied_run_cells=3),
        roi=(8, 5, 10, 8),
    )

    assert not result.detected_door_mask.any()
    assert not result.door_cut_mask.any()


def test_detector_does_not_mutate_inputs_and_disabled_returns_zero_masks():
    free, occ, unk = _blank()
    free[8:11, 5] = True
    occ[5:8, 5] = True
    unk[3:5, 5] = True
    before = (free.copy(), occ.copy(), unk.copy())

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(enabled=False),
        roi=(3, 5, 10, 5),
    )

    assert not result.detected_door_mask.any()
    assert not result.door_cut_mask.any()
    assert result.debug["pre_extension_door_detection_enabled"] is False
    assert np.array_equal(free, before[0])
    assert np.array_equal(occ, before[1])
    assert np.array_equal(unk, before[2])


def test_fallback_roi_ignores_full_map_unknown_extent():
    free, occ, unk = _blank(20, 20)
    unk[:, :] = True
    unk[8:18, 10] = False
    free[15:18, 10] = True
    occ[12:15, 10] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=DoorPatternConfig(
            min_free_run_cells=3,
            min_occupied_run_cells=3,
            unknown_tail_padding_cells=4,
            merge_nearby_door_cells=0,
        ),
    )

    assert result.detected_door_mask.any()
    assert result.debug["pre_extension_door_rois"] == [[8, 10, 17, 10]]
