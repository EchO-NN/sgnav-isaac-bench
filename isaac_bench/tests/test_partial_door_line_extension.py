import numpy as np

from isaac_bench.mapping.door_pattern_detector import DoorPatternConfig, detect_pre_extension_doors


def _partial_cfg(**overrides):
    base = dict(
        strict_rule_enabled=False,
        partial_seed_enabled=True,
        partial_min_free_run_cells=2,
        partial_min_occupied_seed_cells=1,
        partial_seed_min_component_cells=2,
        merge_nearby_door_cells=0,
        door_cut_lateral_radius_cells=0,
        door_line_inner_wall_ratio_max=0.40,
        door_line_max_width_m=2.0,
    )
    base.update(overrides)
    return DoorPatternConfig(**base)


def test_partial_door_seed_extends_to_structural_wall_and_cuts_only_free():
    free = np.zeros((24, 32), dtype=bool)
    occ = np.zeros_like(free)
    unk = np.zeros_like(free)
    occ[11, 8] = True
    occ[11, 22] = True
    free[11, 9:22] = True
    free[12:16, 14] = True
    occ[11, 15:17] = True
    free[11, 15:17] = False

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=_partial_cfg(),
        roi=(11, 8, 15, 22),
        resolution_m=0.1,
    )

    assert int(np.count_nonzero(result.partial_door_seed_mask)) == 2
    assert result.debug["partial_door_line_accepted_count"] == 1
    assert result.partial_door_line_mask[11, 8]
    assert result.partial_door_line_mask[11, 22]
    assert np.count_nonzero(result.partial_door_extension_cut_mask) > 0
    assert np.count_nonzero(result.partial_door_extension_cut_mask & ~free) == 0
    assert np.array_equal(result.door_cut_mask, result.partial_door_extension_cut_mask)


def test_partial_door_seed_rejects_unknown_below_seed():
    free = np.zeros((18, 18), dtype=bool)
    occ = np.zeros_like(free)
    unk = np.zeros_like(free)
    free[12:16, 8] = True
    unk[11, 8] = True
    occ[10, 8:10] = True

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=_partial_cfg(partial_seed_min_component_cells=1, partial_max_unknown_below_cells=0),
        roi=(10, 8, 15, 10),
        resolution_m=0.1,
    )

    assert not result.partial_door_seed_mask.any()
    assert not result.partial_door_extension_cut_mask.any()
    assert result.debug["partial_door_line_candidate_count"] == 0


def test_partial_door_extension_rejects_paths_that_hit_another_door_seed():
    free = np.zeros((24, 36), dtype=bool)
    occ = np.zeros_like(free)
    unk = np.zeros_like(free)
    occ[11, 6] = True
    occ[11, 30] = True
    free[11, 7:30] = True
    free[12:16, 12] = True
    free[12:16, 24] = True
    occ[11, 13:15] = True
    occ[11, 22:24] = True
    free[11, 13:15] = False
    free[11, 22:24] = False

    result = detect_pre_extension_doors(
        free_mask=free,
        occupied_mask=occ,
        unknown_mask=unk,
        config=_partial_cfg(other_door_intersection_radius_cells=0, door_line_max_width_m=3.0),
        roi=(11, 6, 15, 30),
        resolution_m=0.1,
    )

    assert result.debug["partial_door_line_accepted_count"] == 0
    assert result.debug["partial_door_line_reject_reason_counts"]["extension_endpoint_is_other_door"] == 2
    assert result.rejected_door_extension_mask.any()
    assert not result.partial_door_extension_cut_mask.any()
