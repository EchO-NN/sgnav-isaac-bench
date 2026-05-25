import numpy as np

from isaac_bench.mapping.door_pattern_detector import DoorPatternConfig
from isaac_bench.mapping.vertical_free_gap_closure_roomseg import VFGCConfig, run_vertical_free_gap_closure_roomseg


def _cfg(**overrides):
    base = dict(
        resolution_m=0.1,
        close_max_gap_m=1.5,
        close_min_gap_m=0.1,
        virtual_boundary_radius_m=0.06,
        wall_min_component_cells=1,
        free_min_component_cells=1,
        wall_micro_close_radius_cells=0,
        side_support_min_free_cells=3,
        side_support_min_ratio=0.10,
        side_support_balance_min=0.10,
        line_free_ratio_min=0.50,
        line_unknown_ratio_max=0.05,
        line_mid_wall_ratio_max=0.25,
        candidate_score_min=0.05,
        min_room_area_m2=0.50,
        small_component_area_m2=0.10,
    )
    base.update(overrides)
    return VFGCConfig(**base)


def test_original_step1_step2_topology_prune_does_not_use_pre_door_cuts():
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:110] = True
    wall = np.zeros_like(free)
    wall[10:70, 59:61] = True
    free[wall] = False
    wall[35:45, 59:61] = False
    free[35:45, 59:61] = True
    unknown = np.zeros_like(free)

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(pre_extension_door_pattern=DoorPatternConfig(enabled=False)),
    )

    assert result.debug["original_step1_step2_topology_prune_uses_pre_doors"] is False
    assert result.debug["final_virtual_boundary_includes_pre_doors"] is True
    assert result.debug["num_accepted_closures"] >= 1
    assert result.debug["original_step1_step2_virtual_boundary_cells"] == int(
        np.count_nonzero(result.debug["original_step1_step2_virtual_boundary_map"])
    )
    assert np.array_equal(result.virtual_boundary_map, result.debug["original_step1_step2_virtual_boundary_map"])
