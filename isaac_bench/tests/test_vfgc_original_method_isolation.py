import numpy as np

from isaac_bench.mapping.door_pattern_detector import DoorPatternConfig
from isaac_bench.mapping.roomseg_input_sanitizer import RoomsegInputSanitizerConfig
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
    assert np.array_equal(result.accepted_closure_map, result.debug["wall_extension_boundary_mask"])
    assert not result.debug["door_completion_boundary_mask"].any()


def test_wall_extension_and_detected_door_completion_boundaries_are_separate_sources():
    free = np.zeros((24, 32), dtype=bool)
    wall = np.zeros_like(free)
    unknown = np.zeros_like(free)
    wall[11, 8] = True
    wall[11, 22] = True
    free[11, 9:22] = True
    free[12:16, 14] = True
    wall[11, 15:17] = True
    free[11, 15:17] = False

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(
            roomseg_input_sanitizer=RoomsegInputSanitizerConfig(
                wall_core_min_component_cells=1,
                terminal_wall_splat_radius_cells=0,
            ),
            pre_extension_door_pattern=DoorPatternConfig(
                strict_rule_enabled=False,
                partial_seed_enabled=True,
                partial_seed_min_component_cells=2,
                partial_min_free_run_cells=2,
                partial_min_occupied_seed_cells=1,
                partial_max_unknown_below_cells=8,
                merge_nearby_door_cells=0,
                door_cut_lateral_radius_cells=0,
                door_line_inner_wall_ratio_max=0.4,
                door_line_max_width_m=2.0,
            ),
        ),
        navigation_free_mask=free | wall,
    )

    wall_extension = result.debug["wall_extension_boundary_mask"]
    door_completion = result.debug["door_completion_boundary_mask"]

    assert result.debug["wall_extension_boundary_cells"] == int(np.count_nonzero(wall_extension))
    assert result.debug["door_completion_boundary_cells"] == int(np.count_nonzero(door_completion))
    assert result.debug["door_completion_boundary_cells"] > 0
    assert np.array_equal(result.accepted_closure_map, wall_extension)
    assert np.array_equal(door_completion, result.debug["pre_extension_door_cut_mask"])
    assert np.array_equal(result.virtual_boundary_map, wall_extension | door_completion)
    assert set(np.unique(result.debug["virtual_boundary_source_map"])) == {0, 3}
