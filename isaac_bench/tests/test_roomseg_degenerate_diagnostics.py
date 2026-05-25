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
        side_support_min_free_cells=2,
        side_support_min_ratio=0.05,
        side_support_balance_min=0.05,
        line_free_ratio_min=0.50,
        line_unknown_ratio_max=0.05,
        line_mid_wall_ratio_max=0.25,
        candidate_score_min=0.05,
        min_room_area_m2=0.10,
        small_component_area_m2=0.01,
        degenerate_large_free_min_cells=10,
        pre_extension_door_pattern=DoorPatternConfig(enabled=False),
    )
    base.update(overrides)
    return VFGCConfig(**base)


def test_large_single_free_component_without_boundaries_reports_degenerate_one_room():
    free = np.zeros((48, 48), dtype=bool)
    free[5:43, 5:43] = True
    wall = np.zeros_like(free)
    unknown = np.zeros_like(free)

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
        navigation_free_mask=free,
    )

    assert result.debug["raw_free_component_count"] == 1
    assert result.debug["virtual_boundary_cells"] == 0
    assert result.debug["segmentation_degenerate_one_room"] is True
    assert result.debug["corridor_diagnostics"]["endpoint_count"] == 0


def test_terminal_ray_wall_evidence_enters_wall_mask_and_endpoint_diagnostics():
    free = np.zeros((48, 48), dtype=bool)
    free[5:43, 5:43] = True
    wall = np.zeros_like(free)
    unknown = np.zeros_like(free)
    terminal = np.zeros_like(free, dtype=np.uint16)
    terminal[12:36, 24] = 2

    result = run_vertical_free_gap_closure_roomseg(
        free_mask=free,
        wall_mask=wall,
        unknown_mask=unknown,
        resolution_m=0.1,
        config=_cfg(),
        navigation_free_mask=free,
        roomseg_ray_evidence={"terminal_wall_count": terminal},
    )

    assert result.debug["terminal_wall_used_as_roomseg_wall"] is True
    assert result.debug["sanitized_wall_cells"] >= int(np.count_nonzero(terminal))
    assert np.count_nonzero(result.debug["roomseg_sanitized_wall"] & terminal.astype(bool)) == int(np.count_nonzero(terminal))
    assert result.debug["corridor_diagnostics"]["endpoint_count"] > 0
