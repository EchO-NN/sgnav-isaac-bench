import numpy as np

from isaac_bench.mapping.door_pattern_detector import DoorPatternConfig
from isaac_bench.mapping.online_roomseg.evidence_maps import FreeCleanConfig
from isaac_bench.mapping.online_roomseg.online_room_segmenter import (
    OnlineRoseStyleConfig,
    run_online_rose_style_roomseg,
)
from isaac_bench.mapping.online_roomseg.separator_candidates import LineExtensionConfig


def test_online_rose_style_inserts_pre_extension_door_cut_before_line_extension():
    free = np.zeros((18, 18), dtype=bool)
    occ = np.zeros_like(free)
    free[8:11, 5] = True
    occ[8, 6:9] = True
    unknown = ~(free | occ)

    cfg = OnlineRoseStyleConfig(
        resolution_m=0.1,
        min_observed_free_cells=1,
        free_clean=FreeCleanConfig(island_remove_max_area_cells=0, hole_fill_max_area_cells=0),
        line_extension=LineExtensionConfig(enabled=False, passes=1),
        pre_extension_door_pattern=DoorPatternConfig(
            min_free_run_cells=3,
            min_occupied_run_cells=3,
            merge_nearby_door_cells=0,
            door_cut_lateral_radius_cells=0,
        ),
    )

    result = run_online_rose_style_roomseg(
        occupancy_map=occ,
        observed_free_mask=free,
        obstacle_mask=occ,
        unknown_mask=unknown,
        vertical_profile=None,
        config=cfg,
        step=0,
    )

    assert result.debug["pre_extension_door_detection_inserted_before_wall_extension"] is True
    assert result.debug["pre_extension_door_num_accepted"] == 1
    assert result.layers["pre_extension_door_cut_mask"][8, 5]
    assert result.layers["pass2_virtual_targets"][8, 5]
    assert result.layers["accepted_separators"][8, 5]
    assert result.layers["step1_step2_accepted_separator_map"].shape == free.shape
    assert result.debug["navigation_obstacle_written"] is False
