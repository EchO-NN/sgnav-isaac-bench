from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.separator_candidates import (
    DoorNeckConfig,
    LineExtensionConfig,
    build_door_neck_candidates_from_extensions,
    extend_wall_lines_once,
    rasterize_candidates,
)
from isaac_bench.tests.test_online_roomseg_line_extension import _line


def test_pass2_extension_can_hit_pass1_virtual_door_target():
    free = np.zeros((30, 50), dtype=bool)
    free[10, 11:20] = True
    free[12, 21:30] = True
    wall = np.zeros_like(free)
    wall[10, 5:11] = True
    wall[10, 20:25] = True
    wall[12, 30:35] = True
    unknown = ~(free | wall)
    cfg = LineExtensionConfig(min_extension_m=0.4, max_extension_m=1.6, free_ratio_min=0.65, unknown_ratio_max=0.1)

    pass1_hits, _ = extend_wall_lines_once(
        [_line(1, [10, 5], [10, 10])],
        free_clean=free,
        wall_target_mask=wall,
        virtual_target_mask=None,
        unknown_clean=unknown,
        resolution_m=0.1,
        pass_id=1,
        config=cfg,
    )
    pass1_candidates, _ = build_door_neck_candidates_from_extensions(
        [hit for hit in pass1_hits if hit.reject_reason is None],
        accepted_virtual_targets=None,
        resolution_m=0.1,
        config=DoorNeckConfig(min_confidence=0.4),
    )
    virtual = rasterize_candidates(pass1_candidates, free.shape, thickness_cells=1)

    pass2_hits, _ = extend_wall_lines_once(
        [_line(2, [12, 35], [12, 30])],
        free_clean=free,
        wall_target_mask=np.zeros_like(wall),
        virtual_target_mask=virtual,
        unknown_clean=unknown,
        resolution_m=0.1,
        pass_id=2,
        config=cfg,
    )
    accepted = [hit for hit in pass2_hits if hit.reject_reason is None]

    assert accepted
    assert accepted[0].hit_type == "virtual_door"
