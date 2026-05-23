from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg import OnlineRoseStyleConfig, run_online_rose_style_roomseg
from isaac_bench.mapping.online_roomseg.separator_candidates import LineExtensionConfig


def test_open_living_room_without_valid_extension_stays_single_room():
    free = np.zeros((70, 70), dtype=bool)
    free[10:60, 10:60] = True
    wall = np.zeros_like(free)
    wall[8, 10:20] = True
    wall[8, 50:60] = True
    unknown = ~(free | wall)
    cfg = OnlineRoseStyleConfig(
        resolution_m=0.1,
        min_observed_free_cells=1,
        line_extension=LineExtensionConfig(max_extension_m=1.0, max_probe_m=1.1),
    )

    result = run_online_rose_style_roomseg(
        occupancy_map=wall,
        observed_free_mask=free,
        obstacle_mask=wall,
        unknown_mask=unknown,
        vertical_profile=None,
        config=cfg,
        step=0,
    )

    labels = [int(v) for v in np.unique(result.room_label_map) if int(v) > 0]
    assert len(labels) == 1
    assert result.debug["algorithm"] == "roomseg_evidence_line_closure_v3"
