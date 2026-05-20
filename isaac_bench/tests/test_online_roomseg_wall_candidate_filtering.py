from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.evidence_maps import WallCandidateConfig, clean_wall_candidate_map


def _clean(occupied: np.ndarray):
    free = np.zeros_like(occupied, dtype=bool)
    observed = occupied.copy()
    return clean_wall_candidate_map(
        vertical_free_raw=free,
        vertical_occupied_raw=occupied,
        vertical_observed_raw=observed,
        free_clean=free,
        resolution_m=0.05,
        config=WallCandidateConfig(
            mode="jitter_permissive",
            jitter_tolerance_cells=1,
            isolated_remove_max_area_cells=1,
            shape_gate_enabled=False,
        ),
    )


def test_wall_candidate_filtering_keeps_one_cell_jittered_wall_possibility():
    occupied = np.zeros((24, 32), dtype=bool)
    occupied[10, 4:12] = True
    occupied[11, 12:20] = True

    cleaned, debug = _clean(occupied)

    assert debug["mode"] == "jitter_permissive"
    assert debug["shape_gate_enabled"] is False
    assert int(np.count_nonzero(cleaned)) >= int(np.count_nonzero(occupied))
    assert np.all(cleaned[occupied])
    assert cleaned[10:12, 11:13].any()


def test_wall_candidate_filtering_removes_only_isolated_single_cell_noise():
    occupied = np.zeros((20, 20), dtype=bool)
    occupied[5, 5] = True
    occupied[12, 4:12] = True

    cleaned, debug = _clean(occupied)

    assert not bool(cleaned[5, 5])
    assert np.all(cleaned[12, 4:12])
    assert debug["removed_isolated_component_count"] == 1
