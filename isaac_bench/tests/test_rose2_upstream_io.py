from __future__ import annotations

import json

import numpy as np

from isaac_bench.mapping.rose2_upstream_io import export_rose2_upstream_input, normalize_rose2_masks


def test_export_rose2_upstream_input_writes_replay_bundle(tmp_path):
    free = np.zeros((20, 30), dtype=bool)
    free[4:16, 4:26] = True
    occupied = np.zeros_like(free)
    occupied[4:16, 14] = True
    free[8:12, 14] = True
    unknown = ~(free | occupied)
    bundle = export_rose2_upstream_input(
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown,
        resolution_m=0.10,
        out_dir=tmp_path,
        stem="step",
        encoding="auto",
    )

    assert bundle.metric_map_path.exists()
    assert bundle.orebro_input_path.exists()
    assert bundle.free_png_path.exists()
    assert bundle.occupied_png_path.exists()
    assert bundle.unknown_png_path.exists()
    assert bundle.overlay_png_path.exists()
    assert bundle.npz_path.exists()
    payload = json.loads(bundle.summary_json_path.read_text())
    assert payload["chosen_encoding"] in {
        "source_black_wall_white_free",
        "source_white_wall_black_free",
        "source_metric_binary_with_unknown_border",
    }
    assert payload["free_cells"] == int(np.count_nonzero(free & ~occupied))
    assert payload["occupied_cells"] == int(np.count_nonzero(occupied))
    assert payload["resolution_m"] == 0.10


def test_normalize_rose2_masks_uses_occupied_free_unknown_priority():
    free = np.zeros((5, 5), dtype=bool)
    occupied = np.zeros_like(free)
    unknown = np.ones_like(free)
    free[2, 2] = True
    occupied[2, 2] = True
    unknown[2, 2] = True

    norm_free, norm_occ, norm_unknown, debug = normalize_rose2_masks(
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown,
    )

    assert norm_occ[2, 2]
    assert not norm_free[2, 2]
    assert not norm_unknown[2, 2]
    assert debug["free_occupied_overlap"] == 1
    assert debug["occupied_unknown_overlap"] == 1
