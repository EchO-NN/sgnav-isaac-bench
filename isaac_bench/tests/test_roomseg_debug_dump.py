import json
from pathlib import Path

import numpy as np

from isaac_bench.debug.roomseg_layer_dump import ROOMSEG_SNAPSHOT_ARRAY_KEYS, save_roomseg_layer_dump
from isaac_bench.scripts.inspect_roomseg_debug import main as inspect_main


def test_roomseg_debug_dump_writes_headless_artifacts(tmp_path):
    shape = (12, 12)
    labels = np.zeros(shape, dtype=np.int32)
    labels[2:6, 2:6] = 1
    nav = np.zeros(shape, dtype=bool)
    nav[2:8, 2:8] = True
    vertical = np.zeros(shape, dtype=bool)
    vertical[2:6, 2:6] = True
    unknown = np.zeros(shape, dtype=bool)
    frontier = np.zeros(shape, dtype=bool)
    frontier[7, 4:7] = True
    room_debug = {
        "algorithm": "upstream_rose2_vertical_or_free",
        "source_mode": "declutter_reconstruct_mit",
        "navigation_free_room_domain": nav,
        "vertical_free_room_domain": vertical,
        "repaired_roomseg_free": vertical,
        "repaired_roomseg_occupied": ~nav,
        "repaired_roomseg_unknown": unknown,
        "final_room_label_map": labels,
        "context_room_label_map": labels,
        "boundary_map": np.zeros(shape, dtype=bool),
        "representative_lines": [{"p0": [1, 1], "p1": [1, 10]}],
    }

    result = save_roomseg_layer_dump(
        out_dir=tmp_path,
        step=3,
        room_debug=room_debug,
        occupancy_map=~nav,
        observed_free_mask=nav,
        obstacle_mask=~nav,
        unknown_mask=unknown,
        frontier_map=frontier,
        selected_frontier_members=[(7, 4), (7, 5), (7, 6)],
        selected_frontier_center_rc=(7, 5),
        agent_rc=(4, 4),
    )

    paths = result["paths"]
    for key in ("npz", "summary_json", "overlay_png", "layers_png", "navigation_room_masks_png"):
        assert Path(paths[key]).exists()
    summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert summary["algorithm"] == "upstream_rose2_vertical_or_free"
    assert summary["selected_frontier_sector_debug"]["available"] is True

    out_dir = tmp_path / "inspect"
    assert inspect_main(["--input", paths["npz"], "--out-dir", str(out_dir), "--print-summary"]) == 0
    assert (out_dir / "overlay.png").exists()
    assert (out_dir / "layers.png").exists()


def test_roomseg_snapshot_writes_lightweight_outputs_only(tmp_path):
    shape = (8, 8)
    labels = np.zeros(shape, dtype=np.int32)
    labels[1:5, 1:5] = 1
    nav = labels > 0
    room_debug = {
        "navigation_free_room_domain": nav,
        "vertical_free_room_domain": nav,
        "final_room_label_map": labels,
        "context_room_label_map": labels,
    }

    result = save_roomseg_layer_dump(
        out_dir=tmp_path,
        step=9,
        room_debug=room_debug,
        occupancy_map=~nav,
        observed_free_mask=nav,
        obstacle_mask=~nav,
        unknown_mask=np.zeros(shape, dtype=bool),
        save_png=False,
        save_overlay_png=False,
        save_layers_png=False,
        save_navigation_room_masks_png=True,
        npz_keys=ROOMSEG_SNAPSHOT_ARRAY_KEYS,
    )

    paths = result["paths"]
    assert Path(paths["npz"]).exists()
    assert Path(paths["summary_json"]).exists()
    assert Path(paths["navigation_room_masks_png"]).exists()
    assert not Path(paths["overlay_png"]).exists()
    assert not Path(paths["layers_png"]).exists()
    saved = np.load(paths["npz"])
    assert set(saved.files).issubset(set(ROOMSEG_SNAPSHOT_ARRAY_KEYS))
    assert "final_room_label_map" in saved.files
