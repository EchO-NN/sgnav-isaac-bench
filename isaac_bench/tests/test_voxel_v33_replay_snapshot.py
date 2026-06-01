from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED
from isaac_bench.scripts.replay_voxel_roomseg_snapshots import (
    _load_memory_state_from_snapshot,
    _voxel_grid_from_snapshot,
    main as replay_main,
)


def test_replay_single_snapshot_visualize_writes_mask_and_diff_pngs(tmp_path: Path) -> None:
    snapshot = tmp_path / "roomseg_step_000003.npz"
    shape = (18, 20)
    labels = np.zeros(shape, dtype=np.int32)
    labels[2:9, 2:8] = 1
    labels[9:16, 10:18] = 2
    separators = np.zeros(shape, dtype=bool)
    separators[:, 9] = True
    state = np.zeros((4, shape[0], shape[1]), dtype=np.uint8)
    state[1, labels > 0] = int(VOXEL_FREE)
    state[2, separators] = int(VOXEL_OCCUPIED)
    log_odds = np.zeros_like(state, dtype=np.int16)
    log_odds[state == int(VOXEL_FREE)] = -1
    log_odds[state == int(VOXEL_OCCUPIED)] = 2
    sensor = (state != 0).astype(np.uint8)
    np.savez_compressed(
        snapshot,
        occupancy_map=separators,
        observed_free_mask=labels > 0,
        obstacle_mask=separators,
        unknown_mask=labels == 0,
        final_room_label_map=labels,
        voxel_final_room_label_map=labels,
        accepted_separators=separators,
        voxel_final_separator_map=separators,
        voxel_occupancy_state_zyx=state,
        voxel_occupancy_log_odds_zyx=log_odds,
        voxel_sensor_range_count_zyx=sensor,
        voxel_occupancy_z_min_m=np.asarray(0.0, dtype=np.float32),
        voxel_occupancy_z_max_m=np.asarray(0.4, dtype=np.float32),
        voxel_occupancy_z_resolution_m=np.asarray(0.1, dtype=np.float32),
        voxel_occupancy_active_z_min_m=np.asarray(0.0, dtype=np.float32),
        voxel_occupancy_active_z_max_m=np.asarray(0.4, dtype=np.float32),
    )
    out_dir = tmp_path / "replay"

    assert replay_main(["--snapshot", str(snapshot), "--out-dir", str(out_dir), "--mode", "visualize", "--mask-only"]) == 0

    mask_png = out_dir / "roomseg_step_000003.navigation_room_masks.png"
    label_diff_png = out_dir / "roomseg_step_000003.replay_label_diff.png"
    assert mask_png.exists()
    assert label_diff_png.exists()
    assert Image.open(mask_png).size == (shape[1], shape[0])


def test_voxel_grid_from_snapshot_restores_exact_logodds_and_sensor_count() -> None:
    shape = (5, 7)
    state = np.zeros((3, shape[0], shape[1]), dtype=np.uint8)
    log_odds = np.arange(state.size, dtype=np.int16).reshape(state.shape) % 7 - 3
    sensor = (np.arange(state.size, dtype=np.uint8).reshape(state.shape) % 5).astype(np.uint8)
    arrays = {
        "voxel_occupancy_state_zyx": state,
        "voxel_occupancy_log_odds_zyx": log_odds,
        "voxel_sensor_range_count_zyx": sensor,
        "voxel_occupancy_z_min_m": np.asarray(0.0, dtype=np.float32),
        "voxel_occupancy_z_max_m": np.asarray(0.3, dtype=np.float32),
        "voxel_occupancy_z_resolution_m": np.asarray(0.1, dtype=np.float32),
    }
    map_info = MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.35, min_y=0.0, max_y=0.25, width=shape[1], height=shape[0])

    grid = _voxel_grid_from_snapshot(arrays, map_info, {}, recompute_state_from_logodds=False)

    assert np.array_equal(grid.log_odds, log_odds)
    assert np.array_equal(grid.sensor_range_count, sensor)
    assert np.array_equal(grid.state, state)


def test_replay_memory_state_loader_supports_saved_before_json() -> None:
    state = {
        "schema_version": 1,
        "door_memory": {"schema_version": 1, "next_track_id": 3, "tracks": []},
        "separator_memory": {"schema_version": 1, "next_track_id": 4, "tracks": []},
    }
    arrays = {
        "voxel_roomseg_memory_before_json": np.asarray(json.dumps(state), dtype="<U512"),
    }

    loaded = _load_memory_state_from_snapshot(arrays, "saved-before")

    assert loaded == state
