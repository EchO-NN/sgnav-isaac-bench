from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from isaac_bench.config import get_nested, load_config
from isaac_bench.debug.roomseg_layer_dump import save_roomseg_layer_dump
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegmenter
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VOXEL_UNKNOWN,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


REPLAY_NPZ_KEYS = (
    "final_room_label_map",
    "voxel_final_room_label_map",
    "accepted_separators",
    "rejected_separators",
    "voxel_door_seed_mask",
    "voxel_door_centerline_mask",
    "voxel_door_cut_mask",
    "voxel_current_door_cut_mask",
    "voxel_current_door_topology_effective_mask",
    "voxel_stable_door_cut_mask",
    "voxel_stable_door_visual_mask",
    "voxel_door_memory_observed_decay_band_mask",
    "voxel_door_memory_unobserved_track_mask",
    "voxel_door_memory_contradiction_mask",
    "voxel_step1_wall_gap_fill_map",
    "voxel_wall_after_step1_map",
    "voxel_step2_extension_candidate_map",
    "voxel_step2_extension_separator_map",
    "voxel_step2_rejected_extension_map",
    "voxel_final_separator_map",
    "voxel_wall_xy",
    "voxel_vertical_free_xy",
    "voxel_unknown_xy",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay voxel room segmentation from saved roomseg snapshot NPZ files.")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--resolution-m", type=float, default=None)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--reset-memory-per-snapshot", action="store_true")
    parser.add_argument("--save-overlay", action="store_true", default=True)
    parser.add_argument("--no-save-overlay", dest="save_overlay", action="store_false")
    parser.add_argument("--save-layer-grid", action="store_true")
    parser.add_argument("--save-compact-npz", action="store_true", default=True)
    parser.add_argument("--no-save-compact-npz", dest="save_compact_npz", action="store_false")
    parser.add_argument("--mask-only", action="store_true", help="Write only colored navigation room mask PNGs.")
    args = parser.parse_args(argv)
    if bool(args.mask_only):
        args.save_overlay = False
        args.save_layer_grid = False
        args.save_compact_npz = False

    snapshot_dir = Path(args.snapshot_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_paths = sorted(snapshot_dir.glob("roomseg_step_*.npz"), key=_snapshot_sort_key)
    if args.max_snapshots and int(args.max_snapshots) > 0:
        npz_paths = npz_paths[: int(args.max_snapshots)]
    if not npz_paths:
        raise SystemExit("no roomseg_step_*.npz files found in %s" % snapshot_dir)

    cfg = load_config(args.config)
    roomseg_cfg = dict(get_nested(cfg, "mapping.room_segmentation", {}) or {})
    voxel_grid_cfg = dict(roomseg_cfg.get("voxel_grid", {}) or {})
    resolution_m = float(args.resolution_m or get_nested(cfg, "mapping.online_resolution_m", 0.05))
    shape = _snapshot_shape(npz_paths[0])
    map_info = MapInfo(
        resolution_m=resolution_m,
        min_x=0.0,
        max_x=float(shape[1]) * resolution_m,
        min_y=0.0,
        max_y=float(shape[0]) * resolution_m,
        width=int(shape[1]),
        height=int(shape[0]),
    )
    segmenter = VoxelOccupancyDoorWallRoomSegmenter(config=roomseg_cfg, map_info=map_info)

    manifest: dict[str, object] = {
        "snapshot_dir": str(snapshot_dir),
        "out_dir": str(out_dir),
        "config": str(Path(args.config)),
        "resolution_m": float(resolution_m),
        "snapshot_count": int(len(npz_paths)),
        "processed": 0,
        "errors": [],
        "reset_memory_per_snapshot": bool(args.reset_memory_per_snapshot),
        "sensor_range_reconstruction": "from_saved_2d_sensor_range_counts",
        "started_at_unix": time.time(),
        "steps": [],
    }
    for idx, npz_path in enumerate(npz_paths):
        step = _infer_step(npz_path)
        if bool(args.reset_memory_per_snapshot):
            segmenter = VoxelOccupancyDoorWallRoomSegmenter(config=roomseg_cfg, map_info=map_info)
        started = time.perf_counter()
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                arrays = {name: np.asarray(data[name]).copy() for name in data.files}
            voxel_grid = _voxel_grid_from_snapshot(arrays, map_info, voxel_grid_cfg)
            occupancy_map = _bool_array(arrays, "occupancy_map", shape)
            observed_free_mask = _bool_array(arrays, "observed_free_mask", shape)
            obstacle_mask = _bool_array(arrays, "obstacle_mask", shape)
            unknown_mask = _bool_array(arrays, "unknown_mask", shape)
            segmenter.update(
                occupancy_map=occupancy_map,
                observed_free_mask=observed_free_mask,
                obstacle_mask=obstacle_mask,
                unknown_mask=unknown_mask,
                voxel_grid=voxel_grid,
                step=int(step),
            )
            result = segmenter.last_result
            if result is None:
                raise RuntimeError("segmenter did not produce a result")
            room_debug = dict(segmenter.last_debug)
            # The renderer expects these canonical dump keys. The voxel backend
            # also exposes aliases, but make the replay output independent of
            # backend alias drift.
            room_debug["final_room_label_map"] = np.asarray(result.room_label_map, dtype=np.int32)
            room_debug["voxel_final_room_label_map"] = np.asarray(result.room_label_map, dtype=np.int32)
            room_debug["accepted_separators"] = np.asarray(result.separator_map, dtype=bool)
            room_debug["voxel_final_separator_map"] = np.asarray(result.separator_map, dtype=bool)
            frontier_map = arrays.get("frontier_map")
            selected_frontier = _members_from_mask(arrays.get("selected_frontier_mask"))
            selected_center = arrays.get("selected_frontier_center_rc")
            agent_rc = arrays.get("agent_rc")
            dump = save_roomseg_layer_dump(
                out_dir=out_dir,
                step=int(step),
                room_debug=room_debug,
                occupancy_map=occupancy_map,
                observed_free_mask=observed_free_mask,
                obstacle_mask=obstacle_mask,
                unknown_mask=unknown_mask,
                frontier_map=None if frontier_map is None else np.asarray(frontier_map, dtype=bool),
                selected_frontier_members=selected_frontier,
                selected_frontier_center_rc=None if selected_center is None else np.asarray(selected_center, dtype=np.int32).reshape(-1)[:2],
                agent_rc=None if agent_rc is None else np.asarray(agent_rc, dtype=np.int32).reshape(-1)[:2],
                max_saves=max(10000, int(len(npz_paths)) + 10),
                save_npz=bool(args.save_compact_npz),
                save_png=True,
                save_summary_json=not bool(args.mask_only),
                save_overlay_png=bool(args.save_overlay),
                save_layers_png=bool(args.save_layer_grid),
                save_navigation_room_masks_png=True,
                npz_keys=REPLAY_NPZ_KEYS,
                extra_npz_arrays=None,
                include_selected_frontier_sector=False,
            )
            comparison = _compare_with_original(arrays, result.room_label_map, result.separator_map)
            summary_path = Path(dump["paths"]["summary_json"])
            summary = dict(dump.get("summary", {}))
            summary["input_snapshot"] = str(npz_path)
            summary["replay_comparison"] = comparison
            summary["replay_runtime_ms"] = float((time.perf_counter() - started) * 1000.0)
            if not bool(args.mask_only):
                summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest["processed"] = int(manifest["processed"]) + 1
            manifest["steps"].append(
                {
                    "step": int(step),
                    "input": str(npz_path),
                    "summary": None if bool(args.mask_only) else str(summary_path),
                    "navigation_room_masks_png": str(dump["paths"]["navigation_room_masks_png"]),
                    "overlay_png": str(dump["paths"]["overlay_png"]) if bool(args.save_overlay) else None,
                    "runtime_ms": summary["replay_runtime_ms"],
                    **comparison,
                }
            )
            print(
                "[replay] step=%06d rooms old=%s new=%s labeled_changed=%d out=%s"
                % (
                    int(step),
                    comparison.get("original_room_count"),
                    comparison.get("replay_room_count"),
                    int(comparison.get("label_presence_changed_cells", 0)),
                    dump["paths"]["navigation_room_masks_png"],
                ),
                flush=True,
            )
        except Exception as exc:
            err = {"step": int(step), "input": str(npz_path), "error": repr(exc)}
            manifest["errors"].append(err)
            print("[replay] ERROR step=%06d %s" % (int(step), repr(exc)), flush=True)

    manifest["finished_at_unix"] = time.time()
    if not bool(args.mask_only):
        (out_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "processed": manifest["processed"], "errors": len(manifest["errors"])}, ensure_ascii=False))
    return 0 if not manifest["errors"] else 2


def _snapshot_shape(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        if "observed_free_mask" in data:
            arr = np.asarray(data["observed_free_mask"])
        elif "final_room_label_map" in data:
            arr = np.asarray(data["final_room_label_map"])
        else:
            raise ValueError("cannot infer snapshot shape from %s" % path)
    if arr.ndim != 2:
        raise ValueError("snapshot shape source must be 2D")
    return int(arr.shape[0]), int(arr.shape[1])


def _voxel_grid_from_snapshot(
    arrays: Mapping[str, np.ndarray],
    map_info: MapInfo,
    voxel_grid_cfg: Mapping[str, object],
) -> VoxelOccupancyGrid3D:
    if "voxel_occupancy_state_zyx" not in arrays:
        raise KeyError("snapshot missing voxel_occupancy_state_zyx")
    state = np.asarray(arrays["voxel_occupancy_state_zyx"], dtype=np.uint8).copy()
    if state.ndim != 3:
        raise ValueError("voxel_occupancy_state_zyx must have shape [Z,H,W]")
    z_min = _scalar(arrays, "voxel_occupancy_z_min_m", float(voxel_grid_cfg.get("z_min_m", 0.0)))
    z_max = _scalar(arrays, "voxel_occupancy_z_max_m", float(voxel_grid_cfg.get("z_max_m", 3.2)))
    z_res = _scalar(arrays, "voxel_occupancy_z_resolution_m", float(voxel_grid_cfg.get("z_resolution_m", 0.05)))
    active_z_min = _scalar(arrays, "voxel_occupancy_active_z_min_m", float(voxel_grid_cfg.get("active_z_min_m", 0.10)))
    active_z_max = _scalar(arrays, "voxel_occupancy_active_z_max_m", float(voxel_grid_cfg.get("active_z_max_fallback_m", 2.0)))
    ceiling = _scalar(arrays, "voxel_occupancy_ceiling_height_estimate_m", math.nan)
    cfg = VoxelOccupancyGridConfig.from_mapping(
        voxel_grid_cfg,
        z_min_m=z_min,
        z_max_m=z_max,
        z_resolution_m=z_res,
        active_z_min_m=active_z_min,
    )
    log_odds = _log_odds_from_state(state, cfg)
    grid = VoxelOccupancyGrid3D(
        log_odds=log_odds,
        state=state,
        sensor_range_count=np.zeros_like(state, dtype=np.uint8),
        z_min_m=float(z_min),
        z_max_m=float(z_max),
        z_resolution_m=float(z_res),
        map_info=map_info,
        config=cfg,
        active_z_min_m=float(active_z_min),
        active_z_max_m=float(active_z_max),
        ceiling_height_m=None if not np.isfinite(ceiling) else float(ceiling),
        ceiling_estimate_status="snapshot",
    )
    grid.sensor_range_count = _reconstruct_sensor_range_count(arrays, grid)
    return grid


def _reconstruct_sensor_range_count(arrays: Mapping[str, np.ndarray], grid: VoxelOccupancyGrid3D) -> np.ndarray:
    sensor = np.zeros_like(grid.state, dtype=np.uint8)
    active_idx = grid.active_z_indices()
    if active_idx.size == 0:
        return sensor
    active_state = np.asarray(grid.state[active_idx], dtype=np.uint8)
    shape = active_state.shape[1:]
    target_unknown = np.asarray(arrays.get("voxel_sensor_in_range_unknown_count_xy", np.zeros(shape, dtype=np.uint16)), dtype=np.uint16)
    target_total = np.asarray(arrays.get("voxel_sensor_range_count_xy", target_unknown), dtype=np.uint16)
    unknown_bin = active_state == int(VOXEL_UNKNOWN)
    if np.any(target_unknown):
        unknown_rank = np.cumsum(unknown_bin, axis=0, dtype=np.uint16)
        sensor_active = unknown_bin & (unknown_rank <= np.minimum(target_unknown, unknown_rank[-1])[None, :, :])
    else:
        sensor_active = np.zeros_like(active_state, dtype=bool)
    remaining = np.maximum(target_total.astype(np.int32) - np.sum(sensor_active, axis=0, dtype=np.int32), 0).astype(np.uint16)
    non_unknown = ~unknown_bin
    if np.any(remaining):
        known_rank = np.cumsum(non_unknown, axis=0, dtype=np.uint16)
        sensor_active |= non_unknown & (known_rank <= np.minimum(remaining, known_rank[-1])[None, :, :])
    sensor[active_idx] = sensor_active.astype(np.uint8)
    return sensor


def _log_odds_from_state(state: np.ndarray, cfg: VoxelOccupancyGridConfig) -> np.ndarray:
    log_odds = np.zeros_like(state, dtype=np.int16)
    log_odds[state == int(VOXEL_FREE)] = int(cfg.free_logodds_threshold)
    log_odds[state == int(VOXEL_OCCUPIED)] = int(cfg.occupied_logodds_threshold)
    return log_odds


def _compare_with_original(arrays: Mapping[str, np.ndarray], replay_labels: np.ndarray, replay_separator: np.ndarray) -> dict[str, object]:
    original_labels = np.asarray(arrays.get("final_room_label_map", np.zeros_like(replay_labels)), dtype=np.int32)
    replay = np.asarray(replay_labels, dtype=np.int32)
    original_presence = original_labels > 0
    replay_presence = replay > 0
    original_separator = np.asarray(arrays.get("accepted_separators", np.zeros_like(replay_presence)), dtype=bool)
    replay_sep = np.asarray(replay_separator, dtype=bool)
    return {
        "original_room_count": _room_count(original_labels),
        "replay_room_count": _room_count(replay),
        "original_labeled_cells": int(np.count_nonzero(original_presence)),
        "replay_labeled_cells": int(np.count_nonzero(replay_presence)),
        "label_presence_changed_cells": int(np.count_nonzero(original_presence ^ replay_presence)),
        "label_id_changed_labeled_overlap_cells": int(np.count_nonzero((original_labels != replay) & original_presence & replay_presence)),
        "original_separator_cells": int(np.count_nonzero(original_separator)),
        "replay_separator_cells": int(np.count_nonzero(replay_sep)),
        "separator_presence_changed_cells": int(np.count_nonzero(original_separator ^ replay_sep)),
    }


def _room_count(labels: np.ndarray) -> int:
    values = np.unique(np.asarray(labels, dtype=np.int32))
    return int(np.count_nonzero(values > 0))


def _members_from_mask(mask: object, max_members: int = 20000) -> list[tuple[int, int]] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not np.any(arr):
        return None
    coords = np.argwhere(arr)
    if coords.shape[0] > int(max_members):
        coords = coords[: int(max_members)]
    return [(int(r), int(c)) for r, c in coords]


def _bool_array(arrays: Mapping[str, np.ndarray], key: str, shape: tuple[int, int]) -> np.ndarray:
    value = arrays.get(key)
    if value is None:
        return np.zeros(shape, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape != shape:
        raise ValueError("%s shape %s does not match expected %s" % (key, arr.shape, shape))
    return arr


def _scalar(arrays: Mapping[str, np.ndarray], key: str, default: float) -> float:
    if key not in arrays:
        return float(default)
    arr = np.asarray(arrays[key])
    if arr.size == 0:
        return float(default)
    value = float(arr.reshape(-1)[0])
    return float(default) if not np.isfinite(value) and np.isfinite(default) else value


def _infer_step(path: Path) -> int:
    match = re.search(r"roomseg_step_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _snapshot_sort_key(path: Path) -> tuple[int, str]:
    return _infer_step(path), path.name


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
