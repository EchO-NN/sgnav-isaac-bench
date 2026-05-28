from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE, VOXEL_OCCUPIED, VOXEL_CONFLICT, VOXEL_UNKNOWN


STATE_NAMES = {
    int(VOXEL_UNKNOWN): "UNKNOWN",
    int(VOXEL_FREE): "FREE",
    int(VOXEL_OCCUPIED): "OCCUPIED",
    int(VOXEL_CONFLICT): "CONFLICT",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one xy cell from a saved voxel roomseg npz.")
    parser.add_argument("--debug-npz", required=True, help="Path to roomseg debug/snapshot npz")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--col", type=int, required=True)
    args = parser.parse_args()

    path = Path(args.debug_npz)
    data = np.load(path, allow_pickle=True)
    row, col = int(args.row), int(args.col)
    print("cell row=%d col=%d" % (row, col))

    state = _first_present(data, ("voxel_state_zyx", "state"))
    log_odds = _first_present(data, ("voxel_log_odds_zyx", "log_odds"))
    z_min = _scalar(data, "voxel_z_min_m", _scalar(data, "z_min_m", 0.0))
    z_res = _scalar(data, "voxel_z_resolution_m", _scalar(data, "z_resolution_m", 0.05))
    active_min = _scalar(data, "voxel_active_z_min_m", _scalar(data, "active_z_min_m", 0.10))
    active_max = _scalar(data, "voxel_active_z_max_m", _scalar(data, "active_z_max_m", np.nan))
    ceiling = _scalar(data, "voxel_ceiling_height_m", np.nan)
    print("active_z_min=%.2f active_z_max=%s ceiling=%s" % (active_min, _fmt(active_max), _fmt(ceiling)))

    if state is not None:
        state = np.asarray(state)
        if state.ndim != 3:
            raise SystemExit("voxel state must be 3D zyx")
        if row < 0 or col < 0 or row >= state.shape[1] or col >= state.shape[2]:
            raise SystemExit("row/col outside voxel grid shape %s" % (state.shape,))
        log = None if log_odds is None else np.asarray(log_odds)
        for z_idx in range(state.shape[0]):
            z = float(z_min) + (z_idx + 0.5) * float(z_res)
            value = int(state[z_idx, row, col])
            if log is not None and log.shape == state.shape:
                print("z=%.2f state=%s log=%d" % (z, STATE_NAMES.get(value, str(value)), int(log[z_idx, row, col])))
            else:
                print("z=%.2f state=%s" % (z, STATE_NAMES.get(value, str(value))))
    else:
        print("voxel_state_zyx missing in npz; showing 2D classification arrays only")

    free_count = _cell(data, "voxel_active_free_count_xy", row, col)
    occ_count = _cell(data, "voxel_active_occupied_count_xy", row, col)
    unk_count = _cell(data, "voxel_active_unknown_count_xy", row, col)
    ratio = _cell(data, "voxel_occupied_ratio_active_xy", row, col)
    wall = _bool_cell(data, "voxel_wall_xy", row, col)
    vertical_free = _bool_cell(data, "voxel_vertical_free_xy", row, col)
    door_seed = _bool_cell(data, "voxel_door_seed_mask", row, col)
    reject_code = _cell(data, "voxel_door_seed_reject_reason_map", row, col)
    print("")
    print("2D classification:")
    print("free_count=%s" % _fmt(free_count))
    print("occupied_count=%s" % _fmt(occ_count))
    print("unknown_count=%s" % _fmt(unk_count))
    print("occupied_ratio=%s" % _fmt(ratio))
    print("wall=%s" % wall)
    print("vertical_free=%s" % vertical_free)
    print("door_seed=%s" % door_seed)
    print("seed_reject_code=%s" % _fmt(reject_code))
    return 0


def _first_present(data, keys: tuple[str, ...]):
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def _scalar(data, key: str, default):
    if key not in data.files:
        return default
    arr = np.asarray(data[key])
    if arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def _cell(data, key: str, row: int, col: int):
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 2 or row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
        return None
    value = arr[row, col]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _bool_cell(data, key: str, row: int, col: int) -> bool | None:
    value = _cell(data, key, row, col)
    if value is None:
        return None
    return bool(value)


def _fmt(value) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return "%.3f" % float(value)
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
