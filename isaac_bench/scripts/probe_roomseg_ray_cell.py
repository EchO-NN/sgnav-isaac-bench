from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect roomseg ray-valid wall evidence at one grid cell.")
    parser.add_argument("--npz", required=True, help="roomseg layer dump .npz")
    parser.add_argument("--row", required=True, type=int)
    parser.add_argument("--col", required=True, type=int)
    args = parser.parse_args()

    path = Path(args.npz)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    row = int(args.row)
    col = int(args.col)
    shape = _infer_shape(data)
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise IndexError("cell (%d, %d) outside dump shape %s" % (row, col, shape))

    out = {
        "cell": {"row": row, "col": col},
        "shape": [int(shape[0]), int(shape[1])],
        "vertical_free": _bool_cell(data, "vertical_free_room_domain", row, col),
        "vertical_occupied_0p2_2p0": _bool_cell(data, "vertical_occupied_0p2_2p0", row, col),
        "vertical_observed_0p2_2p0": _bool_cell(data, "vertical_observed_0p2_2p0", row, col),
        "roomseg_ray_covered_count": _int_cell(data, "roomseg_ray_covered_count", row, col),
        "terminal_wall_count": _int_cell(data, "roomseg_terminal_wall_count", row, col),
        "terminal_wall_height_min": _float_cell(data, "roomseg_terminal_wall_height_min", row, col),
        "terminal_wall_height_max": _float_cell(data, "roomseg_terminal_wall_height_max", row, col),
        "terminal_wall_depth_min": _float_cell(data, "roomseg_terminal_wall_depth_min", row, col),
        "terminal_wall_splat": _bool_cell(data, "roomseg_terminal_wall_splat", row, col),
        "ray_valid_wall": _bool_cell(data, "ray_valid_wall_inference", row, col),
        "initial_before": {
            "free": _bool_cell(data, "initial_roomseg_free", row, col),
            "occupied": _bool_cell(data, "initial_roomseg_occupied", row, col),
            "unknown": _bool_cell(data, "initial_roomseg_unknown", row, col),
        },
        "initial_after_ray_wall": {
            "free": _bool_cell(data, "initial_roomseg_free_after_ray_wall", row, col),
            "occupied": _bool_cell(data, "initial_roomseg_occupied_after_ray_wall", row, col),
            "unknown": _bool_cell(data, "initial_roomseg_unknown_after_ray_wall", row, col),
        },
        "unknown_before_ray_wall": _bool_cell(data, "unknown_before_ray_wall", row, col),
        "unknown_after_ray_wall": _bool_cell(data, "unknown_after_ray_wall", row, col),
        "unknown_removed_by_ray_wall": _bool_cell(data, "unknown_removed_by_ray_wall", row, col),
        "final_source_label": _int_cell(data, "final_room_label_map", row, col),
        "available_keys": sorted(str(k) for k in data.files),
    }
    print(json.dumps(_json_ready(out), ensure_ascii=False, indent=2))
    return 0


def _infer_shape(data: np.lib.npyio.NpzFile) -> tuple[int, int]:
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError("npz does not contain any 2D roomseg arrays")


def _cell(data: np.lib.npyio.NpzFile, key: str, row: int, col: int):
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 2 or not (0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]):
        return None
    return arr[row, col]


def _bool_cell(data: np.lib.npyio.NpzFile, key: str, row: int, col: int) -> bool | None:
    value = _cell(data, key, row, col)
    return None if value is None else bool(value)


def _int_cell(data: np.lib.npyio.NpzFile, key: str, row: int, col: int) -> int | None:
    value = _cell(data, key, row, col)
    return None if value is None else int(value)


def _float_cell(data: np.lib.npyio.NpzFile, key: str, row: int, col: int) -> float | None:
    value = _cell(data, key, row, col)
    if value is None:
        return None
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
