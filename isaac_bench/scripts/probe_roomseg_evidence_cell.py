from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EVIDENCE_KEYS = (
    "vertical_free_room_domain",
    "vertical_observed_map",
    "vertical_unknown_before_overlay",
    "nav_raw_obstacle",
    "roomseg_static_structural_occupied",
    "nav_obstacle_overlay_candidate",
    "nav_obstacle_overlay_accepted",
    "initial_roomseg_free_after_fusion",
    "initial_roomseg_occupied_after_fusion",
    "initial_roomseg_unknown_after_fusion",
    "walls_rescued_from_unknown",
    "vertical_free_over_nav_obstacle",
    "nav_obstacle_still_unknown_after_fusion",
)

COUNT_PREFIXES = (
    "observed_count",
    "free_ray_count",
    "occupied_count",
    "unknown_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe one roomseg evidence cell from a debug NPZ dump.")
    parser.add_argument("--npz", required=True, help="Path to roomseg_step_*.npz")
    parser.add_argument("--row", type=int, required=True, help="Grid row")
    parser.add_argument("--col", type=int, required=True, help="Grid column")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args(argv)

    path = Path(args.npz)
    if not path.exists():
        raise SystemExit("roomseg evidence NPZ does not exist: %s" % path)

    with np.load(path, allow_pickle=False) as data:
        payload = _probe(data, int(args.row), int(args.col), str(path))

    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("npz: %s" % payload["npz"])
        print("row: %d" % payload["row"])
        print("col: %d" % payload["col"])
        print("in_bounds: %s" % payload["in_bounds"])
        for key in EVIDENCE_KEYS:
            print("%s: %s" % (key, payload["evidence"].get(key)))
        print("final_room_label: %s" % payload["final_room_label"])
        for key, value in payload["vertical_profile_counts"].items():
            print("%s: %s" % (key, value))
    return 0


def _probe(data: Any, row: int, col: int, npz_path: str) -> dict[str, object]:
    shape = _first_2d_shape(data)
    in_bounds = shape is not None and 0 <= row < shape[0] and 0 <= col < shape[1]
    evidence = {key: _cell_value(data, key, row, col, in_bounds) for key in EVIDENCE_KEYS}
    final_label = _cell_value(data, "final_room_label_map", row, col, in_bounds)
    counts: dict[str, object] = {}
    for key in data.files:
        if any(str(key).startswith(prefix) for prefix in COUNT_PREFIXES):
            value = _cell_value(data, key, row, col, in_bounds)
            if value is not None:
                counts[str(key)] = value
    return {
        "npz": npz_path,
        "row": int(row),
        "col": int(col),
        "shape": [int(shape[0]), int(shape[1])] if shape is not None else None,
        "in_bounds": bool(in_bounds),
        "evidence": evidence,
        "final_room_label": final_label,
        "vertical_profile_counts": counts,
    }


def _first_2d_shape(data: Any) -> tuple[int, int] | None:
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])
    return None


def _cell_value(data: Any, key: str, row: int, col: int, in_bounds: bool) -> object:
    if key not in data.files or not bool(in_bounds):
        return None
    arr = np.asarray(data[key])
    if arr.ndim != 2 or row >= arr.shape[0] or col >= arr.shape[1]:
        return None
    value = arr[row, col]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
