from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


def save_online_watershed_debug(
    *,
    out_dir: str | Path,
    layers: Mapping[str, np.ndarray],
    seed_report: Mapping[str, object],
    region_report: Mapping[str, object],
    region_graph: Mapping[str, object],
    frontier_room_context: Mapping[str, object],
) -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    bool_layers = {
        "free_clean": "watershed_free_clean.png",
        "wall_candidate_clean": "watershed_wall_candidate_clean.png",
        "unknown_mask": "watershed_unknown_mask.png",
        "corridor_core": "watershed_corridor_core.png",
        "confirmed_room_seeds": "watershed_confirmed_room_seeds.png",
        "frontier_room_seeds": "watershed_frontier_room_seeds.png",
        "corridor_seeds": "watershed_corridor_seeds.png",
    }
    for key, filename in bool_layers.items():
        if key in layers:
            path = root / filename
            _save_bool(path, np.asarray(layers[key], dtype=bool))
            paths[key] = str(path)
    float_layers = {
        "dist_struct_m": "watershed_dist_struct.png",
        "dist_free_extent_m": "watershed_dist_free_extent.png",
        "elevation": "watershed_elevation.png",
    }
    for key, filename in float_layers.items():
        if key in layers:
            path = root / filename
            _save_float(path, np.asarray(layers[key], dtype=np.float32))
            paths[key] = str(path)
    label_layers = {
        "raw_labels": "watershed_raw_labels.png",
        "final_labels": "watershed_final_labels.png",
        "region_type_map": "watershed_region_types.png",
    }
    for key, filename in label_layers.items():
        if key in layers:
            path = root / filename
            _save_labels(path, np.asarray(layers[key], dtype=np.int32))
            paths[key] = str(path)
    json_payloads = {
        "watershed_seed_report": seed_report,
        "watershed_region_report": region_report,
        "watershed_region_graph": region_graph,
        "watershed_frontier_room_context": frontier_room_context,
    }
    for name, payload in json_payloads.items():
        path = root / ("%s.json" % name)
        path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[name] = str(path)
    return {"output_dir": str(root), "paths": paths}


def _save_bool(path: Path, mask: np.ndarray) -> None:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[:, :] = (20, 20, 20)
    arr[np.asarray(mask, dtype=bool)] = (245, 245, 245)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_float(path: Path, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(finite):
        lo = float(np.min(arr[finite]))
        hi = float(np.max(arr[finite]))
        if hi > lo + 1e-6:
            out[finite] = np.clip((arr[finite] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            out[finite] = 255
    Image.fromarray(out, mode="L").save(path)


def _save_labels(path: Path, labels: np.ndarray) -> None:
    arr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        arr[labels == label] = _label_color(label)
    Image.fromarray(arr, mode="RGB").save(path)


def _label_color(label: int) -> tuple[int, int, int]:
    value = int(label) * 2654435761
    return (
        50 + ((value >> 0) & 0x9F),
        50 + ((value >> 8) & 0x9F),
        50 + ((value >> 16) & 0x9F),
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
