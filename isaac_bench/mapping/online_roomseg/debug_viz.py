from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


BOOL_LAYERS = [
    "vertical_free_raw",
    "vertical_occupied_raw",
    "vertical_observed_raw",
    "vertical_unknown_raw",
    "free_clean",
    "wall_candidate_clean",
    "line_supported_walls",
    "physical_wall_completion_candidates",
    "doorway_virtual_cut_candidates",
    "corridor_skeleton",
    "corridor_candidate_map",
    "corridor_room_neck_cut_candidates",
    "accepted_separators",
    "rejected_separators",
]


def save_online_roomseg_debug(
    *,
    out_dir: str | Path,
    layers: Mapping[str, np.ndarray],
    separator_report: Mapping[str, object],
    save_layers: bool = True,
    save_candidate_json: bool = True,
) -> dict:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    if save_layers:
        for name in BOOL_LAYERS:
            if name in layers:
                path = root / ("%s.png" % name)
                _save_bool(path, np.asarray(layers[name], dtype=bool))
                paths[name] = str(path)
        for name in ("room_labels_before_separators", "room_labels_after_separators", "final_room_labels"):
            if name in layers:
                path = root / ("%s.png" % name)
                _save_labels(path, np.asarray(layers[name], dtype=np.int32))
                paths[name] = str(path)
    if save_candidate_json:
        path = root / "separator_report.json"
        path.write_text(json.dumps(_jsonable(separator_report), indent=2, ensure_ascii=False), encoding="utf-8")
        paths["separator_report"] = str(path)
    return {"paths": paths, "output_dir": str(root)}


def _save_bool(path: Path, mask: np.ndarray) -> None:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[:, :] = (25, 25, 25)
    arr[np.asarray(mask, dtype=bool)] = (245, 245, 245)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_labels(path: Path, labels: np.ndarray) -> None:
    arr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    positive = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    for label in positive:
        color = _label_color(label)
        arr[labels == label] = color
    Image.fromarray(arr, mode="RGB").save(path)


def _label_color(label: int) -> tuple[int, int, int]:
    value = int(label) * 2654435761
    return (
        60 + ((value >> 0) & 0x7F),
        60 + ((value >> 8) & 0x7F),
        60 + ((value >> 16) & 0x7F),
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
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
