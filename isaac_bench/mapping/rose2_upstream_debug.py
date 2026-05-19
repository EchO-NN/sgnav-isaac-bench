from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
from PIL import Image


def save_rose2_upstream_debug_bundle(
    *,
    out_dir: str | Path,
    stem: str,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    room_label_map: np.ndarray,
    source_output_path: str | Path | None,
    summary: Mapping[str, object],
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(room_label_map, dtype=np.int32)
    paths = {
        "parsed_labels_png": out / ("%s.parsed_labels.png" % stem),
        "parsed_labels_npy": out / ("%s.parsed_labels.npy" % stem),
        "overlay_png": out / ("%s.overlay.png" % stem),
        "source_output_png": out / ("%s.source_output.png" % stem),
        "summary_json": out / ("%s.summary.json" % stem),
    }
    label_image(labels).save(paths["parsed_labels_png"])
    np.save(paths["parsed_labels_npy"], labels.astype(np.int32))
    overlay_image(
        observed_free=np.asarray(observed_free, dtype=bool),
        observed_occupied=np.asarray(observed_occupied, dtype=bool),
        unknown=np.asarray(unknown, dtype=bool),
        labels=labels,
    ).save(paths["overlay_png"])
    if source_output_path is not None and Path(source_output_path).exists():
        try:
            shutil.copyfile(str(source_output_path), str(paths["source_output_png"]))
        except Exception:
            paths["source_output_png"] = Path(source_output_path)
    payload = dict(summary)
    payload["debug_paths"] = {key: str(value) for key, value in paths.items()}
    paths["summary_json"].write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"paths": {key: str(value) for key, value in paths.items()}, "summary": payload}


def write_failure_bundle(
    *,
    out_dir: str | Path,
    stem: str,
    reason: str,
    summary: Mapping[str, object] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("%s.failure.json" % stem)
    payload = {"ok": False, "failure_reason": str(reason)}
    payload.update(dict(summary or {}))
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def label_image(labels: np.ndarray) -> Image.Image:
    arr = np.asarray(labels, dtype=np.int32)
    canvas = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        canvas[arr == label_id] = _palette(label_id)
    return Image.fromarray(canvas, mode="RGB")


def overlay_image(*, observed_free: np.ndarray, observed_occupied: np.ndarray, unknown: np.ndarray, labels: np.ndarray) -> Image.Image:
    free = np.asarray(observed_free, dtype=bool)
    occupied = np.asarray(observed_occupied, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    arr = np.zeros((*free.shape, 3), dtype=np.uint8)
    arr[:, :] = (30, 32, 36)
    arr[unknown_arr] = (85, 85, 85)
    arr[free] = (225, 225, 225)
    arr[occupied] = (0, 0, 0)
    label_arr = np.asarray(labels, dtype=np.int32)
    for label_id in sorted(int(v) for v in np.unique(label_arr) if int(v) > 0):
        mask = label_arr == label_id
        color = np.asarray(_palette(label_id), dtype=np.float32)
        arr[mask] = np.clip(arr[mask].astype(np.float32) * 0.45 + color[None, :] * 0.55, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def _palette(label_id: int) -> tuple[int, int, int]:
    palette = [
        (126, 174, 255),
        (255, 156, 102),
        (130, 222, 150),
        (214, 148, 255),
        (250, 216, 95),
        (95, 220, 220),
        (255, 120, 180),
    ]
    return palette[(int(label_id) - 1) % len(palette)]
