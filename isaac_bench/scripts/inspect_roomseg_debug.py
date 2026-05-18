from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from isaac_bench.debug.roomseg_layer_dump import (
    render_roomseg_layers_grid,
    render_roomseg_overlay,
    summarize_roomseg_arrays,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    step = _step_from_name(input_path)
    summary_path = input_path.with_suffix(".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = summarize_roomseg_arrays(arrays, {}, step)
    render_roomseg_overlay(arrays, summary).save(out_dir / "overlay.png")
    render_roomseg_layers_grid(arrays, summary).save(out_dir / "layers.png")
    for key, value in sorted(arrays.items()):
        arr = np.asarray(value)
        if arr.ndim != 2:
            continue
        Image.fromarray(_array_to_uint8(arr)).save(out_dir / ("%s.png" % key))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if bool(args.print_summary):
        counts = summary.get("counts", {})
        print("likely_cause=%s" % summary.get("likely_cause", "unclear"))
        print("navigation_free=%s nav_free_unlabeled=%s vertical_free_unlabeled=%s" % (
            counts.get("navigation_free", 0),
            counts.get("nav_free_unlabeled", 0),
            counts.get("vertical_free_unlabeled", 0),
        ))
        print("summary_json=%s" % str(out_dir / "summary.json"))
    return 0


def _array_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in {"i", "u"} and np.max(arr) > 1:
        vals = arr.astype(np.int32)
        out = np.zeros(vals.shape, dtype=np.uint8)
        positive = vals > 0
        out[positive] = ((vals[positive] * 53) % 230 + 25).astype(np.uint8)
        return out
    if arr.dtype.kind == "f":
        value = arr.astype(np.float32)
        if np.max(value) > np.min(value):
            value = (value - np.min(value)) / (np.max(value) - np.min(value))
        return np.clip(value * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(arr, dtype=bool).astype(np.uint8) * 255


def _step_from_name(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.isdigit():
            return int(part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

