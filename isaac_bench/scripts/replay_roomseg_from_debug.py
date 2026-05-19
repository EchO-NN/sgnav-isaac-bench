from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from isaac_bench.mapping.rose2_source_form import (
    ROSE2SourceFormConfig,
    save_rose2_source_debug,
    source_result_summary,
    run_rose2_source_form_v2,
)
from isaac_bench.mapping.structure_extraction import StructureExtractionConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="NPZ saved by rose2 source debug")
    parser.add_argument("--vertical-free-png", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", default="rose2_source_form_v2", choices=["rose2_source_form_v2"])
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--min-room-area-m2", type=float, default=1.5)
    parser.add_argument("--dump-layers", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    occupied, free, unknown, vertical_observed, vertical_free, wall_confidence, quality = _load_inputs(args)
    structure_cfg = StructureExtractionConfig(
        resolution_m=float(args.resolution_m),
        min_room_area_m2=float(args.min_room_area_m2),
        hough_min_line_length_m=0.45,
        hough_line_gap_m=0.25,
        wall_min_support_ratio=0.10,
    )
    source_cfg = ROSE2SourceFormConfig(
        resolution_m=float(args.resolution_m),
        min_room_area_m2=float(args.min_room_area_m2),
        min_wall_line_length_m=0.45,
        debug_dump=bool(args.dump_layers),
        debug_dir=str(out),
    )
    result = run_rose2_source_form_v2(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        vertical_observed=vertical_observed,
        vertical_free=vertical_free,
        wall_confidence_map=wall_confidence,
        structure_config=structure_cfg,
        source_config=source_cfg,
        object_memory=None,
    )
    result.debug["input_quality"] = quality
    dump = save_rose2_source_debug(
        out_dir=out,
        step=0,
        result=result,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
    )
    summary = source_result_summary(result)
    summary["input_quality"] = quality
    summary["debug_dump"] = dump
    (out / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _load_inputs(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if args.input:
        data = np.load(Path(args.input))
        free = np.asarray(data.get("vertical_free_room_domain", data.get("observed_free", data.get("free"))), dtype=bool)
        occupied = np.asarray(data.get("repaired_roomseg_occupied", data.get("observed_occupied", data.get("occupied", ~free))), dtype=bool)
        unknown = np.asarray(data.get("repaired_roomseg_unknown", data.get("unknown", np.zeros_like(free))), dtype=bool)
        vertical_observed = np.asarray(data.get("vertical_observed", free | occupied), dtype=bool)
        vertical_free = np.asarray(data.get("vertical_free_room_domain", free), dtype=bool)
        wall_confidence = np.asarray(data.get("wall_confidence_map", occupied.astype(np.float32)), dtype=np.float32)
        return occupied, free, unknown, vertical_observed, vertical_free, wall_confidence, "npz_full_masks"
    if args.vertical_free_png:
        arr = np.asarray(Image.open(args.vertical_free_png).convert("L"), dtype=np.uint8)
        vertical_free = arr >= 128
        vertical_observed = np.ones_like(vertical_free, dtype=bool)
        unknown = np.zeros_like(vertical_free, dtype=bool)
        occupied = vertical_observed & ~vertical_free
        wall_confidence = occupied.astype(np.float32)
        return occupied, vertical_free, unknown, vertical_observed, vertical_free, wall_confidence, "vertical_free_png_only"
    raise SystemExit("provide --input or --vertical-free-png")


if __name__ == "__main__":
    raise SystemExit(main())
