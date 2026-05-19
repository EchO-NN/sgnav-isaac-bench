from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from isaac_bench.mapping.rose2_external_runner import run_rose2_external_runner
from isaac_bench.mapping.rose2_source_form import (
    ROSE2SourceFormConfig,
    run_rose2_source_faithful_v1,
    run_rose2_source_form,
    run_rose2_source_form_v2,
    save_rose2_source_debug,
    source_result_summary,
)
from isaac_bench.mapping.structure_extraction import StructureExtractionConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--backend", required=True, choices=["rose2_source_form", "rose2_source_form_v2", "rose2_source_faithful_v1", "rose2_source_external_runner"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--min-room-area-m2", type=float, default=1.5)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--unknown-mode", choices=["gray", "free"], default="gray")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    occupied, free, unknown, vertical_observed, vertical_free, wall_confidence = _load_inputs(Path(args.input))
    if args.unknown_mode == "free":
        free = free | unknown
        unknown = np.zeros_like(free, dtype=bool)

    if args.backend == "rose2_source_external_runner":
        result = run_rose2_external_runner(
            source_root=args.source_root,
            observed_occupied=occupied,
            observed_free=free,
            unknown=unknown,
            work_dir=out,
            timeout_s=float(args.timeout_s),
            unknown_mode=str(args.unknown_mode),
        )
        summary = source_result_summary(result)
        summary["external_runner_summary"] = result.debug.get("external_runner_summary", {})
        _label_image(result.room_label_map).save(out / "external_labels.png")
        _label_image(result.room_label_map).save(out / "room_labels.png")
        _metric_image(occupied, free, unknown).save(out / "input_metric.png")
    else:
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
            thin_wall_min_length_m=0.35,
            thin_wall_max_width_m=0.30,
            thin_wall_min_free_support_ratio=0.20,
            topology_effective_max_candidates=256,
            debug_dump=True,
            debug_dir=str(out),
        )
        if args.backend == "rose2_source_form":
            result = run_rose2_source_form(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                structure_config=structure_cfg,
                source_config=source_cfg,
            )
        elif args.backend == "rose2_source_form_v2":
            result = run_rose2_source_form_v2(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                vertical_observed=vertical_observed,
                vertical_free=vertical_free,
                wall_confidence_map=wall_confidence,
                structure_config=structure_cfg,
                source_config=source_cfg,
            )
        else:
            result = run_rose2_source_faithful_v1(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                vertical_observed=vertical_observed,
                vertical_free=vertical_free,
                wall_confidence_map=wall_confidence,
                structure_config=structure_cfg,
                source_config=source_cfg,
            )
        dump = save_rose2_source_debug(
            out_dir=out,
            step=0,
            result=result,
            observed_occupied=occupied,
            observed_free=free,
            unknown=unknown,
        )
        summary = source_result_summary(result)
        summary["debug_dump"] = dump
        _copy_named_outputs(out)
    summary["backend"] = str(args.backend)
    summary["input"] = str(args.input)
    summary["unknown_mode"] = str(args.unknown_mode)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def _load_inputs(path: Path):
    data = np.load(path)
    free = np.asarray(data.get("vertical_free_room_domain", data.get("observed_free", data.get("free"))), dtype=bool)
    occupied = np.asarray(data.get("repaired_roomseg_occupied", data.get("observed_occupied", data.get("occupied", np.zeros_like(free)))), dtype=bool)
    unknown = np.asarray(data.get("repaired_roomseg_unknown", data.get("unknown", np.zeros_like(free))), dtype=bool)
    vertical_observed = np.asarray(data.get("vertical_observed", free | occupied), dtype=bool)
    vertical_free = np.asarray(data.get("vertical_free_room_domain", free), dtype=bool)
    wall_confidence = np.asarray(data.get("wall_confidence_map", occupied.astype(np.float32)), dtype=np.float32)
    return occupied, free, unknown, vertical_observed, vertical_free, wall_confidence


def _copy_named_outputs(out: Path) -> None:
    stems = list(out.glob("rose2_source_step_000000.*.png"))
    mapping = {
        "metric_input": "input_metric.png",
        "vertical_free": "vertical_free.png",
        "vertical_occupied": "vertical_occupied.png",
        "observed_not_vertical_free": "nonfree_observed.png",
        "thin_wall_candidates": "thin_wall_candidates.png",
        "accepted_separators": "accepted_separators.png",
        "partition_boundary": "partition_boundary.png",
        "room_labels": "room_labels.png",
        "overlay": "overlay.png",
    }
    for path in stems:
        for key, name in mapping.items():
            if path.name.endswith("." + key + ".png"):
                Image.open(path).save(out / name)


def _metric_image(occupied: np.ndarray, free: np.ndarray, unknown: np.ndarray) -> Image.Image:
    arr = np.zeros((*free.shape, 3), dtype=np.uint8)
    arr[unknown] = (128, 128, 128)
    arr[free] = (255, 255, 255)
    arr[occupied] = (0, 0, 0)
    return Image.fromarray(arr)


def _label_image(labels: np.ndarray) -> Image.Image:
    arr = np.zeros((*labels.shape, 3), dtype=np.uint8)
    palette = [(126, 174, 255), (255, 156, 102), (130, 222, 150), (214, 148, 255), (250, 216, 95)]
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        arr[labels == label] = palette[(label - 1) % len(palette)]
    return Image.fromarray(arr)


if __name__ == "__main__":
    raise SystemExit(main())
