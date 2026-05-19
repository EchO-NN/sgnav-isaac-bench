from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from isaac_bench.mapping.rose2_upstream_io import load_rose2_upstream_input_npz


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose an upstream ROSE2 source runner output bundle.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    occupied, free, unknown, *_ = load_rose2_upstream_input_npz(Path(args.input_npz))
    labels_path = Path(args.labels) if args.labels else _labels_from_summary(summary)
    labels = np.load(labels_path).astype(np.int32) if labels_path and labels_path.exists() else np.zeros_like(free, dtype=np.int32)
    report = {
        "backend_is_external_source": str(summary.get("actual_backend", summary.get("backend", ""))) == "rose2_source_external_runner",
        "silent_fallback_used": bool(summary.get("silent_fallback_used", False)),
        "source_form_used_for_final": bool(summary.get("source_form_used_for_final", False)),
        "source_output_exists": bool(summary.get("source_output_path")) and Path(str(summary.get("source_output_path"))).exists(),
        "labels_only_cover_vertical_free": bool(not np.any((labels > 0) & ~free)),
        "labels_unknown_cells": int(np.count_nonzero((labels > 0) & unknown)),
        "labels_on_occupied_cells": int(np.count_nonzero((labels > 0) & occupied)),
        "room_count": int(len([v for v in np.unique(labels) if int(v) > 0])),
        "unlabeled_free_cells": int(np.count_nonzero(free & (labels <= 0))),
        "shape": [int(free.shape[0]), int(free.shape[1])],
        "shape_transform": str((summary.get("parser_debug") or {}).get("shape_transform", "")) if isinstance(summary.get("parser_debug"), dict) else "",
        "foreground_observed_free_iou": float((summary.get("parser_debug") or {}).get("foreground_observed_free_iou", 0.0)) if isinstance(summary.get("parser_debug"), dict) else 0.0,
    }
    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if report["backend_is_external_source"] and not report["silent_fallback_used"] else 2


def _labels_from_summary(summary: dict) -> Path | None:
    debug_paths = summary.get("debug_paths")
    if isinstance(debug_paths, dict):
        value = debug_paths.get("parsed_labels_npy")
        if value:
            return Path(str(value))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
