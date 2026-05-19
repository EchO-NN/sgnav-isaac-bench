from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _diagnose(summary: dict) -> str:
    rooms = int(summary.get("num_rooms_final", summary.get("final_room_count", summary.get("room_count", 0))) or 0)
    accepted = int(summary.get("num_accepted_closures", summary.get("accepted_closure_count", 0)) or 0)
    rejected = int(summary.get("num_rejected_closures", summary.get("rejected_closure_count", 0)) or 0)
    if int(summary.get("labels_outside_vertical_free_cells", 0) or 0) > 0:
        return "label_escaped_vertical_free_domain"
    if int(summary.get("labels_in_unknown_cells", 0) or 0) > 0:
        return "label_escaped_into_unknown"
    if rooms <= 0 and int(summary.get("free_cells", 0) or 0) > 0:
        return "no_room_labels"
    if accepted <= 0 and rejected <= 0 and int(summary.get("num_endpoints", 0) or 0) > 0:
        return "no_gap_candidates"
    if rooms == 1 and accepted <= 0:
        return "undersegmented_no_verified_gap_closure"
    return "good"


def _summary_from_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        data = {name: np.asarray(npz[name]) for name in npz.files}
    labels = np.asarray(data.get("room_label_map", data.get("final_room_label_map", np.zeros((1, 1), dtype=np.int32))), dtype=np.int32)
    free = np.asarray(data.get("free", data.get("clean_free", data.get("input_free", labels > 0))), dtype=bool)
    wall = np.asarray(data.get("wall", data.get("clean_wall", data.get("input_wall", np.zeros_like(free)))), dtype=bool)
    unknown = np.asarray(data.get("unknown", data.get("input_unknown", np.zeros_like(free))), dtype=bool)
    accepted = np.asarray(data.get("accepted_closure_map", data.get("virtual_boundary_map", np.zeros_like(free))), dtype=bool)
    rejected = np.asarray(data.get("rejected_closure_map", np.zeros_like(free)), dtype=bool)
    return {
        "backend": "vertical_free_gap_closure_v1",
        "free_cells": int(np.count_nonzero(free)),
        "wall_cells": int(np.count_nonzero(wall)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "num_rooms_final": int(len([v for v in np.unique(labels) if int(v) > 0])),
        "num_accepted_closure_cells": int(np.count_nonzero(accepted)),
        "num_rejected_closure_cells": int(np.count_nonzero(rejected)),
        "labels_outside_vertical_free_cells": int(np.count_nonzero((labels > 0) & ~free)),
        "labels_in_unknown_cells": int(np.count_nonzero((labels > 0) & unknown)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a vertical-free gap-closure room segmentation artifact.")
    parser.add_argument("--input", required=True, help="vfgc_step_XXXXXX.npz or summary JSON.")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    path = Path(args.input).expanduser()
    if path.suffix == ".json":
        summary = json.loads(path.read_text(encoding="utf-8"))
    else:
        summary = _summary_from_npz(path)
    summary["diagnosis"] = _diagnose(summary)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "inspection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_summary or not args.out_dir:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["diagnosis"] not in {"label_escaped_vertical_free_domain", "label_escaped_into_unknown"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
