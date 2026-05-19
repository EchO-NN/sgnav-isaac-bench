from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _diagnose(summary: dict) -> str:
    if int(summary.get("seed_count", 0) or 0) <= 0:
        return "no_seed_generated"
    if int(summary.get("labels_outside_vertical_free_cells", 0) or 0) > 0:
        return "boundary_absorption_failure"
    room_count = int(summary.get("final_room_count", summary.get("room_count", 0)) or 0)
    if room_count <= 0 and int(summary.get("free_cells", 0) or 0) > 0:
        return "connected_component_only_failure"
    if room_count > 20:
        return "small_fragment_failure"
    if room_count == 1 and int(summary.get("doorway_boundary_count", 0) or 0) == 0:
        return "undersegmented_no_bottleneck"
    return "good"


def _summary_from_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        data = {name: np.asarray(npz[name]) for name in npz.files}
    labels = np.asarray(data.get("final_room_label_map", np.zeros((1, 1), dtype=np.int32)), dtype=np.int32)
    free = np.asarray(data.get("input_free", data.get("clean_free", labels > 0)), dtype=bool)
    wall = np.asarray(data.get("input_wall", data.get("clean_wall", np.zeros_like(free))), dtype=bool)
    unknown = np.asarray(data.get("input_unknown", np.zeros_like(free)), dtype=bool)
    seeds = np.asarray(data.get("seed_label_map", np.zeros_like(labels)), dtype=np.int32)
    boundary = np.asarray(data.get("virtual_boundary_map", data.get("doorway_boundary_map", np.zeros_like(free))), dtype=bool)
    room_stats = []
    for label_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        room_stats.append({"label_id": label_id, "cell_count": int(np.count_nonzero(labels == label_id))})
    return {
        "backend": "vertical_free_geodesic_watershed_v1",
        "free_cells": int(np.count_nonzero(free)),
        "wall_cells": int(np.count_nonzero(wall)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "seed_count": int(len([v for v in np.unique(seeds) if int(v) > 0])),
        "initial_room_count": int(len([v for v in np.unique(data.get("initial_watershed_labels", labels)) if int(v) > 0])),
        "final_room_count": int(len([v for v in np.unique(labels) if int(v) > 0])),
        "doorway_boundary_count": int(np.count_nonzero(boundary)),
        "labels_outside_vertical_free_cells": int(np.count_nonzero((labels > 0) & ~free)),
        "largest_room_area_cells": max([int(item["cell_count"]) for item in room_stats], default=0),
        "room_stats": room_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a vertical-free room segmentation artifact.")
    parser.add_argument("--input", required=True, help="vertical_free_step_XXXXXX.npz or summary JSON.")
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
    return 0 if summary["diagnosis"] != "boundary_absorption_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
