from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)
    summary = diagnose(Path(args.npz), Path(args.out_dir))
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def diagnose(npz_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    vertical_free = _mask(data, "vertical_free_room_domain", "observed_free", "free")
    observed_occupied = _mask(data, "observed_occupied", "occupied", default=np.zeros_like(vertical_free, dtype=bool))
    vertical_occupied = _mask(data, "repaired_roomseg_occupied", "vertical_occupied", default=observed_occupied)
    unknown = _mask(data, "repaired_roomseg_unknown", "unknown", default=np.zeros_like(vertical_free, dtype=bool))
    vertical_observed = _mask(data, "vertical_observed", default=vertical_free | vertical_occupied | observed_occupied)
    nonfree_observed = vertical_observed & ~vertical_free & ~unknown
    occupied_not_free = vertical_occupied & ~vertical_free
    source_labels = _labels(data, "source_room_label_map", "room_label_map")
    room_labels = _labels(data, "room_label_map_after_absorb", "room_label_map", default=source_labels)
    boundary = _mask(data, "partition_boundary_map", "boundary_map", default=np.zeros_like(vertical_free, dtype=bool))
    clean_structure = _mask(data, "clean_structure_map", default=np.zeros_like(vertical_free, dtype=bool))
    accepted = _mask(data, "accepted_separator_mask", default=boundary)
    thin_wall = _mask(data, "thin_wall_separator_mask", default=np.zeros_like(vertical_free, dtype=bool))
    doorway = _mask(data, "doorway_partition_cut_mask", default=np.zeros_like(vertical_free, dtype=bool))
    free_not_labeled = vertical_free & (room_labels <= 0)

    _metric_image(vertical_occupied, vertical_free, unknown).save(out_dir / "01_input_metric_map.png")
    _bool_image(vertical_free, true=(245, 245, 245), false=(0, 0, 0)).save(out_dir / "02_vertical_free.png")
    _bool_image(vertical_occupied, true=(245, 245, 245), false=(0, 0, 0)).save(out_dir / "03_vertical_occupied.png")
    _bool_image(unknown, true=(150, 150, 150), false=(0, 0, 0)).save(out_dir / "04_unknown.png")
    _bool_image(nonfree_observed, true=(255, 255, 255), false=(0, 0, 0)).save(out_dir / "05_nonfree_observed.png")
    _bool_image(clean_structure, true=(255, 180, 40), false=(0, 0, 0)).save(out_dir / "06_clean_structure.png")
    _boundary_overlay(vertical_free, boundary, accepted, thin_wall, doorway).save(out_dir / "07_boundary_and_lines.png")
    _label_image(source_labels).save(out_dir / "08_source_labels.png")
    _label_image(room_labels).save(out_dir / "09_room_labels.png")
    _failure_overlay(vertical_free, nonfree_observed, vertical_occupied, boundary, room_labels, free_not_labeled).save(out_dir / "10_failure_overlay.png")

    proposal_count = _label_count(source_labels)
    final_count = _label_count(room_labels)
    nonfree_cells = int(np.count_nonzero(nonfree_observed))
    occupied_support = int(np.count_nonzero(occupied_not_free))
    line_overlap = int(np.count_nonzero((boundary | accepted) & nonfree_observed))
    thin_overlap = int(np.count_nonzero(thin_wall & nonfree_observed))
    doorway_count = int(np.count_nonzero(doorway))
    summary = {
        "input_npz": str(npz_path),
        "vertical_free_cells": int(np.count_nonzero(vertical_free)),
        "vertical_occupied_cells": int(np.count_nonzero(vertical_occupied)),
        "unknown_cells": int(np.count_nonzero(unknown)),
        "nonfree_observed_cells": nonfree_cells,
        "occupied_not_free_cells": occupied_support,
        "free_not_labeled_cells": int(np.count_nonzero(free_not_labeled)),
        "thin_wall_visible_in_vertical_free": bool(nonfree_cells > 0),
        "thin_wall_has_occupied_support": bool(occupied_support > 0),
        "thin_wall_has_nonfree_observed_support": bool(nonfree_cells > 0),
        "thin_wall_detected_as_line": bool(line_overlap > 0),
        "thin_wall_generated_cut": bool(line_overlap > 0 or thin_overlap > 0),
        "doorway_gap_detected": bool(doorway_count > 0),
        "doorway_partition_cut_added": bool(doorway_count > 0),
        "proposal_room_count": int(proposal_count),
        "final_room_count": int(final_count),
        "boundary_nonfree_overlap_cells": int(line_overlap),
        "thin_wall_nonfree_overlap_cells": int(thin_overlap),
        "likely_cause": _likely_cause(
            nonfree_cells=nonfree_cells,
            occupied_support=occupied_support,
            line_overlap=line_overlap,
            doorway_count=doorway_count,
            proposal_count=proposal_count,
            final_count=final_count,
        ),
        "outputs": {
            "input_metric_map": str(out_dir / "01_input_metric_map.png"),
            "vertical_free": str(out_dir / "02_vertical_free.png"),
            "vertical_occupied": str(out_dir / "03_vertical_occupied.png"),
            "unknown": str(out_dir / "04_unknown.png"),
            "nonfree_observed": str(out_dir / "05_nonfree_observed.png"),
            "clean_structure": str(out_dir / "06_clean_structure.png"),
            "boundary_and_lines": str(out_dir / "07_boundary_and_lines.png"),
            "source_labels": str(out_dir / "08_source_labels.png"),
            "room_labels": str(out_dir / "09_room_labels.png"),
            "failure_overlay": str(out_dir / "10_failure_overlay.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _mask(data, *keys, default=None) -> np.ndarray:
    for key in keys:
        if key and key in data:
            return np.asarray(data[key], dtype=bool)
    if default is not None:
        return np.asarray(default, dtype=bool)
    raise KeyError("none of %s found in npz" % (keys,))


def _labels(data, *keys, default=None) -> np.ndarray:
    for key in keys:
        if key and key in data:
            return np.asarray(data[key], dtype=np.int32)
    if default is not None:
        return np.asarray(default, dtype=np.int32)
    raise KeyError("none of %s found in npz" % (keys,))


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


def _likely_cause(*, nonfree_cells: int, occupied_support: int, line_overlap: int, doorway_count: int, proposal_count: int, final_count: int) -> str:
    if nonfree_cells <= 0:
        return "unknown_not_wall"
    if occupied_support <= 0:
        return "not_promoted_to_separator"
    if line_overlap <= 0:
        return "line_filtered"
    if proposal_count <= 1 and doorway_count <= 0:
        return "no_doorway_partition"
    if proposal_count > final_count:
        return "final_merge_overmerge"
    if proposal_count <= 1:
        return "thin_wall_generated_no_cut"
    return "unclear"


def _bool_image(mask: np.ndarray, *, true=(255, 255, 255), false=(0, 0, 0)) -> Image.Image:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[np.asarray(mask, dtype=bool)] = true
    arr[~np.asarray(mask, dtype=bool)] = false
    return Image.fromarray(arr)


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


def _boundary_overlay(free, boundary, accepted, thin_wall, doorway) -> Image.Image:
    arr = np.zeros((*free.shape, 3), dtype=np.uint8)
    arr[free] = (70, 70, 90)
    arr[boundary] = (255, 0, 255)
    arr[accepted] = (255, 80, 255)
    arr[thin_wall] = (255, 40, 40)
    arr[doorway] = (40, 150, 255)
    return Image.fromarray(arr)


def _failure_overlay(free, nonfree, occupied, boundary, labels, free_not_labeled) -> Image.Image:
    arr = np.asarray(_label_image(labels), dtype=np.uint8).copy()
    arr[free & (labels <= 0)] = (90, 90, 90)
    arr[nonfree] = (255, 40, 40)
    arr[occupied] = (255, 180, 40)
    arr[boundary] = (255, 0, 255)
    arr[free_not_labeled] = (40, 40, 40)
    return Image.fromarray(arr)


if __name__ == "__main__":
    raise SystemExit(main())
