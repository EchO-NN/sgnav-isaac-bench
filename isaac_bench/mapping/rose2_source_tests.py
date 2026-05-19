from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaac_bench.mapping.rose2_source_form import ROSE2SourceFormConfig, SOURCE_FORM_BACKEND, run_rose2_source_form, save_rose2_source_debug
from isaac_bench.mapping.structure_extraction import StructureExtractionConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic ROSE2 source-form smoke fixtures.")
    parser.add_argument("--backend", default=SOURCE_FORM_BACKEND, choices=[SOURCE_FORM_BACKEND])
    parser.add_argument("--out-dir", default="debug/rose2_source_tests")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, builder, min_rooms, max_rooms in (
        ("two_rooms_door", _two_rooms_with_door, 2, 8),
        ("three_rooms_corridor", _three_rooms_with_corridor, 3, 12),
        ("open_plan_clutter", _open_plan_with_clutter, 1, 2),
    ):
        occupied, free, unknown = builder()
        result = run_rose2_source_form(
            observed_occupied=occupied,
            observed_free=free,
            unknown=unknown,
            structure_config=_structure_cfg(),
            source_config=_source_cfg(),
        )
        room_count = _room_count(result.room_label_map)
        ok = min_rooms <= room_count <= max_rooms and not bool(result.debug.get("legacy_connected_component_rooms_used", False))
        dump = save_rose2_source_debug(
            out_dir=out_dir / name,
            step=0,
            result=result,
            observed_occupied=occupied,
            observed_free=free,
            unknown=unknown,
        )
        row = {
            "fixture": name,
            "ok": bool(ok),
            "room_count": int(room_count),
            "min_rooms": int(min_rooms),
            "max_rooms": int(max_rooms),
            "debug_paths": dict(dump.get("paths", {})),
            "summary": dict(dump.get("summary", {})),
        }
        results.append(row)
        print("%s %s rooms=%d" % ("PASS" if ok else "FAIL", name, room_count))
    summary = {"ok": all(bool(row["ok"]) for row in results), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["ok"] else 2


def _structure_cfg() -> StructureExtractionConfig:
    return StructureExtractionConfig(
        resolution_m=0.10,
        min_room_area_m2=0.5,
        hough_min_line_length_m=0.5,
        hough_line_gap_m=0.15,
        wall_min_support_ratio=0.10,
    )


def _source_cfg() -> ROSE2SourceFormConfig:
    return ROSE2SourceFormConfig(
        resolution_m=0.10,
        min_room_area_m2=0.5,
        min_cell_area_m2=0.30,
        min_cut_spacing_m=0.40,
        min_wall_line_length_m=0.5,
    )


def _room_count(labels: np.ndarray) -> int:
    return len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0])


def _two_rooms_with_door() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (70, 90)
    free = np.zeros(shape, dtype=bool)
    free[10:60, 8:82] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 44:46] = True
    free[:, 44:46] = False
    occupied[32:42, 44:46] = False
    free[32:42, 44:46] = True
    return occupied, free, ~(occupied | free)


def _three_rooms_with_corridor() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (90, 110)
    free = np.zeros(shape, dtype=bool)
    free[8:82, 8:102] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[8:82, 52:54] = True
    free[8:82, 52:54] = False
    occupied[38:52, 52:54] = False
    free[38:52, 52:54] = True
    occupied[44:46, 54:102] = True
    free[44:46, 54:102] = False
    occupied[44:46, 75:85] = False
    free[44:46, 75:85] = True
    return occupied, free, ~(occupied | free)


def _open_plan_with_clutter() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (60, 80)
    free = np.zeros(shape, dtype=bool)
    free[8:52, 8:72] = True
    occupied = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in [(18, 20, 23, 28), (34, 42, 42, 50), (20, 58, 25, 65)]:
        occupied[r0:r1, c0:c1] = True
        free[r0:r1, c0:c1] = False
    return occupied, free, ~(occupied | free)


if __name__ == "__main__":
    raise SystemExit(main())
