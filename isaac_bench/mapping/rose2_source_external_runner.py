from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import numpy as np
from PIL import Image

from isaac_bench.mapping.rose2_source_form import (
    ROSE2SourceResult,
    SOURCE_EXTERNAL_RUNNER_BACKEND,
    export_rose2_source_input,
    parse_rose2_source_label_image,
)


REQUIRED_EXTERNAL_FILES = (
    "code/FFT_MQ.py",
    "code/minibatch.py",
    "code/parameters.py",
    "code/runMe.py",
    "code/util/layout.py",
    "code/util/postprocessing.py",
)


def run_rose2_source_external_runner(
    *,
    source_root: str | Path | None,
    observed_occupied: np.ndarray,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    work_dir: str | Path,
    timeout_s: float = 60.0,
) -> ROSE2SourceResult:
    out = Path(work_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    occupied = np.asarray(observed_occupied, dtype=bool)
    free = np.asarray(observed_free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    export_paths = export_rose2_source_input(
        out_dir=out,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown_arr,
        stem="external_source_input",
    )
    orebro = out / "input_orebro.png"
    Image.open(export_paths["metric_map_png"]).save(orebro)
    root = Path(source_root).expanduser() if source_root else None
    failure_reason = ""
    missing = []
    if root is None or not root.exists():
        failure_reason = "missing_rose2_source_root"
    else:
        missing = [name for name in REQUIRED_EXTERNAL_FILES if not (root / name).exists()]
        if missing:
            failure_reason = "missing_required_source_files"

    wrapper = out / "run_external_rose2.py"
    stdout_path = out / "stdout.txt"
    stderr_path = out / "stderr.txt"
    label_map = np.zeros_like(occupied, dtype=np.int32)
    command: list[str] = []
    return_code = None
    output_room_image = None
    if not failure_reason:
        wrapper.write_text(_wrapper_source(root, Path(export_paths["metric_map_png"]), orebro, out), encoding="utf-8")
        command = [sys.executable, str(wrapper)]
        try:
            proc = subprocess.run(
                command,
                cwd=str(root / "code"),
                text=True,
                capture_output=True,
                timeout=float(timeout_s),
                check=False,
            )
            return_code = int(proc.returncode)
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            room_images = sorted(out.glob("**/*rooms*.png"))
            if proc.returncode == 0 and room_images:
                output_room_image = room_images[0]
                label_map = parse_rose2_source_label_image(output_room_image)
            else:
                failure_reason = "external_runner_failed_or_no_room_image"
        except Exception as exc:
            failure_reason = "external_runner_exception:%s" % type(exc).__name__
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
    else:
        wrapper.write_text("# external runner was not launched: %s\n" % failure_reason, encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("missing=%s\n" % missing, encoding="utf-8")

    summary = {
        "backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_root": str(root) if root else None,
        "required_files_present": not bool(missing),
        "missing_required_files": missing,
        "command": command,
        "return_code": return_code,
        "failure_reason": failure_reason,
        "export_paths": export_paths,
        "input_orebro_png": str(orebro),
        "wrapper": str(wrapper),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "output_room_image": str(output_room_image) if output_room_image else None,
        "source_room_count": _label_count(label_map),
    }
    (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _label_image(label_map).save(out / "external_room_label_map.png")
    debug = {
        "source_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_exact_used": bool(output_room_image and not failure_reason),
        "external_runner_summary": summary,
        "external_failure_reason": failure_reason,
        "source_room_count": int(_label_count(label_map)),
        "proposal_room_count": int(_label_count(label_map)),
        "legacy_connected_component_rooms_used": False,
        "failure_mode": failure_reason,
    }
    return ROSE2SourceResult(
        backend=SOURCE_EXTERNAL_RUNNER_BACKEND,
        room_label_map=label_map.astype(np.int32),
        source_room_label_map=label_map.astype(np.int32),
        clean_structure_map=occupied.astype(bool),
        structural_score=occupied.astype(np.float32),
        boundary_map=occupied.astype(bool),
        dominant_directions_rad=[],
        hough_segments=[],
        wall_clusters=[],
        representative_lines=[],
        extended_lines=[],
        faces=[],
        face_adjacency_edges=[],
        cell_edges=[],
        cell_polygons=[],
        timing_ms={},
        debug=debug,
    )


def _wrapper_source(source_root: Path, metric_map: Path, orebro: Path, out: Path) -> str:
    return f"""from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

source_root = Path({str(source_root)!r})
metric_map = Path({str(metric_map)!r})
orebro_seed = Path({str(orebro)!r})
out = Path({str(out)!r})
sys.path.insert(0, str(source_root / 'code' / 'rose_v1_repo'))
sys.path.insert(0, str(source_root / 'code'))
import parameters as par
import FFT_MQ as fft
import minibatch

params = par.ParameterObj()
paths = par.PathObj()
paths.metric_map_name = 'codex_metric_map'
paths.metric_map_path = str(metric_map)
paths.path_folder_output = str(out)
paths.path_log_folder = str(out)
paths.filepath = str(out) + os.sep
paths.path_orebro = str(out / 'OREBRO')
paths.orebro_img = str(out / ('OREBRO_' + str(params.filter_level) + '.png'))
paths.gt_color = ''
Path(paths.path_orebro).mkdir(parents=True, exist_ok=True)
fft.main(paths.metric_map_path, paths.path_orebro, params.filter_level, params)
if not Path(paths.orebro_img).exists():
    shutil.copy(str(orebro_seed), paths.orebro_img)
minibatch.start_main(par, params, paths)
print('external_rose2_done', paths.filepath)
"""


def _label_image(labels: np.ndarray) -> Image.Image:
    arr = np.asarray(labels, dtype=np.int32)
    canvas = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        canvas[arr == label_id] = _palette(label_id)
    return Image.fromarray(canvas)


def _palette(label_id: int) -> tuple[int, int, int]:
    palette = [(126, 174, 255), (255, 156, 102), (130, 222, 150), (214, 148, 255), (250, 216, 95)]
    return palette[(int(label_id) - 1) % len(palette)]


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


def _load_input_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    occupied = np.asarray(data.get("observed_occupied", data.get("occupied")), dtype=bool)
    free = np.asarray(data.get("observed_free", data.get("free")), dtype=bool)
    unknown = np.asarray(data.get("unknown", np.zeros_like(free)), dtype=bool)
    return occupied, free, unknown


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args(argv)
    occupied, free, unknown = _load_input_npz(Path(args.input_npz))
    result = run_rose2_source_external_runner(
        source_root=args.source_root,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        work_dir=args.work_dir,
        timeout_s=float(args.timeout_s),
    )
    print(json.dumps(result.debug.get("external_runner_summary", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
