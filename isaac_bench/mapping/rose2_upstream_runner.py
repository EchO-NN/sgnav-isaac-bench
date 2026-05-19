from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Mapping

import numpy as np

from isaac_bench.mapping.rose2_source_form import ROSE2SourceResult, SOURCE_EXTERNAL_RUNNER_BACKEND
from isaac_bench.mapping.rose2_upstream_debug import (
    label_image,
    save_rose2_upstream_debug_bundle,
    write_failure_bundle,
)
from isaac_bench.mapping.rose2_upstream_io import (
    export_rose2_upstream_input,
    normalize_rose2_masks,
    rose2_input_cache_key,
)
from isaac_bench.mapping.rose2_upstream_parser import parse_rose2_room_output


REQUIRED_UPSTREAM_SOURCE_FILES = (
    "code/FFT_MQ.py",
    "code/minibatch.py",
    "code/parameters.py",
    "code/util/layout.py",
    "code/util/postprocessing.py",
)

REQUIRED_ROSE2_WORKER_MODULES = ("skimage", "skan", "cv2", "PIL", "numpy")
DEFAULT_ROSE2_WORKER_PYTHONS = (
    "/home/echo/SG-Nav/.mamba/envs/sg-nav/bin/python",
)


@dataclass
class ROSE2UpstreamResult:
    backend: str
    room_label_map: np.ndarray
    source_room_label_map: np.ndarray
    source_output_path: Path | None
    metric_map_path: Path
    work_dir: Path
    stdout_path: Path
    stderr_path: Path
    debug: dict
    timing_ms: dict


def run_rose2_external_runner(
    *,
    source_root: str | Path | None,
    observed_free: np.ndarray,
    observed_occupied: np.ndarray,
    unknown: np.ndarray,
    resolution_m: float = 0.05,
    work_dir: str | Path,
    timeout_s: float = 60.0,
    python_executable: str | None = None,
    encoding: str = "auto",
    keep_work_dir: bool = True,
    min_room_area_m2: float = 1.5,
    vertical_observed: np.ndarray | None = None,
    vertical_free: np.ndarray | None = None,
    wall_confidence_map: np.ndarray | None = None,
    parameter_overrides: Mapping[str, object] | None = None,
    cache_enabled: bool = True,
    unknown_mode: str | None = None,
) -> ROSE2SourceResult:
    del keep_work_dir
    t0 = time.perf_counter()
    out = Path(work_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    free, occupied, unknown_arr, _ = normalize_rose2_masks(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        unknown=unknown,
    )
    full_shape = tuple(int(v) for v in free.shape)
    r0, r1, c0, c1 = _crop_to_observed(free=free, occupied=occupied, margin_cells=24)
    crop_slice = np.s_[r0:r1, c0:c1]
    work_free = free[crop_slice]
    work_occupied = occupied[crop_slice]
    work_unknown = unknown_arr[crop_slice]
    work_vertical_observed = np.asarray(vertical_observed, dtype=bool)[crop_slice] if vertical_observed is not None else None
    work_vertical_free = np.asarray(vertical_free, dtype=bool)[crop_slice] if vertical_free is not None else None
    work_wall_confidence = np.asarray(wall_confidence_map, dtype=np.float32)[crop_slice] if wall_confidence_map is not None else None
    source_root_path = Path(source_root).expanduser().resolve() if source_root else None
    source_files_found = _source_files_found(source_root_path)
    audit = {
        "requested_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "actual_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "roomseg_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_root": str(source_root_path) if source_root_path else None,
        "source_files_found": source_files_found,
        "source_form_used": False,
        "source_form_used_for_final": False,
        "legacy_style_used": False,
        "legacy_style_used_for_final": False,
        "silent_fallback_used": False,
        "legacy_connected_component_rooms_used": False,
        "external_unknown_mode": str(unknown_mode or ""),
        "crop_bounds_rc": [int(r0), int(r1), int(c0), int(c1)],
        "full_shape": [int(full_shape[0]), int(full_shape[1])],
        "source_input_shape": [int(work_free.shape[0]), int(work_free.shape[1])],
    }
    cache_key = rose2_input_cache_key(
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown_arr,
        encoding=str(encoding or "auto"),
        parameter_overrides=parameter_overrides,
    )
    cache_dir = out / ".cache" / cache_key
    cached_labels = cache_dir / "parsed_labels.npy"
    cached_summary = cache_dir / "summary.json"
    if bool(cache_enabled) and cached_labels.exists() and cached_summary.exists():
        labels = np.load(cached_labels).astype(np.int32)
        summary = json.loads(cached_summary.read_text(encoding="utf-8"))
        summary.update(audit)
        summary["cache_hit"] = True
        debug = _debug_from_summary(summary, labels)
        return _source_result_from_labels(labels, occupied, wall_confidence_map, debug, timing_ms=dict(summary.get("timing_ms", {})))

    missing = [rel for rel, ok in source_files_found.items() if not ok]
    if source_root_path is None or missing:
        reason = "missing_rose2_source_root" if source_root_path is None or not source_root_path.exists() else "missing_required_source_files"
        summary = {**audit, "ok": False, "failure_reason": reason, "missing_required_files": missing, "cache_key": cache_key}
        write_failure_bundle(out_dir=out, stem="rose2_source_external", reason=reason, summary=summary)
        (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        label_image(np.zeros_like(occupied, dtype=np.int32)).save(out / "external_room_label_map.png")
        raise FileNotFoundError(_missing_source_message(source_root_path, missing))

    worker_python, worker_python_selection = _select_rose2_worker_python(python_executable)
    audit["worker_python"] = worker_python
    audit["worker_python_selection"] = worker_python_selection
    if not worker_python:
        reason = "missing_rose2_worker_python_dependencies"
        summary = {
            **audit,
            "ok": False,
            "failure_reason": reason,
            "failure_detail": _worker_python_error_message(worker_python_selection),
            "cache_key": cache_key,
        }
        write_failure_bundle(out_dir=out, stem="rose2_source_external", reason=reason, summary=summary)
        (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        label_image(np.zeros_like(occupied, dtype=np.int32)).save(out / "external_room_label_map.png")
        raise RuntimeError(summary["failure_detail"])

    bundle = export_rose2_upstream_input(
        observed_free=work_free,
        observed_occupied=work_occupied,
        unknown=work_unknown,
        resolution_m=float(resolution_m),
        out_dir=out,
        stem="rose2_source_step",
        encoding=str(encoding or "auto"),
        vertical_observed=work_vertical_observed,
        vertical_free=work_vertical_free,
        wall_confidence_map=work_wall_confidence,
    )
    stdout_path = out / "stdout.txt"
    stderr_path = out / "stderr.txt"
    worker_json = out / "source_result.json"
    command = [
        str(worker_python),
        "-m",
        "isaac_bench.scripts.rose2_external_worker",
        "--source-root",
        str(source_root_path),
        "--input-npz",
        str(bundle.npz_path),
        "--metric-map",
        str(bundle.metric_map_path),
        "--orebro-input",
        str(bundle.orebro_input_path),
        "--work-dir",
        str(out),
        "--output-json",
        str(worker_json),
        "--parameter-overrides-json",
        json.dumps(dict(parameter_overrides or {}), ensure_ascii=False),
    ]
    return_code = None
    try:
        repo_root = str(Path(__file__).resolve().parents[2])
        env = _rose2_worker_subprocess_env(repo_root=repo_root)
        proc = subprocess.run(
            command,
            cwd=str(source_root_path / "code"),
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
            env=env,
        )
        return_code = int(proc.returncode)
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    except Exception as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("%s: %s" % (type(exc).__name__, exc), encoding="utf-8")
        worker_json.write_text(
            json.dumps({"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    worker_payload = _read_json(worker_json)
    source_output = Path(worker_payload["room_output_path"]).resolve() if worker_payload.get("room_output_path") else None
    if source_output is None or not source_output.exists():
        reason = str(worker_payload.get("error_message") or worker_payload.get("minibatch_error") or "external_runner_failed_or_no_room_image")
        summary = {
            **audit,
            "ok": False,
            "return_code": return_code,
            "failure_reason": "external_runner_failed_or_no_room_image",
            "failure_detail": reason,
            "command": command,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "worker_json": str(worker_json),
            "cache_key": cache_key,
        }
        write_failure_bundle(out_dir=out, stem="rose2_source_external", reason=summary["failure_reason"], summary=summary)
        (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("ROSE2 upstream source runner failed without room output: %s" % reason)

    labels, parser_debug = parse_rose2_room_output(
        output_png=source_output,
        observed_free=work_free,
        unknown=work_unknown,
        min_room_area_m2=float(min_room_area_m2),
        resolution_m=float(resolution_m),
    )
    if (r0, r1, c0, c1) != (0, full_shape[0], 0, full_shape[1]):
        labels_full = np.zeros(full_shape, dtype=np.int32)
        labels_full[crop_slice] = labels.astype(np.int32)
        labels = labels_full
    room_count = _label_count(labels)
    if room_count <= 0:
        reason = "source_output_parsed_zero_rooms"
        summary = {
            **audit,
            "ok": False,
            "failure_reason": reason,
            "source_output_path": str(source_output),
            "parser_debug": parser_debug,
            "cache_key": cache_key,
        }
        write_failure_bundle(out_dir=out, stem="rose2_source_external", reason=reason, summary=summary)
        (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("ROSE2 upstream source output parsed zero rooms: %s" % source_output)

    timing_ms = dict(worker_payload.get("timing_ms") or {})
    timing_ms["total_with_parse"] = float((time.perf_counter() - t0) * 1000.0)
    summary = {
        **audit,
        "ok": True,
        "external_source_ok": True,
        "return_code": return_code,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "worker_json": str(worker_json),
        "metric_map_path": str(bundle.metric_map_path),
        "work_dir": str(out),
        "orebro_input_path": str(bundle.orebro_input_path),
        "input_npz": str(bundle.npz_path),
        "input_summary_json": str(bundle.summary_json_path),
        "chosen_encoding": str(bundle.chosen_encoding),
        "encoding_candidates": list(bundle.encoding_candidates),
        "source_output_path": str(source_output),
        "source_room_count": int(room_count),
        "parsed_label_count": int(room_count),
        "room_count": int(room_count),
        "source_form_debug_room_count": None,
        "source_form_used_for_final": False,
        "cache_hit": False,
        "cache_key": cache_key,
        "parser_debug": parser_debug,
        "timing_ms": timing_ms,
        "worker": worker_payload,
    }
    debug_paths = save_rose2_upstream_debug_bundle(
        out_dir=out,
        stem="rose2_source_step",
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown_arr,
        room_label_map=labels,
        source_output_path=source_output,
        summary=summary,
    )
    summary["debug_paths"] = debug_paths.get("paths", {})
    (out / "external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    label_image(labels).save(out / "external_room_label_map.png")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cached_labels, labels.astype(np.int32))
    cached_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    debug = _debug_from_summary(summary, labels)
    return _source_result_from_labels(labels, occupied, wall_confidence_map, debug, timing_ms=timing_ms)


def run_rose2_source_external_runner(**kwargs) -> ROSE2SourceResult:
    return run_rose2_external_runner(**kwargs)


def _source_files_found(source_root: Path | None) -> dict[str, bool]:
    if source_root is None:
        return {rel: False for rel in REQUIRED_UPSTREAM_SOURCE_FILES}
    return {rel: bool((source_root / rel).exists()) for rel in REQUIRED_UPSTREAM_SOURCE_FILES}


def _select_rose2_worker_python(python_executable: str | None) -> tuple[str | None, dict]:
    candidates = _rose2_worker_python_candidates(python_executable)
    attempts: list[dict] = []
    stopped_after_strict_candidate = False
    for candidate in candidates:
        result = _check_rose2_worker_python(str(candidate["path"]))
        attempt = {**candidate, **result}
        attempts.append(attempt)
        if bool(result.get("ok")):
            return str(result.get("resolved_path") or candidate["path"]), {
                "ok": True,
                "selected": str(result.get("resolved_path") or candidate["path"]),
                "selected_source": str(candidate["source"]),
                "required_modules": list(REQUIRED_ROSE2_WORKER_MODULES),
                "attempts": attempts,
            }
        if bool(candidate.get("strict")):
            stopped_after_strict_candidate = True
            break
    return None, {
        "ok": False,
        "selected": None,
        "required_modules": list(REQUIRED_ROSE2_WORKER_MODULES),
        "attempts": attempts,
        "stopped_after_strict_candidate": stopped_after_strict_candidate,
    }


def _rose2_worker_python_candidates(python_executable: str | None) -> list[dict]:
    explicit = str(python_executable or "").strip()
    if explicit:
        return [{"source": "config.external_runner.python_executable", "path": explicit, "strict": True}]

    env_python = str(os.environ.get("ROSE2_SOURCE_PYTHON", "")).strip()
    if env_python:
        return [{"source": "env.ROSE2_SOURCE_PYTHON", "path": env_python, "strict": True}]

    candidates: list[dict] = []
    seen: set[str] = set()
    for path in (*DEFAULT_ROSE2_WORKER_PYTHONS, sys.executable, shutil.which("python3") or ""):
        text = str(path or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append({"source": "auto", "path": text, "strict": False})
    return candidates


def _check_rose2_worker_python(path: str) -> dict:
    path_text = str(path or "").strip()
    if not path_text:
        return {"ok": False, "error": "empty_python_path"}
    resolved = Path(path_text).expanduser()
    if not resolved.exists():
        return {"ok": False, "resolved_path": str(resolved), "error": "python_not_found"}
    if resolved.is_dir():
        return {"ok": False, "resolved_path": str(resolved), "error": "python_path_is_directory"}
    script = (
        "import importlib\n"
        "mods = %r\n"
        "for name in mods:\n"
        "    importlib.import_module(name)\n"
        "print('rose2-worker-python-ok')\n"
    ) % (tuple(REQUIRED_ROSE2_WORKER_MODULES),)
    try:
        proc = subprocess.run(
            [str(resolved), "-c", script],
            text=True,
            capture_output=True,
            timeout=8.0,
            check=False,
            env=_rose2_worker_subprocess_env(repo_root=None),
        )
    except Exception as exc:
        return {
            "ok": False,
            "resolved_path": str(resolved),
            "error": type(exc).__name__,
            "stderr": str(exc),
        }
    return {
        "ok": proc.returncode == 0,
        "resolved_path": str(resolved),
        "return_code": int(proc.returncode),
        "stdout": (proc.stdout or "").strip()[-500:],
        "stderr": (proc.stderr or "").strip()[-1000:],
    }


def _rose2_worker_subprocess_env(*, repo_root: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    if repo_root:
        env["PYTHONPATH"] = str(repo_root)
    else:
        env.pop("PYTHONPATH", None)
    return env


def _worker_python_error_message(selection: Mapping[str, object]) -> str:
    attempts = selection.get("attempts")
    attempt_text = ""
    if isinstance(attempts, list):
        parts = []
        for item in attempts:
            if isinstance(item, Mapping):
                parts.append(
                    "%s=%s ok=%s error=%s stderr=%s"
                    % (
                        item.get("source"),
                        item.get("path"),
                        item.get("ok"),
                        item.get("error", ""),
                        str(item.get("stderr", ""))[:300],
                    )
                )
        attempt_text = "; ".join(parts)
    return (
        "ROSE2 upstream source runner requires a Python environment with modules %s. "
        "Set ROSE2_SOURCE_PYTHON=/home/echo/SG-Nav/.mamba/envs/sg-nav/bin/python "
        "or set mapping.room_segmentation.external_runner.python_executable. Attempts: %s"
        % (", ".join(REQUIRED_ROSE2_WORKER_MODULES), attempt_text)
    )


def _crop_to_observed(*, free: np.ndarray, occupied: np.ndarray, margin_cells: int = 24) -> tuple[int, int, int, int]:
    observed = np.asarray(free, dtype=bool) | np.asarray(occupied, dtype=bool)
    h, w = observed.shape
    if not np.any(observed):
        return 0, int(h), 0, int(w)
    rr, cc = np.nonzero(observed)
    margin = max(0, int(margin_cells))
    return (
        max(0, int(rr.min()) - margin),
        min(int(h), int(rr.max()) + margin + 1),
        max(0, int(cc.min()) - margin),
        min(int(w), int(cc.max()) + margin + 1),
    )


def _missing_source_message(source_root: Path | None, missing: list[str]) -> str:
    root_text = str(source_root) if source_root is not None else "<unset>"
    return (
        "Missing upstream ROSE2 source root or required files for rose2_source_external_runner. "
        "root=%s missing=%s. Set ROSE2_SOURCE_ROOT to goldleaf3i/declutter-reconstruct."
        % (root_text, ", ".join(missing or REQUIRED_UPSTREAM_SOURCE_FILES))
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)}


def _debug_from_summary(summary: Mapping[str, object], labels: np.ndarray) -> dict:
    room_count = _label_count(labels)
    external_summary = dict(summary)
    return {
        "source_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "roomseg_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "requested_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "actual_backend": SOURCE_EXTERNAL_RUNNER_BACKEND,
        "source_exact_used": True,
        "external_source_ok": bool(summary.get("ok", True)),
        "external_runner_summary": external_summary,
        "rose2_source_external": external_summary,
        "external_failure_reason": str(summary.get("failure_reason", "")),
        "source_room_count": int(room_count),
        "proposal_room_count": int(room_count),
        "source_form_used": False,
        "source_form_used_for_final": False,
        "legacy_style_used": False,
        "legacy_style_used_for_final": False,
        "silent_fallback_used": False,
        "legacy_connected_component_rooms_used": False,
        "failure_mode": str(summary.get("failure_reason", "")),
        "source_output_path": str(summary.get("source_output_path", "")),
        "metric_map_path": str(summary.get("metric_map_path", "")),
        "parsed_labels_png": str((summary.get("debug_paths") or {}).get("parsed_labels_png", "")) if isinstance(summary.get("debug_paths"), dict) else "",
        "source_output_png": str((summary.get("debug_paths") or {}).get("source_output_png", "")) if isinstance(summary.get("debug_paths"), dict) else "",
        "summary_json": str((summary.get("debug_paths") or {}).get("summary_json", "")) if isinstance(summary.get("debug_paths"), dict) else "",
    }


def _source_result_from_labels(
    labels: np.ndarray,
    observed_occupied: np.ndarray,
    wall_confidence_map: np.ndarray | None,
    debug: Mapping[str, object],
    *,
    timing_ms: Mapping[str, object] | None = None,
) -> ROSE2SourceResult:
    label_arr = np.asarray(labels, dtype=np.int32)
    occupied = np.asarray(observed_occupied, dtype=bool)
    score = np.asarray(wall_confidence_map, dtype=np.float32) if wall_confidence_map is not None else occupied.astype(np.float32)
    if score.shape != occupied.shape:
        score = occupied.astype(np.float32)
    return ROSE2SourceResult(
        backend=SOURCE_EXTERNAL_RUNNER_BACKEND,
        room_label_map=label_arr,
        source_room_label_map=label_arr.copy(),
        clean_structure_map=occupied.astype(bool),
        structural_score=score.astype(np.float32),
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
        timing_ms=dict(timing_ms or {}),
        debug=dict(debug),
    )


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))
