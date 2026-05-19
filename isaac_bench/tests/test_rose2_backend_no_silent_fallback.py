from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.rose2_source_form import SOURCE_EXTERNAL_RUNNER_BACKEND
import isaac_bench.mapping.rose2_upstream_runner as rose2_runner
from isaac_bench.mapping.rose2_upstream_runner import run_rose2_external_runner
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import UpstreamROSE2Config, UpstreamROSE2PurePythonSegmenter
from isaac_bench.metrics.result_schema import BenchmarkAssetError, validate_strict_benchmark_assets
from isaac_bench.scripts.rose2_external_worker import _install_source_file_compat_patches, find_rose2_room_output, score_room_output_candidate


def test_upstream_segmenter_strict_default_is_external_runner(tmp_path):
    shape = (20, 20)
    cfg = UpstreamROSE2Config(source_root=str(tmp_path / "missing"), debug_dump=False)

    assert cfg.backend == SOURCE_EXTERNAL_RUNNER_BACKEND
    with pytest.raises(FileNotFoundError):
        UpstreamROSE2PurePythonSegmenter(cfg, _map_info(shape))


def test_strict_contract_rejects_source_form_backend(tmp_path):
    args = Namespace(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode="upstream_rose2_vertical_or_free",
        room_segmentation_config={
            "backend": "rose2_source_faithful_v1",
            "source_root": str(tmp_path / "missing"),
            "require_upstream_source_for_strict": True,
        },
        ablation_name="",
    )

    with pytest.raises(BenchmarkAssetError, match="debug/ablation-only"):
        validate_strict_benchmark_assets(args)


def test_missing_external_source_writes_failure_bundle_and_does_not_fallback(tmp_path):
    free = np.zeros((24, 32), dtype=bool)
    free[4:20, 4:28] = True
    occupied = np.zeros_like(free)
    unknown = ~(free | occupied)

    with pytest.raises(FileNotFoundError):
        run_rose2_external_runner(
            source_root=tmp_path / "missing_declutter",
            observed_free=free,
            observed_occupied=occupied,
            unknown=unknown,
            work_dir=tmp_path / "external",
            timeout_s=1.0,
        )

    summary = json.loads((tmp_path / "external" / "external_summary.json").read_text())
    assert summary["actual_backend"] == "rose2_source_external_runner"
    assert summary["source_form_used"] is False
    assert summary["legacy_style_used"] is False
    assert summary["silent_fallback_used"] is False
    assert summary["failure_reason"] in {"missing_rose2_source_root", "missing_required_source_files"}


def test_external_runner_parses_fake_upstream_source_without_source_form(tmp_path):
    source_root = _fake_declutter_source(tmp_path)
    free = np.zeros((36, 50), dtype=bool)
    free[5:31, 5:45] = True
    occupied = np.zeros_like(free)
    occupied[5:31, 25] = True
    free[occupied] = False
    unknown = ~(free | occupied)

    result = run_rose2_external_runner(
        source_root=source_root,
        observed_free=free,
        observed_occupied=occupied,
        unknown=unknown,
        resolution_m=0.10,
        min_room_area_m2=0.05,
        work_dir=tmp_path / "external_success",
        timeout_s=10.0,
        encoding="source_black_wall_white_free",
        cache_enabled=False,
    )

    summary = result.debug["external_runner_summary"]
    assert result.backend == "rose2_source_external_runner"
    assert summary["actual_backend"] == "rose2_source_external_runner"
    assert summary["source_form_used_for_final"] is False
    assert summary["silent_fallback_used"] is False
    assert summary["source_output_path"]
    assert summary["room_count"] >= 2
    assert np.all(result.room_label_map[unknown] == 0)
    assert np.all(result.room_label_map[occupied] == 0)


def test_rose2_worker_python_uses_env_when_config_not_set(monkeypatch):
    monkeypatch.setenv("ROSE2_SOURCE_PYTHON", "/tmp/rose2-env-python")

    def fake_check(path: str) -> dict:
        return {"ok": True, "resolved_path": path, "return_code": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(rose2_runner, "_check_rose2_worker_python", fake_check)

    selected, debug = rose2_runner._select_rose2_worker_python(None)

    assert selected == "/tmp/rose2-env-python"
    assert debug["ok"] is True
    assert debug["selected_source"] == "env.ROSE2_SOURCE_PYTHON"
    assert debug["attempts"][0]["path"] == "/tmp/rose2-env-python"


def test_configured_bad_rose2_worker_python_does_not_silently_fallback(monkeypatch):
    monkeypatch.delenv("ROSE2_SOURCE_PYTHON", raising=False)

    def fake_check(path: str) -> dict:
        return {"ok": False, "resolved_path": path, "return_code": 1, "stdout": "", "stderr": "missing skimage"}

    monkeypatch.setattr(rose2_runner, "_check_rose2_worker_python", fake_check)

    selected, debug = rose2_runner._select_rose2_worker_python("/tmp/bad-python")

    assert selected is None
    assert debug["ok"] is False
    assert debug["stopped_after_strict_candidate"] is True
    assert len(debug["attempts"]) == 1
    assert debug["attempts"][0]["source"] == "config.external_runner.python_executable"


def test_rose2_worker_subprocess_env_strips_isaac_pythonpath(monkeypatch):
    monkeypatch.setenv(
        "PYTHONPATH",
        "/home/echo/isaac-sim-standalone-5.1.0-linux-x86_64/exts/omni.isaac.ml_archive/pip_prebundle:/tmp/other",
    )
    monkeypatch.setenv("PYTHONHOME", "/tmp/poison")

    check_env = rose2_runner._rose2_worker_subprocess_env(repo_root=None)
    run_env = rose2_runner._rose2_worker_subprocess_env(repo_root="/home/echo/sgnav")

    assert "PYTHONPATH" not in check_env
    assert "PYTHONHOME" not in check_env
    assert run_env["PYTHONPATH"] == "/home/echo/sgnav"
    assert "PYTHONHOME" not in run_env
    assert "isaac-sim-standalone" not in run_env["PYTHONPATH"]


def test_worker_patches_upstream_y1_short_circuit(tmp_path):
    code_dir = tmp_path / "code"
    rose_dir = code_dir / "rose_v1_repo"
    rose_dir.mkdir(parents=True)
    source = rose_dir / "fft_structure_extraction.py"
    source.write_text(
        "if np.abs(Y1) > 3 * np.max(self.binary_map.shape) or b == 0:\n"
        "    pass\n"
        "if np.abs(Y1) > 3 * np.max(self.binary_map.shape) or b == 0:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (code_dir / "minibatch.py").write_text(
        "while True:\n"
        "    if param_obj.filter_level <= 0.12:\n"
        "        break\n",
        encoding="utf-8",
    )
    payload = {"compat_patches": []}

    patch_dir = _install_source_file_compat_patches(code_dir, tmp_path / "work", payload)

    assert patch_dir is not None
    patched = (patch_dir / "fft_structure_extraction.py").read_text(encoding="utf-8")
    assert "if b == 0 or np.abs(Y1) > 3 * np.max(self.binary_map.shape):" in patched
    assert "if np.abs(Y1) > 3 * np.max(self.binary_map.shape) or b == 0:" not in patched
    patched_minibatch = (patch_dir / "minibatch.py").read_text(encoding="utf-8")
    assert "param_obj.filter_level >= getattr(param_obj, 'max_filter_level', 0.50)" in patched_minibatch
    assert "fft_structure_extraction_y1_short_circuit" in payload["compat_patches"]
    assert "minibatch_filter_level_upper_bound" in payload["compat_patches"]


def test_worker_prefers_nonblank_raw_rooms_over_blank_post_overlay(tmp_path):
    blank = np.full((12, 16, 3), 255, dtype=np.uint8)
    raw = blank.copy()
    raw[2:10, 2:8] = (80, 140, 255)
    raw[2:10, 9:14] = (255, 160, 80)
    overlay = blank.copy()
    overlay[4:8, 4:6] = (255, 160, 80)
    Image.fromarray(blank).save(tmp_path / "8b_rooms_th1_on_map_post.png")
    Image.fromarray(raw).save(tmp_path / "8b_rooms_th1.png")
    Image.fromarray(overlay).save(tmp_path / "8b_rooms_th1_on_map_th10.png")

    assert score_room_output_candidate(tmp_path / "8b_rooms_th1_on_map_post.png") == 0
    assert score_room_output_candidate(tmp_path / "8b_rooms_th1.png") > score_room_output_candidate(
        tmp_path / "8b_rooms_th1_on_map_th10.png"
    )
    assert find_rose2_room_output(tmp_path).name == "8b_rooms_th1.png"


def _map_info(shape: tuple[int, int]) -> MapInfo:
    h, w = shape
    return MapInfo(resolution_m=0.10, min_x=0.0, max_x=w * 0.10, min_y=0.0, max_y=h * 0.10, width=w, height=h)


def _fake_declutter_source(tmp_path: Path) -> Path:
    root = tmp_path / "declutter-reconstruct"
    code = root / "code"
    util = code / "util"
    util.mkdir(parents=True)
    (util / "layout.py").write_text("def external_contour(img_rgb):\n    return [], [[0,0],[1,0],[1,1]]\n", encoding="utf-8")
    (util / "postprocessing.py").write_text("# fake postprocessing\n", encoding="utf-8")
    (code / "parameters.py").write_text(
        """
class PathObj:
    pass

class ParameterObj:
    def __init__(self):
        self.filter_level = 0.18
        self.thresholdHough = 25
        self.minLineLength = 10
        self.maxLineGap = 5
        self.th_post = 750
        self.th1 = 0.1
        self.distance_extended_segment = 20
""",
        encoding="utf-8",
    )
    (code / "FFT_MQ.py").write_text(
        """
from pathlib import Path
import shutil

def main(metric_map_path, path_orebro, filter_level, param_obj):
    out = Path(path_orebro)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metric_map_path, out / ('OREBRO_%s.png' % str(filter_level)))
""",
        encoding="utf-8",
    )
    (code / "minibatch.py").write_text(
        """
from pathlib import Path
import numpy as np
from PIL import Image

def start_main(par, param_obj, path_obj):
    metric = np.asarray(Image.open(path_obj.metric_map_path).convert('L'))
    free = metric > 200
    h, w = free.shape
    cc = np.indices(free.shape)[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[free & (cc < w // 2)] = (80, 140, 255)
    out[free & (cc > w // 2)] = (255, 160, 80)
    path = Path(path_obj.filepath) / '8b_rooms_th1.png'
    Image.fromarray(out).save(path)
    return str(path), [(80, 140, 255), (255, 160, 80)]
""",
        encoding="utf-8",
    )
    return root
