from __future__ import annotations

import base64
import io
import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from isaac_bench.config import repo_root
from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.detector_base import DetectorBase


def detection_to_payload(det: Detection2D) -> Dict[str, Any]:
    payload = {
        "category": str(det.category),
        "raw_label": str(det.raw_label),
        "confidence": float(det.confidence),
        "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
        "class_id": None if det.class_id is None else int(det.class_id),
    }
    if det.mask is not None:
        payload["mask_rle"] = encode_mask_rle(det.mask)
    return payload


def detection_from_payload(data: Dict[str, Any]) -> Detection2D:
    bbox = data.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    mask = decode_mask_rle(data["mask_rle"]) if data.get("mask_rle") else None
    return Detection2D(
        category=normalize_category(str(data.get("category", "unknown"))),
        raw_label=str(data.get("raw_label", data.get("category", "unknown"))),
        confidence=float(data.get("confidence", 0.0)),
        bbox_xyxy=tuple(float(v) for v in bbox[:4]),
        class_id=None if data.get("class_id") is None else int(data["class_id"]),
        mask=mask,
    )


def encode_mask_rle(mask: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2:
        raise ValueError("expected a 2D boolean mask")
    packed = np.packbits(arr.reshape(-1).astype(np.uint8))
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "data_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def decode_mask_rle(payload: Dict[str, Any]) -> np.ndarray:
    shape = payload.get("shape", [0, 0])
    height, width = int(shape[0]), int(shape[1])
    raw = base64.b64decode(str(payload.get("data_b64", "")).encode("ascii"))
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=height * width)
    return unpacked.reshape((height, width)).astype(bool)


def encode_rgb_png_b64(rgb: np.ndarray) -> str:
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("expected an RGB image with shape HxWx3")
    image = Image.fromarray(arr[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_rgb_png_b64(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def decode_rgb_image_path(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


class SubprocessDetector(DetectorBase):
    """Detector proxy that keeps heavyweight perception outside Isaac Python."""

    def __init__(
        self,
        detector_name: str,
        model_name: str,
        conf: float = 0.08,
        iou: float = 0.5,
        python_executable: Optional[str] = None,
        startup_timeout_s: float = 180.0,
        response_timeout_s: float = 180.0,
        ipc_jpeg_quality: int = 85,
        segmenter: str = "none",
        sam2_checkpoint: str = "",
        sam2_model_cfg: str = "",
        sam2_device: str = "cuda",
    ) -> None:
        self.detector_name = str(detector_name)
        self.model_name = str(model_name)
        self.conf = float(conf)
        self.iou = float(iou)
        self.python_executable = python_executable or self._default_python_executable()
        self.startup_timeout_s = float(startup_timeout_s)
        self.response_timeout_s = float(response_timeout_s)
        self.ipc_jpeg_quality = max(30, min(95, int(ipc_jpeg_quality)))
        self.segmenter = str(segmenter or "none")
        self.sam2_checkpoint = str(sam2_checkpoint or "")
        self.sam2_model_cfg = str(sam2_model_cfg or "")
        self.sam2_device = str(sam2_device or "cuda")
        self._ipc_dir = Path(tempfile.gettempdir()) / ("sgnav_detector_%d" % os.getpid())
        self._frame_path = self._ipc_dir / "latest.jpg"
        self.vocab: List[str] = []
        self._proc: Optional[subprocess.Popen[str]] = None

    @staticmethod
    def _default_python_executable() -> str:
        explicit = os.environ.get("SG_NAV_PYTHON")
        if explicit:
            return explicit
        env_root = Path(os.environ.get("SG_NAV_ENV", "/home/echo/SG-Nav/.mamba/envs/sg-nav"))
        candidate = env_root / "bin" / "python"
        if candidate.exists():
            return str(candidate)
        return sys.executable

    def set_vocabulary(self, categories: List[str]) -> None:
        self.vocab = [normalize_category(cat) for cat in categories]
        self._ensure_started()
        self._send({"type": "set_vocabulary", "categories": self.vocab})
        self._read_response("ok", timeout_s=self.response_timeout_s)

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        self._ensure_started()
        self._send({"type": "detect_path", "path": self._write_rgb_jpeg(rgb)})
        response = self._read_response("detections", timeout_s=self.response_timeout_s)
        return [detection_from_payload(item) for item in response.get("detections", [])]

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        root = repo_root()
        cmd = [
            self.python_executable,
            "-m",
            "isaac_bench.perception.detector_ipc_worker",
            "--detector",
            self.detector_name,
            "--model",
            self.model_name,
            "--conf",
            str(self.conf),
            "--iou",
            str(self.iou),
            "--categories-json",
            json.dumps(self.vocab, ensure_ascii=False),
            "--segmenter",
            self.segmenter,
            "--sam2-checkpoint",
            self.sam2_checkpoint,
            "--sam2-model-cfg",
            self.sam2_model_cfg,
            "--sam2-device",
            self.sam2_device,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["PYTHONUNBUFFERED"] = "1"
        if os.environ.get("ISAAC_BENCH_KEEP_DETECTOR_LD_LIBRARY_PATH", "").lower() not in {"1", "true", "yes"}:
            env.pop("LD_LIBRARY_PATH", None)
        print(
            "[detector-ipc] starting %s worker with %s" % (self.detector_name, self.python_executable),
            file=sys.stderr,
            flush=True,
        )
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._read_response("ready", timeout_s=self.startup_timeout_s)

    def _write_rgb_jpeg(self, rgb: np.ndarray) -> str:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("expected an RGB image with shape HxWx3")
        self._ipc_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._ipc_dir / "latest.tmp.jpg"
        Image.fromarray(arr[:, :, :3], mode="RGB").save(tmp_path, format="JPEG", quality=self.ipc_jpeg_quality)
        os.replace(tmp_path, self._frame_path)
        return str(self._frame_path)

    def _send(self, payload: Dict[str, Any]) -> None:
        self._ensure_process_alive()
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _read_response(self, expected_type: str, timeout_s: float) -> Dict[str, Any]:
        self._ensure_process_alive()
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            ready, _, _ = select.select([stdout], [], [], max(0.0, float(timeout_s)))
            if not ready:
                raise RuntimeError("Timed out waiting for detector IPC response: %s" % expected_type)
            line = stdout.readline()
            if not line:
                code = self._proc.poll()
                raise RuntimeError("Detector IPC worker exited before %s response (code=%s)" % (expected_type, code))
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                print("[detector-ipc] non-json stdout from worker: %s" % line.rstrip(), file=sys.stderr, flush=True)
                continue
            if response.get("type") == "error":
                raise RuntimeError("Detector IPC worker error: %s" % response.get("message", "unknown error"))
            if response.get("type") != expected_type:
                raise RuntimeError(
                    "Detector IPC protocol error: expected %s, got %s" % (expected_type, response.get("type"))
                )
            return response

    def _ensure_process_alive(self) -> None:
        if self._proc is None:
            raise RuntimeError("Detector IPC worker has not been started")
        code = self._proc.poll()
        if code is not None:
            raise RuntimeError("Detector IPC worker exited unexpectedly with code %s" % code)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
