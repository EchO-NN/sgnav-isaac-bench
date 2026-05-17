from __future__ import annotations

import contextlib
from pathlib import Path
from typing import List, Optional

import numpy as np

from isaac_bench.perception.detection_types import MIN_VALID_DETECTION_CONFIDENCE, Detection2D, detection_confidence_is_valid
from isaac_bench.perception.detector_base import DetectorBase


class SAM2BoxSegmenter:
    """Prompt SAM2 with detector boxes and attach masks to detections."""

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        device: str = "cuda",
        autocast_bfloat16: bool = True,
        min_valid_confidence: float = MIN_VALID_DETECTION_CONFIDENCE,
    ) -> None:
        try:
            import torch
            from sam2.build_sam import build_sam2
            try:
                from sam2.build_sam import build_sam2_hf
            except Exception:
                build_sam2_hf = None
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(
                "SAM2 is required for mask projection. Install facebookresearch/sam2 in the SG-Nav env."
            ) from exc

        self.torch = torch
        self.device = str(device or "cuda")
        self.autocast_bfloat16 = bool(autocast_bfloat16)
        self.min_valid_confidence = float(min_valid_confidence)
        checkpoint_path = Path(str(checkpoint)) if checkpoint else None
        if checkpoint_path is not None and str(checkpoint_path) and not checkpoint_path.exists():
            raise RuntimeError("SAM2 checkpoint not found: %s" % checkpoint_path)
        if not model_cfg:
            raise RuntimeError("SAM2 model config is required")

        if checkpoint_path is None or not str(checkpoint_path):
            if build_sam2_hf is None:
                raise RuntimeError("SAM2 checkpoint is required by this installed sam2 package")
            model = build_sam2_hf(str(model_cfg), device=self.device)
        else:
            try:
                model = build_sam2(str(model_cfg), str(checkpoint_path), device=self.device)
            except TypeError:
                model = build_sam2(str(model_cfg), str(checkpoint_path))
                if hasattr(model, "to"):
                    model = model.to(self.device)
        self.predictor = SAM2ImagePredictor(model)

    def segment(self, rgb: np.ndarray, detections: List[Detection2D]) -> List[Detection2D]:
        detections = [
            det
            for det in detections
            if detection_confidence_is_valid(float(det.confidence), self.min_valid_confidence)
        ]
        if not detections:
            return []
        image = np.asarray(rgb)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("expected RGB image with shape HxWx3")

        torch = self.torch
        use_cuda_autocast = self.device.startswith("cuda") and torch.cuda.is_available() and self.autocast_bfloat16
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda_autocast
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            self.predictor.set_image(image[:, :, :3])
            out: List[Detection2D] = []
            for det in detections:
                box = np.asarray(det.bbox_xyxy, dtype=np.float32)
                masks, scores, _logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False,
                )
                mask = _select_mask(masks, scores)
                out.append(
                    Detection2D(
                        category=det.category,
                        raw_label=det.raw_label,
                        confidence=det.confidence,
                        bbox_xyxy=det.bbox_xyxy,
                        class_id=det.class_id,
                        mask=mask,
                    )
                )
        return out


class SegmentingDetector(DetectorBase):
    def __init__(self, detector: DetectorBase, segmenter: SAM2BoxSegmenter):
        self.detector = detector
        self.segmenter = segmenter

    def set_vocabulary(self, categories: List[str]) -> None:
        self.detector.set_vocabulary(categories)

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        return self.segmenter.segment(rgb, self.detector.detect(rgb))


def _select_mask(masks, scores) -> Optional[np.ndarray]:
    arr = np.asarray(masks)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr.astype(bool)
    score_arr = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((arr.shape[0],), dtype=np.float32)
    idx = int(np.argmax(score_arr)) if len(score_arr) else 0
    return np.asarray(arr[idx]).astype(bool)


def build_sam2_segmenter(
    mode: str,
    checkpoint: str,
    model_cfg: str,
    device: str = "cuda",
    required: bool = False,
) -> Optional[SAM2BoxSegmenter]:
    mode_norm = str(mode or "none").strip().lower()
    if mode_norm in {"none", "false", "0", ""}:
        return None
    if mode_norm not in {"sam2", "auto"}:
        raise ValueError("Unsupported segmenter: %s" % mode)
    try:
        return SAM2BoxSegmenter(checkpoint=checkpoint, model_cfg=model_cfg, device=device)
    except Exception as exc:
        if required or mode_norm == "sam2":
            raise
        print("[sam2] unavailable; continuing without masks: %s" % exc, flush=True)
        return None
