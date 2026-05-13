from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.detector_base import DetectorBase, DryRunDetector


class YOLOWorldDetector(DetectorBase):
    def __init__(self, model_name: str = "data/models/yolov8s-worldv2.pt", conf: float = 0.08, iou: float = 0.5):
        try:
            from ultralytics import YOLOWorld
        except Exception as exc:
            raise RuntimeError("ultralytics is required for YOLO-World. Install it in the controller env.") from exc
        self.model_name = str(model_name)
        model_path = Path(self.model_name)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        load_name = self.model_name if model_path.exists() else model_path.name
        self.model = YOLOWorld(load_name)
        self.conf = float(conf)
        self.iou = float(iou)
        self.vocab: List[str] = []

    def set_vocabulary(self, categories: List[str]) -> None:
        self.vocab = [normalize_category(cat) for cat in categories]
        self.model.set_classes(self.vocab)

    def _warp_or_tensor_to_torch(self, rgb: Any):
        try:
            import torch
        except Exception:
            return None

        if isinstance(rgb, torch.Tensor):
            tensor = rgb
        else:
            tensor = None
            try:
                import warp as wp

                if isinstance(rgb, wp.array):
                    tensor = wp.to_torch(rgb)
            except Exception:
                tensor = None
            if tensor is None:
                return None

        if tensor.ndim == 3:
            if tensor.shape[-1] >= 3:
                tensor = tensor[..., :3]
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            elif tensor.shape[0] >= 3:
                tensor = tensor[:3].unsqueeze(0)
            else:
                return None
        elif tensor.ndim == 4:
            if tensor.shape[-1] >= 3:
                tensor = tensor[..., :3].permute(0, 3, 1, 2)
            elif tensor.shape[1] >= 3:
                tensor = tensor[:, :3]
            else:
                return None
        else:
            return None

        tensor = tensor.contiguous()
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        elif not tensor.is_floating_point():
            tensor = tensor.float()
        return tensor

    def _predict(self, rgb: Any):
        tensor = self._warp_or_tensor_to_torch(rgb)
        if tensor is not None:
            kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False}
            if str(tensor.device).startswith("cuda"):
                kwargs["device"] = str(tensor.device)
            return self.model.predict(tensor, **kwargs)
        return self.model.predict(rgb, conf=self.conf, iou=self.iou, verbose=False)

    def detect(self, rgb: Any) -> List[Detection2D]:
        results = self._predict(rgb)
        out: List[Detection2D] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                label = self.vocab[cls_id] if 0 <= cls_id < len(self.vocab) else str(cls_id)
                out.append(
                    Detection2D(
                        category=normalize_category(label),
                        raw_label=label,
                        confidence=float(box.conf.item()) if box.conf is not None else 0.0,
                        bbox_xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                        class_id=cls_id,
                    )
                )
        return out


def build_detector(name: str, model: str, conf: float = 0.08, iou: float = 0.5) -> DetectorBase:
    if name in ("dry_run", "none"):
        return DryRunDetector()
    if name == "yolo_world":
        return YOLOWorldDetector(model, conf=conf, iou=iou)
    raise ValueError("Unsupported detector: %s" % name)
