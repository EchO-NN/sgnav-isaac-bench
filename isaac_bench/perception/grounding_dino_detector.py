from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
from PIL import Image

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.detector_base import DetectorBase


DEFAULT_GROUNDING_DINO_REPO_ID = "ShilongLiu/GroundingDINO"
DEFAULT_GROUNDING_DINO_CHECKPOINT = "data/models/groundingdino_swinb_cogcoor.pth"
DEFAULT_GROUNDING_DINO_CONFIG_CANDIDATES = (
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py",
    "groundingdino/config/GroundingDINO_SwinB.py",
)
GROUNDING_DINO_FIXED_VOCABULARY = ("door", "doorframe")


class GroundingDINODetector(DetectorBase):
    """GroundingDINO-B/Swin-B open-vocabulary detector."""

    backend_name = "grounding_dino"
    variant = "GroundingDINO-B/Swin-B"

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_GROUNDING_DINO_CHECKPOINT,
        *,
        config_path: str | None = None,
        conf: float = 0.45,
        iou: float = 0.5,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ) -> None:
        try:
            from groundingdino.datasets import transforms as _transforms
            from groundingdino.util.inference import load_model as _load_model
            from groundingdino.util.inference import predict as _predict
            from torchvision.ops import box_convert as _box_convert
        except Exception as exc:
            raise RuntimeError(
                "groundingdino, torchvision, and their dependencies are required for GroundingDINO-B/Swin-B. "
                "Install/use the SG-Nav env that contains GroundingDINO."
            ) from exc

        self.checkpoint_path = str(Path(str(checkpoint_path)).expanduser())
        self.config_path = str(resolve_grounding_dino_config(config_path))
        _require_existing_file("GroundingDINO-B/Swin-B checkpoint", self.checkpoint_path)
        _require_existing_file("GroundingDINO-B/Swin-B config", self.config_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.text_threshold = float(text_threshold)
        self.device = str(device or "cuda")
        self.vocab: List[str] = []
        self._transforms = _transforms
        self._predict = _predict
        self._box_convert = _box_convert
        self.model = _load_model(self.config_path, self.checkpoint_path, device=self.device).to(self.device)

    def set_vocabulary(self, categories: List[str]) -> None:
        self.vocab = grounding_dino_vocabulary(categories)

    def detect(self, rgb: Any) -> List[Detection2D]:
        if not self.vocab:
            return []
        image = _rgb_to_numpy_uint8(rgb)
        tensor = self._preprocess_rgb(image)
        caption = grounding_dino_caption(self.vocab)
        boxes, logits, phrases = self._predict(
            model=self.model,
            image=tensor,
            caption=caption,
            box_threshold=self.conf,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        if boxes is None or len(boxes) == 0:
            return []
        source_h, source_w = int(image.shape[0]), int(image.shape[1])
        scale = _torch_tensor([source_w, source_h, source_w, source_h])
        xyxy = self._box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        scores = logits.detach().cpu().numpy().astype(np.float32)
        keep = _nms_numpy(xyxy, scores, self.iou)
        out: list[Detection2D] = []
        for idx in keep:
            phrase = str(phrases[int(idx)] if int(idx) < len(phrases) else "")
            class_id, label = _match_phrase_to_vocab(phrase, self.vocab)
            score = float(scores[int(idx)])
            box = np.asarray(xyxy[int(idx)], dtype=np.float32)
            x1 = float(np.clip(box[0], 0, max(0, source_w - 1)))
            y1 = float(np.clip(box[1], 0, max(0, source_h - 1)))
            x2 = float(np.clip(box[2], 0, max(0, source_w - 1)))
            y2 = float(np.clip(box[3], 0, max(0, source_h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                Detection2D(
                    category=normalize_category(label),
                    raw_label=phrase or label,
                    confidence=score,
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_id=class_id,
                )
            )
        return out

    def _preprocess_rgb(self, image: np.ndarray):
        transform = self._transforms.Compose(
            [
                self._transforms.RandomResize([800], max_size=1333),
                self._transforms.ToTensor(),
                self._transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        pil = Image.fromarray(image[:, :, :3], mode="RGB")
        tensor, _ = transform(pil, None)
        return tensor


def grounding_dino_caption(categories: Sequence[str]) -> str:
    tokens = []
    seen: set[str] = set()
    for category in categories:
        normalized = normalize_category(category)
        if not normalized or normalized == "unknown" or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(_caption_token_for_category(normalized))
    return ". ".join(tokens) + "."


def grounding_dino_vocabulary(categories: Sequence[str]) -> list[str]:
    """Normalize a GroundingDINO vocabulary and always include structural door cues."""

    seen: set[str] = set()
    vocab: list[str] = []
    for category in list(categories or []) + list(GROUNDING_DINO_FIXED_VOCABULARY):
        normalized = normalize_category(category)
        if normalized in seen or normalized == "unknown":
            continue
        seen.add(normalized)
        vocab.append(normalized)
    return vocab


def _caption_token_for_category(category: str) -> str:
    normalized = normalize_category(category)
    if normalized == "doorframe":
        return "door frame"
    return normalized.replace("_", " ")


def resolve_grounding_dino_config(config_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path).expanduser())
    env_config = os.environ.get("GROUNDING_DINO_CONFIG")
    if env_config:
        candidates.append(Path(env_config).expanduser())
    root_values = [
        os.environ.get("GROUNDING_DINO_ROOT"),
        os.environ.get("SGNAV_GROUNDING_DINO_ROOT"),
        "/home/echo/SG-Nav",
        "/home/joey/SG-Nav",
    ]
    for root in root_values:
        if not root:
            continue
        root_path = Path(root).expanduser()
        for rel in DEFAULT_GROUNDING_DINO_CONFIG_CANDIDATES:
            candidates.append(root_path / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Missing GroundingDINO-B/Swin-B config. Set GROUNDING_DINO_CONFIG or GROUNDING_DINO_ROOT. Tried: %s" % tried
    )


def _require_existing_file(kind: str, path: str) -> None:
    if not path or not Path(path).expanduser().exists():
        raise FileNotFoundError("Missing %s: %s" % (kind, path))


def _rgb_to_numpy_uint8(rgb: Any) -> np.ndarray:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(rgb, torch.Tensor):
        tensor = rgb.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] >= 3 and tensor.shape[-1] != 3:
            tensor = tensor[:3].permute(1, 2, 0)
        arr = tensor.numpy()
    else:
        arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("GroundingDINO expects an RGB image with shape HxWx3")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if float(np.nanmax(arr)) <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _torch_tensor(values: Sequence[float]):
    import torch

    return torch.tensor(list(values), dtype=torch.float32)


def _match_phrase_to_vocab(phrase: str, vocab: Sequence[str]) -> tuple[int | None, str]:
    normalized_phrase = normalize_category(phrase)
    normalized_words = {normalize_category(part) for part in str(phrase).replace(".", " ").split()}
    for idx, category in enumerate(vocab):
        normalized_category = normalize_category(category)
        compact_phrase = normalized_phrase.replace("_", "")
        compact_category = normalized_category.replace("_", "")
        if normalized_phrase == normalized_category or compact_phrase == compact_category:
            return int(idx), normalized_category
    for idx, category in enumerate(vocab):
        normalized_category = normalize_category(category)
        if normalized_category in normalized_words:
            return int(idx), normalized_category
        if normalized_category in normalized_phrase or normalized_phrase in normalized_category:
            return int(idx), normalized_category
    return None, normalized_phrase if normalized_phrase != "unknown" else (vocab[0] if vocab else "unknown")


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, np.maximum(union, 1e-9))
        order = order[1:][iou <= float(iou_threshold)]
    return keep
