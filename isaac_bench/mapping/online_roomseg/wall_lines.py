from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .utils import component_metrics, label_components, rasterize_line


@dataclass
class LineWallsConfig:
    enabled: bool = True
    hough_enabled: bool = True
    pca_enabled: bool = True
    min_line_length_m: float = 0.45
    min_support_ratio: float = 0.35
    max_angle_to_dominant_deg: float = 15.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "LineWallsConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class WallSegment:
    segment_id: int
    p0_rc: np.ndarray
    p1_rc: np.ndarray
    theta: float
    length_m: float
    support_ratio: float
    mean_wall_score: float
    source: str

    def to_dict(self) -> dict:
        return {
            "segment_id": int(self.segment_id),
            "p0_rc": [int(round(float(v))) for v in self.p0_rc.tolist()],
            "p1_rc": [int(round(float(v))) for v in self.p1_rc.tolist()],
            "theta": float(self.theta),
            "length_m": float(self.length_m),
            "support_ratio": float(self.support_ratio),
            "mean_wall_score": float(self.mean_wall_score),
            "source": str(self.source),
        }


def extract_line_supported_walls(
    wall_candidate_clean: np.ndarray,
    *,
    resolution_m: float,
    config: LineWallsConfig | Mapping[str, object] | None = None,
) -> tuple[list[WallSegment], dict]:
    cfg = config if isinstance(config, LineWallsConfig) else LineWallsConfig.from_mapping(config)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    if not bool(cfg.enabled) or not np.any(wall):
        return [], {"enabled": bool(cfg.enabled), "segments": [], "dominant_directions": []}
    segments: list[WallSegment] = []
    next_id = 1
    if bool(cfg.hough_enabled):
        hough = _axis_run_segments(wall, float(resolution_m), cfg, next_id)
        segments.extend(hough)
        next_id += len(hough)
    if bool(cfg.pca_enabled):
        pca = _pca_component_segments(wall, float(resolution_m), cfg, next_id)
        segments.extend(pca)
    segments = _dedupe_segments(segments)
    dominant = _dominant_directions(segments)
    debug = {
        "enabled": True,
        "hough_enabled": bool(cfg.hough_enabled),
        "pca_enabled": bool(cfg.pca_enabled),
        "segment_count": int(len(segments)),
        "segments": [segment.to_dict() for segment in segments[:512]],
        "dominant_directions": dominant,
    }
    return segments, debug


def line_supported_wall_mask(segments: Sequence[WallSegment], shape: tuple[int, int], radius_cells: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for segment in segments:
        out |= rasterize_line(segment.p0_rc, segment.p1_rc, shape)
    if int(radius_cells) > 0:
        from .utils import dilate

        out = dilate(out, int(radius_cells))
    return out.astype(bool)


def _axis_run_segments(wall: np.ndarray, resolution: float, cfg: LineWallsConfig, start_id: int) -> list[WallSegment]:
    min_cells = max(2, int(round(float(cfg.min_line_length_m) / max(float(resolution), 1e-9))))
    out: list[WallSegment] = []
    sid = int(start_id)
    h, w = wall.shape
    for row in range(h):
        for c0, c1 in _runs(wall[row, :]):
            if c1 - c0 < min_cells:
                continue
            out.append(
                WallSegment(
                    segment_id=sid,
                    p0_rc=np.asarray([row, c0], dtype=np.float32),
                    p1_rc=np.asarray([row, c1 - 1], dtype=np.float32),
                    theta=0.0,
                    length_m=float((c1 - c0) * resolution),
                    support_ratio=1.0,
                    mean_wall_score=1.0,
                    source="hough_axis_run",
                )
            )
            sid += 1
    for col in range(w):
        for r0, r1 in _runs(wall[:, col]):
            if r1 - r0 < min_cells:
                continue
            out.append(
                WallSegment(
                    segment_id=sid,
                    p0_rc=np.asarray([r0, col], dtype=np.float32),
                    p1_rc=np.asarray([r1 - 1, col], dtype=np.float32),
                    theta=float(np.pi / 2.0),
                    length_m=float((r1 - r0) * resolution),
                    support_ratio=1.0,
                    mean_wall_score=1.0,
                    source="hough_axis_run",
                )
            )
            sid += 1
    return out


def _pca_component_segments(wall: np.ndarray, resolution: float, cfg: LineWallsConfig, start_id: int) -> list[WallSegment]:
    labels, count = label_components(wall, 8)
    out: list[WallSegment] = []
    sid = int(start_id)
    for idx in range(1, int(count) + 1):
        comp = labels == idx
        metrics = component_metrics(comp, resolution)
        if metrics["length_m"] < float(cfg.min_line_length_m) or metrics["elongation"] < 1.5:
            continue
        rows, cols = np.nonzero(comp)
        coords = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=1)
        center = np.mean(coords, axis=0)
        if coords.shape[0] < 2:
            continue
        cov = np.cov((coords - center).T)
        vals, vecs = np.linalg.eigh(cov)
        direction = vecs[:, int(np.argmax(vals))]
        projection = (coords - center) @ direction
        p0 = center + direction * float(np.min(projection))
        p1 = center + direction * float(np.max(projection))
        length_m = float(np.linalg.norm(p1 - p0) * resolution)
        if length_m < float(cfg.min_line_length_m):
            continue
        line = rasterize_line(p0, p1, wall.shape)
        support = float(np.count_nonzero(line & wall)) / float(max(1, np.count_nonzero(line)))
        if support < float(cfg.min_support_ratio):
            continue
        theta = float(np.arctan2(float(p1[0] - p0[0]), float(p1[1] - p0[1])))
        out.append(
            WallSegment(
                segment_id=sid,
                p0_rc=p0.astype(np.float32),
                p1_rc=p1.astype(np.float32),
                theta=theta,
                length_m=length_m,
                support_ratio=support,
                mean_wall_score=support,
                source="pca_component",
            )
        )
        sid += 1
    return out


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(values, dtype=bool)
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr.tolist() + [False]):
        if value and start is None:
            start = int(idx)
        elif not value and start is not None:
            out.append((int(start), int(idx)))
            start = None
    return out


def _dedupe_segments(segments: Sequence[WallSegment]) -> list[WallSegment]:
    out: list[WallSegment] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for segment in segments:
        p0 = tuple(int(round(float(v))) for v in segment.p0_rc)
        p1 = tuple(int(round(float(v))) for v in segment.p1_rc)
        key = (*min(p0, p1), *max(p0, p1), str(segment.source))
        if key in seen:
            continue
        seen.add(key)
        out.append(segment)
    for idx, segment in enumerate(out, start=1):
        segment.segment_id = int(idx)
    return out


def _dominant_directions(segments: Sequence[WallSegment]) -> list[dict]:
    buckets = {"horizontal": 0.0, "vertical": 0.0, "other": 0.0}
    for segment in segments:
        theta = abs(float(segment.theta)) % float(np.pi)
        if theta < np.pi / 6 or theta > 5 * np.pi / 6:
            buckets["horizontal"] += float(segment.length_m)
        elif abs(theta - np.pi / 2) < np.pi / 6:
            buckets["vertical"] += float(segment.length_m)
        else:
            buckets["other"] += float(segment.length_m)
    total = max(1e-9, sum(buckets.values()))
    return [
        {"name": name, "support_m": float(value), "confidence": float(value / total)}
        for name, value in sorted(buckets.items(), key=lambda item: -item[1])
        if value > 0
    ]
