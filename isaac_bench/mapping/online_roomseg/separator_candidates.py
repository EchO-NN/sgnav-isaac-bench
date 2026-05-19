from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .utils import rasterize_line
from .wall_lines import WallSegment


@dataclass
class PhysicalWallCompletionConfig:
    enabled: bool = True
    max_gap_m: float = 0.45
    max_lateral_offset_m: float = 0.15
    max_angle_deg: float = 10.0
    min_endpoint_support: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "PhysicalWallCompletionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorwayVirtualCutConfig:
    enabled: bool = True
    min_width_m: float = 0.45
    max_width_m: float = 1.60
    max_angle_deg: float = 12.0
    min_wall_endpoint_support: float = 0.4
    require_free_gap: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorwayVirtualCutConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SeparatorCandidate:
    candidate_id: int
    kind: str
    p0_rc: np.ndarray
    p1_rc: np.ndarray
    theta: float
    length_m: float
    confidence: float
    source_segment_ids: list[int]
    wall_support_score: float = 0.0
    free_gap_score: float = 0.0
    doorway_score: float = 0.0
    corridor_preservation_score: float = 0.0
    topology_gain_score: float = 0.0
    fragmentation_penalty: float = 0.0
    corridor_split_penalty: float = 0.0
    accepted: bool = False
    reject_reason: str = ""
    debug: dict = field(default_factory=dict)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        return rasterize_line(self.p0_rc, self.p1_rc, shape)

    def to_dict(self) -> dict:
        return {
            "candidate_id": int(self.candidate_id),
            "kind": str(self.kind),
            "p0_rc": [int(round(float(v))) for v in self.p0_rc.tolist()],
            "p1_rc": [int(round(float(v))) for v in self.p1_rc.tolist()],
            "theta": float(self.theta),
            "length_m": float(self.length_m),
            "confidence": float(self.confidence),
            "source_segment_ids": [int(v) for v in self.source_segment_ids],
            "wall_support_score": float(self.wall_support_score),
            "free_gap_score": float(self.free_gap_score),
            "doorway_score": float(self.doorway_score),
            "corridor_preservation_score": float(self.corridor_preservation_score),
            "topology_gain_score": float(self.topology_gain_score),
            "fragmentation_penalty": float(self.fragmentation_penalty),
            "corridor_split_penalty": float(self.corridor_split_penalty),
            "accepted": bool(self.accepted),
            "reject_reason": str(self.reject_reason),
            **_jsonable(self.debug),
        }


def generate_wall_gap_candidates(
    segments: Sequence[WallSegment],
    *,
    free_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    physical_config: PhysicalWallCompletionConfig | Mapping[str, object] | None = None,
    doorway_config: DoorwayVirtualCutConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    physical = physical_config if isinstance(physical_config, PhysicalWallCompletionConfig) else PhysicalWallCompletionConfig.from_mapping(physical_config)
    doorway = doorway_config if isinstance(doorway_config, DoorwayVirtualCutConfig) else DoorwayVirtualCutConfig.from_mapping(doorway_config)
    candidates: list[SeparatorCandidate] = []
    sid = int(start_id)
    axis_segments = [_axis_segment(seg) for seg in segments]
    for idx, a in enumerate(axis_segments):
        if a is None:
            continue
        for b in axis_segments[idx + 1 :]:
            if b is None or a["axis"] != b["axis"]:
                continue
            if abs(int(a["line"]) - int(b["line"])) > max(0, int(round(float(physical.max_lateral_offset_m) / max(float(resolution_m), 1e-9)))):
                continue
            gap_start = min(int(a["end"]), int(b["end"])) + 1
            gap_end = max(int(a["start"]), int(b["start"])) - 1
            if gap_end < gap_start:
                continue
            length_cells = int(gap_end - gap_start + 1)
            length_m = float(length_cells * float(resolution_m))
            if a["axis"] == "horizontal":
                p0 = np.asarray([int(round((int(a["line"]) + int(b["line"])) / 2)), gap_start], dtype=np.float32)
                p1 = np.asarray([int(round((int(a["line"]) + int(b["line"])) / 2)), gap_end], dtype=np.float32)
                theta = 0.0
            else:
                p0 = np.asarray([gap_start, int(round((int(a["line"]) + int(b["line"])) / 2))], dtype=np.float32)
                p1 = np.asarray([gap_end, int(round((int(a["line"]) + int(b["line"])) / 2))], dtype=np.float32)
                theta = float(np.pi / 2.0)
            line = rasterize_line(p0, p1, free_clean.shape)
            line_cells = max(1, int(np.count_nonzero(line)))
            free_ratio = float(np.count_nonzero(line & free_clean)) / float(line_cells)
            unknown_ratio = float(np.count_nonzero(line & unknown_clean)) / float(line_cells)
            source_ids = [int(a["segment_id"]), int(b["segment_id"])]
            if bool(physical.enabled) and length_m <= float(physical.max_gap_m):
                candidates.append(
                    SeparatorCandidate(
                        candidate_id=sid,
                        kind="physical_wall_completion",
                        p0_rc=p0,
                        p1_rc=p1,
                        theta=theta,
                        length_m=length_m,
                        confidence=float(np.clip(0.7 + 0.3 * unknown_ratio, 0.0, 1.0)),
                        source_segment_ids=source_ids,
                        wall_support_score=1.0,
                        free_gap_score=free_ratio,
                        debug={"unknown_ratio": unknown_ratio, "free_ratio": free_ratio},
                    )
                )
                sid += 1
            if (
                bool(doorway.enabled)
                and float(doorway.min_width_m) <= length_m <= float(doorway.max_width_m)
                and (not bool(doorway.require_free_gap) or free_ratio >= 0.55)
            ):
                candidates.append(
                    SeparatorCandidate(
                        candidate_id=sid,
                        kind="doorway_virtual_cut",
                        p0_rc=p0,
                        p1_rc=p1,
                        theta=theta,
                        length_m=length_m,
                        confidence=float(np.clip(0.5 + 0.5 * free_ratio, 0.0, 1.0)),
                        source_segment_ids=source_ids,
                        wall_support_score=1.0,
                        free_gap_score=free_ratio,
                        doorway_score=free_ratio,
                        debug={"unknown_ratio": unknown_ratio, "free_ratio": free_ratio},
                    )
                )
                sid += 1
    return candidates, {
        "physical_wall_completion_enabled": bool(physical.enabled),
        "doorway_virtual_cut_enabled": bool(doorway.enabled),
        "wall_gap_candidate_count": int(len(candidates)),
    }


def _axis_segment(segment: WallSegment) -> dict | None:
    p0 = np.rint(segment.p0_rc).astype(np.int32)
    p1 = np.rint(segment.p1_rc).astype(np.int32)
    if abs(int(p0[0]) - int(p1[0])) <= 1:
        line = int(round(float((int(p0[0]) + int(p1[0])) / 2.0)))
        start, end = sorted((int(p0[1]), int(p1[1])))
        return {"axis": "horizontal", "line": line, "start": start, "end": end, "segment_id": int(segment.segment_id)}
    if abs(int(p0[1]) - int(p1[1])) <= 1:
        line = int(round(float((int(p0[1]) + int(p1[1])) / 2.0)))
        start, end = sorted((int(p0[0]), int(p1[0])))
        return {"axis": "vertical", "line": line, "start": start, "end": end, "segment_id": int(segment.segment_id)}
    return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
