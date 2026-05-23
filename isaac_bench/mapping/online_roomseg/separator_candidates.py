from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .utils import dilate, rasterize_line
from .wall_lines import FilteredWallLine, SnappedWallRun, WallSegment, snap_wall_segments_to_runs


@dataclass
class PhysicalWallCompletionConfig:
    enabled: bool = True
    max_gap_m: float = 0.45
    max_lateral_offset_m: float = 0.20
    max_angle_deg: float = 10.0
    min_endpoint_support: float = 0.35
    min_confidence: float = 0.55

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "PhysicalWallCompletionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class MissedScanGapClosureConfig:
    enabled: bool = True
    max_gap_m: float = 0.45
    max_unknown_ratio: float = 0.70
    max_free_ratio: float = 0.65
    min_wall_endpoint_support: float = 0.25

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "MissedScanGapClosureConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class ShortUnknownGapClosureConfig:
    enabled: bool = True
    max_gap_m: float = 0.60
    min_unknown_ratio: float = 0.45
    min_wall_endpoint_support: float = 0.25

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "ShortUnknownGapClosureConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class NoiseWallGapFillConfig:
    enabled: bool = True
    max_gap_m: float = 0.40
    max_lateral_offset_m: float = 0.20
    min_endpoint_support: float = 0.25
    thickness_cells: int = 0
    corner_gap_enabled: bool = True
    corner_thickness_cells: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "NoiseWallGapFillConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorwayVirtualCutConfig:
    enabled: bool = True
    min_width_m: float = 0.45
    max_width_m: float = 1.80
    preferred_min_width_m: float = 0.55
    preferred_max_width_m: float = 1.20
    max_angle_deg: float = 12.0
    min_wall_endpoint_support: float = 0.30
    require_free_gap: bool = True
    free_gap_min_ratio: float = 0.55

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorwayVirtualCutConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SingleSidedWallExtensionConfig:
    enabled: bool = True
    max_extension_m: float = 1.20
    max_open_free_ratio: float = 0.70
    require_anchor: bool = True
    reject_if_endpoint_none: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "SingleSidedWallExtensionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class SeparatorAnchorConfig:
    thickness_cells: int = 1
    doorway_thickness_cells: int = 1
    wall_completion_thickness_cells: int = 1
    corridor_neck_thickness_cells: int = 1
    max_anchor_extension_m: float = 0.35
    require_two_anchors: bool = True
    accept_anchor_score_min: float = 0.75
    require_two_anchors_for_wall_completion: bool = True
    require_two_anchors_for_doorway: bool = True
    allow_one_strong_one_unknown_frontier: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "SeparatorAnchorConfig":
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
    anchor_score: float = 0.0
    p0_anchor_type: str = "none"
    p1_anchor_type: str = "none"
    p0_anchor_score: float = 0.0
    p1_anchor_score: float = 0.0
    final_score: float = 0.0
    visibility_drop_score: float = 0.0
    temporal_score: float = 0.0
    axis_alignment_score: float = 0.0
    open_space_penalty: float = 0.0
    frontier_penalty: float = 0.0
    fragmentation_penalty: float = 0.0
    corridor_split_penalty: float = 0.0
    accepted: bool = False
    reject_reason: str = ""
    debug: dict = field(default_factory=dict)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        custom = _candidate_mask_from_debug(self.debug, shape)
        if custom is not None:
            return custom
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
            "anchor_score": float(self.anchor_score),
            "p0_anchor_type": str(self.p0_anchor_type),
            "p1_anchor_type": str(self.p1_anchor_type),
            "p0_anchor_score": float(self.p0_anchor_score),
            "p1_anchor_score": float(self.p1_anchor_score),
            "final_score": float(self.final_score),
            "visibility_drop_score": float(self.visibility_drop_score),
            "temporal_score": float(self.temporal_score),
            "axis_alignment_score": float(self.axis_alignment_score),
            "open_space_penalty": float(self.open_space_penalty),
            "frontier_penalty": float(self.frontier_penalty),
            "fragmentation_penalty": float(self.fragmentation_penalty),
            "corridor_split_penalty": float(self.corridor_split_penalty),
            "accepted": bool(self.accepted),
            "reject_reason": str(self.reject_reason),
            **_jsonable(self.debug),
        }


@dataclass
class SeparatorScoringConfig:
    wall_support_weight: float = 0.20
    doorway_score_weight: float = 0.20
    visibility_drop_weight: float = 0.15
    topology_gain_weight: float = 0.25
    temporal_weight: float = 0.10
    anchor_weight: float = 0.10
    open_space_penalty_weight: float = 0.15
    frontier_penalty_weight: float = 0.05
    confidence_weight: float = 0.10

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "SeparatorScoringConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class LineExtensionConfig:
    enabled: bool = True
    passes: int = 2
    min_extension_m: float = 0.35
    max_extension_m: float = 1.60
    max_probe_m: float = 1.80
    sample_step_m: float = 0.05
    hit_radius_m: float = 0.10
    free_ratio_min: float = 0.65
    unknown_ratio_max: float = 0.50
    wall_mid_ratio_max: float = 0.15
    require_free_between_start_and_hit: bool = True
    min_free_cells_between_start_and_hit: int = 3
    allow_hit_virtual_door_on_pass2: bool = True
    virtual_target_dilation_m: float = 0.08
    reject_duplicate_extensions: bool = True
    duplicate_endpoint_distance_m: float = 0.10

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "LineExtensionConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class DoorNeckConfig:
    enabled: bool = True
    min_width_m: float = 0.40
    max_width_m: float = 1.80
    min_confidence: float = 0.50
    require_topology_gain: bool = True
    reject_unknown_only_support: bool = True
    reject_endpoint_on_other_door_middle: bool = True
    endpoint_on_other_middle_min_ratio: float = 0.15
    endpoint_on_other_middle_max_ratio: float = 0.85
    endpoint_on_other_tolerance_cells: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "DoorNeckConfig":
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class LineExtensionHit:
    extension_id: int
    source_line_id: int
    source_endpoint: str
    pass_id: int
    p_start_rc: np.ndarray
    p_hit_rc: np.ndarray
    theta: float
    length_m: float
    interior_free_ratio: float
    interior_unknown_ratio: float
    interior_wall_ratio: float
    hit_type: str
    hit_candidate_id: int | None
    confidence: float
    reject_reason: str | None
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "extension_id": int(self.extension_id),
            "source_line_id": int(self.source_line_id),
            "source_endpoint": str(self.source_endpoint),
            "pass_id": int(self.pass_id),
            "p_start_rc": [float(v) for v in np.asarray(self.p_start_rc, dtype=np.float32).tolist()],
            "p_hit_rc": [float(v) for v in np.asarray(self.p_hit_rc, dtype=np.float32).tolist()],
            "theta": float(self.theta),
            "length_m": float(self.length_m),
            "interior_free_ratio": float(self.interior_free_ratio),
            "interior_unknown_ratio": float(self.interior_unknown_ratio),
            "interior_wall_ratio": float(self.interior_wall_ratio),
            "hit_type": str(self.hit_type),
            "hit_candidate_id": None if self.hit_candidate_id is None else int(self.hit_candidate_id),
            "confidence": float(self.confidence),
            "accepted_as_candidate": self.reject_reason is None,
            "reject_reason": None if self.reject_reason is None else str(self.reject_reason),
            "debug": _jsonable(self.debug),
        }


def extend_wall_lines_once(
    filtered_lines: Sequence[FilteredWallLine],
    *,
    free_clean: np.ndarray,
    wall_target_mask: np.ndarray,
    virtual_target_mask: np.ndarray | None,
    unknown_clean: np.ndarray,
    resolution_m: float,
    pass_id: int,
    config: LineExtensionConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[LineExtensionHit], dict]:
    cfg = config if isinstance(config, LineExtensionConfig) else LineExtensionConfig.from_mapping(config)
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_target_mask, dtype=bool)
    virtual = None if virtual_target_mask is None else np.asarray(virtual_target_mask, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    if not bool(cfg.enabled):
        return [], {"enabled": False, "pass_id": int(pass_id), "extension_count": 0}
    hit_radius = max(0, int(round(float(cfg.hit_radius_m) / max(float(resolution_m), 1e-9))))
    virtual_radius = max(0, int(round(float(cfg.virtual_target_dilation_m) / max(float(resolution_m), 1e-9))))
    real_target = dilate(wall, hit_radius)
    virtual_target = dilate(virtual, max(hit_radius, virtual_radius)) if virtual is not None else None
    hits: list[LineExtensionHit] = []
    eid = int(start_id)
    for line in filtered_lines:
        for endpoint in ("p0", "p1"):
            hit = trace_line_extension(
                line,
                endpoint,
                free_clean=free,
                real_wall_target=real_target,
                wall_mid_mask=wall,
                virtual_target=virtual_target,
                virtual_mid_mask=virtual,
                unknown_clean=unknown,
                resolution_m=float(resolution_m),
                pass_id=int(pass_id),
                extension_id=eid,
                config=cfg,
            )
            hits.append(hit)
            eid += 1
    debug = {
        "enabled": True,
        "pass_id": int(pass_id),
        "extension_count": int(len(hits)),
        "accepted_extension_count": int(sum(1 for hit in hits if hit.reject_reason is None)),
        "rejected_by_reason": _extension_reason_counts(hits),
        "extensions": [hit.to_dict() for hit in hits[:1024]],
    }
    return hits, debug


def trace_line_extension(
    line: FilteredWallLine,
    endpoint: str,
    *,
    free_clean: np.ndarray,
    real_wall_target: np.ndarray,
    wall_mid_mask: np.ndarray,
    virtual_target: np.ndarray | None,
    virtual_mid_mask: np.ndarray | None,
    unknown_clean: np.ndarray,
    resolution_m: float,
    pass_id: int,
    extension_id: int,
    config: LineExtensionConfig,
) -> LineExtensionHit:
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(real_wall_target, dtype=bool)
    wall_mid = np.asarray(wall_mid_mask, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    virtual = None if virtual_target is None else np.asarray(virtual_target, dtype=bool)
    virtual_mid = None if virtual_mid_mask is None else np.asarray(virtual_mid_mask, dtype=bool)
    p_start, direction = _extension_start_and_direction(line, endpoint)
    max_steps = max(1, int(round(float(config.max_probe_m) / max(float(resolution_m), 1e-9))))
    hit_radius = max(0, int(round(float(config.hit_radius_m) / max(float(resolution_m), 1e-9))))
    virtual_radius = max(
        hit_radius,
        int(round(float(config.virtual_target_dilation_m) / max(float(resolution_m), 1e-9))),
    )
    sampled: list[tuple[int, int]] = []
    hit_type = "none"
    hit_candidate_id = None
    hit_point = np.asarray(p_start, dtype=np.float32)
    hit_debug: dict[str, object] = {}
    last_point = np.asarray(p_start, dtype=np.float32)
    for step in range(1, max_steps + 1):
        rc = np.rint(np.asarray(p_start, dtype=np.float32) + np.asarray(direction, dtype=np.float32) * float(step)).astype(np.int32)
        if not _inside(rc, free.shape):
            return _line_extension_result(
                line,
                endpoint,
                p_start,
                last_point,
                sampled,
                hit_type="none",
                hit_candidate_id=None,
                reject_reason="reject_extension_no_wall_or_virtual_door_hit",
                pass_id=pass_id,
                extension_id=extension_id,
                resolution_m=float(resolution_m),
                config=config,
                debug={"stop": "out_of_bounds", "_free_clean": free, "_wall_target": wall_mid, "_unknown_clean": unknown},
            )
        sampled.append((int(rc[0]), int(rc[1])))
        last_point = rc.astype(np.float32)
        free_count = int(sum(1 for r, c in sampled if bool(free[r, c])))
        if bool(wall_mid[int(rc[0]), int(rc[1])]):
            if free_count < int(config.min_free_cells_between_start_and_hit):
                return _line_extension_result(
                    line,
                    endpoint,
                    p_start,
                    last_point,
                    sampled,
                    hit_type="blocked_wall",
                    hit_candidate_id=None,
                    reject_reason="reject_extension_blocked_by_near_wall",
                    pass_id=pass_id,
                    extension_id=extension_id,
                    resolution_m=float(resolution_m),
                    config=config,
                    debug={
                        "stop": "blocked_by_near_wall_before_min_free",
                        "blocked_cell": [int(rc[0]), int(rc[1])],
                        "_free_clean": free,
                        "_wall_target": wall_mid,
                        "_unknown_clean": unknown,
                    },
                )
            hit_type = "real_wall"
            hit_point = _snap_hit_point_to_target(rc, wall_mid, direction, radius_cells=hit_radius)
            hit_debug = {
                "hit_cell_before_snap": [int(rc[0]), int(rc[1])],
                "hit_cell_snap_target": "real_wall",
            }
            break
        if free_count < int(config.min_free_cells_between_start_and_hit):
            continue
        if bool(wall[int(rc[0]), int(rc[1])]):
            hit_type = "real_wall"
            hit_point = _snap_hit_point_to_target(rc, wall_mid, direction, radius_cells=hit_radius)
            hit_debug = {
                "hit_cell_before_snap": [int(rc[0]), int(rc[1])],
                "hit_cell_snap_target": "real_wall",
            }
            break
        if (
            virtual is not None
            and bool(config.allow_hit_virtual_door_on_pass2)
            and int(pass_id) >= 2
            and bool(virtual[int(rc[0]), int(rc[1])])
        ):
            hit_type = "virtual_door"
            hit_point = _snap_hit_point_to_target(
                rc,
                virtual_mid if virtual_mid is not None else virtual,
                direction,
                radius_cells=virtual_radius,
            )
            hit_debug = {
                "hit_cell_before_snap": [int(rc[0]), int(rc[1])],
                "hit_cell_snap_target": "virtual_door",
            }
            break
    if hit_type == "none":
        reason = "reject_extension_hit_unknown" if _sample_ratio(sampled, unknown) > float(config.unknown_ratio_max) else "reject_extension_no_wall_or_virtual_door_hit"
        return _line_extension_result(
            line,
            endpoint,
            p_start,
            last_point,
            sampled,
            hit_type=hit_type,
            hit_candidate_id=hit_candidate_id,
            reject_reason=reason,
            pass_id=pass_id,
            extension_id=extension_id,
            resolution_m=float(resolution_m),
            config=config,
            debug={"stop": "max_probe", "_free_clean": free, "_wall_target": wall_mid, "_unknown_clean": unknown},
        )
    return _line_extension_result(
        line,
        endpoint,
        p_start,
        hit_point,
        sampled,
        hit_type=hit_type,
        hit_candidate_id=hit_candidate_id,
        reject_reason=None,
        pass_id=pass_id,
        extension_id=extension_id,
        resolution_m=float(resolution_m),
        config=config,
        debug={**hit_debug, "stop": "target_hit", "_free_clean": free, "_wall_target": wall_mid, "_unknown_clean": unknown},
    )


def build_door_neck_candidates_from_extensions(
    extensions: Sequence[LineExtensionHit],
    *,
    accepted_virtual_targets: Sequence[SeparatorCandidate] | None,
    resolution_m: float,
    config: DoorNeckConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    cfg = config if isinstance(config, DoorNeckConfig) else DoorNeckConfig.from_mapping(config)
    if not bool(cfg.enabled):
        return [], {"enabled": False, "candidate_count": 0}
    _ = accepted_virtual_targets, resolution_m
    candidates: list[SeparatorCandidate] = []
    rejected: list[dict] = []
    cid = int(start_id)
    for hit in extensions:
        if hit.reject_reason is not None:
            rejected.append(hit.to_dict())
            continue
        if not (float(cfg.min_width_m) <= float(hit.length_m) <= float(cfg.max_width_m)):
            hit.reject_reason = "reject_extension_length_out_of_door_range"
            rejected.append(hit.to_dict())
            continue
        if float(hit.confidence) < float(cfg.min_confidence):
            hit.reject_reason = "reject_low_line_extension_confidence"
            rejected.append(hit.to_dict())
            continue
        pass_bonus = 1.0 if int(hit.pass_id) == 1 else (0.85 if str(hit.hit_type) == "virtual_door" else 0.95)
        confidence = float(
            np.clip(
                0.30 * float(hit.interior_free_ratio)
                + 0.25 * float(hit.confidence)
                + 0.20 * (1.0 if str(hit.hit_type) in {"real_wall", "virtual_door", "virtual_neck"} else 0.0)
                + 0.15 * 1.0
                + 0.10 * pass_bonus,
                0.0,
                1.0,
            )
        )
        candidate = SeparatorCandidate(
            candidate_id=cid,
            kind="line_extension_door_neck",
            p0_rc=np.asarray(hit.p_start_rc, dtype=np.float32),
            p1_rc=np.asarray(hit.p_hit_rc, dtype=np.float32),
            theta=float(hit.theta),
            length_m=float(hit.length_m),
            confidence=confidence,
            source_segment_ids=[int(hit.source_line_id)],
            wall_support_score=float(hit.confidence),
            free_gap_score=float(hit.interior_free_ratio),
            doorway_score=float(hit.interior_free_ratio if str(hit.hit_type) in {"real_wall", "virtual_door"} else 0.0),
            debug={
                "candidate_source": "line_extension",
                "kind_detail": "door_neck",
                "pass_id": int(hit.pass_id),
                "source_extension_ids": [int(hit.extension_id)],
                "source_line_ids": [int(hit.source_line_id)],
                "hit_types": [str(hit.hit_type)],
                "width_m": float(hit.length_m),
                "neck_score": float(hit.interior_free_ratio),
                "doorway_score": float(hit.interior_free_ratio),
                "corridor_false_pair_score": 0.0,
                "extension": hit.to_dict(),
            },
        )
        candidates.append(candidate)
        cid += 1
    debug = {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "rejected_extension_count": int(len(rejected)),
        "candidates": [candidate.to_dict() for candidate in candidates[:1024]],
        "rejected_extensions": rejected[:1024],
    }
    return candidates, debug


def build_l_corner_door_neck_candidates(
    filtered_lines: Sequence[FilteredWallLine],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    line_config: LineExtensionConfig | Mapping[str, object] | None = None,
    door_config: DoorNeckConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    line_cfg = line_config if isinstance(line_config, LineExtensionConfig) else LineExtensionConfig.from_mapping(line_config)
    door_cfg = door_config if isinstance(door_config, DoorNeckConfig) else DoorNeckConfig.from_mapping(door_config)
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    if not bool(line_cfg.enabled) or not bool(door_cfg.enabled):
        return [], {"enabled": False, "candidate_count": 0}
    horizontal: list[tuple[FilteredWallLine, dict]] = []
    vertical: list[tuple[FilteredWallLine, dict]] = []
    for line in filtered_lines:
        info = _filtered_line_axis_info(line)
        if info is None:
            continue
        if str(info["axis"]) == "horizontal":
            horizontal.append((line, info))
        elif str(info["axis"]) == "vertical":
            vertical.append((line, info))

    candidates: list[SeparatorCandidate] = []
    rejected: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = set()
    cid = int(start_id)
    for h_line, h_info in horizontal:
        for v_line, v_info in vertical:
            corner_rc = (int(h_info["line"]), int(v_info["line"]))
            for h_endpoint_name, h_endpoint in h_info["endpoints"]:
                if not _endpoint_faces_corner(str(h_endpoint_name), int(h_endpoint[1]), int(corner_rc[1])):
                    continue
                for v_endpoint_name, v_endpoint in v_info["endpoints"]:
                    if not _endpoint_faces_corner(str(v_endpoint_name), int(v_endpoint[0]), int(corner_rc[0])):
                        continue
                    path_cells = _l_corner_path_cells(
                        horizontal_line=int(corner_rc[0]),
                        horizontal_endpoint_col=int(h_endpoint[1]),
                        vertical_line=int(corner_rc[1]),
                        vertical_endpoint_row=int(v_endpoint[0]),
                        shape=free.shape,
                    )
                    if len(path_cells) == 0:
                        continue
                    gap_cells = int(abs(int(h_endpoint[1]) - int(corner_rc[1])) + abs(int(v_endpoint[0]) - int(corner_rc[0])))
                    length_m = float(gap_cells * float(resolution_m))
                    key = (
                        (int(h_endpoint[0]), int(h_endpoint[1])),
                        (int(v_endpoint[0]), int(v_endpoint[1])),
                        (int(corner_rc[0]), int(corner_rc[1])),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    free_ratio = float(np.count_nonzero(free[path_cells[:, 0], path_cells[:, 1]])) / float(len(path_cells))
                    unknown_ratio = float(np.count_nonzero(unknown[path_cells[:, 0], path_cells[:, 1]])) / float(len(path_cells))
                    reject_reason = ""
                    if length_m < float(door_cfg.min_width_m) or length_m > float(door_cfg.max_width_m):
                        reject_reason = "reject_l_corner_length_out_of_door_range"
                    elif free_ratio < float(line_cfg.free_ratio_min):
                        reject_reason = "reject_l_corner_not_enough_free"
                    elif unknown_ratio > float(line_cfg.unknown_ratio_max):
                        reject_reason = "reject_l_corner_hit_unknown"
                    endpoint_support = float(min(float(h_line.confidence), float(v_line.confidence)))
                    confidence = float(
                        np.clip(
                            0.30 * float(free_ratio)
                            + 0.25 * float(endpoint_support)
                            + 0.20 * (1.0 - min(1.0, unknown_ratio))
                            + 0.15 * 1.0
                            + 0.10 * 0.95,
                            0.0,
                            1.0,
                        )
                    )
                    if confidence < float(door_cfg.min_confidence):
                        reject_reason = reject_reason or "reject_low_l_corner_confidence"
                    item_debug = {
                        "candidate_source": "l_corner_door_neck",
                        "kind_detail": "door_neck",
                        "pass_id": 2,
                        "source_line_ids": [int(h_line.line_id), int(v_line.line_id)],
                        "horizontal_endpoint": str(h_endpoint_name),
                        "vertical_endpoint": str(v_endpoint_name),
                        "horizontal_endpoint_rc": [int(h_endpoint[0]), int(h_endpoint[1])],
                        "vertical_endpoint_rc": [int(v_endpoint[0]), int(v_endpoint[1])],
                        "corner_rc": [int(corner_rc[0]), int(corner_rc[1])],
                        "width_m": float(length_m),
                        "neck_score": float(free_ratio),
                        "doorway_score": float(free_ratio),
                        "l_corner_free_ratio": float(free_ratio),
                        "l_corner_unknown_ratio": float(unknown_ratio),
                        "mask_cells_rc": [[int(r), int(c)] for r, c in path_cells.tolist()],
                    }
                    if reject_reason:
                        rejected.append({**item_debug, "reject_reason": str(reject_reason), "confidence": float(confidence)})
                        continue
                    candidate = SeparatorCandidate(
                        candidate_id=cid,
                        kind="line_extension_door_neck",
                        p0_rc=np.asarray(h_endpoint, dtype=np.float32),
                        p1_rc=np.asarray(v_endpoint, dtype=np.float32),
                        theta=float(np.pi / 4.0),
                        length_m=float(length_m),
                        confidence=float(confidence),
                        source_segment_ids=[int(h_line.line_id), int(v_line.line_id)],
                        wall_support_score=float(endpoint_support),
                        free_gap_score=float(free_ratio),
                        doorway_score=float(free_ratio),
                        debug=item_debug,
                    )
                    candidates.append(candidate)
                    cid += 1
    debug = {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "rejected_candidate_count": int(len(rejected)),
        "candidates": [candidate.to_dict() for candidate in candidates[:1024]],
        "rejected_candidates": rejected[:1024],
    }
    return candidates, debug


def build_door_neck_candidates_from_extension_intersections(
    extensions: Sequence[LineExtensionHit],
    *,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    line_config: LineExtensionConfig | Mapping[str, object] | None = None,
    door_config: DoorNeckConfig | Mapping[str, object] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], np.ndarray, dict]:
    """Create door/neck candidates where two wall-extension probes cross.

    A red extension probe crossing another red extension probe is meaningful
    structural evidence, even when neither probe hits an already materialized
    wall cell.  This helper turns that crossing into virtual-neck endpoints so
    the normal topology test can decide whether the resulting separator really
    splits navigable free space.
    """
    line_cfg = line_config if isinstance(line_config, LineExtensionConfig) else LineExtensionConfig.from_mapping(line_config)
    door_cfg = door_config if isinstance(door_config, DoorNeckConfig) else DoorNeckConfig.from_mapping(door_config)
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    target = np.zeros_like(free, dtype=bool)
    if not bool(line_cfg.enabled) or not bool(door_cfg.enabled):
        return [], target, {"enabled": False, "candidate_count": 0, "reason": "disabled"}

    usable: list[tuple[LineExtensionHit, str, int, int, int]] = []
    for hit in extensions:
        if not _extension_can_seed_intersection(hit):
            continue
        axis_info = _extension_axis_info(hit, free.shape)
        if axis_info is None:
            continue
        axis, fixed, lo, hi = axis_info
        if hi <= lo:
            continue
        usable.append((hit, axis, fixed, lo, hi))

    candidates: list[SeparatorCandidate] = []
    events: list[dict] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    cid = int(start_id)
    for idx, (a, axis_a, fixed_a, lo_a, hi_a) in enumerate(usable):
        for b, axis_b, fixed_b, lo_b, hi_b in usable[idx + 1 :]:
            if int(a.source_line_id) == int(b.source_line_id):
                continue
            if axis_a == axis_b:
                continue
            if axis_a == "horizontal":
                rc = np.asarray([fixed_a, fixed_b], dtype=np.int32)
                inside = lo_a <= fixed_b <= hi_a and lo_b <= fixed_a <= hi_b
            else:
                rc = np.asarray([fixed_b, fixed_a], dtype=np.int32)
                inside = lo_b <= fixed_a <= hi_b and lo_a <= fixed_b <= hi_a
            if not inside or not _inside(rc, free.shape) or not bool(free[int(rc[0]), int(rc[1])]):
                continue
            pair_events = []
            for hit in (a, b):
                candidate = _candidate_from_extension_intersection(
                    hit,
                    other_hit=b if hit is a else a,
                    intersection_rc=rc,
                    free_clean=free,
                    unknown_clean=unknown,
                    resolution_m=float(resolution_m),
                    line_config=line_cfg,
                    door_config=door_cfg,
                    candidate_id=cid,
                )
                if candidate is None:
                    continue
                key = (
                    int(candidate.source_segment_ids[0]),
                    int(candidate.source_segment_ids[1]),
                    int(round(float(candidate.p0_rc[0]))),
                    int(round(float(candidate.p0_rc[1]))),
                    int(rc[0]) * int(free.shape[1]) + int(rc[1]),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                target[int(rc[0]), int(rc[1])] = True
                pair_events.append(
                    {
                        "candidate_id": int(candidate.candidate_id),
                        "source_extension_ids": list(candidate.debug.get("source_extension_ids", [])),
                        "length_m": float(candidate.length_m),
                    }
                )
                cid += 1
            if pair_events:
                events.append(
                    {
                        "intersection_rc": [int(rc[0]), int(rc[1])],
                        "source_extension_ids": [int(a.extension_id), int(b.extension_id)],
                        "source_line_ids": [int(a.source_line_id), int(b.source_line_id)],
                        "candidates": pair_events,
                    }
                )
    debug = {
        "enabled": True,
        "usable_extension_count": int(len(usable)),
        "intersection_count": int(len(events)),
        "candidate_count": int(len(candidates)),
        "virtual_target_cell_count": int(np.count_nonzero(target)),
        "events": events[:512],
    }
    return candidates, target.astype(bool), debug


def rasterize_candidates(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int], thickness_cells: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= separator_mask_for_candidate(candidate, shape, thickness_cells)
    return out.astype(bool)


def reject_candidates_ending_on_other_door_middle(
    candidates: Sequence[SeparatorCandidate],
    shape: tuple[int, int],
    *,
    config: DoorNeckConfig | Mapping[str, object] | None = None,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], dict]:
    cfg = config if isinstance(config, DoorNeckConfig) else DoorNeckConfig.from_mapping(config)
    items = list(candidates)
    if not bool(cfg.reject_endpoint_on_other_door_middle):
        return items, [], {"enabled": False, "rejected_count": 0}
    door_like = {"doorway_virtual_cut", "line_extension_door_neck", "extension_intersection_cut", "corridor_room_neck_cut"}
    min_ratio = float(np.clip(float(cfg.endpoint_on_other_middle_min_ratio), 0.0, 1.0))
    max_ratio = float(np.clip(float(cfg.endpoint_on_other_middle_max_ratio), 0.0, 1.0))
    if max_ratio < min_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio
    tolerance = max(0, int(cfg.endpoint_on_other_tolerance_cells))
    kept: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    events: list[dict] = []
    path_cache: dict[int, np.ndarray] = {}

    for candidate in items:
        if str(candidate.kind) not in {"line_extension_door_neck", "extension_intersection_cut"}:
            kept.append(candidate)
            continue
        reject_event = None
        for endpoint_name, endpoint in (
            ("p0", np.rint(np.asarray(candidate.p0_rc, dtype=np.float32)).astype(np.int32)),
            ("p1", np.rint(np.asarray(candidate.p1_rc, dtype=np.float32)).astype(np.int32)),
        ):
            for other in items:
                if other is candidate or str(other.kind) not in door_like:
                    continue
                other_path = path_cache.get(id(other))
                if other_path is None:
                    other_path = _candidate_ordered_path_cells(other, shape)
                    path_cache[id(other)] = other_path
                hit = _endpoint_middle_hit(endpoint, other_path, min_ratio=min_ratio, max_ratio=max_ratio, tolerance_cells=tolerance)
                if hit is None:
                    continue
                reject_event = {
                    "candidate_id": int(candidate.candidate_id),
                    "candidate_kind": str(candidate.kind),
                    "candidate_source": str(candidate.debug.get("candidate_source", "")),
                    "endpoint": str(endpoint_name),
                    "endpoint_rc": [int(endpoint[0]), int(endpoint[1])],
                    "other_candidate_id": int(other.candidate_id),
                    "other_candidate_kind": str(other.kind),
                    "other_candidate_source": str(other.debug.get("candidate_source", "")),
                    **hit,
                }
                break
            if reject_event is not None:
                break
        if reject_event is None:
            kept.append(candidate)
            continue
        candidate.accepted = False
        candidate.reject_reason = "reject_endpoint_on_other_door_middle"
        candidate.debug.update(
            {
                "endpoint_on_other_door_middle_rejected": True,
                "endpoint_on_other_door_middle": reject_event,
            }
        )
        rejected.append(candidate)
        events.append(reject_event)

    return kept, rejected, {
        "enabled": True,
        "middle_min_ratio": float(min_ratio),
        "middle_max_ratio": float(max_ratio),
        "tolerance_cells": int(tolerance),
        "input_count": int(len(items)),
        "kept_count": int(len(kept)),
        "rejected_count": int(len(rejected)),
        "events": events[:512],
    }


def fill_noise_wall_gaps_from_runs(
    runs: Sequence[SnappedWallRun],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    config: NoiseWallGapFillConfig | Mapping[str, object] | None = None,
) -> tuple[np.ndarray, dict]:
    cfg = config if isinstance(config, NoiseWallGapFillConfig) else NoiseWallGapFillConfig.from_mapping(config)
    out = np.zeros(shape, dtype=bool)
    if not bool(cfg.enabled):
        return out, {"enabled": False, "filled_gap_count": 0, "filled_cell_count": 0, "strict_less_than_max_gap": True}
    runs_list = list(runs)
    max_gap_m = float(cfg.max_gap_m)
    max_lateral_cells = max(0, int(round(float(cfg.max_lateral_offset_m) / max(float(resolution_m), 1e-9))))
    thickness = max(0, int(cfg.thickness_cells))
    corner_thickness = max(0, int(cfg.corner_thickness_cells))
    events: list[dict] = []
    seen: set[tuple[str, int, int, int]] = set()
    for idx, a in enumerate(runs_list):
        for b in runs_list[idx + 1 :]:
            if str(a.axis) != str(b.axis):
                continue
            if abs(int(a.line) - int(b.line)) > max_lateral_cells:
                continue
            gap = _run_gap(a, b)
            if gap is None:
                continue
            gap_start, gap_end = gap
            gap_cells = int(gap_end - gap_start + 1)
            if gap_cells <= 0:
                continue
            length_m = float(gap_cells * float(resolution_m))
            if not (length_m < max_gap_m - 1e-9):
                continue
            endpoint_support = float(min(float(a.confidence), float(b.confidence)))
            if endpoint_support < float(cfg.min_endpoint_support):
                continue
            line = int(round(float((int(a.line) + int(b.line)) / 2.0)))
            key = (str(a.axis), int(line), int(gap_start), int(gap_end))
            if key in seen:
                continue
            seen.add(key)
            p0, p1, _theta = _candidate_points(str(a.axis), line, gap_start, gap_end)
            mask = rasterize_line(p0, p1, shape)
            if thickness > 0:
                mask = dilate(mask, thickness)
            out |= mask
            events.append(
                {
                    "axis": str(a.axis),
                    "line": int(line),
                    "gap_start": int(gap_start),
                    "gap_end": int(gap_end),
                    "gap_cells": int(gap_cells),
                    "gap_m": float(length_m),
                    "endpoint_support": float(endpoint_support),
                    "source_run_ids": [int(a.run_id), int(b.run_id)],
                    "strict_less_than_max_gap_m": float(max_gap_m),
                    "kind": "collinear_gap",
                }
            )
    if bool(cfg.corner_gap_enabled):
        corner_events, corner_fill = _fill_l_corner_gaps_from_runs(
            runs_list,
            shape=shape,
            resolution_m=float(resolution_m),
            max_gap_m=max_gap_m,
            min_endpoint_support=float(cfg.min_endpoint_support),
            thickness_cells=corner_thickness,
        )
        out |= corner_fill
        events.extend(corner_events)
    return out.astype(bool), {
        "enabled": True,
        "max_gap_m": float(max_gap_m),
        "corner_gap_enabled": bool(cfg.corner_gap_enabled),
        "corner_thickness_cells": int(corner_thickness),
        "strict_less_than_max_gap": True,
        "max_lateral_offset_cells": int(max_lateral_cells),
        "input_run_count": int(len(runs_list)),
        "filled_gap_count": int(len(events)),
        "corner_filled_gap_count": int(sum(1 for event in events if str(event.get("kind", "")) == "l_corner_gap")),
        "filled_cell_count": int(np.count_nonzero(out)),
        "events": events[:512],
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
    missed_scan_config: MissedScanGapClosureConfig | Mapping[str, object] | None = None,
    short_unknown_config: ShortUnknownGapClosureConfig | Mapping[str, object] | None = None,
    single_sided_config: SingleSidedWallExtensionConfig | Mapping[str, object] | None = None,
    runs: Sequence[SnappedWallRun] | None = None,
    start_id: int = 1,
) -> tuple[list[SeparatorCandidate], dict]:
    physical = physical_config if isinstance(physical_config, PhysicalWallCompletionConfig) else PhysicalWallCompletionConfig.from_mapping(physical_config)
    doorway = doorway_config if isinstance(doorway_config, DoorwayVirtualCutConfig) else DoorwayVirtualCutConfig.from_mapping(doorway_config)
    missed = missed_scan_config if isinstance(missed_scan_config, MissedScanGapClosureConfig) else MissedScanGapClosureConfig.from_mapping(missed_scan_config)
    short_unknown = (
        short_unknown_config
        if isinstance(short_unknown_config, ShortUnknownGapClosureConfig)
        else ShortUnknownGapClosureConfig.from_mapping(short_unknown_config)
    )
    single_sided = (
        single_sided_config
        if isinstance(single_sided_config, SingleSidedWallExtensionConfig)
        else SingleSidedWallExtensionConfig.from_mapping(single_sided_config)
    )
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    if runs is None:
        runs, snap_debug = snap_wall_segments_to_runs(
            segments,
            wall,
            resolution_m=float(resolution_m),
            max_angle_to_axis_deg=25.0,
            support_band_cells=2,
            min_run_length_m=0.25,
            min_support_ratio=0.15,
        )
    else:
        snap_debug = {"used_precomputed_runs": True, "run_count": int(len(runs))}
    candidates: list[SeparatorCandidate] = []
    sid = int(start_id)
    seen: set[tuple[str, str, int, int, int]] = set()
    max_lateral_cells = max(0, int(round(max(float(physical.max_lateral_offset_m), 0.20) / max(float(resolution_m), 1e-9))))
    for idx, a in enumerate(runs):
        for b in runs[idx + 1 :]:
            if str(a.axis) != str(b.axis):
                continue
            if abs(int(a.line) - int(b.line)) > max_lateral_cells:
                continue
            gap = _run_gap(a, b)
            if gap is None:
                continue
            gap_start, gap_end = gap
            gap_cells = int(gap_end - gap_start + 1)
            if gap_cells <= 0:
                continue
            length_m = float(gap_cells * float(resolution_m))
            line = int(round(float((int(a.line) + int(b.line)) / 2.0)))
            p0, p1, theta = _candidate_points(str(a.axis), line, gap_start, gap_end)
            stats = _line_stats(p0, p1, free, wall, unknown)
            source_ids = sorted({*map(int, a.source_segment_ids), *map(int, b.source_segment_ids)})
            wall_support = float(min(float(a.confidence), float(b.confidence)))
            base_debug = {
                "axis": str(a.axis),
                "line": int(line),
                "gap_m": float(length_m),
                "gap_cells": int(gap_cells),
                "free_ratio": float(stats["free_ratio"]),
                "unknown_ratio": float(stats["unknown_ratio"]),
                "wall_ratio": float(stats["wall_ratio"]),
                "source_run_ids": [int(a.run_id), int(b.run_id)],
                "p0_before_anchor_extension": [int(v) for v in np.rint(p0).astype(int).tolist()],
                "p1_before_anchor_extension": [int(v) for v in np.rint(p1).astype(int).tolist()],
            }
            for kind, confidence, score in _candidate_kinds_for_gap(
                length_m=length_m,
                free_ratio=float(stats["free_ratio"]),
                unknown_ratio=float(stats["unknown_ratio"]),
                wall_support=wall_support,
                physical=physical,
                missed=missed,
                short_unknown=short_unknown,
                doorway=doorway,
            ):
                key = (kind, str(a.axis), int(line), int(gap_start), int(gap_end))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SeparatorCandidate(
                        candidate_id=sid,
                        kind=kind,
                        p0_rc=p0.copy(),
                        p1_rc=p1.copy(),
                        theta=theta,
                        length_m=length_m,
                        confidence=float(confidence),
                        source_segment_ids=source_ids,
                        wall_support_score=wall_support,
                        free_gap_score=float(stats["free_ratio"]),
                        doorway_score=float(score if kind == "doorway_virtual_cut" else 0.0),
                        debug={**base_debug, "candidate_source": "wall_run_gap"},
                    )
                )
                sid += 1
    if bool(single_sided.enabled):
        extra, sid = _single_sided_candidates(
            runs,
            free,
            wall,
            unknown,
            resolution_m=float(resolution_m),
            config=single_sided,
            start_id=sid,
            seen=seen,
        )
        candidates.extend(extra)
    return candidates, {
        "physical_wall_completion_enabled": bool(physical.enabled),
        "missed_scan_gap_closure_enabled": bool(missed.enabled),
        "short_unknown_gap_closure_enabled": bool(short_unknown.enabled),
        "doorway_virtual_cut_enabled": bool(doorway.enabled),
        "single_sided_wall_extension_enabled": bool(single_sided.enabled),
        "wall_gap_candidate_count": int(len(candidates)),
        "candidate_count_by_kind": _kind_counts(candidates),
        "snap_debug": snap_debug,
    }


def extend_candidate_to_anchors(
    candidate: SeparatorCandidate,
    *,
    free_clean: np.ndarray,
    wall_candidate_clean: np.ndarray,
    unknown_clean: np.ndarray,
    existing_separator_map: np.ndarray,
    resolution_m: float,
    max_anchor_extension_m: float,
) -> tuple[SeparatorCandidate, dict]:
    free = np.asarray(free_clean, dtype=bool)
    wall = np.asarray(wall_candidate_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    existing = np.asarray(existing_separator_map, dtype=bool)
    max_steps = max(0, int(round(float(max_anchor_extension_m) / max(float(resolution_m), 1e-9))))
    p0_before = np.rint(candidate.p0_rc).astype(np.int32)
    p1_before = np.rint(candidate.p1_rc).astype(np.int32)
    direction = _unit_grid_direction(p0_before, p1_before)
    p0_after, a0 = _extend_endpoint_to_anchor(p0_before, -direction, free, wall, unknown, existing, max_steps)
    p1_after, a1 = _extend_endpoint_to_anchor(p1_before, direction, free, wall, unknown, existing, max_steps)
    candidate.p0_rc = p0_after.astype(np.float32)
    candidate.p1_rc = p1_after.astype(np.float32)
    candidate.length_m = float(max(1, int(np.max(np.abs(p1_after - p0_after)) + 1)) * float(resolution_m))
    score0 = _anchor_score(str(a0["type"]))
    score1 = _anchor_score(str(a1["type"]))
    score = float(0.5 * (score0 + score1))
    candidate.p0_anchor_type = str(a0["type"])
    candidate.p1_anchor_type = str(a1["type"])
    candidate.p0_anchor_score = float(score0)
    candidate.p1_anchor_score = float(score1)
    candidate.anchor_score = float(score)
    debug = {
        "p0_before_anchor_extension": [int(v) for v in p0_before.tolist()],
        "p1_before_anchor_extension": [int(v) for v in p1_before.tolist()],
        "p0_after_anchor_extension": [int(v) for v in p0_after.tolist()],
        "p1_after_anchor_extension": [int(v) for v in p1_after.tolist()],
        "anchor_type_p0": str(a0["type"]),
        "anchor_type_p1": str(a1["type"]),
        "p0_anchor_type": str(a0["type"]),
        "p1_anchor_type": str(a1["type"]),
        "p0_anchor_score": float(score0),
        "p1_anchor_score": float(score1),
        "anchor_score": float(score),
        "anchor_steps_p0": int(a0["steps"]),
        "anchor_steps_p1": int(a1["steps"]),
    }
    candidate.debug.update(debug)
    return candidate, debug


def _run_gap(a: SnappedWallRun, b: SnappedWallRun) -> tuple[int, int] | None:
    left, right = (a, b) if int(a.start) <= int(b.start) else (b, a)
    if int(left.end) >= int(right.start) - 1:
        return None
    return int(left.end) + 1, int(right.start) - 1


def _fill_l_corner_gaps_from_runs(
    runs: Sequence[SnappedWallRun],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    max_gap_m: float,
    min_endpoint_support: float,
    thickness_cells: int,
) -> tuple[list[dict], np.ndarray]:
    out = np.zeros(shape, dtype=bool)
    events: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for idx, a in enumerate(runs):
        for b in runs[idx + 1 :]:
            if str(a.axis) == str(b.axis):
                continue
            if str(a.axis) == "horizontal" and str(b.axis) == "vertical":
                horizontal, vertical = a, b
            elif str(a.axis) == "vertical" and str(b.axis) == "horizontal":
                horizontal, vertical = b, a
            else:
                continue
            endpoint_support = float(min(float(horizontal.confidence), float(vertical.confidence)))
            if endpoint_support < float(min_endpoint_support):
                continue
            h_line = int(horizontal.line)
            v_line = int(vertical.line)
            for h_endpoint_name, h_col in (("start", int(horizontal.start)), ("end", int(horizontal.end))):
                if not _endpoint_faces_corner(str(h_endpoint_name), int(h_col), int(v_line)):
                    continue
                for v_endpoint_name, v_row in (("start", int(vertical.start)), ("end", int(vertical.end))):
                    if not _endpoint_faces_corner(str(v_endpoint_name), int(v_row), int(h_line)):
                        continue
                    gap_cells = int(abs(h_col - v_line) + abs(v_row - h_line))
                    if gap_cells <= 0:
                        continue
                    length_m = float(gap_cells * float(resolution_m))
                    if not (length_m < float(max_gap_m) - 1e-9):
                        continue
                    key = (int(h_line), int(v_line), int(h_col), int(v_row))
                    if key in seen:
                        continue
                    seen.add(key)
                    mask = _l_corner_gap_mask(
                        shape,
                        horizontal_line=h_line,
                        horizontal_endpoint_col=int(h_col),
                        vertical_line=v_line,
                        vertical_endpoint_row=int(v_row),
                    )
                    if not np.any(mask):
                        continue
                    if int(thickness_cells) > 0:
                        mask = dilate(mask, int(thickness_cells))
                    out |= mask
                    events.append(
                        {
                            "kind": "l_corner_gap",
                            "axis": "l_corner",
                            "corner_rc": [int(h_line), int(v_line)],
                            "horizontal_endpoint_rc": [int(h_line), int(h_col)],
                            "vertical_endpoint_rc": [int(v_row), int(v_line)],
                            "horizontal_endpoint": str(h_endpoint_name),
                            "vertical_endpoint": str(v_endpoint_name),
                            "gap_cells": int(gap_cells),
                            "gap_m": float(length_m),
                            "endpoint_support": float(endpoint_support),
                            "source_run_ids": [int(horizontal.run_id), int(vertical.run_id)],
                            "strict_less_than_max_gap_m": float(max_gap_m),
                            "corner_thickness_cells": int(thickness_cells),
                        }
                    )
    return events, out.astype(bool)


def _endpoint_faces_corner(endpoint_name: str, endpoint_coord: int, corner_coord: int) -> bool:
    if str(endpoint_name) == "start":
        return int(corner_coord) <= int(endpoint_coord)
    return int(corner_coord) >= int(endpoint_coord)


def _l_corner_gap_mask(
    shape: tuple[int, int],
    *,
    horizontal_line: int,
    horizontal_endpoint_col: int,
    vertical_line: int,
    vertical_endpoint_row: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _paint_horizontal_gap_leg(mask, int(horizontal_line), int(horizontal_endpoint_col), int(vertical_line))
    _paint_vertical_gap_leg(mask, int(vertical_line), int(vertical_endpoint_row), int(horizontal_line))
    return mask


def _paint_horizontal_gap_leg(mask: np.ndarray, row: int, endpoint_col: int, corner_col: int) -> None:
    if endpoint_col == corner_col or not (0 <= int(row) < mask.shape[0]):
        return
    if endpoint_col < corner_col:
        start, end = endpoint_col + 1, corner_col
    else:
        start, end = corner_col, endpoint_col - 1
    start = max(0, int(start))
    end = min(mask.shape[1] - 1, int(end))
    if start <= end:
        mask[int(row), start : end + 1] = True


def _paint_vertical_gap_leg(mask: np.ndarray, col: int, endpoint_row: int, corner_row: int) -> None:
    if endpoint_row == corner_row or not (0 <= int(col) < mask.shape[1]):
        return
    if endpoint_row < corner_row:
        start, end = endpoint_row + 1, corner_row
    else:
        start, end = corner_row, endpoint_row - 1
    start = max(0, int(start))
    end = min(mask.shape[0] - 1, int(end))
    if start <= end:
        mask[start : end + 1, int(col)] = True


def _line_extension_result(
    line: FilteredWallLine,
    endpoint: str,
    p_start: np.ndarray,
    p_hit: np.ndarray,
    sampled: Sequence[tuple[int, int]],
    *,
    hit_type: str,
    hit_candidate_id: int | None,
    reject_reason: str | None,
    pass_id: int,
    extension_id: int,
    resolution_m: float,
    config: LineExtensionConfig,
    debug: Mapping[str, object] | None = None,
) -> LineExtensionHit:
    # Ratios are computed below from explicit masks by the caller-populated sampled
    # cells. Empty paths stay rejected by length/free-cell checks.
    return _finalize_line_extension_hit(
        line,
        endpoint,
        p_start,
        p_hit,
        sampled,
        hit_type=hit_type,
        hit_candidate_id=hit_candidate_id,
        reject_reason=reject_reason,
        pass_id=pass_id,
        extension_id=extension_id,
        resolution_m=float(resolution_m),
        config=config,
        debug=dict(debug or {}),
    )


def _finalize_line_extension_hit(
    line: FilteredWallLine,
    endpoint: str,
    p_start: np.ndarray,
    p_hit: np.ndarray,
    sampled: Sequence[tuple[int, int]],
    *,
    hit_type: str,
    hit_candidate_id: int | None,
    reject_reason: str | None,
    pass_id: int,
    extension_id: int,
    resolution_m: float,
    config: LineExtensionConfig,
    debug: dict,
) -> LineExtensionHit:
    # The masks are attached by trace_line_extension just before this helper is
    # called. Keeping them out of the public debug prevents huge JSON payloads.
    free = np.asarray(debug.pop("_free_clean"), dtype=bool)
    wall = np.asarray(debug.pop("_wall_target"), dtype=bool)
    unknown = np.asarray(debug.pop("_unknown_clean"), dtype=bool)
    interior = list(sampled[:-1]) if len(sampled) > 1 else list(sampled)
    length_m = float(max(0, len(sampled)) * float(resolution_m))
    free_ratio = _sample_ratio(interior, free)
    unknown_ratio = _sample_ratio(interior, unknown)
    wall_ratio = _sample_ratio(interior, wall)
    reason = reject_reason
    if reason is None:
        if length_m < float(config.min_extension_m) or length_m > float(config.max_extension_m):
            reason = "reject_extension_length_out_of_door_range"
        elif free_ratio < float(config.free_ratio_min):
            reason = "reject_extension_not_enough_free"
        elif unknown_ratio > float(config.unknown_ratio_max):
            reason = "reject_extension_hit_unknown"
        elif wall_ratio > float(config.wall_mid_ratio_max):
            reason = "reject_extension_crosses_mid_wall"
        elif str(hit_type) not in {"real_wall", "virtual_door", "virtual_neck"}:
            reason = "reject_extension_no_wall_or_virtual_door_hit"
    confidence = float(
        np.clip(
            0.45 * float(line.confidence)
            + 0.35 * float(free_ratio)
            + 0.10 * (1.0 - min(1.0, unknown_ratio))
            + 0.10 * (1.0 if str(hit_type) in {"real_wall", "virtual_door", "virtual_neck"} else 0.0),
            0.0,
            1.0,
        )
    )
    return LineExtensionHit(
        extension_id=int(extension_id),
        source_line_id=int(line.line_id),
        source_endpoint=str(endpoint),
        pass_id=int(pass_id),
        p_start_rc=np.asarray(p_start, dtype=np.float32),
        p_hit_rc=np.asarray(p_hit, dtype=np.float32),
        theta=float(line.theta),
        length_m=length_m,
        interior_free_ratio=float(free_ratio),
        interior_unknown_ratio=float(unknown_ratio),
        interior_wall_ratio=float(wall_ratio),
        hit_type=str(hit_type),
        hit_candidate_id=hit_candidate_id,
        confidence=confidence,
        reject_reason=reason,
        debug={
            **dict(debug),
            "source_line_confidence": float(line.confidence),
            "source_line_length_m": float(line.length_m),
            "sampled_cell_count": int(len(sampled)),
        },
    )


def _extension_start_and_direction(line: FilteredWallLine, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.asarray(line.p0_rc, dtype=np.float32)
    p1 = np.asarray(line.p1_rc, dtype=np.float32)
    if str(endpoint) == "p0":
        delta = p0 - p1
        start = p0
    else:
        delta = p1 - p0
        start = p1
    if abs(float(delta[0])) >= abs(float(delta[1])):
        direction = np.asarray([1.0 if float(delta[0]) >= 0.0 else -1.0, 0.0], dtype=np.float32)
    else:
        direction = np.asarray([0.0, 1.0 if float(delta[1]) >= 0.0 else -1.0], dtype=np.float32)
    return start.astype(np.float32), direction


def _sample_ratio(cells: Sequence[tuple[int, int]], mask: np.ndarray) -> float:
    if not cells:
        return 0.0
    arr = np.asarray(mask, dtype=bool)
    count = 0
    valid = 0
    for r, c in cells:
        if 0 <= int(r) < arr.shape[0] and 0 <= int(c) < arr.shape[1]:
            valid += 1
            if bool(arr[int(r), int(c)]):
                count += 1
    return float(count) / float(max(1, valid))


def _snap_hit_point_to_target(
    rc: np.ndarray,
    target_mask: np.ndarray | None,
    direction: np.ndarray,
    *,
    radius_cells: int,
) -> np.ndarray:
    if target_mask is None:
        return np.asarray(rc, dtype=np.float32)
    target = np.asarray(target_mask, dtype=bool)
    point = np.asarray(rc, dtype=np.int32)
    if not _inside(point, target.shape):
        return point.astype(np.float32)
    if bool(target[int(point[0]), int(point[1])]):
        return point.astype(np.float32)
    radius = max(0, int(radius_cells))
    r0, r1 = max(0, int(point[0]) - radius), min(target.shape[0], int(point[0]) + radius + 1)
    c0, c1 = max(0, int(point[1]) - radius), min(target.shape[1], int(point[1]) + radius + 1)
    rows, cols = np.nonzero(target[r0:r1, c0:c1])
    if rows.size == 0:
        return point.astype(np.float32)
    coords = np.stack([rows + r0, cols + c0], axis=1).astype(np.int32)
    delta = coords.astype(np.float32) - point.astype(np.float32)
    unit = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-6:
        unit = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        unit = unit / norm
    projection = delta @ unit
    forward = projection >= -1e-6
    if np.any(forward):
        coords = coords[forward]
        delta = delta[forward]
        projection = projection[forward]
    if abs(float(unit[0])) >= abs(float(unit[1])):
        lateral = np.abs(delta[:, 1])
    else:
        lateral = np.abs(delta[:, 0])
    dist2 = np.sum(delta * delta, axis=1)
    order = np.lexsort((dist2, np.abs(projection), lateral))
    return coords[int(order[0])].astype(np.float32)


def _extension_reason_counts(hits: Sequence[LineExtensionHit]) -> dict:
    out: dict[str, int] = {}
    for hit in hits:
        reason = str(hit.reject_reason or "accepted")
        out[reason] = out.get(reason, 0) + 1
    return out


def _extension_can_seed_intersection(hit: LineExtensionHit) -> bool:
    reason = "" if hit.reject_reason is None else str(hit.reject_reason)
    if reason in {
        "reject_extension_blocked_by_near_wall",
        "reject_extension_crosses_mid_wall",
        "reject_extension_not_enough_free",
        "reject_extension_length_out_of_door_range",
        "reject_low_line_extension_confidence",
    }:
        return False
    if str(hit.hit_type) == "blocked_wall":
        return False
    return bool(float(hit.length_m) > 0.0)


def _extension_axis_info(hit: LineExtensionHit, shape: tuple[int, int]) -> tuple[str, int, int, int] | None:
    p0 = np.rint(np.asarray(hit.p_start_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(hit.p_hit_rc, dtype=np.float32)).astype(np.int32)
    if not _inside(p0, shape) or not _inside(p1, shape):
        return None
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    if abs(dc) >= abs(dr):
        fixed = int(round(float((int(p0[0]) + int(p1[0])) / 2.0)))
        return "horizontal", fixed, int(min(p0[1], p1[1])), int(max(p0[1], p1[1]))
    fixed = int(round(float((int(p0[1]) + int(p1[1])) / 2.0)))
    return "vertical", fixed, int(min(p0[0], p1[0])), int(max(p0[0], p1[0]))


def _ordered_line_cells(p0_rc: np.ndarray, p1_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    p0 = np.rint(np.asarray(p0_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(p1_rc, dtype=np.float32)).astype(np.int32)
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    steps = max(abs(dr), abs(dc))
    if steps <= 0:
        rows = np.asarray([int(p0[0])], dtype=np.int32)
        cols = np.asarray([int(p0[1])], dtype=np.int32)
    else:
        rows = np.rint(np.linspace(int(p0[0]), int(p1[0]), steps + 1)).astype(np.int32)
        cols = np.rint(np.linspace(int(p0[1]), int(p1[1]), steps + 1)).astype(np.int32)
    inside = (rows >= 0) & (rows < int(shape[0])) & (cols >= 0) & (cols < int(shape[1]))
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    if len(coords) <= 1:
        return coords.astype(np.int32)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _candidate_ordered_path_cells(candidate: SeparatorCandidate, shape: tuple[int, int]) -> np.ndarray:
    raw = candidate.debug.get("l_corner_cut_cells_rc", candidate.debug.get("mask_cells_rc"))
    if raw is not None:
        try:
            cells = np.asarray(raw, dtype=np.int32)
        except Exception:
            cells = np.empty((0, 2), dtype=np.int32)
        if cells.ndim == 2 and cells.shape[1] == 2 and len(cells) > 0:
            inside = (
                (cells[:, 0] >= 0)
                & (cells[:, 0] < int(shape[0]))
                & (cells[:, 1] >= 0)
                & (cells[:, 1] < int(shape[1]))
            )
            cells = cells[inside]
            if len(cells) > 0:
                keep = np.ones(len(cells), dtype=bool)
                keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
                return cells[keep].astype(np.int32)
    return _ordered_line_cells(candidate.p0_rc, candidate.p1_rc, shape)


def _endpoint_middle_hit(
    endpoint_rc: np.ndarray,
    other_path: np.ndarray,
    *,
    min_ratio: float,
    max_ratio: float,
    tolerance_cells: int,
) -> dict | None:
    path = np.asarray(other_path, dtype=np.int32)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        return None
    point = np.rint(np.asarray(endpoint_rc, dtype=np.float32)).astype(np.int32)
    delta = path.astype(np.int32) - point[None, :]
    cheb = np.max(np.abs(delta), axis=1)
    best_idx = int(np.argmin(cheb))
    best_dist = int(cheb[best_idx])
    if best_dist > int(tolerance_cells):
        return None
    ratio = float(best_idx) / float(max(1, len(path) - 1))
    if ratio < float(min_ratio) or ratio > float(max_ratio):
        return None
    return {
        "other_path_index": int(best_idx),
        "other_path_cell_count": int(len(path)),
        "other_path_ratio": float(ratio),
        "other_path_cell_rc": [int(path[best_idx, 0]), int(path[best_idx, 1])],
        "endpoint_to_other_chebyshev_cells": int(best_dist),
    }


def _candidate_from_extension_intersection(
    hit: LineExtensionHit,
    *,
    other_hit: LineExtensionHit,
    intersection_rc: np.ndarray,
    free_clean: np.ndarray,
    unknown_clean: np.ndarray,
    resolution_m: float,
    line_config: LineExtensionConfig,
    door_config: DoorNeckConfig,
    candidate_id: int,
) -> SeparatorCandidate | None:
    free = np.asarray(free_clean, dtype=bool)
    unknown = np.asarray(unknown_clean, dtype=bool)
    start = np.rint(np.asarray(hit.p_start_rc, dtype=np.float32)).astype(np.int32)
    end = np.rint(np.asarray(intersection_rc, dtype=np.float32)).astype(np.int32)
    if not _inside(start, free.shape) or not _inside(end, free.shape):
        return None
    cells = _ordered_line_cells(start.astype(np.float32), end.astype(np.float32), free.shape)
    if len(cells) <= 1:
        return None
    interior = [(int(r), int(c)) for r, c in cells[1:]]
    length_m = float((len(cells) - 1) * float(resolution_m))
    if length_m < float(door_config.min_width_m) or length_m > float(door_config.max_width_m):
        return None
    free_ratio = _sample_ratio(interior, free)
    unknown_ratio = _sample_ratio(interior, unknown)
    if free_ratio < float(line_config.free_ratio_min) or unknown_ratio > float(line_config.unknown_ratio_max):
        return None
    theta = 0.0 if int(start[0]) == int(end[0]) else float(np.pi / 2.0)
    confidence = float(
        np.clip(
            0.35 * float(hit.confidence)
            + 0.25 * float(other_hit.confidence)
            + 0.30 * float(free_ratio)
            + 0.10 * (1.0 - min(1.0, unknown_ratio)),
            0.0,
            1.0,
        )
    )
    if confidence < float(door_config.min_confidence):
        return None
    return SeparatorCandidate(
        candidate_id=int(candidate_id),
        kind="extension_intersection_cut",
        p0_rc=start.astype(np.float32),
        p1_rc=end.astype(np.float32),
        theta=float(theta),
        length_m=float(length_m),
        confidence=float(confidence),
        source_segment_ids=[int(hit.source_line_id), int(other_hit.source_line_id)],
        wall_support_score=float(max(float(hit.confidence), float(other_hit.confidence))),
        free_gap_score=float(free_ratio),
        doorway_score=float(free_ratio),
        debug={
            "candidate_source": "extension_intersection",
            "kind_detail": "extension_intersection_cut",
            "pass_id": int(max(int(hit.pass_id), int(other_hit.pass_id))),
            "source_extension_ids": [int(hit.extension_id), int(other_hit.extension_id)],
            "source_line_ids": [int(hit.source_line_id), int(other_hit.source_line_id)],
            "hit_types": ["virtual_neck"],
            "intersection_rc": [int(end[0]), int(end[1])],
            "width_m": float(length_m),
            "neck_score": float(free_ratio),
            "doorway_score": float(free_ratio),
            "intersection_free_ratio": float(free_ratio),
            "intersection_unknown_ratio": float(unknown_ratio),
            "primary_extension": hit.to_dict(),
            "cross_extension": other_hit.to_dict(),
        },
    )


def _candidate_points(axis: str, line: int, gap_start: int, gap_end: int) -> tuple[np.ndarray, np.ndarray, float]:
    if axis == "horizontal":
        return (
            np.asarray([line, gap_start], dtype=np.float32),
            np.asarray([line, gap_end], dtype=np.float32),
            0.0,
        )
    return (
        np.asarray([gap_start, line], dtype=np.float32),
        np.asarray([gap_end, line], dtype=np.float32),
        float(np.pi / 2.0),
    )


def _filtered_line_axis_info(line: FilteredWallLine) -> dict | None:
    p0 = np.rint(np.asarray(line.p0_rc, dtype=np.float32)).astype(np.int32)
    p1 = np.rint(np.asarray(line.p1_rc, dtype=np.float32)).astype(np.int32)
    dr = int(p1[0] - p0[0])
    dc = int(p1[1] - p0[1])
    if abs(dc) >= abs(dr):
        fixed = int(round(float((int(p0[0]) + int(p1[0])) / 2.0)))
        start = int(min(int(p0[1]), int(p1[1])))
        end = int(max(int(p0[1]), int(p1[1])))
        return {
            "axis": "horizontal",
            "line": int(fixed),
            "start": int(start),
            "end": int(end),
            "endpoints": [
                ("start", np.asarray([fixed, start], dtype=np.int32)),
                ("end", np.asarray([fixed, end], dtype=np.int32)),
            ],
        }
    fixed = int(round(float((int(p0[1]) + int(p1[1])) / 2.0)))
    start = int(min(int(p0[0]), int(p1[0])))
    end = int(max(int(p0[0]), int(p1[0])))
    return {
        "axis": "vertical",
        "line": int(fixed),
        "start": int(start),
        "end": int(end),
        "endpoints": [
            ("start", np.asarray([start, fixed], dtype=np.int32)),
            ("end", np.asarray([end, fixed], dtype=np.int32)),
        ],
    }


def _l_corner_path_cells(
    *,
    horizontal_line: int,
    horizontal_endpoint_col: int,
    vertical_line: int,
    vertical_endpoint_row: int,
    shape: tuple[int, int],
) -> np.ndarray:
    cells: list[tuple[int, int]] = []
    h_row = int(horizontal_line)
    h_col = int(horizontal_endpoint_col)
    v_col = int(vertical_line)
    v_row = int(vertical_endpoint_row)
    if h_col < v_col:
        h_cols = range(h_col + 1, v_col + 1)
    else:
        h_cols = range(h_col - 1, v_col - 1, -1)
    for col in h_cols:
        cells.append((h_row, int(col)))
    if h_row < v_row:
        v_rows = range(h_row + 1, v_row)
    else:
        v_rows = range(h_row - 1, v_row, -1)
    for row in v_rows:
        cells.append((int(row), v_col))
    if not cells:
        return np.empty((0, 2), dtype=np.int32)
    rows = np.asarray([item[0] for item in cells], dtype=np.int32)
    cols = np.asarray([item[1] for item in cells], dtype=np.int32)
    inside = (rows >= 0) & (rows < int(shape[0])) & (cols >= 0) & (cols < int(shape[1]))
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    if len(coords) <= 1:
        return coords.astype(np.int32)
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _line_stats(p0: np.ndarray, p1: np.ndarray, free: np.ndarray, wall: np.ndarray, unknown: np.ndarray) -> dict:
    line = rasterize_line(p0, p1, free.shape)
    cells = max(1, int(np.count_nonzero(line)))
    return {
        "free_ratio": float(np.count_nonzero(line & free)) / float(cells),
        "unknown_ratio": float(np.count_nonzero(line & unknown)) / float(cells),
        "wall_ratio": float(np.count_nonzero(line & wall)) / float(cells),
        "cell_count": int(cells),
    }


def _candidate_kinds_for_gap(
    *,
    length_m: float,
    free_ratio: float,
    unknown_ratio: float,
    wall_support: float,
    physical: PhysicalWallCompletionConfig,
    missed: MissedScanGapClosureConfig,
    short_unknown: ShortUnknownGapClosureConfig,
    doorway: DoorwayVirtualCutConfig,
) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    shorter_than_door = bool(length_m < float(doorway.min_width_m))
    gap_tolerance_m = 0.051
    if bool(physical.enabled) and length_m <= float(physical.max_gap_m) + gap_tolerance_m and wall_support >= float(physical.min_endpoint_support):
        if shorter_than_door or free_ratio <= 0.80 or unknown_ratio >= 0.15:
            confidence = float(np.clip(0.75 + 0.2 * wall_support + 0.05 * unknown_ratio, 0.0, 1.0))
            if confidence >= float(physical.min_confidence):
                out.append(("physical_wall_completion", confidence, 0.0))
    if (
        bool(missed.enabled)
        and length_m <= float(missed.max_gap_m) + gap_tolerance_m
        and wall_support >= float(missed.min_wall_endpoint_support)
        and (shorter_than_door or free_ratio <= float(missed.max_free_ratio))
    ):
        if unknown_ratio <= float(missed.max_unknown_ratio):
            out.append(("missed_scan_gap_closure", float(np.clip(0.65 + 0.25 * wall_support + 0.1 * (1.0 - free_ratio), 0.0, 1.0)), 0.0))
    if (
        bool(short_unknown.enabled)
        and length_m <= float(short_unknown.max_gap_m)
        and wall_support >= float(short_unknown.min_wall_endpoint_support)
        and unknown_ratio >= float(short_unknown.min_unknown_ratio)
    ):
        out.append(("short_unknown_gap_closure", float(np.clip(0.55 + 0.35 * unknown_ratio + 0.1 * wall_support, 0.0, 1.0)), 0.0))
    if (
        bool(doorway.enabled)
        and float(doorway.min_width_m) <= length_m <= float(doorway.max_width_m)
        and wall_support >= float(doorway.min_wall_endpoint_support)
        and (not bool(doorway.require_free_gap) or free_ratio >= float(doorway.free_gap_min_ratio))
    ):
        preferred_min = float(doorway.preferred_min_width_m)
        preferred_max = float(doorway.preferred_max_width_m)
        if preferred_min <= float(length_m) <= preferred_max:
            width_score = 1.0
        else:
            mid = 0.5 * (preferred_min + preferred_max)
            width_score = 1.0 - min(1.0, abs(float(length_m) - mid) / max(mid, 1e-6))
        out.append(("doorway_virtual_cut", float(np.clip(0.60 + 0.30 * free_ratio + 0.10 * width_score, 0.0, 1.0)), free_ratio))
    return out


def _single_sided_candidates(
    runs: Sequence[SnappedWallRun],
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    *,
    resolution_m: float,
    config: SingleSidedWallExtensionConfig,
    start_id: int,
    seen: set[tuple[str, str, int, int, int]],
) -> tuple[list[SeparatorCandidate], int]:
    max_steps = max(1, int(round(float(config.max_extension_m) / max(float(resolution_m), 1e-9))))
    candidates: list[SeparatorCandidate] = []
    cid = int(start_id)
    for run in runs:
        for side, sign in (("start", -1), ("end", 1)):
            if run.axis == "horizontal":
                base = np.asarray([run.line, run.start if side == "start" else run.end], dtype=np.int32)
                direction = np.asarray([0, sign], dtype=np.int32)
            else:
                base = np.asarray([run.start if side == "start" else run.end, run.line], dtype=np.int32)
                direction = np.asarray([sign, 0], dtype=np.int32)
            endpoint, anchor = _extend_endpoint_to_anchor(base, direction, free, wall, unknown, np.zeros_like(free, dtype=bool), max_steps)
            if bool(config.reject_if_endpoint_none) and str(anchor.get("type", "none")) == "none":
                continue
            if bool(config.require_anchor) and not bool(anchor["anchored"]):
                continue
            if int(anchor["steps"]) <= 1:
                continue
            p0 = base.astype(np.float32)
            p1 = endpoint.astype(np.float32)
            stats = _line_stats(p0, p1, free, wall, unknown)
            if stats["free_ratio"] > float(config.max_open_free_ratio):
                continue
            key = ("single_sided_wall_extension", str(run.axis), int(run.line), int(base[0] * free.shape[1] + base[1]), int(endpoint[0] * free.shape[1] + endpoint[1]))
            if key in seen:
                continue
            seen.add(key)
            theta = 0.0 if run.axis == "horizontal" else float(np.pi / 2.0)
            candidates.append(
                SeparatorCandidate(
                    candidate_id=cid,
                    kind="single_sided_wall_extension",
                    p0_rc=p0,
                    p1_rc=p1,
                    theta=theta,
                    length_m=float(max(1, int(anchor["steps"])) * float(resolution_m)),
                    confidence=float(np.clip(0.45 + 0.45 * float(run.confidence), 0.0, 1.0)),
                    source_segment_ids=[int(v) for v in run.source_segment_ids],
                    wall_support_score=float(run.confidence),
                    free_gap_score=float(stats["free_ratio"]),
                    debug={
                        "axis": str(run.axis),
                        "line": int(run.line),
                        "gap_m": float(max(1, int(anchor["steps"])) * float(resolution_m)),
                        "free_ratio": float(stats["free_ratio"]),
                        "unknown_ratio": float(stats["unknown_ratio"]),
                        "source_run_ids": [int(run.run_id)],
                        "candidate_source": "single_sided_wall_extension",
                        "extension_anchor_type": str(anchor["type"]),
                    },
                )
            )
            cid += 1
    return candidates, cid


def _unit_grid_direction(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    delta = np.asarray(p1, dtype=np.int32) - np.asarray(p0, dtype=np.int32)
    if abs(int(delta[0])) >= abs(int(delta[1])):
        return np.asarray([1 if int(delta[0]) >= 0 else -1, 0], dtype=np.int32)
    return np.asarray([0, 1 if int(delta[1]) >= 0 else -1], dtype=np.int32)


def _extend_endpoint_to_anchor(
    start: np.ndarray,
    direction: np.ndarray,
    free: np.ndarray,
    wall: np.ndarray,
    unknown: np.ndarray,
    existing: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, dict]:
    point = np.asarray(start, dtype=np.int32).copy()
    anchor = _anchor_type(point, free, wall, unknown, existing)
    if anchor != "":
        return point, {"anchored": True, "type": anchor, "steps": 0}
    last = point.copy()
    for step in range(1, int(max_steps) + 1):
        candidate = point + np.asarray(direction, dtype=np.int32) * int(step)
        if not _inside(candidate, free.shape):
            return last, {"anchored": True, "type": "map_boundary", "steps": int(step - 1)}
        last = candidate
        anchor = _anchor_type(candidate, free, wall, unknown, existing)
        if anchor != "":
            return candidate, {"anchored": True, "type": anchor, "steps": int(step)}
    return last, {"anchored": False, "type": "none", "steps": int(max_steps)}


def _anchor_type(point: np.ndarray, free: np.ndarray, wall: np.ndarray, unknown: np.ndarray, existing: np.ndarray) -> str:
    r, c = int(point[0]), int(point[1])
    if not _inside(point, free.shape):
        return "map_boundary"
    if r == 0 or c == 0 or r == free.shape[0] - 1 or c == free.shape[1] - 1:
        return "map_boundary"
    r0, r1 = max(0, r - 1), min(free.shape[0], r + 2)
    c0, c1 = max(0, c - 1), min(free.shape[1], c + 2)
    if np.any(existing[r0:r1, c0:c1]):
        return "other_separator"
    if np.any(wall[r0:r1, c0:c1]):
        endpoint_like = int(np.count_nonzero(wall[r0:r1, c0:c1])) <= 2
        return "wall_endpoint" if endpoint_like else "wall"
    if np.any(unknown[r0:r1, c0:c1]):
        return "unknown_frontier"
    if not bool(free[r, c]) or np.any(~free[r0:r1, c0:c1]):
        return "room_boundary"
    return ""


def _anchor_score(anchor_type: str) -> float:
    return float(
        {
            "structural_wall": 1.0,
            "wall": 1.0,
            "wall_endpoint": 0.95,
            "accepted_separator": 0.90,
            "other_separator": 0.90,
            "extension_intersection": 0.80,
            "navigation_obstacle_edge": 0.55,
            "map_boundary": 0.75,
            "room_boundary": 0.75,
            "unknown_frontier": 0.40,
            "none": 0.0,
            "": 0.0,
            # Backward-compatible aliases from v1 debug payloads.
            "touch_wall": 1.0,
            "touch_existing_separator": 0.90,
            "touch_unknown_boundary": 0.40,
            "touch_map_boundary": 0.75,
            "touch_room_boundary": 0.75,
        }.get(str(anchor_type), 0.0)
    )


def score_separator_candidate(
    candidate: SeparatorCandidate,
    config: SeparatorScoringConfig | Mapping[str, object] | None = None,
) -> float:
    cfg = config if isinstance(config, SeparatorScoringConfig) else SeparatorScoringConfig.from_mapping(config)
    score = (
        float(cfg.wall_support_weight) * float(candidate.wall_support_score)
        + float(cfg.doorway_score_weight) * float(candidate.doorway_score)
        + float(cfg.visibility_drop_weight) * float(candidate.visibility_drop_score)
        + float(cfg.topology_gain_weight) * float(candidate.topology_gain_score)
        + float(cfg.temporal_weight) * float(candidate.temporal_score)
        + float(cfg.anchor_weight) * float(candidate.anchor_score)
        + float(cfg.confidence_weight) * float(candidate.confidence)
        - float(cfg.open_space_penalty_weight) * float(candidate.open_space_penalty)
        - float(cfg.frontier_penalty_weight) * float(candidate.frontier_penalty)
    )
    candidate.final_score = float(np.clip(score, 0.0, 1.0))
    return float(candidate.final_score)


def _inside(point: np.ndarray, shape: tuple[int, int]) -> bool:
    return 0 <= int(point[0]) < int(shape[0]) and 0 <= int(point[1]) < int(shape[1])


def separator_mask_for_candidate(candidate: SeparatorCandidate, shape: tuple[int, int], thickness_cells: int = 0) -> np.ndarray:
    mask = candidate.mask(shape)
    if int(thickness_cells) > 0:
        mask = dilate(mask, int(thickness_cells))
    return mask.astype(bool)


def _candidate_mask_from_debug(debug: Mapping[str, object], shape: tuple[int, int]) -> np.ndarray | None:
    raw = debug.get("l_corner_cut_cells_rc", debug.get("mask_cells_rc"))
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] <= 0:
        return None
    rows = arr[:, 0]
    cols = arr[:, 1]
    inside = (rows >= 0) & (rows < int(shape[0])) & (cols >= 0) & (cols < int(shape[1]))
    if not np.any(inside):
        return np.zeros(shape, dtype=bool)
    mask = np.zeros(shape, dtype=bool)
    mask[rows[inside], cols[inside]] = True
    return mask.astype(bool)


def _kind_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        out[str(candidate.kind)] = out.get(str(candidate.kind), 0) + 1
    return out


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
