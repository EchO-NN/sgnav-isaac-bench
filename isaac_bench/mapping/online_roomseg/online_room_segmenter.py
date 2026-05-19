from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import RoomMask, RoomProposalState, RoomSegmentationConfig
from isaac_bench.mapping.room_segmentation import _proposal_masks_debug, _room_from_mask  # reuse canonical metadata/stable room shape
from isaac_bench.mapping.vertical_profile import VerticalProfileMap

from .corridor import CorridorConfig, CorridorRoomNeckCutConfig, generate_corridor_room_neck_candidates
from .debug_viz import save_online_roomseg_debug
from .evidence_maps import FreeCleanConfig, WallCandidateConfig, build_evidence_maps
from .separator_candidates import (
    DoorwayVirtualCutConfig,
    PhysicalWallCompletionConfig,
    SeparatorCandidate,
    generate_wall_gap_candidates,
)
from .topology_tests import TopologyTestConfig, greedily_select_separators
from .utils import relabel_compact
from .wall_lines import LineWallsConfig, extract_line_supported_walls, line_supported_wall_mask


ONLINE_ROSE_STYLE_BACKEND = "online_rose_style_v1"
ONLINE_ROSE_STYLE_CONTEXT = "online_rose_style_v1_vlm"


@dataclass
class OnlineRoseStyleConfig:
    enabled: bool = True
    backend: str = ONLINE_ROSE_STYLE_BACKEND
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    z_min_m: float = 0.20
    z_max_m: float = 2.00
    min_free_rays: int = 1
    min_observed_rays: int = 1
    min_observed_free_cells: int = 20
    min_room_area_m2: float = 1.0
    debug_dir: str = "debug/online_roomseg"
    free_clean: FreeCleanConfig = field(default_factory=FreeCleanConfig)
    wall_candidate: WallCandidateConfig = field(default_factory=WallCandidateConfig)
    line_walls: LineWallsConfig = field(default_factory=LineWallsConfig)
    physical_wall_completion: PhysicalWallCompletionConfig = field(default_factory=PhysicalWallCompletionConfig)
    doorway_virtual_cut: DoorwayVirtualCutConfig = field(default_factory=DoorwayVirtualCutConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    corridor_room_neck_cut: CorridorRoomNeckCutConfig = field(default_factory=CorridorRoomNeckCutConfig)
    topology_test: TopologyTestConfig = field(default_factory=TopologyTestConfig)
    debug: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "OnlineRoseStyleConfig":
        raw_root = dict(data or {})
        raw = dict(raw_root.get("online_roomseg", {}) or {})
        for key in ("enabled", "backend"):
            if key in raw_root and key not in raw:
                raw[key] = raw_root[key]
        vertical_or_free = dict(raw_root.get("vertical_or_free", {}) or {})
        if vertical_or_free:
            raw.setdefault("z_min_m", vertical_or_free.get("z_min_m", 0.20))
            raw.setdefault("z_max_m", vertical_or_free.get("z_max_m", 2.00))
            raw.setdefault("min_free_rays", vertical_or_free.get("min_free_rays", 1))
            raw.setdefault("min_observed_rays", vertical_or_free.get("min_observed_rays", 1))
        debug_layers = dict(raw_root.get("debug_layers", {}) or {})
        debug = dict(raw.get("debug", {}) or {})
        if "enabled" in debug_layers:
            debug.setdefault("save_layers", bool(debug_layers.get("enabled")))
            debug.setdefault("save_candidate_json", bool(debug_layers.get("enabled")))
        if "output_dir" in debug_layers:
            raw.setdefault("debug_dir", str(debug_layers.get("output_dir")))
        raw["debug"] = debug
        raw.update({key: value for key, value in overrides.items() if value is not None})
        nested = {
            "free_clean": FreeCleanConfig.from_mapping(raw.get("free_clean")),
            "wall_candidate": WallCandidateConfig.from_mapping(raw.get("wall_candidate")),
            "line_walls": LineWallsConfig.from_mapping(raw.get("line_walls")),
            "physical_wall_completion": PhysicalWallCompletionConfig.from_mapping(raw.get("physical_wall_completion")),
            "doorway_virtual_cut": DoorwayVirtualCutConfig.from_mapping(raw.get("doorway_virtual_cut")),
            "corridor": CorridorConfig.from_mapping(raw.get("corridor")),
            "corridor_room_neck_cut": CorridorRoomNeckCutConfig.from_mapping(raw.get("corridor_room_neck_cut")),
            "topology_test": TopologyTestConfig.from_mapping(raw.get("topology_test")),
        }
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key not in nested}
        base.update(nested)
        return cls(**base)

    def room_config(self) -> RoomSegmentationConfig:
        return RoomSegmentationConfig(
            algorithm=ONLINE_ROSE_STYLE_BACKEND,
            source_grid="vertical_profile_free_0p2_2p0",
            proposal_mode=ONLINE_ROSE_STYLE_BACKEND,
            finalization_mode="no_merge",
            min_observed_free_cells=int(self.min_observed_free_cells),
            min_room_area_m2=float(self.min_room_area_m2),
            resolution_m=float(self.resolution_m),
            map_info=self.map_info,
        )


@dataclass
class OnlineRoseStyleResult:
    room_label_map: np.ndarray
    separator_map: np.ndarray
    accepted_candidates: list[SeparatorCandidate]
    rejected_candidates: list[SeparatorCandidate]
    layers: dict[str, np.ndarray]
    debug: dict


class OnlineRoseStyleRoomSegmenter:
    context_source = ONLINE_ROSE_STYLE_CONTEXT

    def __init__(self, config: OnlineRoseStyleConfig | Mapping[str, object] | None = None, map_info: MapInfo | None = None):
        if isinstance(config, OnlineRoseStyleConfig):
            self.config = config
        else:
            self.config = OnlineRoseStyleConfig.from_mapping(config or {}, map_info=map_info)
        if map_info is not None:
            self.config.map_info = map_info
            self.config.resolution_m = float(map_info.resolution_m)
        self.last_debug: dict = {}
        self.last_result: OnlineRoseStyleResult | None = None

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Iterable[object] | None = None,
        vertical_profile: VerticalProfileMap | None = None,
        roomseg_static_structural_occupied: np.ndarray | None = None,
        roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    ) -> list[RoomMask]:
        proposals, state = self.build_proposals(
            occupancy_map,
            observed_free_mask,
            obstacle_mask,
            unknown_mask,
            step=step,
            object_memory=object_memory,
            vertical_profile=vertical_profile,
            roomseg_static_structural_occupied=roomseg_static_structural_occupied,
            roomseg_ray_evidence=roomseg_ray_evidence,
        )
        _ = proposals
        return self.finalize_proposals(state, proposal_semantic_labels=None)

    def build_proposals(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Iterable[object] | None = None,
        vertical_profile: VerticalProfileMap | None = None,
        roomseg_static_structural_occupied: np.ndarray | None = None,
        roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[list[RoomMask], RoomProposalState]:
        result = run_online_rose_style_roomseg(
            occupancy_map=occupancy_map,
            observed_free_mask=observed_free_mask,
            obstacle_mask=obstacle_mask,
            unknown_mask=unknown_mask,
            vertical_profile=vertical_profile,
            config=self.config,
            step=int(step),
        )
        self.last_result = result
        self.last_debug = dict(result.debug)
        room_cfg = self.config.room_config()
        rooms = _rooms_from_labels(result.room_label_map, result.layers["unknown_clean"], room_cfg, int(step), result.debug)
        state = RoomProposalState(
            proposal_labels=np.asarray(result.room_label_map, dtype=np.int32),
            structural_free_mask=np.asarray(result.layers["free_clean"], dtype=bool),
            structural_obstacle_mask=np.asarray(result.separator_map, dtype=bool),
            unknown_mask=np.asarray(result.layers["unknown_clean"], dtype=bool),
            distance_m=ndimage.distance_transform_edt(np.asarray(result.layers["free_clean"], dtype=bool)) * float(self.config.resolution_m),
            step=int(step),
            debug=dict(result.debug),
        )
        return rooms, state

    def finalize_proposals(
        self,
        proposal_state: RoomProposalState,
        proposal_semantic_labels: Mapping[int, object] | None = None,
    ) -> list[RoomMask]:
        _ = proposal_semantic_labels
        labels = np.asarray(proposal_state.proposal_labels, dtype=np.int32)
        rooms = _rooms_from_labels(labels, proposal_state.unknown_mask, self.config.room_config(), int(proposal_state.step), proposal_state.debug)
        self.last_debug = {
            **dict(proposal_state.debug),
            "room_count": int(len(rooms)),
            "rooms": _proposal_masks_debug(labels),
        }
        return rooms


def run_online_rose_style_roomseg(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    vertical_profile: VerticalProfileMap | None,
    config: OnlineRoseStyleConfig,
    step: int = 0,
) -> OnlineRoseStyleResult:
    evidence = build_evidence_maps(
        occupancy_map=occupancy_map,
        observed_free_mask=observed_free_mask,
        obstacle_mask=obstacle_mask,
        unknown_mask=unknown_mask,
        vertical_profile=vertical_profile,
        resolution_m=float(config.resolution_m),
        free_clean_config=config.free_clean,
        wall_candidate_config=config.wall_candidate,
        z_min_m=float(config.z_min_m),
        z_max_m=float(config.z_max_m),
        min_free_rays=int(config.min_free_rays),
        min_observed_rays=int(config.min_observed_rays),
    )
    segments, wall_debug = extract_line_supported_walls(
        evidence.wall_candidate_clean,
        resolution_m=float(config.resolution_m),
        config=config.line_walls,
    )
    line_wall_map = line_supported_wall_mask(segments, evidence.free_clean.shape)
    wall_candidates, wall_gap_debug = generate_wall_gap_candidates(
        segments,
        free_clean=evidence.free_clean,
        wall_candidate_clean=evidence.wall_candidate_clean,
        unknown_clean=evidence.unknown_clean,
        resolution_m=float(config.resolution_m),
        physical_config=config.physical_wall_completion,
        doorway_config=config.doorway_virtual_cut,
        start_id=1,
    )
    corridor_candidates, corridor_debug = generate_corridor_room_neck_candidates(
        evidence.free_clean,
        resolution_m=float(config.resolution_m),
        corridor_config=config.corridor,
        neck_config=config.corridor_room_neck_cut,
        start_id=len(wall_candidates) + 1,
    )
    candidates = [*wall_candidates, *corridor_candidates]
    candidate_layers = _candidate_layers(candidates, evidence.free_clean.shape)
    before_labels, _ = ndimage.label(evidence.free_clean, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    before_labels = relabel_compact(before_labels)
    accepted, rejected, separator_map, final_labels, topology_debug = greedily_select_separators(
        candidates,
        free_clean=evidence.free_clean,
        unknown_clean=evidence.unknown_clean,
        resolution_m=float(config.resolution_m),
        config=config.topology_test,
    )
    rejected_map = _rasterize_many(rejected, evidence.free_clean.shape)
    layers = {
        "vertical_free_raw": evidence.vertical_free_raw,
        "vertical_occupied_raw": evidence.vertical_occupied_raw,
        "vertical_observed_raw": evidence.vertical_observed_raw,
        "vertical_unknown_raw": evidence.vertical_unknown_raw,
        "free_clean": evidence.free_clean,
        "wall_candidate_clean": evidence.wall_candidate_clean,
        "unknown_clean": evidence.unknown_clean,
        "line_supported_walls": line_wall_map,
        "physical_wall_completion_candidates": candidate_layers["physical_wall_completion"],
        "doorway_virtual_cut_candidates": candidate_layers["doorway_virtual_cut"],
        "corridor_skeleton": np.asarray(corridor_debug.get("corridor_skeleton", np.zeros_like(evidence.free_clean)), dtype=bool),
        "corridor_candidate_map": np.asarray(corridor_debug.get("corridor_candidate_map", np.zeros_like(evidence.free_clean)), dtype=bool),
        "corridor_room_neck_cut_candidates": candidate_layers["corridor_room_neck_cut"],
        "accepted_separators": separator_map,
        "rejected_separators": rejected_map,
        "room_labels_before_separators": before_labels,
        "room_labels_after_separators": final_labels,
        "final_room_labels": final_labels,
    }
    report = {
        "step": int(step),
        "backend": ONLINE_ROSE_STYLE_BACKEND,
        "algorithm": ONLINE_ROSE_STYLE_BACKEND,
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "candidates": [candidate.to_dict() for candidate in [*accepted, *rejected]],
    }
    debug = {
        "backend": ONLINE_ROSE_STYLE_BACKEND,
        "actual_backend": ONLINE_ROSE_STYLE_BACKEND,
        "source_backend": ONLINE_ROSE_STYLE_BACKEND,
        "roomseg_backend": ONLINE_ROSE_STYLE_BACKEND,
        "algorithm": ONLINE_ROSE_STYLE_BACKEND,
        "source": ONLINE_ROSE_STYLE_BACKEND,
        "context_source": ONLINE_ROSE_STYLE_CONTEXT,
        "room_map_mode": ONLINE_ROSE_STYLE_CONTEXT,
        "strict_fallback_used": False,
        "silent_fallback_used": False,
        "legacy_style_used": False,
        "navigation_obstacle_written": False,
        "step": int(step),
        "resolution_m": float(config.resolution_m),
        "final_room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
        "room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
        "separator_report": report,
        "accepted_separators": [candidate.to_dict() for candidate in accepted],
        "rejected_separators": [candidate.to_dict() for candidate in rejected],
        "proposal_room_masks": _proposal_masks_debug(final_labels),
        **evidence.debug,
        "line_walls": wall_debug,
        "wall_gap_candidates": wall_gap_debug,
        "corridor": _strip_arrays(corridor_debug),
        "topology": topology_debug,
    }
    debug_cfg = dict(config.debug or {})
    if bool(debug_cfg.get("save_layers", False)) or bool(debug_cfg.get("save_candidate_json", False)):
        dump = save_online_roomseg_debug(
            out_dir=Path(str(config.debug_dir)) / ("online_roomseg_step_%06d" % int(step)),
            layers=layers,
            separator_report=report,
            save_layers=bool(debug_cfg.get("save_layers", True)),
            save_candidate_json=bool(debug_cfg.get("save_candidate_json", True)),
        )
        debug["online_roomseg_debug_paths"] = dict(dump.get("paths", {}))
    return OnlineRoseStyleResult(
        room_label_map=final_labels.astype(np.int32),
        separator_map=separator_map.astype(bool),
        accepted_candidates=list(accepted),
        rejected_candidates=list(rejected),
        layers=layers,
        debug=debug,
    )


def _rooms_from_labels(labels: np.ndarray, unknown: np.ndarray, config: RoomSegmentationConfig, step: int, debug: Mapping[str, object]) -> list[RoomMask]:
    out: list[RoomMask] = []
    min_cells = max(1, int(config.min_observed_free_cells))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = np.asarray(labels == label, dtype=bool)
        if int(np.count_nonzero(mask)) < min_cells and out:
            continue
        room = _room_from_mask("pending", mask, unknown, [], config, int(step))
        room.source = ONLINE_ROSE_STYLE_BACKEND
        room.metadata["label_id"] = int(label)
        room.metadata["proposal_labels"] = [int(label)]
        room.metadata["source_finalization_mode"] = ONLINE_ROSE_STYLE_BACKEND
        room.metadata["accepted_separator_count"] = int(len(debug.get("accepted_separators", []) or []))
        out.append(room)
    return out


def _candidate_layers(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    out = {
        "physical_wall_completion": np.zeros(shape, dtype=bool),
        "doorway_virtual_cut": np.zeros(shape, dtype=bool),
        "corridor_room_neck_cut": np.zeros(shape, dtype=bool),
    }
    for candidate in candidates:
        if candidate.kind in out:
            out[candidate.kind] |= candidate.mask(shape)
    return out


def _rasterize_many(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= candidate.mask(shape)
    return out


def _strip_arrays(value):
    if isinstance(value, dict):
        return {str(k): _strip_arrays(v) for k, v in value.items() if not isinstance(v, np.ndarray)}
    if isinstance(value, list):
        return [_strip_arrays(v) for v in value]
    if isinstance(value, tuple):
        return [_strip_arrays(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
