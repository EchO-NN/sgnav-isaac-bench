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

from .corridor import (
    CorridorConfig,
    CorridorMergeConfig,
    CorridorRoomNeckCutConfig,
    build_corridor_debug,
    generate_corridor_room_neck_candidates,
    merge_false_parallel_door_corridor_regions,
)
from .debug_viz import save_online_roomseg_debug
from .evidence_maps import FreeCleanConfig, WallCandidateConfig, build_evidence_maps
from .separator_candidates import (
    DoorwayVirtualCutConfig,
    DoorNeckConfig,
    LineExtensionConfig,
    MissedScanGapClosureConfig,
    NoiseWallGapFillConfig,
    PhysicalWallCompletionConfig,
    SeparatorCandidate,
    ShortUnknownGapClosureConfig,
    SingleSidedWallExtensionConfig,
    build_door_neck_candidates_from_extension_intersections,
    build_door_neck_candidates_from_extensions,
    build_l_corner_door_neck_candidates,
    extend_wall_lines_once,
    fill_noise_wall_gaps_from_runs,
    generate_wall_gap_candidates,
    rasterize_candidates,
    reject_candidates_ending_on_other_door_middle,
)
from .topology_tests import TopologyTestConfig, greedily_select_separators
from .utils import label_components, relabel_compact
from .wall_lines import (
    LineWallsConfig,
    LineFilteringConfig,
    WallRunMergeConfig,
    WallRunSnapConfig,
    extract_line_supported_walls,
    filtered_wall_line_mask,
    filter_and_snap_wall_lines,
    line_supported_wall_mask,
    merge_snapped_wall_runs,
    snap_wall_segments_to_runs,
    snapped_wall_run_mask,
)


ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND = "online_line_extend_roomseg_v2"
ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT = "online_line_extend_roomseg_v2_vlm"
ONLINE_ROSE_STYLE_BACKEND = "online_rose_style_v1"
ONLINE_ROSE_STYLE_CONTEXT = "online_rose_style_v1_vlm"


@dataclass
class OnlineRoseStyleConfig:
    enabled: bool = True
    backend: str = ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    z_min_m: float = 0.10
    z_max_m: float = 2.50
    min_free_rays: int = 1
    min_observed_rays: int = 1
    min_observed_free_cells: int = 20
    min_room_area_m2: float = 1.0
    debug_dir: str = "debug/online_roomseg"
    free_clean: FreeCleanConfig = field(default_factory=FreeCleanConfig)
    wall_candidate: WallCandidateConfig = field(default_factory=WallCandidateConfig)
    line_walls: LineWallsConfig = field(default_factory=LineWallsConfig)
    line_filtering: LineFilteringConfig = field(default_factory=LineFilteringConfig)
    line_extension: LineExtensionConfig = field(default_factory=LineExtensionConfig)
    door_neck: DoorNeckConfig = field(default_factory=DoorNeckConfig)
    wall_run_snap: WallRunSnapConfig = field(default_factory=WallRunSnapConfig)
    wall_run_merge: WallRunMergeConfig = field(default_factory=WallRunMergeConfig)
    physical_wall_completion: PhysicalWallCompletionConfig = field(default_factory=PhysicalWallCompletionConfig)
    missed_scan_gap_closure: MissedScanGapClosureConfig = field(default_factory=MissedScanGapClosureConfig)
    short_unknown_gap_closure: ShortUnknownGapClosureConfig = field(default_factory=ShortUnknownGapClosureConfig)
    noise_wall_gap_fill: NoiseWallGapFillConfig = field(default_factory=NoiseWallGapFillConfig)
    doorway_virtual_cut: DoorwayVirtualCutConfig = field(default_factory=DoorwayVirtualCutConfig)
    single_sided_wall_extension: SingleSidedWallExtensionConfig = field(default_factory=SingleSidedWallExtensionConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    corridor_room_neck_cut: CorridorRoomNeckCutConfig = field(default_factory=CorridorRoomNeckCutConfig)
    corridor_merge: CorridorMergeConfig = field(default_factory=CorridorMergeConfig)
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
            raw.setdefault("z_min_m", vertical_or_free.get("z_min_m", 0.10))
            raw.setdefault("z_max_m", vertical_or_free.get("z_max_m", 2.50))
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
            "line_filtering": LineFilteringConfig.from_mapping(raw.get("line_filtering")),
            "line_extension": LineExtensionConfig.from_mapping(raw.get("line_extension")),
            "door_neck": DoorNeckConfig.from_mapping(raw.get("door_neck")),
            "wall_run_snap": WallRunSnapConfig.from_mapping(raw.get("wall_run_snap")),
            "wall_run_merge": WallRunMergeConfig.from_mapping(raw.get("wall_run_merge")),
            "physical_wall_completion": PhysicalWallCompletionConfig.from_mapping(raw.get("physical_wall_completion")),
            "missed_scan_gap_closure": MissedScanGapClosureConfig.from_mapping(raw.get("missed_scan_gap_closure")),
            "short_unknown_gap_closure": ShortUnknownGapClosureConfig.from_mapping(raw.get("short_unknown_gap_closure")),
            "noise_wall_gap_fill": NoiseWallGapFillConfig.from_mapping(raw.get("noise_wall_gap_fill")),
            "doorway_virtual_cut": DoorwayVirtualCutConfig.from_mapping(raw.get("doorway_virtual_cut")),
            "single_sided_wall_extension": SingleSidedWallExtensionConfig.from_mapping(raw.get("single_sided_wall_extension")),
            "corridor": CorridorConfig.from_mapping(raw.get("corridor")),
            "corridor_room_neck_cut": CorridorRoomNeckCutConfig.from_mapping(raw.get("corridor_room_neck_cut")),
            "corridor_merge": CorridorMergeConfig.from_mapping(raw.get("corridor_merge")),
            "topology_test": TopologyTestConfig.from_mapping(_merge_topology_config(raw)),
        }
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key not in nested}
        base.update(nested)
        return cls(**base)

    def room_config(self) -> RoomSegmentationConfig:
        return RoomSegmentationConfig(
            algorithm=ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            source_grid="vertical_profile_free_0p1_2p5",
            proposal_mode=ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
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
    context_source = ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT

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
            roomseg_ray_evidence=roomseg_ray_evidence,
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
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
) -> OnlineRoseStyleResult:
    evidence = build_evidence_maps(
        occupancy_map=occupancy_map,
        observed_free_mask=observed_free_mask,
        obstacle_mask=obstacle_mask,
        unknown_mask=unknown_mask,
        vertical_profile=vertical_profile,
        roomseg_ray_evidence=roomseg_ray_evidence,
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
    raw_line_map = line_supported_wall_mask(segments, evidence.free_clean.shape)
    filtered_lines, filter_debug = filter_and_snap_wall_lines(
        segments,
        wall_candidate_clean=evidence.wall_candidate_clean,
        free_clean=evidence.free_clean,
        resolution_m=float(config.resolution_m),
        config=config.line_filtering,
    )
    filtered_line_map = filtered_wall_line_mask(filtered_lines, evidence.free_clean.shape)
    noise_gap_runs, noise_gap_snap_debug = snap_wall_segments_to_runs(
        segments,
        evidence.wall_candidate_clean,
        resolution_m=float(config.resolution_m),
        max_angle_to_axis_deg=float(config.wall_run_snap.max_angle_to_axis_deg),
        support_band_cells=int(config.wall_run_snap.support_band_cells),
        min_run_length_m=float(config.wall_run_snap.min_run_length_m),
        min_support_ratio=float(config.wall_run_snap.min_support_ratio),
        close_holes_m=0.0,
    )
    noise_gap_fill_map, noise_gap_debug = fill_noise_wall_gaps_from_runs(
        noise_gap_runs,
        shape=evidence.free_clean.shape,
        resolution_m=float(config.resolution_m),
        config=config.noise_wall_gap_fill,
    )
    structural_wall_free_overlap_map = (evidence.wall_candidate_clean | noise_gap_fill_map) & evidence.free_clean
    noise_gap_room_separator_map = noise_gap_fill_map & evidence.free_clean
    roomseg_free_clean = evidence.free_clean & ~structural_wall_free_overlap_map
    roomseg_unknown_clean = evidence.unknown_clean & ~noise_gap_room_separator_map
    roomseg_wall_target_map = evidence.wall_candidate_clean | raw_line_map | filtered_line_map | noise_gap_fill_map
    corridor_debug = build_corridor_debug(
        roomseg_free_clean,
        resolution_m=float(config.resolution_m),
        config=config.corridor,
    )
    corridor_skeleton = np.asarray(corridor_debug.get("corridor_skeleton", np.zeros_like(roomseg_free_clean)), dtype=bool)

    pass1_gap_candidates, pass1_gap_candidate_debug = generate_wall_gap_candidates(
        segments,
        free_clean=roomseg_free_clean,
        wall_candidate_clean=roomseg_wall_target_map,
        unknown_clean=roomseg_unknown_clean,
        resolution_m=float(config.resolution_m),
        physical_config=config.physical_wall_completion,
        doorway_config=config.doorway_virtual_cut,
        missed_scan_config=config.missed_scan_gap_closure,
        short_unknown_config=config.short_unknown_gap_closure,
        single_sided_config=config.single_sided_wall_extension,
        runs=noise_gap_runs,
        start_id=1,
    )
    pass1_extension_enabled = bool(config.line_extension.enabled) and int(config.line_extension.passes) >= 1
    if pass1_extension_enabled:
        pass1_extensions, pass1_extension_debug = extend_wall_lines_once(
            filtered_lines,
            free_clean=roomseg_free_clean,
            wall_target_mask=roomseg_wall_target_map,
            virtual_target_mask=None,
            unknown_clean=roomseg_unknown_clean,
            resolution_m=float(config.resolution_m),
            pass_id=1,
            config=config.line_extension,
            start_id=1,
        )
        pass1_extension_debug = {
            **dict(pass1_extension_debug),
            "stage": "line_extension_pass1",
        }
        pass1_extension_candidates, pass1_extension_candidate_debug = build_door_neck_candidates_from_extensions(
            pass1_extensions,
            accepted_virtual_targets=None,
            resolution_m=float(config.resolution_m),
            config=config.door_neck,
            start_id=1 + len(pass1_gap_candidates),
        )
    else:
        pass1_extensions = []
        pass1_extension_candidates = []
        pass1_extension_debug = {
            "enabled": False,
            "pass_id": 1,
            "stage": "line_extension_pass1",
            "extension_count": 0,
            "accepted_extension_count": 0,
            "rejected_by_reason": {},
            "extensions": [],
            "skipped_reason": "line_extension_passes_lt_1",
        }
        pass1_extension_candidate_debug = {
            "enabled": bool(config.door_neck.enabled),
            "pass_id": 1,
            "stage": "line_extension_pass1",
            "candidate_count": 0,
            "rejected_extension_count": 0,
            "candidates": [],
            "rejected_extensions": [],
            "skipped_reason": "line_extension_passes_lt_1",
        }
    pass1_candidates = [*pass1_gap_candidates, *pass1_extension_candidates]
    pass1_accepted, pass1_rejected, pass1_separator_map, pass1_labels, pass1_topology = greedily_select_separators(
        pass1_candidates,
        free_clean=roomseg_free_clean,
        unknown_clean=roomseg_unknown_clean,
        wall_candidate_clean=roomseg_wall_target_map,
        corridor_skeleton=corridor_skeleton,
        resolution_m=float(config.resolution_m),
        config=config.topology_test,
    )
    pass1_topology = {**dict(pass1_topology), "pass_id": 1, "stage": "prepass_anchor_topology"}
    pass1_candidate_debug = {
        "enabled": True,
        "pass_id": 1,
        "stage": "prepass_anchor_topology",
        "wall_gap_candidates": pass1_gap_candidate_debug,
        "line_extension_candidates": pass1_extension_candidate_debug,
        "candidate_count": int(len(pass1_candidates)),
        "accepted_count": int(len(pass1_accepted)),
        "rejected_count": int(len(pass1_rejected)),
        "candidate_count_by_kind": _kind_counts(pass1_candidates),
        "accepted_count_by_kind": _kind_counts(pass1_accepted),
        "rejected_count_by_reason": _reason_counts(pass1_rejected),
        "candidates": [candidate.to_dict() for candidate in pass1_candidates[:1024]],
    }
    before_labels, _ = ndimage.label(roomseg_free_clean, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
    before_labels = relabel_compact(before_labels)
    virtual_target_map = pass1_separator_map.astype(bool)
    pre_topology_rejected: list[SeparatorCandidate] = [*pass1_rejected]
    pass2_enabled = bool(config.line_extension.enabled) and int(config.line_extension.passes) >= 2
    pass2_start_id = 1 + len(pass1_candidates) + len(pass1_rejected)
    if pass2_enabled:
        pass2_extensions, pass2_extension_debug = extend_wall_lines_once(
            filtered_lines,
            free_clean=roomseg_free_clean,
            wall_target_mask=roomseg_wall_target_map,
            virtual_target_mask=virtual_target_map,
            unknown_clean=roomseg_unknown_clean,
            resolution_m=float(config.resolution_m),
            pass_id=2,
            config=config.line_extension,
            start_id=1 + len(pass1_extensions),
        )
        pass2_extension_debug = {
            **dict(pass2_extension_debug),
            "stage": "line_extension_pass2",
            "virtual_target_cell_count": int(np.count_nonzero(virtual_target_map)),
        }
        pass2_candidates, pass2_candidate_debug = build_door_neck_candidates_from_extensions(
            pass2_extensions,
            accepted_virtual_targets=pass1_accepted,
            resolution_m=float(config.resolution_m),
            config=config.door_neck,
            start_id=pass2_start_id,
        )
        pass2_intersection_candidates, pass2_extension_intersection_targets, pass2_intersection_debug = (
            build_door_neck_candidates_from_extension_intersections(
                pass2_extensions,
                free_clean=roomseg_free_clean,
                unknown_clean=roomseg_unknown_clean,
                resolution_m=float(config.resolution_m),
                line_config=config.line_extension,
                door_config=config.door_neck,
                start_id=pass2_start_id + len(pass2_candidates),
            )
        )
        pass2_l_corner_candidates, pass2_l_corner_debug = build_l_corner_door_neck_candidates(
            filtered_lines,
            free_clean=roomseg_free_clean,
            unknown_clean=roomseg_unknown_clean,
            resolution_m=float(config.resolution_m),
            line_config=config.line_extension,
            door_config=config.door_neck,
            start_id=pass2_start_id + len(pass2_candidates) + len(pass2_intersection_candidates),
        )
        pass2_candidates = [*pass2_candidates, *pass2_intersection_candidates, *pass2_l_corner_candidates]
        pass2_candidate_debug = {
            **dict(pass2_candidate_debug),
            "pass_id": 2,
            "stage": "line_extension_pass2",
            "extension_intersection": pass2_intersection_debug,
            "l_corner_door_neck": pass2_l_corner_debug,
            "candidate_count": int(len(pass2_candidates)),
        }
    else:
        pass2_extensions = []
        pass2_candidates = []
        pass2_l_corner_candidates = []
        pass2_extension_intersection_targets = np.zeros_like(roomseg_free_clean, dtype=bool)
        pass2_intersection_debug = {"enabled": False, "candidate_count": 0, "reason": "line_extension_passes_lt_2"}
        pass2_l_corner_debug = {"enabled": False, "candidate_count": 0, "reason": "line_extension_passes_lt_2"}
        pass2_extension_debug = {
            "enabled": False,
            "pass_id": 2,
            "stage": "line_extension_pass2",
            "extension_count": 0,
            "accepted_extension_count": 0,
            "skipped_reason": "line_extension_passes_lt_2",
            "configured_passes": int(config.line_extension.passes),
        }
        pass2_candidate_debug = {
            "enabled": bool(config.door_neck.enabled),
            "pass_id": 2,
            "stage": "line_extension_pass2",
            "candidate_count": 0,
            "rejected_extension_count": 0,
            "skipped_reason": "line_extension_passes_lt_2",
            "configured_passes": int(config.line_extension.passes),
            "extension_intersection": pass2_intersection_debug,
            "l_corner_door_neck": pass2_l_corner_debug,
        }
    pass2_corridor_room_neck_candidates, pass2_corridor_room_neck_debug = generate_corridor_room_neck_candidates(
        roomseg_free_clean,
        resolution_m=float(config.resolution_m),
        corridor_config=config.corridor,
        neck_config=config.corridor_room_neck_cut,
        corridor_debug=corridor_debug,
        wall_candidate_clean=roomseg_wall_target_map,
        start_id=pass2_start_id + len(pass2_candidates),
    )
    pass2_candidates = [*pass2_candidates, *pass2_corridor_room_neck_candidates]
    pass2_candidate_debug = {
        **dict(pass2_candidate_debug),
        "corridor_room_neck": pass2_corridor_room_neck_debug,
        "candidate_count": int(len(pass2_candidates)),
    }
    pass2_candidates, endpoint_middle_rejected, endpoint_middle_filter_debug = (
        reject_candidates_ending_on_other_door_middle(
            pass2_candidates,
            roomseg_free_clean.shape,
            config=config.door_neck,
        )
    )
    pre_topology_rejected.extend(endpoint_middle_rejected)
    if endpoint_middle_rejected and pass2_enabled:
        kept_intersection_cells = {
            tuple(int(v) for v in candidate.debug.get("intersection_rc", []))
            for candidate in pass2_candidates
            if str(candidate.debug.get("candidate_source", "")) == "extension_intersection"
            and len(candidate.debug.get("intersection_rc", []) or []) == 2
        }
        for rejected_candidate in endpoint_middle_rejected:
            if str(rejected_candidate.debug.get("candidate_source", "")) != "extension_intersection":
                continue
            raw_rc = rejected_candidate.debug.get("intersection_rc", [])
            if len(raw_rc or []) != 2:
                continue
            rc = tuple(int(v) for v in raw_rc)
            if rc in kept_intersection_cells:
                continue
            if 0 <= rc[0] < pass2_extension_intersection_targets.shape[0] and 0 <= rc[1] < pass2_extension_intersection_targets.shape[1]:
                pass2_extension_intersection_targets[rc[0], rc[1]] = False
    pass2_candidate_debug = {
        **dict(pass2_candidate_debug),
        "endpoint_on_other_door_middle_filter": endpoint_middle_filter_debug,
        "candidate_count": int(len(pass2_candidates)),
        "pre_topology_rejected_count": int(len(pre_topology_rejected)),
    }
    candidates = [*pass1_accepted, *pass2_candidates]
    pass2_virtual_target_map = (virtual_target_map | pass2_extension_intersection_targets) if pass2_enabled else np.zeros_like(virtual_target_map, dtype=bool)
    accepted, topology_rejected, separator_map, raw_labels, topology_debug = greedily_select_separators(
        candidates,
        free_clean=roomseg_free_clean,
        unknown_clean=roomseg_unknown_clean,
        wall_candidate_clean=roomseg_wall_target_map,
        corridor_skeleton=corridor_skeleton,
        resolution_m=float(config.resolution_m),
        config=config.topology_test,
    )
    rejected = [*pre_topology_rejected, *topology_rejected]
    initial_accepted_virtual_boundary_map = (separator_map | structural_wall_free_overlap_map).astype(bool)
    initial_raw_labels, _ = label_components(roomseg_free_clean & ~initial_accepted_virtual_boundary_map, 4)
    initial_raw_labels = relabel_compact(initial_raw_labels)
    _initial_corridor_labels, initial_corridor_merge_debug = merge_false_parallel_door_corridor_regions(
        initial_raw_labels,
        accepted_candidates=accepted,
        free_clean=roomseg_free_clean,
        unknown_clean=roomseg_unknown_clean,
        wall_candidate_clean=roomseg_wall_target_map,
        filtered_lines=filtered_lines,
        resolution_m=float(config.resolution_m),
        config=config.corridor_merge,
    )
    active_accepted = [candidate for candidate in accepted if not bool(candidate.debug.get("rejected_after_corridor_merge", False))]
    accepted_virtual_boundary_map = (_rasterize_many(active_accepted, roomseg_free_clean.shape) | structural_wall_free_overlap_map).astype(bool)
    raw_labels, _ = label_components(roomseg_free_clean & ~accepted_virtual_boundary_map, 4)
    raw_labels = relabel_compact(raw_labels)
    final_labels_raw, corridor_merge_debug = merge_false_parallel_door_corridor_regions(
        raw_labels,
        accepted_candidates=active_accepted,
        free_clean=roomseg_free_clean,
        unknown_clean=roomseg_unknown_clean,
        wall_candidate_clean=roomseg_wall_target_map,
        filtered_lines=filtered_lines,
        resolution_m=float(config.resolution_m),
        config=config.corridor_merge,
    )
    newly_rejected_after_merge = [candidate for candidate in active_accepted if bool(candidate.debug.get("rejected_after_corridor_merge", False))]
    if newly_rejected_after_merge:
        active_accepted = [candidate for candidate in accepted if not bool(candidate.debug.get("rejected_after_corridor_merge", False))]
        accepted_virtual_boundary_map = (_rasterize_many(active_accepted, roomseg_free_clean.shape) | structural_wall_free_overlap_map).astype(bool)
        raw_labels, _ = label_components(roomseg_free_clean & ~accepted_virtual_boundary_map, 4)
        raw_labels = relabel_compact(raw_labels)
        final_labels_raw = raw_labels.copy()
    corridor_merge_debug = {
        **dict(corridor_merge_debug),
        "pre_rejection_pass": _strip_arrays(initial_corridor_merge_debug),
        "accepted_separator_count_after_corridor_merge": int(len(active_accepted)),
        "accepted_virtual_boundary_cells": int(np.count_nonzero(accepted_virtual_boundary_map)),
    }
    corridor_rejected = [candidate for candidate in accepted if bool(candidate.debug.get("rejected_after_corridor_merge", False))]
    for candidate in corridor_rejected:
        candidate.accepted = False
        candidate.reject_reason = str(candidate.reject_reason or candidate.debug.get("corridor_merge_reject_reason", "reject_corridor_false_parallel_door"))
    final_accepted = [candidate for candidate in accepted if not bool(candidate.debug.get("rejected_after_corridor_merge", False))]
    for candidate in final_accepted:
        candidate.accepted = True
        candidate.reject_reason = ""
    rejected = [*rejected, *corridor_rejected]
    final_labels_before_virtual_fill = final_labels_raw.copy()
    final_labels, virtual_separator_fill_debug = _fill_virtual_separator_label_gaps(
        final_labels_raw,
        free_clean=roomseg_free_clean,
        virtual_separator_map=accepted_virtual_boundary_map,
    )
    candidate_layers = _candidate_layers([*pass1_candidates, *pass2_candidates, *final_accepted, *rejected], roomseg_free_clean.shape)
    rejected_map = _rasterize_many(rejected, roomseg_free_clean.shape)
    pass1_extension_layers = _extension_layers(pass1_extensions, roomseg_free_clean.shape)
    pass2_extension_layers = _extension_layers(pass2_extensions, roomseg_free_clean.shape)
    pass2_extension_completion_map = _rasterize_many(
        [
            c
            for c in final_accepted
            if str(c.kind) in {"line_extension_door_neck", "extension_intersection_cut"}
        ],
        roomseg_free_clean.shape,
    )
    wall_target_after_line_extension = roomseg_wall_target_map.copy()
    final_separator_map = accepted_virtual_boundary_map.copy()
    accepted_before_corridor_merge = initial_accepted_virtual_boundary_map.copy()
    accepted_after_corridor_merge = accepted_virtual_boundary_map.copy()
    false_parallel_rejected_map = _rasterize_many(corridor_rejected, roomseg_free_clean.shape)
    corridor_like_regions, open_living_room_like_regions = _region_type_layers(final_labels, corridor_merge_debug)
    room_confidence_map = _room_confidence_layer(final_labels, roomseg_free_clean, accepted_virtual_boundary_map)
    functional_zone_map = np.zeros_like(final_labels, dtype=np.int32)
    topology_reject_reason_map = _reject_reason_layer(rejected, roomseg_free_clean.shape)
    layers = {
        "vertical_free_raw": evidence.vertical_free_raw,
        "vertical_occupied_raw": evidence.vertical_occupied_raw,
        "vertical_observed_raw": evidence.vertical_observed_raw,
        "vertical_unknown_raw": evidence.vertical_unknown_raw,
        "free_clean_before_noise_wall_gap_fill": evidence.free_clean,
        "free_clean": roomseg_free_clean,
        "wall_candidate_clean": evidence.wall_candidate_clean,
        "noise_wall_gap_fill": noise_gap_room_separator_map,
        "noise_wall_gap_fill_all": noise_gap_fill_map,
        "structural_wall_free_overlap": structural_wall_free_overlap_map,
        "wall_target_after_noise_gap_fill": roomseg_wall_target_map,
        "pass2_line_extension_completion": pass2_extension_completion_map,
        "wall_target_after_line_extension": wall_target_after_line_extension,
        "completed_wall_after_line_extension": wall_target_after_line_extension,
        "unknown_clean": roomseg_unknown_clean,
        "line_supported_walls": raw_line_map,
        "line_supported_wall_mask": raw_line_map,
        "raw_line_supported_walls": raw_line_map,
        "filtered_wall_lines": filtered_line_map,
        "filtered_wall_endpoints": _filtered_line_endpoint_mask(filtered_lines, roomseg_free_clean.shape),
        "physical_wall_completion_candidates": candidate_layers["physical_wall_completion"],
        "missed_scan_gap_closure_candidates": candidate_layers["missed_scan_gap_closure"],
        "short_unknown_gap_closure_candidates": candidate_layers["short_unknown_gap_closure"],
        "doorway_virtual_cut_candidates": candidate_layers["doorway_virtual_cut"],
        "extension_intersection_cut_candidates": candidate_layers["extension_intersection_cut"],
        "single_sided_wall_extension_candidates": candidate_layers["single_sided_wall_extension"],
        "corridor_skeleton": corridor_skeleton,
        "corridor_candidate_map": np.asarray(corridor_debug.get("corridor_candidate_map", np.zeros_like(evidence.free_clean)), dtype=bool),
        "corridor_room_neck_cut_candidates": candidate_layers["corridor_room_neck_cut"],
        "pass1_line_extensions_all": pass1_extension_layers["all"],
        "pass1_line_extensions_accepted": pass1_extension_layers["accepted"],
        "pass1_line_extensions_rejected": pass1_extension_layers["rejected"],
        "pass1_door_neck_candidates": _rasterize_many(pass1_candidates, roomseg_free_clean.shape),
        "pass1_accepted_separators": pass1_separator_map,
        "pass1_rejected_separators": _rasterize_many(pass1_rejected, roomseg_free_clean.shape),
        "pass2_virtual_targets": pass2_virtual_target_map,
        "pass2_extension_intersection_targets": pass2_extension_intersection_targets,
        "pass2_line_extensions_all": pass2_extension_layers["all"],
        "pass2_line_extensions_accepted": pass2_extension_layers["accepted"],
        "pass2_line_extensions_rejected": pass2_extension_layers["rejected"],
        "pass2_door_neck_candidates": _rasterize_many(pass2_candidates, roomseg_free_clean.shape),
        "accepted_separators_before_corridor_merge": accepted_before_corridor_merge,
        "accepted_separators_after_corridor_merge": accepted_after_corridor_merge,
        "accepted_virtual_boundary_map": accepted_virtual_boundary_map,
        "separator_map": accepted_virtual_boundary_map,
        "rejected_false_parallel_doors": false_parallel_rejected_map,
        "accepted_separators": final_separator_map,
        "rejected_separators": rejected_map,
        "room_labels_before_separators": before_labels,
        "room_labels_after_separators": raw_labels,
        "raw_room_labels_before_corridor_merge": raw_labels,
        "room_labels_after_corridor_merge_before_virtual_fill": final_labels_before_virtual_fill,
        "room_labels_after_corridor_merge": final_labels,
        "final_room_labels": final_labels,
        "room_confidence_map": room_confidence_map,
        "functional_zone_map": functional_zone_map,
        "topology_reject_reason_map": topology_reject_reason_map,
        "virtual_separator_label_fill": np.asarray(virtual_separator_fill_debug.get("_filled_mask", np.zeros_like(roomseg_free_clean)), dtype=bool),
        "corridor_like_regions": corridor_like_regions,
        "open_living_room_like_regions": open_living_room_like_regions,
    }
    report = {
        "step": int(step),
        "backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "algorithm": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "context_source": ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT,
        "stage_order": [
            "vertical_evidence_projection",
            "evidence_cleaning",
            "wall_line_extraction",
            "prepass_separator_candidates",
            "pass1_topology_verification",
            "pass1_accepted_virtual_targets",
            "line_extension_pass2",
            "extension_intersection_candidates",
            "final_greedy_topology_verification",
            "accepted_virtual_boundary_connected_components",
            "post_label_refinement",
            "temporal_id_matching",
        ],
        "pass1_stage": "prepass_anchor_topology",
        "pass2_stage": "line_extension_pass2",
        "wall_segment_count": int(len(segments)),
        "snapped_wall_run_count": int(filter_debug.get("snapped_wall_run_count", 0)),
        "merged_wall_run_count": int(filter_debug.get("merged_wall_run_count", 0)),
        "filtered_wall_line_count": int(len(filtered_lines)),
        "noise_wall_gap_fill_enabled": bool(noise_gap_debug.get("enabled", False)),
        "noise_wall_gap_fill_count": int(noise_gap_debug.get("filled_gap_count", 0)),
        "noise_wall_gap_fill_cells": int(np.count_nonzero(noise_gap_room_separator_map)),
        "structural_wall_free_overlap_cells": int(np.count_nonzero(structural_wall_free_overlap_map)),
        "noise_wall_gap_fill_strict_less_than_max_gap_m": float(noise_gap_debug.get("max_gap_m", 0.0)),
        "line_extension_passes_requested": int(config.line_extension.passes),
        "pass2_extension_enabled": bool(pass2_enabled),
        "pass1_extension_count": int(len(pass1_extensions)),
        "pass1_candidate_count": int(len(pass1_candidates)),
        "pass1_accepted_count": int(len(pass1_accepted)),
        "pass2_extension_count": int(len(pass2_extensions)),
        "pass2_candidate_count": int(len(pass2_candidates)),
        "pass2_extension_intersection_candidate_count": int(len(pass2_intersection_candidates)) if pass2_enabled else 0,
        "pass2_l_corner_candidate_count": int(len(pass2_l_corner_candidates)) if pass2_enabled else 0,
        "pass2_corridor_room_neck_candidate_count": int(len(pass2_corridor_room_neck_candidates)),
        "pre_topology_rejected_count": int(len(pre_topology_rejected)),
        "endpoint_on_other_door_middle_rejected_count": int(len(endpoint_middle_rejected)),
        "pass2_extension_intersection_target_cells": int(np.count_nonzero(pass2_extension_intersection_targets)),
        "pass2_line_extension_completion_cells": int(np.count_nonzero(pass2_extension_completion_map)),
        "accepted_virtual_boundary_cells": int(np.count_nonzero(accepted_virtual_boundary_map)),
        "connected_components_formula": "raw_room_labels = connected_components(free_clean & ~accepted_virtual_boundary_map, connectivity=4)",
        "wall_target_after_line_extension_cells": int(np.count_nonzero(wall_target_after_line_extension)),
        "wall_target_after_line_extension_source": "wall_target_after_noise_gap_fill_only",
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(final_accepted)),
        "rejected_count": int(len(rejected)),
        "candidate_count_by_kind": _kind_counts(candidates),
        "accepted_count_by_kind": _kind_counts(final_accepted),
        "rejected_count_by_reason": _reason_counts(rejected),
        "unanchored_candidate_count": int(sum(1 for item in rejected if str(item.reject_reason).startswith("reject_unanchored") or str(item.reject_reason) == "reject_one_sided_unanchored")),
        "corridor_to_corridor_rejected_count": int(topology_debug.get("corridor_to_corridor_rejected_count", 0)),
        "corridor_to_room_accepted_count": int(topology_debug.get("corridor_to_room_accepted_count", 0)),
        "topology_min_area_by_kind": dict(config.topology_test.per_kind_min_split_area_m2 or {}),
        "room_count_after_separators": int(len([v for v in np.unique(raw_labels) if int(v) > 0])),
        "room_count_after_corridor_merge_before_virtual_fill": int(len([v for v in np.unique(final_labels_before_virtual_fill) if int(v) > 0])),
        "final_room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
        "largest_room_area_ratio": _largest_room_area_ratio(final_labels),
        "corridor_merge_event_count": int(len(corridor_merge_debug.get("merge_events", []) or [])),
        "corridor_sliver_merge_event_count": int(len(corridor_merge_debug.get("sliver_merge_events", []) or [])),
        "virtual_separator_label_fill_cells": int(virtual_separator_fill_debug.get("filled_cell_count", 0)),
        "virtual_separator_label_fill_remaining_unlabeled_cells": int(virtual_separator_fill_debug.get("remaining_unlabeled_separator_cells", 0)),
        "final_free_assignment_ratio": _free_assignment_ratio(final_labels, roomseg_free_clean),
        "navigation_obstacle_written": False,
        "no_fallback": True,
        "candidates": [candidate.to_dict() for candidate in [*final_accepted, *rejected]],
    }
    debug = {
        "backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "actual_backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "source_backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "roomseg_backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "algorithm": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "source": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
        "context_source": ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT,
        "room_map_mode": ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT,
        "strict_fallback_used": False,
        "silent_fallback_used": False,
        "legacy_style_used": False,
        "navigation_obstacle_written": False,
        "step": int(step),
        "resolution_m": float(config.resolution_m),
        "final_room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
        "room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
        "separator_report": report,
        "filtered_wall_lines_report": {
            "filtered_wall_lines": [line.to_dict() for line in filtered_lines],
            **filter_debug,
        },
        "line_extension_report": {
            "pass1": pass1_extension_debug,
            "pass2": pass2_extension_debug,
        },
        "noise_wall_gap_fill_report": {
            **noise_gap_debug,
            "snap_debug": noise_gap_snap_debug,
        },
        "door_neck_candidate_report": {
            "pass1": pass1_candidate_debug,
            "pass2": pass2_candidate_debug,
        },
        "topology_report": {
            "pass1": pass1_topology,
            "final": topology_debug,
        },
        "corridor_merge_report": corridor_merge_debug,
        "virtual_separator_label_fill": {key: value for key, value in virtual_separator_fill_debug.items() if not str(key).startswith("_")},
        "final_region_report": {
            "final_room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
            "largest_room_area_ratio": _largest_room_area_ratio(final_labels),
            "rooms": _proposal_masks_debug(final_labels),
        },
        "accepted_separators": [candidate.to_dict() for candidate in final_accepted],
        "rejected_separators": [candidate.to_dict() for candidate in rejected],
        "proposal_room_masks": _proposal_masks_debug(final_labels),
        **evidence.debug,
        "line_walls": wall_debug,
        "line_filtering": filter_debug,
        "noise_wall_gap_fill": {
            **noise_gap_debug,
            "snap_debug": noise_gap_snap_debug,
        },
        "corridor": _strip_arrays(corridor_debug),
        "corridor_merge": corridor_merge_debug,
        "topology": topology_debug,
    }
    debug["evidence_report"] = dict(evidence.debug)
    debug.update(
        {
            "backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            "actual_backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            "source_backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            "roomseg_backend": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            "algorithm": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            "source": ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
            "context_source": ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT,
            "room_map_mode": ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT,
        }
    )
    debug_cfg = dict(config.debug or {})
    if bool(debug_cfg.get("save_layers", False)) or bool(debug_cfg.get("save_candidate_json", False)):
        dump = save_online_roomseg_debug(
            out_dir=Path(str(config.debug_dir)) / ("online_roomseg_step_%06d" % int(step)),
            layers=layers,
            separator_report=report,
            extra_reports={
                "filtered_wall_lines": debug["filtered_wall_lines_report"],
                "line_extension_report": debug["line_extension_report"],
                "noise_wall_gap_fill_report": debug["noise_wall_gap_fill_report"],
                "door_neck_candidate_report": debug["door_neck_candidate_report"],
                "topology_report": debug["topology_report"],
                "corridor_merge_report": debug["corridor_merge_report"],
                "final_region_report": debug["final_region_report"],
            },
            save_layers=bool(debug_cfg.get("save_layers", True)),
            save_candidate_json=bool(debug_cfg.get("save_candidate_json", True)),
        )
        debug["online_roomseg_debug_paths"] = dict(dump.get("paths", {}))
    return OnlineRoseStyleResult(
        room_label_map=final_labels.astype(np.int32),
        separator_map=final_separator_map.astype(bool),
        accepted_candidates=list(final_accepted),
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
        room.source = ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND
        room.metadata["label_id"] = int(label)
        room.metadata["proposal_labels"] = [int(label)]
        room.metadata["source_finalization_mode"] = ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND
        room.metadata["context_source"] = ONLINE_LINE_EXTEND_ROOMSEG_V2_CONTEXT
        room.metadata["room_type"] = room.metadata.get("room_type", "room")
        room.metadata["functional_zone_label"] = room.metadata.get("functional_zone_label", "unknown_functional_zone")
        room.metadata["navigation_obstacle_written"] = False
        room.metadata["accepted_separator_count"] = int(len(debug.get("accepted_separators", []) or []))
        out.append(room)
    return out


def _candidate_layers(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    out = {
        "physical_wall_completion": np.zeros(shape, dtype=bool),
        "missed_scan_gap_closure": np.zeros(shape, dtype=bool),
        "short_unknown_gap_closure": np.zeros(shape, dtype=bool),
        "doorway_virtual_cut": np.zeros(shape, dtype=bool),
        "line_extension_door_neck": np.zeros(shape, dtype=bool),
        "extension_intersection_cut": np.zeros(shape, dtype=bool),
        "single_sided_wall_extension": np.zeros(shape, dtype=bool),
        "corridor_room_neck_cut": np.zeros(shape, dtype=bool),
    }
    for candidate in candidates:
        if candidate.kind in out:
            out[candidate.kind] |= candidate.mask(shape)
    return out


def _kind_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        out[str(candidate.kind)] = out.get(str(candidate.kind), 0) + 1
    return out


def _reason_counts(candidates: Sequence[SeparatorCandidate]) -> dict:
    out: dict[str, int] = {}
    for candidate in candidates:
        reason = str(candidate.reject_reason or "")
        out[reason] = out.get(reason, 0) + 1
    return out


def _largest_room_area_ratio(labels: np.ndarray) -> float:
    arr = np.asarray(labels, dtype=np.int32)
    counts = [int(np.count_nonzero(arr == int(label))) for label in np.unique(arr) if int(label) > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    return float(max(counts) / total)


def _free_assignment_ratio(labels: np.ndarray, free_clean: np.ndarray) -> float:
    free = np.asarray(free_clean, dtype=bool)
    if int(np.count_nonzero(free)) <= 0:
        return 0.0
    return float(np.count_nonzero((np.asarray(labels, dtype=np.int32) > 0) & free) / max(1, int(np.count_nonzero(free))))


def _room_confidence_layer(labels: np.ndarray, free_clean: np.ndarray, accepted_virtual_boundary_map: np.ndarray) -> np.ndarray:
    label_arr = np.asarray(labels, dtype=np.int32)
    free = np.asarray(free_clean, dtype=bool)
    boundary = np.asarray(accepted_virtual_boundary_map, dtype=bool)
    confidence = np.zeros(label_arr.shape, dtype=np.float32)
    confidence[free & (label_arr > 0)] = 0.90
    confidence[boundary & free & (label_arr > 0)] = 0.72
    return confidence.astype(np.float32)


def _reject_reason_layer(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int32)
    reason_to_code: dict[str, int] = {}
    next_code = 1
    for candidate in candidates:
        reason = str(candidate.reject_reason or candidate.debug.get("corridor_merge_reject_reason", "") or "rejected")
        if reason not in reason_to_code:
            reason_to_code[reason] = next_code
            next_code += 1
        mask = candidate.mask(shape)
        out[mask] = int(reason_to_code[reason])
    return out.astype(np.int32)


def _fill_virtual_separator_label_gaps(
    labels: np.ndarray,
    *,
    free_clean: np.ndarray,
    virtual_separator_map: np.ndarray,
) -> tuple[np.ndarray, dict]:
    base = np.asarray(labels, dtype=np.int32)
    free = np.asarray(free_clean, dtype=bool)
    separator = np.asarray(virtual_separator_map, dtype=bool)
    fill_mask = separator & free & (base <= 0)
    positive = base > 0
    debug = {
        "enabled": True,
        "candidate_separator_cells": int(np.count_nonzero(separator & free)),
        "fill_candidate_cells": int(np.count_nonzero(fill_mask)),
        "filled_cell_count": 0,
        "remaining_unlabeled_separator_cells": int(np.count_nonzero(fill_mask)),
        "_filled_mask": np.zeros_like(free, dtype=bool),
    }
    if not np.any(fill_mask) or not np.any(positive):
        return base.copy(), debug
    _distance, indices = ndimage.distance_transform_edt(~positive, return_indices=True)
    nearest = base[indices[0], indices[1]]
    assignable = fill_mask & (nearest > 0)
    out = base.copy()
    out[assignable] = nearest[assignable]
    out = relabel_compact(out)
    filled_mask = assignable.astype(bool)
    debug.update(
        {
            "filled_cell_count": int(np.count_nonzero(filled_mask)),
            "remaining_unlabeled_separator_cells": int(np.count_nonzero(fill_mask & (out <= 0))),
            "room_count_before_fill": int(len([v for v in np.unique(base) if int(v) > 0])),
            "room_count_after_fill": int(len([v for v in np.unique(out) if int(v) > 0])),
            "_filled_mask": filled_mask,
        }
    )
    return out.astype(np.int32), debug


def _merge_topology_config(raw: Mapping[str, object]) -> dict:
    topology = dict(raw.get("topology_test", raw.get("topology", {})) or {})
    if "topology" in raw:
        topology.update(dict(raw.get("topology", {}) or {}))
    separator = dict(raw.get("separator", {}) or {})
    if separator:
        topology["separator"] = separator
    return topology


def _rasterize_many(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= candidate.mask(shape)
    return out


def _extension_layers(extensions, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    out = {
        "all": np.zeros(shape, dtype=bool),
        "accepted": np.zeros(shape, dtype=bool),
        "rejected": np.zeros(shape, dtype=bool),
    }
    for hit in extensions:
        line = _line_mask(hit.p_start_rc, hit.p_hit_rc, shape)
        out["all"] |= line
        if getattr(hit, "reject_reason", None) is None:
            out["accepted"] |= line
        else:
            out["rejected"] |= line
    return out


def _line_mask(p0, p1, shape: tuple[int, int]) -> np.ndarray:
    from .utils import rasterize_line

    return rasterize_line(np.asarray(p0, dtype=np.float32), np.asarray(p1, dtype=np.float32), shape)


def _filtered_line_endpoint_mask(lines, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for line in lines:
        for p in (line.p0_rc, line.p1_rc):
            r, c = np.rint(np.asarray(p, dtype=np.float32)).astype(int).tolist()
            if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
                out[int(r), int(c)] = True
    return out


def _region_type_layers(labels: np.ndarray, corridor_merge_debug: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(labels, dtype=np.int32)
    corridor = np.zeros_like(arr, dtype=bool)
    open_room = np.zeros_like(arr, dtype=bool)
    for info in corridor_merge_debug.get("region_infos_after_merge", []) or []:
        label = int(info.get("label", 0) or 0)
        if label <= 0:
            continue
        if bool(info.get("corridor_like", False)):
            corridor |= arr == label
        if bool(info.get("open_living_room_like", False)):
            open_room |= arr == label
    return corridor, open_room


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
