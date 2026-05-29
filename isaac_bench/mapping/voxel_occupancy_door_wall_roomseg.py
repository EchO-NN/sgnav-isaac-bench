from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.height_profile_door_wall_roomseg import extend_step2_wall_lines
from isaac_bench.mapping.online_roomseg.debug_viz import save_online_roomseg_debug
from isaac_bench.mapping.online_roomseg.separator_candidates import (
    DoorNeckConfig,
    LineExtensionConfig,
    LineExtensionHit,
    NoiseWallGapFillConfig,
    SeparatorCandidate,
    build_door_neck_candidates_from_extension_intersections,
    build_step2_separator_candidates_from_extensions,
    fill_noise_wall_gaps_from_runs,
)
from isaac_bench.mapping.online_roomseg.topology_tests import TopologyTestConfig, greedily_select_separators
from isaac_bench.mapping.online_roomseg.utils import conn, dilate, rasterize_line, relabel_compact
from isaac_bench.mapping.online_roomseg.wall_lines import (
    FilteredWallLine,
    LineFilteringConfig,
    LineWallsConfig,
    WallRunMergeConfig,
    WallRunSnapConfig,
    extract_line_supported_walls,
    filtered_wall_line_mask,
    filter_and_snap_wall_lines,
    line_supported_wall_mask,
    snap_wall_segments_to_runs,
)
from isaac_bench.mapping.room_segmentation import RoomMask, RoomSegmentationConfig
from isaac_bench.mapping.room_segmentation import _proposal_masks_debug, _room_from_mask
from isaac_bench.mapping.voxel_door_detector import (
    DOOR_ANCHOR_FILTERED_LINE,
    DOOR_ANCHOR_PROJECTED,
    DOOR_ANCHOR_PROJECTED_ANCHOR,
    DOOR_ANCHOR_STEP1,
    DOOR_ANCHOR_STRICT_RAW,
    VoxelDoorCompletionResult,
    VoxelDoorDetectorConfig,
    VoxelDoorMemory,
    classify_voxel_door_seeds,
    complete_voxel_doors_from_seeds,
)
from isaac_bench.mapping.voxel_occupancy_grid import VoxelOccupancyGrid3D
from isaac_bench.mapping.voxel_roomseg_evidence import (
    VoxelRoomsegEvidence,
    VoxelRoomsegEvidenceConfig,
    build_voxel_roomseg_evidence,
)
from isaac_bench.mapping.wall_projection import (
    ProjectedWallLine,
    WallProjectionConfig,
    project_wall_evidence_to_axis_accumulator_lines,
)


VOXEL_OCCUPANCY_ROOMSEG_BACKEND = "voxel_occupancy_door_wall_v9"
VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM = "voxel_occupancy_door_wall_v9"
VOXEL_OCCUPANCY_ROOMSEG_CONTEXT = "voxel_occupancy_door_wall_v9_vlm"


@dataclass
class Step2StageMaps:
    extension_hits_all_map: np.ndarray
    extension_hits_pre_topology_map: np.ndarray
    separator_candidates_pre_topology_map: np.ndarray
    topology_rejected_separator_map: np.ndarray
    accepted_separator_map: np.ndarray
    accepted_partition_cut_map: np.ndarray


@dataclass
class Step2LinePool:
    source_lines: list[FilteredWallLine]
    source_line_map: np.ndarray
    target_wall_map: np.ndarray
    target_source_map: np.ndarray
    debug: dict[str, object]


@dataclass
class StableSeparatorTrack:
    track_id: int
    kind: str
    first_seen_step: int
    last_seen_step: int
    first_seen_update_index: int
    last_seen_update_index: int
    confidence: float
    center_rc: tuple[float, float]
    major_dir_rc: tuple[float, float]
    cut_cells: list[tuple[int, int]]
    visual_cells: list[tuple[int, int]]
    hard_contradiction_count: int = 0
    missed_update_count: int = 0

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        for r, c in self.cut_cells:
            if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
                out[int(r), int(c)] = True
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "kind": str(self.kind),
            "first_seen_step": int(self.first_seen_step),
            "last_seen_step": int(self.last_seen_step),
            "first_seen_update_index": int(self.first_seen_update_index),
            "last_seen_update_index": int(self.last_seen_update_index),
            "confidence": float(self.confidence),
            "center_rc": [float(self.center_rc[0]), float(self.center_rc[1])],
            "major_dir_rc": [float(self.major_dir_rc[0]), float(self.major_dir_rc[1])],
            "cut_cells": [[int(r), int(c)] for r, c in self.cut_cells],
            "visual_cells": [[int(r), int(c)] for r, c in self.visual_cells],
            "hard_contradiction_count": int(self.hard_contradiction_count),
            "missed_update_count": int(self.missed_update_count),
        }


@dataclass
class StableSeparatorMemoryResult:
    stable_separator_mask: np.ndarray
    stable_separator_visual_mask: np.ndarray
    tracks: list[StableSeparatorTrack]
    debug: dict[str, object]


class VoxelSeparatorMemory:
    def __init__(self, ttl_updates: int = 40, decay_per_update: float = 0.05, min_confidence: float = 0.20, hard_contradiction_min_updates: int = 3):
        self.ttl_updates = int(ttl_updates)
        self.decay_per_update = float(decay_per_update)
        self.min_confidence = float(min_confidence)
        self.hard_contradiction_min_updates = int(hard_contradiction_min_updates)
        self._tracks: list[StableSeparatorTrack] = []
        self._next_track_id = 1

    def stable_mask(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        for track in self._tracks:
            out |= track.mask(shape)
        return out.astype(bool)

    def update(
        self,
        accepted_mask: np.ndarray,
        *,
        step: int,
        update_index: int,
        shape: tuple[int, int],
        contradiction_wall_map: np.ndarray | None = None,
    ) -> StableSeparatorMemoryResult:
        accepted = np.asarray(accepted_mask, dtype=bool)
        if accepted.shape != shape:
            raise ValueError("accepted separator mask must match shape")
        strict_wall = np.zeros(shape, dtype=bool) if contradiction_wall_map is None else np.asarray(contradiction_wall_map, dtype=bool)
        labels, count = ndimage.label(accepted, structure=conn(8))
        current_tracks: list[StableSeparatorTrack] = []
        for label in range(1, int(count) + 1):
            comp = labels == int(label)
            cells = [tuple((int(r), int(c))) for r, c in zip(*np.nonzero(comp))]
            if not cells:
                continue
            rows, cols = np.nonzero(comp)
            center = (float(np.mean(rows)), float(np.mean(cols)))
            major = _mask_major_dir(comp)
            current_tracks.append(
                StableSeparatorTrack(
                    track_id=-1,
                    kind="step2_corridor",
                    first_seen_step=int(step),
                    last_seen_step=int(step),
                    first_seen_update_index=int(update_index),
                    last_seen_update_index=int(update_index),
                    confidence=1.0,
                    center_rc=center,
                    major_dir_rc=major,
                    cut_cells=cells,
                    visual_cells=cells,
                    hard_contradiction_count=0,
                    missed_update_count=0,
                )
            )
        matched_existing: set[int] = set()
        new_tracks: list[StableSeparatorTrack] = []
        for current in current_tracks:
            best_idx = None
            best_score = 0.0
            current_mask = current.mask(shape)
            for idx, track in enumerate(self._tracks):
                if idx in matched_existing:
                    continue
                old_mask = track.mask(shape)
                union = int(np.count_nonzero(current_mask | old_mask))
                iou = float(np.count_nonzero(current_mask & old_mask)) / float(max(1, union))
                dist = float(np.linalg.norm(np.asarray(current.center_rc) - np.asarray(track.center_rc)))
                score = iou - 0.02 * dist
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score >= 0.05:
                track = self._tracks[int(best_idx)]
                matched_existing.add(int(best_idx))
                track.last_seen_step = int(step)
                track.last_seen_update_index = int(update_index)
                track.confidence = min(1.0, float(track.confidence) + 0.35)
                track.center_rc = current.center_rc
                track.major_dir_rc = current.major_dir_rc
                track.cut_cells = list(current.cut_cells)
                track.visual_cells = list(current.visual_cells)
                track.hard_contradiction_count = 0
                track.missed_update_count = 0
                new_tracks.append(track)
            else:
                current.track_id = int(self._next_track_id)
                self._next_track_id += 1
                new_tracks.append(current)
        prune_counts: Counter[str] = Counter()
        for idx, track in enumerate(self._tracks):
            if idx in matched_existing:
                continue
            missed = max(0, int(update_index) - int(track.last_seen_update_index))
            track.missed_update_count = int(missed)
            track.confidence = max(0.0, float(track.confidence) - float(self.decay_per_update) * float(max(1, missed)))
            track_mask = track.mask(shape)
            if np.any(track_mask):
                strict_overlap = float(np.count_nonzero(track_mask & strict_wall)) / float(max(1, int(np.count_nonzero(track_mask))))
                if strict_overlap >= 0.70:
                    track.hard_contradiction_count += 1
            if missed > int(self.ttl_updates):
                prune_counts["ttl_updates_exceeded"] += 1
                continue
            if track.confidence < float(self.min_confidence):
                prune_counts["confidence_below_min"] += 1
                continue
            if track.hard_contradiction_count >= int(self.hard_contradiction_min_updates):
                prune_counts["strict_wall_contradiction"] += 1
                continue
            new_tracks.append(track)
        self._tracks = list(new_tracks)
        stable = np.zeros(shape, dtype=bool)
        visual = np.zeros(shape, dtype=bool)
        for track in self._tracks:
            stable |= track.mask(shape)
            visual |= _cells_to_mask(track.visual_cells, shape)
        debug = {
            "voxel_separator_memory_track_count": int(len(self._tracks)),
            "voxel_separator_memory_update_index": int(update_index),
            "voxel_separator_memory_prune_reason_counts": dict(prune_counts),
            "voxel_separator_memory_tracks": [track.to_dict() for track in self._tracks],
            "voxel_stable_step2_separator_cells": int(np.count_nonzero(stable)),
        }
        return StableSeparatorMemoryResult(stable.astype(bool), visual.astype(bool), list(self._tracks), debug)


@dataclass
class PartitionMapBundle:
    wall_anchor_support_map: np.ndarray
    door_anchor_wall_map: np.ndarray
    step2_target_wall_map: np.ndarray
    partition_real_wall_map: np.ndarray
    base_partition_free: np.ndarray
    partition_unknown: np.ndarray
    removed_by_seed_carve_map: np.ndarray
    debug: dict[str, object]


@dataclass
class DoorSeedMasksForStep2:
    raw_seed_mask: np.ndarray
    accepted_seed_cluster_mask: np.ndarray
    accepted_door_cut_mask: np.ndarray
    stable_door_cut_mask: np.ndarray
    accepted_door_visual_mask: np.ndarray
    step2_suppression_band: np.ndarray
    wall_carve_seed_mask: np.ndarray


@dataclass
class VoxelStep2TopologyConfig:
    corridor_min_split_area_m2: float = 0.05
    corridor_min_new_component_width_m: float = 0.10
    corridor_reject_tiny_side_width_cells_leq: int = 2
    corridor_tiny_side_min_area_m2: float = 0.03
    corridor_tiny_side_min_length_m: float = 0.35
    corridor_accept_long_narrow_side: bool = True
    corridor_local_topology_radius_cells: int = 20

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None) -> "VoxelStep2TopologyConfig":
        if isinstance(data, cls):
            return data
        raw = dict(data or {})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class VoxelOccupancyDoorWallRoomSegConfig:
    enabled: bool = True
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    voxel_evidence: VoxelRoomsegEvidenceConfig = field(default_factory=VoxelRoomsegEvidenceConfig)
    wall_projection: WallProjectionConfig = field(default_factory=WallProjectionConfig)
    door: VoxelDoorDetectorConfig = field(default_factory=VoxelDoorDetectorConfig)
    line_walls: LineWallsConfig = field(default_factory=LineWallsConfig)
    line_filtering: LineFilteringConfig = field(default_factory=LineFilteringConfig)
    wall_run_snap: WallRunSnapConfig = field(default_factory=WallRunSnapConfig)
    wall_run_merge: WallRunMergeConfig = field(default_factory=WallRunMergeConfig)
    step1_gap_fill_enabled: bool = True
    step1_gap_fill_max_gap_m: float = 0.30
    step1_gap_fill_max_lateral_offset_m: float = 0.15
    step1_gap_fill_min_endpoint_support: float = 0.25
    step1_gap_fill_forbidden_on_door: bool = True
    line_extension: LineExtensionConfig = field(default_factory=LineExtensionConfig)
    door_neck: DoorNeckConfig = field(default_factory=DoorNeckConfig)
    topology_test: TopologyTestConfig = field(default_factory=TopologyTestConfig)
    reject_step2_if_intersects_door: bool = True
    reject_step2_if_intersects_door_visual: bool = False
    reject_step2_if_intersects_door_seed: bool = False
    reject_step2_if_intersects_existing_extension: bool = True
    reject_step2_if_hits_extension_not_wall: bool = True
    reject_step2_if_hits_frontier_residual_wall: bool = True
    reject_step2_if_crosses_frontier_unknown_band: bool = True
    door_intersection_dilation_cells: int = 1
    step2_source_min_line_length_m: float = 0.60
    step2_source_min_support_ratio: float = 0.30
    step2_source_min_support_cells: int = 6
    step2_source_forbid_frontier_unknown_band: bool = True
    step2_source_forbid_door_band: bool = True
    corridor_neck_source_enabled: bool = True
    corridor_neck_source_min_line_length_m: float = 0.35
    corridor_neck_source_min_support_cells: int = 4
    corridor_neck_source_min_support_ratio: float = 0.45
    corridor_neck_source_must_touch_large_free: bool = True
    corridor_neck_source_min_adjacent_free_area_cells: int = 60
    corridor_neck_source_forbid_raw_seed_band: bool = False
    corridor_neck_source_forbid_accepted_door_band: bool = True
    corridor_neck_source_max_unknown_ratio_on_line: float = 0.25
    corridor_neck_source_allow_only_projected_or_strict: bool = True
    extension_intersection_policy: str = "reject"
    enable_extension_intersection_fallback: bool = False
    separator_memory_enabled: bool = True
    separator_memory_ttl_updates: int = 40
    separator_memory_decay_per_update: float = 0.05
    separator_memory_min_confidence_to_keep: float = 0.20
    separator_memory_hard_contradiction_min_updates: int = 3
    reject_step2_if_tiny_side_width_cells_leq: int = 3
    final_connectivity: int = 4
    merge_small_components_enabled: bool = False
    min_observed_free_cells: int = 1
    min_room_area_m2: float = 0.05
    use_real_wall_as_partition_barrier: bool = True
    real_wall_barrier_dilation_cells: int = 0
    door_seed_wall_carve_radius_cells: int = 2
    voxel_step2_topology: VoxelStep2TopologyConfig = field(default_factory=VoxelStep2TopologyConfig)
    voxel_show_wall_diagnostics: bool = False
    voxel_show_raw_wall_support: bool = False
    voxel_show_wall_support_rejected_unknown: bool = False
    voxel_show_frontier_unknown_band: bool = False
    voxel_show_step2_source_candidates: bool = False
    voxel_show_step2_rejected_candidates: bool = False
    voxel_show_door_visual_only_candidates: bool = False
    voxel_show_raw_door_seed_band: bool = False
    voxel_show_corridor_neck_sources: bool = False
    voxel_show_door_seed_reject_reason_overlay: bool = False
    debug_dump: bool = False
    debug_dir: str = "debug/voxel_occupancy_door_wall_roomseg"

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "VoxelOccupancyDoorWallRoomSegConfig":
        raw_root = dict(data or {})
        raw = dict(raw_root)
        online = dict(raw_root.get("online_roomseg", {}) or {})
        step1 = dict(raw_root.get("voxel_step1", {}) or {})
        step2 = dict(raw_root.get("voxel_step2", {}) or {})
        voxel_roomseg = dict(raw_root.get("voxel_roomseg", {}) or {})
        voxel_visualization = dict(raw_root.get("voxel_visualization", {}) or {})
        debug_layers = dict(raw_root.get("debug_layers", {}) or {})

        if "enabled" in debug_layers:
            raw["debug_dump"] = bool(debug_layers.get("enabled"))
        if "output_dir" in debug_layers:
            raw["debug_dir"] = str(debug_layers.get("output_dir"))

        topology_raw = dict(raw_root.get("topology_test", online.get("topology_test", online.get("topology", {}))) or {})
        if "reject_if_tiny_side_width_cells_leq" in step2:
            topology_raw["reject_if_side_width_cells_leq"] = int(step2["reject_if_tiny_side_width_cells_leq"])
        else:
            topology_raw.setdefault("reject_if_side_width_cells_leq", int(raw.get("reject_step2_if_tiny_side_width_cells_leq", 3)))
        topology_raw.setdefault("min_new_component_width_m", 0.20)

        line_extension_raw = dict(online.get("line_extension", raw_root.get("line_extension", {})) or {})
        line_extension_raw.update({key: step2[key] for key in step2 if key in LineExtensionConfig.__dataclass_fields__})
        line_extension_raw.setdefault("min_extension_m", 0.40)
        line_extension_raw.setdefault("max_extension_m", 1.60)
        line_extension_raw.setdefault("max_probe_m", 1.60)
        line_extension_raw.setdefault("allow_hit_virtual_door_on_pass2", False)
        line_extension_raw.setdefault("require_free_between_start_and_hit", True)
        line_extension_raw.setdefault("min_free_cells_between_start_and_hit", 3)

        nested = {
            "voxel_evidence": VoxelRoomsegEvidenceConfig.from_mapping(raw_root.get("voxel_roomseg_evidence", {})),
            "wall_projection": WallProjectionConfig.from_mapping(raw_root.get("voxel_wall_projection", {})),
            "door": VoxelDoorDetectorConfig.from_mapping(raw_root.get("voxel_door", {})),
            "line_walls": LineWallsConfig.from_mapping(online.get("line_walls", raw_root.get("line_walls"))),
            "line_filtering": LineFilteringConfig.from_mapping(online.get("line_filtering", raw_root.get("line_filtering"))),
            "wall_run_snap": WallRunSnapConfig.from_mapping(online.get("wall_run_snap", raw_root.get("wall_run_snap"))),
            "wall_run_merge": WallRunMergeConfig.from_mapping(online.get("wall_run_merge", raw_root.get("wall_run_merge"))),
            "line_extension": LineExtensionConfig.from_mapping(line_extension_raw),
            "door_neck": DoorNeckConfig.from_mapping(online.get("door_neck", raw_root.get("door_neck"))),
            "topology_test": TopologyTestConfig.from_mapping(topology_raw),
            "voxel_step2_topology": VoxelStep2TopologyConfig.from_mapping(raw_root.get("voxel_step2_topology", {})),
        }
        step1_key_map = {
            "gap_fill_enabled": "step1_gap_fill_enabled",
            "gap_fill_max_gap_m": "step1_gap_fill_max_gap_m",
            "gap_fill_max_lateral_offset_m": "step1_gap_fill_max_lateral_offset_m",
            "gap_fill_min_endpoint_support": "step1_gap_fill_min_endpoint_support",
            "gap_fill_forbidden_on_door": "step1_gap_fill_forbidden_on_door",
        }
        for src, dst in step1_key_map.items():
            if src in step1:
                raw[dst] = step1[src]
        step2_key_map = {
            "reject_if_intersects_door": "reject_step2_if_intersects_door",
            "reject_if_intersects_door_visual": "reject_step2_if_intersects_door_visual",
            "reject_if_intersects_door_seed": "reject_step2_if_intersects_door_seed",
            "reject_if_intersects_existing_extension": "reject_step2_if_intersects_existing_extension",
            "reject_if_hits_extension_not_wall": "reject_step2_if_hits_extension_not_wall",
            "reject_if_hits_frontier_residual_wall": "reject_step2_if_hits_frontier_residual_wall",
            "reject_if_crosses_frontier_unknown_band": "reject_step2_if_crosses_frontier_unknown_band",
            "door_intersection_dilation_cells": "door_intersection_dilation_cells",
            "source_min_line_length_m": "step2_source_min_line_length_m",
            "source_min_support_ratio": "step2_source_min_support_ratio",
            "source_min_support_cells": "step2_source_min_support_cells",
            "source_forbid_frontier_unknown_band": "step2_source_forbid_frontier_unknown_band",
            "source_forbid_door_band": "step2_source_forbid_door_band",
            "corridor_neck_source_enabled": "corridor_neck_source_enabled",
            "corridor_neck_source_min_line_length_m": "corridor_neck_source_min_line_length_m",
            "corridor_neck_source_min_support_cells": "corridor_neck_source_min_support_cells",
            "corridor_neck_source_min_support_ratio": "corridor_neck_source_min_support_ratio",
            "corridor_neck_source_must_touch_large_free": "corridor_neck_source_must_touch_large_free",
            "corridor_neck_source_min_adjacent_free_area_cells": "corridor_neck_source_min_adjacent_free_area_cells",
            "corridor_neck_source_forbid_raw_seed_band": "corridor_neck_source_forbid_raw_seed_band",
            "corridor_neck_source_forbid_accepted_door_band": "corridor_neck_source_forbid_accepted_door_band",
            "corridor_neck_source_max_unknown_ratio_on_line": "corridor_neck_source_max_unknown_ratio_on_line",
            "corridor_neck_source_allow_only_projected_or_strict": "corridor_neck_source_allow_only_projected_or_strict",
            "extension_intersection_policy": "extension_intersection_policy",
            "enable_extension_intersection_fallback": "enable_extension_intersection_fallback",
            "separator_memory_enabled": "separator_memory_enabled",
            "separator_memory_ttl_updates": "separator_memory_ttl_updates",
            "separator_memory_decay_per_update": "separator_memory_decay_per_update",
            "separator_memory_min_confidence_to_keep": "separator_memory_min_confidence_to_keep",
            "separator_memory_hard_contradiction_min_updates": "separator_memory_hard_contradiction_min_updates",
            "reject_if_tiny_side_width_cells_leq": "reject_step2_if_tiny_side_width_cells_leq",
        }
        for src, dst in step2_key_map.items():
            if src in step2:
                raw[dst] = step2[src]
        for key in ("use_real_wall_as_partition_barrier", "real_wall_barrier_dilation_cells", "door_seed_wall_carve_radius_cells"):
            if key in voxel_roomseg:
                raw[key] = voxel_roomseg[key]
        if "voxel_show_wall_diagnostics" in voxel_visualization:
            raw["voxel_show_wall_diagnostics"] = voxel_visualization["voxel_show_wall_diagnostics"]
        for src, dst in {
            "show_raw_wall_support": "voxel_show_raw_wall_support",
            "show_wall_support_rejected_unknown": "voxel_show_wall_support_rejected_unknown",
            "show_frontier_unknown_band": "voxel_show_frontier_unknown_band",
            "show_step2_source_candidates": "voxel_show_step2_source_candidates",
            "show_step2_rejected_candidates": "voxel_show_step2_rejected_candidates",
            "show_door_visual_only_candidates": "voxel_show_door_visual_only_candidates",
            "show_raw_door_seed_band": "voxel_show_raw_door_seed_band",
            "show_corridor_neck_sources": "voxel_show_corridor_neck_sources",
            "show_door_seed_reject_reason_overlay": "voxel_show_door_seed_reject_reason_overlay",
        }.items():
            if src in voxel_visualization:
                raw[dst] = voxel_visualization[src]
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key not in nested}
        base.update(nested)
        cfg = cls(**base)
        cfg.voxel_evidence.active_z_min_m = float(cfg.voxel_evidence.active_z_min_m)
        return cfg

    def room_config(self) -> RoomSegmentationConfig:
        return RoomSegmentationConfig(
            algorithm=VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
            source_grid="voxel_vertical_free",
            proposal_mode=VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
            finalization_mode="no_merge",
            min_observed_free_cells=int(self.min_observed_free_cells),
            min_room_area_m2=float(self.min_room_area_m2),
            resolution_m=float(self.resolution_m),
            map_info=self.map_info,
        )


@dataclass
class VoxelOccupancyDoorWallRoomSegResult:
    room_label_map: np.ndarray
    separator_map: np.ndarray
    wall_map: np.ndarray
    door_cut_map: np.ndarray
    step1_wall_gap_fill_map: np.ndarray
    step2_extension_separator_map: np.ndarray
    layers: dict[str, np.ndarray]
    debug: dict[str, object]


class VoxelOccupancyDoorWallRoomSegmenter:
    context_source = VOXEL_OCCUPANCY_ROOMSEG_CONTEXT

    def __init__(self, config: VoxelOccupancyDoorWallRoomSegConfig | Mapping[str, object] | None = None, map_info: MapInfo | None = None):
        self.config = config if isinstance(config, VoxelOccupancyDoorWallRoomSegConfig) else VoxelOccupancyDoorWallRoomSegConfig.from_mapping(config or {}, map_info=map_info)
        if map_info is not None:
            self.config.map_info = map_info
            self.config.resolution_m = float(map_info.resolution_m)
        self.last_debug: dict[str, object] = {}
        self.last_result: VoxelOccupancyDoorWallRoomSegResult | None = None
        self.door_memory = VoxelDoorMemory(self.config.door)
        self.separator_memory = VoxelSeparatorMemory(
            ttl_updates=int(self.config.separator_memory_ttl_updates),
            decay_per_update=float(self.config.separator_memory_decay_per_update),
            min_confidence=float(self.config.separator_memory_min_confidence_to_keep),
            hard_contradiction_min_updates=int(self.config.separator_memory_hard_contradiction_min_updates),
        )
        self._roomseg_update_index = 0

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        *,
        step: int,
        voxel_grid: VoxelOccupancyGrid3D | None = None,
        object_memory: Iterable[object] | None = None,
        **kwargs,
    ) -> list[RoomMask]:
        _ = object_memory, kwargs
        if voxel_grid is None:
            self.last_result = None
            self.last_debug = {
                "backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
                "algorithm": VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
                "context_source": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
                "source": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
                "reason": "voxel_grid_missing",
                "room_count": 0,
                "rooms": [],
            }
            return []
        self._roomseg_update_index += 1
        result = run_voxel_occupancy_door_wall_roomseg(
            occupancy_map=occupancy_map,
            observed_free_mask=observed_free_mask,
            obstacle_mask=obstacle_mask,
            unknown_mask=unknown_mask,
            voxel_grid=voxel_grid,
            navigation_free_mask=observed_free_mask,
            navigation_obstacle_mask=obstacle_mask,
            resolution_m=float(self.config.resolution_m),
            config=self.config,
            step=int(step),
            door_memory_update_index=int(self._roomseg_update_index),
            door_memory=self.door_memory,
            separator_memory=self.separator_memory if bool(self.config.separator_memory_enabled) else None,
        )
        self.last_result = result
        self.last_debug = dict(result.debug)
        return _rooms_from_labels(result.room_label_map, np.asarray(result.layers["voxel_unknown_xy"], dtype=bool), self.config.room_config(), int(step), result.debug)


def run_voxel_occupancy_door_wall_roomseg(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    voxel_grid: VoxelOccupancyGrid3D,
    navigation_free_mask: np.ndarray,
    navigation_obstacle_mask: np.ndarray,
    resolution_m: float,
    config: VoxelOccupancyDoorWallRoomSegConfig | Mapping[str, object] | None = None,
    step: int = 0,
    door_memory_update_index: int = 0,
    door_memory: VoxelDoorMemory | None = None,
    separator_memory: VoxelSeparatorMemory | None = None,
) -> VoxelOccupancyDoorWallRoomSegResult:
    cfg = config if isinstance(config, VoxelOccupancyDoorWallRoomSegConfig) else VoxelOccupancyDoorWallRoomSegConfig.from_mapping(config, resolution_m=resolution_m)
    shape = np.asarray(observed_free_mask, dtype=bool).shape
    for name, arr in {
        "occupancy_map": occupancy_map,
        "obstacle_mask": obstacle_mask,
        "unknown_mask": unknown_mask,
        "navigation_free_mask": navigation_free_mask,
        "navigation_obstacle_mask": navigation_obstacle_mask,
    }.items():
        if np.asarray(arr).shape != shape:
            raise ValueError("%s must match observed_free_mask shape" % name)
    if tuple(voxel_grid.shape) != tuple(shape):
        raise ValueError("voxel_grid shape must match roomseg masks")

    evidence = build_voxel_roomseg_evidence(
        voxel_grid=voxel_grid,
        navigation_free_mask=np.asarray(navigation_free_mask, dtype=bool),
        navigation_obstacle_mask=np.asarray(navigation_obstacle_mask, dtype=bool),
        unknown_mask_from_navigation=np.asarray(unknown_mask, dtype=bool),
        resolution_m=float(resolution_m),
        config=cfg.voxel_evidence,
    )
    door_seed_result = classify_voxel_door_seeds(
        voxel_grid=voxel_grid,
        config=cfg.door,
        sensor_range_count=getattr(voxel_grid, "sensor_range_count", None),
        sensor_range_threshold=int(getattr(cfg.voxel_evidence, "sensor_range_count_threshold_for_roomseg", 1)),
        roomseg_evidence=evidence,
    )
    door_seed_mask = np.asarray(door_seed_result.door_seed_mask, dtype=bool)

    projection_seed = np.asarray(evidence.support_seed_for_projection_xy if evidence.support_seed_for_projection_xy is not None else evidence.wall_support_strong_xy, dtype=bool)
    projection_bridge = np.asarray(evidence.support_bridge_for_projection_xy if evidence.support_bridge_for_projection_xy is not None else np.zeros(shape, dtype=bool), dtype=bool)
    projection_input = np.asarray(evidence.wall_support_for_projection_xy if evidence.wall_support_for_projection_xy is not None else (projection_seed | projection_bridge), dtype=bool)
    projection_weight = np.asarray(evidence.wall_support_weight_xy if evidence.wall_support_weight_xy is not None else evidence.wall_line_support_weight_xy, dtype=np.float32)
    frontier_unknown_band = np.asarray(evidence.frontier_unknown_band_xy if evidence.frontier_unknown_band_xy is not None else np.zeros(shape, dtype=bool), dtype=bool)
    forbidden_frontier_residual = np.asarray(evidence.forbidden_frontier_residual_support_xy if evidence.forbidden_frontier_residual_support_xy is not None else np.zeros(shape, dtype=bool), dtype=bool)
    forbidden_unknown_boundary = np.asarray(evidence.forbidden_unknown_boundary_support_xy if evidence.forbidden_unknown_boundary_support_xy is not None else np.zeros(shape, dtype=bool), dtype=bool)
    forbidden_residual_support = forbidden_frontier_residual | forbidden_unknown_boundary
    protected_structural_wall_band = np.asarray(evidence.protected_structural_wall_band_xy if evidence.protected_structural_wall_band_xy is not None else np.zeros(shape, dtype=bool), dtype=bool)
    wall_projection = project_wall_evidence_to_axis_accumulator_lines(
        support_seed_map=projection_seed,
        support_bridge_map=projection_bridge,
        forbidden_frontier_residual_map=forbidden_residual_support,
        protected_structural_wall_band=protected_structural_wall_band,
        support_weight=projection_weight,
        door_forbidden_mask=door_seed_mask,
        vertical_free_map=evidence.vertical_free_xy,
        unknown_map=evidence.unknown_xy,
        unknown_ratio_map=evidence.unknown_ratio_active_xy,
        navigation_unknown_map=np.asarray(unknown_mask, dtype=bool),
        frontier_unknown_band=frontier_unknown_band,
        structural_side_support_map=np.asarray(evidence.wall_xy, dtype=bool) | np.asarray(evidence.structural_wall_ratio_xy, dtype=bool) | projection_seed,
        resolution_m=float(resolution_m),
        config=cfg.wall_projection,
    )
    anchor_wall_projection = wall_projection
    projected_wall_map = np.asarray(wall_projection.projected_wall_display_map if wall_projection.projected_wall_display_map is not None else wall_projection.projected_wall_map, dtype=bool)
    anchor_projected_wall_map = np.asarray(wall_projection.projected_wall_anchor_map if wall_projection.projected_wall_anchor_map is not None else projected_wall_map, dtype=bool)
    step2_source_projected_wall_map = np.asarray(wall_projection.projected_wall_step2_source_map if wall_projection.projected_wall_step2_source_map is not None else projected_wall_map, dtype=bool)
    projected_display_lines = list(wall_projection.projected_display_lines or wall_projection.projected_lines)
    projected_step2_source_lines = list(wall_projection.projected_step2_source_lines or [])
    projected_corridor_neck_source_lines = list(getattr(wall_projection, "projected_corridor_neck_source_lines", []) or [])
    wall_for_line_extraction = np.asarray(evidence.wall_xy, dtype=bool) | projected_wall_map
    segments, wall_debug = extract_line_supported_walls(
        wall_for_line_extraction,
        resolution_m=float(resolution_m),
        config=cfg.line_walls,
    )
    raw_line_map = line_supported_wall_mask(segments, shape)
    filtered_lines, filter_debug = filter_and_snap_wall_lines(
        segments,
        wall_candidate_clean=wall_for_line_extraction,
        free_clean=evidence.vertical_free_xy,
        resolution_m=float(resolution_m),
        config=cfg.line_filtering,
    )
    seed_line_filter_cfg = replace(
        cfg.line_filtering,
        min_filtered_line_length_m=min(float(cfg.line_filtering.min_filtered_line_length_m), 0.15),
        min_filtered_support_ratio=min(float(cfg.line_filtering.min_filtered_support_ratio), 0.20),
        endpoint_min_wall_support_m=min(float(cfg.line_filtering.endpoint_min_wall_support_m), 0.10),
        min_confidence=min(float(cfg.line_filtering.min_confidence), 0.20),
    )
    extension_seed_lines, seed_filter_debug = filter_and_snap_wall_lines(
        segments,
        wall_candidate_clean=wall_for_line_extraction | anchor_projected_wall_map,
        free_clean=evidence.vertical_free_xy,
        resolution_m=float(resolution_m),
        config=seed_line_filter_cfg,
    )
    filtered_line_map = filtered_wall_line_mask(filtered_lines, shape)
    extension_seed_line_map = filtered_wall_line_mask(extension_seed_lines, shape)
    filtered_line_map_clean = filtered_line_map.copy()
    validated_extension_seed_line_map = extension_seed_line_map.copy()
    wall_base_pre_step1 = np.asarray(evidence.wall_xy, dtype=bool) | projected_wall_map

    wall_runs, wall_run_debug = snap_wall_segments_to_runs(
        segments,
        wall_base_pre_step1,
        resolution_m=float(resolution_m),
        max_angle_to_axis_deg=float(cfg.wall_run_snap.max_angle_to_axis_deg),
        support_band_cells=int(cfg.wall_run_snap.support_band_cells),
        min_run_length_m=float(cfg.wall_run_snap.min_run_length_m),
        min_support_ratio=float(cfg.wall_run_snap.min_support_ratio),
        close_holes_m=0.0,
    )
    step1_gap_fill_map, step1_gap_debug = fill_noise_wall_gaps_from_runs(
        wall_runs,
        shape=shape,
        resolution_m=float(resolution_m),
        config=NoiseWallGapFillConfig(
            enabled=bool(cfg.step1_gap_fill_enabled),
            max_gap_m=float(cfg.step1_gap_fill_max_gap_m),
            max_lateral_offset_m=float(cfg.step1_gap_fill_max_lateral_offset_m),
            min_endpoint_support=float(cfg.step1_gap_fill_min_endpoint_support),
            thickness_cells=0,
        ),
    )
    step1_gap_fill_map &= np.asarray(evidence.active_observed_xy, dtype=bool)
    if bool(cfg.step1_gap_fill_forbidden_on_door):
        step1_gap_fill_map &= ~dilate(door_seed_mask, int(cfg.door_intersection_dilation_cells))
    step1_gap_fill_map &= ~np.asarray(evidence.vertical_free_xy, dtype=bool)
    step1_completed_wall_map = wall_base_pre_step1 | step1_gap_fill_map
    pre_partition_maps = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_carve_mask=np.zeros(shape, dtype=bool),
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=anchor_projected_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        step1_gap_fill_map=step1_gap_fill_map,
        cfg=cfg,
    )
    real_wall_barrier_map = np.asarray(pre_partition_maps.partition_real_wall_map, dtype=bool)
    if int(cfg.real_wall_barrier_dilation_cells) > 0:
        real_wall_barrier_for_partition = dilate(real_wall_barrier_map, int(cfg.real_wall_barrier_dilation_cells))
    else:
        real_wall_barrier_for_partition = real_wall_barrier_map.copy()
    base_partition_free = np.asarray(pre_partition_maps.base_partition_free, dtype=bool) & ~real_wall_barrier_for_partition
    free_after_step1 = base_partition_free.copy()
    unknown_after_step1 = np.asarray(pre_partition_maps.partition_unknown, dtype=bool)
    door_anchor_source_map = _door_anchor_source_map(
        shape,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall=projected_wall_map,
        anchor_projected_wall=anchor_projected_wall_map,
        step1_gap_fill=step1_gap_fill_map,
        filtered_line=np.zeros(shape, dtype=bool),
    )
    door_anchor_source_map = np.where(pre_partition_maps.door_anchor_wall_map, door_anchor_source_map, 0).astype(np.uint8)
    door_anchor_source_map[(pre_partition_maps.door_anchor_wall_map) & (door_anchor_source_map == 0)] = DOOR_ANCHOR_STRICT_RAW
    door_anchor_wall_map = np.asarray(pre_partition_maps.door_anchor_wall_map, dtype=bool)
    door_cluster_barrier = np.asarray(pre_partition_maps.partition_real_wall_map, dtype=bool) & ~dilate(door_seed_mask, 2)
    door_completion = complete_voxel_doors_from_seeds(
        seed_result=door_seed_result,
        free_map=evidence.vertical_free_xy,
        free_map_for_visual_validation=evidence.vertical_free_xy,
        base_partition_free=base_partition_free,
        anchor_wall_map=door_anchor_wall_map,
        unknown_map=unknown_after_step1,
        resolution_m=float(resolution_m),
        config=cfg.door,
        anchor_source_map=door_anchor_source_map,
        real_wall_barrier_map=real_wall_barrier_for_partition,
        seed_cluster_barrier_map=door_cluster_barrier,
    )
    current_door_visual_mask = np.asarray(door_completion.door_centerline_visual_mask, dtype=bool)
    current_door_cut_mask = np.asarray(door_completion.door_cut_mask_for_partition, dtype=bool)
    if door_memory is not None:
        door_memory_result = door_memory.update(
            door_completion.candidates,
            step=int(step),
            update_index=int(door_memory_update_index),
            shape=shape,
            contradiction_wall_map=np.asarray(evidence.wall_xy, dtype=bool),
            contradiction_unknown_map=None,
        )
        stable_door_cut_mask = np.asarray(door_memory_result.stable_door_cut_mask, dtype=bool)
        stable_door_visual_mask = np.asarray(door_memory_result.stable_door_visual_mask, dtype=bool)
        door_memory_debug = dict(door_memory_result.debug)
    else:
        stable_door_cut_mask = np.zeros(shape, dtype=bool)
        stable_door_visual_mask = np.zeros(shape, dtype=bool)
        door_memory_debug = {
            "voxel_door_memory_enabled": bool(cfg.door.door_memory_enabled),
            "voxel_door_memory_active": False,
            "voxel_door_memory_track_count": 0,
        }
    door_cut_mask = current_door_cut_mask | stable_door_cut_mask
    door_visual_mask = current_door_visual_mask | stable_door_visual_mask
    accepted_door_visual_mask = np.asarray(door_completion.debug.get("voxel_accepted_door_centerline_mask", current_door_cut_mask), dtype=bool) | stable_door_visual_mask
    accepted_seed_cluster_mask = _accepted_door_seed_cluster_mask(
        np.asarray(door_completion.debug.get("voxel_door_seed_cluster_map", np.zeros(shape, dtype=np.int32)), dtype=np.int32),
        door_completion.debug.get("voxel_door_seed_clusters", []),
    )
    accepted_seed_for_partition = accepted_seed_cluster_mask & np.asarray(door_completion.debug.get("voxel_door_seed_mask", door_seed_mask), dtype=bool)
    partition_maps = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_carve_mask=accepted_seed_for_partition,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=anchor_projected_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        step1_gap_fill_map=step1_gap_fill_map,
        cfg=cfg,
    )
    real_wall_barrier_map = np.asarray(partition_maps.partition_real_wall_map, dtype=bool)
    if int(cfg.real_wall_barrier_dilation_cells) > 0:
        real_wall_barrier_for_partition = dilate(real_wall_barrier_map, int(cfg.real_wall_barrier_dilation_cells))
    else:
        real_wall_barrier_for_partition = real_wall_barrier_map.copy()
    base_partition_free = np.asarray(partition_maps.base_partition_free, dtype=bool) & ~real_wall_barrier_for_partition
    free_after_step1 = base_partition_free.copy()
    unknown_after_step1 = np.asarray(partition_maps.partition_unknown, dtype=bool)
    door_suppression_band = dilate(
        door_cut_mask | stable_door_cut_mask | accepted_door_visual_mask,
        int(cfg.door_intersection_dilation_cells),
    )
    raw_door_seed_band_for_debug = dilate(door_seed_mask, 2)
    door_seed_masks_for_step2 = DoorSeedMasksForStep2(
        raw_seed_mask=door_seed_mask.astype(bool),
        accepted_seed_cluster_mask=accepted_seed_cluster_mask.astype(bool),
        accepted_door_cut_mask=door_cut_mask.astype(bool),
        stable_door_cut_mask=stable_door_cut_mask.astype(bool),
        accepted_door_visual_mask=accepted_door_visual_mask.astype(bool),
        step2_suppression_band=door_suppression_band.astype(bool),
        wall_carve_seed_mask=accepted_seed_for_partition.astype(bool),
    )
    step2_target_wall_override = np.asarray(partition_maps.step2_target_wall_map, dtype=bool) | np.asarray(pre_partition_maps.step2_target_wall_map, dtype=bool)
    step2_target_wall_override &= ~door_suppression_band
    step1_wall_mask = step1_completed_wall_map
    step2_line_pool = build_step2_line_pool(
        filtered_lines=filtered_lines,
        extension_seed_lines=extension_seed_lines,
        projected_display_lines=projected_display_lines,
        projected_source_lines=projected_step2_source_lines,
        projected_corridor_neck_source_lines=projected_corridor_neck_source_lines,
        source_support_map=np.asarray(evidence.support_for_step2_source_xy if evidence.support_for_step2_source_xy is not None else projection_seed, dtype=bool) | np.asarray(evidence.wall_xy, dtype=bool) | projected_wall_map,
        frontier_unknown_band=frontier_unknown_band,
        door_suppression_band=door_suppression_band,
        raw_door_seed_band=raw_door_seed_band_for_debug,
        vertical_free_map=evidence.vertical_free_xy,
        unknown_ratio_map=evidence.unknown_ratio_active_xy,
        source_min_line_length_m=float(cfg.step2_source_min_line_length_m),
        source_min_support_ratio=float(cfg.step2_source_min_support_ratio),
        source_min_support_cells=int(cfg.step2_source_min_support_cells),
        source_forbid_frontier_unknown_band=bool(cfg.step2_source_forbid_frontier_unknown_band),
        source_forbid_door_band=bool(cfg.step2_source_forbid_door_band),
        corridor_neck_source_enabled=bool(cfg.corridor_neck_source_enabled),
        corridor_neck_source_min_line_length_m=float(cfg.corridor_neck_source_min_line_length_m),
        corridor_neck_source_min_support_cells=int(cfg.corridor_neck_source_min_support_cells),
        corridor_neck_source_min_support_ratio=float(cfg.corridor_neck_source_min_support_ratio),
        corridor_neck_source_must_touch_large_free=bool(cfg.corridor_neck_source_must_touch_large_free),
        corridor_neck_source_min_adjacent_free_area_cells=int(cfg.corridor_neck_source_min_adjacent_free_area_cells),
        corridor_neck_source_forbid_raw_seed_band=bool(cfg.corridor_neck_source_forbid_raw_seed_band),
        corridor_neck_source_forbid_accepted_door_band=bool(cfg.corridor_neck_source_forbid_accepted_door_band),
        corridor_neck_source_max_unknown_ratio_on_line=float(cfg.corridor_neck_source_max_unknown_ratio_on_line),
        corridor_neck_source_allow_only_projected_or_strict=bool(cfg.corridor_neck_source_allow_only_projected_or_strict),
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=anchor_projected_wall_map,
        step1_completed_wall_map=step1_completed_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        projected_source_line_map=step2_source_projected_wall_map,
        shape=shape,
        resolution_m=float(resolution_m),
        target_wall_override=step2_target_wall_override,
    )
    step2_door_reject_mask = build_step2_door_reject_mask(
        current_door_cut_mask=door_cut_mask,
        stable_door_cut_mask=stable_door_cut_mask,
        accepted_door_visual_mask=accepted_door_visual_mask,
        accepted_seed_cluster_mask=accepted_seed_cluster_mask,
        door_intersection_dilation_cells=int(cfg.door_intersection_dilation_cells),
        reject_if_intersects_door_visual=bool(getattr(cfg, "reject_step2_if_intersects_door_visual", False)),
        reject_if_intersects_door_seed=bool(getattr(cfg, "reject_step2_if_intersects_door_seed", False)),
    )
    previous_stable_step2_separator_map = (
        separator_memory.stable_mask(shape)
        if separator_memory is not None
        else np.zeros(shape, dtype=bool)
    )

    line_cfg = replace(
        cfg.line_extension,
        min_extension_m=0.40 if cfg.line_extension.min_extension_m is None else float(cfg.line_extension.min_extension_m),
        max_extension_m=1.60 if cfg.line_extension.max_extension_m is None else float(cfg.line_extension.max_extension_m),
        max_probe_m=min(float(cfg.line_extension.max_probe_m), 1.60),
        allow_hit_virtual_door_on_pass2=False,
        require_free_between_start_and_hit=True,
        min_free_cells_between_start_and_hit=max(3, int(cfg.line_extension.min_free_cells_between_start_and_hit)),
    )
    step2_extensions, step2_extension_debug = extend_step2_wall_lines(
        filtered_lines=step2_line_pool.source_lines,
        free_after_step1=free_after_step1,
        step1_wall_mask=step2_line_pool.target_wall_map,
        unknown_after_step1=unknown_after_step1,
        door_mask=step2_door_reject_mask,
        resolution_m=float(resolution_m),
        config=line_cfg,
        reject_if_intersects_door=bool(cfg.reject_step2_if_intersects_door),
        door_intersection_dilation_cells=int(cfg.door_intersection_dilation_cells),
    )
    hit_validation_debug = reject_step2_extension_hits_v23(
        step2_extensions,
        forbidden_frontier_residual_map=forbidden_residual_support,
        frontier_unknown_band=frontier_unknown_band,
        target_wall_map=step2_line_pool.target_wall_map,
        target_source_map=step2_line_pool.target_source_map,
        anchor_projected_wall_map=anchor_projected_wall_map,
        existing_separator_map=previous_stable_step2_separator_map,
        shape=shape,
        reject_if_hits_frontier_residual=bool(getattr(cfg, "reject_step2_if_hits_frontier_residual_wall", True)),
        reject_if_crosses_frontier_unknown_band=bool(getattr(cfg, "reject_step2_if_crosses_frontier_unknown_band", True)),
        reject_if_intersects_existing_separator=bool(getattr(cfg, "reject_step2_if_intersects_existing_extension", True)),
    )
    step2_extension_debug = {**dict(step2_extension_debug), **hit_validation_debug}
    step2_candidates, step2_candidate_debug = build_step2_separator_candidates_from_extensions(
        step2_extensions,
        accepted_virtual_targets=None,
        resolution_m=float(resolution_m),
        config=cfg.door_neck,
        start_id=1,
    )
    intersection_policy = str(getattr(cfg, "extension_intersection_policy", "reject") or "reject").strip().lower()
    debug_intersection_candidates: list[SeparatorCandidate] = []
    intersection_debug: dict[str, object] = {
        "voxel_step2_extension_intersection_policy": str(intersection_policy),
        "voxel_step2_extension_intersection_fallback_enabled": bool(getattr(cfg, "enable_extension_intersection_fallback", False)),
    }
    if intersection_policy in {"candidate", "debug_only"} or bool(getattr(cfg, "enable_extension_intersection_fallback", False)):
        debug_intersection_candidates, debug_intersection_target_map, intersection_debug_raw = build_door_neck_candidates_from_extension_intersections(
            step2_extensions,
            free_clean=free_after_step1,
            unknown_clean=unknown_after_step1,
            resolution_m=float(resolution_m),
            line_config=line_cfg,
            door_config=cfg.door_neck,
            start_id=int(len(step2_candidates) + 1),
        )
        intersection_debug.update(dict(intersection_debug_raw))
        intersection_debug["voxel_step2_debug_intersection_candidate_count"] = int(len(debug_intersection_candidates))
    else:
        debug_intersection_target_map = np.zeros(shape, dtype=bool)
    if intersection_policy == "candidate" and bool(getattr(cfg, "enable_extension_intersection_fallback", False)):
        intersection_candidates = list(debug_intersection_candidates)
        intersection_target_map = np.asarray(debug_intersection_target_map, dtype=bool)
        for candidate in intersection_candidates:
            candidate.kind = "line_extension_corridor_separator"
            candidate.debug["kind_detail"] = "corridor_separator"
            candidate.debug["candidate_source"] = "step2_extension_intersection"
        intersection_candidates, intersection_pre_rejected = _reject_step2_candidates_intersecting_doors(
            intersection_candidates,
            door_block_mask=step2_door_reject_mask,
            shape=shape,
        )
    else:
        intersection_candidates = []
        intersection_pre_rejected = []
        intersection_target_map = np.zeros(shape, dtype=bool)
        intersection_debug["voxel_step2_extension_intersection_candidates_suppressed"] = int(len(debug_intersection_candidates))
    all_step2_candidates = [*step2_candidates, *intersection_candidates]
    all_step2_candidates, conflict_rejected_step2, conflict_debug = reject_step2_candidate_conflicts_v22(
        all_step2_candidates,
        accepted_door_mask=door_cut_mask,
        stable_door_mask=stable_door_cut_mask,
        accepted_step2_memory_mask=previous_stable_step2_separator_map,
        real_wall_map=step2_line_pool.target_wall_map,
        shape=shape,
        cfg=cfg,
    )
    topology_cfg = replace(
        cfg.topology_test,
        reject_if_side_width_cells_leq=int(cfg.reject_step2_if_tiny_side_width_cells_leq),
        corridor_min_split_area_m2=float(cfg.voxel_step2_topology.corridor_min_split_area_m2),
        corridor_min_new_component_width_m=float(cfg.voxel_step2_topology.corridor_min_new_component_width_m),
        corridor_reject_tiny_side_width_cells_leq=int(cfg.voxel_step2_topology.corridor_reject_tiny_side_width_cells_leq),
        corridor_tiny_side_min_area_m2=float(cfg.voxel_step2_topology.corridor_tiny_side_min_area_m2),
        corridor_tiny_side_min_length_m=float(cfg.voxel_step2_topology.corridor_tiny_side_min_length_m),
        corridor_accept_long_narrow_side=bool(cfg.voxel_step2_topology.corridor_accept_long_narrow_side),
        corridor_local_topology_radius_cells=int(cfg.voxel_step2_topology.corridor_local_topology_radius_cells),
    )
    accepted_step2, rejected_step2, accepted_step2_map, _raw_topology_labels, topology_debug = greedily_select_separators(
        all_step2_candidates,
        free_clean=free_after_step1,
        unknown_clean=unknown_after_step1,
        wall_candidate_clean=step2_line_pool.target_wall_map | intersection_target_map,
        corridor_skeleton=np.zeros(shape, dtype=bool),
        resolution_m=float(resolution_m),
        config=topology_cfg,
    )
    rejected_step2 = [*intersection_pre_rejected, *conflict_rejected_step2, *rejected_step2]
    step2_candidate_map = _rasterize_candidates(all_step2_candidates, shape)
    step2_partition_cut_accepted_map, step2_partition_cut_candidate_from_accepted_map, step2_partition_cut_debug = build_step2_partition_cut_v16(
        accepted_step2,
        accepted_topology_map=accepted_step2_map,
        base_partition_free=base_partition_free,
        partition_unknown=unknown_after_step1,
        real_wall_barrier=real_wall_barrier_for_partition,
        shape=shape,
        max_unknown_bridge_cells=2,
        max_nonfree_bridge_cells=1,
    )
    step2_partition_cut_candidate_map = step2_candidate_map.astype(bool)
    step2_extension_separator_map = step2_partition_cut_accepted_map & base_partition_free
    if separator_memory is not None:
        separator_memory_result = separator_memory.update(
            step2_extension_separator_map,
            step=int(step),
            update_index=int(door_memory_update_index),
            shape=shape,
            contradiction_wall_map=np.asarray(evidence.wall_xy, dtype=bool),
        )
        stable_step2_separator_map = np.asarray(separator_memory_result.stable_separator_mask, dtype=bool)
        separator_memory_debug = dict(separator_memory_result.debug)
    else:
        stable_step2_separator_map = np.zeros(shape, dtype=bool)
        separator_memory_debug = {
            "voxel_separator_memory_track_count": 0,
            "voxel_separator_memory_update_index": int(door_memory_update_index),
            "voxel_stable_step2_separator_cells": 0,
        }
    final_step2_separator_map = step2_extension_separator_map | stable_step2_separator_map
    final_virtual_separator_map = door_cut_mask | final_step2_separator_map
    partition_free_for_label = (base_partition_free | accepted_seed_for_partition) & ~final_virtual_separator_map
    partition_free = partition_free_for_label.copy()
    labels, _count = ndimage.label(partition_free, structure=conn(int(cfg.final_connectivity)))
    labels = relabel_compact(labels.astype(np.int32))
    labels[unknown_after_step1] = 0
    labels[~partition_free_for_label] = 0
    final_separator_map = real_wall_barrier_for_partition | door_cut_mask | final_step2_separator_map

    boundary_source = np.zeros(shape, dtype=np.uint8)
    boundary_source[real_wall_barrier_for_partition] = 1
    boundary_source[step1_gap_fill_map] = 4
    boundary_source[door_cut_mask] = 2
    boundary_source[final_step2_separator_map] = 3
    step2_layers = _extension_layers(step2_extensions, shape)
    step2_topology_rejected_map = _rasterize_candidates(rejected_step2, shape)
    step2_reject_reason_map = _step2_reject_reason_map(step2_extensions, rejected_step2, shape)
    step2_intersection_candidate_map = _rasterize_candidates(intersection_candidates, shape)
    corridor_neck_source_lines = [
        line for line in step2_line_pool.source_lines if str(line.debug.get("step2_source_role", "")) == "corridor_neck_source"
    ]
    corridor_neck_source_map = filtered_wall_line_mask(corridor_neck_source_lines, shape)
    raw_seed_not_suppressing_step2_map = raw_door_seed_band_for_debug & ~door_suppression_band
    step2_suppression_from_accepted_door_map = door_suppression_band.astype(bool)
    step2_stage_maps = Step2StageMaps(
        extension_hits_all_map=step2_layers["all"].astype(bool),
        extension_hits_pre_topology_map=step2_layers["accepted"].astype(bool),
        separator_candidates_pre_topology_map=step2_candidate_map.astype(bool),
        topology_rejected_separator_map=step2_topology_rejected_map.astype(bool),
        accepted_separator_map=accepted_step2_map.astype(bool),
        accepted_partition_cut_map=final_step2_separator_map.astype(bool),
    )
    step2_partition_cut_empty_count = int(len(accepted_step2)) if np.any(accepted_step2_map) and not np.any(step2_extension_separator_map) else 0
    display_wall_map = (
        np.asarray(evidence.wall_xy, dtype=bool)
        | projected_wall_map
        | step1_gap_fill_map
    )
    layers = {
        "voxel_nav_free_xy": np.asarray(navigation_free_mask, dtype=bool),
        "voxel_nav_occupied_xy": np.asarray(navigation_obstacle_mask, dtype=bool),
        "voxel_nav_observed_xy": ~np.asarray(unknown_mask, dtype=bool),
        "voxel_nav_unknown_xy": np.asarray(unknown_mask, dtype=bool),
        "voxel_vertical_free_xy": evidence.vertical_free_xy,
        "voxel_wall_xy": evidence.wall_xy,
        "voxel_display_wall_xy": display_wall_map,
        "voxel_wall_projected_xy": projected_wall_map,
        "voxel_projected_wall_map": projected_wall_map,
        "voxel_projected_wall_display_map": projected_wall_map,
        "voxel_projected_wall_anchor_map": anchor_projected_wall_map,
        "voxel_projected_wall_step2_source_map": step2_source_projected_wall_map,
        "voxel_projected_corridor_neck_source_map": corridor_neck_source_map,
        "voxel_projected_structural_wall_map": projected_wall_map,
        "voxel_anchor_projected_wall_map": anchor_projected_wall_map,
        "voxel_unknown_xy": evidence.unknown_xy,
        "voxel_unknown_dominant_xy": np.asarray(evidence.unknown_dominant_xy, dtype=bool),
        "voxel_structural_wall_seed_xy": np.asarray(evidence.structural_wall_seed_xy, dtype=bool),
        "voxel_structural_wall_ratio_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_wall_ratio_raw_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_wall_line_support_xy": np.asarray(evidence.wall_line_support_xy, dtype=bool),
        "voxel_wall_line_support_raw_xy": np.asarray(evidence.wall_line_support_raw_xy, dtype=bool),
        "voxel_wall_line_support_strong_xy": np.asarray(evidence.wall_line_support_strong_xy, dtype=bool),
        "voxel_wall_line_support_conflict_xy": np.asarray(evidence.wall_line_support_conflict_xy, dtype=bool),
        "voxel_wall_line_support_near_free_boundary_xy": np.asarray(evidence.wall_line_support_near_free_boundary_xy, dtype=bool),
        "voxel_wall_line_support_rejected_furniture_xy": np.asarray(evidence.wall_line_support_rejected_furniture_xy, dtype=bool),
        "voxel_wall_line_support_rejected_unknown_xy": np.asarray(evidence.wall_line_support_rejected_unknown_xy, dtype=bool),
        "voxel_wall_line_support_weight_xy": np.asarray(evidence.wall_line_support_weight_xy, dtype=np.float32),
        "voxel_wall_line_support_rejected_by_free_xy": np.asarray(evidence.wall_line_support_rejected_by_free_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_unknown_xy": np.asarray(evidence.wall_line_support_rejected_by_unknown_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_observed_xy": np.asarray(evidence.wall_line_support_rejected_by_observed_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_nav_edge_xy": np.asarray(evidence.wall_line_support_rejected_by_nav_edge_xy, dtype=bool),
        "voxel_wall_projection_support_input_xy": projection_input,
        "voxel_wall_support_raw_occupied_xy": np.asarray(evidence.wall_support_raw_occupied_xy, dtype=bool),
        "voxel_wall_support_known_xy": np.asarray(evidence.wall_support_known_xy, dtype=bool),
        "voxel_wall_support_for_projection_xy": np.asarray(evidence.wall_support_for_projection_xy, dtype=bool),
        "voxel_wall_support_rejected_by_unknown_ratio_xy": np.asarray(evidence.wall_support_unknown_rejected_xy, dtype=bool),
        "voxel_wall_support_rejected_by_nav_unknown_xy": np.asarray(evidence.wall_support_nav_unknown_rejected_xy, dtype=bool),
        "voxel_wall_support_rejected_by_frontier_band_xy": np.asarray(evidence.wall_support_frontier_band_rejected_xy, dtype=bool),
        "voxel_wall_support_free_conflict_xy": np.asarray(evidence.wall_support_free_conflict_xy, dtype=bool),
        "voxel_frontier_unknown_band_xy": np.asarray(evidence.frontier_unknown_band_xy, dtype=bool),
        "voxel_v23_protected_structural_wall_band_xy": protected_structural_wall_band,
        "voxel_v23_frontier_residual_band_xy": np.asarray(evidence.debug.get("voxel_v23_frontier_residual_band_xy", forbidden_residual_support), dtype=bool),
        "voxel_v23_nav_unknown_band_xy": np.asarray(evidence.debug.get("voxel_v23_nav_unknown_band_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_v23_unknown_dominant_band_xy": np.asarray(evidence.debug.get("voxel_v23_unknown_dominant_band_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_v23_free_boundary_band_xy": np.asarray(evidence.debug.get("voxel_v23_free_boundary_band_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_strong_structural_support_xy": np.asarray(evidence.strong_structural_support_xy if evidence.strong_structural_support_xy is not None else projection_seed, dtype=bool),
        "voxel_bridge_only_support_xy": np.asarray(evidence.bridge_only_support_xy if evidence.bridge_only_support_xy is not None else projection_bridge, dtype=bool),
        "voxel_forbidden_frontier_residual_support_xy": forbidden_frontier_residual,
        "voxel_forbidden_unknown_boundary_support_xy": forbidden_unknown_boundary,
        "voxel_free_conflict_support_xy": np.asarray(evidence.free_conflict_support_xy if evidence.free_conflict_support_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_protected_structural_wall_band_xy": protected_structural_wall_band,
        "voxel_support_seed_for_projection_xy": projection_seed,
        "voxel_support_bridge_for_projection_xy": projection_bridge,
        "voxel_support_for_projection_display_xy": projection_input,
        "voxel_support_for_step2_target_xy": np.asarray(evidence.support_for_step2_target_xy if evidence.support_for_step2_target_xy is not None else partition_maps.step2_target_wall_map, dtype=bool),
        "voxel_support_for_step2_source_xy": np.asarray(evidence.support_for_step2_source_xy if evidence.support_for_step2_source_xy is not None else projection_seed, dtype=bool),
        "voxel_wall_rejected_by_free_xy": np.asarray(evidence.wall_rejected_by_free_xy, dtype=bool),
        "voxel_wall_rejected_by_unknown_xy": np.asarray(evidence.wall_rejected_by_unknown_xy, dtype=bool),
        "voxel_nonstructural_occupied_xy": np.asarray(evidence.nonstructural_occupied_xy, dtype=bool),
        "voxel_small_unknown_hole_filled_map": np.asarray(evidence.small_unknown_hole_filled_xy, dtype=bool),
        "voxel_wall_support_unknown_gated_xy": np.asarray(evidence.wall_support_unknown_gated_xy, dtype=bool),
        "voxel_wall_support_rejected_unknown_xy": np.asarray(evidence.wall_support_rejected_unknown_xy, dtype=bool),
        "voxel_free_raw_xy": np.asarray(evidence.debug.get("voxel_free_raw_xy", evidence.vertical_free_xy), dtype=bool),
        "voxel_wall_raw_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_sensor_range_count_xy": np.asarray(evidence.debug.get("voxel_sensor_range_count_xy", np.zeros(shape, dtype=np.uint16)), dtype=np.uint16),
        "voxel_sensor_range_ratio_xy": np.asarray(evidence.debug.get("voxel_sensor_range_ratio_xy", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_sensor_in_range_unknown_count_xy": np.asarray(evidence.debug.get("voxel_sensor_in_range_unknown_count_xy", np.zeros(shape, dtype=np.uint16)), dtype=np.uint16),
        "voxel_sensor_outside_range_unknown_count_xy": np.asarray(evidence.debug.get("voxel_sensor_outside_range_unknown_count_xy", np.zeros(shape, dtype=np.uint16)), dtype=np.uint16),
        "voxel_in_range_unknown_ratio_xy": np.asarray(evidence.debug.get("voxel_in_range_unknown_ratio_xy", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_outside_unknown_ratio_xy": np.asarray(evidence.debug.get("voxel_outside_unknown_ratio_xy", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_generalized_occupied_count_xy": np.asarray(evidence.debug.get("voxel_generalized_occupied_count_xy", np.zeros(shape, dtype=np.uint16)), dtype=np.uint16),
        "voxel_generalized_occupied_ratio_xy": np.asarray(evidence.debug.get("voxel_generalized_occupied_ratio_xy", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_generalized_raw_xy": np.asarray(evidence.debug.get("voxel_wall_generalized_raw_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_wall_actual_ratio_raw_xy": np.asarray(evidence.debug.get("voxel_wall_actual_ratio_raw_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_wall_actual_occupied_requirement_xy": np.asarray(evidence.debug.get("voxel_wall_actual_occupied_requirement_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_wall_rejected_by_outside_unknown_xy": np.asarray(evidence.debug.get("voxel_wall_rejected_by_outside_unknown_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_wall_from_in_range_unknown_xy": np.asarray(evidence.debug.get("voxel_wall_from_in_range_unknown_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_roomseg_nav_obstacle_suppressed_by_vertical_free_xy": np.asarray(evidence.debug.get("voxel_roomseg_nav_obstacle_suppressed_by_vertical_free_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_outside_unknown_dominant_xy": np.asarray(evidence.debug.get("voxel_outside_unknown_dominant_xy", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_occupied_any_xy": np.asarray(evidence.occupied_any_xy, dtype=bool),
        "voxel_raw_occupied_wall_support_xy": np.asarray(evidence.raw_occupied_wall_support_xy, dtype=bool),
        "voxel_strict_raw_wall_xy": np.asarray(evidence.strict_raw_wall_xy, dtype=bool),
        "voxel_wall_suppressed_by_free_xy": np.asarray(evidence.wall_suppressed_by_free_xy, dtype=bool),
        "voxel_ratio_wall_debug_xy": np.asarray(evidence.ratio_wall_debug_xy, dtype=bool),
        "voxel_door_anchor_wall_xy": door_anchor_wall_map,
        "voxel_door_anchor_wall_union_map": door_anchor_wall_map,
        "voxel_door_anchor_source_map": door_anchor_source_map,
        "voxel_wall_anchor_support_map": partition_maps.wall_anchor_support_map,
        "voxel_door_anchor_wall_map": partition_maps.door_anchor_wall_map,
        "voxel_pre_door_wall_anchor_support_map": pre_partition_maps.wall_anchor_support_map,
        "voxel_pre_door_step2_target_wall_map": pre_partition_maps.step2_target_wall_map,
        "voxel_final_step2_target_wall_map": partition_maps.step2_target_wall_map,
        "voxel_step2_target_wall_union_pre_final_map": step2_target_wall_override,
        "voxel_partition_real_wall_map": partition_maps.partition_real_wall_map,
        "voxel_partition_real_wall_removed_by_seed_carve_map": partition_maps.removed_by_seed_carve_map,
        "voxel_pre_door_partition_real_wall_map": pre_partition_maps.partition_real_wall_map,
        "voxel_pre_door_partition_real_wall_removed_by_seed_carve_map": pre_partition_maps.removed_by_seed_carve_map,
        "voxel_partition_unknown": partition_maps.partition_unknown,
        "voxel_wall_projection_support_map": np.asarray(wall_projection.support_map, dtype=bool),
        "voxel_wall_projection_rejected_support_map": np.asarray(wall_projection.rejected_support_map, dtype=bool),
        "voxel_wall_projection_forbidden_unknown_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_forbidden_unknown_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_wall_projection_accumulator_h_votes": np.asarray(wall_projection.debug.get("voxel_wall_projection_accumulator_h_votes", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_projection_accumulator_v_votes": np.asarray(wall_projection.debug.get("voxel_wall_projection_accumulator_v_votes", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_projection_reject_reason_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_free_wall_conflict_xy": np.asarray(evidence.free_wall_conflict_xy, dtype=bool),
        "voxel_vertical_observed_xy": evidence.active_observed_xy,
        "voxel_active_observed_xy": evidence.active_observed_xy,
        "voxel_line_supported_wall_map": raw_line_map,
        "voxel_filtered_wall_line_mask": filtered_line_map,
        "voxel_extension_seed_wall_line_mask": extension_seed_line_map,
        "voxel_wall_base_map": wall_base_pre_step1,
        "voxel_door_seed_mask": door_seed_mask,
        "voxel_raw_door_seed_mask": door_seed_masks_for_step2.raw_seed_mask,
        "voxel_raw_door_seed_band_for_debug": raw_door_seed_band_for_debug,
        "voxel_raw_door_seed_not_suppressing_step2_map": raw_seed_not_suppressing_step2_map,
        "voxel_step2_suppression_from_accepted_door_map": step2_suppression_from_accepted_door_map,
        "voxel_step2_suppression_band": door_suppression_band,
        "voxel_accepted_seed_cluster_mask": accepted_seed_cluster_mask,
        "voxel_door_seed_wall_carve_mask": accepted_seed_for_partition,
        "voxel_door_seed_component_map": door_seed_result.door_seed_component_map,
        "voxel_door_extension_attempt_all_mask": door_completion.door_extension_attempt_all_mask,
        "voxel_door_extension_attempt_selected_mask": np.asarray(door_completion.debug.get("voxel_door_extension_attempt_selected_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_extension_attempt_rejected_mask": door_completion.door_extension_attempt_rejected_mask,
        "voxel_door_extension_attempt_reason_map": np.asarray(door_completion.debug.get("voxel_door_extension_attempt_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_candidate_lines_map": door_completion.door_centerline_candidate_mask,
        "voxel_door_rejected_lines_map": door_completion.rejected_door_centerline_mask,
        "voxel_door_centerline_mask": door_visual_mask,
        "voxel_door_centerline_visual_mask": door_visual_mask,
        "voxel_door_current_centerline_visual_mask": current_door_visual_mask,
        "voxel_door_visual_only_mask": np.asarray(door_completion.debug.get("voxel_door_visual_only_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_partition_cut_candidate_mask": door_completion.door_partition_cut_candidate_mask,
        "voxel_door_partition_cut_accepted_mask": door_cut_mask,
        "voxel_door_current_cut_mask": current_door_cut_mask,
        "voxel_stable_door_cut_mask": stable_door_cut_mask,
        "voxel_stable_door_visual_mask": stable_door_visual_mask,
        "voxel_final_door_cut_mask": door_cut_mask,
        "voxel_door_partition_cut_rejected_mask": np.asarray(door_completion.debug.get("voxel_door_partition_cut_rejected_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_door_partition_reject_reason_map": np.asarray(door_completion.debug.get("voxel_door_partition_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_door_reject_reason_map": door_completion.door_reject_reason_map,
        "voxel_door_cut_mask": door_cut_mask,
        "voxel_door_topology_accepted_cut_mask": np.asarray(door_completion.debug.get("voxel_door_topology_accepted_cut_mask", current_door_cut_mask), dtype=bool),
        "voxel_door_topology_warning_cut_mask": np.asarray(door_completion.debug.get("voxel_door_topology_warning_cut_mask", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_accepted_door_centerline_mask": accepted_door_visual_mask,
        "voxel_rejected_door_centerline_mask": door_completion.rejected_door_centerline_mask,
        "voxel_step1_wall_gap_fill_map": step1_gap_fill_map,
        "voxel_wall_after_step1_map": step1_wall_mask,
        "voxel_step1_completed_wall_map": step1_wall_mask,
        "voxel_real_wall_barrier_map": real_wall_barrier_map,
        "voxel_real_wall_barrier_for_partition": real_wall_barrier_for_partition,
        "voxel_base_partition_free": base_partition_free,
        "voxel_step2_source_line_map": step2_line_pool.source_line_map,
        "voxel_step2_corridor_neck_source_line_map": corridor_neck_source_map,
        "voxel_step2_target_wall_map": step2_line_pool.target_wall_map,
        "voxel_step2_target_source_map": step2_line_pool.target_source_map,
        "voxel_step2_door_reject_mask": step2_door_reject_mask,
        "voxel_step2_extension_candidate_map": step2_layers["all"],
        "voxel_step2_extension_hits_all_map": step2_stage_maps.extension_hits_all_map,
        "voxel_step2_extension_hits_pre_topology_map": step2_stage_maps.extension_hits_pre_topology_map,
        "voxel_step2_separator_candidates_pre_topology_map": step2_stage_maps.separator_candidates_pre_topology_map,
        "voxel_step2_partition_cut_candidate_map": step2_partition_cut_candidate_map,
        "voxel_step2_partition_cut_candidate_from_accepted_map": step2_partition_cut_candidate_from_accepted_map,
        "voxel_step2_partition_cut_accepted_map": final_step2_separator_map,
        "voxel_step2_current_partition_cut_accepted_map": step2_extension_separator_map,
        "voxel_stable_step2_separator_map": stable_step2_separator_map,
        "voxel_final_step2_separator_map": final_step2_separator_map,
        "voxel_previous_stable_step2_separator_map": previous_stable_step2_separator_map,
        "voxel_step2_topology_rejected_separator_map": step2_stage_maps.topology_rejected_separator_map,
        "voxel_step2_reject_reason_map": step2_reject_reason_map,
        "voxel_step2_hit_frontier_residual_rejected_map": np.asarray(hit_validation_debug.get("voxel_step2_hit_frontier_residual_rejected_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_step2_v23_rejected_hit_map": np.asarray(hit_validation_debug.get("voxel_step2_v23_rejected_hit_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_step2_accepted_separator_map": step2_stage_maps.accepted_separator_map,
        "voxel_step2_extension_separator_map": final_step2_separator_map,
        "voxel_step2_current_extension_separator_map": step2_extension_separator_map,
        "voxel_step2_intersection_candidate_map": step2_intersection_candidate_map,
        "voxel_step2_intersection_target_map": intersection_target_map,
        "voxel_step2_rejected_extension_map": step2_layers["rejected"] | step2_topology_rejected_map,
        "voxel_boundary_source_map": boundary_source,
        "voxel_partition_free": partition_free,
        "voxel_partition_free_before_label": partition_free_for_label,
        "voxel_final_virtual_separator_map": final_virtual_separator_map,
        "voxel_final_room_label_map": labels,
        "voxel_room_label_map_visual": labels,
        "voxel_final_separator_map": final_separator_map,
        "voxel_final_separator_source_map": boundary_source,
        "final_room_labels": labels,
        "accepted_separators": final_separator_map,
        "rejected_separators": _rasterize_candidates(rejected_step2, shape),
    }
    layers.update(_height_profile_alias_layers(layers))
    report = {
        "step": int(step),
        "backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "algorithm": VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
        "stage_order": ["voxel_evidence", "voxel_door_seed", "wall_projection", "step1_real_wall_gap_fill", "voxel_door_completion", "step2_wall_line_extension", "final_4conn_labels"],
        "wall_segment_count": int(len(segments)),
        "filtered_wall_line_count": int(len(filtered_lines)),
        "extension_seed_wall_line_count": int(len(extension_seed_lines)),
        "voxel_step2_source_line_count": int(step2_line_pool.debug.get("voxel_step2_source_line_count", 0)),
        "voxel_step2_source_line_count_main": int(step2_line_pool.debug.get("voxel_step2_source_line_count_main", 0)),
        "voxel_step2_source_line_count_corridor_neck": int(step2_line_pool.debug.get("voxel_step2_source_line_count_corridor_neck", 0)),
        "voxel_step2_filtered_line_count": int(step2_line_pool.debug.get("voxel_step2_filtered_line_count", 0)),
        "voxel_step2_extension_seed_line_count": int(step2_line_pool.debug.get("voxel_step2_extension_seed_line_count", 0)),
        "voxel_step2_projected_source_line_count": int(step2_line_pool.debug.get("voxel_step2_projected_source_line_count", 0)),
        "voxel_step2_source_line_dedup_count": int(step2_line_pool.debug.get("voxel_step2_source_line_dedup_count", 0)),
        "voxel_step2_target_wall_cells": int(step2_line_pool.debug.get("voxel_step2_target_wall_cells", 0)),
        "voxel_step2_target_wall_pre_door_cells": int(np.count_nonzero(pre_partition_maps.step2_target_wall_map)),
        "voxel_step2_target_wall_final_cells": int(np.count_nonzero(partition_maps.step2_target_wall_map)),
        "voxel_step2_target_wall_union_pre_final_cells": int(np.count_nonzero(step2_target_wall_override)),
        "voxel_step2_target_source_counts": dict(step2_line_pool.debug.get("voxel_step2_target_source_counts", {}) or {}),
        "voxel_wall_raw_cells": int(np.count_nonzero(evidence.structural_wall_ratio_xy)),
        "voxel_wall_ratio_raw_cells": int(np.count_nonzero(evidence.structural_wall_ratio_xy)),
        "voxel_raw_occupied_wall_support_cells": int(np.count_nonzero(evidence.raw_occupied_wall_support_xy)),
        "voxel_wall_line_support_cells": int(np.count_nonzero(evidence.wall_line_support_xy)),
        "voxel_wall_line_support_raw_cells": int(np.count_nonzero(evidence.wall_line_support_raw_xy)),
        "voxel_wall_line_support_strong_cells": int(np.count_nonzero(evidence.wall_line_support_strong_xy)),
        "voxel_wall_line_support_conflict_cells": int(np.count_nonzero(evidence.wall_line_support_conflict_xy)),
        "voxel_wall_support_for_projection_cells": int(np.count_nonzero(np.asarray(evidence.wall_support_for_projection_xy, dtype=bool))),
        "voxel_wall_support_rejected_by_unknown_ratio_cells": int(np.count_nonzero(np.asarray(evidence.wall_support_unknown_rejected_xy, dtype=bool))),
        "voxel_wall_support_rejected_by_nav_unknown_cells": int(np.count_nonzero(np.asarray(evidence.wall_support_nav_unknown_rejected_xy, dtype=bool))),
        "voxel_wall_support_rejected_by_frontier_band_cells": int(np.count_nonzero(np.asarray(evidence.wall_support_frontier_band_rejected_xy, dtype=bool))),
        "voxel_frontier_unknown_band_cells": int(np.count_nonzero(np.asarray(evidence.frontier_unknown_band_xy, dtype=bool))),
        "voxel_sensor_range_xy_cells": int(evidence.debug.get("voxel_sensor_range_xy_cells", 0) or 0),
        "voxel_in_range_unknown_cells_xy": int(evidence.debug.get("voxel_in_range_unknown_cells_xy", 0) or 0),
        "voxel_outside_unknown_dominant_cells": int(evidence.debug.get("voxel_outside_unknown_dominant_cells", 0) or 0),
        "voxel_generalized_wall_raw_cells": int(evidence.debug.get("voxel_generalized_wall_raw_cells", 0) or 0),
        "voxel_wall_from_generalized_unknown_cells": int(evidence.debug.get("voxel_wall_from_generalized_unknown_cells", 0) or 0),
        "voxel_wall_rejected_by_outside_unknown_cells": int(evidence.debug.get("voxel_wall_rejected_by_outside_unknown_cells", 0) or 0),
        "voxel_wall_actual_occupied_requirement_cells": int(evidence.debug.get("voxel_wall_actual_occupied_requirement_cells", 0) or 0),
        "voxel_forbidden_frontier_residual_support_cells": int(np.count_nonzero(forbidden_frontier_residual)),
        "voxel_forbidden_unknown_boundary_support_cells": int(np.count_nonzero(forbidden_unknown_boundary)),
        "voxel_bridge_only_support_cells": int(np.count_nonzero(projection_bridge)),
        "voxel_support_seed_for_projection_cells": int(np.count_nonzero(projection_seed)),
        "voxel_support_bridge_for_projection_cells": int(np.count_nonzero(projection_bridge)),
        "voxel_support_for_step2_source_cells": int(np.count_nonzero(np.asarray(evidence.support_for_step2_source_xy if evidence.support_for_step2_source_xy is not None else projection_seed, dtype=bool))),
        "voxel_strict_raw_wall_cells": int(np.count_nonzero(evidence.strict_raw_wall_xy)),
        "voxel_wall_projected_cells": int(np.count_nonzero(projected_wall_map)),
        "voxel_projected_structural_wall_cells": int(np.count_nonzero(projected_wall_map)),
        "voxel_display_wall_cells": int(np.count_nonzero(display_wall_map)),
        "voxel_anchor_projected_wall_cells": int(np.count_nonzero(anchor_projected_wall_map)),
        "voxel_step2_source_projected_wall_cells": int(np.count_nonzero(step2_source_projected_wall_map)),
        "voxel_door_anchor_wall_cells": int(np.count_nonzero(door_anchor_wall_map)),
        "voxel_pre_door_partition_debug": dict(pre_partition_maps.debug),
        **partition_maps.debug,
        "voxel_door_accepted_count": int(door_completion.debug.get("voxel_door_accepted_count", 0)),
        "voxel_door_visual_accepted_count": int(door_completion.debug.get("voxel_door_visual_accepted_count", 0)),
        "voxel_door_partition_accepted_count": int(door_completion.debug.get("voxel_door_partition_accepted_count", 0)),
        "voxel_door_current_cut_cells": int(np.count_nonzero(current_door_cut_mask)),
        "voxel_stable_door_cut_cells": int(np.count_nonzero(stable_door_cut_mask)),
        "voxel_final_door_cut_cells": int(np.count_nonzero(door_cut_mask)),
        **door_memory_debug,
        "voxel_door_rejected_count": int(door_completion.debug.get("voxel_door_rejected_count", 0)),
        "voxel_real_wall_barrier_cells": int(np.count_nonzero(real_wall_barrier_for_partition)),
        "voxel_base_partition_free_cells": int(np.count_nonzero(base_partition_free)),
        "voxel_step1_gap_fill_cells": int(np.count_nonzero(step1_gap_fill_map)),
        "voxel_step2_extension_count": int(len(step2_extensions)),
        "voxel_step2_candidate_count": int(len(all_step2_candidates)),
        "voxel_step2_hit_candidate_count": int(len(step2_candidates)),
        "voxel_step2_extension_hit_count": int(sum(1 for hit in step2_extensions if hit.reject_reason is None)),
        "voxel_step2_intersection_candidate_count": int(len(intersection_candidates)),
        "voxel_step2_intersection_pre_rejected_count": int(len(intersection_pre_rejected)),
        "voxel_step2_accepted_count": int(len(accepted_step2)),
        "voxel_step2_rejected_count": int(len(rejected_step2)),
        "voxel_step2_extension_reject_reason_counts": _extension_reason_counts(step2_extensions),
        "voxel_step2_v23_hit_reject_reason_counts": dict(hit_validation_debug.get("voxel_step2_v23_hit_reject_reason_counts", {}) or {}),
        "voxel_step2_topology_reject_reason_counts": _candidate_reason_counts(rejected_step2),
        "voxel_step2_hit_target_source_counts": dict(Counter(int(v) for v in np.asarray(step2_line_pool.target_source_map, dtype=np.uint8)[step2_layers["accepted"] & step2_line_pool.target_wall_map].tolist())),
        "voxel_step2_blocked_by_raw_seed_count": int(step2_line_pool.debug.get("voxel_step2_blocked_by_raw_seed_count", 0)),
        "voxel_step2_blocked_by_accepted_door_count": int(step2_line_pool.debug.get("voxel_step2_blocked_by_accepted_door_count", 0)),
        "voxel_raw_door_seed_not_suppressing_step2_cells": int(np.count_nonzero(raw_seed_not_suppressing_step2_map)),
        "voxel_step2_suppression_from_accepted_door_cells": int(np.count_nonzero(step2_suppression_from_accepted_door_map)),
        "voxel_step2_partition_cut_empty_count": int(step2_partition_cut_empty_count),
        "voxel_step2_partition_cut_candidate_cells": int(np.count_nonzero(step2_partition_cut_candidate_map)),
        "voxel_step2_partition_cut_accepted_cells": int(np.count_nonzero(step2_extension_separator_map)),
        "voxel_stable_step2_separator_cells": int(np.count_nonzero(stable_step2_separator_map)),
        "voxel_final_step2_separator_cells": int(np.count_nonzero(final_step2_separator_map)),
        "voxel_step2_partition_cut_debug": dict(step2_partition_cut_debug),
        "voxel_step2_reject_reason_counts": _extension_and_candidate_reasons(step2_extensions, rejected_step2),
        **separator_memory_debug,
        **conflict_debug,
        "voxel_step2_candidate_debug_list": [candidate.to_dict() for candidate in [*accepted_step2, *rejected_step2]],
        "voxel_step2_corridor_topology_debug_per_candidate": [
            candidate.to_dict()
            for candidate in [*accepted_step2, *rejected_step2]
            if str(candidate.kind) == "line_extension_corridor_separator"
        ],
        "voxel_room_count": _room_count(labels),
        "final_connectivity": int(cfg.final_connectivity),
        "merge_small_components_enabled": bool(cfg.merge_small_components_enabled),
        "frontier_source": "voxel_vertical_free",
        "frontier_vertical_free_cells": int(np.count_nonzero(evidence.vertical_free_xy)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(evidence.vertical_free_xy)),
        "use_real_wall_as_partition_barrier": bool(cfg.use_real_wall_as_partition_barrier),
        "real_wall_barrier_dilation_cells": int(cfg.real_wall_barrier_dilation_cells),
        "candidates": [candidate.to_dict() for candidate in [*accepted_step2, *rejected_step2]],
    }
    debug = {
        "backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "actual_backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "source_backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "roomseg_backend": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "algorithm": VOXEL_OCCUPANCY_ROOMSEG_ALGORITHM,
        "source": VOXEL_OCCUPANCY_ROOMSEG_BACKEND,
        "context_source": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
        "room_map_mode": VOXEL_OCCUPANCY_ROOMSEG_CONTEXT,
        "strict_fallback_used": False,
        "silent_fallback_used": False,
        "legacy_style_used": False,
        "navigation_obstacle_written": False,
        "step": int(step),
        "resolution_m": float(resolution_m),
        "voxel_room_count": _room_count(labels),
        "room_count": _room_count(labels),
        "merge_small_components_enabled": bool(cfg.merge_small_components_enabled),
        "voxel_show_wall_diagnostics": bool(cfg.voxel_show_wall_diagnostics),
        "voxel_show_raw_wall_support": bool(cfg.voxel_show_raw_wall_support),
        "voxel_show_wall_support_rejected_unknown": bool(cfg.voxel_show_wall_support_rejected_unknown),
        "voxel_show_frontier_unknown_band": bool(cfg.voxel_show_frontier_unknown_band),
        "voxel_show_step2_source_candidates": bool(cfg.voxel_show_step2_source_candidates),
        "voxel_show_step2_rejected_candidates": bool(cfg.voxel_show_step2_rejected_candidates),
        "voxel_show_door_visual_only_candidates": bool(cfg.voxel_show_door_visual_only_candidates),
        "voxel_show_raw_door_seed_band": bool(cfg.voxel_show_raw_door_seed_band),
        "voxel_show_corridor_neck_sources": bool(cfg.voxel_show_corridor_neck_sources),
        "voxel_show_door_seed_reject_reason_overlay": bool(cfg.voxel_show_door_seed_reject_reason_overlay),
        "voxel_step2_reject_reason_counts": report["voxel_step2_reject_reason_counts"],
        "voxel_step2_extension_reject_reason_counts": report["voxel_step2_extension_reject_reason_counts"],
        "voxel_step2_v23_hit_reject_reason_counts": report["voxel_step2_v23_hit_reject_reason_counts"],
        "voxel_step2_topology_reject_reason_counts": report["voxel_step2_topology_reject_reason_counts"],
        "voxel_step2_conflict_reject_reason_counts": report.get("voxel_step2_conflict_reject_reason_counts", {}),
        "voxel_step2_partition_cut_empty_count": int(step2_partition_cut_empty_count),
        "voxel_step2_partition_cut_debug": dict(step2_partition_cut_debug),
        "voxel_step2_corridor_topology_debug_per_candidate": report["voxel_step2_corridor_topology_debug_per_candidate"],
        **step2_line_pool.debug,
        "separator_report": report,
        "filtered_wall_lines_report": {
            "filtered_wall_lines": [line.to_dict() for line in filtered_lines],
            **filter_debug,
        },
        "extension_seed_wall_lines_report": {
            "filtered_wall_lines": [line.to_dict() for line in extension_seed_lines],
            **seed_filter_debug,
        },
        "line_extension_report": {"pass2": step2_extension_debug},
        "voxel_step2_v23_hit_validation": hit_validation_debug,
        "voxel_step1_gap_fill_report": step1_gap_debug,
        "voxel_wall_run_report": wall_run_debug,
        "topology_report": {"final": topology_debug, "step2_candidates": step2_candidate_debug},
        "voxel_step2_intersection_debug": intersection_debug,
        "voxel_pre_door_partition_debug": dict(pre_partition_maps.debug),
        **conflict_debug,
        **separator_memory_debug,
        "accepted_separators": [candidate.to_dict() for candidate in accepted_step2],
        "rejected_separators": [candidate.to_dict() for candidate in rejected_step2],
        "proposal_room_masks": _proposal_masks_debug(labels),
        "frontier_source": "voxel_vertical_free",
        "frontier_vertical_free_cells": int(np.count_nonzero(evidence.vertical_free_xy)),
        "frontier_vertical_observed_cells": int(np.count_nonzero(evidence.active_observed_xy)),
        "frontier_vertical_wall_cells": int(np.count_nonzero(display_wall_map)),
        "frontier_cells_from_vertical_free": int(np.count_nonzero(evidence.vertical_free_xy)),
        "frontier_cells_removed_by_navigation_unreachable": 0,
        **voxel_grid.to_debug_dict(),
        **evidence.debug,
        **wall_projection.debug,
        **_prefixed_projection_debug(anchor_wall_projection.debug, "voxel_anchor_wall_projection"),
        **door_seed_result.debug,
        **door_completion.debug,
        **door_memory_debug,
        **partition_maps.debug,
        **layers,
        "line_walls": wall_debug,
    }
    debug.update(_height_profile_alias_debug(debug))
    if bool(cfg.debug_dump):
        dump = save_online_roomseg_debug(
            out_dir=Path(str(cfg.debug_dir)) / ("voxel_roomseg_step_%06d" % int(step)),
            layers=layers,
            separator_report=report,
            extra_reports={
                "filtered_wall_lines": debug["filtered_wall_lines_report"],
                "line_extension_report": debug["line_extension_report"],
                "topology_report": debug["topology_report"],
            },
            save_layers=True,
            save_candidate_json=True,
        )
        debug["online_roomseg_debug_paths"] = dict(dump.get("paths", {}))
    return VoxelOccupancyDoorWallRoomSegResult(
        room_label_map=labels.astype(np.int32),
        separator_map=final_separator_map.astype(bool),
        wall_map=step1_wall_mask.astype(bool),
        door_cut_map=door_cut_mask.astype(bool),
        step1_wall_gap_fill_map=step1_gap_fill_map.astype(bool),
        step2_extension_separator_map=final_step2_separator_map.astype(bool),
        layers=layers,
        debug=debug,
    )


def build_step2_partition_cut_v16(
    candidates: Sequence[SeparatorCandidate],
    *,
    accepted_topology_map: np.ndarray,
    base_partition_free: np.ndarray,
    partition_unknown: np.ndarray,
    real_wall_barrier: np.ndarray,
    shape: tuple[int, int],
    max_unknown_bridge_cells: int = 2,
    max_nonfree_bridge_cells: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    free = np.asarray(base_partition_free, dtype=bool)
    unknown = np.asarray(partition_unknown, dtype=bool)
    real_wall = np.asarray(real_wall_barrier, dtype=bool)
    accepted_topology = np.asarray(accepted_topology_map, dtype=bool)
    accepted_cut = accepted_topology & free
    candidate_cut = accepted_topology.copy()
    bridge_gap_count = 0
    bridge_cell_count = 0
    for candidate in candidates:
        cells = _ordered_line_cells_rc(candidate.p0_rc, candidate.p1_rc, shape)
        if len(cells) <= 0:
            continue
        valid = np.asarray([bool(free[int(r), int(c)]) for r, c in cells], dtype=bool)
        unknown_flags = np.asarray([bool(unknown[int(r), int(c)] and not real_wall[int(r), int(c)]) for r, c in cells], dtype=bool)
        nonfree_flags = np.asarray([bool((not free[int(r), int(c)]) and not unknown[int(r), int(c)] and not real_wall[int(r), int(c)]) for r, c in cells], dtype=bool)
        bridged, gap_count, bridged_indices = _bridge_step2_line_gaps_v16(
            valid,
            unknown_flags=unknown_flags,
            nonfree_flags=nonfree_flags,
            max_unknown_bridge_cells=int(max_unknown_bridge_cells),
            max_nonfree_bridge_cells=int(max_nonfree_bridge_cells),
        )
        bridge_gap_count += int(gap_count)
        bridge_cell_count += int(len(bridged_indices))
        if not np.any(bridged):
            continue
        line_mask = np.zeros(shape, dtype=bool)
        bridged_cells = cells[np.flatnonzero(bridged)]
        line_mask[bridged_cells[:, 0], bridged_cells[:, 1]] = True
        candidate_cut |= line_mask
        accepted_cut |= line_mask & free
    return accepted_cut.astype(bool), candidate_cut.astype(bool), {
        "step2_partition_cut_v16": True,
        "step2_partition_cut_bridge_gap_count": int(bridge_gap_count),
        "step2_partition_cut_bridge_cell_count": int(bridge_cell_count),
        "step2_partition_cut_candidate_cells_from_accepted": int(np.count_nonzero(candidate_cut)),
        "step2_partition_cut_accepted_cells": int(np.count_nonzero(accepted_cut)),
    }


def _ordered_line_cells_rc(p0_rc: np.ndarray, p1_rc: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
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
    if not np.any(inside):
        return np.zeros((0, 2), dtype=np.int32)
    coords = np.stack([rows[inside], cols[inside]], axis=1)
    keep = np.ones(len(coords), dtype=bool)
    if len(coords) > 1:
        keep[1:] = np.any(coords[1:] != coords[:-1], axis=1)
    return coords[keep].astype(np.int32)


def _bridge_step2_line_gaps_v16(
    valid: np.ndarray,
    *,
    unknown_flags: np.ndarray,
    nonfree_flags: np.ndarray,
    max_unknown_bridge_cells: int,
    max_nonfree_bridge_cells: int,
) -> tuple[np.ndarray, int, list[int]]:
    out = np.asarray(valid, dtype=bool).copy()
    unknown = np.asarray(unknown_flags, dtype=bool)
    nonfree = np.asarray(nonfree_flags, dtype=bool)
    if out.shape != unknown.shape or out.shape != nonfree.shape:
        raise ValueError("step2 bridge arrays must have the same shape")
    bridged: list[int] = []
    gap_count = 0
    idx = 0
    while idx < out.size:
        if bool(out[idx]):
            idx += 1
            continue
        start = idx
        while idx < out.size and not bool(out[idx]):
            idx += 1
        end = idx
        if start == 0 or end >= out.size:
            continue
        unknown_count = int(np.count_nonzero(unknown[start:end]))
        nonfree_count = int(np.count_nonzero(nonfree[start:end]))
        hard_block_count = int((end - start) - unknown_count - nonfree_count)
        if (
            bool(out[start - 1])
            and bool(out[end])
            and hard_block_count == 0
            and unknown_count <= int(max_unknown_bridge_cells)
            and nonfree_count <= int(max_nonfree_bridge_cells)
        ):
            out[start:end] = True
            bridged.extend(range(start, end))
            gap_count += 1
    return out, int(gap_count), bridged


def _rooms_from_labels(labels: np.ndarray, unknown: np.ndarray, config: RoomSegmentationConfig, step: int, debug: Mapping[str, object]) -> list[RoomMask]:
    out: list[RoomMask] = []
    min_cells = max(1, int(config.min_observed_free_cells))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = np.asarray(labels == label, dtype=bool)
        if int(np.count_nonzero(mask)) < min_cells:
            continue
        room = _room_from_mask("pending", mask, unknown, [], config, int(step))
        room.source = VOXEL_OCCUPANCY_ROOMSEG_BACKEND
        room.metadata["label_id"] = int(label)
        room.metadata["proposal_labels"] = [int(label)]
        room.metadata["source_finalization_mode"] = VOXEL_OCCUPANCY_ROOMSEG_BACKEND
        room.metadata["accepted_separator_count"] = _accepted_separator_count(debug)
        out.append(room)
    return out


def _accepted_separator_count(debug: Mapping[str, object]) -> int:
    report = debug.get("separator_report")
    if isinstance(report, Mapping):
        for key in ("voxel_step2_accepted_count", "accepted_count"):
            if key in report:
                try:
                    return int(report[key])
                except (TypeError, ValueError):
                    pass
    value = debug.get("accepted_separators")
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(np.count_nonzero(value))
    try:
        return int(len(value))  # type: ignore[arg-type]
    except TypeError:
        return 0


def _accepted_door_seed_cluster_mask(cluster_map: np.ndarray, clusters: object) -> np.ndarray:
    labels = np.asarray(cluster_map, dtype=np.int32)
    accepted_ids: set[int] = set()
    if isinstance(clusters, Sequence) and not isinstance(clusters, (str, bytes)):
        for item in clusters:
            if not isinstance(item, Mapping):
                continue
            if bool(item.get("accepted_for_completion", False)):
                try:
                    accepted_ids.add(int(item.get("cluster_id", 0)))
                except (TypeError, ValueError):
                    pass
    out = np.zeros_like(labels, dtype=bool)
    for cluster_id in accepted_ids:
        if int(cluster_id) > 0:
            out |= labels == int(cluster_id)
    return out.astype(bool)


def build_voxel_partition_maps(
    *,
    evidence: VoxelRoomsegEvidence,
    strict_raw_wall: np.ndarray,
    projected_wall_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    filtered_line_map: np.ndarray,
    extension_seed_line_map: np.ndarray,
    step1_gap_fill_map: np.ndarray,
    cfg: VoxelOccupancyDoorWallRoomSegConfig,
    door_seed_carve_mask: np.ndarray | None = None,
    door_seed_mask: np.ndarray | None = None,
) -> PartitionMapBundle:
    seed_source = door_seed_carve_mask if door_seed_carve_mask is not None else door_seed_mask
    if seed_source is None:
        raise ValueError("door_seed_carve_mask is required")
    seed = np.asarray(seed_source, dtype=bool)
    strict = np.asarray(strict_raw_wall, dtype=bool)
    projected = np.asarray(projected_wall_map, dtype=bool)
    anchor_projected = np.asarray(anchor_projected_wall_map, dtype=bool)
    filtered = np.asarray(filtered_line_map, dtype=bool)
    extension_seed = np.asarray(extension_seed_line_map, dtype=bool)
    step1 = np.asarray(step1_gap_fill_map, dtype=bool)
    raw_support = np.asarray(evidence.raw_occupied_wall_support_xy, dtype=bool)
    unknown_dominant = np.asarray(getattr(evidence, "unknown_dominant_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    unknown_rejected = np.asarray(getattr(evidence, "wall_rejected_by_unknown_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    forbidden_frontier_residual = np.asarray(getattr(evidence, "forbidden_frontier_residual_support_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    forbidden_unknown_boundary = np.asarray(getattr(evidence, "forbidden_unknown_boundary_support_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    forbidden_residual = forbidden_frontier_residual | forbidden_unknown_boundary
    bridge_only = np.asarray(getattr(evidence, "bridge_only_support_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    vertical_free = np.asarray(evidence.vertical_free_xy, dtype=bool)
    shape = vertical_free.shape
    for name, arr in {
        "strict_raw_wall": strict,
        "projected_wall_map": projected,
        "anchor_projected_wall_map": anchor_projected,
        "filtered_line_map": filtered,
        "extension_seed_line_map": extension_seed,
        "step1_gap_fill_map": step1,
        "door_seed_carve_mask": seed,
    }.items():
        if arr.shape != shape:
            raise ValueError("%s must match voxel evidence shape" % name)
    seed_carve_radius = int(getattr(cfg, "door_seed_wall_carve_radius_cells", cfg.door.seed_cluster_morph_close_radius_cells))
    seed_carve = dilate(seed, max(0, seed_carve_radius))
    structural_forbid = vertical_free | unknown_dominant | forbidden_residual
    clean_structural_wall = strict & ~structural_forbid
    projected_structural_wall = projected & ~structural_forbid
    anchor_projected_structural_wall = anchor_projected & ~structural_forbid
    step1_structural_wall = step1 & ~structural_forbid
    wall_anchor_support_map = clean_structural_wall | projected_structural_wall | anchor_projected_structural_wall | step1_structural_wall
    door_anchor_wall_map = wall_anchor_support_map.copy()
    step2_target_wall_map = clean_structural_wall | projected_structural_wall | step1_structural_wall
    step2_target_wall_map &= ~seed_carve
    step2_target_wall_map &= ~extension_seed
    step2_target_wall_map &= ~bridge_only
    step2_target_wall_map &= ~forbidden_residual
    step2_target_wall_map &= ~unknown_dominant
    partition_real_wall_raw = clean_structural_wall | projected_structural_wall | step1_structural_wall
    if not bool(cfg.use_real_wall_as_partition_barrier):
        partition_real_wall_raw = np.zeros(shape, dtype=bool)
    partition_real_wall_map = np.asarray(partition_real_wall_raw, dtype=bool).copy()
    removed_by_seed_carve = partition_real_wall_map & seed_carve
    partition_real_wall_map &= ~seed_carve
    base_partition_free = vertical_free & ~partition_real_wall_map
    partition_unknown = ~(base_partition_free | partition_real_wall_map)
    old_raw_support_loss = vertical_free & raw_support & ~partition_real_wall_map
    debug = {
        "voxel_partition_maps_stage": "v19_wall_recovery_partition_sources",
        "voxel_wall_anchor_support_cells": int(np.count_nonzero(wall_anchor_support_map)),
        "voxel_door_anchor_wall_cells": int(np.count_nonzero(door_anchor_wall_map)),
        "voxel_door_anchor_allowed_sources": ["structural_wall", "projected_structural_wall", "anchor_projected_wall", "step1"],
        "voxel_step2_target_allowed_sources": ["structural_wall", "projected_structural_wall", "step1"],
        "voxel_door_anchor_unknown_overlap_cells": int(np.count_nonzero(door_anchor_wall_map & partition_unknown)),
        "voxel_step2_target_wall_cells_v18": int(np.count_nonzero(step2_target_wall_map)),
        "voxel_step2_target_wall_cells_v15": int(np.count_nonzero(step2_target_wall_map)),
        "voxel_step2_target_wall_unknown_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & unknown_dominant)),
        "voxel_step2_target_wall_frontier_residual_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & forbidden_frontier_residual)),
        "voxel_step2_target_wall_unknown_boundary_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & forbidden_unknown_boundary)),
        "voxel_step2_target_wall_bridge_only_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & bridge_only)),
        "voxel_step2_target_wall_door_seed_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & seed_carve)),
        "voxel_step2_false_frontier_wall_rejected_map": unknown_rejected.astype(bool),
        "voxel_step2_frontier_residual_wall_rejected_map": forbidden_residual.astype(bool),
        "voxel_partition_extension_seed_excluded_cells": int(np.count_nonzero(extension_seed & ~partition_real_wall_map)),
        "voxel_partition_filtered_line_excluded_cells": int(np.count_nonzero(filtered & ~partition_real_wall_map)),
        "voxel_partition_anchor_projection_excluded_cells": int(np.count_nonzero(anchor_projected & ~partition_real_wall_map)),
        "voxel_partition_real_wall_cells": int(np.count_nonzero(partition_real_wall_map)),
        "voxel_partition_real_wall_removed_by_seed_carve_cells": int(np.count_nonzero(removed_by_seed_carve)),
        "voxel_partition_seed_carve_source": "accepted_seed_cluster_only",
        "voxel_partition_raw_seed_carve_enabled": False,
        "voxel_base_partition_free_cells": int(np.count_nonzero(base_partition_free)),
        "voxel_base_partition_free_lost_to_old_raw_support_cells": int(np.count_nonzero(old_raw_support_loss)),
        "voxel_seed_wall_carve_radius_cells": int(seed_carve_radius),
    }
    return PartitionMapBundle(
        wall_anchor_support_map=wall_anchor_support_map.astype(bool),
        door_anchor_wall_map=door_anchor_wall_map.astype(bool),
        step2_target_wall_map=step2_target_wall_map.astype(bool),
        partition_real_wall_map=partition_real_wall_map.astype(bool),
        base_partition_free=base_partition_free.astype(bool),
        partition_unknown=partition_unknown.astype(bool),
        removed_by_seed_carve_map=removed_by_seed_carve.astype(bool),
        debug=debug,
    )


def build_step2_line_pool(
    *,
    filtered_lines: Sequence[FilteredWallLine],
    extension_seed_lines: Sequence[FilteredWallLine],
    projected_display_lines: Sequence[ProjectedWallLine] | None = None,
    projected_source_lines: Sequence[ProjectedWallLine] | None = None,
    projected_corridor_neck_source_lines: Sequence[ProjectedWallLine] | None = None,
    source_support_map: np.ndarray | None = None,
    frontier_unknown_band: np.ndarray | None = None,
    door_suppression_band: np.ndarray | None = None,
    raw_door_seed_band: np.ndarray | None = None,
    vertical_free_map: np.ndarray | None = None,
    unknown_ratio_map: np.ndarray | None = None,
    source_min_line_length_m: float = 0.60,
    source_min_support_ratio: float = 0.30,
    source_min_support_cells: int = 6,
    source_forbid_frontier_unknown_band: bool = True,
    source_forbid_door_band: bool = True,
    corridor_neck_source_enabled: bool = False,
    corridor_neck_source_min_line_length_m: float = 0.35,
    corridor_neck_source_min_support_cells: int = 4,
    corridor_neck_source_min_support_ratio: float = 0.45,
    corridor_neck_source_must_touch_large_free: bool = True,
    corridor_neck_source_min_adjacent_free_area_cells: int = 60,
    corridor_neck_source_forbid_raw_seed_band: bool = False,
    corridor_neck_source_forbid_accepted_door_band: bool = True,
    corridor_neck_source_max_unknown_ratio_on_line: float = 0.25,
    corridor_neck_source_allow_only_projected_or_strict: bool = True,
    strict_raw_wall: np.ndarray,
    projected_wall_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    step1_completed_wall_map: np.ndarray,
    filtered_line_map: np.ndarray,
    extension_seed_line_map: np.ndarray,
    projected_source_line_map: np.ndarray | None = None,
    shape: tuple[int, int],
    resolution_m: float,
    target_wall_override: np.ndarray | None = None,
) -> Step2LinePool:
    support = np.ones(shape, dtype=bool) if source_support_map is None else np.asarray(source_support_map, dtype=bool)
    frontier_band = np.zeros(shape, dtype=bool) if frontier_unknown_band is None else np.asarray(frontier_unknown_band, dtype=bool)
    door_band = np.zeros(shape, dtype=bool) if door_suppression_band is None else np.asarray(door_suppression_band, dtype=bool)
    raw_seed_band = np.zeros(shape, dtype=bool) if raw_door_seed_band is None else np.asarray(raw_door_seed_band, dtype=bool)
    vertical_free = None if vertical_free_map is None else np.asarray(vertical_free_map, dtype=bool)
    unknown_ratio = np.zeros(shape, dtype=np.float32) if unknown_ratio_map is None else np.asarray(unknown_ratio_map, dtype=np.float32)
    if support.shape != shape or frontier_band.shape != shape or door_band.shape != shape:
        raise ValueError("Step2 source validation masks must match shape")
    if raw_seed_band.shape != shape or unknown_ratio.shape != shape:
        raise ValueError("Step2 raw seed and unknown-ratio masks must match shape")
    if vertical_free is not None and vertical_free.shape != shape:
        raise ValueError("Step2 vertical_free_map must match shape")
    projected_source_filtered = projected_wall_lines_to_filtered_lines(
        list(projected_source_lines or []),
        shape=shape,
        resolution_m=float(resolution_m),
        source_name="projected_step2_source_wall",
    )
    corridor_neck_projected_filtered = projected_wall_lines_to_filtered_lines(
        list(projected_corridor_neck_source_lines or []),
        shape=shape,
        resolution_m=float(resolution_m),
        source_name="projected_corridor_neck_source_wall",
    )
    source_debug: list[dict[str, object]] = []
    rejected_source_debug: list[dict[str, object]] = []

    def validate(line: FilteredWallLine, role: str) -> bool:
        accepted, reason, debug = validate_step2_source_line_v22(
            line,
            shape=shape,
            resolution_m=float(resolution_m),
            support_map=support,
            frontier_unknown_band=frontier_band,
            door_suppression_band=door_band,
            min_line_length_m=float(source_min_line_length_m),
            min_support_ratio=float(source_min_support_ratio),
            min_support_cells=int(source_min_support_cells),
            forbid_frontier_unknown_band=bool(source_forbid_frontier_unknown_band),
            forbid_door_band=bool(source_forbid_door_band),
        )
        line.debug = dict(line.debug)
        line.debug["step2_source_role"] = str(role)
        line.debug["step2_source_validation"] = dict(debug)
        record = {"role": str(role), "accepted": bool(accepted), "reject_reason": reason, **dict(debug)}
        if accepted:
            source_debug.append(record)
        else:
            rejected_source_debug.append(record)
        return bool(accepted)

    def validate_corridor_neck(line: FilteredWallLine) -> bool:
        accepted, reason, debug = validate_corridor_neck_step2_source_line_v25(
            line,
            shape=shape,
            resolution_m=float(resolution_m),
            support_map=support,
            frontier_unknown_band=frontier_band,
            accepted_door_suppression_band=door_band,
            raw_door_seed_band=raw_seed_band,
            vertical_free_map=vertical_free,
            unknown_ratio_map=unknown_ratio,
            min_line_length_m=float(corridor_neck_source_min_line_length_m),
            min_support_ratio=float(corridor_neck_source_min_support_ratio),
            min_support_cells=int(corridor_neck_source_min_support_cells),
            must_touch_large_free=bool(corridor_neck_source_must_touch_large_free),
            min_adjacent_free_area_cells=int(corridor_neck_source_min_adjacent_free_area_cells),
            forbid_raw_seed_band=bool(corridor_neck_source_forbid_raw_seed_band),
            forbid_accepted_door_band=bool(corridor_neck_source_forbid_accepted_door_band),
            max_unknown_ratio_on_line=float(corridor_neck_source_max_unknown_ratio_on_line),
            allow_only_projected_or_strict=bool(corridor_neck_source_allow_only_projected_or_strict),
        )
        line.debug = dict(line.debug)
        line.debug["step2_source_role"] = "corridor_neck_source"
        line.debug["step2_source_validation"] = dict(debug)
        record = {"role": "corridor_neck_source", "accepted": bool(accepted), "reject_reason": reason, **dict(debug)}
        if accepted:
            source_debug.append(record)
        else:
            rejected_source_debug.append(record)
        return bool(accepted)

    filtered_valid = [line for line in filtered_lines if validate(line, "filtered_wall_line")]
    extension_valid = [line for line in extension_seed_lines if validate(line, "extension_seed_line")]
    projected_source_valid = [line for line in projected_source_filtered if validate(line, "projected_step2_source_wall")]
    corridor_neck_valid = [
        line
        for line in corridor_neck_projected_filtered
        if bool(corridor_neck_source_enabled) and validate_corridor_neck(line)
    ]
    source_lines, dedup_count = _dedup_step2_lines(
        [
            *filtered_valid,
            *extension_valid,
            *projected_source_valid,
            *corridor_neck_valid,
        ]
    )
    source_line_map = filtered_wall_line_mask(source_lines, shape)
    target_source = np.zeros(shape, dtype=np.uint8)
    projected_source_map = np.zeros(shape, dtype=bool) if projected_source_line_map is None else np.asarray(projected_source_line_map, dtype=bool)
    target_source[np.asarray(strict_raw_wall, dtype=bool)] = 1
    target_source[np.asarray(projected_wall_map, dtype=bool)] = 2
    target_source[np.asarray(step1_completed_wall_map, dtype=bool)] = 4
    target_source[projected_source_map] = 7
    target_wall = target_source > 0
    if target_wall_override is not None:
        target_wall = np.asarray(target_wall_override, dtype=bool)
        clean_source = np.zeros(shape, dtype=np.uint8)
        clean_source[np.asarray(strict_raw_wall, dtype=bool) & target_wall] = 1
        clean_source[np.asarray(projected_wall_map, dtype=bool) & target_wall] = 2
        clean_source[np.asarray(step1_completed_wall_map, dtype=bool) & target_wall] = 4
        clean_source[projected_source_map & target_wall & (clean_source == 0)] = 7
        target_source = clean_source
    source_counts = {
        "strict_raw_wall": int(np.count_nonzero(target_source == 1)),
        "projected_wall": int(np.count_nonzero(target_source == 2)),
        "anchor_projected_wall": int(np.count_nonzero(target_source == 3)),
        "step1_completed_wall": int(np.count_nonzero(target_source == 4)),
        "filtered_wall_line": int(np.count_nonzero(target_source == 5)),
        "extension_seed_line": int(np.count_nonzero(target_source == 6)),
        "projected_step2_source_wall": int(np.count_nonzero(target_source == 7)),
    }
    debug = {
        "voxel_step2_source_line_count": int(len(source_lines)),
        "voxel_step2_source_line_count_main": int(len(filtered_valid) + len(extension_valid) + len(projected_source_valid)),
        "voxel_step2_source_line_count_corridor_neck": int(len(corridor_neck_valid)),
        "voxel_step2_filtered_line_count": int(len(filtered_valid)),
        "voxel_step2_extension_seed_line_count": int(len(extension_valid)),
        "voxel_step2_projected_display_line_count": 0,
        "voxel_step2_projected_source_line_count": int(len(projected_source_valid)),
        "voxel_step2_corridor_neck_projected_line_count": int(len(corridor_neck_projected_filtered)),
        "voxel_step2_corridor_neck_source_line_count": int(len(corridor_neck_valid)),
        "voxel_step2_rejected_source_line_count": int(len(rejected_source_debug)),
        "voxel_step2_source_line_validation": source_debug[:1024],
        "voxel_step2_rejected_source_line_validation": rejected_source_debug[:1024],
        "voxel_step2_source_reject_reason_counts": dict(Counter(str(item.get("reject_reason")) for item in rejected_source_debug)),
        "voxel_step2_source_line_dedup_count": int(dedup_count),
        "voxel_step2_blocked_by_raw_seed_count": 0,
        "voxel_step2_blocked_by_accepted_door_count": int(
            sum(1 for item in rejected_source_debug if str(item.get("reject_reason")) in {"reject_step2_source_door_frame_duplicate", "reject_corridor_neck_source_accepted_door_band"})
        ),
        "voxel_step2_target_wall_cells": int(np.count_nonzero(target_wall)),
        "voxel_step2_target_source_counts": source_counts,
        "voxel_step2_source_lines": [line.to_dict() for line in source_lines[:1024]],
        "voxel_step2_line_pool_resolution_m": float(resolution_m),
    }
    return Step2LinePool(
        source_lines=list(source_lines),
        source_line_map=source_line_map.astype(bool),
        target_wall_map=target_wall.astype(bool),
        target_source_map=target_source.astype(np.uint8),
        debug=debug,
    )


def build_step2_door_reject_mask(
    *,
    current_door_cut_mask: np.ndarray,
    stable_door_cut_mask: np.ndarray,
    accepted_door_visual_mask: np.ndarray,
    accepted_seed_cluster_mask: np.ndarray,
    door_intersection_dilation_cells: int,
    reject_if_intersects_door_visual: bool = False,
    reject_if_intersects_door_seed: bool = False,
) -> np.ndarray:
    current_cut = np.asarray(current_door_cut_mask, dtype=bool)
    stable_cut = np.asarray(stable_door_cut_mask, dtype=bool)
    shape = current_cut.shape
    if stable_cut.shape != shape:
        raise ValueError("stable_door_cut_mask must match current_door_cut_mask")
    visual = np.asarray(accepted_door_visual_mask, dtype=bool)
    seed = np.asarray(accepted_seed_cluster_mask, dtype=bool)
    if visual.shape != shape or seed.shape != shape:
        raise ValueError("door visual/seed masks must match current_door_cut_mask")
    reject = dilate(current_cut | stable_cut, int(door_intersection_dilation_cells))
    if bool(reject_if_intersects_door_visual):
        reject |= visual
    if bool(reject_if_intersects_door_seed):
        reject |= seed
    return reject.astype(bool)


def reject_step2_extension_hits_v23(
    extensions: Sequence[LineExtensionHit],
    *,
    forbidden_frontier_residual_map: np.ndarray,
    frontier_unknown_band: np.ndarray,
    target_wall_map: np.ndarray,
    target_source_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    existing_separator_map: np.ndarray,
    shape: tuple[int, int],
    reject_if_hits_frontier_residual: bool = True,
    reject_if_crosses_frontier_unknown_band: bool = True,
    reject_if_intersects_existing_separator: bool = True,
    max_frontier_cross_ratio: float = 0.20,
) -> dict[str, object]:
    residual = np.asarray(forbidden_frontier_residual_map, dtype=bool)
    frontier = np.asarray(frontier_unknown_band, dtype=bool)
    target_wall = np.asarray(target_wall_map, dtype=bool)
    target_source = np.asarray(target_source_map, dtype=np.uint8)
    anchor = np.asarray(anchor_projected_wall_map, dtype=bool)
    existing = np.asarray(existing_separator_map, dtype=bool)
    for name, arr in {
        "forbidden_frontier_residual_map": residual,
        "frontier_unknown_band": frontier,
        "target_wall_map": target_wall,
        "target_source_map": target_source,
        "anchor_projected_wall_map": anchor,
        "existing_separator_map": existing,
    }.items():
        if arr.shape != shape:
            raise ValueError("%s must match shape" % name)
    rejected_map = np.zeros(shape, dtype=bool)
    residual_hit_map = np.zeros(shape, dtype=bool)
    reason_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    for hit in extensions:
        if hit.reject_reason is not None:
            continue
        line = rasterize_line(hit.p_start_rc, hit.p_hit_rc, shape)
        line_cells = int(np.count_nonzero(line))
        hit_cell = np.zeros(shape, dtype=bool)
        r, c = np.rint(np.asarray(hit.p_hit_rc, dtype=np.float32)).astype(np.int32).tolist()
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
            hit_cell[int(r), int(c)] = True
        hit_disk = dilate(hit_cell, 1)
        target_overlap = bool(np.any(hit_disk & target_wall))
        residual_overlap = bool(np.any(hit_disk & residual))
        anchor_overlap = bool(np.any(hit_disk & anchor & ~target_wall))
        source_values = [int(v) for v in np.unique(target_source[hit_disk]) if int(v) > 0]
        frontier_ratio = float(np.count_nonzero(line & frontier)) / float(max(1, line_cells))
        existing_overlap = bool(np.any(line & existing & ~target_wall))
        reason = None
        if bool(reject_if_hits_frontier_residual) and residual_overlap and not target_overlap:
            reason = "reject_step2_hit_frontier_residual_wall"
            residual_hit_map |= hit_disk & residual
        elif anchor_overlap and not target_overlap:
            reason = "reject_step2_hit_anchor_only_wall"
        elif bool(reject_if_crosses_frontier_unknown_band) and frontier_ratio > float(max_frontier_cross_ratio):
            reason = "reject_step2_crosses_frontier_unknown_band"
        elif bool(reject_if_intersects_existing_separator) and existing_overlap:
            reason = "reject_step2_intersects_existing_separator"
        record = {
            "extension_id": int(hit.extension_id),
            "source_line_id": int(hit.source_line_id),
            "hit_cell": [int(r), int(c)],
            "target_overlap": bool(target_overlap),
            "target_source_values": source_values,
            "residual_overlap": bool(residual_overlap),
            "anchor_only_overlap": bool(anchor_overlap),
            "frontier_unknown_ratio": float(frontier_ratio),
            "existing_separator_overlap": bool(existing_overlap),
            "reject_reason": reason,
        }
        hit.debug["v23_hit_validation"] = dict(record)
        if reason is not None:
            hit.reject_reason = str(reason)
            rejected_map |= line
            reason_counts[str(reason)] += 1
        records.append(record)
    return {
        "voxel_step2_v23_hit_validation": records[:1024],
        "voxel_step2_v23_hit_reject_reason_counts": dict(reason_counts),
        "voxel_step2_hit_frontier_residual_rejected_map": residual_hit_map.astype(bool),
        "voxel_step2_v23_rejected_hit_map": rejected_map.astype(bool),
    }


def validate_step2_source_line_v22(
    line: FilteredWallLine,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    support_map: np.ndarray,
    frontier_unknown_band: np.ndarray,
    door_suppression_band: np.ndarray,
    min_line_length_m: float,
    min_support_ratio: float,
    min_support_cells: int,
    forbid_frontier_unknown_band: bool = True,
    forbid_door_band: bool = True,
) -> tuple[bool, str | None, dict[str, object]]:
    mask = _filtered_line_mask(line, shape)
    support = np.asarray(support_map, dtype=bool)
    frontier = np.asarray(frontier_unknown_band, dtype=bool)
    door_band = np.asarray(door_suppression_band, dtype=bool)
    if support.shape != shape or frontier.shape != shape or door_band.shape != shape:
        raise ValueError("Step2 source validation masks must match shape")
    support_cells = int(np.count_nonzero(mask & support))
    projected_cells = int(np.count_nonzero(mask))
    support_ratio = float(getattr(line, "support_ratio", 0.0) or 0.0)
    if projected_cells > 0:
        support_ratio = max(support_ratio, float(support_cells) / float(projected_cells))
    length_m = float(getattr(line, "length_m", 0.0) or 0.0)
    if length_m <= 0.0:
        length_m = float(projected_cells) * float(resolution_m)
    overlaps_frontier = bool(np.any(mask & frontier))
    overlaps_door = bool(np.any(mask & door_band))
    debug = {
        "length_m": float(length_m),
        "projected_cells": int(projected_cells),
        "support_cells": int(support_cells),
        "support_ratio": float(support_ratio),
        "overlaps_frontier_unknown_band": bool(overlaps_frontier),
        "overlaps_door_band": bool(overlaps_door),
    }
    if length_m + 1e-9 < float(min_line_length_m) or projected_cells < 8:
        return False, "reject_step2_source_too_short", debug
    if support_ratio + 1e-9 < float(min_support_ratio):
        return False, "reject_step2_source_support_ratio_low", debug
    if support_cells < max(1, int(min_support_cells)):
        return False, "reject_step2_source_support_cells_low", debug
    if bool(forbid_frontier_unknown_band) and overlaps_frontier:
        return False, "reject_step2_source_frontier_unknown_edge", debug
    if bool(forbid_door_band) and overlaps_door:
        return False, "reject_step2_source_door_frame_duplicate", debug
    return True, None, debug


def validate_corridor_neck_step2_source_line_v25(
    line: FilteredWallLine,
    *,
    shape: tuple[int, int],
    resolution_m: float,
    support_map: np.ndarray,
    frontier_unknown_band: np.ndarray,
    accepted_door_suppression_band: np.ndarray,
    raw_door_seed_band: np.ndarray,
    vertical_free_map: np.ndarray | None,
    unknown_ratio_map: np.ndarray,
    min_line_length_m: float,
    min_support_ratio: float,
    min_support_cells: int,
    must_touch_large_free: bool,
    min_adjacent_free_area_cells: int,
    forbid_raw_seed_band: bool,
    forbid_accepted_door_band: bool,
    max_unknown_ratio_on_line: float,
    allow_only_projected_or_strict: bool = True,
) -> tuple[bool, str | None, dict[str, object]]:
    mask = _filtered_line_mask(line, shape)
    support = np.asarray(support_map, dtype=bool)
    frontier = np.asarray(frontier_unknown_band, dtype=bool)
    accepted_door = np.asarray(accepted_door_suppression_band, dtype=bool)
    raw_seed = np.asarray(raw_door_seed_band, dtype=bool)
    unknown_ratio = np.asarray(unknown_ratio_map, dtype=np.float32)
    if support.shape != shape or frontier.shape != shape or accepted_door.shape != shape or raw_seed.shape != shape or unknown_ratio.shape != shape:
        raise ValueError("corridor neck source validation masks must match shape")
    projected_cells = int(np.count_nonzero(mask))
    support_cells = int(np.count_nonzero(mask & support))
    support_ratio = float(getattr(line, "support_ratio", 0.0) or 0.0)
    if projected_cells > 0:
        support_ratio = max(support_ratio, float(support_cells) / float(projected_cells))
    length_m = float(getattr(line, "length_m", 0.0) or 0.0)
    if length_m <= 0.0:
        length_m = float(projected_cells) * float(resolution_m)
    unknown_ratio_on_line = float(np.mean(unknown_ratio[mask])) if projected_cells > 0 else 1.0
    overlaps_frontier = bool(np.any(mask & frontier))
    overlaps_raw_seed = bool(np.any(mask & raw_seed))
    overlaps_accepted_door = bool(np.any(mask & accepted_door))
    adjacent_free_area = 0
    if vertical_free_map is not None:
        free = np.asarray(vertical_free_map, dtype=bool)
        if free.shape != shape:
            raise ValueError("vertical_free_map must match shape")
        labels, count = ndimage.label(free, structure=conn(4))
        around = dilate(mask, 1) & free
        touched = [int(v) for v in np.unique(labels[around]) if int(v) > 0]
        if touched:
            areas = np.bincount(labels.reshape(-1).astype(np.int32))
            adjacent_free_area = int(max(int(areas[label]) for label in touched))
    debug = {
        "length_m": float(length_m),
        "projected_cells": int(projected_cells),
        "support_cells": int(support_cells),
        "support_ratio": float(support_ratio),
        "unknown_ratio_on_line": float(unknown_ratio_on_line),
        "overlaps_frontier_unknown_band": bool(overlaps_frontier),
        "overlaps_raw_door_seed_band": bool(overlaps_raw_seed),
        "overlaps_accepted_door_band": bool(overlaps_accepted_door),
        "adjacent_free_area_cells": int(adjacent_free_area),
        "allow_only_projected_or_strict": bool(allow_only_projected_or_strict),
    }
    if bool(allow_only_projected_or_strict):
        source_text = str(line.debug.get("source", line.endpoint_quality.get("source", ""))).lower()
        if "projected" not in source_text and "strict" not in source_text:
            return False, "reject_corridor_neck_source_not_projected_or_strict", debug
    if length_m + 1e-9 < float(min_line_length_m):
        return False, "reject_corridor_neck_source_too_short", debug
    if support_ratio + 1e-9 < float(min_support_ratio):
        return False, "reject_corridor_neck_source_support_ratio_low", debug
    if support_cells < max(1, int(min_support_cells)):
        return False, "reject_corridor_neck_source_support_cells_low", debug
    if unknown_ratio_on_line > float(max_unknown_ratio_on_line) + 1e-9:
        return False, "reject_corridor_neck_source_unknown_ratio_high", debug
    if overlaps_frontier:
        return False, "reject_corridor_neck_source_frontier_unknown_edge", debug
    if bool(forbid_raw_seed_band) and overlaps_raw_seed:
        return False, "reject_corridor_neck_source_raw_seed_band", debug
    if bool(forbid_accepted_door_band) and overlaps_accepted_door:
        return False, "reject_corridor_neck_source_accepted_door_band", debug
    if bool(must_touch_large_free):
        if vertical_free_map is None:
            return False, "reject_corridor_neck_source_missing_free_map", debug
        if adjacent_free_area < max(1, int(min_adjacent_free_area_cells)):
            return False, "reject_corridor_neck_source_adjacent_free_too_small", debug
    return True, None, debug


def _filtered_line_mask(line: FilteredWallLine, shape: tuple[int, int]) -> np.ndarray:
    return rasterize_line(np.asarray(line.p0_rc, dtype=np.float32), np.asarray(line.p1_rc, dtype=np.float32), shape)


def projected_wall_lines_to_filtered_lines(
    lines: Sequence[ProjectedWallLine],
    *,
    shape: tuple[int, int],
    resolution_m: float,
    source_name: str,
) -> list[FilteredWallLine]:
    out: list[FilteredWallLine] = []
    base_segment_id = 1_000_000 if str(source_name) == "projected_display_wall" else 2_000_000
    for idx, line in enumerate(lines, start=1):
        if str(line.axis) == "h":
            p0 = np.asarray([int(line.line), int(line.start)], dtype=np.float32)
            p1 = np.asarray([int(line.line), int(line.end)], dtype=np.float32)
            theta = 0.0
        elif str(line.axis) == "v":
            p0 = np.asarray([int(line.start), int(line.line)], dtype=np.float32)
            p1 = np.asarray([int(line.end), int(line.line)], dtype=np.float32)
            theta = float(np.pi / 2.0)
        else:
            continue
        p0[0] = float(np.clip(p0[0], 0, int(shape[0]) - 1))
        p1[0] = float(np.clip(p1[0], 0, int(shape[0]) - 1))
        p0[1] = float(np.clip(p0[1], 0, int(shape[1]) - 1))
        p1[1] = float(np.clip(p1[1], 0, int(shape[1]) - 1))
        length_m = float(max(abs(float(p1[0] - p0[0])), abs(float(p1[1] - p0[1]))) + 1.0) * float(resolution_m)
        confidence = float(
            np.clip(
                0.30 + 0.50 * float(line.support_ratio) + 0.20 * float(line.structural_side_score),
                0.0,
                1.0,
            )
        )
        out.append(
            FilteredWallLine(
                line_id=int(base_segment_id + idx),
                p0_rc=p0,
                p1_rc=p1,
                theta=float(theta),
                normal_theta=float(theta + np.pi / 2.0),
                length_m=float(length_m),
                support_ratio=float(line.support_ratio),
                mean_wall_score=float(confidence),
                thickness_m=0.05,
                source_segment_ids=[int(base_segment_id + int(line.line_id))],
                endpoint_quality={"source": str(source_name), "projected_line_id": int(line.line_id)},
                confidence=float(confidence),
                debug={"source": str(source_name), "projected_wall_line": line.to_dict()},
            )
        )
    return out


def _reject_step2_candidates_intersecting_doors(
    candidates: Sequence[SeparatorCandidate],
    *,
    door_block_mask: np.ndarray,
    shape: tuple[int, int],
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate]]:
    door_block = np.asarray(door_block_mask, dtype=bool)
    kept: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    for candidate in candidates:
        mask = candidate.mask(shape)
        if np.any(mask & door_block):
            candidate.accepted = False
            candidate.reject_reason = "reject_intersection_candidate_crosses_accepted_door"
            candidate.debug["pre_topology_reject_reason"] = candidate.reject_reason
            rejected.append(candidate)
        else:
            kept.append(candidate)
    return kept, rejected


def reject_step2_candidate_conflicts_v22(
    candidates: Sequence[SeparatorCandidate],
    *,
    accepted_door_mask: np.ndarray,
    stable_door_mask: np.ndarray,
    accepted_step2_memory_mask: np.ndarray,
    real_wall_map: np.ndarray,
    shape: tuple[int, int],
    cfg: VoxelOccupancyDoorWallRoomSegConfig,
) -> tuple[list[SeparatorCandidate], list[SeparatorCandidate], dict[str, object]]:
    door = np.asarray(accepted_door_mask, dtype=bool) | np.asarray(stable_door_mask, dtype=bool)
    existing = np.asarray(accepted_step2_memory_mask, dtype=bool)
    real_wall = np.asarray(real_wall_map, dtype=bool)
    kept: list[SeparatorCandidate] = []
    rejected: list[SeparatorCandidate] = []
    conflicts: list[dict[str, object]] = []
    for candidate in candidates:
        mask = candidate.mask(shape)
        if bool(getattr(cfg, "reject_step2_if_intersects_door", True)) and np.any(mask & door):
            candidate.accepted = False
            candidate.reject_reason = "reject_step2_intersects_door"
            candidate.debug["pre_topology_reject_reason"] = candidate.reject_reason
            rejected.append(candidate)
            conflicts.append({"candidate_id": int(candidate.candidate_id), "conflict_type": candidate.reject_reason})
            continue
        if bool(getattr(cfg, "reject_step2_if_intersects_existing_extension", True)) and np.any(mask & existing & ~real_wall):
            candidate.accepted = False
            candidate.reject_reason = "reject_step2_intersects_existing_extension"
            candidate.debug["pre_topology_reject_reason"] = candidate.reject_reason
            rejected.append(candidate)
            conflicts.append({"candidate_id": int(candidate.candidate_id), "conflict_type": candidate.reject_reason})
            continue
        kept.append(candidate)

    final_kept: list[SeparatorCandidate] = []
    rejected_ids: set[int] = set()
    masks = [candidate.mask(shape) for candidate in kept]
    for i, cand_a in enumerate(kept):
        if int(cand_a.candidate_id) in rejected_ids:
            continue
        for j in range(i + 1, len(kept)):
            cand_b = kept[j]
            if int(cand_b.candidate_id) in rejected_ids:
                continue
            overlap = masks[i] & masks[j] & ~real_wall
            if not np.any(overlap):
                continue
            score_a = _step2_candidate_pre_score(cand_a, masks[i])
            score_b = _step2_candidate_pre_score(cand_b, masks[j])
            cells = [[int(r), int(c)] for r, c in zip(*np.nonzero(overlap))]
            if abs(score_a - score_b) <= 0.05:
                for cand in (cand_a, cand_b):
                    cand.accepted = False
                    cand.reject_reason = "reject_step2_pairwise_crossing_ambiguous"
                    cand.debug["pre_topology_reject_reason"] = cand.reject_reason
                    rejected_ids.add(int(cand.candidate_id))
                conflicts.append(
                    {
                        "candidate_id_a": int(cand_a.candidate_id),
                        "candidate_id_b": int(cand_b.candidate_id),
                        "conflict_type": "reject_step2_pairwise_crossing_ambiguous",
                        "intersection_cells": cells[:64],
                        "resolved_winner": None,
                        "rejected_candidate_ids": [int(cand_a.candidate_id), int(cand_b.candidate_id)],
                    }
                )
            else:
                loser = cand_a if score_a < score_b else cand_b
                winner = cand_b if score_a < score_b else cand_a
                loser.accepted = False
                loser.reject_reason = "reject_step2_pairwise_crossing_loser"
                loser.debug["pre_topology_reject_reason"] = loser.reject_reason
                rejected_ids.add(int(loser.candidate_id))
                conflicts.append(
                    {
                        "candidate_id_a": int(cand_a.candidate_id),
                        "candidate_id_b": int(cand_b.candidate_id),
                        "conflict_type": "reject_step2_pairwise_crossing_loser",
                        "intersection_cells": cells[:64],
                        "resolved_winner": int(winner.candidate_id),
                        "rejected_candidate_ids": [int(loser.candidate_id)],
                    }
                )
    for candidate in kept:
        if int(candidate.candidate_id) in rejected_ids:
            rejected.append(candidate)
        else:
            final_kept.append(candidate)
    debug = {
        "voxel_step2_conflict_rejected_count": int(len(rejected)),
        "voxel_step2_conflict_graph": conflicts,
        "voxel_step2_conflict_reject_reason_counts": dict(Counter(str(c.reject_reason) for c in rejected if c.reject_reason)),
    }
    return final_kept, rejected, debug


def _step2_candidate_pre_score(candidate: SeparatorCandidate, mask: np.ndarray) -> float:
    source_conf = float(candidate.debug.get("source_line_confidence", candidate.debug.get("confidence", 0.5)) or 0.5)
    topology = float(candidate.debug.get("topology_gain_score", 0.5) or 0.5)
    length_score = min(1.0, float(np.count_nonzero(mask)) / 16.0)
    unknown_ratio = float(candidate.debug.get("unknown_ratio", 0.0) or 0.0)
    return float(0.35 * source_conf + 0.20 * topology + 0.20 * length_score - 0.30 * unknown_ratio)


def _dedup_step2_lines(lines: Sequence[FilteredWallLine]) -> tuple[list[FilteredWallLine], int]:
    kept: list[FilteredWallLine] = []
    dropped = 0
    for line in lines:
        duplicate_index = None
        for idx, old in enumerate(kept):
            if _step2_lines_equivalent(line, old):
                duplicate_index = idx
                break
        if duplicate_index is None:
            kept.append(line)
            continue
        dropped += 1
        old = kept[int(duplicate_index)]
        if (float(line.confidence), float(line.length_m)) > (float(old.confidence), float(old.length_m)):
            kept[int(duplicate_index)] = line
    for idx, line in enumerate(kept, start=1):
        line.debug = dict(line.debug)
        line.debug["step2_line_pool_id"] = int(idx)
    return kept, int(dropped)


def _step2_lines_equivalent(a: FilteredWallLine, b: FilteredWallLine) -> bool:
    a0 = np.rint(np.asarray(a.p0_rc, dtype=np.float32)).astype(np.int32)
    a1 = np.rint(np.asarray(a.p1_rc, dtype=np.float32)).astype(np.int32)
    b0 = np.rint(np.asarray(b.p0_rc, dtype=np.float32)).astype(np.int32)
    b1 = np.rint(np.asarray(b.p1_rc, dtype=np.float32)).astype(np.int32)
    endpoints_same = max(_cheb(a0, b0), _cheb(a1, b1)) <= 1
    endpoints_reversed = max(_cheb(a0, b1), _cheb(a1, b0)) <= 1
    if not (endpoints_same or endpoints_reversed):
        return False
    angle = abs(float(a.theta) - float(b.theta))
    angle = min(angle, abs(float(np.pi) - angle))
    if float(np.degrees(angle)) >= 5.0:
        return False
    if set(int(v) for v in a.source_segment_ids) & set(int(v) for v in b.source_segment_ids):
        return True
    return endpoints_same or endpoints_reversed


def _cheb(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32))))


def _cells_to_mask(cells: Sequence[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        if 0 <= int(r) < shape[0] and 0 <= int(c) < shape[1]:
            out[int(r), int(c)] = True
    return out


def _mask_major_dir(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size < 2:
        return (0.0, 1.0)
    pts = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=1)
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(pts, full_matrices=False)
        vec = vt[0].astype(np.float32)
    except np.linalg.LinAlgError:
        vec = np.asarray([0.0, 1.0], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return (0.0, 1.0)
    vec = vec / norm
    return (float(vec[0]), float(vec[1]))


def _extension_layers(extensions: Sequence[LineExtensionHit], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    out = {
        "all": np.zeros(shape, dtype=bool),
        "accepted": np.zeros(shape, dtype=bool),
        "rejected": np.zeros(shape, dtype=bool),
    }
    for hit in extensions:
        line = rasterize_line(hit.p_start_rc, hit.p_hit_rc, shape)
        out["all"] |= line
        if hit.reject_reason is None:
            out["accepted"] |= line
        else:
            out["rejected"] |= line
    return out


def _rasterize_candidates(candidates: Sequence[SeparatorCandidate], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for candidate in candidates:
        out |= candidate.mask(shape)
    return out.astype(bool)


def _step2_reject_reason_map(
    extensions: Sequence[LineExtensionHit],
    rejected_candidates: Sequence[SeparatorCandidate],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for hit in extensions:
        if hit.reject_reason is None:
            continue
        mask = rasterize_line(hit.p_start_rc, hit.p_hit_rc, shape)
        out[mask] = _step2_reject_code(str(hit.reject_reason))
    for candidate in rejected_candidates:
        reason = str(candidate.reject_reason or candidate.debug.get("pre_topology_reject_reason") or "step2_candidate_rejected")
        out[candidate.mask(shape)] = _step2_reject_code(reason)
    return out.astype(np.uint8)


def _step2_reject_code(reason: str) -> int:
    values = {
        "reject_intersection_candidate_crosses_accepted_door": 1,
        "candidate_intersects_door": 1,
        "line_extension_crosses_door": 1,
        "topology_no_gain": 2,
        "topology_tiny_side": 3,
        "topology_small_component": 4,
        "door_cut_no_global_gain": 5,
        "door_cut_no_topology_gain": 5,
        "no_hit": 6,
        "hit_too_close": 7,
        "hit_not_wall": 8,
        "reject_step2_intersects_door": 9,
        "reject_step2_intersects_existing_extension": 10,
        "reject_step2_intersects_existing_separator": 10,
        "reject_step2_hits_extension_not_wall": 11,
        "reject_step2_pairwise_crossing_ambiguous": 12,
        "reject_step2_pairwise_crossing_loser": 13,
        "reject_step2_hit_frontier_residual_wall": 14,
        "reject_step2_hit_anchor_only_wall": 15,
        "reject_step2_crosses_frontier_unknown_band": 16,
        "reject_step2_source_too_short": 21,
        "reject_step2_source_support_ratio_low": 22,
        "reject_step2_source_support_cells_low": 23,
        "reject_step2_source_frontier_unknown_edge": 24,
        "reject_step2_source_door_frame_duplicate": 25,
        "step2_candidate_rejected": 255,
    }
    return int(values.get(str(reason), 254))


def _door_anchor_source_map(
    shape: tuple[int, int],
    *,
    strict_raw_wall: np.ndarray,
    projected_wall: np.ndarray,
    anchor_projected_wall: np.ndarray,
    step1_gap_fill: np.ndarray,
    filtered_line: np.ndarray,
) -> np.ndarray:
    source = np.zeros(shape, dtype=np.uint8)
    source[np.asarray(filtered_line, dtype=bool)] = DOOR_ANCHOR_FILTERED_LINE
    source[np.asarray(anchor_projected_wall, dtype=bool)] = DOOR_ANCHOR_PROJECTED_ANCHOR
    source[np.asarray(step1_gap_fill, dtype=bool)] = DOOR_ANCHOR_STEP1
    source[np.asarray(projected_wall, dtype=bool)] = DOOR_ANCHOR_PROJECTED
    source[np.asarray(strict_raw_wall, dtype=bool)] = DOOR_ANCHOR_STRICT_RAW
    return source


def _prefixed_projection_debug(debug: Mapping[str, object], prefix: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in debug.items():
        if str(key).startswith("voxel_wall_projection_"):
            suffix = str(key)[len("voxel_wall_projection_") :]
            out[f"{prefix}_{suffix}"] = value
    return out


def _room_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(labels) if int(v) > 0]))


def _extension_reason_counts(extensions: Sequence[LineExtensionHit]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for hit in extensions:
        if hit.reject_reason is not None:
            counts[str(hit.reject_reason)] += 1
    return dict(counts)


def _extension_and_candidate_reasons(extensions: Sequence[LineExtensionHit], rejected_candidates: Sequence[SeparatorCandidate]) -> dict[str, int]:
    counts: Counter[str] = Counter(_extension_reason_counts(extensions))
    for candidate in rejected_candidates:
        counts[str(candidate.reject_reason)] += 1
    return dict(counts)


def _candidate_reason_counts(candidates: Sequence[SeparatorCandidate]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts[str(candidate.reject_reason)] += 1
    return dict(counts)


def _height_profile_alias_layers(layers: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    alias = {
        "height_profile_vertical_free_xy": "voxel_vertical_free_xy",
        "height_profile_wall_xy": "voxel_wall_xy",
        "height_profile_unknown_xy": "voxel_unknown_xy",
        "height_profile_vertical_observed_xy": "voxel_vertical_observed_xy",
        "height_profile_wall_base_mask": "voxel_wall_base_map",
        "height_profile_line_supported_wall_mask": "voxel_line_supported_wall_map",
        "height_profile_filtered_wall_line_mask": "voxel_filtered_wall_line_mask",
        "height_profile_door_seed_mask": "voxel_door_seed_mask",
        "height_profile_door_cut_mask": "voxel_door_cut_mask",
        "height_profile_accepted_door_centerline_mask": "voxel_accepted_door_centerline_mask",
        "height_profile_rejected_door_centerline_mask": "voxel_rejected_door_centerline_mask",
        "height_profile_step1_wall_gap_fill_map": "voxel_step1_wall_gap_fill_map",
        "height_profile_step1_wall_mask": "voxel_wall_after_step1_map",
        "height_profile_step2_line_extensions_all": "voxel_step2_extension_candidate_map",
        "height_profile_step2_line_extensions_rejected": "voxel_step2_rejected_extension_map",
        "height_profile_step2_extension_separator_map": "voxel_step2_extension_separator_map",
        "height_profile_boundary_source_map": "voxel_boundary_source_map",
        "height_profile_partition_free": "voxel_partition_free",
        "height_profile_final_room_label_map": "voxel_final_room_label_map",
        "height_profile_room_label_map_visual": "voxel_room_label_map_visual",
        "height_profile_final_separator_map": "voxel_final_separator_map",
    }
    return {dst: np.asarray(layers[src]) for dst, src in alias.items() if src in layers}


def _height_profile_alias_debug(debug: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for dst, src in {
        "height_profile_room_count": "voxel_room_count",
        "height_profile_step2_reject_reason_counts": "voxel_step2_reject_reason_counts",
        "height_profile_active_z_max_m": "voxel_active_z_max_m",
        "height_profile_ceiling_height_m": "voxel_ceiling_height_m",
    }.items():
        if src in debug:
            out[dst] = debug[src]
    return out
