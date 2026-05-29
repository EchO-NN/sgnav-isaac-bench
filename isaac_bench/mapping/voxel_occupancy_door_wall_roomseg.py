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
from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_axis_accumulator_lines


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
    door_intersection_dilation_cells: int = 1
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
            "door_intersection_dilation_cells": "door_intersection_dilation_cells",
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
            door_memory=self.door_memory,
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
    door_memory: VoxelDoorMemory | None = None,
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
    door_seed_result = classify_voxel_door_seeds(voxel_grid=voxel_grid, config=cfg.door)
    door_seed_mask = np.asarray(door_seed_result.door_seed_mask, dtype=bool)

    wall_line_support_strong_xy = np.asarray(
        evidence.wall_line_support_strong_xy
        if evidence.wall_line_support_strong_xy is not None
        else evidence.wall_line_support_xy,
        dtype=bool,
    )
    wall_line_support_conflict_xy = np.asarray(
        evidence.wall_line_support_conflict_xy
        if evidence.wall_line_support_conflict_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    wall_line_support_rejected_unknown_xy = np.asarray(
        evidence.wall_line_support_rejected_unknown_xy
        if evidence.wall_line_support_rejected_unknown_xy is not None
        else evidence.wall_line_support_rejected_by_unknown_xy,
        dtype=bool,
    )
    wall_line_support_weight_xy = np.asarray(
        evidence.wall_line_support_weight_xy
        if evidence.wall_line_support_weight_xy is not None
        else wall_line_support_strong_xy.astype(np.float32),
        dtype=np.float32,
    )
    projection_seed = np.asarray(
        evidence.support_seed_for_projection_xy
        if evidence.support_seed_for_projection_xy is not None
        else (
            evidence.wall_support_strong_xy
            if evidence.wall_support_strong_xy is not None
            else wall_line_support_strong_xy
        ),
        dtype=bool,
    )
    projection_bridge = np.asarray(
        evidence.support_bridge_for_projection_xy
        if evidence.support_bridge_for_projection_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    projection_input = np.asarray(
        evidence.wall_support_for_projection_xy
        if evidence.wall_support_for_projection_xy is not None
        else (projection_seed | projection_bridge),
        dtype=bool,
    )
    projection_weight = np.asarray(
        evidence.wall_support_weight_xy
        if evidence.wall_support_weight_xy is not None
        else wall_line_support_weight_xy,
        dtype=np.float32,
    )
    frontier_unknown_band = np.asarray(
        evidence.frontier_unknown_band_xy
        if evidence.frontier_unknown_band_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    forbidden_frontier_residual = np.asarray(
        evidence.forbidden_frontier_residual_support_xy
        if evidence.forbidden_frontier_residual_support_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    forbidden_unknown_boundary = np.asarray(
        evidence.forbidden_unknown_boundary_support_xy
        if evidence.forbidden_unknown_boundary_support_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
    forbidden_residual_support = forbidden_frontier_residual | forbidden_unknown_boundary
    protected_structural_wall_band = np.asarray(
        evidence.protected_structural_wall_band_xy
        if evidence.protected_structural_wall_band_xy is not None
        else np.zeros(shape, dtype=bool),
        dtype=bool,
    )
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
        structural_side_support_map=np.asarray(evidence.wall_xy, dtype=bool)
        | np.asarray(evidence.structural_wall_ratio_xy, dtype=bool)
        | projection_seed,
        resolution_m=float(resolution_m),
        config=cfg.wall_projection,
    )
    anchor_wall_projection = wall_projection
    projected_wall_map = np.asarray(
        wall_projection.projected_wall_display_map
        if wall_projection.projected_wall_display_map is not None
        else wall_projection.projected_wall_map,
        dtype=bool,
    )
    anchor_projected_wall_map = np.asarray(
        wall_projection.projected_wall_anchor_map
        if wall_projection.projected_wall_anchor_map is not None
        else projected_wall_map,
        dtype=bool,
    )
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
    partition_maps = build_voxel_partition_maps(
        evidence=evidence,
        door_seed_mask=door_seed_mask,
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
    door_anchor_source_map = _door_anchor_source_map(
        shape,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall=projected_wall_map,
        anchor_projected_wall=np.zeros(shape, dtype=bool),
        step1_gap_fill=step1_gap_fill_map,
        filtered_line=np.zeros(shape, dtype=bool),
    )
    door_anchor_source_map = np.where(partition_maps.door_anchor_wall_map, door_anchor_source_map, 0).astype(np.uint8)
    door_anchor_source_map[(partition_maps.door_anchor_wall_map) & (door_anchor_source_map == 0)] = DOOR_ANCHOR_STRICT_RAW
    door_anchor_wall_map = np.asarray(partition_maps.door_anchor_wall_map, dtype=bool)
    door_cluster_barrier = np.asarray(partition_maps.partition_real_wall_map, dtype=bool) & ~dilate(door_seed_mask, 2)
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
        door_memory_result = door_memory.update(door_completion.candidates, step=int(step), shape=shape)
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
    step1_wall_mask = step1_completed_wall_map
    step2_line_pool = build_step2_line_pool(
        filtered_lines=filtered_lines,
        extension_seed_lines=extension_seed_lines,
        strict_raw_wall=np.asarray(evidence.wall_xy, dtype=bool),
        projected_wall_map=projected_wall_map,
        anchor_projected_wall_map=np.zeros(shape, dtype=bool),
        step1_completed_wall_map=step1_completed_wall_map,
        filtered_line_map=filtered_line_map_clean,
        extension_seed_line_map=validated_extension_seed_line_map,
        shape=shape,
        resolution_m=float(resolution_m),
        target_wall_override=partition_maps.step2_target_wall_map,
    )
    accepted_seed_cluster_mask = _accepted_door_seed_cluster_mask(
        np.asarray(door_completion.debug.get("voxel_door_seed_cluster_map", np.zeros(shape, dtype=np.int32)), dtype=np.int32),
        door_completion.debug.get("voxel_door_seed_clusters", []),
    )
    accepted_seed_for_partition = accepted_seed_cluster_mask & np.asarray(door_completion.debug.get("voxel_door_seed_mask", door_seed_mask), dtype=bool)
    step2_door_reject_mask = accepted_door_visual_mask | door_cut_mask | accepted_seed_cluster_mask

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
    step2_candidates, step2_candidate_debug = build_step2_separator_candidates_from_extensions(
        step2_extensions,
        accepted_virtual_targets=None,
        resolution_m=float(resolution_m),
        config=cfg.door_neck,
        start_id=1,
    )
    intersection_candidates, intersection_target_map, intersection_debug = build_door_neck_candidates_from_extension_intersections(
        step2_extensions,
        free_clean=free_after_step1,
        unknown_clean=unknown_after_step1,
        resolution_m=float(resolution_m),
        line_config=line_cfg,
        door_config=cfg.door_neck,
        start_id=int(len(step2_candidates) + 1),
    )
    for candidate in intersection_candidates:
        candidate.kind = "line_extension_corridor_separator"
        candidate.debug["kind_detail"] = "corridor_separator"
        candidate.debug["candidate_source"] = "step2_extension_intersection"
    intersection_candidates, intersection_pre_rejected = _reject_step2_candidates_intersecting_doors(
        intersection_candidates,
        door_block_mask=step2_door_reject_mask,
        shape=shape,
    )
    all_step2_candidates = [*step2_candidates, *intersection_candidates]
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
    rejected_step2 = [*intersection_pre_rejected, *rejected_step2]
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
    final_virtual_separator_map = door_cut_mask | step2_extension_separator_map
    partition_free_for_label = (base_partition_free | accepted_seed_for_partition) & ~final_virtual_separator_map
    partition_free = partition_free_for_label.copy()
    labels, _count = ndimage.label(partition_free, structure=conn(int(cfg.final_connectivity)))
    labels = relabel_compact(labels.astype(np.int32))
    labels[unknown_after_step1] = 0
    labels[~partition_free_for_label] = 0
    final_separator_map = real_wall_barrier_for_partition | door_cut_mask | step2_extension_separator_map

    boundary_source = np.zeros(shape, dtype=np.uint8)
    boundary_source[real_wall_barrier_for_partition] = 1
    boundary_source[step1_gap_fill_map] = 4
    boundary_source[door_cut_mask] = 2
    boundary_source[step2_extension_separator_map] = 3
    step2_layers = _extension_layers(step2_extensions, shape)
    step2_topology_rejected_map = _rasterize_candidates(rejected_step2, shape)
    step2_intersection_candidate_map = _rasterize_candidates(intersection_candidates, shape)
    step2_stage_maps = Step2StageMaps(
        extension_hits_all_map=step2_layers["all"].astype(bool),
        extension_hits_pre_topology_map=step2_layers["accepted"].astype(bool),
        separator_candidates_pre_topology_map=step2_candidate_map.astype(bool),
        topology_rejected_separator_map=step2_topology_rejected_map.astype(bool),
        accepted_separator_map=accepted_step2_map.astype(bool),
        accepted_partition_cut_map=step2_extension_separator_map.astype(bool),
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
        "voxel_projected_structural_wall_map": projected_wall_map,
        "voxel_anchor_projected_wall_map": anchor_projected_wall_map,
        "voxel_unknown_xy": evidence.unknown_xy,
        "voxel_unknown_dominant_xy": np.asarray(evidence.unknown_dominant_xy, dtype=bool),
        "voxel_structural_wall_seed_xy": np.asarray(evidence.structural_wall_seed_xy, dtype=bool),
        "voxel_structural_wall_ratio_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_wall_ratio_raw_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
        "voxel_wall_line_support_xy": np.asarray(evidence.wall_line_support_xy, dtype=bool),
        "voxel_wall_line_support_raw_xy": np.asarray(evidence.wall_line_support_raw_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_free_xy": np.asarray(evidence.wall_line_support_rejected_by_free_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_unknown_xy": np.asarray(evidence.wall_line_support_rejected_by_unknown_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_observed_xy": np.asarray(evidence.wall_line_support_rejected_by_observed_xy, dtype=bool),
        "voxel_wall_line_support_rejected_by_nav_edge_xy": np.asarray(evidence.wall_line_support_rejected_by_nav_edge_xy, dtype=bool),
        "voxel_wall_projection_support_input_xy": projection_input,
        "voxel_wall_support_raw_occupied_xy": np.asarray(evidence.wall_support_raw_occupied_xy if evidence.wall_support_raw_occupied_xy is not None else evidence.wall_line_support_raw_xy, dtype=bool),
        "voxel_wall_support_known_xy": np.asarray(evidence.wall_support_known_xy if evidence.wall_support_known_xy is not None else wall_line_support_strong_xy, dtype=bool),
        "voxel_wall_support_for_projection_xy": np.asarray(evidence.wall_support_for_projection_xy if evidence.wall_support_for_projection_xy is not None else projection_input, dtype=bool),
        "voxel_wall_support_weight_xy": projection_weight,
        "voxel_wall_support_unknown_rejected_xy": np.asarray(evidence.wall_support_unknown_rejected_xy if evidence.wall_support_unknown_rejected_xy is not None else wall_line_support_rejected_unknown_xy, dtype=bool),
        "voxel_wall_support_nav_unknown_rejected_xy": np.asarray(evidence.wall_support_nav_unknown_rejected_xy if evidence.wall_support_nav_unknown_rejected_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_wall_support_frontier_band_rejected_xy": np.asarray(evidence.wall_support_frontier_band_rejected_xy if evidence.wall_support_frontier_band_rejected_xy is not None else forbidden_frontier_residual, dtype=bool),
        "voxel_wall_support_free_conflict_xy": np.asarray(evidence.wall_support_free_conflict_xy if evidence.wall_support_free_conflict_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_frontier_unknown_band_xy": frontier_unknown_band,
        "voxel_strong_structural_support_xy": np.asarray(evidence.strong_structural_support_xy if evidence.strong_structural_support_xy is not None else projection_seed, dtype=bool),
        "voxel_bridge_only_support_xy": np.asarray(evidence.bridge_only_support_xy if evidence.bridge_only_support_xy is not None else projection_bridge, dtype=bool),
        "voxel_forbidden_frontier_residual_support_xy": forbidden_frontier_residual,
        "voxel_forbidden_unknown_boundary_support_xy": forbidden_unknown_boundary,
        "voxel_free_conflict_support_xy": np.asarray(evidence.free_conflict_support_xy if evidence.free_conflict_support_xy is not None else np.zeros(shape, dtype=bool), dtype=bool),
        "voxel_protected_structural_wall_band_xy": protected_structural_wall_band,
        "voxel_support_seed_for_projection_xy": projection_seed,
        "voxel_support_bridge_for_projection_xy": projection_bridge,
        "voxel_support_for_projection_display_xy": projection_input,
        "voxel_projected_wall_display_map": np.asarray(wall_projection.projected_wall_display_map if wall_projection.projected_wall_display_map is not None else projected_wall_map, dtype=bool),
        "voxel_projected_wall_anchor_map": anchor_projected_wall_map,
        "voxel_projected_wall_step2_source_map": np.asarray(wall_projection.projected_wall_step2_source_map if wall_projection.projected_wall_step2_source_map is not None else projected_wall_map, dtype=bool),
        "voxel_wall_projection_accumulator_h_votes": np.asarray(wall_projection.debug.get("voxel_wall_projection_accumulator_h_votes", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_projection_accumulator_v_votes": np.asarray(wall_projection.debug.get("voxel_wall_projection_accumulator_v_votes", np.zeros(shape, dtype=np.float32)), dtype=np.float32),
        "voxel_wall_projection_reject_reason_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_wall_projection_step2_source_reject_reason_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_step2_source_reject_reason_map", np.zeros(shape, dtype=np.uint8)), dtype=np.uint8),
        "voxel_wall_rejected_by_free_xy": np.asarray(evidence.wall_rejected_by_free_xy, dtype=bool),
        "voxel_wall_rejected_by_unknown_xy": np.asarray(evidence.wall_rejected_by_unknown_xy, dtype=bool),
        "voxel_nonstructural_occupied_xy": np.asarray(evidence.nonstructural_occupied_xy, dtype=bool),
        "voxel_small_unknown_hole_filled_map": np.asarray(evidence.small_unknown_hole_filled_xy, dtype=bool),
        "voxel_wall_support_unknown_gated_xy": np.asarray(evidence.wall_support_unknown_gated_xy, dtype=bool),
        "voxel_wall_support_rejected_unknown_xy": np.asarray(evidence.wall_support_rejected_unknown_xy, dtype=bool),
        "voxel_free_raw_xy": np.asarray(evidence.debug.get("voxel_free_raw_xy", evidence.vertical_free_xy), dtype=bool),
        "voxel_wall_raw_xy": np.asarray(evidence.structural_wall_ratio_xy, dtype=bool),
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
        "voxel_partition_real_wall_map": partition_maps.partition_real_wall_map,
        "voxel_partition_real_wall_removed_by_seed_carve_map": partition_maps.removed_by_seed_carve_map,
        "voxel_partition_unknown": partition_maps.partition_unknown,
        "voxel_wall_projection_support_map": np.asarray(wall_projection.support_map, dtype=bool),
        "voxel_wall_projection_rejected_support_map": np.asarray(wall_projection.rejected_support_map, dtype=bool),
        "voxel_wall_projection_forbidden_unknown_map": np.asarray(wall_projection.debug.get("voxel_wall_projection_forbidden_unknown_map", np.zeros(shape, dtype=bool)), dtype=bool),
        "voxel_free_wall_conflict_xy": np.asarray(evidence.free_wall_conflict_xy, dtype=bool),
        "voxel_vertical_observed_xy": evidence.active_observed_xy,
        "voxel_active_observed_xy": evidence.active_observed_xy,
        "voxel_line_supported_wall_map": raw_line_map,
        "voxel_filtered_wall_line_mask": filtered_line_map,
        "voxel_extension_seed_wall_line_mask": extension_seed_line_map,
        "voxel_wall_base_map": wall_base_pre_step1,
        "voxel_door_seed_mask": door_seed_mask,
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
        "voxel_step2_target_wall_map": step2_line_pool.target_wall_map,
        "voxel_step2_target_source_map": step2_line_pool.target_source_map,
        "voxel_step2_door_reject_mask": step2_door_reject_mask,
        "voxel_step2_extension_candidate_map": step2_layers["all"],
        "voxel_step2_extension_hits_all_map": step2_stage_maps.extension_hits_all_map,
        "voxel_step2_extension_hits_pre_topology_map": step2_stage_maps.extension_hits_pre_topology_map,
        "voxel_step2_separator_candidates_pre_topology_map": step2_stage_maps.separator_candidates_pre_topology_map,
        "voxel_step2_partition_cut_candidate_map": step2_partition_cut_candidate_map,
        "voxel_step2_partition_cut_candidate_from_accepted_map": step2_partition_cut_candidate_from_accepted_map,
        "voxel_step2_partition_cut_accepted_map": step2_extension_separator_map,
        "voxel_step2_topology_rejected_separator_map": step2_stage_maps.topology_rejected_separator_map,
        "voxel_step2_accepted_separator_map": step2_stage_maps.accepted_separator_map,
        "voxel_step2_extension_separator_map": step2_extension_separator_map,
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
        "voxel_step2_filtered_line_count": int(step2_line_pool.debug.get("voxel_step2_filtered_line_count", 0)),
        "voxel_step2_extension_seed_line_count": int(step2_line_pool.debug.get("voxel_step2_extension_seed_line_count", 0)),
        "voxel_step2_source_line_dedup_count": int(step2_line_pool.debug.get("voxel_step2_source_line_dedup_count", 0)),
        "voxel_step2_target_wall_cells": int(step2_line_pool.debug.get("voxel_step2_target_wall_cells", 0)),
        "voxel_step2_target_source_counts": dict(step2_line_pool.debug.get("voxel_step2_target_source_counts", {}) or {}),
        "voxel_wall_raw_cells": int(np.count_nonzero(evidence.structural_wall_ratio_xy)),
        "voxel_wall_ratio_raw_cells": int(np.count_nonzero(evidence.structural_wall_ratio_xy)),
        "voxel_raw_occupied_wall_support_cells": int(np.count_nonzero(evidence.raw_occupied_wall_support_xy)),
        "voxel_wall_line_support_cells": int(np.count_nonzero(evidence.wall_line_support_xy)),
        "voxel_wall_line_support_raw_cells": int(np.count_nonzero(evidence.wall_line_support_raw_xy)),
        "voxel_wall_line_support_strong_cells": int(np.count_nonzero(wall_line_support_strong_xy)),
        "voxel_wall_line_support_conflict_cells": int(np.count_nonzero(wall_line_support_conflict_xy)),
        "voxel_frontier_unknown_band_cells": int(np.count_nonzero(frontier_unknown_band)),
        "voxel_forbidden_frontier_residual_support_cells": int(np.count_nonzero(forbidden_frontier_residual)),
        "voxel_forbidden_unknown_boundary_support_cells": int(np.count_nonzero(forbidden_unknown_boundary)),
        "voxel_support_seed_for_projection_cells": int(np.count_nonzero(projection_seed)),
        "voxel_support_bridge_for_projection_cells": int(np.count_nonzero(projection_bridge)),
        "voxel_strict_raw_wall_cells": int(np.count_nonzero(evidence.strict_raw_wall_xy)),
        "voxel_wall_projected_cells": int(np.count_nonzero(projected_wall_map)),
        "voxel_projected_structural_wall_cells": int(np.count_nonzero(projected_wall_map)),
        "voxel_display_wall_cells": int(np.count_nonzero(display_wall_map)),
        "voxel_anchor_projected_wall_cells": int(np.count_nonzero(anchor_projected_wall_map)),
        "voxel_door_anchor_wall_cells": int(np.count_nonzero(door_anchor_wall_map)),
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
        "voxel_step2_intersection_candidate_count": int(len(intersection_candidates)),
        "voxel_step2_intersection_pre_rejected_count": int(len(intersection_pre_rejected)),
        "voxel_step2_accepted_count": int(len(accepted_step2)),
        "voxel_step2_rejected_count": int(len(rejected_step2)),
        "voxel_step2_extension_reject_reason_counts": _extension_reason_counts(step2_extensions),
        "voxel_step2_topology_reject_reason_counts": _candidate_reason_counts(rejected_step2),
        "voxel_step2_partition_cut_empty_count": int(step2_partition_cut_empty_count),
        "voxel_step2_partition_cut_candidate_cells": int(np.count_nonzero(step2_partition_cut_candidate_map)),
        "voxel_step2_partition_cut_accepted_cells": int(np.count_nonzero(step2_extension_separator_map)),
        "voxel_step2_partition_cut_debug": dict(step2_partition_cut_debug),
        "voxel_step2_reject_reason_counts": _extension_and_candidate_reasons(step2_extensions, rejected_step2),
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
        "variant": "voxel_v25_1_wall_lines_on_v19",
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
        "voxel_step2_reject_reason_counts": report["voxel_step2_reject_reason_counts"],
        "voxel_step2_extension_reject_reason_counts": report["voxel_step2_extension_reject_reason_counts"],
        "voxel_step2_topology_reject_reason_counts": report["voxel_step2_topology_reject_reason_counts"],
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
        "voxel_step1_gap_fill_report": step1_gap_debug,
        "voxel_wall_run_report": wall_run_debug,
        "topology_report": {"final": topology_debug, "step2_candidates": step2_candidate_debug},
        "voxel_step2_intersection_debug": intersection_debug,
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
        step2_extension_separator_map=step2_extension_separator_map.astype(bool),
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
    door_seed_mask: np.ndarray,
    strict_raw_wall: np.ndarray,
    projected_wall_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    filtered_line_map: np.ndarray,
    extension_seed_line_map: np.ndarray,
    step1_gap_fill_map: np.ndarray,
    cfg: VoxelOccupancyDoorWallRoomSegConfig,
) -> PartitionMapBundle:
    seed = np.asarray(door_seed_mask, dtype=bool)
    strict = np.asarray(strict_raw_wall, dtype=bool)
    projected = np.asarray(projected_wall_map, dtype=bool)
    anchor_projected = np.asarray(anchor_projected_wall_map, dtype=bool)
    filtered = np.asarray(filtered_line_map, dtype=bool)
    extension_seed = np.asarray(extension_seed_line_map, dtype=bool)
    step1 = np.asarray(step1_gap_fill_map, dtype=bool)
    raw_support = np.asarray(evidence.raw_occupied_wall_support_xy, dtype=bool)
    unknown_dominant = np.asarray(getattr(evidence, "unknown_dominant_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    unknown_rejected = np.asarray(getattr(evidence, "wall_rejected_by_unknown_xy", np.zeros_like(strict, dtype=bool)), dtype=bool)
    vertical_free = np.asarray(evidence.vertical_free_xy, dtype=bool)
    shape = vertical_free.shape
    for name, arr in {
        "strict_raw_wall": strict,
        "projected_wall_map": projected,
        "anchor_projected_wall_map": anchor_projected,
        "filtered_line_map": filtered,
        "extension_seed_line_map": extension_seed,
        "step1_gap_fill_map": step1,
        "door_seed_mask": seed,
    }.items():
        if arr.shape != shape:
            raise ValueError("%s must match voxel evidence shape" % name)
    seed_carve_radius = int(getattr(cfg, "door_seed_wall_carve_radius_cells", cfg.door.seed_cluster_morph_close_radius_cells))
    seed_carve = dilate(seed, max(0, seed_carve_radius))
    clean_structural_wall = strict & ~vertical_free
    projected_structural_wall = projected & ~vertical_free
    step1_structural_wall = step1 & ~vertical_free
    validated_extension_seed = extension_seed & ~vertical_free
    wall_anchor_support_map = clean_structural_wall | projected_structural_wall | step1_structural_wall
    door_anchor_wall_map = wall_anchor_support_map.copy()
    step2_target_wall_map = wall_anchor_support_map | validated_extension_seed
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
        "voxel_door_anchor_allowed_sources": ["structural_wall", "projected_structural_wall", "step1"],
        "voxel_step2_target_allowed_sources": ["structural_wall", "projected_structural_wall", "step1", "validated_extension_seed"],
        "voxel_door_anchor_unknown_overlap_cells": int(np.count_nonzero(door_anchor_wall_map & partition_unknown)),
        "voxel_step2_target_wall_cells_v18": int(np.count_nonzero(step2_target_wall_map)),
        "voxel_step2_target_wall_cells_v15": int(np.count_nonzero(step2_target_wall_map)),
        "voxel_step2_target_wall_unknown_rejected_cells": int(np.count_nonzero((strict | projected | anchor_projected | filtered | extension_seed | step1) & unknown_dominant)),
        "voxel_step2_false_frontier_wall_rejected_map": unknown_rejected.astype(bool),
        "voxel_partition_extension_seed_excluded_cells": int(np.count_nonzero(extension_seed & ~partition_real_wall_map)),
        "voxel_partition_filtered_line_excluded_cells": int(np.count_nonzero(filtered & ~partition_real_wall_map)),
        "voxel_partition_anchor_projection_excluded_cells": int(np.count_nonzero(anchor_projected & ~partition_real_wall_map)),
        "voxel_partition_real_wall_cells": int(np.count_nonzero(partition_real_wall_map)),
        "voxel_partition_real_wall_removed_by_seed_carve_cells": int(np.count_nonzero(removed_by_seed_carve)),
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
    strict_raw_wall: np.ndarray,
    projected_wall_map: np.ndarray,
    anchor_projected_wall_map: np.ndarray,
    step1_completed_wall_map: np.ndarray,
    filtered_line_map: np.ndarray,
    extension_seed_line_map: np.ndarray,
    shape: tuple[int, int],
    resolution_m: float,
    target_wall_override: np.ndarray | None = None,
) -> Step2LinePool:
    source_lines, dedup_count = _dedup_step2_lines([*list(filtered_lines), *list(extension_seed_lines)])
    source_line_map = filtered_wall_line_mask(source_lines, shape)
    target_source = np.zeros(shape, dtype=np.uint8)
    target_source[np.asarray(strict_raw_wall, dtype=bool)] = 1
    target_source[np.asarray(projected_wall_map, dtype=bool)] = 2
    target_source[np.asarray(anchor_projected_wall_map, dtype=bool)] = 3
    target_source[np.asarray(step1_completed_wall_map, dtype=bool)] = 4
    target_source[np.asarray(filtered_line_map, dtype=bool)] = 5
    target_source[np.asarray(extension_seed_line_map, dtype=bool)] = 6
    target_wall = target_source > 0
    if target_wall_override is not None:
        target_wall = np.asarray(target_wall_override, dtype=bool)
        clean_source = np.zeros(shape, dtype=np.uint8)
        clean_source[np.asarray(strict_raw_wall, dtype=bool) & target_wall] = 1
        clean_source[np.asarray(projected_wall_map, dtype=bool) & target_wall] = 2
        clean_source[np.asarray(step1_completed_wall_map, dtype=bool) & target_wall] = 4
        clean_source[np.asarray(extension_seed_line_map, dtype=bool) & target_wall & (clean_source == 0)] = 6
        target_source = clean_source
    source_counts = {
        "strict_raw_wall": int(np.count_nonzero(target_source == 1)),
        "projected_wall": int(np.count_nonzero(target_source == 2)),
        "anchor_projected_wall": int(np.count_nonzero(target_source == 3)),
        "step1_completed_wall": int(np.count_nonzero(target_source == 4)),
        "filtered_wall_line": int(np.count_nonzero(target_source == 5)),
        "extension_seed_line": int(np.count_nonzero(target_source == 6)),
    }
    debug = {
        "voxel_step2_source_line_count": int(len(source_lines)),
        "voxel_step2_filtered_line_count": int(len(filtered_lines)),
        "voxel_step2_extension_seed_line_count": int(len(extension_seed_lines)),
        "voxel_step2_source_line_dedup_count": int(dedup_count),
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
