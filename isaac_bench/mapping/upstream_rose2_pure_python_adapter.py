from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, world_xy_to_grid
from isaac_bench.mapping.room_segmentation import (
    RoomMask,
    RoomProposalState,
    RoomSegmentationConfig,
    merge_open_plan_proposals,
)
from isaac_bench.mapping.roomseg_evidence_fusion import (
    EVIDENCE_FUSION_MODE,
    fuse_vertical_profile_with_navigation_obstacles,
)
from isaac_bench.mapping.roomseg_ray_valid_wall import (
    RAY_VALID_WALL_INFERENCE_MODE,
    build_ray_valid_wall_inference,
)
from isaac_bench.mapping.room_context_overlay import build_navigation_free_room_context_overlay
from isaac_bench.mapping.structure_extraction import (
    StructureExtractionConfig,
    StructureExtractionResult,
    bresenham_line,
    connected_components,
    face_adjacency_edges,
    faces_from_boundary_map,
    extract_rose2_structure,
)
from isaac_bench.mapping.rose2_source_form import (
    LEGACY_STYLE_BACKEND,
    ROSE2SourceFormConfig,
    ROSE2SourceResult,
    SOURCE_EXTERNAL_BACKEND,
    SOURCE_EXTERNAL_RUNNER_BACKEND,
    SOURCE_FAITHFUL_BACKEND,
    SOURCE_FORM_BACKEND,
    run_rose2_source_external,
    run_rose2_source_faithful_v1,
    run_rose2_source_form,
    run_rose2_source_form_v2,
    save_rose2_source_debug,
    source_result_summary,
)
from isaac_bench.mapping.rose2_source_external_runner import run_rose2_source_external_runner
from isaac_bench.mapping.vertical_profile import VerticalProfileMap, band_index, ensure_vertical_profile
from isaac_bench.mapping.vertical_free_roomseg import (
    VERTICAL_FREE_ROOMSEG_ALGORITHM,
    VERTICAL_FREE_ROOMSEG_BACKEND,
    VERTICAL_FREE_ROOMSEG_CONTEXT,
    VerticalFreeRoomSegConfig,
    run_vertical_free_roomseg,
    save_vertical_free_roomseg_debug,
    vertical_free_result_to_source_result,
)
from isaac_bench.mapping.vertical_free_gap_closure_roomseg import (
    VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
    VERTICAL_FREE_GAP_CLOSURE_BACKEND,
    VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
    VFGCConfig,
    run_vertical_free_gap_closure_roomseg,
    save_vertical_free_gap_closure_debug,
    vfgc_result_to_source_result,
)


UPSTREAM_SOURCE_MODE = "declutter_reconstruct_external"
UPSTREAM_ALGORITHM = "upstream_rose2_vertical_or_free"
UPSTREAM_CONTEXT_SOURCE = "%s_vlm" % UPSTREAM_ALGORITHM
UPSTREAM_ALGORITHM_ALIASES = {
    SOURCE_FAITHFUL_BACKEND,
    "%s_vlm" % SOURCE_FAITHFUL_BACKEND,
    SOURCE_FORM_BACKEND,
    "%s_vlm" % SOURCE_FORM_BACKEND,
    UPSTREAM_CONTEXT_SOURCE,
    SOURCE_EXTERNAL_RUNNER_BACKEND,
    "upstream_rose2_vertical_or_free",
    "upstream_rose2_vertical_or_free_vlm",
    "upstream_rose2_pure_python",
    "upstream_rose2_pure_python_vlm",
    VERTICAL_FREE_ROOMSEG_ALGORITHM,
    VERTICAL_FREE_ROOMSEG_CONTEXT,
    VERTICAL_FREE_ROOMSEG_BACKEND,
    VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
    VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
    VERTICAL_FREE_GAP_CLOSURE_BACKEND,
}
DEFAULT_SOURCE_ENV = "ROSE2_SOURCE_ROOT"
REQUIRED_SOURCE_FILES = (
    "code/FFT_MQ.py",
    "code/minibatch.py",
    "code/parameters.py",
    "code/util/layout.py",
    "code/util/postprocessing.py",
)


@dataclass
class UpstreamROSE2Config:
    source_root: str | None
    source_mode: str = UPSTREAM_SOURCE_MODE
    backend: str = SOURCE_EXTERNAL_RUNNER_BACKEND
    filter_value: float = 0.18
    spatial_clustering_line_segments_threshold: float = 5.0
    lines_th1: float = 0.1
    lines_distance: float = 20.0
    edges_th: float = 0.1
    rooms_voronoi: bool = False
    fail_on_missing_source: bool = True
    debug_dump: bool = True
    finalization_mode: str = "no_merge_until_source_backend_verified"
    upstream_repo_env: str = DEFAULT_SOURCE_ENV
    resolution_m: float = 0.05
    min_room_area_m2: float = 1.5
    hough_min_line_length_m: float = 1.0
    hough_line_gap_m: float = 0.25
    wall_cluster_distance_m: float = 0.35
    wall_cluster_gap_m: float = 0.50
    wall_min_support_ratio: float = 0.25
    wall_extension_enabled: bool = True
    wall_extension_band_m: float = 0.45
    wall_extension_margin_m: float = 0.15
    clutter_component_max_area_m2: float = 4.0
    debug_dir: str = "debug/roomseg"
    wall_confidence_threshold: float = 0.55
    vertical_free_suppression_weight: float = 0.35
    object_overlap_suppression_weight: float = 0.45
    perimeter_support_weight: float = 0.15
    line_support_weight: float = 0.25
    continuity_weight: float = 0.45
    persistence_weight: float = 0.10
    single_view_penalty: float = 0.20
    surrounded_by_free_penalty: float = 0.25
    bulky_component_penalty: float = 0.25
    furniture_suppression_radius_m: float = 0.90
    exterior_margin_cells: int = 2
    wall_like_aspect_ratio_min: float = 4.0
    vertical_or_free_enabled: bool = True
    vertical_or_free_z_min_m: float = 0.10
    vertical_or_free_z_max_m: float = 2.50
    vertical_or_free_min_free_rays: int = 1
    vertical_or_free_min_observed_rays: int = 1
    roomseg_evidence_fusion_enabled: bool = True
    roomseg_evidence_fusion_mode: str = EVIDENCE_FUSION_MODE
    roomseg_evidence_fusion_vertical_free_priority: bool = True
    roomseg_use_navigation_obstacle_for_vertical_unknown: bool = False
    roomseg_use_navigation_free_for_roomseg_free: bool = False
    roomseg_use_inflated_obstacle: bool = False
    roomseg_use_depth_valid_as_wall: bool = False
    roomseg_use_static_structural_occupied: bool = False
    ray_valid_wall_inference_enabled: bool = True
    ray_valid_wall_inference_mode: str = RAY_VALID_WALL_INFERENCE_MODE
    ray_valid_wall_depth_max_m: float = 3.0
    ray_valid_wall_min_endpoint_height_m: float = 0.10
    ray_valid_wall_max_endpoint_height_m: float = 2.50
    ray_valid_wall_min_terminal_wall_count: int = 1
    ray_valid_wall_terminal_wall_splat_radius_cells: int = 1
    ray_valid_wall_require_no_vertical_free: bool = True
    ray_valid_wall_mark_ray_covered_debug: bool = True
    ray_valid_wall_decouple_roomseg_rays_from_nav_clear: bool = True
    ray_valid_wall_strict_no_navigation_obstacle_overlay: bool = True
    ray_valid_wall_strict_no_navigation_free_overlay: bool = True
    navigation_free_context_overlay_enabled: bool = False
    navigation_free_context_overlay_use_for_room_nodes: bool = True
    navigation_free_context_overlay_use_for_visualization: bool = True
    navigation_free_context_overlay_use_for_frontier_room_assignment: bool = True
    navigation_free_context_overlay_max_absorb_distance_m: float = 1.25
    navigation_free_context_overlay_min_seed_room_area_cells: int = 20
    navigation_free_context_overlay_protect_unknown: bool = True
    navigation_free_context_overlay_do_not_cross_structural_boundary: bool = True
    navigation_free_context_overlay_do_not_cross_obstacle: bool = True
    navigation_free_context_overlay_reliability_for_absorbed_cells: float = 0.35
    wall_gating_fix_enabled: bool = False
    wall_gating_fix_mode: str = "candidate_wall_component_gate"
    wall_gating_fix_keep_perimeter_walls: bool = True
    wall_gating_fix_keep_high_confidence_walls: bool = True
    wall_gating_fix_keep_line_supported_walls: bool = True
    wall_gating_fix_min_component_area_cells: int = 4
    debug_rose2_source: bool = False
    rose2_source_work_dir: str = "debug/rose2_source"
    rose2_compare_legacy: bool = False
    strict_disallow_legacy_fallback: bool = True
    allow_source_form_in_metric: bool = False
    allow_silent_fallback: bool = False
    external_runner_timeout_s: float = 60.0
    external_runner_python_executable: str | None = None
    external_runner_encoding: str = "auto"
    external_runner_keep_work_dir: bool = True
    external_runner_cache_enabled: bool = True
    external_runner_parameter_overrides: Mapping[str, object] | None = None
    source_form_min_cell_area_m2: float = 0.35
    source_form_min_cell_free_ratio: float = 0.12
    source_form_min_cut_spacing_m: float = 0.45
    source_form_axis_snap_angle_rad: float = 0.488692191
    source_form_max_cuts_per_axis: int = 96
    source_form_thin_wall_separator_enabled: bool = True
    source_form_thin_wall_min_length_m: float = 0.45
    source_form_thin_wall_max_width_m: float = 0.25
    source_form_thin_wall_min_aspect_ratio: float = 3.0
    source_form_thin_wall_free_support_band_m: float = 0.30
    source_form_thin_wall_min_free_support_ratio: float = 0.25
    source_form_topology_effective_separator_enabled: bool = True
    source_form_topology_effective_min_largest_component_drop: float = 0.08
    source_form_topology_effective_max_candidates: int = 128
    source_form_doorway_partition_enabled: bool = True
    source_form_doorway_width_min_m: float = 0.45
    source_form_doorway_width_max_m: float = 1.60
    source_form_doorway_min_wall_support_on_sides_m: float = 0.35
    source_form_merge_guard_enabled: bool = True
    vertical_free_roomseg: Mapping[str, object] | None = None
    vertical_free_gap_closure: Mapping[str, object] | None = None

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, object]] = None, **overrides) -> "UpstreamROSE2Config":
        raw = dict(data or {})
        debug_layers = dict(raw.get("debug_layers", {}) or {})
        if "enabled" in debug_layers:
            raw["debug_dump"] = bool(debug_layers.get("enabled"))
        if "output_dir" in debug_layers:
            raw["debug_dir"] = str(debug_layers.get("output_dir"))
        rose2 = dict(raw.get("rose2", {}) or {})
        for key, value in rose2.items():
            raw.setdefault(key, value)
        vertical_or_free = dict(raw.get("vertical_or_free", {}) or {})
        for key, value in vertical_or_free.items():
            raw.setdefault("vertical_or_free_%s" % key, value)
        evidence_fusion = dict(raw.get("evidence_fusion", {}) or {})
        if evidence_fusion:
            raw.setdefault("roomseg_evidence_fusion", evidence_fusion)
        evidence_key_map = {
            "enabled": "roomseg_evidence_fusion_enabled",
            "mode": "roomseg_evidence_fusion_mode",
            "vertical_free_priority": "roomseg_evidence_fusion_vertical_free_priority",
            "use_navigation_obstacle_for_vertical_unknown": "roomseg_use_navigation_obstacle_for_vertical_unknown",
            "use_navigation_free_for_roomseg_free": "roomseg_use_navigation_free_for_roomseg_free",
            "use_inflated_obstacle": "roomseg_use_inflated_obstacle",
            "use_depth_valid_as_wall": "roomseg_use_depth_valid_as_wall",
            "use_static_structural_occupied": "roomseg_use_static_structural_occupied",
        }
        for key, value in evidence_fusion.items():
            if key in evidence_key_map:
                raw.setdefault(evidence_key_map[key], value)
        ray_valid_wall = dict(raw.get("ray_valid_wall_inference", {}) or {})
        ray_key_map = {
            "enabled": "ray_valid_wall_inference_enabled",
            "mode": "ray_valid_wall_inference_mode",
            "depth_max_m": "ray_valid_wall_depth_max_m",
            "min_endpoint_height_m": "ray_valid_wall_min_endpoint_height_m",
            "max_endpoint_height_m": "ray_valid_wall_max_endpoint_height_m",
            "min_terminal_wall_count": "ray_valid_wall_min_terminal_wall_count",
            "terminal_wall_splat_radius_cells": "ray_valid_wall_terminal_wall_splat_radius_cells",
            "require_no_vertical_free": "ray_valid_wall_require_no_vertical_free",
            "mark_ray_covered_debug": "ray_valid_wall_mark_ray_covered_debug",
            "decouple_roomseg_rays_from_nav_clear": "ray_valid_wall_decouple_roomseg_rays_from_nav_clear",
            "strict_no_navigation_obstacle_overlay": "ray_valid_wall_strict_no_navigation_obstacle_overlay",
            "strict_no_navigation_free_overlay": "ray_valid_wall_strict_no_navigation_free_overlay",
        }
        for key, value in ray_valid_wall.items():
            if key in ray_key_map:
                raw.setdefault(ray_key_map[key], value)
        for section in ("navigation_free_context_overlay", "wall_gating_fix"):
            nested = dict(raw.get(section, {}) or {})
            for key, value in nested.items():
                raw.setdefault("%s_%s" % (section, key), value)
        source_form = dict(raw.get("source_form", {}) or {})
        for key, value in source_form.items():
            raw.setdefault("source_form_%s" % key, value)
        external_runner = dict(raw.get("external_runner", {}) or {})
        for key, value in external_runner.items():
            if key == "parameter_overrides":
                raw.setdefault("external_runner_parameter_overrides", value)
            elif key == "work_dir":
                raw.setdefault("rose2_source_work_dir", value)
            else:
                raw.setdefault("external_runner_%s" % key, value)
        if "source_root" not in raw:
            env_name = str(raw.get("upstream_repo_env", DEFAULT_SOURCE_ENV) or DEFAULT_SOURCE_ENV)
            raw["source_root"] = os.environ.get(env_name)
        raw.update({key: value for key, value in overrides.items() if value is not None})
        fields = {name for name in cls.__dataclass_fields__}
        return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class UpstreamROSE2Result:
    room_masks: list[RoomMask]
    clean_structure_map: np.ndarray
    main_directions: list[float]
    wall_lines: list[dict]
    extended_lines: list[dict]
    edges: list[dict]
    room_polygons: list[np.ndarray]
    room_label_map: np.ndarray
    debug: dict


class UpstreamROSE2PurePythonSegmenter:
    source = UPSTREAM_ALGORITHM
    context_source = UPSTREAM_CONTEXT_SOURCE

    def __init__(self, config: UpstreamROSE2Config, map_info: MapInfo):
        self.config = config
        self.map_info = map_info
        backend = str(config.backend or SOURCE_EXTERNAL_RUNNER_BACKEND).strip().lower()
        if backend in {VERTICAL_FREE_GAP_CLOSURE_BACKEND, VERTICAL_FREE_GAP_CLOSURE_ALGORITHM}:
            self.source = VERTICAL_FREE_GAP_CLOSURE_ALGORITHM
            self.context_source = VERTICAL_FREE_GAP_CLOSURE_CONTEXT
        elif backend in {VERTICAL_FREE_ROOMSEG_BACKEND, VERTICAL_FREE_ROOMSEG_ALGORITHM}:
            self.source = VERTICAL_FREE_ROOMSEG_ALGORITHM
            self.context_source = VERTICAL_FREE_ROOMSEG_CONTEXT
        else:
            self.source = UPSTREAM_ALGORITHM
            self.context_source = UPSTREAM_CONTEXT_SOURCE
        if backend in {
            VERTICAL_FREE_ROOMSEG_BACKEND,
            VERTICAL_FREE_ROOMSEG_ALGORITHM,
            VERTICAL_FREE_GAP_CLOSURE_BACKEND,
            VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        }:
            self.source_root = None
        else:
            self.source_root = validate_upstream_rose2_source_root(
                config.source_root,
                env_name=config.upstream_repo_env,
                fail=bool(config.fail_on_missing_source and backend in {SOURCE_EXTERNAL_BACKEND, SOURCE_EXTERNAL_RUNNER_BACKEND}),
            )
        self._room_config = RoomSegmentationConfig(
            algorithm=self.source,
            source_grid="online_depth_observed",
            finalization_mode=str(config.finalization_mode),
            resolution_m=float(config.resolution_m),
            min_room_area_m2=float(config.min_room_area_m2),
            min_wall_line_length_m=float(config.hough_min_line_length_m),
            max_clutter_component_area_m2=float(config.clutter_component_max_area_m2),
            debug_dump=bool(config.debug_dump),
            debug_dir=str(config.debug_dir),
            map_info=map_info,
        )
        self._structure_config = StructureExtractionConfig(
            resolution_m=float(config.resolution_m),
            min_room_area_m2=float(config.min_room_area_m2),
            hough_min_line_length_m=float(config.hough_min_line_length_m),
            hough_line_gap_m=float(config.hough_line_gap_m),
            wall_cluster_distance_m=float(config.wall_cluster_distance_m),
            wall_cluster_gap_m=float(config.wall_cluster_gap_m),
            wall_min_support_ratio=float(config.wall_min_support_ratio),
            wall_extension_enabled=bool(config.wall_extension_enabled),
            wall_extension_band_m=float(config.wall_extension_band_m),
            wall_extension_margin_m=float(config.wall_extension_margin_m),
            clutter_component_max_area_m2=float(config.clutter_component_max_area_m2),
        )
        self._source_form_config = ROSE2SourceFormConfig(
            resolution_m=float(config.resolution_m),
            min_room_area_m2=float(config.min_room_area_m2),
            min_cell_area_m2=float(config.source_form_min_cell_area_m2),
            min_cell_free_ratio=float(config.source_form_min_cell_free_ratio),
            min_cut_spacing_m=float(config.source_form_min_cut_spacing_m),
            min_wall_line_length_m=float(config.hough_min_line_length_m),
            axis_snap_angle_rad=float(config.source_form_axis_snap_angle_rad),
            max_cuts_per_axis=int(config.source_form_max_cuts_per_axis),
            wall_raster_radius_cells=int(self._structure_config.wall_raster_radius_cells),
            debug_dump=bool(config.debug_rose2_source),
            debug_dir=str(config.rose2_source_work_dir),
            thin_wall_separator_enabled=bool(config.source_form_thin_wall_separator_enabled),
            thin_wall_min_length_m=float(config.source_form_thin_wall_min_length_m),
            thin_wall_max_width_m=float(config.source_form_thin_wall_max_width_m),
            thin_wall_min_aspect_ratio=float(config.source_form_thin_wall_min_aspect_ratio),
            thin_wall_free_support_band_m=float(config.source_form_thin_wall_free_support_band_m),
            thin_wall_min_free_support_ratio=float(config.source_form_thin_wall_min_free_support_ratio),
            topology_effective_separator_enabled=bool(config.source_form_topology_effective_separator_enabled),
            topology_effective_min_largest_component_drop=float(config.source_form_topology_effective_min_largest_component_drop),
            topology_effective_max_candidates=int(config.source_form_topology_effective_max_candidates),
            doorway_partition_enabled=bool(config.source_form_doorway_partition_enabled),
            doorway_width_min_m=float(config.source_form_doorway_width_min_m),
            doorway_width_max_m=float(config.source_form_doorway_width_max_m),
            doorway_min_wall_support_on_sides_m=float(config.source_form_doorway_min_wall_support_on_sides_m),
            merge_guard_enabled=bool(config.source_form_merge_guard_enabled),
        )
        self.last_debug: dict = {}
        self.last_result: UpstreamROSE2Result | None = None
        self._previous: dict[str, RoomMask] = {}

    def segment(self, occupancy_grid: np.ndarray, observed_free: np.ndarray | None = None) -> UpstreamROSE2Result:
        occupied = np.asarray(occupancy_grid, dtype=bool)
        free = np.asarray(observed_free, dtype=bool) if observed_free is not None else ~occupied
        unknown = ~(occupied | free)
        proposals, state = self.build_proposals(occupied, free, occupied, unknown, step=0, object_memory=None)
        _ = proposals
        room_masks = self.finalize_proposals(state, proposal_semantic_labels=None)
        source_result = state.debug.get("_source_result")
        if isinstance(source_result, ROSE2SourceResult):
            clean = source_result.clean_structure_map
            main_directions = [float(v) for v in source_result.dominant_directions_rad]
            wall_lines = list(source_result.hough_segments)
            extended_lines = list(source_result.extended_lines)
            edges = list(source_result.face_adjacency_edges)
        else:
            clean = np.zeros_like(occupied, dtype=bool)
            main_directions = []
            wall_lines = []
            extended_lines = []
            edges = []
        room_polygons = [_room_polygon_from_mask(room.mask, self.map_info) for room in room_masks]
        final_label_map = np.asarray(self.last_debug.get("final_room_label_map", state.proposal_labels), dtype=np.int32)
        result = UpstreamROSE2Result(
            room_masks=room_masks,
            clean_structure_map=np.asarray(clean, dtype=bool),
            main_directions=main_directions,
            wall_lines=wall_lines,
            extended_lines=extended_lines,
            edges=edges,
            room_polygons=room_polygons,
            room_label_map=final_label_map,
            debug=dict(self.last_debug),
        )
        self.last_result = result
        return result

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Optional[Sequence[object]] = None,
        vertical_profile: VerticalProfileMap | None = None,
        roomseg_static_structural_occupied: np.ndarray | None = None,
        roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    ) -> List[RoomMask]:
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
        object_memory: Optional[Sequence[object]] = None,
        vertical_profile: VerticalProfileMap | None = None,
        roomseg_static_structural_occupied: np.ndarray | None = None,
        roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[List[RoomMask], RoomProposalState]:
        free = np.asarray(observed_free_mask, dtype=bool)
        occupied = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool)
        unknown = np.asarray(unknown_mask, dtype=bool)
        if free.shape != occupied.shape or free.shape != unknown.shape:
            raise ValueError("upstream ROSE2 masks must have the same HxW shape")
        structural = _vertical_profile_structural_maps(
            occupied=occupied,
            free=free,
            unknown=unknown,
            vertical_profile=ensure_vertical_profile(vertical_profile, occupied.shape),
            vertical_profile_provided=vertical_profile is not None,
            roomseg_static_structural_occupied=roomseg_static_structural_occupied,
            roomseg_ray_evidence=roomseg_ray_evidence,
            object_memory=object_memory or [],
            map_info=self.map_info,
            config=self.config,
            room_config=self._room_config,
        )
        if int(structural.get("navigation_free_added_to_strict_roomseg_cells", -1)) != 0:
            raise AssertionError("strict room segmentation must not add navigation-free cells to vertical-free input")
        backend = str(self.config.backend or SOURCE_EXTERNAL_RUNNER_BACKEND).strip().lower()
        source_result, legacy_structure = self._run_source_backend(
            backend=backend,
            structural=structural,
            object_memory=object_memory,
            step=int(step),
        )
        proposal_labels = np.asarray(source_result.room_label_map, dtype=np.int32)
        proposal_rooms = self._rooms_from_labels(proposal_labels, unknown, [], int(step), source_labels=proposal_labels)
        debug = self._debug_from_source_result(source_result, proposal_labels, int(step), legacy_structure=legacy_structure)
        debug.update(structural)
        debug["_source_result"] = source_result
        debug["_structure_result"] = legacy_structure
        debug["_input_occupancy_map"] = occupied.copy()
        state = RoomProposalState(
            proposal_labels=proposal_labels,
            structural_free_mask=np.asarray(structural["repaired_roomseg_free"], dtype=bool),
            structural_obstacle_mask=np.asarray(source_result.boundary_map, dtype=bool),
            unknown_mask=np.asarray(structural["repaired_roomseg_unknown"], dtype=bool),
            distance_m=np.asarray(source_result.structural_score, dtype=np.float32),
            step=int(step),
            debug=debug,
        )
        return proposal_rooms, state

    def _run_source_backend(
        self,
        *,
        backend: str,
        structural: Mapping[str, object],
        object_memory: Optional[Sequence[object]],
        step: int,
    ) -> tuple[ROSE2SourceResult, StructureExtractionResult | None]:
        occupied = np.asarray(structural["repaired_roomseg_occupied"], dtype=bool)
        free = np.asarray(structural["repaired_roomseg_free"], dtype=bool)
        unknown = np.asarray(structural["repaired_roomseg_unknown"], dtype=bool)
        backend_name = str(backend or SOURCE_EXTERNAL_RUNNER_BACKEND).strip().lower()
        legacy_structure: StructureExtractionResult | None = None
        if backend_name == SOURCE_FAITHFUL_BACKEND:
            if bool(self.config.strict_disallow_legacy_fallback) and not bool(self.config.allow_source_form_in_metric):
                raise ValueError(
                    "%s is debug/ablation-only; strict room segmentation requires %s"
                    % (SOURCE_FAITHFUL_BACKEND, SOURCE_EXTERNAL_RUNNER_BACKEND)
                )
            source_result = run_rose2_source_faithful_v1(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                vertical_observed=np.asarray(structural.get("vertical_observed_map", structural.get("vertical_observed")), dtype=bool),
                vertical_free=np.asarray(structural.get("vertical_free_room_domain", free), dtype=bool),
                wall_confidence_map=np.asarray(structural.get("wall_confidence_map", np.zeros_like(occupied, dtype=np.float32)), dtype=np.float32),
                structure_config=self._structure_config,
                source_config=self._source_form_config,
                object_memory=object_memory,
            )
        elif backend_name in {SOURCE_FORM_BACKEND, "rose2_source_form"}:
            if bool(self.config.strict_disallow_legacy_fallback) and not bool(self.config.allow_source_form_in_metric):
                raise ValueError(
                    "%s is debug/ablation-only; strict room segmentation requires %s"
                    % (backend_name, SOURCE_EXTERNAL_RUNNER_BACKEND)
                )
            source_result = run_rose2_source_form_v2(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                vertical_observed=np.asarray(structural.get("vertical_observed_map", structural.get("vertical_observed")), dtype=bool),
                vertical_free=np.asarray(structural.get("vertical_free_room_domain", free), dtype=bool),
                wall_confidence_map=np.asarray(structural.get("wall_confidence_map", np.zeros_like(occupied, dtype=np.float32)), dtype=np.float32),
                structure_config=self._structure_config,
                source_config=self._source_form_config,
                object_memory=object_memory,
            )
        elif backend_name in {SOURCE_EXTERNAL_BACKEND, SOURCE_EXTERNAL_RUNNER_BACKEND}:
            step_dir = Path(str(self.config.rose2_source_work_dir)) / ("rose2_source_step_%06d" % int(step))
            source_result = run_rose2_source_external_runner(
                source_root=self.source_root,
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                vertical_observed=np.asarray(structural.get("vertical_observed_map", structural.get("vertical_observed")), dtype=bool),
                vertical_free=np.asarray(structural.get("vertical_free_room_domain", free), dtype=bool),
                wall_confidence_map=np.asarray(structural.get("wall_confidence_map", np.zeros_like(occupied, dtype=np.float32)), dtype=np.float32),
                resolution_m=float(self.config.resolution_m),
                min_room_area_m2=float(self.config.min_room_area_m2),
                work_dir=str(step_dir),
                timeout_s=float(self.config.external_runner_timeout_s),
                python_executable=self.config.external_runner_python_executable,
                encoding=str(self.config.external_runner_encoding or "auto"),
                keep_work_dir=bool(self.config.external_runner_keep_work_dir),
                parameter_overrides=dict(self.config.external_runner_parameter_overrides or {}),
                cache_enabled=bool(self.config.external_runner_cache_enabled),
            )
        elif backend_name in {VERTICAL_FREE_ROOMSEG_BACKEND, VERTICAL_FREE_ROOMSEG_ALGORITHM}:
            vf_cfg = VerticalFreeRoomSegConfig.from_mapping(
                self.config.vertical_free_roomseg or {},
                resolution_m=float(self.config.resolution_m),
                min_room_area_m2=float(self.config.min_room_area_m2),
                debug_dump=bool(
                    self.config.debug_dump
                    and bool(dict(self.config.vertical_free_roomseg or {}).get("debug_dump", self.config.debug_dump))
                ),
                debug_dir=str(dict(self.config.vertical_free_roomseg or {}).get("debug_dir", self.config.debug_dir)),
            )
            vf_result = run_vertical_free_roomseg(
                observed_free=free,
                observed_occupied=occupied,
                unknown=unknown,
                resolution_m=float(self.config.resolution_m),
                config=vf_cfg,
            )
            if bool(vf_cfg.debug_dump):
                dump = save_vertical_free_roomseg_debug(
                    result=vf_result,
                    out_dir=str(vf_cfg.debug_dir),
                    step=int(step),
                )
                vf_result.debug["vertical_free_roomseg_layers"] = dict(dump.get("paths", {}))
                vf_result.debug["vertical_free_roomseg_debug_summary"] = dict(dump.get("summary", {}))
            source_result = vertical_free_result_to_source_result(vf_result)
        elif backend_name in {VERTICAL_FREE_GAP_CLOSURE_BACKEND, VERTICAL_FREE_GAP_CLOSURE_ALGORITHM}:
            vfgc_cfg = VFGCConfig.from_mapping(
                self.config.vertical_free_gap_closure or {},
                resolution_m=float(self.config.resolution_m),
                min_room_area_m2=float(self.config.min_room_area_m2),
                debug_dump=bool(
                    self.config.debug_dump
                    and bool(dict(self.config.vertical_free_gap_closure or {}).get("debug_dump", self.config.debug_dump))
                ),
                debug_dir=str(dict(self.config.vertical_free_gap_closure or {}).get("debug_dir", self.config.debug_dir)),
            )
            vfgc_result = run_vertical_free_gap_closure_roomseg(
                free_mask=free,
                wall_mask=occupied,
                unknown_mask=unknown,
                resolution_m=float(self.config.resolution_m),
                config=vfgc_cfg,
            )
            if bool(vfgc_cfg.debug_dump):
                dump = save_vertical_free_gap_closure_debug(
                    result=vfgc_result,
                    out_dir=str(vfgc_cfg.debug_dir),
                    step=int(step),
                )
                vfgc_result.debug["vertical_free_gap_closure_layers"] = dict(dump.get("paths", {}))
                vfgc_result.debug["vertical_free_gap_closure_debug_summary"] = dict(dump.get("summary", {}))
            source_result = vfgc_result_to_source_result(vfgc_result)
        elif backend_name == LEGACY_STYLE_BACKEND:
            if bool(self.config.strict_disallow_legacy_fallback):
                raise ValueError(
                    "%s is debug/ablation-only; strict room segmentation uses %s"
                    % (LEGACY_STYLE_BACKEND, SOURCE_EXTERNAL_RUNNER_BACKEND)
                )
            legacy_structure = extract_rose2_structure(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                config=self._structure_config,
                object_memory=object_memory,
            )
            source_result = _source_result_from_legacy_structure(legacy_structure)
        else:
            raise ValueError(
                "unsupported roomseg backend %s; expected %s, %s, %s, %s, %s, %s, or %s"
                    % (
                        backend_name,
                        SOURCE_FAITHFUL_BACKEND,
                        SOURCE_FORM_BACKEND,
                        SOURCE_EXTERNAL_BACKEND,
                        SOURCE_EXTERNAL_RUNNER_BACKEND,
                        LEGACY_STYLE_BACKEND,
                        VERTICAL_FREE_ROOMSEG_BACKEND,
                        VERTICAL_FREE_GAP_CLOSURE_BACKEND,
                    )
                )
        if bool(self.config.rose2_compare_legacy) and backend_name != LEGACY_STYLE_BACKEND:
            legacy_structure = extract_rose2_structure(
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
                config=self._structure_config,
                object_memory=object_memory,
            )
            source_result.debug["legacy_compare"] = {
                "enabled": True,
                "legacy_room_count": _label_count(legacy_structure.face_labels),
                "legacy_wall_line_count": int(len(legacy_structure.representative_lines)),
                "legacy_connected_component_rooms_used": False,
            }
        source_result.debug.update(
            {
                "vertical_free_room_domain": np.asarray(structural.get("vertical_free_room_domain", free), dtype=bool),
                "vertical_observed": np.asarray(structural.get("vertical_observed_map", structural.get("vertical_observed", occupied | free)), dtype=bool),
                "observed_not_vertical_free": np.asarray(structural.get("vertical_observed_map", occupied | free), dtype=bool)
                & ~np.asarray(structural.get("vertical_free_room_domain", free), dtype=bool),
                "repaired_roomseg_free": free,
                "repaired_roomseg_occupied": occupied,
                "repaired_roomseg_unknown": unknown,
                "wall_confidence_map": np.asarray(structural.get("wall_confidence_map", np.zeros_like(occupied, dtype=np.float32)), dtype=np.float32),
            }
        )
        if bool(self.config.debug_rose2_source):
            dump = save_rose2_source_debug(
                out_dir=str(self.config.rose2_source_work_dir),
                step=int(step),
                result=source_result,
                observed_occupied=occupied,
                observed_free=free,
                unknown=unknown,
            )
            source_result.debug["rose2_source_debug_paths"] = dict(dump.get("paths", {}))
            source_result.debug["rose2_source_debug_summary"] = dict(dump.get("summary", {}))
        return source_result, legacy_structure

    def _apply_window_door_gap_repair(
        self,
        structure: StructureExtractionResult,
        *,
        occupied: np.ndarray,
        free: np.ndarray,
        unknown: np.ndarray,
        vertical_profile: VerticalProfileMap,
        object_memory: Sequence[object],
        wall_confidence_map: np.ndarray,
    ) -> dict:
        base_count = _label_count(structure.face_labels)
        closure = _classify_and_repair_wall_gaps(
            occupied=np.asarray(occupied, dtype=bool),
            free=np.asarray(free, dtype=bool),
            unknown=np.asarray(unknown, dtype=bool),
            boundary=np.asarray(structure.boundary_map, dtype=bool),
            vertical_profile=vertical_profile,
            object_memory=object_memory,
            wall_confidence_map=np.asarray(wall_confidence_map, dtype=np.float32),
            config=self._room_config,
            adapter_config=self.config,
        )
        debug = {
            "window_gap_repair": {
                "repaired_window_gaps": list(closure.get("repaired_window_gaps") or []),
                "verified_doorway_gaps": list(closure.get("verified_doorway_gaps") or []),
                "closed_nontraversable_gaps": list(closure.get("closed_nontraversable_gaps") or []),
            },
            "repaired_window_gaps": list(closure.get("repaired_window_gaps") or []),
            "verified_doorway_gaps": list(closure.get("verified_doorway_gaps") or []),
        }
        if not closure["lines"]:
            return debug
        boundary = np.asarray(closure["boundary"], dtype=bool)
        labels, faces = faces_from_boundary_map(free, boundary, self._structure_config)
        new_count = _label_count(labels)
        if new_count <= max(1, base_count):
            return debug
        structure.boundary_map = boundary
        structure.face_labels = labels.astype(np.int32)
        structure.faces = faces
        structure.face_adjacency_edges = face_adjacency_edges(labels, boundary)
        offset = len(structure.representative_lines)
        for idx, line in enumerate(closure["lines"]):
            row = dict(line)
            row["line_id"] = int(offset + idx)
            row["source"] = str(row.get("source", "vertical_profile_gap_repair"))
            structure.representative_lines.append(row)
        debug["gap_repair_applied"] = True
        return debug

    def finalize_proposals(
        self,
        proposal_state: RoomProposalState,
        proposal_semantic_labels: Optional[Mapping[int, object]] = None,
    ) -> List[RoomMask]:
        labels = np.asarray(proposal_state.proposal_labels, dtype=np.int32)
        finalization_mode = str(self._room_config.finalization_mode or "doorway_constrained_merge").strip().lower()
        if not np.any(labels > 0):
            debug = {
                key: value
                for key, value in dict(proposal_state.debug or {}).items()
                if key not in {"_source_result", "_structure_result", "_input_occupancy_map"}
            }
            debug.update(self._empty_debug(int(proposal_state.step)))
            debug["finalization_mode"] = finalization_mode
            self.last_debug = debug
            return []
        if finalization_mode in {
            "no_merge",
            "proposal_only",
            "premerge_proposals",
            "no_merge_until_source_backend_verified",
            "no_merge_until_geometry_verified",
        }:
            final_labels = labels.copy()
            doorway_edges: list[dict] = []
            finalization_debug = {
                "proposal_room_count": int(len([v for v in np.unique(labels) if int(v) > 0])),
                "final_room_count": int(len([v for v in np.unique(final_labels) if int(v) > 0])),
                "proposal_room_masks": _proposal_masks_debug(labels),
                "finalization_mode": finalization_mode,
                "merge_disabled": True,
                "merge_operations": [],
                "doorway_edges": [],
                "functional_split_edges": [],
                "adjacency_evidence": [],
                "adjacency_decisions": [],
                "merge_guard_enabled": bool(self.config.source_form_merge_guard_enabled),
                "merge_blocked_by_strong_boundary_count": 0,
            }
        elif finalization_mode == "doorway_constrained_merge":
            final_labels, finalization_debug, doorway_edges = merge_open_plan_proposals(
                proposal_labels=labels,
                structural_free_mask=proposal_state.structural_free_mask,
                structural_obstacle_mask=proposal_state.structural_obstacle_mask,
                unknown_mask=proposal_state.unknown_mask,
                distance_m=proposal_state.distance_m,
                config=self._room_config,
                proposal_semantic_labels=proposal_semantic_labels,
            )
        else:
            raise ValueError("unsupported upstream ROSE2 finalization_mode: %s" % self._room_config.finalization_mode)
        final_labels, free_absorption_debug = _absorb_unlabeled_structural_free_into_rooms(
            final_labels,
            structural_free_mask=proposal_state.structural_free_mask,
            unknown_mask=proposal_state.unknown_mask,
        )
        input_occupancy = np.asarray(proposal_state.debug.get("_input_occupancy_map", proposal_state.structural_obstacle_mask), dtype=bool)
        context_labels = np.asarray(final_labels, dtype=np.int32).copy()
        context_reliability = np.zeros_like(context_labels, dtype=np.float32)
        context_reliability[context_labels > 0] = 1.0
        overlay_debug = {"nav_free_overlay_enabled": False}
        if bool(self.config.navigation_free_context_overlay_enabled):
            context_labels, context_reliability, overlay_debug = build_navigation_free_room_context_overlay(
                final_labels,
                navigation_free=np.asarray(proposal_state.debug.get("navigation_free_room_domain", proposal_state.structural_free_mask), dtype=bool),
                unknown=proposal_state.unknown_mask,
                obstacle=input_occupancy if bool(self.config.navigation_free_context_overlay_do_not_cross_obstacle) else None,
                structural_boundary=proposal_state.structural_obstacle_mask
                if bool(self.config.navigation_free_context_overlay_do_not_cross_structural_boundary)
                else None,
                resolution_m=float(self.config.resolution_m),
                max_absorb_distance_m=float(self.config.navigation_free_context_overlay_max_absorb_distance_m),
                min_seed_room_area_cells=int(self.config.navigation_free_context_overlay_min_seed_room_area_cells),
                protect_unknown=bool(self.config.navigation_free_context_overlay_protect_unknown),
                do_not_cross_structural_boundary=bool(self.config.navigation_free_context_overlay_do_not_cross_structural_boundary),
                do_not_cross_obstacle=bool(self.config.navigation_free_context_overlay_do_not_cross_obstacle),
                absorbed_reliability=float(self.config.navigation_free_context_overlay_reliability_for_absorbed_cells),
            )
        room_labels_for_nodes = context_labels if (
            bool(self.config.navigation_free_context_overlay_enabled)
            and bool(self.config.navigation_free_context_overlay_use_for_room_nodes)
        ) else final_labels
        rooms = self._rooms_from_labels(room_labels_for_nodes, proposal_state.unknown_mask, doorway_edges, int(proposal_state.step), source_labels=labels)
        if bool(self.config.navigation_free_context_overlay_enabled):
            for room in rooms:
                label_id = int((room.metadata or {}).get("label_id", 0) or 0)
                strict_cells = int(np.count_nonzero(final_labels == label_id))
                context_cells = int(np.count_nonzero(context_labels == label_id))
                room.metadata["context_overlay_applied"] = bool(self.config.navigation_free_context_overlay_use_for_room_nodes)
                room.metadata["strict_mask_area_cells"] = int(strict_cells)
                room.metadata["context_mask_area_cells"] = int(context_cells)
                room.metadata["absorbed_context_cells"] = int(max(0, context_cells - strict_cells))
        rooms = self._assign_stable_ids(rooms)
        debug = {
            key: value
            for key, value in dict(proposal_state.debug or {}).items()
            if key not in {"_source_result", "_structure_result", "_input_occupancy_map"}
        }
        debug_algorithm = str(debug.get("algorithm", getattr(self, "source", UPSTREAM_ALGORITHM)) or getattr(self, "source", UPSTREAM_ALGORITHM))
        debug_source = str(debug.get("source", debug_algorithm) or debug_algorithm)
        source_repository = None if debug_algorithm == VERTICAL_FREE_ROOMSEG_ALGORITHM else "https://github.com/goldleaf3i/declutter-reconstruct"
        debug.update(
            {
                "algorithm": debug_algorithm,
                "source": debug_source,
                "source_mode": str(self.config.source_mode),
                "roomseg_backend": str(self.config.backend or SOURCE_EXTERNAL_RUNNER_BACKEND),
                "source_backend": str(self.config.backend or SOURCE_EXTERNAL_RUNNER_BACKEND),
                "source_root": str(self.source_root) if self.source_root is not None else None,
                "source_repository": source_repository,
                "source_provenance": _source_provenance(self.source_root),
                "strict_fallback_used": False,
                "step": int(proposal_state.step),
                "finalization_mode": finalization_mode,
                "merge_disabled": bool(finalization_debug.get("merge_disabled", False)),
                "proposal_room_count": int(finalization_debug.get("proposal_room_count", 0)),
                "final_room_count": int(len(rooms)),
                "room_count": int(len(rooms)),
                "num_rooms": int(len(rooms)),
                "rooms": [room_mask_to_debug(room) for room in rooms],
                "room_masks": [room_mask_to_debug(room) for room in rooms],
                "room_labels": [],
                "merge_operations": list(finalization_debug.get("merge_operations") or []),
                "merge_guard_enabled": bool(finalization_debug.get("merge_guard_enabled", self.config.source_form_merge_guard_enabled)),
                "merge_blocked_by_strong_boundary_count": int(finalization_debug.get("merge_blocked_by_strong_boundary_count", 0) or 0),
                "merge_split_decisions": list(finalization_debug.get("adjacency_decisions") or []),
                "functional_split_edges": list(finalization_debug.get("functional_split_edges") or []),
                "adjacency_evidence": list(finalization_debug.get("adjacency_evidence") or []),
                "adjacency_decisions": list(finalization_debug.get("adjacency_decisions") or []),
                "doorway_edges": list(finalization_debug.get("doorway_edges") or []),
                "room_free_absorption": free_absorption_debug,
                "unlabeled_structural_free_cells_before_absorb": int(free_absorption_debug["before_unlabeled_free_cells"]),
                "absorbed_structural_free_cells": int(free_absorption_debug["absorbed_free_cells"]),
                "unlabeled_structural_free_cells_after_absorb": int(free_absorption_debug["after_unlabeled_free_cells"]),
                "final_room_label_map": np.asarray(final_labels, dtype=np.int32),
                "room_labels_after_merge": np.asarray(final_labels, dtype=np.int32),
                "context_room_label_map": np.asarray(context_labels, dtype=np.int32),
                "context_room_reliability_map": np.asarray(context_reliability, dtype=np.float32),
                "navigation_free_context_overlay": dict(overlay_debug),
            }
        )
        self.last_debug = debug
        self._previous = {room.room_id: room for room in rooms}
        if bool(self.config.debug_dump):
            self._write_debug_dump(int(proposal_state.step), rooms, debug, input_occupancy)
        return rooms

    def _rooms_from_labels(
        self,
        labels: np.ndarray,
        unknown: np.ndarray,
        doorway_edges: Sequence[Mapping[str, object]],
        step: int,
        source_labels: np.ndarray,
    ) -> List[RoomMask]:
        rooms: List[RoomMask] = []
        for label_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            mask = labels == label_id
            room = _room_from_mask("pending", mask, unknown, doorway_edges, self.map_info, int(step), int(label_id))
            source_name = str(getattr(self, "source", UPSTREAM_ALGORITHM))
            room.source = source_name
            room.metadata["algorithm"] = source_name
            room.metadata["segmentation_source"] = source_name
            room.metadata["source_mode"] = str(self.config.source_mode)
            room.metadata["label_id"] = int(label_id)
            room.metadata["proposal_labels"] = sorted(int(v) for v in np.unique(source_labels[mask]) if int(v) > 0)
            rooms.append(room)
        return rooms

    def _assign_stable_ids(self, rooms: Sequence[RoomMask]) -> List[RoomMask]:
        previous = dict(self._previous)
        used: set[str] = set()
        next_index = 1
        for room in rooms:
            best_id = None
            best_iou = 0.0
            for old_id, old_room in previous.items():
                if old_id in used:
                    continue
                iou = _mask_iou(room.mask, old_room.mask)
                if iou > best_iou:
                    best_id = old_id
                    best_iou = iou
            if best_id is None or best_iou < 0.35:
                while "room_%04d" % next_index in used:
                    next_index += 1
                best_id = "room_%04d" % next_index
                next_index += 1
            room.room_id = best_id
            used.add(best_id)
        return list(rooms)

    def _debug_from_structure(self, structure: StructureExtractionResult, proposal_labels: np.ndarray, step: int) -> dict:
        return {
            "algorithm": UPSTREAM_ALGORITHM,
            "source": UPSTREAM_ALGORITHM,
            "source_mode": str(self.config.source_mode),
            "roomseg_backend": str(self.config.backend or SOURCE_EXTERNAL_RUNNER_BACKEND),
            "source_backend": str(self.config.backend or SOURCE_EXTERNAL_RUNNER_BACKEND),
            "source_root": str(self.source_root) if self.source_root is not None else None,
            "source_repository": "https://github.com/goldleaf3i/declutter-reconstruct",
            "source_provenance": _source_provenance(self.source_root),
            "strict_fallback_used": False,
            "step": int(step),
            "dominant_directions_rad": [float(v) for v in structure.dominant_directions_rad],
            "main_directions": [float(v) for v in structure.dominant_directions_rad],
            "clean_structure_map": np.asarray(structure.clean_structure_map, dtype=bool),
            "boundary_map": np.asarray(structure.boundary_map, dtype=bool),
            "hough_segments": list(structure.hough_segments),
            "wall_lines": list(structure.hough_segments),
            "wall_clusters": list(structure.wall_clusters),
            "representative_lines": list(structure.representative_lines),
            "extended_lines": list(structure.representative_lines),
            "edges": list(structure.face_adjacency_edges),
            "faces": list(structure.faces),
            "room_polygons": [_room_polygon_from_mask(proposal_labels == label_id, self.map_info) for label_id in sorted(int(v) for v in np.unique(proposal_labels) if int(v) > 0)],
            "timing_ms": dict(structure.timing_ms),
            "num_hough_segments": int(len(structure.hough_segments)),
            "num_wall_lines": int(len(structure.hough_segments)),
            "num_wall_clusters": int(len(structure.wall_clusters)),
            "num_representative_lines": int(len(structure.representative_lines)),
            "num_physical_rooms": int(len([v for v in np.unique(proposal_labels) if int(v) > 0])),
            "proposal_room_count": int(len([v for v in np.unique(proposal_labels) if int(v) > 0])),
            "proposal_room_masks": _proposal_masks_debug(proposal_labels),
            "room_proposal_labels_before_merge": np.asarray(proposal_labels, dtype=np.int32),
        }

    def _debug_from_source_result(
        self,
        source_result: ROSE2SourceResult,
        proposal_labels: np.ndarray,
        step: int,
        *,
        legacy_structure: StructureExtractionResult | None = None,
    ) -> dict:
        debug = {
            "algorithm": UPSTREAM_ALGORITHM,
            "source": UPSTREAM_ALGORITHM,
            "source_mode": str(self.config.source_mode),
            "roomseg_backend": str(source_result.backend),
            "source_backend": str(source_result.backend),
            "source_root": str(self.source_root) if self.source_root is not None else None,
            "source_repository": "https://github.com/goldleaf3i/declutter-reconstruct",
            "source_provenance": _source_provenance(self.source_root),
            "strict_fallback_used": False,
            "step": int(step),
            "dominant_directions_rad": [float(v) for v in source_result.dominant_directions_rad],
            "main_directions": [float(v) for v in source_result.dominant_directions_rad],
            "clean_structure_map": np.asarray(source_result.clean_structure_map, dtype=bool),
            "boundary_map": np.asarray(source_result.boundary_map, dtype=bool),
            "hough_segments": list(source_result.hough_segments),
            "wall_lines": list(source_result.hough_segments),
            "wall_clusters": list(source_result.wall_clusters),
            "representative_lines": list(source_result.extended_lines or source_result.representative_lines),
            "extended_lines": list(source_result.extended_lines or source_result.representative_lines),
            "edges": list(source_result.face_adjacency_edges),
            "cell_edges": list(source_result.cell_edges),
            "faces": list(source_result.faces),
            "source_cell_polygons": list(source_result.cell_polygons),
            "room_polygons": [
                _room_polygon_from_mask(proposal_labels == label_id, self.map_info)
                for label_id in sorted(int(v) for v in np.unique(proposal_labels) if int(v) > 0)
            ],
            "timing_ms": dict(source_result.timing_ms),
            "num_hough_segments": int(len(source_result.hough_segments)),
            "num_wall_lines": int(len(source_result.hough_segments)),
            "num_wall_clusters": int(len(source_result.wall_clusters)),
            "num_representative_lines": int(len(source_result.extended_lines or source_result.representative_lines)),
            "num_source_cells": int(len(source_result.cell_polygons)),
            "num_source_cell_edges": int(len(source_result.cell_edges)),
            "num_physical_rooms": int(len([v for v in np.unique(proposal_labels) if int(v) > 0])),
            "proposal_room_count": int(len([v for v in np.unique(proposal_labels) if int(v) > 0])),
            "proposal_room_masks": _proposal_masks_debug(proposal_labels),
            "room_proposal_labels_before_merge": np.asarray(proposal_labels, dtype=np.int32),
            "source_room_label_map": np.asarray(source_result.source_room_label_map, dtype=np.int32),
            "source_result_summary": source_result_summary(source_result),
        }
        debug.update(dict(source_result.debug or {}))
        if legacy_structure is not None and "legacy_compare" not in debug:
            debug["legacy_compare"] = {
                "enabled": bool(source_result.backend == LEGACY_STYLE_BACKEND),
                "legacy_room_count": _label_count(legacy_structure.face_labels),
                "legacy_wall_line_count": int(len(legacy_structure.representative_lines)),
                "legacy_connected_component_rooms_used": bool(source_result.backend == LEGACY_STYLE_BACKEND),
            }
        return debug

    def _empty_debug(self, step: int) -> dict:
        return {
            "algorithm": UPSTREAM_ALGORITHM,
            "source": UPSTREAM_ALGORITHM,
            "source_mode": str(self.config.source_mode),
            "source_root": str(self.source_root) if self.source_root is not None else None,
            "source_repository": "https://github.com/goldleaf3i/declutter-reconstruct",
            "source_provenance": _source_provenance(self.source_root),
            "strict_fallback_used": False,
            "step": int(step),
            "num_rooms": 0,
            "num_wall_lines": 0,
            "main_directions": [],
            "room_masks": [],
            "room_labels": [],
            "rooms": [],
        }

    def _write_debug_dump(self, step: int, rooms: Sequence[RoomMask], debug: Mapping[str, object], occupancy: np.ndarray) -> None:
        from isaac_bench.mapping.room_segmentation_debug import save_rose2_roomseg_debug

        save_rose2_roomseg_debug(
            out_dir=Path(str(self.config.debug_dir or "debug/roomseg")),
            episode_id="episode",
            step=int(step),
            occupancy=np.asarray(occupancy, dtype=bool),
            room_masks=rooms,
            debug=debug,
            room_labels={},
        )


def occupancy_to_rose_image(occupancy_grid: np.ndarray, observed_free: np.ndarray | None) -> np.ndarray:
    occupied = np.asarray(occupancy_grid, dtype=bool)
    if observed_free is None:
        free = ~occupied
        unknown = np.zeros_like(occupied, dtype=bool)
    else:
        free = np.asarray(observed_free, dtype=bool)
        unknown = ~(occupied | free)
    image = np.full(occupied.shape, 127, dtype=np.uint8)
    image[free] = 0
    image[occupied] = 255
    image[unknown] = 127
    return image


def _source_result_from_legacy_structure(structure: StructureExtractionResult) -> ROSE2SourceResult:
    debug = {
        "source_backend": LEGACY_STYLE_BACKEND,
        "source_form_used": False,
        "source_exact_used": False,
        "legacy_style_used": True,
        "legacy_connected_component_rooms_used": True,
        "source_room_count": _label_count(structure.face_labels),
        "proposal_room_count": _label_count(structure.face_labels),
        "failure_mode": "",
    }
    return ROSE2SourceResult(
        backend=LEGACY_STYLE_BACKEND,
        room_label_map=np.asarray(structure.face_labels, dtype=np.int32),
        source_room_label_map=np.asarray(structure.face_labels, dtype=np.int32),
        clean_structure_map=np.asarray(structure.clean_structure_map, dtype=bool),
        structural_score=np.asarray(structure.structural_score, dtype=np.float32),
        boundary_map=np.asarray(structure.boundary_map, dtype=bool),
        dominant_directions_rad=[float(v) for v in structure.dominant_directions_rad],
        hough_segments=[dict(item) for item in structure.hough_segments],
        wall_clusters=[dict(item) for item in structure.wall_clusters],
        representative_lines=[dict(item) for item in structure.representative_lines],
        extended_lines=[dict(item) for item in structure.representative_lines],
        faces=[dict(item) for item in structure.faces],
        face_adjacency_edges=[dict(item) for item in structure.face_adjacency_edges],
        cell_edges=[],
        cell_polygons=[],
        timing_ms=dict(structure.timing_ms),
        debug=debug,
    )


def rose_polygons_to_room_masks(polygons: list[np.ndarray], shape: tuple[int, int], map_info: MapInfo) -> list[RoomMask]:
    rooms: list[RoomMask] = []
    h, w = int(shape[0]), int(shape[1])
    for idx, polygon in enumerate(polygons, start=1):
        arr = np.asarray(polygon, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            continue
        cells = []
        for x, y in arr[:, :2]:
            r, c = world_xy_to_grid(float(x), float(y), map_info)
            cells.append((int(c), int(r)))
        image = Image.new("L", (w, h), 0)
        ImageDraw.Draw(image).polygon(cells, fill=1)
        mask = np.asarray(image, dtype=np.uint8).astype(bool)
        if not np.any(mask):
            continue
        room = _room_from_mask("room_%04d" % idx, mask, np.zeros((h, w), dtype=bool), [], map_info, 0, idx)
        room.metadata["algorithm"] = UPSTREAM_ALGORITHM
        room.metadata["segmentation_source"] = UPSTREAM_ALGORITHM
        rooms.append(room)
    return rooms


def validate_upstream_rose2_source_root(source_root: str | None, env_name: str = DEFAULT_SOURCE_ENV, fail: bool = True) -> Path | None:
    root_value = source_root or os.environ.get(str(env_name or DEFAULT_SOURCE_ENV))
    if root_value:
        root = Path(str(root_value)).expanduser()
        if _looks_like_declutter_reconstruct(root):
            return root
    if fail:
        hint = (
            "Missing upstream ROSE2 pure-Python source root. "
            "Set %s to a goldleaf3i/declutter-reconstruct checkout containing %s."
            % (str(env_name or DEFAULT_SOURCE_ENV), ", ".join(REQUIRED_SOURCE_FILES))
        )
        raise FileNotFoundError(hint)
    return None


def _looks_like_declutter_reconstruct(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    lower_files = {str(path.relative_to(root)).replace("\\", "/").lower() for path in root.rglob("*") if path.is_file()}
    required = [item.lower() for item in REQUIRED_SOURCE_FILES]
    return all(item in lower_files for item in required)


def _source_provenance(root: Path | None) -> dict:
    if root is None:
        return {"available": False, "required_files": list(REQUIRED_SOURCE_FILES), "files": []}
    files = []
    for rel in REQUIRED_SOURCE_FILES:
        path = root / rel
        if not path.exists():
            files.append({"path": rel, "exists": False, "sha256": None})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": rel, "exists": True, "sha256": digest})
    return {"available": all(item["exists"] for item in files), "required_files": list(REQUIRED_SOURCE_FILES), "files": files}


def _room_from_mask(
    room_id: str,
    mask: np.ndarray,
    unknown: np.ndarray,
    doorway_edges: Sequence[Mapping[str, object]],
    map_info: MapInfo,
    step: int,
    label_id: int,
) -> RoomMask:
    arr = np.asarray(mask, dtype=bool)
    rr, cc = np.nonzero(arr)
    if rr.size == 0:
        centroid_xy = (0.0, 0.0)
    else:
        centroid_xy = grid_to_world_xy(int(round(float(np.mean(rr)))), int(round(float(np.mean(cc)))), map_info)
    area_m2 = float(np.count_nonzero(arr)) * float(map_info.resolution_m) ** 2
    boundary = _mask_boundary(arr)
    unknown_arr = np.asarray(unknown, dtype=bool)
    boundary_unknown = float(np.count_nonzero(boundary & unknown_arr)) / float(max(1, np.count_nonzero(boundary)))
    room_edges = []
    for edge in doorway_edges:
        if int(edge.get("room_a_label", -1)) == int(label_id) or int(edge.get("room_b_label", -1)) == int(label_id):
            room_edges.append(dict(edge))
    confidence = float(max(0.0, min(1.0, 1.0 - 0.5 * boundary_unknown)))
    return RoomMask(
        room_id=room_id,
        mask=arr,
        centroid_xy=(float(centroid_xy[0]), float(centroid_xy[1])),
        area_m2=area_m2,
        boundary_unknown_fraction=boundary_unknown,
        doorway_edges=room_edges,
        confidence=confidence,
        source=UPSTREAM_ALGORITHM,
        observed_free_cells=int(np.count_nonzero(arr)),
        mask_confidence=confidence,
        is_partial=bool(boundary_unknown > 0.25),
        step=int(step),
        metadata={"label_id": int(label_id), "centroid_grid": [int(round(float(np.mean(rr)))) if rr.size else 0, int(round(float(np.mean(cc)))) if cc.size else 0]},
    )


def _room_polygon_from_mask(mask: np.ndarray, map_info: MapInfo) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if not np.any(arr):
        return np.zeros((0, 2), dtype=np.float32)
    rr, cc = np.nonzero(arr)
    r0, r1 = int(np.min(rr)), int(np.max(rr))
    c0, c1 = int(np.min(cc)), int(np.max(cc))
    corners = [grid_to_world_xy(r0, c0, map_info), grid_to_world_xy(r0, c1, map_info), grid_to_world_xy(r1, c1, map_info), grid_to_world_xy(r1, c0, map_info)]
    return np.asarray(corners, dtype=np.float32)


def _proposal_masks_debug(labels: np.ndarray) -> list[dict]:
    arr = np.asarray(labels, dtype=np.int32)
    out: list[dict] = []
    for label_id in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        mask = arr == label_id
        out.append({"label_id": int(label_id), "mask": mask.astype(np.uint8).tolist(), "cell_count": int(np.count_nonzero(mask))})
    return out


def _absorb_unlabeled_structural_free_into_rooms(
    labels: np.ndarray,
    *,
    structural_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Assign ROSE-free cells left at label 0 to adjacent final rooms.

    ROSE uses rasterized structural boundaries to split faces. Those boundary
    pixels can still be valid free space in the dedicated roomseg input, but
    `faces_from_boundary_map` must temporarily exclude them so the split works.
    This pass is deliberately after merge/finalization: it fills visual and
    semantic room masks without allowing those pixels to merge rooms.
    """

    out = np.asarray(labels, dtype=np.int32).copy()
    structural_free = np.asarray(structural_free_mask, dtype=bool)
    unknown = np.asarray(unknown_mask, dtype=bool)
    if out.shape != structural_free.shape or out.shape != unknown.shape:
        raise ValueError("room free absorption inputs must have the same HxW shape")
    candidates = structural_free & ~unknown & (out <= 0)
    before = int(np.count_nonzero(candidates))
    debug = {
        "enabled": True,
        "source_mask": "structural_free_mask",
        "policy": "adjacent_room_flood_fill_then_nearest_label",
        "before_unlabeled_free_cells": before,
        "absorbed_free_cells": 0,
        "after_unlabeled_free_cells": before,
        "nearest_fallback_cells": 0,
    }
    if before == 0 or not np.any(out > 0):
        return out, debug

    from collections import deque

    queue: deque[tuple[int, int]] = deque()
    for r, c in zip(*np.nonzero(candidates)):
        neighbor_labels = _neighbor_room_labels(out, int(r), int(c))
        if not neighbor_labels:
            continue
        out[int(r), int(c)] = _majority_label(neighbor_labels)
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        label_id = int(out[r, c])
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and candidates[nr, nc] and out[nr, nc] <= 0:
                out[nr, nc] = label_id
                queue.append((nr, nc))

    remaining = structural_free & ~unknown & (out <= 0)
    nearest_fallback = 0
    if np.any(remaining):
        labeled_coords = np.argwhere(out > 0)
        for r, c in zip(*np.nonzero(remaining)):
            rr = int(r)
            cc = int(c)
            nearest = _nearest_room_label_from_coords(out, labeled_coords, rr, cc)
            if nearest > 0:
                out[rr, cc] = nearest
                nearest_fallback += 1

    after = int(np.count_nonzero(structural_free & ~unknown & (out <= 0)))
    debug["absorbed_free_cells"] = int(before - after)
    debug["after_unlabeled_free_cells"] = after
    debug["nearest_fallback_cells"] = int(nearest_fallback)
    return out, debug


def _neighbor_room_labels(labels: np.ndarray, row: int, col: int) -> list[int]:
    out: list[int] = []
    for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if 0 <= nr < labels.shape[0] and 0 <= nc < labels.shape[1]:
            label_id = int(labels[nr, nc])
            if label_id > 0:
                out.append(label_id)
    return out


def _majority_label(labels: Sequence[int]) -> int:
    counts: dict[int, int] = {}
    for label_id in labels:
        counts[int(label_id)] = counts.get(int(label_id), 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else 0


def _nearest_room_label_from_coords(labels: np.ndarray, labeled_coords: np.ndarray, row: int, col: int) -> int:
    if labeled_coords.size == 0:
        return 0
    target = np.asarray([[int(row), int(col)]], dtype=np.int32)
    d2 = np.sum((labeled_coords.astype(np.int32) - target) ** 2, axis=1)
    nearest = labeled_coords[int(np.argmin(d2))]
    return int(labels[int(nearest[0]), int(nearest[1])])


def room_mask_to_debug(room: RoomMask) -> dict:
    return {
        "room_id": str(room.room_id),
        "area_m2": float(room.area_m2),
        "centroid_xy": [float(room.centroid_xy[0]), float(room.centroid_xy[1])],
        "source": UPSTREAM_ALGORITHM,
        "boundary_confidence": float(room.mask_confidence),
        "polygon_xy": _room_polygon_from_mask(room.mask, _map_info_from_room(room)).astype(float).tolist(),
        "proposal_labels": list((room.metadata or {}).get("proposal_labels", [])),
    }


def _map_info_from_room(room: RoomMask) -> MapInfo:
    info = (room.metadata or {}).get("_map_info")
    if isinstance(info, MapInfo):
        return info
    h, w = np.asarray(room.mask).shape
    return MapInfo(resolution_m=1.0, min_x=0.0, max_x=float(w), min_y=0.0, max_y=float(h), width=int(w), height=int(h))


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    neighbors = padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]
    return arr & ~neighbors


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape:
        return 0.0
    inter = float(np.count_nonzero(aa & bb))
    union = float(np.count_nonzero(aa | bb))
    return 0.0 if union <= 0 else inter / union


def _label_count(labels: np.ndarray) -> int:
    return int(len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0]))


FURNITURE_LIKE_CATEGORIES = {
    "cabinet",
    "kitchen_cabinet",
    "bathroom_cabinet",
    "shelf",
    "curtain",
    "wardrobe",
    "refrigerator",
    "fridge",
    "sofa",
    "bed",
    "table",
    "dining_table",
    "coffee_table",
    "desk",
    "chair",
    "kitchen_rack",
    "rack",
    "bookshelf",
    "dresser",
    "counter",
    "tv_stand",
    "stand",
}

WINDOW_LIKE_CATEGORIES = {
    "window",
    "curtain",
    "glass",
    "glass_door",
    "blind",
}


def _normalize_roomseg_ray_evidence(
    evidence: Mapping[str, np.ndarray] | None,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    raw = dict(evidence or {})

    def pick(*names: str, dtype=np.uint16, fill=0):
        for name in names:
            if name not in raw:
                continue
            arr = np.asarray(raw[name], dtype=dtype)
            if arr.shape == tuple(shape):
                return arr
        return np.full(tuple(shape), fill, dtype=dtype)

    return {
        "ray_covered_count": pick("ray_covered_count", "roomseg_ray_covered_count", dtype=np.uint16, fill=0),
        "terminal_wall_count": pick("terminal_wall_count", "roomseg_terminal_wall_count", dtype=np.uint16, fill=0),
        "terminal_wall_splat": pick("terminal_wall_splat", "roomseg_terminal_wall_splat", dtype=np.uint8, fill=0),
        "terminal_wall_height_min": pick("terminal_wall_height_min", "roomseg_terminal_wall_height_min", dtype=np.float32, fill=np.inf),
        "terminal_wall_height_max": pick("terminal_wall_height_max", "roomseg_terminal_wall_height_max", dtype=np.float32, fill=-np.inf),
        "terminal_wall_depth_min": pick("terminal_wall_depth_min", "roomseg_terminal_wall_depth_min", dtype=np.float32, fill=np.inf),
    }


def _legacy_fusion_audit(
    *,
    vertical_free: np.ndarray,
    vertical_observed: np.ndarray,
    nav_raw_obstacle: np.ndarray,
    static_structural_occupied: np.ndarray | None,
    navigation_free: np.ndarray,
    config: UpstreamROSE2Config,
    initial_free: np.ndarray,
    initial_occupied: np.ndarray,
    initial_unknown: np.ndarray,
) -> dict[str, object]:
    strict_no_nav_obstacle = bool(config.ray_valid_wall_strict_no_navigation_obstacle_overlay)
    strict_no_nav_free = bool(config.ray_valid_wall_strict_no_navigation_free_overlay)
    audit_config = {
        "enabled": bool(config.roomseg_evidence_fusion_enabled),
        "mode": "ray_valid_wall_no_navigation_overlay",
        "vertical_free_priority": True,
        "use_navigation_obstacle_for_vertical_unknown": False if strict_no_nav_obstacle else bool(config.roomseg_use_navigation_obstacle_for_vertical_unknown),
        "use_navigation_free_for_roomseg_free": False if strict_no_nav_free else bool(config.roomseg_use_navigation_free_for_roomseg_free),
        "use_inflated_obstacle": False,
        "use_depth_valid_as_wall": False,
        "use_static_structural_occupied": False if strict_no_nav_obstacle else bool(config.roomseg_use_static_structural_occupied),
    }
    audit = fuse_vertical_profile_with_navigation_obstacles(
        vertical_free=vertical_free,
        vertical_observed=vertical_observed,
        nav_raw_obstacle=nav_raw_obstacle,
        static_structural_occupied=static_structural_occupied,
        navigation_free=navigation_free,
        inflated_obstacle=None,
        config=audit_config,
    )
    audit["initial_roomseg_free"] = np.asarray(initial_free, dtype=bool)
    audit["initial_roomseg_occupied"] = np.asarray(initial_occupied, dtype=bool)
    audit["initial_roomseg_unknown"] = np.asarray(initial_unknown, dtype=bool)
    audit["initial_roomseg_free_after_fusion"] = np.asarray(initial_free, dtype=bool)
    audit["initial_roomseg_occupied_after_fusion"] = np.asarray(initial_occupied, dtype=bool)
    audit["initial_roomseg_unknown_after_fusion"] = np.asarray(initial_unknown, dtype=bool)
    debug = dict(audit.get("debug", {}) or {})
    debug.update(
        {
            "evidence_fusion_mode": "ray_valid_wall_no_navigation_overlay",
            "use_navigation_obstacle_for_vertical_unknown": bool(audit_config["use_navigation_obstacle_for_vertical_unknown"]),
            "use_navigation_free_for_roomseg_free": bool(audit_config["use_navigation_free_for_roomseg_free"]),
            "use_static_structural_occupied": bool(audit_config["use_static_structural_occupied"]),
            "navigation_obstacle_overlay_for_roomseg_occupied": False,
            "navigation_free_overlay_for_roomseg_free": False,
            "initial_roomseg_free_cells": int(np.count_nonzero(initial_free)),
            "initial_roomseg_occupied_cells": int(np.count_nonzero(initial_occupied)),
            "initial_roomseg_unknown_cells": int(np.count_nonzero(initial_unknown)),
        }
    )
    audit["debug"] = debug
    return audit


def _vertical_profile_structural_maps(
    *,
    occupied: np.ndarray,
    free: np.ndarray,
    unknown: np.ndarray,
    vertical_profile: VerticalProfileMap,
    vertical_profile_provided: bool,
    roomseg_static_structural_occupied: np.ndarray | None,
    roomseg_ray_evidence: Mapping[str, np.ndarray] | None,
    object_memory: Sequence[object],
    map_info: MapInfo,
    config: UpstreamROSE2Config,
    room_config: RoomSegmentationConfig,
) -> dict:
    occ = np.asarray(occupied, dtype=bool)
    free_arr = np.asarray(free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    vp = ensure_vertical_profile(vertical_profile, occ.shape)
    occ_counts = np.asarray(vp.occupied_count, dtype=np.float32)
    observed_counts = np.asarray(vp.observed_count, dtype=np.float32)

    evidence = vp.evidence_fields(
        min_free_rays=int(config.vertical_or_free_min_free_rays),
        min_observed_rays=int(config.vertical_or_free_min_observed_rays),
    )
    vertical_band_indices = _vertical_band_indices_for_height_range(
        vp,
        z_min_m=float(config.vertical_or_free_z_min_m),
        z_max_m=float(config.vertical_or_free_z_max_m),
    )
    if vertical_band_indices:
        vertical_observed_count = np.sum(np.asarray(vp.observed_count[vertical_band_indices], dtype=np.uint32), axis=0)
    else:
        vertical_observed_count = np.zeros_like(occ, dtype=np.uint32)
    vertical_observed = vertical_observed_count >= int(config.vertical_or_free_min_observed_rays)
    vertical_or_free = (
        _vertical_free_mask_for_height_range(
            vp,
            z_min_m=float(config.vertical_or_free_z_min_m),
            z_max_m=float(config.vertical_or_free_z_max_m),
            min_free_rays=int(config.vertical_or_free_min_free_rays),
            min_observed_rays=int(config.vertical_or_free_min_observed_rays),
        )
        if bool(config.vertical_or_free_enabled)
        else np.zeros_like(occ, dtype=bool)
    )
    nav_free_room_domain = free_arr.copy()
    vertical_free_room_domain = vertical_or_free.copy()
    synthetic_nav_free_bootstrap = False
    # Compatibility for tiny synthetic tests without ray evidence. Strict Isaac
    # runs feed this from OnlineMapper.vertical_profile; this branch keeps unit
    # fixtures readable without changing the planner/frontier maps.
    if not bool(vertical_profile_provided) and not np.any(vertical_observed) and (np.any(free_arr) or np.any(occ)):
        vertical_observed = free_arr | occ
        vertical_free_room_domain = free_arr.copy()
        synthetic_nav_free_bootstrap = True

    if vertical_band_indices:
        vertical_occupied_count = np.sum(np.asarray(vp.occupied_count[vertical_band_indices], dtype=np.uint32), axis=0)
        vertical_unknown_count = np.sum(np.asarray(vp.unknown_count[vertical_band_indices], dtype=np.uint32), axis=0)
    else:
        vertical_occupied_count = np.zeros_like(occ, dtype=np.uint32)
        vertical_unknown_count = np.ones_like(occ, dtype=np.uint32)
    vertical_occupied_before_unknown_guard = vertical_occupied_count >= 1
    vertical_partial_unknown = vertical_unknown_count > 0
    vertical_occupied_map = vertical_occupied_before_unknown_guard & ~vertical_partial_unknown
    if synthetic_nav_free_bootstrap:
        vertical_occupied_map |= occ & ~vertical_free_room_domain
    ray_evidence = _normalize_roomseg_ray_evidence(roomseg_ray_evidence, occ.shape)
    ray_wall = build_ray_valid_wall_inference(
        vertical_free=vertical_free_room_domain,
        vertical_occupied=vertical_occupied_map,
        vertical_observed=vertical_observed,
        terminal_wall_count=ray_evidence.get("terminal_wall_count"),
        terminal_wall_splat=ray_evidence.get("terminal_wall_splat"),
        ray_covered_count=ray_evidence.get("ray_covered_count"),
        terminal_wall_height_min=ray_evidence.get("terminal_wall_height_min"),
        terminal_wall_height_max=ray_evidence.get("terminal_wall_height_max"),
        terminal_wall_depth_min=ray_evidence.get("terminal_wall_depth_min"),
        vertical_profile=vp,
        grid_free=free_arr,
        grid_occupied=occ,
        grid_observed=(~unknown_arr) | free_arr | occ | vertical_observed,
        resolution_m=float(room_config.resolution_m),
        config=config,
    )
    initial_roomseg_free = np.asarray(ray_wall["initial_roomseg_free"], dtype=bool)
    initial_roomseg_occupied = np.asarray(ray_wall["initial_roomseg_occupied"], dtype=bool)
    initial_roomseg_unknown = np.asarray(ray_wall["initial_roomseg_unknown"], dtype=bool)
    vertical_observed = np.asarray(ray_wall["vertical_observed_map"], dtype=bool)
    fusion = _legacy_fusion_audit(
        vertical_free=vertical_free_room_domain,
        vertical_observed=np.asarray(ray_wall["vertical_observed_map"], dtype=bool),
        nav_raw_obstacle=occ,
        static_structural_occupied=roomseg_static_structural_occupied,
        navigation_free=free_arr,
        config=config,
        initial_free=initial_roomseg_free,
        initial_occupied=initial_roomseg_occupied,
        initial_unknown=initial_roomseg_unknown,
    )
    fusion_debug = dict(fusion.get("debug", {}) or {})
    ray_wall_debug = dict(ray_wall.get("debug", {}) or {})
    fusion_debug.update(ray_wall_debug)
    fusion_debug["compat_synthetic_nav_free_bootstrap"] = bool(synthetic_nav_free_bootstrap)
    fusion_debug["vertical_partial_unknown_occupied_suppressed_cells"] = int(
        np.count_nonzero(vertical_occupied_before_unknown_guard & vertical_partial_unknown)
    )
    fusion_debug["vertical_occupied_before_unknown_guard_cells"] = int(np.count_nonzero(vertical_occupied_before_unknown_guard))
    fusion_debug["vertical_unknown_band_cells"] = int(np.count_nonzero(vertical_partial_unknown))

    occupied_bands = (occ_counts > 0) & initial_roomseg_occupied[None, :, :]
    continuity = np.count_nonzero(occupied_bands, axis=0).astype(np.float32) / float(max(1, occ_counts.shape[0]))
    persistence = np.minimum(1.0, np.sum(observed_counts, axis=0) / 4.0)
    vertical_free = vertical_or_free.astype(np.float32)
    line_support = _axis_line_support(initial_roomseg_occupied, min_run_cells=max(3, int(round(float(config.hough_min_line_length_m) / max(float(config.resolution_m), 1e-6)))))
    perimeter_support = _perimeter_network_support(initial_roomseg_occupied, margin=max(1, int(config.exterior_margin_cells)))
    object_overlap = np.zeros_like(occ, dtype=bool)
    surrounded_free = _component_surrounded_by_free_mask(initial_roomseg_occupied, free_arr | initial_roomseg_free)
    bulky = _bulky_component_mask(initial_roomseg_occupied, min_area_cells=4)
    low_observation = (np.sum(observed_counts, axis=0) <= 1).astype(np.float32)

    confidence = np.zeros_like(continuity, dtype=np.float32)
    confidence += float(config.continuity_weight) * continuity
    confidence += float(config.line_support_weight) * line_support
    confidence += float(config.perimeter_support_weight) * perimeter_support
    confidence += float(config.persistence_weight) * persistence
    confidence += initial_roomseg_occupied.astype(np.float32) * 0.20
    wall_shape_support = np.clip(np.maximum(line_support, perimeter_support), 0.0, 1.0)
    vertical_free_conflict = vertical_free * (1.0 - wall_shape_support)
    confidence -= float(config.single_view_penalty) * low_observation * initial_roomseg_occupied.astype(np.float32)
    no_vertical_evidence = np.sum(observed_counts, axis=0) <= 0
    legacy_2d_confidence = np.clip(0.70 * line_support + 0.20 * perimeter_support + 0.25 * initial_roomseg_occupied.astype(np.float32), 0.0, 1.0)
    confidence[no_vertical_evidence & initial_roomseg_occupied] = np.maximum(confidence[no_vertical_evidence & initial_roomseg_occupied], legacy_2d_confidence[no_vertical_evidence & initial_roomseg_occupied])
    threshold = float(config.wall_confidence_threshold)
    object_component = np.zeros_like(occ, dtype=bool)
    interior_clutter = np.zeros_like(occ, dtype=bool)
    hard_suppressed = np.zeros_like(occ, dtype=bool)
    confidence[hard_suppressed] = np.minimum(confidence[hard_suppressed], max(0.0, threshold - 0.10))
    confidence[unknown_arr & ~initial_roomseg_occupied] = 0.0
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

    candidate_wall = initial_roomseg_occupied & ~hard_suppressed & (confidence >= threshold)
    component_gate = _structural_wall_component_gate(
        candidate_wall,
        confidence,
        line_support,
        perimeter_support,
        config,
    )
    pre_repair_wall = initial_roomseg_occupied.copy()
    if bool(config.wall_gating_fix_enabled):
        repaired_occupied, wall_gating_fix_debug = _apply_wall_gating_fix(
            initial_occupied=pre_repair_wall,
            candidate_wall=candidate_wall,
            component_gate=component_gate,
            wall_confidence=confidence,
            perimeter_support=perimeter_support,
            line_support=line_support,
            surrounded_free=surrounded_free,
            bulky=bulky,
            config=config,
        )
    else:
        repaired_occupied = pre_repair_wall.copy()
        wall_gating_fix_debug = {
            "enabled": False,
            "initial_occupied_count": int(np.count_nonzero(pre_repair_wall)),
            "candidate_wall_count": int(np.count_nonzero(candidate_wall)),
            "suppressed_occupied_count": 0,
            "repaired_occupied_count": int(np.count_nonzero(repaired_occupied)),
        }
    closed_gap_mask = np.zeros_like(occ, dtype=bool)
    repaired_free = np.asarray(initial_roomseg_free, dtype=bool).copy()
    repaired_unknown = initial_roomseg_unknown & ~(repaired_occupied | repaired_free)
    vertical_carved = repaired_occupied.copy()
    return {
        "vertical_profile_bands": vp.to_debug_dict(),
        "vertical_cell_evidence_summary": _vertical_evidence_summary(
            vp,
            evidence,
            z_min_m=float(config.vertical_or_free_z_min_m),
            z_max_m=float(config.vertical_or_free_z_max_m),
        ),
        "vertical_or_free_z_min_m": float(config.vertical_or_free_z_min_m),
        "vertical_or_free_z_max_m": float(config.vertical_or_free_z_max_m),
        "vertical_free_source": "vertical_profile_0p1_2p5",
        "vertical_free_overrides_occupied": True,
        "vertical_free_overridden_occupied_cells": int(np.count_nonzero(occ & vertical_or_free)),
        "roomseg_input_source": "vertical_profile_plus_ray_valid_terminal_wall",
        "ray_valid_wall_inference": np.asarray(ray_wall["ray_valid_wall_inference"], dtype=bool),
        "ray_valid_wall_inference_enabled": bool(ray_wall_debug.get("ray_valid_wall_inference_enabled", True)),
        "ray_valid_wall_inference_mode": str(ray_wall_debug.get("ray_valid_wall_inference_mode", RAY_VALID_WALL_INFERENCE_MODE)),
        "ray_valid_wall_inference_debug": ray_wall_debug,
        "vertical_occupied_0p1_2p5": np.asarray(ray_wall["vertical_occupied_0p1_2p5"], dtype=bool),
        "vertical_occupied_0p1_2p5_cells": int(np.count_nonzero(ray_wall["vertical_occupied_0p1_2p5"])),
        "vertical_occupied_0p2_2p0": np.asarray(ray_wall["vertical_occupied_0p2_2p0"], dtype=bool),
        "vertical_occupied_0p2_2p0_cells": int(np.count_nonzero(ray_wall["vertical_occupied_0p2_2p0"])),
        "vertical_occupied_before_unknown_guard": vertical_occupied_before_unknown_guard.astype(bool),
        "vertical_occupied_before_unknown_guard_cells": int(np.count_nonzero(vertical_occupied_before_unknown_guard)),
        "vertical_partial_unknown_occupied_suppressed_cells": int(
            np.count_nonzero(vertical_occupied_before_unknown_guard & vertical_partial_unknown)
        ),
        "vertical_unknown_band_map": vertical_partial_unknown.astype(bool),
        "vertical_unknown_band_cells": int(np.count_nonzero(vertical_partial_unknown)),
        "vertical_observed_0p1_2p5": np.asarray(ray_wall["vertical_observed_0p1_2p5"], dtype=bool),
        "vertical_observed_0p1_2p5_cells": int(np.count_nonzero(ray_wall["vertical_observed_0p1_2p5"])),
        "vertical_observed_0p2_2p0": np.asarray(ray_wall["vertical_observed_0p2_2p0"], dtype=bool),
        "vertical_observed_0p2_2p0_cells": int(np.count_nonzero(ray_wall["vertical_observed_0p2_2p0"])),
        "roomseg_ray_covered_count": np.asarray(ray_wall["roomseg_ray_covered_count"], dtype=np.uint16),
        "roomseg_terminal_wall_count": np.asarray(ray_wall["roomseg_terminal_wall_count"], dtype=np.uint16),
        "roomseg_terminal_wall_height_min": np.asarray(ray_wall["roomseg_terminal_wall_height_min"], dtype=np.float32),
        "roomseg_terminal_wall_height_max": np.asarray(ray_wall["roomseg_terminal_wall_height_max"], dtype=np.float32),
        "roomseg_terminal_wall_depth_min": np.asarray(ray_wall["roomseg_terminal_wall_depth_min"], dtype=np.float32),
        "roomseg_terminal_wall_splat": np.asarray(ray_wall["roomseg_terminal_wall_splat"], dtype=bool),
        "roomseg_ray_covered_cells": int(ray_wall_debug.get("roomseg_ray_covered_cells", 0)),
        "terminal_wall_cells": int(ray_wall_debug.get("terminal_wall_cells", 0)),
        "terminal_wall_splat_cells": int(ray_wall_debug.get("terminal_wall_splat_cells", 0)),
        "unknown_before_ray_wall": np.asarray(ray_wall["unknown_before_ray_wall"], dtype=bool),
        "unknown_after_ray_wall": np.asarray(ray_wall["unknown_after_ray_wall"], dtype=bool),
        "unknown_removed_by_ray_wall": np.asarray(ray_wall["unknown_removed_by_ray_wall"], dtype=bool),
        "unknown_before_ray_wall_cells": int(ray_wall_debug.get("unknown_before_cells", 0)),
        "unknown_after_ray_wall_cells": int(ray_wall_debug.get("unknown_after_cells", 0)),
        "unknown_removed_by_ray_wall_cells": int(ray_wall_debug.get("unknown_removed_by_ray_wall_cells", 0)),
        "vertical_free_overridden_by_wall_cells": int(ray_wall_debug.get("vertical_free_overridden_by_wall_cells", 0)),
        "evidence_fusion": fusion_debug,
        "occupied_source_audit": dict(fusion_debug.get("occupied_source_audit", {})),
        "vertical_observed_map": vertical_observed,
        "vertical_observed_cells": int(np.count_nonzero(vertical_observed)),
        "vertical_unknown_before_overlay": np.asarray(fusion["vertical_unknown_before_overlay"], dtype=bool),
        "vertical_unknown_before_overlay_cells": int(fusion_debug.get("vertical_unknown_before_overlay_cells", 0)),
        "nav_raw_obstacle": np.asarray(fusion["nav_raw_obstacle"], dtype=bool),
        "nav_raw_obstacle_cells": int(fusion_debug.get("nav_raw_obstacle_cells", 0)),
        "roomseg_static_structural_occupied": np.asarray(fusion["roomseg_static_structural_occupied"], dtype=bool),
        "roomseg_static_structural_occupied_cells": int(fusion_debug.get("static_structural_occupied_cells", 0)),
        "nav_obstacle_overlay_candidate": np.asarray(fusion["nav_obstacle_overlay_candidate"], dtype=bool),
        "nav_obstacle_overlay_candidate_cells": int(fusion_debug.get("nav_obstacle_overlay_candidate_cells", 0)),
        "nav_obstacle_overlay_accepted": np.asarray(fusion["nav_obstacle_overlay_accepted"], dtype=bool),
        "nav_obstacle_overlay_accepted_cells": int(fusion_debug.get("nav_obstacle_overlay_accepted_cells", 0)),
        "walls_rescued_from_unknown": np.asarray(fusion["walls_rescued_from_unknown"], dtype=bool),
        "walls_rescued_from_unknown_cells": int(fusion_debug.get("walls_rescued_from_unknown_cells", 0)),
        "vertical_free_over_nav_obstacle": np.asarray(fusion["vertical_free_over_nav_obstacle"], dtype=bool),
        "vertical_free_over_nav_obstacle_cells": int(fusion_debug.get("vertical_free_over_nav_obstacle_cells", 0)),
        "nav_obstacle_still_unknown_after_fusion": np.asarray(fusion["nav_obstacle_still_unknown_after_fusion"], dtype=bool),
        "nav_obstacle_still_unknown_after_fusion_cells": int(fusion_debug.get("nav_obstacle_still_unknown_after_fusion_cells", 0)),
        "observed_not_vertical_free": vertical_observed & ~vertical_free_room_domain,
        "observed_not_vertical_free_cells": int(np.count_nonzero(vertical_observed & ~vertical_free_room_domain)),
        "vertical_or_free_map": vertical_or_free,
        "vertical_or_free_cells": int(np.count_nonzero(vertical_or_free)),
        "vertical_free_conflict_map": vertical_free_conflict > 0.0,
        "vertical_free_conflict_cells": int(np.count_nonzero(vertical_free_conflict > 0.0)),
        "navigation_free_room_domain": nav_free_room_domain,
        "navigation_free_room_domain_cells": int(np.count_nonzero(nav_free_room_domain)),
        "vertical_free_room_domain": vertical_free_room_domain,
        "vertical_free_added_to_roomseg_cells": int(np.count_nonzero(vertical_free_room_domain)),
        "navigation_free_added_to_roomseg_cells": 0,
        "navigation_free_added_to_strict_roomseg_cells": 0,
        "navigation_free_not_added_to_roomseg_cells": int(np.count_nonzero(nav_free_room_domain & ~vertical_free_room_domain)),
        "initial_roomseg_free": initial_roomseg_free,
        "initial_roomseg_occupied": initial_roomseg_occupied,
        "initial_roomseg_unknown": initial_roomseg_unknown,
        "initial_roomseg_free_after_ray_wall": np.asarray(ray_wall["initial_roomseg_free_after_ray_wall"], dtype=bool),
        "initial_roomseg_occupied_after_ray_wall": np.asarray(ray_wall["initial_roomseg_occupied_after_ray_wall"], dtype=bool),
        "initial_roomseg_unknown_after_ray_wall": np.asarray(ray_wall["initial_roomseg_unknown_after_ray_wall"], dtype=bool),
        "initial_roomseg_free_after_fusion": np.asarray(fusion["initial_roomseg_free_after_fusion"], dtype=bool),
        "initial_roomseg_occupied_after_fusion": np.asarray(fusion["initial_roomseg_occupied_after_fusion"], dtype=bool),
        "initial_roomseg_unknown_after_fusion": np.asarray(fusion["initial_roomseg_unknown_after_fusion"], dtype=bool),
        "initial_roomseg_free_cells": int(fusion_debug.get("initial_roomseg_free_cells", np.count_nonzero(initial_roomseg_free))),
        "initial_roomseg_occupied_cells": int(fusion_debug.get("initial_roomseg_occupied_cells", np.count_nonzero(initial_roomseg_occupied))),
        "initial_roomseg_unknown_cells": int(fusion_debug.get("initial_roomseg_unknown_cells", np.count_nonzero(initial_roomseg_unknown))),
        "repaired_roomseg_free": repaired_free,
        "repaired_roomseg_occupied": repaired_occupied,
        "repaired_roomseg_unknown": repaired_unknown,
        "structural_free_mask": repaired_free,
        "vertical_carved_map": vertical_carved,
        "wall_confidence_map": confidence,
        "candidate_wall": candidate_wall,
        "component_gate": component_gate,
        "structural_wall_mask": repaired_occupied,
        "pre_repair_structural_wall_mask": pre_repair_wall,
        "furniture_suppression_mask": object_overlap,
        "interior_clutter_suppression_mask": interior_clutter,
        "structural_component_gate_mask": component_gate,
        "structural_component_rejected_mask": candidate_wall & ~component_gate,
        "repaired_window_gaps": [],
        "verified_doorway_gaps": [],
        "closed_nontraversable_gaps": [],
        "closed_gap_mask": closed_gap_mask,
        "doorway_virtual_boundary_mask": np.zeros_like(occ, dtype=bool),
        "wall_confidence_threshold": float(config.wall_confidence_threshold),
        "repaired_roomseg_free_cells": int(np.count_nonzero(repaired_free)),
        "repaired_roomseg_occupied_cells": int(np.count_nonzero(repaired_occupied)),
        "vertical_carved_cells": int(np.count_nonzero(vertical_carved)),
        "structural_wall_cells": int(np.count_nonzero(repaired_occupied)),
        "furniture_suppressed_cells": int(np.count_nonzero(object_overlap & occ)),
        "interior_clutter_suppressed_cells": int(np.count_nonzero(interior_clutter & occ)),
        "structural_component_rejected_cells": int(np.count_nonzero(candidate_wall & ~component_gate)),
        "wall_confidence_stats": _array_stats(confidence[occ]) if np.any(occ) else {},
        "wall_gating_fix": wall_gating_fix_debug,
    }


def _classify_and_repair_wall_gaps(
    *,
    occupied: np.ndarray,
    free: np.ndarray,
    unknown: np.ndarray,
    boundary: np.ndarray,
    vertical_profile: VerticalProfileMap,
    object_memory: Sequence[object],
    wall_confidence_map: np.ndarray,
    config: RoomSegmentationConfig,
    adapter_config: UpstreamROSE2Config,
) -> dict:
    occ = np.asarray(occupied, dtype=bool)
    free_arr = np.asarray(free, dtype=bool)
    unknown_arr = np.asarray(unknown, dtype=bool)
    out = np.asarray(boundary, dtype=bool).copy()
    if occ.shape != free_arr.shape or occ.shape != out.shape:
        return {
            "boundary": out,
            "lines": [],
            "repaired_window_gaps": [],
            "verified_doorway_gaps": [],
            "closed_nontraversable_gaps": [],
            "closed_gap_mask": np.zeros_like(out, dtype=bool),
            "doorway_virtual_boundary_mask": np.zeros_like(out, dtype=bool),
        }
    resolution = max(float(config.resolution_m), 1e-6)
    min_len_cells = max(3, int(round(float(config.min_wall_line_length_m) / resolution)))
    max_gap_cells = max(1, int(round(float(config.doorway_width_max_m) / resolution)))
    max_repair_gap_cells = max(max_gap_cells, int(round(3.00 / resolution)))
    min_gap_cells = max(1, int(round(float(config.doorway_width_min_m) / resolution)))
    min_support = max(0.05, float(config.doorway_wall_support_min_ratio))
    lines: list[dict] = []
    repaired_window_gaps: list[dict] = []
    verified_doorway_gaps: list[dict] = []
    closed_nontraversable_gaps: list[dict] = []
    closed_gap_mask = np.zeros_like(out, dtype=bool)
    doorway_virtual_boundary_mask = np.zeros_like(out, dtype=bool)

    def add_span(axis: str, index: int, start: int, end: int, support_ratio: float, gap_runs: list[dict]) -> None:
        if end - start + 1 < min_len_cells:
            return
        if axis == "vertical":
            p0 = [int(start), int(index)]
            p1 = [int(end), int(index)]
            orientation = 1.5707963267948966
        else:
            p0 = [int(index), int(start)]
            p1 = [int(index), int(end)]
            orientation = 0.0
        for gap in gap_runs:
            if axis == "vertical":
                out[int(gap["start"]) : int(gap["end"]) + 1, index] = True
                if str(gap.get("kind", "")) == "doorway_virtual_boundary":
                    doorway_virtual_boundary_mask[int(gap["start"]) : int(gap["end"]) + 1, index] = True
                else:
                    closed_gap_mask[int(gap["start"]) : int(gap["end"]) + 1, index] = True
            else:
                out[index, int(gap["start"]) : int(gap["end"]) + 1] = True
                if str(gap.get("kind", "")) == "doorway_virtual_boundary":
                    doorway_virtual_boundary_mask[index, int(gap["start"]) : int(gap["end"]) + 1] = True
                else:
                    closed_gap_mask[index, int(gap["start"]) : int(gap["end"]) + 1] = True
        lines.append(
            {
                "p0": p0,
                "p1": p1,
                "orientation_rad": float(orientation),
                "support_ratio": float(support_ratio),
                "length_m": float((end - start + 1) * resolution),
                "closed_gap_count": int(len(gap_runs)),
                "closed_gaps": [[int(gap["start"]), int(gap["end"])] for gap in gap_runs],
                "source": "vertical_profile_window_door_repair",
            }
        )

    def close_axis(axis: str) -> None:
        count = occ.shape[1] if axis == "vertical" else occ.shape[0]
        for idx in range(count):
            vec_occ = occ[:, idx] if axis == "vertical" else occ[idx, :]
            vec_free = free_arr[:, idx] if axis == "vertical" else free_arr[idx, :]
            vec_unknown = unknown_arr[:, idx] if axis == "vertical" else unknown_arr[idx, :]
            occ_idx = np.flatnonzero(vec_occ)
            if occ_idx.size < min_len_cells:
                continue
            start, end = int(occ_idx[0]), int(occ_idx[-1])
            span = max(1, end - start + 1)
            support = float(occ_idx.size) / float(span)
            if support < min_support:
                continue
            gap_runs: list[dict] = []
            pos = start
            while pos <= end:
                if vec_occ[pos]:
                    pos += 1
                    continue
                gap_start = pos
                while pos <= end and not vec_occ[pos]:
                    pos += 1
                gap_end = pos - 1
                gap_len = gap_end - gap_start + 1
                if gap_len > max_repair_gap_cells or gap_len < min_gap_cells:
                    continue
                classification = _classify_gap(
                    axis=axis,
                    index=idx,
                    gap_start=gap_start,
                    gap_end=gap_end,
                    free=free_arr,
                    unknown=unknown_arr,
                    vertical_profile=vertical_profile,
                    object_memory=object_memory,
                    wall_confidence_map=wall_confidence_map,
                    map_info_resolution=float(resolution),
                    doorway_width_min_m=float(config.doorway_width_min_m),
                    doorway_width_max_m=float(config.doorway_width_max_m),
                    adapter_config=adapter_config,
                )
                unknown_ratio = float(np.count_nonzero(vec_unknown[gap_start : gap_end + 1])) / float(max(1, gap_len))
                if classification["kind"] == "doorway" and unknown_ratio <= float(config.doorway_unknown_support_max_ratio):
                    classification["decision"] = "keep_open"
                    classification["gap_type"] = "doorway"
                    classification["gap_id"] = "gap_%s_%04d_%04d_%04d" % (axis[0], int(idx), int(gap_start), int(gap_end))
                    verified_doorway_gaps.append(classification)
                    gap_runs.append({"start": gap_start, "end": gap_end, "kind": "doorway_virtual_boundary"})
                elif classification["kind"] == "window":
                    classification["decision"] = "close_as_wall"
                    classification["gap_type"] = "window_or_nontraversable_gap"
                    classification["gap_id"] = "gap_%s_%04d_%04d_%04d" % (axis[0], int(idx), int(gap_start), int(gap_end))
                    repaired_window_gaps.append(classification)
                    gap_runs.append({"start": gap_start, "end": gap_end, "kind": "window_repair"})
                else:
                    classification["decision"] = "close_as_wall"
                    classification["gap_type"] = "closed_nontraversable_gap"
                    classification["gap_id"] = "gap_%s_%04d_%04d_%04d" % (axis[0], int(idx), int(gap_start), int(gap_end))
                    closed_nontraversable_gaps.append(classification)
                    gap_runs.append({"start": gap_start, "end": gap_end, "kind": "nontraversable_repair"})
            if not gap_runs:
                continue
            add_span(axis, idx, start, end, support, gap_runs)

    close_axis("vertical")
    close_axis("horizontal")
    return {
        "boundary": out,
        "lines": lines,
        "repaired_window_gaps": repaired_window_gaps,
        "verified_doorway_gaps": verified_doorway_gaps,
        "closed_nontraversable_gaps": closed_nontraversable_gaps,
        "closed_gap_mask": closed_gap_mask,
        "doorway_virtual_boundary_mask": doorway_virtual_boundary_mask,
    }


def _classify_gap(
    *,
    axis: str,
    index: int,
    gap_start: int,
    gap_end: int,
    free: np.ndarray,
    unknown: np.ndarray,
    vertical_profile: VerticalProfileMap,
    object_memory: Sequence[object],
    wall_confidence_map: np.ndarray,
    map_info_resolution: float,
    doorway_width_min_m: float,
    doorway_width_max_m: float,
    adapter_config: UpstreamROSE2Config,
) -> dict:
    cells = _gap_cells(axis, index, gap_start, gap_end)
    nav_free = all(_cell_value(free, cell) for cell in cells)
    floor_free_score = _gap_band_free_score(cells, vertical_profile, ("low",))
    robot_body_free_score = _gap_band_free_score(cells, vertical_profile, ("robot_body",))
    high_free_score = _gap_band_free_score(cells, vertical_profile, ("mid", "upper"))
    floor_traversable = bool(nav_free and floor_free_score >= 0.50 and robot_body_free_score >= 0.50)
    side_free = _gap_has_free_on_both_sides(axis, index, gap_start, gap_end, free)
    connects_indoor = _gap_connects_indoor_components(cells, free)
    exterior = _gap_touches_exterior(cells, free.shape, int(adapter_config.exterior_margin_cells))
    window_object = _gap_overlaps_object_category(cells, object_memory, WINDOW_LIKE_CATEGORIES)
    vertical_free = _gap_vertical_free_score(cells, vertical_profile)
    confidence = _gap_wall_confidence_support(axis, index, gap_start, gap_end, wall_confidence_map)
    side_support = _gap_side_wall_support(axis, index, gap_start, gap_end, wall_confidence_map)
    width_m = float((gap_end - gap_start + 1) * map_info_resolution)
    doorway_width_ok = bool(float(doorway_width_min_m) <= width_m <= float(doorway_width_max_m))
    reason = {
        "axis": str(axis),
        "index": int(index),
        "start": int(gap_start),
        "end": int(gap_end),
        "width_m": width_m,
        "doorway_width_ok": bool(doorway_width_ok),
        "floor_traversable": bool(floor_traversable),
        "nav_free": bool(nav_free),
        "floor_free_score": float(floor_free_score),
        "robot_body_free_score": float(robot_body_free_score),
        "high_free": bool(high_free_score >= 0.50),
        "high_free_score": float(high_free_score),
        "side_free": bool(side_free),
        "connects_indoor_components": bool(connects_indoor),
        "exterior_or_perimeter": bool(exterior),
        "window_or_curtain_object_overlap": bool(window_object),
        "vertical_free_score": float(vertical_free),
        "wall_confidence_support": float(confidence),
        "wall_support_left": float(side_support[0]),
        "wall_support_right": float(side_support[1]),
        "object_evidence": _gap_object_evidence(cells, object_memory, WINDOW_LIKE_CATEGORIES),
        "room_graph_edge_created": False,
    }
    has_wall_support = bool(confidence >= 0.25 or min(side_support) >= 0.10)
    if doorway_width_ok and floor_traversable and connects_indoor and side_free and not exterior and has_wall_support:
        reason["kind"] = "doorway"
        reason["repair_action"] = "virtual_room_boundary_with_navigable_portal"
        reason["room_graph_edge_created"] = True
        return reason
    if window_object:
        reason["semantic_hint"] = "window"
        reason["reason"] = "window_or_curtain_object_without_floor_traversability" if not floor_traversable else "window_or_curtain_object"
    elif not floor_traversable and high_free_score >= 0.50:
        reason["reason"] = "high_gap_but_floor_not_traversable"
    elif not connects_indoor:
        reason["reason"] = "gap_does_not_connect_indoor_components"
    elif exterior:
        reason["reason"] = "exterior_or_perimeter_gap"
    elif not has_wall_support:
        reason["reason"] = "insufficient_wall_support_for_doorway"
    else:
        reason["reason"] = "nontraversable_gap"
    reason["kind"] = "window" if (window_object or exterior or vertical_free > 0.25 or not floor_traversable) else "closed_nontraversable"
    reason["repair_action"] = "close_gap_as_wall_for_room_segmentation"
    return reason


def _axis_line_support(mask: np.ndarray, min_run_cells: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(arr.shape, dtype=np.float32)
    for r in range(arr.shape[0]):
        _mark_runs(out[r, :], arr[r, :], int(min_run_cells))
    for c in range(arr.shape[1]):
        _mark_runs(out[:, c], arr[:, c], int(min_run_cells))
    return out


def _mark_runs(out_line: np.ndarray, mask_line: np.ndarray, min_run_cells: int) -> None:
    idx = 0
    while idx < len(mask_line):
        if not bool(mask_line[idx]):
            idx += 1
            continue
        start = idx
        while idx < len(mask_line) and bool(mask_line[idx]):
            idx += 1
        if idx - start >= int(min_run_cells):
            out_line[start:idx] = 1.0


def _perimeter_network_support(mask: np.ndarray, margin: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(arr.shape, dtype=np.float32)
    for comp in connected_components(arr):
        rows = [cell[0] for cell in comp]
        cols = [cell[1] for cell in comp]
        touches = min(rows) <= margin or min(cols) <= margin or max(rows) >= arr.shape[0] - 1 - margin or max(cols) >= arr.shape[1] - 1 - margin
        if touches:
            for r, c in comp:
                out[r, c] = 1.0
    return out


def _component_surrounded_by_free_mask(mask: np.ndarray, free: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    free_arr = np.asarray(free, dtype=bool)
    out = np.zeros(arr.shape, dtype=bool)
    for comp in connected_components(arr):
        boundary = _component_neighbor_cells(comp, arr.shape)
        if not boundary:
            continue
        ratio = float(sum(1 for cell in boundary if _cell_value(free_arr, cell))) / float(len(boundary))
        if ratio >= 0.65:
            for r, c in comp:
                out[r, c] = True
    return out


def _vertical_free_mask_for_height_range(
    vertical_profile: VerticalProfileMap,
    *,
    z_min_m: float,
    z_max_m: float,
    min_free_rays: int,
    min_observed_rays: int,
) -> np.ndarray:
    vp = vertical_profile
    indices = _vertical_band_indices_for_height_range(vp, z_min_m=z_min_m, z_max_m=z_max_m)
    if not indices:
        return np.zeros(vp.shape, dtype=bool)
    return vp.reliable_free_mask(
        min_free_rays=int(min_free_rays),
        min_observed_rays=int(min_observed_rays),
        band_names=tuple(str(vp.band_names[idx]) for idx in indices),
    )


def _vertical_band_indices_for_height_range(
    vertical_profile: VerticalProfileMap,
    *,
    z_min_m: float,
    z_max_m: float,
) -> list[int]:
    vp = vertical_profile
    lo = float(z_min_m)
    hi = float(z_max_m)
    return [
        int(idx)
        for idx, (_name, (band_lo, band_hi)) in enumerate(zip(vp.band_names, vp.band_ranges_m))
        if float(band_hi) > lo and float(band_lo) < hi
    ]


def _object_overlap_component_mask(
    mask: np.ndarray,
    object_overlap: np.ndarray,
    perimeter_support: np.ndarray,
    line_support: np.ndarray,
    free: np.ndarray,
    config: UpstreamROSE2Config,
) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    obj = np.asarray(object_overlap, dtype=bool)
    perim = np.asarray(perimeter_support, dtype=np.float32)
    line = np.asarray(line_support, dtype=np.float32)
    free_arr = np.asarray(free, dtype=bool)
    out = np.zeros(arr.shape, dtype=bool)
    resolution = max(float(config.resolution_m), 1e-6)
    max_area_cells = max(1, int(round(float(config.clutter_component_max_area_m2) / max(resolution * resolution, 1e-9))))
    max_wall_thickness_cells = max(2, int(round(max(0.25, float(config.wall_cluster_distance_m)) / resolution)))
    for comp in connected_components(arr):
        object_hits = sum(1 for r, c in comp if obj[r, c])
        if object_hits <= 0:
            continue
        if any(perim[r, c] >= 0.5 for r, c in comp):
            continue
        rows = [cell[0] for cell in comp]
        cols = [cell[1] for cell in comp]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        thickness = min(height, width)
        aspect = max(height, width) / float(max(1, thickness))
        boundary = _component_neighbor_cells(comp, arr.shape)
        free_ratio = 0.0 if not boundary else float(sum(1 for cell in boundary if _cell_value(free_arr, cell))) / float(len(boundary))
        line_fraction = float(np.mean([line[r, c] for r, c in comp])) if comp else 0.0
        strong_wall_like = bool(
            aspect >= float(config.wall_like_aspect_ratio_min)
            and line_fraction >= 0.75
            and thickness <= max_wall_thickness_cells * 2
        )
        if strong_wall_like:
            continue
        object_fraction = float(object_hits) / float(max(1, len(comp)))
        furniture_like_component = bool(
            object_fraction >= 0.02
            and (
                len(comp) <= max_area_cells
                or aspect < float(config.wall_like_aspect_ratio_min)
                or thickness > max_wall_thickness_cells
                or free_ratio >= 0.50
            )
        )
        if not furniture_like_component:
            continue
        for r, c in comp:
            out[r, c] = True
    return out


def _bulky_component_mask(mask: np.ndarray, min_area_cells: int = 4) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    out = np.zeros(arr.shape, dtype=bool)
    for comp in connected_components(arr):
        if len(comp) < int(min_area_cells):
            continue
        rows = [cell[0] for cell in comp]
        cols = [cell[1] for cell in comp]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        aspect = max(height, width) / float(max(1, min(height, width)))
        if aspect < 3.0:
            for r, c in comp:
                out[r, c] = True
    return out


def _interior_clutter_component_mask(
    mask: np.ndarray,
    free: np.ndarray,
    object_overlap: np.ndarray,
    perimeter_support: np.ndarray,
    line_support: np.ndarray,
    config: UpstreamROSE2Config,
) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    free_arr = np.asarray(free, dtype=bool)
    obj = np.asarray(object_overlap, dtype=bool)
    perim = np.asarray(perimeter_support, dtype=np.float32)
    line = np.asarray(line_support, dtype=np.float32)
    out = np.zeros(arr.shape, dtype=bool)
    resolution = max(float(config.resolution_m), 1e-6)
    max_area_cells = max(1, int(round(float(config.clutter_component_max_area_m2) / max(resolution * resolution, 1e-9))))
    max_wall_thickness_cells = max(2, int(round(max(0.25, float(config.wall_cluster_distance_m)) / resolution)))
    for comp in connected_components(arr):
        if any(perim[r, c] >= 0.5 for r, c in comp):
            continue
        rows = [cell[0] for cell in comp]
        cols = [cell[1] for cell in comp]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        thickness = min(height, width)
        aspect = max(height, width) / float(max(1, thickness))
        area = len(comp)
        boundary = _component_neighbor_cells(comp, arr.shape)
        free_ratio = 0.0 if not boundary else float(sum(1 for cell in boundary if _cell_value(free_arr, cell))) / float(len(boundary))
        line_fraction = float(np.mean([line[r, c] for r, c in comp])) if comp else 0.0
        too_thick_for_wall = thickness > max_wall_thickness_cells
        compact_or_small = aspect < 5.0 or area <= max_area_cells
        surrounded = free_ratio >= 0.50
        suppress = bool(
            (not any(perim[r, c] >= 0.5 for r, c in comp) and aspect < float(config.wall_like_aspect_ratio_min))
            or (not any(perim[r, c] >= 0.5 for r, c in comp) and too_thick_for_wall)
            or (surrounded and too_thick_for_wall)
            or (surrounded and compact_or_small and line_fraction < 0.80)
        )
        if not suppress:
            continue
        for r, c in comp:
            out[r, c] = True
    return out


def _structural_wall_component_gate(
    candidate_wall: np.ndarray,
    confidence: np.ndarray,
    line_support: np.ndarray,
    perimeter_support: np.ndarray,
    config: UpstreamROSE2Config,
) -> np.ndarray:
    cand = np.asarray(candidate_wall, dtype=bool)
    conf = np.asarray(confidence, dtype=np.float32)
    line = np.asarray(line_support, dtype=np.float32)
    perim = np.asarray(perimeter_support, dtype=np.float32)
    out = np.zeros(cand.shape, dtype=bool)
    resolution = max(float(config.resolution_m), 1e-6)
    max_wall_thickness_cells = max(2, int(round(max(0.25, float(config.wall_cluster_distance_m)) / resolution)))
    min_aspect = max(1.0, float(config.wall_like_aspect_ratio_min))
    min_length_cells = max(3, int(round(float(config.hough_min_line_length_m) / resolution)))
    for comp in connected_components(cand):
        rows = [cell[0] for cell in comp]
        cols = [cell[1] for cell in comp]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        length = max(height, width)
        thickness = min(height, width)
        aspect = float(length) / float(max(1, thickness))
        touches_network = any(perim[r, c] >= 0.5 for r, c in comp)
        line_fraction = float(np.mean([line[r, c] for r, c in comp])) if comp else 0.0
        mean_confidence = float(np.mean([conf[r, c] for r, c in comp])) if comp else 0.0
        long_thin_wall = bool(length >= min_length_cells and thickness <= max_wall_thickness_cells and aspect >= min_aspect)
        network_supported_wall = bool(touches_network and line_fraction >= 0.75 and thickness <= max_wall_thickness_cells * 2)
        very_confident_line = bool(line_fraction >= 0.90 and aspect >= min_aspect * 1.5 and mean_confidence >= float(config.wall_confidence_threshold) + 0.10)
        if not (long_thin_wall or network_supported_wall or very_confident_line):
            continue
        for r, c in comp:
            out[r, c] = True
    return out


def _apply_wall_gating_fix(
    *,
    initial_occupied: np.ndarray,
    candidate_wall: np.ndarray,
    component_gate: np.ndarray,
    wall_confidence: np.ndarray,
    perimeter_support: np.ndarray,
    line_support: np.ndarray,
    surrounded_free: np.ndarray,
    bulky: np.ndarray,
    config: UpstreamROSE2Config,
) -> tuple[np.ndarray, dict]:
    initial = np.asarray(initial_occupied, dtype=bool)
    candidate = np.asarray(candidate_wall, dtype=bool)
    gate = np.asarray(component_gate, dtype=bool)
    confidence = np.asarray(wall_confidence, dtype=np.float32)
    perimeter = np.asarray(perimeter_support, dtype=np.float32) > 0.0
    line = np.asarray(line_support, dtype=np.float32) > 0.0
    clutter_like = np.asarray(surrounded_free, dtype=bool) | np.asarray(bulky, dtype=bool)
    threshold = float(config.wall_confidence_threshold)
    suppress = initial & ~candidate & ~gate & (confidence < threshold) & clutter_like
    if bool(config.wall_gating_fix_keep_perimeter_walls):
        suppress &= ~perimeter
    if bool(config.wall_gating_fix_keep_line_supported_walls):
        suppress &= ~line
    if bool(config.wall_gating_fix_keep_high_confidence_walls):
        suppress &= confidence < max(0.0, threshold - 0.05)
    suppress &= _small_component_mask(suppress, min_area_cells=int(config.wall_gating_fix_min_component_area_cells))
    repaired = initial & ~suppress
    return repaired.astype(bool), {
        "enabled": True,
        "mode": str(config.wall_gating_fix_mode),
        "initial_occupied_count": int(np.count_nonzero(initial)),
        "candidate_wall_count": int(np.count_nonzero(candidate)),
        "component_gate_count": int(np.count_nonzero(gate)),
        "suppressed_occupied_count": int(np.count_nonzero(suppress)),
        "repaired_occupied_count": int(np.count_nonzero(repaired)),
    }


def _small_component_mask(mask: np.ndarray, min_area_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    out = np.zeros(src.shape, dtype=bool)
    if not np.any(src):
        return out
    for comp in connected_components(src):
        if len(comp) <= max(1, int(min_area_cells)):
            for r, c in comp:
                out[int(r), int(c)] = True
    return out


def _stable_object_overlap_mask(
    object_memory: Sequence[object],
    shape: tuple[int, int],
    map_info: MapInfo,
    config: UpstreamROSE2Config,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    center_radius_m = min(float(config.furniture_suppression_radius_m), 0.25)
    bbox_padding_m = min(float(config.furniture_suppression_radius_m), 0.10)
    center_radius_cells = max(1, int(round(center_radius_m / max(float(map_info.resolution_m), 1e-6))))
    bbox_padding_cells = max(1, int(round(bbox_padding_m / max(float(map_info.resolution_m), 1e-6))))
    for node in object_memory or []:
        category = _object_category_key(getattr(node, "category", ""))
        if not _is_furniture_like_category(category):
            continue
        center = getattr(node, "center_grid", None)
        if center is None:
            continue
        try:
            row, col = int(center[0]), int(center[1])
        except Exception:
            continue
        for rr in range(row - center_radius_cells, row + center_radius_cells + 1):
            for cc in range(col - center_radius_cells, col + center_radius_cells + 1):
                if 0 <= rr < shape[0] and 0 <= cc < shape[1] and (rr - row) ** 2 + (cc - col) ** 2 <= center_radius_cells**2:
                    out[rr, cc] = True
        bbox_world = getattr(node, "bbox_world", None)
        if bbox_world is not None:
            try:
                bbox = np.asarray(bbox_world, dtype=np.float32)
            except Exception:
                bbox = np.zeros((0, 0), dtype=np.float32)
            if bbox.shape == (2, 3) and np.all(np.isfinite(bbox[:, :2])):
                r0, c0 = world_xy_to_grid(float(np.min(bbox[:, 0])), float(np.min(bbox[:, 1])), map_info)
                r1, c1 = world_xy_to_grid(float(np.max(bbox[:, 0])), float(np.max(bbox[:, 1])), map_info)
                rr0, rr1 = sorted((int(r0), int(r1)))
                cc0, cc1 = sorted((int(c0), int(c1)))
                rr0, rr1 = max(0, rr0 - bbox_padding_cells), min(shape[0] - 1, rr1 + bbox_padding_cells)
                cc0, cc1 = max(0, cc0 - bbox_padding_cells), min(shape[1] - 1, cc1 + bbox_padding_cells)
                out[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return out


def _object_category_key(category: object) -> str:
    text = str(category or "").strip()
    norm = normalize_category(text)
    return str(norm or text).strip().lower().replace(" ", "_")


def _is_furniture_like_category(category: str) -> bool:
    cat = _object_category_key(category)
    if cat in FURNITURE_LIKE_CATEGORIES:
        return True
    tokens = [part for part in cat.replace("-", "_").split("_") if part]
    return any(token in FURNITURE_LIKE_CATEGORIES for token in tokens)


def _component_neighbor_cells(comp: Sequence[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells = set((int(r), int(c)) for r, c in comp)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < shape[0] and 0 <= nc < shape[1] and (nr, nc) not in cells:
                out.add((nr, nc))
    return sorted(out)


def _gap_cells(axis: str, index: int, start: int, end: int) -> list[tuple[int, int]]:
    if axis == "vertical":
        return [(int(pos), int(index)) for pos in range(int(start), int(end) + 1)]
    return [(int(index), int(pos)) for pos in range(int(start), int(end) + 1)]


def _gap_has_free_on_both_sides(axis: str, index: int, start: int, end: int, free: np.ndarray) -> bool:
    arr = np.asarray(free, dtype=bool)
    cells_a: list[tuple[int, int]] = []
    cells_b: list[tuple[int, int]] = []
    for pos in range(int(start), int(end) + 1):
        if axis == "vertical":
            cells_a.append((pos, int(index) - 1))
            cells_b.append((pos, int(index) + 1))
        else:
            cells_a.append((int(index) - 1, pos))
            cells_b.append((int(index) + 1, pos))
    return any(_cell_value(arr, cell) for cell in cells_a) and any(_cell_value(arr, cell) for cell in cells_b)


def _gap_touches_exterior(cells: Sequence[tuple[int, int]], shape: tuple[int, int], margin: int) -> bool:
    m = max(0, int(margin))
    for r, c in cells:
        if r <= m or c <= m or r >= shape[0] - 1 - m or c >= shape[1] - 1 - m:
            return True
    return False


def _gap_overlaps_object_category(cells: Sequence[tuple[int, int]], object_memory: Sequence[object], categories: set[str]) -> bool:
    cell_set = set((int(r), int(c)) for r, c in cells)
    for node in object_memory or []:
        category = _object_category_key(getattr(node, "category", ""))
        if category not in categories:
            continue
        center = getattr(node, "center_grid", None)
        if center is None:
            continue
        try:
            row, col = int(center[0]), int(center[1])
        except Exception:
            continue
        if any((rr, cc) in cell_set for rr in range(row - 2, row + 3) for cc in range(col - 2, col + 3)):
            return True
    return False


def _gap_vertical_free_score(cells: Sequence[tuple[int, int]], vertical_profile: VerticalProfileMap) -> float:
    if not cells:
        return 0.0
    vp = vertical_profile
    scores = []
    for r, c in cells:
        if not (0 <= r < vp.free_ray_count.shape[1] and 0 <= c < vp.free_ray_count.shape[2]):
            continue
        free_count = float(np.sum(vp.free_ray_count[:, r, c], dtype=np.float64))
        observed = float(np.sum(vp.observed_count[:, r, c], dtype=np.float64))
        scores.append(0.0 if observed <= 0 else free_count / observed)
    return 0.0 if not scores else float(np.mean(scores))


def _gap_band_free_score(cells: Sequence[tuple[int, int]], vertical_profile: VerticalProfileMap, band_names: Sequence[str]) -> float:
    if not cells:
        return 0.0
    vp = vertical_profile
    indices = [band_index(name) for name in band_names]
    scores = []
    for r, c in cells:
        if not (0 <= r < vp.free_ray_count.shape[1] and 0 <= c < vp.free_ray_count.shape[2]):
            continue
        free_count = float(np.sum(vp.free_ray_count[indices, r, c], dtype=np.float64))
        observed = float(np.sum(vp.observed_count[indices, r, c], dtype=np.float64))
        scores.append(0.0 if observed <= 0 else min(1.0, free_count / observed))
    return 0.0 if not scores else float(np.mean(scores))


def _gap_wall_confidence_support(axis: str, index: int, start: int, end: int, wall_confidence_map: np.ndarray) -> float:
    conf = np.asarray(wall_confidence_map, dtype=np.float32)
    values = []
    for pos in (int(start) - 1, int(end) + 1):
        for offset in range(-1, 2):
            cell = (pos, int(index) + offset) if axis == "vertical" else (int(index) + offset, pos)
            if 0 <= cell[0] < conf.shape[0] and 0 <= cell[1] < conf.shape[1]:
                values.append(float(conf[cell[0], cell[1]]))
    return 0.0 if not values else float(np.mean(values))


def _gap_side_wall_support(axis: str, index: int, start: int, end: int, wall_confidence_map: np.ndarray) -> tuple[float, float]:
    conf = np.asarray(wall_confidence_map, dtype=np.float32)
    values_a: list[float] = []
    values_b: list[float] = []
    for pos in range(int(start) - 2, int(end) + 3):
        for side, values in ((-1, values_a), (1, values_b)):
            for outward in (1, 2):
                cell = (pos, int(index) + side * outward) if axis == "vertical" else (int(index) + side * outward, pos)
                if 0 <= cell[0] < conf.shape[0] and 0 <= cell[1] < conf.shape[1]:
                    values.append(float(conf[cell[0], cell[1]]))
    return (
        0.0 if not values_a else float(np.percentile(values_a, 75)),
        0.0 if not values_b else float(np.percentile(values_b, 75)),
    )


def _gap_connects_indoor_components(cells: Sequence[tuple[int, int]], free: np.ndarray) -> bool:
    arr = np.asarray(free, dtype=bool).copy()
    if arr.ndim != 2:
        return False
    for cell in cells:
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            arr[r, c] = False
    labels = np.zeros(arr.shape, dtype=np.int32)
    current = 0
    for comp in connected_components(arr):
        current += 1
        for r, c in comp:
            labels[r, c] = current
    neighbor_labels = set()
    for r, c in cells:
        for nr, nc in ((int(r) - 1, int(c)), (int(r) + 1, int(c)), (int(r), int(c) - 1), (int(r), int(c) + 1)):
            if 0 <= nr < labels.shape[0] and 0 <= nc < labels.shape[1] and labels[nr, nc] > 0:
                neighbor_labels.add(int(labels[nr, nc]))
    return bool(len(neighbor_labels) >= 2)


def _gap_object_evidence(cells: Sequence[tuple[int, int]], object_memory: Sequence[object], categories: set[str]) -> list[str]:
    cell_set = set((int(r), int(c)) for r, c in cells)
    out: list[str] = []
    for node in object_memory or []:
        category = _object_category_key(getattr(node, "category", ""))
        if category not in categories:
            continue
        center = getattr(node, "center_grid", None)
        if center is None:
            continue
        try:
            row, col = int(center[0]), int(center[1])
        except Exception:
            continue
        if any((rr, cc) in cell_set for rr in range(row - 2, row + 3) for cc in range(col - 2, col + 3)):
            out.append(category)
    return sorted(set(out))


def _cell_value(arr: np.ndarray, cell: tuple[int, int]) -> bool:
    r, c = int(cell[0]), int(cell[1])
    return bool(0 <= r < arr.shape[0] and 0 <= c < arr.shape[1] and arr[r, c])


def _array_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _vertical_evidence_summary(vp: VerticalProfileMap, fields: Mapping[str, np.ndarray], *, z_min_m: float, z_max_m: float) -> dict:
    return {
        "z_min_m": float(z_min_m),
        "z_max_m": float(z_max_m),
        "free_count_sum": int(np.sum(vp.free_ray_count, dtype=np.uint64)),
        "occupied_count_sum": int(np.sum(vp.occupied_count, dtype=np.uint64)),
        "observed_count_sum": int(np.sum(vp.observed_count, dtype=np.uint64)),
        "unknown_count_sum": int(np.sum(vp.unknown_count, dtype=np.uint64)),
        "has_floor_level_free_cells": int(np.count_nonzero(fields.get("has_floor_level_free", []))),
        "has_robot_body_free_cells": int(np.count_nonzero(fields.get("has_robot_body_free", []))),
        "has_upper_free_cells": int(np.count_nonzero(fields.get("has_upper_free", []))),
        "has_any_reliable_free_0p1_2p0_cells": int(np.count_nonzero(fields.get("has_any_reliable_free_0p1_2p0", []))),
        "has_low_obstacle_or_sill_cells": int(np.count_nonzero(fields.get("has_low_obstacle_or_sill", []))),
        "has_window_like_high_gap_cells": int(np.count_nonzero(fields.get("has_window_like_high_gap", []))),
    }
