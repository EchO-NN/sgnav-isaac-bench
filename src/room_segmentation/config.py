from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _subconfig(cls, raw: Mapping[str, object], key: str):
    value = raw.get(key, {})
    if hasattr(cls, "from_mapping"):
        return cls.from_mapping(value if isinstance(value, Mapping) else {})
    return cls()


def _from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object):
    raw = dict(data or {})
    raw.update({key: value for key, value in overrides.items() if value is not None})
    fields = {name for name in cls.__dataclass_fields__}
    return cls(**{key: raw[key] for key in raw if key in fields})


@dataclass
class MapConfig:
    resolution_m: float = 0.05
    z_resolution_m: float = 0.05
    z_min_m: float = 0.20
    z_max_m: float = 2.00
    unknown_value: int = -1
    free_value: int = 0
    occupied_value: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class DepthConfig:
    depth_min_m: float = 0.20
    depth_max_m: float = 2.00
    ray_step_m: float = 0.025
    endpoint_margin_m: float = 0.03
    max_ray_points_per_frame: int = 120000
    depth_stride_px: int = 2
    use_depth_median_filter: bool = True
    depth_median_kernel: int = 3
    use_depth_bilateral_filter: bool = False
    invalid_depth_value: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class VerticalProjectionConfig:
    min_free_count_for_xy_free: int = 1
    min_occupied_count_for_xy_occupied: int = 3
    occupied_height_ratio_threshold: float = 0.82
    free_height_ratio_threshold: float = 0.08
    unknown_dominance_ratio_threshold: float = 0.75
    endpoint_occupied_confidence: float = 0.85
    through_ray_free_confidence: float = 0.70
    decay_per_update: float = 0.995
    max_log_odds: float = 8.0
    min_log_odds: float = -8.0
    free_log_odds_update: float = -0.65
    occupied_log_odds_update: float = 0.90
    endpoint_log_odds_update: float = 1.20

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class StructuralWallConfig:
    wall_probability_threshold: float = 0.62
    hard_wall_probability_threshold: float = 0.82
    free_probability_threshold: float = 0.55
    unknown_probability_threshold: float = 0.65
    min_wall_line_length_m: float = 0.60
    min_structural_component_area_m2: float = 0.03
    max_island_area_m2: float = 0.20
    close_kernel_m: float = 0.10
    open_kernel_m: float = 0.05
    fill_gap_max_m: float = 0.18
    preserve_openings_min_m: float = 0.35

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        raw = dict(data or {})
        morphology = dict(raw.pop("morphology", {}) or {})
        line_detection = dict(raw.pop("line_detection", {}) or {})
        raw.setdefault("close_kernel_m", morphology.get("close_kernel_m", cls.close_kernel_m))
        raw.setdefault("open_kernel_m", morphology.get("open_kernel_m", cls.open_kernel_m))
        raw.setdefault("fill_gap_max_m", morphology.get("fill_gap_max_m", cls.fill_gap_max_m))
        raw.setdefault("preserve_openings_min_m", morphology.get("preserve_openings_min_m", cls.preserve_openings_min_m))
        raw.setdefault("min_wall_line_length_m", line_detection.get("min_line_length_m", raw.get("min_wall_line_length_m", cls.min_wall_line_length_m)))
        return _from_mapping(cls, raw)


@dataclass
class FreeSpaceConfig:
    robot_radius_m: float = 0.18
    safety_margin_m: float = 0.05
    erosion_radius_m: float = 0.20
    roomseg_erosion_radius_m: float | None = None
    navigation_erosion_radius_m: float | None = None
    min_free_component_area_m2: float = 0.12
    frontier_band_m: float = 0.25
    unknown_as_boundary_for_distance: bool = True
    unknown_as_wall_for_segmentation: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class SkeletonConfig:
    method: str = "medial_axis"
    prune_branch_length_m: float = 0.35
    min_skeleton_component_length_m: float = 0.50
    smooth_width_window_nodes: int = 5
    local_minima_window_m: float = 0.45
    local_minima_prominence_m: float = 0.10
    max_nodes: int = 6000

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class BottleneckConfig:
    max_candidates: int = 120
    candidate_width_min_m: float = 0.35
    candidate_width_preferred_min_m: float = 0.45
    candidate_width_preferred_max_m: float = 1.60
    candidate_width_max_m: float = 2.80
    min_area_each_side_m2: float = 0.25
    min_area_ratio_after_cut: float = 0.04
    max_frontier_ratio_near_cut: float = 0.55
    frontier_penalty_band_m: float = 0.40
    cut_extension_step_m: float = 0.05
    cut_max_length_m: float = 3.00
    cut_connect_wall_distance_m: float = 0.20
    cut_raster_thickness_m: float = 0.07
    endpoint_wall_support_radius_m: float = 0.20
    side_sampling_distance_m: float = 0.35
    side_sampling_count: int = 7

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class VisibilityConfig:
    enabled: bool = True
    ray_count_per_node: int = 72
    ray_angle_span_deg: float = 360.0
    ray_max_range_m: float = 4.00
    ray_step_m: float = 0.05
    visibility_cell_sample_limit: int = 2000
    jaccard_epsilon: float = 1.0e-6
    min_visibility_cells: int = 20
    visibility_cache_size: int = 2000

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class PlaceGraphConfig:
    keyframe_interval_m: float = 0.35
    keyframe_interval_rad: float = 0.35
    skeleton_node_stride_m: float = 0.20
    room_center_min_distance_m: float = 0.70
    edge_max_geodesic_m: float = 1.50
    edge_max_euclidean_m: float = 1.20
    line_of_sight_required_for_short_edges: bool = False
    weight_geodesic: float = 0.30
    weight_visibility_jaccard: float = 0.30
    weight_line_of_sight: float = 0.15
    weight_semantic_similarity: float = 0.10
    weight_bottleneck_penalty: float = 0.45
    weight_frontier_penalty: float = 0.15

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        raw = dict(data or {})
        weights = dict(raw.pop("weights", {}) or {})
        for key, value in weights.items():
            raw.setdefault("weight_" + str(key), value)
        return _from_mapping(cls, raw)


@dataclass
class PartitionConfig:
    method: str = "union_find_with_cut"
    spectral_enabled: bool = True
    normalized_cut_enabled: bool = True
    split_score_threshold: float = 0.72
    soft_split_score_threshold: float = 0.55
    merge_score_threshold: float = 0.42
    min_room_area_m2: float = 0.60
    min_room_observed_area_m2: float = 0.35
    min_room_graph_nodes: int = 3
    oversegmentation_merge_enabled: bool = True
    max_visibility_drop_for_merge: float = 0.35
    min_shared_boundary_m: float = 0.25
    max_boundary_score_for_merge: float = 0.45

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        raw = dict(data or {})
        merge = dict(raw.pop("oversegmentation_merge", {}) or {})
        for key, value in merge.items():
            raw.setdefault("oversegmentation_merge_" + str(key), value)
        return _from_mapping(cls, raw)


@dataclass
class SeparatorScoringConfig:
    weight_width: float = 0.18
    weight_wall_support: float = 0.18
    weight_visibility_drop: float = 0.24
    weight_graph_conductance: float = 0.18
    weight_temporal: float = 0.12
    weight_frontier_penalty: float = 0.15
    weight_alcove_penalty: float = 0.12
    weight_area_balance: float = 0.10
    weight_corridor_consistency: float = 0.12
    sigmoid_temperature: float = 1.00
    score_clip_min: float = 0.0
    score_clip_max: float = 1.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        raw = dict(data or {})
        weights = dict(raw.pop("weights", {}) or {})
        aliases = {"graph": "graph_conductance", "wall": "wall_support", "visibility": "visibility_drop", "area": "area_balance", "corridor": "corridor_consistency"}
        for key, value in weights.items():
            raw.setdefault("weight_" + aliases.get(str(key), str(key)), value)
        return _from_mapping(cls, raw)


@dataclass
class CorridorConfig:
    enabled: bool = True
    width_min_m: float = 0.45
    width_max_m: float = 1.80
    length_to_width_ratio_min: float = 2.80
    width_cv_max: float = 0.28
    skeleton_degree2_ratio_min: float = 0.65
    parallel_wall_support_min: float = 0.45
    max_internal_cut_score: float = 0.62
    side_branch_angle_min_deg: float = 45.0
    side_branch_angle_max_deg: float = 135.0
    merge_accidental_splits: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class OpenSpaceConfig:
    enabled: bool = True
    min_open_area_m2: float = 1.50
    bottleneck_absence_threshold: float = 0.50
    allow_functional_subzones: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class FunctionalZoneConfig:
    enabled: bool = True
    geometry_only_fallback: bool = True
    object_cluster_radius_m: float = 1.20
    min_objects_per_zone: int = 1
    semantic_embedding_weight: float = 0.45
    object_density_weight: float = 0.35
    geometry_region_weight: float = 0.20
    default_zone_label: str = "unknown_functional_zone"
    labels: tuple[str, ...] = (
        "kitchen_zone",
        "living_zone",
        "dining_zone",
        "sleeping_zone",
        "working_zone",
        "bathroom_zone",
        "storage_zone",
        "corridor_zone",
        "unknown_functional_zone",
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        raw = dict(data or {})
        if "labels" in raw and isinstance(raw["labels"], list):
            raw["labels"] = tuple(str(item) for item in raw["labels"])
        return _from_mapping(cls, raw)


@dataclass
class TemporalConfig:
    enabled: bool = True
    split_persistence_observations: int = 3
    merge_persistence_observations: int = 3
    room_id_iou_match_threshold: float = 0.35
    room_id_centroid_distance_threshold_m: float = 1.00
    max_room_id_age_frames: int = 50
    score_ema_alpha: float = 0.35
    mask_ema_alpha: float = 0.40
    confidence_decay_unobserved: float = 0.98
    hard_split_threshold: float = 0.75
    soft_split_threshold: float = 0.55
    merge_threshold: float = 0.45

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class OutputConfig:
    output_soft_masks: bool = True
    output_debug_layers: bool = True
    output_room_graph: bool = True
    output_functional_zones: bool = True
    output_corridor_mask: bool = True
    output_open_space_mask: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class RuntimeConfig:
    max_update_time_ms: int = 80
    full_recompute_interval_frames: int = 10
    incremental_update: bool = True
    max_grid_cells_for_full_skeleton: int = 400000
    downsample_for_skeleton_if_needed: bool = True
    skeleton_downsample_factor: int = 2

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        return _from_mapping(cls, data)


@dataclass
class DebugConfig:
    save_debug_images: bool = False
    debug_image_dir: str = "debug/room_segmentation"
    visualize_skeleton: bool = True
    visualize_bottlenecks: bool = True
    visualize_visibility_graph: bool = True
    visualize_room_masks: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None):
        raw = dict(data or {})
        if "save_layers" in raw:
            raw.setdefault("save_debug_images", bool(raw["save_layers"]))
        if "output_dir" in raw:
            raw.setdefault("debug_image_dir", str(raw["output_dir"]))
        return _from_mapping(cls, raw)


@dataclass
class OVBConfig:
    map: MapConfig = field(default_factory=MapConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    vertical_projection: VerticalProjectionConfig = field(default_factory=VerticalProjectionConfig)
    structural_wall: StructuralWallConfig = field(default_factory=StructuralWallConfig)
    free_space: FreeSpaceConfig = field(default_factory=FreeSpaceConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    place_graph: PlaceGraphConfig = field(default_factory=PlaceGraphConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    separator_scoring: SeparatorScoringConfig = field(default_factory=SeparatorScoringConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    open_space: OpenSpaceConfig = field(default_factory=OpenSpaceConfig)
    functional_zone: FunctionalZoneConfig = field(default_factory=FunctionalZoneConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "OVBConfig":
        raw = dict(data or {})
        if "online_visibility_bottleneck" in raw:
            raw = {**raw, **dict(raw.get("online_visibility_bottleneck", {}) or {})}
        out = cls(
            map=_subconfig(MapConfig, raw, "map"),
            depth=_subconfig(DepthConfig, raw, "depth"),
            vertical_projection=_subconfig(VerticalProjectionConfig, raw, "vertical_projection"),
            structural_wall=_subconfig(StructuralWallConfig, raw, "structural_wall"),
            free_space=_subconfig(FreeSpaceConfig, raw, "free_space"),
            skeleton=_subconfig(SkeletonConfig, raw, "skeleton"),
            bottleneck=_subconfig(BottleneckConfig, raw, "bottleneck"),
            visibility=_subconfig(VisibilityConfig, raw, "visibility"),
            place_graph=_subconfig(PlaceGraphConfig, raw, "place_graph"),
            partition=_subconfig(PartitionConfig, raw, "partition"),
            separator_scoring=_subconfig(SeparatorScoringConfig, raw, "separator_scoring"),
            corridor=_subconfig(CorridorConfig, raw, "corridor"),
            open_space=_subconfig(OpenSpaceConfig, raw, "open_space"),
            functional_zone=_subconfig(FunctionalZoneConfig, raw, "functional_zone"),
            temporal=_subconfig(TemporalConfig, raw, "temporal"),
            output=_subconfig(OutputConfig, raw, "output"),
            runtime=_subconfig(RuntimeConfig, raw, "runtime"),
            debug=_subconfig(DebugConfig, raw, "debug"),
        )
        for key, value in overrides.items():
            if value is None:
                continue
            if "." in key:
                root, leaf = key.split(".", 1)
                sub = getattr(out, root, None)
                if sub is not None and hasattr(sub, leaf):
                    setattr(sub, leaf, value)
            elif hasattr(out, key):
                setattr(out, key, value)
        return out

    def to_dict(self) -> dict:
        def convert(value: Any):
            if is_dataclass(value):
                return {field_name: convert(getattr(value, field_name)) for field_name in value.__dataclass_fields__}
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            return value

        return convert(self)


def load_ovb_config(path: str | Path | None = None, overrides: Mapping[str, object] | None = None) -> OVBConfig:
    raw: dict[str, object] = {}
    if path is not None and Path(path).exists():
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                raw = dict(loaded)
        except Exception:
            raw = {}
    if overrides:
        raw.update(dict(overrides))
    return OVBConfig.from_mapping(raw)
