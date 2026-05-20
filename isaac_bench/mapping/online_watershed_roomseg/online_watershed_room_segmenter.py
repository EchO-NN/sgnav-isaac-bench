from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.online_roomseg.evidence_maps import FreeCleanConfig, WallCandidateConfig, build_evidence_maps
from isaac_bench.mapping.room_segmentation import RoomMask, RoomProposalState, RoomSegmentationConfig
from isaac_bench.mapping.room_segmentation import _proposal_masks_debug, _room_from_mask
from isaac_bench.mapping.vertical_profile import VerticalProfileMap

from .corridor import WatershedCorridorConfig, compute_watershed_corridors
from .debug import save_online_watershed_debug
from .distance import compute_watershed_distance_fields
from .regions import (
    WatershedRegionConfig,
    build_frontier_room_context,
    postprocess_watershed_regions,
    run_marker_controlled_watershed,
)
from .seeds import WatershedSeedConfig, build_watershed_markers


ONLINE_WATERSHED_ROOMSEG_BACKEND = "online_watershed_roomseg_v1"
ONLINE_WATERSHED_ROOMSEG_CONTEXT = "online_watershed_roomseg_v1_vlm"


@dataclass
class OnlineWatershedRoomSegConfig:
    enabled: bool = True
    backend: str = ONLINE_WATERSHED_ROOMSEG_BACKEND
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    z_min_m: float = 0.20
    z_max_m: float = 2.00
    min_free_rays: int = 1
    min_observed_rays: int = 1
    min_observed_free_cells: int = 20
    min_room_area_m2: float = 0.40
    alpha_narrow: float = 0.35
    debug_dir: str = "debug/online_watershed_roomseg"
    free_clean: FreeCleanConfig = field(default_factory=FreeCleanConfig)
    wall_candidate: WallCandidateConfig = field(default_factory=WallCandidateConfig)
    corridor: WatershedCorridorConfig = field(default_factory=WatershedCorridorConfig)
    seeds: WatershedSeedConfig = field(default_factory=WatershedSeedConfig)
    regions: WatershedRegionConfig = field(default_factory=WatershedRegionConfig)
    frontier_context_radius_m: float = 0.45
    debug: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "OnlineWatershedRoomSegConfig":
        raw_root = dict(data or {})
        raw = dict(raw_root.get("online_watershed_roomseg", raw_root.get("online_watershed_roomseg_v1", {})) or {})
        for key in ("enabled", "backend", "min_observed_free_cells", "min_room_area_m2"):
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
            debug.setdefault("save_json", bool(debug_layers.get("enabled")))
        if "output_dir" in debug_layers:
            raw.setdefault("debug_dir", str(debug_layers.get("output_dir")))
        raw["debug"] = debug
        raw.update({key: value for key, value in overrides.items() if value is not None})
        nested = {
            "free_clean": FreeCleanConfig.from_mapping(raw.get("free_clean")),
            "wall_candidate": WallCandidateConfig.from_mapping(raw.get("wall_candidate")),
            "corridor": WatershedCorridorConfig.from_mapping(raw.get("corridor")),
            "seeds": WatershedSeedConfig.from_mapping(raw.get("seeds")),
            "regions": WatershedRegionConfig.from_mapping(raw.get("regions")),
        }
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key not in nested}
        base.update(nested)
        return cls(**base)

    def room_config(self) -> RoomSegmentationConfig:
        return RoomSegmentationConfig(
            algorithm=ONLINE_WATERSHED_ROOMSEG_BACKEND,
            source_grid="vertical_profile_free_0p2_2p0",
            proposal_mode="marker_controlled_watershed",
            finalization_mode="region_graph_postprocess",
            min_observed_free_cells=int(self.min_observed_free_cells),
            min_room_area_m2=float(self.min_room_area_m2),
            resolution_m=float(self.resolution_m),
            map_info=self.map_info,
        )


@dataclass
class OnlineWatershedRoomSegResult:
    room_label_map: np.ndarray
    region_type_map: np.ndarray
    layers: dict[str, np.ndarray]
    region_infos: list[dict]
    region_graph: dict
    frontier_room_context: dict
    debug: dict


class OnlineWatershedRoomSegmenter:
    context_source = ONLINE_WATERSHED_ROOMSEG_CONTEXT

    def __init__(self, config: OnlineWatershedRoomSegConfig | Mapping[str, object] | None = None, map_info: MapInfo | None = None):
        if isinstance(config, OnlineWatershedRoomSegConfig):
            self.config = config
        else:
            self.config = OnlineWatershedRoomSegConfig.from_mapping(config or {}, map_info=map_info)
        if map_info is not None:
            self.config.map_info = map_info
            self.config.resolution_m = float(map_info.resolution_m)
        self.last_debug: dict = {}
        self.last_result: OnlineWatershedRoomSegResult | None = None

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
        frontier_clusters: Sequence[object] | None = None,
    ) -> tuple[list[RoomMask], RoomProposalState]:
        _ = object_memory, roomseg_static_structural_occupied, roomseg_ray_evidence
        result = run_online_watershed_roomseg(
            occupancy_map=occupancy_map,
            observed_free_mask=observed_free_mask,
            obstacle_mask=obstacle_mask,
            unknown_mask=unknown_mask,
            vertical_profile=vertical_profile,
            frontier_clusters=frontier_clusters,
            config=self.config,
            step=int(step),
        )
        self.last_result = result
        self.last_debug = dict(result.debug)
        rooms = _rooms_from_watershed_labels(result.room_label_map, result.layers["unknown_mask"], self.config.room_config(), int(step), result.debug, result.region_infos)
        state = RoomProposalState(
            proposal_labels=np.asarray(result.room_label_map, dtype=np.int32),
            structural_free_mask=np.asarray(result.layers["free_clean"], dtype=bool),
            structural_obstacle_mask=np.asarray(result.layers["wall_candidate_clean"], dtype=bool),
            unknown_mask=np.asarray(result.layers["unknown_mask"], dtype=bool),
            distance_m=np.asarray(result.layers["dist_free_extent_m"], dtype=np.float32),
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
        region_infos = list(dict(proposal_state.debug).get("watershed_region_infos", []) or [])
        rooms = _rooms_from_watershed_labels(labels, proposal_state.unknown_mask, self.config.room_config(), int(proposal_state.step), proposal_state.debug, region_infos)
        self.last_debug = {
            **dict(proposal_state.debug),
            "room_count": int(len(rooms)),
            "rooms": _proposal_masks_debug(labels),
        }
        return rooms


def run_online_watershed_roomseg(
    *,
    occupancy_map: np.ndarray,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
    vertical_profile: VerticalProfileMap | None,
    config: OnlineWatershedRoomSegConfig,
    step: int = 0,
    frontier_clusters: Sequence[object] | None = None,
) -> OnlineWatershedRoomSegResult:
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
    distances = compute_watershed_distance_fields(
        free_clean=evidence.free_clean,
        wall_candidate_clean=evidence.wall_candidate_clean,
        resolution_m=float(config.resolution_m),
        alpha_narrow=float(config.alpha_narrow),
    )
    corridor = compute_watershed_corridors(
        free_clean=evidence.free_clean,
        dist_free_extent_m=distances.dist_free_extent_m,
        resolution_m=float(config.resolution_m),
        config=config.corridor,
    )
    seeds = build_watershed_markers(
        free_clean=evidence.free_clean,
        dist_struct_m=distances.dist_struct_m,
        corridor_seed_mask=corridor.corridor_seeds,
        frontier_clusters=frontier_clusters,
        resolution_m=float(config.resolution_m),
        config=config.seeds,
    )
    raw_labels = run_marker_controlled_watershed(
        free_clean=evidence.free_clean,
        elevation=distances.elevation,
        marker_labels=seeds.marker_labels,
    )
    regions = postprocess_watershed_regions(
        raw_labels=raw_labels,
        free_clean=evidence.free_clean,
        marker_labels=seeds.marker_labels,
        seed_type_by_label=seeds.seed_type_by_label,
        corridor_core=corridor.corridor_core,
        dist_free_extent_m=distances.dist_free_extent_m,
        resolution_m=float(config.resolution_m),
        config=config.regions,
    )
    frontier_context = build_frontier_room_context(
        frontier_clusters=frontier_clusters,
        final_labels=regions.final_labels,
        region_infos=regions.region_infos,
        free_clean=evidence.free_clean,
        resolution_m=float(config.resolution_m),
        search_radius_m=float(config.frontier_context_radius_m),
    )
    layers = {
        "vertical_free_raw": evidence.vertical_free_raw,
        "vertical_occupied_raw": evidence.vertical_occupied_raw,
        "vertical_observed_raw": evidence.vertical_observed_raw,
        "free_clean": evidence.free_clean,
        "wall_candidate_clean": evidence.wall_candidate_clean,
        "unknown_mask": evidence.unknown_clean,
        "dist_struct_m": distances.dist_struct_m,
        "dist_free_extent_m": distances.dist_free_extent_m,
        "elevation": distances.elevation,
        "corridor_core": corridor.corridor_core,
        "confirmed_room_seeds": seeds.confirmed_room_seeds,
        "frontier_room_seeds": seeds.frontier_room_seeds,
        "corridor_seeds": seeds.corridor_seeds,
        "raw_labels": regions.raw_labels,
        "final_labels": regions.final_labels,
        "room_labels_after_merge": regions.final_labels,
        "final_room_label_map": regions.final_labels,
        "region_type_map": regions.region_type_map,
    }
    region_report = {**regions.report, "region_infos": regions.region_infos}
    debug = {
        "backend": ONLINE_WATERSHED_ROOMSEG_BACKEND,
        "actual_backend": ONLINE_WATERSHED_ROOMSEG_BACKEND,
        "source_backend": ONLINE_WATERSHED_ROOMSEG_BACKEND,
        "roomseg_backend": ONLINE_WATERSHED_ROOMSEG_BACKEND,
        "algorithm": ONLINE_WATERSHED_ROOMSEG_BACKEND,
        "source": ONLINE_WATERSHED_ROOMSEG_BACKEND,
        "context_source": ONLINE_WATERSHED_ROOMSEG_CONTEXT,
        "room_map_mode": ONLINE_WATERSHED_ROOMSEG_CONTEXT,
        "strict_fallback_used": False,
        "silent_fallback_used": False,
        "legacy_style_used": False,
        "navigation_obstacle_written": False,
        "marker_controlled_watershed": True,
        "unknown_is_not_watershed_barrier": True,
        "step": int(step),
        "resolution_m": float(config.resolution_m),
        "final_room_count": int(len([v for v in np.unique(regions.final_labels) if int(v) > 0])),
        "room_count": int(len([v for v in np.unique(regions.final_labels) if int(v) > 0])),
        "watershed_seed_report": seeds.report,
        "watershed_corridor_report": corridor.report,
        "watershed_region_report": region_report,
        "watershed_region_infos": regions.region_infos,
        "watershed_region_graph": regions.graph,
        "watershed_frontier_room_context": frontier_context,
        "proposal_room_masks": _proposal_masks_debug(regions.final_labels),
        "room_labels_after_merge": regions.final_labels,
        "final_room_label_map": regions.final_labels,
        **evidence.debug,
        **layers,
    }
    debug_cfg = dict(config.debug or {})
    if bool(debug_cfg.get("save_layers", False)) or bool(debug_cfg.get("save_json", False)):
        dump = save_online_watershed_debug(
            out_dir=Path(str(config.debug_dir)) / ("online_watershed_roomseg_step_%06d" % int(step)),
            layers=layers,
            seed_report=seeds.report,
            region_report=region_report,
            region_graph=regions.graph,
            frontier_room_context=frontier_context,
        )
        debug["online_watershed_roomseg_debug_paths"] = dict(dump.get("paths", {}))
    return OnlineWatershedRoomSegResult(
        room_label_map=regions.final_labels.astype(np.int32),
        region_type_map=regions.region_type_map.astype(np.int32),
        layers=layers,
        region_infos=regions.region_infos,
        region_graph=regions.graph,
        frontier_room_context=frontier_context,
        debug=debug,
    )


def _rooms_from_watershed_labels(
    labels: np.ndarray,
    unknown: np.ndarray,
    config: RoomSegmentationConfig,
    step: int,
    debug: Mapping[str, object],
    region_infos: Sequence[Mapping[str, object]],
) -> list[RoomMask]:
    info_by_label = {int(info.get("label", 0)): dict(info) for info in region_infos}
    out: list[RoomMask] = []
    min_cells = max(1, int(config.min_observed_free_cells))
    for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        mask = np.asarray(labels == label, dtype=bool)
        if int(np.count_nonzero(mask)) < min_cells and out:
            continue
        room = _room_from_mask("pending", mask, unknown, [], config, int(step))
        room.source = ONLINE_WATERSHED_ROOMSEG_BACKEND
        info = info_by_label.get(int(label), {})
        room.metadata["label_id"] = int(label)
        room.metadata["proposal_labels"] = [int(label)]
        room.metadata["source_finalization_mode"] = "region_graph_postprocess"
        room.metadata["region_type"] = str(info.get("region_type", "unassigned_free"))
        room.metadata["watershed_backend"] = ONLINE_WATERSHED_ROOMSEG_BACKEND
        room.metadata["marker_controlled_watershed"] = bool(debug.get("marker_controlled_watershed", True))
        out.append(room)
    return out
