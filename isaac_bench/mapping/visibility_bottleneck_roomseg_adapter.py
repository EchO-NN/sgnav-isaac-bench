from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterable, Mapping

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.mapping.room_segmentation import RoomMask
from src.room_segmentation import GridSpec, OVBConfig
from src.room_segmentation.room_segmenter import OnlineRoomSegmenter as OVBOnlineRoomSegmenter


ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND = "online_visibility_bottleneck_roomseg_v1"
ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_CONTEXT = "online_visibility_bottleneck_roomseg_v1_vlm"


@dataclass
class OVBRoomSegConfig:
    enabled: bool = True
    backend: str = ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND
    resolution_m: float = 0.05
    map_info: MapInfo | None = None
    min_observed_free_cells: int = 20
    min_room_area_m2: float = 0.35
    ovb: OVBConfig = field(default_factory=OVBConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None = None, **overrides: object) -> "OVBRoomSegConfig":
        raw_root = dict(data or {})
        raw = dict(raw_root.get("online_visibility_bottleneck", {}) or raw_root.get("ovb_room_segmentation", {}) or {})
        for key in ("enabled", "backend"):
            if key in raw_root and key not in raw:
                raw[key] = raw_root[key]
        raw.update({key: value for key, value in overrides.items() if value is not None})
        ovb_raw = dict(raw)
        ovb_raw.setdefault("map", {})
        if isinstance(ovb_raw["map"], Mapping):
            ovb_raw["map"] = {**dict(ovb_raw["map"]), "resolution_m": float(raw.get("resolution_m", overrides.get("resolution_m", 0.05)))}
        cfg = OVBConfig.from_mapping(ovb_raw)
        fields = {name for name in cls.__dataclass_fields__}
        base = {key: raw[key] for key in raw if key in fields and key != "ovb"}
        base["ovb"] = cfg
        return cls(**base)


class OnlineVisibilityBottleneckRoomSegmenter:
    context_source = ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_CONTEXT

    def __init__(self, config: OVBRoomSegConfig | Mapping[str, object] | None = None, map_info: MapInfo | None = None):
        if isinstance(config, OVBRoomSegConfig):
            self.config = config
        else:
            self.config = OVBRoomSegConfig.from_mapping(config or {}, map_info=map_info)
        if map_info is not None:
            self.config.map_info = map_info
            self.config.resolution_m = float(map_info.resolution_m)
            self.config.ovb.map.resolution_m = float(map_info.resolution_m)
        self.segmenter = OVBOnlineRoomSegmenter(self.config.ovb, self._grid_spec())
        self.last_output = None
        self.last_result = None
        self.last_debug: dict = {}

    def update(
        self,
        occupancy_map: np.ndarray,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        unknown_mask: np.ndarray,
        step: int,
        object_memory: Iterable[object] | None = None,
        vertical_profile=None,
        roomseg_static_structural_occupied: np.ndarray | None = None,
        roomseg_ray_evidence: Mapping[str, np.ndarray] | None = None,
    ) -> list[RoomMask]:
        _ = vertical_profile, roomseg_ray_evidence
        obstacle = np.asarray(obstacle_mask if obstacle_mask is not None else occupancy_map, dtype=bool)
        if roomseg_static_structural_occupied is not None and np.asarray(roomseg_static_structural_occupied).shape == obstacle.shape:
            obstacle = obstacle | np.asarray(roomseg_static_structural_occupied, dtype=bool)
        semantic_objects = _semantic_objects_from_memory(object_memory, self.config.map_info)
        output = self.segmenter.update_from_masks(
            observed_free_mask=np.asarray(observed_free_mask, dtype=bool),
            obstacle_mask=obstacle,
            unknown_mask=np.asarray(unknown_mask, dtype=bool),
            frame_id=int(step),
            semantic_objects=semantic_objects,
            existing_2d_map=np.asarray(occupancy_map) if occupancy_map is not None else None,
        )
        self.last_output = output
        rooms = self._room_masks_from_output(output, int(step))
        legacy_layers = _legacy_roomseg_layers(
            output=output,
            observed_free_mask=np.asarray(observed_free_mask, dtype=bool),
            obstacle_mask=obstacle,
            unknown_mask=np.asarray(unknown_mask, dtype=bool),
        )
        self.last_result = SimpleNamespace(
            layers=legacy_layers,
            room_label_map=np.asarray(output.room_id_map, dtype=np.int32),
        )
        self.last_debug = {
            "backend": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
            "actual_backend": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
            "source_backend": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
            "roomseg_backend": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
            "algorithm": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
            "source": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
            "context_source": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_CONTEXT,
            "room_map_mode": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_CONTEXT,
            "strict_fallback_used": False,
            "silent_fallback_used": False,
            "legacy_style_used": False,
            "step": int(step),
            "room_count": int(len(rooms)),
            "num_rooms": int(len(rooms)),
            "num_final_rooms": int(len(rooms)),
            "final_room_count": int(len(rooms)),
            "cut_candidate_count": int(len(output.cut_candidates)),
            "soft_separator_count": int(sum(1 for c in output.cut_candidates if c.is_soft_separator)),
            "hard_separator_count": int(sum(1 for c in output.cut_candidates if c.is_hard_separator)),
            "accepted_separators": [c.to_dict() for c in output.cut_candidates if c.is_soft_separator],
            "debug_layer_keys": sorted(str(k) for k in output.debug_layers.keys()),
            **legacy_layers,
            "rooms": [
                {
                    "room_id": room.room_id,
                    "area_m2": float(room.area_m2),
                    "confidence": float(room.confidence),
                    "metadata": dict(room.metadata),
                }
                for room in rooms
            ],
        }
        return rooms

    def _room_masks_from_output(self, output, step: int) -> list[RoomMask]:
        labels = np.asarray(output.room_id_map, dtype=np.int32)
        unknown = labels < 0
        rooms: list[RoomMask] = []
        min_cells = max(1, int(self.config.min_observed_free_cells))
        for instance in output.room_instances:
            room_id_int = int(instance.room_id)
            mask = labels == room_id_int
            if int(np.count_nonzero(mask)) < min_cells and rooms:
                continue
            centroid = self._centroid_xy(mask)
            room = RoomMask(
                room_id="room_%04d" % room_id_int,
                mask=mask.astype(bool),
                centroid_xy=centroid,
                area_m2=float(np.count_nonzero(mask)) * float(self.config.resolution_m) ** 2,
                boundary_unknown_fraction=_boundary_unknown_fraction(mask, unknown),
                doorway_edges=[],
                confidence=float(instance.confidence),
                source=ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
                observed_free_cells=int(np.count_nonzero(mask)),
                mask_confidence=float(np.mean(output.room_confidence_map[mask])) if np.any(mask) else 0.0,
                is_partial=bool(np.any(mask & output.debug_layers.get("frontier_mask", np.zeros_like(mask, dtype=bool)))),
                step=int(step),
                stale=False,
                metadata={
                    "label_id": room_id_int,
                    "room_type": str(instance.room_type),
                    "functional_zone_label": str(instance.functional_zone_label),
                    "is_corridor": bool(instance.is_corridor),
                    "is_open_space": bool(instance.is_open_space),
                    "connected_room_ids": ["room_%04d" % int(v) for v in instance.connected_room_ids],
                    "object_ids": list(instance.object_ids),
                    "source_finalization_mode": ONLINE_VISIBILITY_BOTTLENECK_ROOMSEG_BACKEND,
                },
            )
            rooms.append(room)
        return rooms

    def _centroid_xy(self, mask: np.ndarray) -> tuple[float, float]:
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return (0.0, 0.0)
        row = int(round(float(np.mean(rows))))
        col = int(round(float(np.mean(cols))))
        if self.config.map_info is not None:
            return tuple(float(v) for v in grid_to_world_xy(row, col, self.config.map_info))
        return (float(col) * float(self.config.resolution_m), float(row) * float(self.config.resolution_m))

    def _grid_spec(self) -> GridSpec:
        info = self.config.map_info
        if info is not None:
            return GridSpec(
                resolution_m=float(info.resolution_m),
                origin_xy=(float(info.min_x), float(info.min_y)),
                width=int(info.width),
                height=int(info.height),
            )
        return GridSpec(
            resolution_m=float(self.config.resolution_m),
            origin_xy=(0.0, 0.0),
            width=1,
            height=1,
        )


def _semantic_objects_from_memory(object_memory: Iterable[object] | None, map_info: MapInfo | None) -> list[dict]:
    out: list[dict] = []
    if object_memory is None:
        return out
    for idx, obj in enumerate(object_memory):
        label = getattr(obj, "category", getattr(obj, "label", getattr(obj, "name", "")))
        object_id = getattr(obj, "object_id", getattr(obj, "id", idx))
        xy = getattr(obj, "centroid_xy", getattr(obj, "xy", None))
        if xy is None and hasattr(obj, "centroid"):
            centroid = getattr(obj, "centroid")
            if centroid is not None and len(centroid) >= 2:
                xy = [float(centroid[0]), float(centroid[1])]
        item = {"object_id": int(object_id) if str(object_id).isdigit() else idx, "label": str(label), "confidence": float(getattr(obj, "confidence", 1.0))}
        if xy is not None:
            item["xy"] = [float(xy[0]), float(xy[1])]
        out.append(item)
    return out


def _boundary_unknown_fraction(mask: np.ndarray, unknown: np.ndarray) -> float:
    from src.room_segmentation.utils import dilate

    boundary = dilate(mask, 1) & ~mask
    total = int(np.count_nonzero(boundary))
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(boundary & unknown)) / float(total)


def _legacy_roomseg_layers(
    *,
    output,
    observed_free_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Expose OVB layers using the existing roomseg debug dump schema."""
    labels = np.asarray(output.room_id_map, dtype=np.int32)
    shape = labels.shape
    free = _shape_bool(observed_free_mask, shape)
    obstacle = _shape_bool(obstacle_mask, shape)
    unknown = _shape_bool(unknown_mask, shape)
    debug_layers = dict(getattr(output, "debug_layers", {}) or {})
    observed_free = _debug_bool(debug_layers, "observed_free_mask", shape, free)
    hard_wall = _debug_bool(debug_layers, "hard_wall_mask", shape, obstacle)
    p_wall = np.asarray(debug_layers.get("p_wall", np.zeros(shape, dtype=np.float32)), dtype=np.float32)
    if p_wall.shape != shape:
        p_wall = np.zeros(shape, dtype=np.float32)
    candidate_wall = hard_wall | (p_wall >= 0.62)
    p_unknown = np.asarray(debug_layers.get("p_unknown", unknown.astype(np.float32)), dtype=np.float32)
    if p_unknown.shape != shape:
        p_unknown = unknown.astype(np.float32)
    vertical_observed = ~(p_unknown >= 0.65)
    if not np.any(vertical_observed):
        vertical_observed = free | obstacle
    soft_separator = _debug_bool(debug_layers, "soft_separator_map", shape, np.zeros(shape, dtype=bool))
    hard_separator = _debug_bool(debug_layers, "hard_separator_map", shape, np.zeros(shape, dtype=bool))
    return {
        "navigation_free_room_domain": observed_free.astype(bool),
        "vertical_free_room_domain": observed_free.astype(bool),
        "vertical_occupied_0p2_2p0": candidate_wall.astype(bool),
        "vertical_observed_map": vertical_observed.astype(bool),
        "vertical_observed_0p2_2p0": vertical_observed.astype(bool),
        "vertical_unknown_before_overlay": unknown.astype(bool),
        "initial_roomseg_free": observed_free.astype(bool),
        "initial_roomseg_occupied": candidate_wall.astype(bool),
        "initial_roomseg_unknown": unknown.astype(bool),
        "initial_roomseg_free_after_fusion": observed_free.astype(bool),
        "initial_roomseg_occupied_after_fusion": candidate_wall.astype(bool),
        "initial_roomseg_unknown_after_fusion": unknown.astype(bool),
        "repaired_roomseg_free": observed_free.astype(bool),
        "repaired_roomseg_occupied": candidate_wall.astype(bool),
        "repaired_roomseg_unknown": unknown.astype(bool),
        "structural_free_mask": observed_free.astype(bool),
        "boundary_map": hard_wall.astype(bool),
        "wall_boundary_map": candidate_wall.astype(bool),
        "candidate_wall": candidate_wall.astype(bool),
        "accepted_separators": soft_separator.astype(bool),
        "rejected_separators": hard_separator & ~soft_separator,
        "final_room_label_map": labels,
        "room_labels_after_merge": labels,
        "room_proposal_labels_before_merge": np.asarray(debug_layers.get("raw_room_id_map", labels), dtype=np.int32),
        "corridor_mask": _debug_bool(debug_layers, "corridor_mask", shape, np.zeros(shape, dtype=bool)),
        "open_space_mask": _debug_bool(debug_layers, "open_space_mask", shape, np.zeros(shape, dtype=bool)),
        "wall_confidence_map": p_wall.astype(np.float32),
    }


def _debug_bool(layers: Mapping[str, object], key: str, shape: tuple[int, int], default: np.ndarray) -> np.ndarray:
    if key not in layers:
        return _shape_bool(default, shape)
    return _shape_bool(layers[key], shape)


def _shape_bool(value: object, shape: tuple[int, int], default: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.shape == shape:
        return arr
    return np.full(shape, bool(default), dtype=bool)
