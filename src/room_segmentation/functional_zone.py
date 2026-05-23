from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np

from .config import FunctionalZoneConfig, OpenSpaceConfig
from .data_types import GridSpec, MaskAssignmentResult, PlaceGraph, StructuralMap
from .utils import world_to_grid


KITCHEN_OBJECTS = {"sink", "refrigerator", "microwave", "oven", "stove", "dishwasher", "kettle", "toaster", "counter", "cabinet"}
LIVING_OBJECTS = {"sofa", "couch", "tv", "television", "coffee_table", "armchair", "bookshelf"}
DINING_OBJECTS = {"dining_table", "chair", "table"}
SLEEPING_OBJECTS = {"bed", "nightstand", "wardrobe", "dresser"}
WORKING_OBJECTS = {"desk", "office_chair", "monitor", "keyboard", "laptop"}
BATHROOM_OBJECTS = {"toilet", "sink", "bathtub", "shower"}
STORAGE_OBJECTS = {"shelf", "closet", "wardrobe", "cabinet"}

ZONE_OBJECTS = {
    "kitchen_zone": KITCHEN_OBJECTS,
    "living_zone": LIVING_OBJECTS,
    "dining_zone": DINING_OBJECTS,
    "sleeping_zone": SLEEPING_OBJECTS,
    "working_zone": WORKING_OBJECTS,
    "bathroom_zone": BATHROOM_OBJECTS,
    "storage_zone": STORAGE_OBJECTS,
}


class FunctionalZoneAssigner:
    def __init__(
        self,
        config: FunctionalZoneConfig | Mapping[str, object] | None = None,
        open_space_config: OpenSpaceConfig | Mapping[str, object] | None = None,
        grid_spec: GridSpec | None = None,
    ):
        self.config = config if isinstance(config, FunctionalZoneConfig) else FunctionalZoneConfig.from_mapping(config or {})
        self.open_space_config = (
            open_space_config if isinstance(open_space_config, OpenSpaceConfig) else OpenSpaceConfig.from_mapping(open_space_config or {})
        )
        self.grid_spec = grid_spec
        self.zone_to_id = {label: idx + 1 for idx, label in enumerate(self.config.labels)}

    def assign(
        self,
        mask_result: MaskAssignmentResult,
        semantic_objects: list[dict] | None,
        structural_map: StructuralMap,
        place_graph: PlaceGraph,
    ) -> MaskAssignmentResult:
        labels = np.asarray(mask_result.room_id_map, dtype=np.int32)
        functional_map = np.zeros(labels.shape, dtype=np.int32)
        open_space = self._open_space_mask(mask_result, place_graph)
        objects_by_room = self._objects_by_room(labels, semantic_objects or [])
        for instance in mask_result.room_instances:
            room_label = int(instance.room_id)
            room_mask = labels == room_label
            if not np.any(room_mask):
                continue
            if instance.is_corridor:
                zone = "corridor_zone"
            else:
                zone = self._zone_for_room(instance, objects_by_room.get(room_label, []), bool(np.any(open_space & room_mask)))
            instance.functional_zone_label = zone
            instance.is_open_space = bool(np.any(open_space & room_mask))
            if instance.is_open_space:
                instance.room_type = "open_space"
            functional_map[room_mask] = int(self.zone_to_id.get(zone, self.zone_to_id.get(self.config.default_zone_label, 0)))
            instance.object_ids = [int(obj.get("object_id", idx)) for idx, obj in enumerate(objects_by_room.get(room_label, []))]
        out = MaskAssignmentResult(
            room_id_map=mask_result.room_id_map,
            room_confidence_map=mask_result.room_confidence_map,
            room_soft_masks=mask_result.room_soft_masks,
            corridor_mask=mask_result.corridor_mask,
            open_space_mask=open_space,
            functional_zone_map=functional_map,
            room_instances=mask_result.room_instances,
            room_graph_edges=mask_result.room_graph_edges,
            debug_layers=dict(mask_result.debug_layers),
        )
        out.debug_layers["open_space_mask"] = open_space.astype(np.uint8)
        out.debug_layers["functional_zone_map"] = functional_map.astype(np.int32)
        out.debug_layers["functional_zone_legend"] = np.asarray([self.zone_to_id[label] for label in self.config.labels], dtype=np.int32)
        setattr(out, "_grid_spec", getattr(mask_result, "_grid_spec", self.grid_spec))
        return out

    def _zone_for_room(self, instance, objects: list[dict], is_open_space: bool) -> str:
        if not bool(self.config.enabled):
            return str(self.config.default_zone_label)
        votes: Counter[str] = Counter()
        for obj in objects:
            label = str(obj.get("label", obj.get("category", ""))).strip().lower().replace(" ", "_")
            for zone, names in ZONE_OBJECTS.items():
                if label in names:
                    votes[zone] += float(obj.get("confidence", 1.0))
        if votes:
            return str(votes.most_common(1)[0][0])
        if instance.is_corridor:
            return "corridor_zone"
        if float(instance.area_m2) < 0.8:
            return "storage_zone"
        if is_open_space:
            return str(self.config.default_zone_label)
        return str(self.config.default_zone_label)

    def _open_space_mask(self, mask_result: MaskAssignmentResult, place_graph: PlaceGraph) -> np.ndarray:
        labels = np.asarray(mask_result.room_id_map, dtype=np.int32)
        cut_score = np.asarray(mask_result.debug_layers.get("cut_score_map", np.zeros(labels.shape)), dtype=np.float32)
        out = np.zeros(labels.shape, dtype=bool)
        if not bool(self.open_space_config.enabled):
            return out
        for instance in mask_result.room_instances:
            room_mask = labels == int(instance.room_id)
            if not np.any(room_mask) or instance.is_corridor:
                continue
            area_ok = float(instance.area_m2) >= float(self.open_space_config.min_open_area_m2)
            soft_mean = float(np.mean(cut_score[room_mask])) if np.any(room_mask) else 1.0
            center_count = sum(1 for node in place_graph.nodes if node.node_type == "room_center" and room_mask[node.uv])
            if area_ok and soft_mean < float(self.open_space_config.bottleneck_absence_threshold) and center_count >= 1:
                out |= room_mask
        return out

    def _objects_by_room(self, labels: np.ndarray, semantic_objects: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        if self.grid_spec is None:
            return out
        for obj in semantic_objects:
            if "uv" in obj:
                row, col = int(obj["uv"][0]), int(obj["uv"][1])
            elif "xy" in obj:
                row, col = world_to_grid(float(obj["xy"][0]), float(obj["xy"][1]), self.grid_spec)
            else:
                continue
            if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
                room_id = int(labels[row, col])
                if room_id > 0:
                    out.setdefault(room_id, []).append(dict(obj))
        return out
