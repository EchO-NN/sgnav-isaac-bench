from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .data_types import CutCandidate, FreeSpaceState, GridSpec, MaskAssignmentResult, PartitionResult, PlaceGraph, RoomInstance, StructuralMap
from .utils import adjacency_pairs, boundary_cells, centroid_xy, dijkstra_cost, dilate, in_bounds, label_components, radius_cells, relabel_compact


class MaskAssigner:
    def __init__(self, grid_spec: GridSpec, temperature: float = 1.0):
        self.grid_spec = grid_spec
        self.temperature = float(temperature)

    def assign(
        self,
        partition: PartitionResult,
        place_graph: PlaceGraph,
        cut_candidates: list[CutCandidate],
        structural_map: StructuralMap,
        free_space: FreeSpaceState,
    ) -> MaskAssignmentResult:
        free = np.asarray(free_space.observed_free_mask, dtype=bool)
        unknown = np.asarray(structural_map.p_unknown, dtype=np.float32) >= 0.65
        if not np.any(free):
            empty = np.full(free.shape, -1, dtype=np.int32)
            return MaskAssignmentResult(
                room_id_map=empty,
                room_confidence_map=np.zeros(free.shape, dtype=np.float32),
                room_soft_masks={},
                corridor_mask=np.zeros(free.shape, dtype=bool),
                open_space_mask=np.zeros(free.shape, dtype=bool),
                functional_zone_map=np.zeros(free.shape, dtype=np.int32),
                room_instances=[],
                room_graph_edges=[],
                debug_layers={},
            )
        cut_score = np.zeros(free.shape, dtype=np.float32)
        hard_separator = np.zeros(free.shape, dtype=bool)
        for candidate in cut_candidates:
            if not bool(candidate.is_soft_separator):
                continue
            for cell in candidate.cut_cells:
                if in_bounds(cell[0], cell[1], free.shape):
                    cut_score[cell] = max(float(cut_score[cell]), float(candidate.final_score))
                    hard_separator[cell] = hard_separator[cell] or bool(candidate.is_hard_separator)
        split_mask = cut_score >= 0.55
        component_labels, component_count = label_components(free & ~split_mask, connectivity=8)
        component_labels = relabel_compact(component_labels)
        partition_seeds: dict[int, list[tuple[int, int]]] = {
            int(label): [(int(cell[0]), int(cell[1])) for cell in cells]
            for label, cells in partition.seed_cells_by_label.items()
            if int(label) > 0 and cells
        }
        component_seeds: dict[int, list[tuple[int, int]]] = {}
        if component_count > 0:
            min_component_cells = max(1, int(round(0.35 / max(float(self.grid_spec.resolution_m) ** 2, 1e-9))))
            next_label = 1
            for label in sorted(int(v) for v in np.unique(component_labels) if int(v) > 0):
                comp = component_labels == label
                if int(np.count_nonzero(comp)) < min_component_cells and len(component_seeds) > 0:
                    continue
                rows, cols = np.nonzero(comp)
                component_seeds[next_label] = [(int(rows[len(rows) // 2]), int(cols[len(cols) // 2]))]
                next_label += 1
        component_positive_count = int(len(component_seeds))
        partition_reasonable = bool(
            np.any(np.asarray(partition.separator_map, dtype=bool))
            and
            partition_seeds
            and (
                component_positive_count <= 1
                or len(partition_seeds) <= component_positive_count + 1
            )
        )
        if partition_reasonable:
            seeds = partition_seeds
            seed_source = "graph_partition"
        else:
            seeds = component_seeds
            seed_source = "connected_components_fallback"

        unknown_penalty = dilate(unknown, radius_cells(0.20, self.grid_spec.resolution_m)).astype(np.float32)
        wall_boundary = dilate(np.asarray(structural_map.p_wall, dtype=np.float32) >= 0.62, 1).astype(np.float32)
        frontier_penalty = np.asarray(free_space.frontier_mask, dtype=bool).astype(np.float32)
        transition_cost = (
            1.0
            + 2.0 * cut_score
            + 1.2 * unknown_penalty
            + 1.0 * wall_boundary
            + 0.5 * frontier_penalty
        ).astype(np.float32)
        costs: dict[int, np.ndarray] = {}
        for label, label_seeds in seeds.items():
            costs[int(label)] = dijkstra_cost(free, label_seeds, transition_cost)
        if not costs:
            rows, cols = np.nonzero(free)
            seed = (int(rows[len(rows) // 2]), int(cols[len(cols) // 2]))
            costs[1] = dijkstra_cost(free, [seed], transition_cost)

        label_order = sorted(costs)
        cost_stack = np.stack([costs[label] for label in label_order], axis=0)
        finite = np.isfinite(cost_stack)
        best_idx = np.argmin(cost_stack, axis=0)
        best_cost = np.take_along_axis(cost_stack, best_idx[None, :, :], axis=0)[0]
        masked = cost_stack.copy()
        np.put_along_axis(masked, best_idx[None, :, :], np.inf, axis=0)
        second_cost = np.min(masked, axis=0) if len(label_order) > 1 else best_cost + 1.0
        labels_arr = np.asarray(label_order, dtype=np.int32)
        room_id_map = np.zeros(free.shape, dtype=np.int32)
        room_id_map[free] = labels_arr[best_idx[free]]
        room_id_map[unknown | ~free] = -1
        confidence = np.zeros(free.shape, dtype=np.float32)
        best_finite = np.isfinite(best_cost)
        second_safe = np.where(np.isfinite(second_cost), second_cost, best_cost + 1.0)
        conf_raw = np.zeros_like(confidence, dtype=np.float32)
        valid_conf = free & best_finite
        conf_raw[valid_conf] = np.clip(
            (second_safe[valid_conf] - best_cost[valid_conf]) / (second_safe[valid_conf] + 1e-6),
            0.0,
            1.0,
        )
        confidence[free] = conf_raw[free]
        confidence[free & (cut_score >= 0.55)] *= 0.75
        confidence[free & np.asarray(free_space.frontier_mask, dtype=bool)] *= 0.65
        if len(label_order) == 1:
            confidence[free] = np.maximum(confidence[free], 0.85)
        temperature = max(1e-6, float(self.temperature))
        stable_cost = np.where(finite, cost_stack, np.inf)
        min_cost = np.min(stable_cost, axis=0, keepdims=True)
        min_cost = np.where(np.isfinite(min_cost), min_cost, 0.0)
        finite_cost = np.where(np.isfinite(stable_cost), stable_cost, min_cost + 80.0)
        logits = np.exp(-np.clip(finite_cost - min_cost, 0.0, 80.0) / temperature)
        logits[:, ~free] = 0.0
        denom = np.sum(logits, axis=0, keepdims=True) + 1e-6
        probs = logits / denom
        soft_masks = {int(label): probs[idx].astype(np.float32) for idx, label in enumerate(label_order)}
        instances = _instances_from_map(room_id_map, confidence, self.grid_spec)
        edges = _room_edges(room_id_map, cut_score)
        result = MaskAssignmentResult(
            room_id_map=room_id_map.astype(np.int32),
            room_confidence_map=confidence.astype(np.float32),
            room_soft_masks=soft_masks,
            corridor_mask=np.zeros(free.shape, dtype=bool),
            open_space_mask=np.zeros(free.shape, dtype=bool),
            functional_zone_map=np.zeros(free.shape, dtype=np.int32),
            room_instances=instances,
            room_graph_edges=edges,
            debug_layers={
                "cut_score_map": cut_score.astype(np.float32),
                "hard_separator_map": hard_separator.astype(np.uint8),
                "soft_separator_map": (cut_score >= 0.55).astype(np.uint8),
                "raw_room_id_map": room_id_map.astype(np.int32),
                "seed_source": np.asarray([0 if seed_source == "graph_partition" else 1], dtype=np.int32),
            },
        )
        setattr(result, "_grid_spec", self.grid_spec)
        return result


def _instances_from_map(room_id_map: np.ndarray, confidence: np.ndarray, grid_spec: GridSpec) -> list[RoomInstance]:
    out: list[RoomInstance] = []
    for label in sorted(int(v) for v in np.unique(room_id_map) if int(v) > 0):
        mask = room_id_map == label
        area_m2 = float(np.count_nonzero(mask)) * float(grid_spec.resolution_m) ** 2
        out.append(
            RoomInstance(
                room_id=int(label),
                room_type="room",
                functional_zone_label="unknown_functional_zone",
                area_m2=area_m2,
                centroid_xy=centroid_xy(mask, grid_spec),
                confidence=float(np.mean(confidence[mask])) if np.any(mask) else 0.0,
                is_corridor=False,
                is_open_space=False,
                boundary_cells=boundary_cells(mask)[:512],
                connected_room_ids=[],
                object_ids=[],
            )
        )
    edges = _room_edges(room_id_map, np.zeros(room_id_map.shape, dtype=np.float32))
    connected: dict[int, set[int]] = {room.room_id: set() for room in out}
    for a, b, _w in edges:
        connected.setdefault(int(a), set()).add(int(b))
        connected.setdefault(int(b), set()).add(int(a))
    for room in out:
        room.connected_room_ids = sorted(connected.get(room.room_id, set()))
    return out


def _room_edges(room_id_map: np.ndarray, cut_score_map: np.ndarray) -> list[tuple[int, int, float]]:
    pairs = adjacency_pairs(np.where(room_id_map > 0, room_id_map, 0))
    out = []
    for a, b in sorted(pairs):
        mask_a = room_id_map == int(a)
        mask_b = room_id_map == int(b)
        contact = dilate(mask_a, 1) & mask_b
        shared = int(np.count_nonzero(contact))
        if shared <= 0:
            continue
        boundary_score = float(np.mean(cut_score_map[dilate(contact, 1)])) if np.any(dilate(contact, 1)) else 0.0
        weight = float(np.clip(0.35 * min(1.0, shared / 20.0) + 0.35 * boundary_score + 0.30 * (1.0 - boundary_score), 0.0, 1.0))
        out.append((int(a), int(b), weight))
    return out
