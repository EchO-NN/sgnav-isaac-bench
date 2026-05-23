from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import CorridorConfig
from .data_types import CutCandidate, MaskAssignmentResult, SkeletonGraph, StructuralMap
from .mask_assignment import _instances_from_map, _room_edges
from .utils import adjacency_pairs, boundary_cells, centroid_xy, dilate, label_components, relabel_compact


class CorridorAnalyzer:
    def __init__(self, config: CorridorConfig | Mapping[str, object] | None = None, resolution_m: float = 0.05):
        self.config = config if isinstance(config, CorridorConfig) else CorridorConfig.from_mapping(config or {})
        self.resolution_m = float(resolution_m)

    def refine(
        self,
        mask_result: MaskAssignmentResult,
        skeleton_graph: SkeletonGraph,
        structural_map: StructuralMap,
        cut_candidates: list[CutCandidate],
    ) -> MaskAssignmentResult:
        if not bool(self.config.enabled):
            return mask_result
        labels = np.asarray(mask_result.room_id_map, dtype=np.int32).copy()
        cut_score = np.asarray(mask_result.debug_layers.get("cut_score_map", np.zeros(labels.shape)), dtype=np.float32)
        corridor_labels = self._corridor_labels(labels, skeleton_graph, structural_map)
        if bool(self.config.merge_accidental_splits):
            labels = self._merge_corridor_neighbors(labels, corridor_labels, cut_score)
            corridor_labels = self._corridor_labels(labels, skeleton_graph, structural_map)
        corridor_mask = _corridor_mask_from_skeleton(labels > 0, skeleton_graph, self.config, self.resolution_m)
        for label in corridor_labels:
            corridor_mask |= labels == int(label)
        grid_spec = getattr(mask_result, "_grid_spec", None)
        instances = mask_result.room_instances
        edges = mask_result.room_graph_edges
        if grid_spec is not None:
            instances = _instances_from_map(labels, mask_result.room_confidence_map, grid_spec)
            edges = _room_edges(labels, cut_score)
        out = MaskAssignmentResult(
            room_id_map=labels,
            room_confidence_map=mask_result.room_confidence_map,
            room_soft_masks=mask_result.room_soft_masks,
            corridor_mask=corridor_mask,
            open_space_mask=mask_result.open_space_mask,
            functional_zone_map=mask_result.functional_zone_map,
            room_instances=instances,
            room_graph_edges=edges,
            debug_layers=dict(mask_result.debug_layers),
        )
        setattr(out, "_grid_spec", grid_spec)
        out.debug_layers["corridor_mask"] = corridor_mask.astype(np.uint8)
        for instance in out.room_instances:
            room_mask = labels == int(instance.room_id)
            overlap = float(np.count_nonzero(room_mask & corridor_mask)) / float(max(1, np.count_nonzero(room_mask)))
            if int(instance.room_id) in corridor_labels or overlap >= 0.15:
                instance.is_corridor = True
                instance.room_type = "corridor"
                instance.functional_zone_label = "corridor_zone"
        return out

    def _corridor_labels(self, labels: np.ndarray, skeleton_graph: SkeletonGraph, structural_map: StructuralMap) -> set[int]:
        out: set[int] = set()
        for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            mask = labels == label
            widths = []
            degree2 = 0
            total_nodes = 0
            node_ids = []
            for node in skeleton_graph.nodes:
                row, col = node.uv
                if mask[row, col]:
                    total_nodes += 1
                    node_ids.append(node.node_id)
                    widths.append(2.0 * float(node.clearance_m))
                    if int(node.degree) == 2:
                        degree2 += 1
            if widths:
                mean_width = float(np.mean(widths))
                width_cv = float(np.std(widths) / max(mean_width, 1e-6))
                degree2_ratio = float(degree2) / float(max(1, total_nodes))
                skeleton_length = 0.0
                node_set = set(node_ids)
                for edge in skeleton_graph.edges:
                    if edge.src in node_set and edge.dst in node_set:
                        skeleton_length += float(edge.length_m)
            else:
                rows, cols = np.nonzero(mask)
                if rows.size == 0:
                    continue
                span_r = float(rows.max() - rows.min() + 1) * self.resolution_m
                span_c = float(cols.max() - cols.min() + 1) * self.resolution_m
                mean_width = min(span_r, span_c)
                width_cv = 0.0
                degree2_ratio = 1.0
                skeleton_length = max(span_r, span_c)
            length_to_width = skeleton_length / max(mean_width, 1e-6)
            wall_support = _parallel_wall_support(mask, structural_map, self.resolution_m)
            if (
                float(self.config.width_min_m) <= mean_width <= float(self.config.width_max_m)
                and length_to_width >= float(self.config.length_to_width_ratio_min)
                and width_cv <= float(self.config.width_cv_max)
                and degree2_ratio >= float(self.config.skeleton_degree2_ratio_min)
                and wall_support >= max(0.20, float(self.config.parallel_wall_support_min) * 0.5)
            ):
                out.add(int(label))
        return out

    def _merge_corridor_neighbors(self, labels: np.ndarray, corridor_labels: set[int], cut_score: np.ndarray) -> np.ndarray:
        if not corridor_labels:
            return labels
        parent = {int(label): int(label) for label in np.unique(labels) if int(label) > 0}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in adjacency_pairs(labels):
            if a not in corridor_labels or b not in corridor_labels:
                continue
            boundary = dilate(labels == int(a), 1) & (labels == int(b))
            score = float(np.max(cut_score[dilate(boundary, 1)])) if np.any(boundary) else 0.0
            if score < float(self.config.max_internal_cut_score):
                union(int(a), int(b))
        out = labels.copy()
        mapping: dict[int, int] = {}
        next_label = 1
        for label in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
            root = find(label)
            if root not in mapping:
                mapping[root] = next_label
                next_label += 1
            out[labels == label] = mapping[root]
        return out


def _parallel_wall_support(mask: np.ndarray, structural_map: StructuralMap, resolution_m: float) -> float:
    wall = np.asarray(structural_map.p_wall, dtype=np.float32) >= 0.55
    boundary = dilate(mask, max(1, int(round(0.25 / max(resolution_m, 1e-9))))) & ~mask
    total = int(np.count_nonzero(boundary))
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(boundary & wall)) / float(total)


def _corridor_mask_from_skeleton(free: np.ndarray, skeleton_graph: SkeletonGraph, config: CorridorConfig, resolution_m: float) -> np.ndarray:
    out = np.zeros_like(free, dtype=bool)
    if not skeleton_graph.nodes:
        return out
    for node in skeleton_graph.nodes:
        width = 2.0 * float(node.clearance_m)
        if not (float(config.width_min_m) <= width <= float(config.width_max_m)):
            continue
        if int(node.degree) > 2:
            continue
        radius = max(1, int(round(float(node.clearance_m) / max(float(resolution_m), 1e-9))))
        rr, cc = node.uv
        r0, r1 = max(0, rr - radius), min(out.shape[0], rr + radius + 1)
        c0, c1 = max(0, cc - radius), min(out.shape[1], cc + radius + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        disk = (yy - rr) * (yy - rr) + (xx - cc) * (xx - cc) <= radius * radius
        out[r0:r1, c0:c1] |= disk
    out &= np.asarray(free, dtype=bool)
    return out
