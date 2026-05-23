from __future__ import annotations

from typing import Mapping

import numpy as np

from .config import PartitionConfig
from .data_types import CutCandidate, PartitionResult, PlaceGraph, StructuralMap
from .utils import in_bounds, path_crosses, relabel_compact


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        item = int(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class GraphPartitioner:
    def __init__(self, config: PartitionConfig | Mapping[str, object] | None = None):
        self.config = config if isinstance(config, PartitionConfig) else PartitionConfig.from_mapping(config or {})

    def partition(self, place_graph: PlaceGraph, cut_candidates: list[CutCandidate], structural_map: StructuralMap) -> PartitionResult:
        shape = np.asarray(structural_map.p_wall).shape
        separator_map = np.zeros(shape, dtype=np.float32)
        hard_map = np.zeros(shape, dtype=bool)
        for candidate in cut_candidates:
            if not candidate.is_soft_separator and float(candidate.final_score) < float(self.config.soft_split_score_threshold):
                continue
            for cell in candidate.cut_cells:
                if in_bounds(cell[0], cell[1], shape):
                    separator_map[cell] = max(float(separator_map[cell]), float(candidate.final_score))
                    hard_map[cell] = hard_map[cell] or bool(candidate.is_hard_separator)
        uf = UnionFind([node.node_id for node in place_graph.nodes])
        for edge in place_graph.edges:
            crosses_soft = path_crosses(separator_map >= float(self.config.soft_split_score_threshold), place_graph.nodes[edge.src].uv, place_graph.nodes[edge.dst].uv)
            crosses_hard = path_crosses(hard_map, place_graph.nodes[edge.src].uv, place_graph.nodes[edge.dst].uv)
            weight = float(edge.weight)
            if crosses_soft:
                weight -= float(edge.bottleneck_penalty) * 0.45
            if not crosses_hard and weight >= float(self.config.merge_score_threshold):
                uf.union(edge.src, edge.dst)
        root_to_label: dict[int, int] = {}
        node_labels: dict[int, int] = {}
        next_label = 1
        for node in place_graph.nodes:
            root = uf.find(node.node_id)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            node_labels[node.node_id] = root_to_label[root]
        if self.config.normalized_cut_enabled:
            node_labels = self._normalized_cut_fallback(node_labels, place_graph, cut_candidates, structural_map)
        seed_cells: dict[int, list[tuple[int, int]]] = {}
        for node in place_graph.nodes:
            label = int(node_labels.get(node.node_id, 0))
            if label <= 0:
                continue
            seed_cells.setdefault(label, []).append((int(node.uv[0]), int(node.uv[1])))
        return PartitionResult(
            node_labels=node_labels,
            separator_map=(separator_map >= float(self.config.soft_split_score_threshold)).astype(bool),
            seed_cells_by_label=seed_cells,
            debug={
                "partition_method": str(self.config.method),
                "node_count": int(len(place_graph.nodes)),
                "edge_count": int(len(place_graph.edges)),
                "partition_count": int(len(set(node_labels.values()))),
                "soft_separator_cells": int(np.count_nonzero(separator_map >= float(self.config.soft_split_score_threshold))),
                "hard_separator_cells": int(np.count_nonzero(hard_map)),
            },
        )

    def _normalized_cut_fallback(
        self,
        node_labels: dict[int, int],
        place_graph: PlaceGraph,
        cut_candidates: list[CutCandidate],
        structural_map: StructuralMap,
    ) -> dict[int, int]:
        if not node_labels:
            return node_labels
        counts: dict[int, int] = {}
        for label in node_labels.values():
            counts[int(label)] = counts.get(int(label), 0) + 1
        if not counts:
            return node_labels
        largest = max(counts, key=counts.get)
        high_soft = [c for c in cut_candidates if float(c.final_score) >= float(self.config.soft_split_score_threshold)]
        if counts[largest] < 6 or len(high_soft) < 2:
            return node_labels
        best = max(high_soft, key=lambda c: c.final_score)
        out = dict(node_labels)
        next_label = max(out.values()) + 1
        center = np.asarray(best.center_uv, dtype=np.float32)
        normal = np.asarray([best.normal_xy[1], best.normal_xy[0]], dtype=np.float32)
        for node in place_graph.nodes:
            if out.get(node.node_id) != largest:
                continue
            side = float(np.dot(np.asarray(node.uv, dtype=np.float32) - center, normal))
            if side > 0:
                out[node.node_id] = next_label
        return out

