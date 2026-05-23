from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from .config import SkeletonConfig
from .data_types import FreeSpaceState, GridSpec, SkeletonEdge, SkeletonGraph, SkeletonNode, StructuralMap
from .utils import grid_to_world, in_bounds, neighbor_offsets, radius_cells


class SkeletonExtractor:
    def __init__(self, config: SkeletonConfig | Mapping[str, object] | None = None, grid_spec: GridSpec | None = None):
        self.config = config if isinstance(config, SkeletonConfig) else SkeletonConfig.from_mapping(config or {})
        self.grid_spec = grid_spec

    def extract(self, free_space: FreeSpaceState, structural_map: StructuralMap) -> SkeletonGraph:
        free = np.asarray(free_space.free_eroded_mask, dtype=bool)
        distance_m = np.asarray(free_space.distance_transform_m, dtype=np.float32)
        skeleton = _ridge_skeleton(free, distance_m)
        skeleton = _thin_plateaus(skeleton, distance_m)
        skeleton = _remove_short_components(skeleton, distance_m, float(self.config.min_skeleton_component_length_m), self._resolution())
        skeleton = _prune_short_branches(
            skeleton,
            frontier=np.asarray(free_space.frontier_mask, dtype=bool),
            prune_length_m=float(self.config.prune_branch_length_m),
            resolution_m=self._resolution(),
        )
        node_id_map = np.full(free.shape, -1, dtype=np.int32)
        rows, cols = np.nonzero(skeleton)
        max_nodes = max(1, int(self.config.max_nodes))
        if rows.size > max_nodes:
            keep = np.linspace(0, rows.size - 1, max_nodes).astype(np.int64)
            rows = rows[keep]
            cols = cols[keep]
            sampled = np.zeros_like(skeleton)
            sampled[rows, cols] = True
            skeleton = sampled
        nodes: list[SkeletonNode] = []
        for idx, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
            node_id_map[int(row), int(col)] = int(idx)
            xy = grid_to_world(int(row), int(col), self.grid_spec) if self.grid_spec is not None else (float(col), float(row))
            nodes.append(
                SkeletonNode(
                    node_id=int(idx),
                    xy=(float(xy[0]), float(xy[1])),
                    uv=(int(row), int(col)),
                    clearance_m=float(distance_m[int(row), int(col)]),
                    degree=0,
                )
            )
        edges: list[SkeletonEdge] = []
        adjacency: dict[int, list[int]] = {node.node_id: [] for node in nodes}
        for node in nodes:
            row, col = node.uv
            for dr, dc in neighbor_offsets(8):
                nr, nc = row + dr, col + dc
                if not in_bounds(nr, nc, skeleton.shape):
                    continue
                other = int(node_id_map[nr, nc])
                if other < 0 or other <= node.node_id:
                    continue
                step = float(np.hypot(dr, dc)) * self._resolution()
                mean_clearance = 0.5 * (float(distance_m[row, col]) + float(distance_m[nr, nc]))
                edges.append(SkeletonEdge(src=int(node.node_id), dst=int(other), length_m=step, mean_clearance_m=mean_clearance))
                adjacency[int(node.node_id)].append(int(other))
                adjacency[int(other)].append(int(node.node_id))
        for node in nodes:
            node.degree = int(len(adjacency.get(node.node_id, [])))
        return SkeletonGraph(
            nodes=nodes,
            edges=edges,
            adjacency=adjacency,
            node_id_map=node_id_map,
            skeleton_mask=skeleton.astype(bool),
            distance_transform_m=distance_m.astype(np.float32),
        )

    def local_minima_node_ids(self, graph: SkeletonGraph) -> list[int]:
        if not graph.nodes:
            return []
        radius = radius_cells(float(self.config.local_minima_window_m), self._resolution())
        prominence = float(self.config.local_minima_prominence_m)
        out: list[int] = []
        skeleton = np.asarray(graph.skeleton_mask, dtype=bool)
        dist = np.asarray(graph.distance_transform_m, dtype=np.float32)
        for node in graph.nodes:
            row, col = node.uv
            r0, r1 = max(0, row - radius), min(skeleton.shape[0], row + radius + 1)
            c0, c1 = max(0, col - radius), min(skeleton.shape[1], col + radius + 1)
            local = dist[r0:r1, c0:c1][skeleton[r0:r1, c0:c1]]
            if local.size < 3:
                continue
            width = 2.0 * float(node.clearance_m)
            if 0.35 <= width <= 2.80 and float(node.clearance_m) < float(np.mean(local)) - prominence:
                out.append(int(node.node_id))
        return out

    def _resolution(self) -> float:
        return float(self.grid_spec.resolution_m) if self.grid_spec is not None else 1.0


def _ridge_skeleton(free: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    if not np.any(free):
        return np.zeros_like(free, dtype=bool)
    dist = np.asarray(distance_m, dtype=np.float32)
    max_neigh = np.zeros_like(dist, dtype=np.float32)
    for dr, dc in neighbor_offsets(8):
        shifted = np.zeros_like(dist, dtype=np.float32)
        sr0 = max(0, -dr)
        sr1 = min(dist.shape[0], dist.shape[0] - dr)
        sc0 = max(0, -dc)
        sc1 = min(dist.shape[1], dist.shape[1] - dc)
        shifted[sr0 + dr : sr1 + dr, sc0 + dc : sc1 + dc] = dist[sr0:sr1, sc0:sc1]
        max_neigh = np.maximum(max_neigh, shifted)
    skeleton = free & (dist >= (max_neigh - 1e-5)) & (dist > 0.0)
    if np.count_nonzero(skeleton) < 2:
        threshold = max(0.0, float(np.percentile(dist[free], 70.0)) if np.any(free) else 0.0)
        skeleton = free & (dist >= threshold)
    return skeleton.astype(bool)


def _thin_plateaus(skeleton: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    src = np.asarray(skeleton, dtype=bool)
    out = np.zeros_like(src, dtype=bool)
    labels = np.zeros(src.shape, dtype=np.int32)
    label = 0
    for start in zip(*np.nonzero(src)):
        if labels[start] != 0:
            continue
        label += 1
        q: deque[tuple[int, int]] = deque([(int(start[0]), int(start[1]))])
        labels[start] = label
        cells: list[tuple[int, int]] = []
        while q:
            row, col = q.popleft()
            cells.append((row, col))
            for dr, dc in neighbor_offsets(8):
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc, src.shape) and src[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = label
                    q.append((nr, nc))
        if len(cells) <= 2:
            for cell in cells:
                out[cell] = True
            continue
        rows = np.asarray([c[0] for c in cells], dtype=np.int32)
        cols = np.asarray([c[1] for c in cells], dtype=np.int32)
        span_r = int(rows.max() - rows.min() + 1)
        span_c = int(cols.max() - cols.min() + 1)
        if max(span_r, span_c) / max(1, min(span_r, span_c)) < 1.5:
            for cell in cells:
                out[cell] = True
            continue
        if span_c >= span_r:
            for col in sorted(set(cols.tolist())):
                same = [cell for cell in cells if cell[1] == col]
                best = max(same, key=lambda cell: (float(distance_m[cell]), -abs(cell[0] - float(np.mean(rows)))))
                out[best] = True
        else:
            for row in sorted(set(rows.tolist())):
                same = [cell for cell in cells if cell[0] == row]
                best = max(same, key=lambda cell: (float(distance_m[cell]), -abs(cell[1] - float(np.mean(cols)))))
                out[best] = True
    return out


def _remove_short_components(skeleton: np.ndarray, distance_m: np.ndarray, min_length_m: float, resolution_m: float) -> np.ndarray:
    src = np.asarray(skeleton, dtype=bool)
    labels = np.zeros(src.shape, dtype=np.int32)
    out = np.zeros_like(src, dtype=bool)
    label = 0
    for start in zip(*np.nonzero(src)):
        if labels[start] != 0:
            continue
        label += 1
        q: deque[tuple[int, int]] = deque([(int(start[0]), int(start[1]))])
        labels[start] = label
        cells: list[tuple[int, int]] = []
        while q:
            row, col = q.popleft()
            cells.append((row, col))
            for dr, dc in neighbor_offsets(8):
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc, src.shape) and src[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = label
                    q.append((nr, nc))
        length = max(float(len(cells) - 1) * float(resolution_m), float(np.max(distance_m[tuple(np.asarray(cells).T)]) if cells else 0.0))
        if length >= float(min_length_m) or len(cells) >= 2:
            for row, col in cells:
                out[row, col] = True
    return out


def _degrees(skeleton: np.ndarray) -> np.ndarray:
    deg = np.zeros(skeleton.shape, dtype=np.int16)
    rows, cols = np.nonzero(skeleton)
    for row, col in zip(rows.tolist(), cols.tolist()):
        count = 0
        for dr, dc in neighbor_offsets(8):
            nr, nc = int(row) + dr, int(col) + dc
            if in_bounds(nr, nc, skeleton.shape) and skeleton[nr, nc]:
                count += 1
        deg[int(row), int(col)] = count
    return deg


def _prune_short_branches(skeleton: np.ndarray, frontier: np.ndarray, prune_length_m: float, resolution_m: float) -> np.ndarray:
    skel = np.asarray(skeleton, dtype=bool).copy()
    if not np.any(skel):
        return skel
    prune_cells = max(1, int(round(float(prune_length_m) / max(float(resolution_m), 1e-9))))
    changed = True
    while changed:
        changed = False
        deg = _degrees(skel)
        leaves = [(int(r), int(c)) for r, c in zip(*np.nonzero(skel & (deg == 1)))]
        for leaf in leaves:
            if not skel[leaf]:
                continue
            path = [leaf]
            prev: tuple[int, int] | None = None
            cur = leaf
            while True:
                neigh = []
                for dr, dc in neighbor_offsets(8):
                    nr, nc = cur[0] + dr, cur[1] + dc
                    if in_bounds(nr, nc, skel.shape) and skel[nr, nc] and (prev is None or (nr, nc) != prev):
                        neigh.append((nr, nc))
                if len(neigh) != 1:
                    break
                prev = cur
                cur = neigh[0]
                path.append(cur)
                if deg[cur] != 2:
                    break
                if len(path) > prune_cells:
                    break
            touches_frontier = any(frontier[row, col] for row, col in path)
            if len(path) <= prune_cells and not touches_frontier:
                for row, col in path:
                    skel[row, col] = False
                changed = True
    return skel
