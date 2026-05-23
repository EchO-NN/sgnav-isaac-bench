from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .config import BottleneckConfig, SeparatorScoringConfig, StructuralWallConfig, TemporalConfig
from .data_types import CutCandidate, FreeSpaceState, GridSpec, SkeletonGraph, StructuralMap
from .structural_wall import wall_support_near
from .utils import (
    clamp,
    dilate,
    flood_fill,
    grid_to_world,
    in_bounds,
    label_components,
    line_mask,
    neighbor_offsets,
    radius_cells,
    rasterize_line,
    sigmoid,
)
from .visibility import VisibilityComputer


class BottleneckDetector:
    def __init__(
        self,
        config: BottleneckConfig | Mapping[str, object] | None = None,
        structural_config: StructuralWallConfig | Mapping[str, object] | None = None,
        grid_spec: GridSpec | None = None,
    ):
        self.config = config if isinstance(config, BottleneckConfig) else BottleneckConfig.from_mapping(config or {})
        self.structural_config = (
            structural_config
            if isinstance(structural_config, StructuralWallConfig)
            else StructuralWallConfig.from_mapping(structural_config or {})
        )
        self.grid_spec = grid_spec

    def generate(self, skeleton_graph: SkeletonGraph, structural_map: StructuralMap, free_space: FreeSpaceState) -> list[CutCandidate]:
        candidates: list[CutCandidate] = []
        next_id = 1
        minima = _local_minimum_nodes(skeleton_graph, self.config)
        for node_id in minima:
            cand = self._candidate_from_node(
                next_id,
                skeleton_graph,
                node_id,
                structural_map,
                free_space,
                kind="skeleton_local_minimum",
            )
            if cand is not None:
                candidates.append(cand)
                next_id += 1
        for node_id in _degree_branch_nodes(skeleton_graph):
            cand = self._candidate_from_node(
                next_id,
                skeleton_graph,
                node_id,
                structural_map,
                free_space,
                kind="corridor_side_branch",
            )
            if cand is not None:
                candidates.append(cand)
                next_id += 1
        for node_id in _narrow_nodes(skeleton_graph, self.config, limit=80):
            cand = self._candidate_from_node(
                next_id,
                skeleton_graph,
                node_id,
                structural_map,
                free_space,
                kind="narrow_passage",
            )
            if cand is not None:
                candidates.append(cand)
                next_id += 1
        for cand in self._wall_endpoint_pair_candidates(next_id, structural_map, free_space):
            candidates.append(cand)
            next_id += 1
        if not candidates:
            weak = self._open_space_weak_candidate(next_id, structural_map, free_space)
            if weak is not None:
                candidates.append(weak)
        candidates = _deduplicate_candidates(candidates, self.config)
        candidates.sort(key=lambda item: (item.debug.get("local_minimum_prominence", 0.0), width_score(item.width_m, self.config), -item.width_m), reverse=True)
        return candidates[: max(1, int(self.config.max_candidates))]

    def _candidate_from_node(
        self,
        candidate_id: int,
        graph: SkeletonGraph,
        node_id: int,
        structural_map: StructuralMap,
        free_space: FreeSpaceState,
        *,
        kind: str,
    ) -> CutCandidate | None:
        node = graph.nodes[int(node_id)]
        tangent = _local_tangent(graph, int(node_id), radius_m=0.25)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        return self._candidate_from_center(
            candidate_id,
            node.uv,
            tangent,
            normal,
            structural_map,
            free_space,
            kind=kind,
            local_prominence=float(_local_prominence(graph, int(node_id))),
        )

    def _candidate_from_center(
        self,
        candidate_id: int,
        center_rc: tuple[int, int],
        tangent_rc: np.ndarray,
        normal_rc: np.ndarray,
        structural_map: StructuralMap,
        free_space: FreeSpaceState,
        *,
        kind: str,
        local_prominence: float = 0.0,
    ) -> CutCandidate | None:
        free = np.asarray(free_space.observed_free_mask, dtype=bool)
        if not in_bounds(center_rc[0], center_rc[1], free.shape) or not free[center_rc]:
            return None
        n_norm = float(np.linalg.norm(normal_rc))
        t_norm = float(np.linalg.norm(tangent_rc))
        if n_norm <= 1e-6 or t_norm <= 1e-6:
            return None
        normal = normal_rc.astype(np.float32) / n_norm
        tangent = tangent_rc.astype(np.float32) / t_norm
        cells = _ray_cut_cells(center_rc, normal, structural_map, free_space, self.config, self.structural_config, self._resolution())
        if not cells:
            return None
        rel = np.asarray(cells, dtype=np.float32) - np.asarray(center_rc, dtype=np.float32)[None, :]
        proj = rel @ normal
        raster_width_m = (float(np.max(proj) - np.min(proj)) + 1.0) * self._resolution() if proj.size else 0.0
        width_m = max(raster_width_m, 2.0 * float(free_space.distance_transform_m[center_rc]))
        if width_m < float(self.config.candidate_width_min_m) or width_m > float(self.config.candidate_width_max_m):
            return None
        left, right = _side_seed_cells(center_rc, tangent, free, self.config, self._resolution())
        if not left or not right:
            return None
        if not _valid_side_areas(free, cells, left, right, self.config, self._resolution()):
            return None
        if _frontier_ratio(cells, free_space.frontier_mask, self.config, self._resolution()) > float(self.config.max_frontier_ratio_near_cut):
            return None
        xy = grid_to_world(center_rc[0], center_rc[1], self.grid_spec) if self.grid_spec is not None else (float(center_rc[1]), float(center_rc[0]))
        normal_xy = (float(normal[1]), float(normal[0]))
        tangent_xy = (float(tangent[1]), float(tangent[0]))
        return CutCandidate(
            candidate_id=int(candidate_id),
            center_uv=(int(center_rc[0]), int(center_rc[1])),
            center_xy=(float(xy[0]), float(xy[1])),
            normal_xy=normal_xy,
            tangent_xy=tangent_xy,
            width_m=float(width_m),
            cut_cells=list(cells),
            left_seed_cells=list(left),
            right_seed_cells=list(right),
            kind=str(kind),
            debug={"local_minimum_prominence": float(local_prominence)},
        )

    def _wall_endpoint_pair_candidates(
        self,
        start_id: int,
        structural_map: StructuralMap,
        free_space: FreeSpaceState,
    ) -> list[CutCandidate]:
        hard = np.asarray(structural_map.hard_wall_mask, dtype=bool)
        free = np.asarray(free_space.observed_free_mask, dtype=bool)
        endpoints = _wall_endpoints(hard, max_points=96)
        out: list[CutCandidate] = []
        next_id = int(start_id)
        max_len_cells = int(round(float(self.config.candidate_width_max_m) / max(self._resolution(), 1e-9)))
        min_len_cells = int(round(float(self.config.candidate_width_min_m) / max(self._resolution(), 1e-9)))
        for i, p0 in enumerate(endpoints):
            for p1 in endpoints[i + 1 :]:
                dist = float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))
                if dist < min_len_cells or dist > max_len_cells:
                    continue
                cells = rasterize_line(p0, p1, free.shape)
                if not cells:
                    continue
                free_ratio = float(sum(1 for cell in cells if free[cell])) / float(len(cells))
                if free_ratio < 0.65:
                    continue
                center = cells[len(cells) // 2]
                tangent = np.asarray([float(p1[0] - p0[0]), float(p1[1] - p0[1])], dtype=np.float32)
                if np.linalg.norm(tangent) <= 1e-6:
                    continue
                normal = tangent / float(np.linalg.norm(tangent))
                tangent_axis = np.asarray([-normal[1], normal[0]], dtype=np.float32)
                cand = self._candidate_from_center(
                    next_id,
                    center,
                    tangent_axis,
                    normal,
                    structural_map,
                    free_space,
                    kind="wall_endpoint_pair",
                    local_prominence=0.05,
                )
                if cand is not None:
                    out.append(cand)
                    next_id += 1
                if len(out) >= 24:
                    return out
        return out

    def _open_space_weak_candidate(self, candidate_id: int, structural_map: StructuralMap, free_space: FreeSpaceState) -> CutCandidate | None:
        free = np.asarray(free_space.observed_free_mask, dtype=bool)
        if int(np.count_nonzero(free)) <= 0:
            return None
        dist = np.asarray(free_space.distance_transform_m, dtype=np.float32)
        row, col = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return self._candidate_from_center(
            int(candidate_id),
            (int(row), int(col)),
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([1.0, 0.0], dtype=np.float32),
            structural_map,
            free_space,
            kind="open_space_weak_separator",
            local_prominence=0.0,
        )

    def _resolution(self) -> float:
        return float(self.grid_spec.resolution_m) if self.grid_spec is not None else 0.05


class SeparatorScorer:
    def __init__(
        self,
        config: SeparatorScoringConfig | Mapping[str, object] | None = None,
        bottleneck_config: BottleneckConfig | Mapping[str, object] | None = None,
        temporal_config: TemporalConfig | Mapping[str, object] | None = None,
        visibility: VisibilityComputer | None = None,
        resolution_m: float = 0.05,
    ):
        self.config = config if isinstance(config, SeparatorScoringConfig) else SeparatorScoringConfig.from_mapping(config or {})
        self.bottleneck_config = (
            bottleneck_config if isinstance(bottleneck_config, BottleneckConfig) else BottleneckConfig.from_mapping(bottleneck_config or {})
        )
        self.temporal_config = temporal_config if isinstance(temporal_config, TemporalConfig) else TemporalConfig.from_mapping(temporal_config or {})
        self.visibility = visibility or VisibilityComputer(resolution_m=resolution_m)
        self.resolution_m = float(resolution_m)

    def score_all(
        self,
        cut_candidates: list[CutCandidate],
        skeleton_graph: SkeletonGraph,
        structural_map: StructuralMap,
        free_space: FreeSpaceState,
        temporal_state: Mapping[tuple[int, int], Mapping[str, float]] | None = None,
    ) -> list[CutCandidate]:
        out = []
        for candidate in cut_candidates:
            out.append(self.score(candidate, skeleton_graph, structural_map, free_space, temporal_state or {}))
        out.sort(key=lambda item: item.final_score, reverse=True)
        return out

    def score(
        self,
        candidate: CutCandidate,
        skeleton_graph: SkeletonGraph,
        structural_map: StructuralMap,
        free_space: FreeSpaceState,
        temporal_state: Mapping[tuple[int, int], Mapping[str, float]],
    ) -> CutCandidate:
        cfg = self.config
        bcfg = self.bottleneck_config
        candidate.width_score = width_score(candidate.width_m, bcfg)
        radius = radius_cells(float(bcfg.endpoint_wall_support_radius_m), self.resolution_m)
        endpoints = [candidate.cut_cells[0], candidate.cut_cells[-1]] if candidate.cut_cells else []
        candidate.wall_support_score = wall_support_near(np.asarray(structural_map.p_wall, dtype=np.float32), endpoints, radius)
        if candidate.wall_support_score > 0.60:
            candidate.wall_support_score = clamp(candidate.wall_support_score + 0.15)
        candidate.visibility_drop_score = self.visibility.visibility_drop(candidate.left_seed_cells, candidate.right_seed_cells, structural_map)
        graph_score, area_balance, alcove = _graph_conductance_and_area(candidate, free_space, self.resolution_m, bcfg)
        candidate.graph_conductance_score = graph_score
        candidate.area_balance_score = area_balance
        candidate.alcove_penalty = max(alcove, _alcove_penalty(candidate, skeleton_graph, free_space, self.resolution_m))
        candidate.frontier_penalty = clamp(_frontier_ratio(candidate.cut_cells, free_space.frontier_mask, bcfg, self.resolution_m) / max(1e-6, float(bcfg.max_frontier_ratio_near_cut)))
        candidate.corridor_consistency_score = clamp(1.0 - abs(candidate.width_m - 1.05) / 1.80)
        key = (int(candidate.center_uv[0]), int(candidate.center_uv[1]))
        hist = temporal_state.get(key, {})
        candidate.temporal_score = clamp(float(hist.get("score", 0.0)))
        candidate.temporal_observations = int(hist.get("observations", 0))
        weighted = (
            float(cfg.weight_width) * candidate.width_score
            + float(cfg.weight_wall_support) * candidate.wall_support_score
            + float(cfg.weight_visibility_drop) * candidate.visibility_drop_score
            + float(cfg.weight_graph_conductance) * candidate.graph_conductance_score
            + float(cfg.weight_temporal) * candidate.temporal_score
            + float(cfg.weight_area_balance) * candidate.area_balance_score
            + float(cfg.weight_corridor_consistency) * candidate.corridor_consistency_score
            - float(cfg.weight_frontier_penalty) * candidate.frontier_penalty
            - float(cfg.weight_alcove_penalty) * candidate.alcove_penalty
        )
        candidate.final_score = clamp(float(sigmoid(weighted / max(float(cfg.sigmoid_temperature), 1e-6))), float(cfg.score_clip_min), float(cfg.score_clip_max))
        if candidate.kind == "open_space_weak_separator":
            candidate.final_score = min(candidate.final_score, 0.49)
        prominence_ok = not (
            candidate.kind in {"skeleton_local_minimum", "narrow_passage"}
            and float(candidate.debug.get("local_minimum_prominence", 0.0)) < 0.03
        )
        quality_ok = bool((candidate.wall_support_score >= 0.30 or candidate.visibility_drop_score >= 0.45) and prominence_ok)
        candidate.is_soft_separator = bool(
            candidate.final_score >= float(self.temporal_config.soft_split_threshold)
            and quality_ok
            and candidate.alcove_penalty < 0.95
        )
        candidate.is_hard_separator = bool(
            candidate.final_score >= float(self.temporal_config.hard_split_threshold)
            and candidate.temporal_observations >= int(self.temporal_config.split_persistence_observations)
        )
        candidate.debug.update(
            {
                "side_area_balance": float(candidate.area_balance_score),
                "frontier_ratio": float(candidate.frontier_penalty),
                "quality_ok": bool(quality_ok),
            }
        )
        return candidate


def width_score(width_m: float, config: BottleneckConfig | None = None) -> float:
    cfg = config or BottleneckConfig()
    width = float(width_m)
    if width < float(cfg.candidate_width_min_m):
        return 0.0
    if float(cfg.candidate_width_preferred_min_m) <= width <= float(cfg.candidate_width_preferred_max_m):
        return 1.0
    if float(cfg.candidate_width_min_m) <= width < float(cfg.candidate_width_preferred_min_m):
        return clamp((width - float(cfg.candidate_width_min_m)) / max(1e-6, float(cfg.candidate_width_preferred_min_m) - float(cfg.candidate_width_min_m)))
    if float(cfg.candidate_width_preferred_max_m) < width <= float(cfg.candidate_width_max_m):
        return float(np.exp(-((width - float(cfg.candidate_width_preferred_max_m)) ** 2) / (2.0 * 0.55**2)))
    return 0.0


def _local_minimum_nodes(graph: SkeletonGraph, config: BottleneckConfig) -> list[int]:
    if not graph.nodes:
        return []
    out: list[int] = []
    for node in graph.nodes:
        if 2.0 * float(node.clearance_m) < float(config.candidate_width_min_m) or 2.0 * float(node.clearance_m) > float(config.candidate_width_max_m):
            continue
        neigh = [graph.nodes[n].clearance_m for n in graph.adjacency.get(node.node_id, [])]
        if not neigh:
            continue
        if float(node.clearance_m) <= float(np.mean(neigh)) + 0.02:
            out.append(int(node.node_id))
    if not out:
        ordered = sorted(graph.nodes, key=lambda n: n.clearance_m)
        out = [int(node.node_id) for node in ordered[: min(32, len(ordered))]]
    return out


def _degree_branch_nodes(graph: SkeletonGraph) -> list[int]:
    return [int(node.node_id) for node in graph.nodes if int(node.degree) >= 3]


def _narrow_nodes(graph: SkeletonGraph, config: BottleneckConfig, limit: int) -> list[int]:
    nodes = [
        node
        for node in graph.nodes
        if float(config.candidate_width_min_m) <= 2.0 * float(node.clearance_m) <= float(config.candidate_width_preferred_max_m)
    ]
    nodes.sort(key=lambda node: node.clearance_m)
    return [int(node.node_id) for node in nodes[: int(limit)]]


def _local_tangent(graph: SkeletonGraph, node_id: int, radius_m: float) -> np.ndarray:
    node = graph.nodes[int(node_id)]
    q = [int(node_id)]
    seen = {int(node_id)}
    cells: list[tuple[int, int]] = [node.uv]
    while q and len(cells) < 24:
        cur = q.pop(0)
        for nxt in graph.adjacency.get(cur, []):
            if nxt in seen:
                continue
            if _dist_uv(node.uv, graph.nodes[nxt].uv) * max(1e-6, node.clearance_m) > 20.0 * max(radius_m, 1e-6):
                continue
            seen.add(nxt)
            q.append(nxt)
            cells.append(graph.nodes[nxt].uv)
    if len(cells) < 2:
        neigh = graph.adjacency.get(node_id, [])
        if neigh:
            other = graph.nodes[neigh[0]].uv
            vec = np.asarray([float(other[0] - node.uv[0]), float(other[1] - node.uv[1])], dtype=np.float32)
        else:
            vec = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        coords = np.asarray(cells, dtype=np.float32)
        coords -= np.mean(coords, axis=0, keepdims=True)
        vals, vecs = np.linalg.eigh(np.cov(coords.T))
        vec = vecs[:, int(np.argmax(vals))].astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return vec / norm


def _local_prominence(graph: SkeletonGraph, node_id: int) -> float:
    node = graph.nodes[int(node_id)]
    neigh = [graph.nodes[n].clearance_m for n in graph.adjacency.get(node.node_id, [])]
    if not neigh:
        return 0.0
    return float(max(0.0, float(np.mean(neigh)) - float(node.clearance_m)))


def _dist_uv(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(int(a[0]) - int(b[0]), int(a[1]) - int(b[1])))


def _ray_cut_cells(
    center_rc: tuple[int, int],
    normal: np.ndarray,
    structural_map: StructuralMap,
    free_space: FreeSpaceState,
    config: BottleneckConfig,
    structural_config: StructuralWallConfig,
    resolution_m: float,
) -> list[tuple[int, int]]:
    free = np.asarray(free_space.observed_free_mask, dtype=bool)
    p_wall = np.asarray(structural_map.p_wall, dtype=np.float32)
    unknown = np.asarray(structural_map.p_unknown, dtype=np.float32) >= float(structural_config.unknown_probability_threshold)
    hard = np.asarray(structural_map.hard_wall_mask, dtype=bool)
    max_steps = max(1, int(round(float(config.cut_max_length_m) / max(float(resolution_m), 1e-9))))
    cells: list[tuple[int, int]] = [center_rc]
    for sign in (-1.0, 1.0):
        side: list[tuple[int, int]] = []
        for step in range(1, max_steps + 1):
            row = int(round(float(center_rc[0]) + sign * float(normal[0]) * float(step)))
            col = int(round(float(center_rc[1]) + sign * float(normal[1]) * float(step)))
            if not in_bounds(row, col, free.shape):
                break
            if hard[row, col] or p_wall[row, col] >= float(structural_config.wall_probability_threshold):
                break
            if unknown[row, col] or not free[row, col]:
                break
            side.append((row, col))
        if sign < 0:
            cells = list(reversed(side)) + cells
        else:
            cells = cells + side
    thickness = max(0, radius_cells(float(config.cut_raster_thickness_m), resolution_m) - 1)
    if thickness > 0:
        mask = np.zeros(free.shape, dtype=bool)
        for cell in cells:
            mask[cell] = True
        mask = dilate(mask, thickness)
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(mask & free))]
    return cells


def _side_seed_cells(
    center_rc: tuple[int, int],
    tangent: np.ndarray,
    free: np.ndarray,
    config: BottleneckConfig,
    resolution_m: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    base = float(config.side_sampling_distance_m) / max(float(resolution_m), 1e-9)
    offsets = np.linspace(-0.5, 0.5, max(1, int(config.side_sampling_count)))
    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    for sign, out in ((-1.0, left), (1.0, right)):
        for off in offsets:
            rr = float(center_rc[0]) + sign * float(tangent[0]) * base + float(normal[0]) * float(off) * 2.0
            cc = float(center_rc[1]) + sign * float(tangent[1]) * base + float(normal[1]) * float(off) * 2.0
            row, col = int(round(rr)), int(round(cc))
            if in_bounds(row, col, free.shape) and free[row, col]:
                out.append((row, col))
    return left, right


def _valid_side_areas(
    free: np.ndarray,
    cut_cells: list[tuple[int, int]],
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
    config: BottleneckConfig,
    resolution_m: float,
) -> bool:
    blocked = np.asarray(free, dtype=bool).copy()
    for cell in cut_cells:
        if in_bounds(cell[0], cell[1], blocked.shape):
            blocked[cell] = False
    a = flood_fill(blocked, left, connectivity=4)
    b = flood_fill(blocked, right, connectivity=4)
    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    if area_a <= 0 or area_b <= 0:
        return False
    min_cells = max(1, int(round(float(config.min_area_each_side_m2) / max(float(resolution_m) ** 2, 1e-9))))
    total = max(1, int(np.count_nonzero(free)))
    return area_a >= min_cells and area_b >= min_cells and min(area_a, area_b) / float(total) >= float(config.min_area_ratio_after_cut)


def _frontier_ratio(cut_cells: list[tuple[int, int]], frontier: np.ndarray, config: BottleneckConfig, resolution_m: float) -> float:
    if not cut_cells:
        return 0.0
    mask = np.zeros(frontier.shape, dtype=bool)
    for cell in cut_cells:
        if in_bounds(cell[0], cell[1], frontier.shape):
            mask[cell] = True
    band = dilate(mask, radius_cells(float(config.frontier_penalty_band_m), resolution_m))
    total = int(np.count_nonzero(band))
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(band & np.asarray(frontier, dtype=bool))) / float(total)


def _graph_conductance_and_area(
    candidate: CutCandidate,
    free_space: FreeSpaceState,
    resolution_m: float,
    config: BottleneckConfig,
) -> tuple[float, float, float]:
    free = np.asarray(free_space.observed_free_mask, dtype=bool)
    blocked = free.copy()
    for cell in candidate.cut_cells:
        if in_bounds(cell[0], cell[1], blocked.shape):
            blocked[cell] = False
    a = flood_fill(blocked, candidate.left_seed_cells, connectivity=4)
    b = flood_fill(blocked, candidate.right_seed_cells, connectivity=4)
    area_a = int(np.count_nonzero(a))
    area_b = int(np.count_nonzero(b))
    if area_a <= 0 or area_b <= 0:
        return 0.0, 0.0, 1.0
    boundary_edges = 0
    cut_mask = np.zeros_like(free, dtype=bool)
    for cell in candidate.cut_cells:
        if in_bounds(cell[0], cell[1], cut_mask.shape):
            cut_mask[cell] = True
    for row, col in zip(*np.nonzero(dilate(cut_mask, 1))):
        neigh_labels = set()
        for dr, dc in neighbor_offsets(4):
            nr, nc = int(row) + dr, int(col) + dc
            if in_bounds(nr, nc, free.shape):
                if a[nr, nc]:
                    neigh_labels.add("a")
                if b[nr, nc]:
                    neigh_labels.add("b")
        if len(neigh_labels) >= 2:
            boundary_edges += 1
    conductance = float(boundary_edges) / float(max(1, min(area_a, area_b)))
    graph_score = clamp(1.0 - conductance / 0.15)
    area_balance = clamp(min(area_a, area_b) / max(area_a, area_b))
    min_area_m2 = float(min(area_a, area_b)) * float(resolution_m) ** 2
    alcove = 1.0 if min_area_m2 < max(0.60, float(config.min_area_each_side_m2)) else 0.0
    return graph_score, area_balance, alcove


def _alcove_penalty(candidate: CutCandidate, graph: SkeletonGraph, free_space: FreeSpaceState, resolution_m: float) -> float:
    _ = graph
    if candidate.width_m < 0.40:
        return 0.5
    if _frontier_ratio(candidate.cut_cells, free_space.frontier_mask, BottleneckConfig(), resolution_m) > 0.65:
        return 0.8
    return 0.0


def _wall_endpoints(hard_wall: np.ndarray, max_points: int) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    labels, count = label_components(hard_wall, connectivity=8)
    for label in range(1, int(count) + 1):
        rows, cols = np.nonzero(labels == label)
        if rows.size == 0:
            continue
        coords = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=1)
        if coords.shape[0] == 1:
            endpoints.append((int(rows[0]), int(cols[0])))
            continue
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        try:
            vals, vecs = np.linalg.eigh(np.cov(centered.T))
            axis = vecs[:, int(np.argmax(vals))]
        except np.linalg.LinAlgError:
            axis = np.asarray([1.0, 0.0], dtype=np.float32)
        proj = coords @ axis
        for idx in (int(np.argmin(proj)), int(np.argmax(proj))):
            endpoints.append((int(round(float(coords[idx, 0]))), int(round(float(coords[idx, 1])))))
    if len(endpoints) > max_points:
        idx = np.linspace(0, len(endpoints) - 1, max_points).astype(np.int64)
        endpoints = [endpoints[int(i)] for i in idx]
    return endpoints


def _deduplicate_candidates(candidates: list[CutCandidate], config: BottleneckConfig) -> list[CutCandidate]:
    out: list[CutCandidate] = []
    min_dist_cells = max(2, int(round(0.25 / 0.05)))
    for cand in candidates:
        duplicate = False
        for kept in out:
            if np.hypot(cand.center_uv[0] - kept.center_uv[0], cand.center_uv[1] - kept.center_uv[1]) <= min_dist_cells:
                duplicate = True
                break
        if not duplicate:
            out.append(cand)
    return out[: max(1, int(config.max_candidates))]
