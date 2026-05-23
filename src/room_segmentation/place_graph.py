from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .config import PlaceGraphConfig
from .data_types import CutCandidate, GridSpec, PlaceEdge, PlaceGraph, PlaceNode, SkeletonGraph, StructuralMap
from .utils import grid_to_world, in_bounds, path_crosses
from .visibility import VisibilityComputer, jaccard


class PlaceGraphBuilder:
    def __init__(
        self,
        config: PlaceGraphConfig | Mapping[str, object] | None = None,
        grid_spec: GridSpec | None = None,
        visibility: VisibilityComputer | None = None,
    ):
        self.config = config if isinstance(config, PlaceGraphConfig) else PlaceGraphConfig.from_mapping(config or {})
        self.grid_spec = grid_spec
        self.visibility = visibility or VisibilityComputer(resolution_m=self._resolution())
        self._keyframes: list[tuple[float, float, float, tuple[int, int]]] = []

    def reset(self) -> None:
        self._keyframes.clear()

    def build(
        self,
        skeleton_graph: SkeletonGraph,
        cut_candidates: list[CutCandidate],
        structural_map: StructuralMap,
        semantic_objects: list[dict] | None = None,
        frame_id: int = 0,
        camera_pose_world: np.ndarray | None = None,
    ) -> PlaceGraph:
        nodes: list[PlaceNode] = []
        self._maybe_add_keyframe(camera_pose_world, structural_map.p_wall.shape)
        for pose_idx, (_x, _y, _yaw, uv) in enumerate(self._keyframes):
            xy = grid_to_world(uv[0], uv[1], self.grid_spec) if self.grid_spec is not None else (float(uv[1]), float(uv[0]))
            nodes.append(PlaceNode(node_id=len(nodes), node_type="keyframe", xy=xy, uv=uv))
        stride_cells = max(1, int(round(float(self.config.skeleton_node_stride_m) / max(self._resolution(), 1e-9))))
        for idx, node in enumerate(skeleton_graph.nodes):
            if idx % stride_cells != 0:
                continue
            nodes.append(PlaceNode(node_id=len(nodes), node_type="skeleton", xy=node.xy, uv=node.uv))
        for uv in _room_centers(skeleton_graph.distance_transform_m, structural_map.observed_free_mask, self.config, self._resolution()):
            xy = grid_to_world(uv[0], uv[1], self.grid_spec) if self.grid_spec is not None else (float(uv[1]), float(uv[0]))
            nodes.append(PlaceNode(node_id=len(nodes), node_type="room_center", xy=xy, uv=uv))
        if not nodes and np.any(structural_map.observed_free_mask):
            rows, cols = np.nonzero(structural_map.observed_free_mask)
            uv = (int(np.mean(rows)), int(np.mean(cols)))
            xy = grid_to_world(uv[0], uv[1], self.grid_spec) if self.grid_spec is not None else (float(uv[1]), float(uv[0]))
            nodes.append(PlaceNode(node_id=0, node_type="room_center", xy=xy, uv=uv))
        cut_score_map = np.zeros_like(structural_map.p_wall, dtype=np.float32)
        for candidate in cut_candidates:
            for cell in candidate.cut_cells:
                if in_bounds(cell[0], cell[1], cut_score_map.shape):
                    cut_score_map[cell] = max(float(cut_score_map[cell]), float(candidate.final_score))
        edges: list[PlaceEdge] = []
        adjacency = {node.node_id: [] for node in nodes}
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                euclidean = _xy_distance(a.xy, b.xy)
                if euclidean > max(float(self.config.edge_max_geodesic_m), float(self.config.edge_max_euclidean_m)):
                    continue
                los = _line_of_sight(a.uv, b.uv, structural_map)
                if euclidean > float(self.config.edge_max_geodesic_m) and not (euclidean <= float(self.config.edge_max_euclidean_m) and los):
                    continue
                if bool(self.config.line_of_sight_required_for_short_edges) and not los:
                    continue
                bottleneck_penalty = _path_max(cut_score_map, a.uv, b.uv)
                frontier_penalty = _path_mean(structural_map.frontier_mask.astype(np.float32), a.uv, b.uv)
                if euclidean <= 1.75:
                    sig_a = self.visibility.compute_visibility_signature([a.uv], structural_map)
                    sig_b = self.visibility.compute_visibility_signature([b.uv], structural_map)
                    vis = jaccard(sig_a, sig_b)
                else:
                    vis = 0.5
                semantic = 0.5
                weight = (
                    float(self.config.weight_geodesic) * math.exp(-euclidean / max(float(self.config.edge_max_geodesic_m), 1e-6))
                    + float(self.config.weight_visibility_jaccard) * vis
                    + float(self.config.weight_line_of_sight) * (1.0 if los else 0.0)
                    + float(self.config.weight_semantic_similarity) * semantic
                    - float(self.config.weight_bottleneck_penalty) * bottleneck_penalty
                    - float(self.config.weight_frontier_penalty) * frontier_penalty
                )
                edge = PlaceEdge(
                    src=int(a.node_id),
                    dst=int(b.node_id),
                    weight=float(np.clip(weight, 0.0, 1.0)),
                    geodesic_m=float(euclidean),
                    visibility_jaccard=float(vis),
                    line_of_sight=float(1.0 if los else 0.0),
                    bottleneck_penalty=float(bottleneck_penalty),
                )
                edges.append(edge)
                adjacency[a.node_id].append(b.node_id)
                adjacency[b.node_id].append(a.node_id)
        return PlaceGraph(nodes=nodes, edges=edges, adjacency=adjacency)

    def _maybe_add_keyframe(self, camera_pose_world: np.ndarray | None, shape: tuple[int, int]) -> None:
        if camera_pose_world is None or self.grid_spec is None:
            return
        pose = np.asarray(camera_pose_world, dtype=np.float64)
        if pose.shape != (4, 4):
            return
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        yaw = float(math.atan2(pose[1, 0], pose[0, 0]))
        from .utils import world_to_grid

        uv = world_to_grid(x, y, self.grid_spec)
        if not in_bounds(uv[0], uv[1], shape):
            return
        if not self._keyframes:
            self._keyframes.append((x, y, yaw, uv))
            return
        px, py, pyaw, _ = self._keyframes[-1]
        if math.hypot(x - px, y - py) >= float(self.config.keyframe_interval_m) or abs(_angle_diff(yaw, pyaw)) >= float(self.config.keyframe_interval_rad):
            self._keyframes.append((x, y, yaw, uv))

    def _resolution(self) -> float:
        return float(self.grid_spec.resolution_m) if self.grid_spec is not None else 0.05


def _room_centers(distance_m: np.ndarray, free: np.ndarray, config: PlaceGraphConfig, resolution_m: float) -> list[tuple[int, int]]:
    dist = np.asarray(distance_m, dtype=np.float32)
    free_mask = np.asarray(free, dtype=bool)
    rows, cols = np.nonzero(free_mask)
    candidates = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        val = float(dist[int(row), int(col)])
        if val <= 0.0:
            continue
        r0, r1 = max(0, int(row) - 1), min(dist.shape[0], int(row) + 2)
        c0, c1 = max(0, int(col) - 1), min(dist.shape[1], int(col) + 2)
        if val >= float(np.max(dist[r0:r1, c0:c1])) - 1e-6:
            candidates.append((val, int(row), int(col)))
    candidates.sort(reverse=True)
    min_dist_cells = max(1, int(round(float(config.room_center_min_distance_m) / max(float(resolution_m), 1e-9))))
    centers: list[tuple[int, int]] = []
    for _val, row, col in candidates:
        if all(np.hypot(row - r, col - c) >= min_dist_cells for r, c in centers):
            centers.append((row, col))
        if len(centers) >= 128:
            break
    return centers


def _line_of_sight(a: tuple[int, int], b: tuple[int, int], structural_map: StructuralMap) -> bool:
    blocked = np.asarray(structural_map.hard_wall_mask, dtype=bool) | (np.asarray(structural_map.p_unknown, dtype=np.float32) >= 0.65)
    return not path_crosses(blocked, a, b)


def _path_max(values: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    from .utils import rasterize_line

    cells = rasterize_line(a, b, values.shape)
    if not cells:
        return 0.0
    return float(max(float(values[cell]) for cell in cells))


def _path_mean(values: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    from .utils import rasterize_line

    cells = rasterize_line(a, b, values.shape)
    if not cells:
        return 0.0
    return float(np.mean([float(values[cell]) for cell in cells]))


def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)

