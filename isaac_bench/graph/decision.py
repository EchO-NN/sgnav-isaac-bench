from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.perception.object_memory import ObjectMemory, ObjectNode


@dataclass
class DecisionResult:
    selected_index: Optional[int]
    selected_frontier: Optional[FrontierCluster]
    scenegraph_scores: List[float]
    distance_scores: List[float]
    total_scores: List[float]
    reason: str


@dataclass
class NavigationDecision:
    mode: str
    target_cells: List[Tuple[int, int]]
    stop: bool
    selected_candidate: Optional[ObjectNode]
    frontier_decision: Optional[DecisionResult]
    reason: str


class SGNavDecision:
    def __init__(
        self,
        scenegraph: SGNavSceneGraphAdapter,
        frontier_distance_weight: float = 2.0,
        candidate_min_hits: int = 2,
        candidate_start_min_confidence: float = 0.20,
        candidate_stop_distance_m: float = 1.0,
        candidate_standoff_min_m: float = 0.65,
        candidate_standoff_max_m: float = 1.80,
    ):
        self.scenegraph = scenegraph
        self.frontier_distance_weight = float(frontier_distance_weight)
        self.candidate_min_hits = max(1, int(candidate_min_hits))
        self.candidate_start_min_confidence = float(candidate_start_min_confidence)
        self.candidate_stop_distance_m = float(candidate_stop_distance_m)
        self.candidate_standoff_min_m = float(candidate_standoff_min_m)
        self.candidate_standoff_max_m = max(float(candidate_standoff_max_m), self.candidate_standoff_min_m)

    def choose_frontier(self, frontier_clusters: List[FrontierCluster]) -> DecisionResult:
        if not frontier_clusters:
            return DecisionResult(None, None, [], [], [], "no_frontiers")
        locs = np.asarray([f.center_grid for f in frontier_clusters], dtype=np.int32)
        sg_scores = self.scenegraph.score(locs, len(frontier_clusters))
        dists = np.asarray([f.path_distance_from_agent for f in frontier_clusters], dtype=np.float32)
        if len(dists) == 0:
            dist_scores = np.zeros((0,), dtype=np.float32)
        else:
            clipped = np.clip(dists, 1.6, 11.6)
            dist_scores = 1.0 - (clipped - 1.6) / 10.0
        total = sg_scores + self.frontier_distance_weight * dist_scores
        idx = int(np.argmax(total))
        return DecisionResult(
            idx,
            frontier_clusters[idx],
            [float(x) for x in sg_scores],
            [float(x) for x in dist_scores],
            [float(x) for x in total],
            "selected_frontier",
        )

    def choose_navigation_target(
        self,
        object_memory: ObjectMemory,
        goal_category: str,
        current_grid: Tuple[int, int],
        frontiers: List[FrontierCluster],
        planner: GridAStarPlanner,
        map_info: MapInfo,
        current_pose: Sequence[float],
        allow_frontier: bool = True,
    ) -> NavigationDecision:
        candidate = self.select_goal_candidate(object_memory, goal_category, current_pose)
        if candidate is not None:
            distance_to_candidate = float(
                np.linalg.norm(
                    np.asarray(candidate.center_world[:2], dtype=np.float32)
                    - np.asarray([float(current_pose[0]), float(current_pose[1])], dtype=np.float32)
                )
            )
            if candidate.observed_count >= self.candidate_min_hits and distance_to_candidate <= self.candidate_stop_distance_m:
                return NavigationDecision("stop", [], True, candidate, None, "candidate_verified_stop")
            standoff = self.candidate_standoff_cells(candidate, planner.traversible, map_info)
            if standoff:
                return NavigationDecision("candidate", standoff, False, candidate, None, "navigate_to_goal_candidate")

        if allow_frontier:
            frontier_decision = self.choose_frontier(frontiers)
            if frontier_decision.selected_frontier is not None:
                return NavigationDecision(
                    "frontier",
                    list(frontier_decision.selected_frontier.members),
                    False,
                    None,
                    frontier_decision,
                    frontier_decision.reason,
                )
            return NavigationDecision("none", [], False, None, frontier_decision, frontier_decision.reason)
        return NavigationDecision("none", [], False, None, None, "no_candidate")

    def select_goal_candidate(
        self,
        object_memory: ObjectMemory,
        goal_category: str,
        current_pose: Sequence[float],
    ) -> Optional[ObjectNode]:
        goal = normalize_category(goal_category)
        candidates = []
        agent_xy = np.asarray([float(current_pose[0]), float(current_pose[1])], dtype=np.float32)
        for node in object_memory.nodes:
            cat = normalize_category(node.category)
            if cat != goal and not (goal and (goal in cat or cat in goal)):
                continue
            if float(node.confidence) < self.candidate_start_min_confidence:
                continue
            dist = float(np.linalg.norm(np.asarray(node.center_world[:2], dtype=np.float32) - agent_xy))
            score = float(node.confidence) + 0.25 * min(float(node.observed_count), 5.0) - 0.05 * dist
            candidates.append((score, node))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def candidate_standoff_cells(
        self,
        candidate: ObjectNode,
        traversible: np.ndarray,
        map_info: MapInfo,
    ) -> List[Tuple[int, int]]:
        cx, cy = float(candidate.center_world[0]), float(candidate.center_world[1])
        max_radius_cells = int(np.ceil(self.candidate_standoff_max_m / map_info.resolution_m))
        r0, c0 = int(candidate.center_grid[0]), int(candidate.center_grid[1])
        cells: List[Tuple[int, int]] = []
        nearest: List[Tuple[float, Tuple[int, int]]] = []
        h, w = traversible.shape
        for r in range(max(0, r0 - max_radius_cells), min(h, r0 + max_radius_cells + 1)):
            for c in range(max(0, c0 - max_radius_cells), min(w, c0 + max_radius_cells + 1)):
                if not bool(traversible[r, c]):
                    continue
                wx, wy = grid_to_world_xy(r, c, map_info)
                dist = float(np.hypot(wx - cx, wy - cy))
                nearest.append((dist, (int(r), int(c))))
                if self.candidate_standoff_min_m <= dist <= self.candidate_standoff_max_m:
                    cells.append((int(r), int(c)))
        if cells:
            return cells
        nearest.sort(key=lambda item: item[0])
        return [cell for _dist, cell in nearest[:16]]
