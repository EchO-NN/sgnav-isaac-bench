from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.astar import GridAStarPlanner, astar_distance_map
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
    state: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


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
        reperception_enabled: bool = True,
        reperception_min_observations: int = 3,
        reperception_max_steps: int = 10,
        reperception_same_goal_radius_m: float = 0.8,
        stop_verification_steps: int = 4,
        stop_verification_min_hits: int = 2,
        found_goal_stop_distance_m: float = 0.35,
        score_frontiers_before_candidate: bool = False,
    ):
        self.scenegraph = scenegraph
        self.frontier_distance_weight = float(frontier_distance_weight)
        self.candidate_min_hits = max(1, int(candidate_min_hits))
        self.candidate_start_min_confidence = float(candidate_start_min_confidence)
        self.candidate_stop_distance_m = float(candidate_stop_distance_m)
        self.candidate_standoff_min_m = float(candidate_standoff_min_m)
        self.candidate_standoff_max_m = max(float(candidate_standoff_max_m), self.candidate_standoff_min_m)
        self.reperception_enabled = bool(reperception_enabled)
        self.reperception_min_observations = max(1, int(reperception_min_observations))
        self.reperception_max_steps = max(0, int(reperception_max_steps))
        self.reperception_same_goal_radius_m = max(0.05, float(reperception_same_goal_radius_m))
        self.stop_verification_steps = max(0, int(stop_verification_steps))
        self.stop_verification_min_hits = max(1, int(stop_verification_min_hits))
        self.found_goal_stop_distance_m = max(0.05, float(found_goal_stop_distance_m))
        self.score_frontiers_before_candidate = bool(score_frontiers_before_candidate)
        self.state = "frontier"
        self.reperception_candidate_id: Optional[int] = None
        self.reperception_steps = 0
        self.stop_candidate_id: Optional[int] = None
        self.stop_verification_steps_taken = 0
        self.last_reason = ""

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
        prefetched_frontier_decision: Optional[DecisionResult] = None
        if allow_frontier and self.score_frontiers_before_candidate and frontiers:
            prefetched_frontier_decision = self.choose_frontier(frontiers)

        candidate = self.select_goal_candidate(object_memory, goal_category, current_pose)
        if candidate is not None:
            distance_to_candidate = float(
                np.linalg.norm(
                    np.asarray(candidate.center_world[:2], dtype=np.float32)
                    - np.asarray([float(current_pose[0]), float(current_pose[1])], dtype=np.float32)
                )
            )
            close_for_stop = distance_to_candidate <= self.candidate_stop_distance_m
            very_close = distance_to_candidate <= self.found_goal_stop_distance_m
            enough_candidate_hits = candidate.observed_count >= self.candidate_min_hits
            enough_verification_hits = candidate.observed_count >= max(self.candidate_min_hits, self.stop_verification_min_hits)
            if close_for_stop and enough_candidate_hits:
                if self.stop_candidate_id != candidate.node_id:
                    self.stop_candidate_id = candidate.node_id
                    self.stop_verification_steps_taken = 0
                if (
                    self.stop_verification_steps <= 0
                    or self.stop_verification_steps_taken >= self.stop_verification_steps
                    or (very_close and enough_verification_hits)
                ):
                    self.state = "stop"
                    self.last_reason = "stop_verification_confirmed"
                    return NavigationDecision(
                        "stop",
                        [],
                        True,
                        candidate,
                        prefetched_frontier_decision,
                        "stop_verification_confirmed",
                        state=self.state,
                        metadata={
                            "candidate_distance_m": distance_to_candidate,
                            "candidate_observed_count": int(candidate.observed_count),
                            "stop_verification_steps_taken": int(self.stop_verification_steps_taken),
                        },
                    )
                self.stop_verification_steps_taken += 1
                self.state = "stop_verification"
                self.last_reason = "stop_verification_scanning"
                return NavigationDecision(
                    "reperception",
                    [current_grid],
                    False,
                    candidate,
                    prefetched_frontier_decision,
                    "stop_verification_scanning",
                    state=self.state,
                    metadata={
                        "candidate_distance_m": distance_to_candidate,
                        "candidate_observed_count": int(candidate.observed_count),
                        "stop_verification_steps_taken": int(self.stop_verification_steps_taken),
                    },
                )
            if (
                self.reperception_enabled
                and distance_to_candidate <= self.reperception_same_goal_radius_m
                and candidate.observed_count < self.reperception_min_observations
                and self.reperception_steps < self.reperception_max_steps
            ):
                if self.reperception_candidate_id != candidate.node_id:
                    self.reperception_candidate_id = candidate.node_id
                    self.reperception_steps = 0
                self.reperception_steps += 1
                self.state = "reperception"
                self.last_reason = "reperception_scanning"
                return NavigationDecision(
                    "reperception",
                    [current_grid],
                    False,
                    candidate,
                    prefetched_frontier_decision,
                    "reperception_scanning",
                    state=self.state,
                    metadata={
                        "candidate_distance_m": distance_to_candidate,
                        "candidate_observed_count": int(candidate.observed_count),
                        "reperception_steps": int(self.reperception_steps),
                    },
                )
            standoff = self.candidate_standoff_cells(candidate, planner.traversible, map_info, current_grid=current_grid)
            standoff = self._targets_with_min_progress(standoff, planner, current_grid, map_info)
            if standoff:
                self.state = "candidate"
                self.last_reason = "navigate_to_goal_candidate"
                return NavigationDecision(
                    "candidate",
                    standoff,
                    False,
                    candidate,
                    prefetched_frontier_decision,
                    "navigate_to_goal_candidate",
                    state=self.state,
                    metadata={
                        "candidate_distance_m": distance_to_candidate,
                        "candidate_observed_count": int(candidate.observed_count),
                    },
                )

        if allow_frontier:
            self.reperception_candidate_id = None
            self.reperception_steps = 0
            self.stop_candidate_id = None
            self.stop_verification_steps_taken = 0
            frontier_decision = prefetched_frontier_decision or self.choose_frontier(frontiers)
            if frontier_decision.selected_frontier is not None:
                self.state = "frontier"
                self.last_reason = frontier_decision.reason
                return NavigationDecision(
                    "frontier",
                    list(frontier_decision.selected_frontier.members),
                    False,
                    None,
                    frontier_decision,
                    frontier_decision.reason,
                    state=self.state,
                )
            self.state = "none"
            self.last_reason = frontier_decision.reason
            return NavigationDecision("none", [], False, None, frontier_decision, frontier_decision.reason, state=self.state)
        self.state = "none"
        self.last_reason = "no_candidate"
        return NavigationDecision("none", [], False, None, None, "no_candidate", state=self.state)

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
        current_grid: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        cx, cy = float(candidate.center_world[0]), float(candidate.center_world[1])
        max_radius_cells = int(np.ceil(self.candidate_standoff_max_m / map_info.resolution_m))
        r0, c0 = int(candidate.center_grid[0]), int(candidate.center_grid[1])
        traversible_bool = np.asarray(traversible).astype(bool)
        reachable = traversible_bool
        if current_grid is not None:
            dist_map = astar_distance_map(traversible_bool, current_grid, map_info.resolution_m, allow_diagonal=True)
            reachable = np.isfinite(dist_map)
            if not np.any(reachable):
                return []
        cells: List[Tuple[int, int]] = []
        nearest: List[Tuple[float, Tuple[int, int]]] = []
        h, w = traversible_bool.shape
        for r in range(max(0, r0 - max_radius_cells), min(h, r0 + max_radius_cells + 1)):
            for c in range(max(0, c0 - max_radius_cells), min(w, c0 + max_radius_cells + 1)):
                if not bool(reachable[r, c]):
                    continue
                wx, wy = grid_to_world_xy(r, c, map_info)
                dist = float(np.hypot(wx - cx, wy - cy))
                nearest.append((dist, (int(r), int(c))))
                if self.candidate_standoff_min_m <= dist <= self.candidate_standoff_max_m:
                    cells.append((int(r), int(c)))
        if cells:
            return cells
        nearest.sort(key=lambda item: item[0])
        if nearest:
            return [cell for _dist, cell in nearest[:16]]
        return self.nearest_reachable_cells_to_candidate(candidate, traversible, map_info, current_grid=current_grid)

    def nearest_reachable_cells_to_candidate(
        self,
        candidate: ObjectNode,
        traversible: np.ndarray,
        map_info: MapInfo,
        current_grid: Optional[Tuple[int, int]] = None,
        max_cells: int = 16,
    ) -> List[Tuple[int, int]]:
        traversible_bool = np.asarray(traversible).astype(bool)
        rows, cols = np.nonzero(traversible_bool)
        if rows.size == 0:
            return []
        reachable = traversible_bool
        if current_grid is not None:
            dist_map = astar_distance_map(traversible_bool, current_grid, map_info.resolution_m, allow_diagonal=True)
            reachable = np.isfinite(dist_map)
            rows, cols = np.nonzero(reachable)
            if rows.size == 0:
                return []
        cx, cy = float(candidate.center_world[0]), float(candidate.center_world[1])
        ranked: List[Tuple[float, Tuple[int, int]]] = []
        current = (int(current_grid[0]), int(current_grid[1])) if current_grid is not None else None
        for row, col in zip(rows, cols):
            cell = (int(row), int(col))
            wx, wy = grid_to_world_xy(cell[0], cell[1], map_info)
            candidate_dist = float(np.hypot(wx - cx, wy - cy))
            current_penalty = 0.0
            if current is not None and cell == current:
                current_penalty = 1e3
            ranked.append((candidate_dist + current_penalty, cell))
        ranked.sort(key=lambda item: item[0])
        return [cell for _dist, cell in ranked[: max(1, int(max_cells))]]

    @staticmethod
    def _targets_with_min_progress(
        targets: Sequence[Tuple[int, int]],
        planner: GridAStarPlanner,
        current_grid: Tuple[int, int],
        map_info: MapInfo,
    ) -> List[Tuple[int, int]]:
        if not targets:
            return []
        min_progress_m = max(float(map_info.resolution_m) * 3.0, 0.25)
        dist_map = astar_distance_map(planner.traversible, current_grid, map_info.resolution_m, allow_diagonal=planner.allow_diagonal)
        filtered: List[Tuple[int, int]] = []
        seen = set()
        for target in targets:
            cell = (int(target[0]), int(target[1]))
            if cell in seen or not planner.inside(cell):
                continue
            seen.add(cell)
            dist = float(dist_map[cell])
            if np.isfinite(dist) and dist >= min_progress_m:
                filtered.append(cell)
        return filtered
