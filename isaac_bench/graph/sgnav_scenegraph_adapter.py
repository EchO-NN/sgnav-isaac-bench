from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.perception.object_memory import ObjectMemory


SGNAV_ROOM_NAMES = [
    "bedroom",
    "living room",
    "bathroom",
    "kitchen",
    "dining room",
    "office room",
    "gym",
    "lounge",
    "laundry room",
]


RELATED_CATEGORY_PAIRS = {
    tuple(sorted(pair))
    for pair in [
        ("bed", "nightstand"),
        ("wardrobe", "dresser"),
        ("bookshelf", "chair"),
        ("counter", "stove"),
        ("table", "chair"),
        ("sink", "mirror"),
        ("shower", "bathtub"),
        ("refrigerator", "freezer"),
        ("oven", "microwave"),
        ("washing machine", "dryer"),
        ("sofa", "table"),
        ("desk", "chair"),
        ("computer", "tv"),
        ("tv", "sofa"),
        ("tv", "table"),
        ("vase", "table"),
        ("lamp", "table"),
    ]
}


GOAL_ROOM_PRIORS: Dict[str, Tuple[str, ...]] = {
    "bed": ("bedroom",),
    "nightstand": ("bedroom",),
    "dresser": ("bedroom",),
    "wardrobe": ("bedroom",),
    "sofa": ("living room", "lounge"),
    "tv": ("living room", "lounge", "bedroom"),
    "table": ("dining room", "living room", "kitchen", "office room"),
    "chair": ("dining room", "living room", "office room"),
    "desk": ("office room", "bedroom"),
    "bookshelf": ("office room", "living room", "bedroom"),
    "sink": ("bathroom", "kitchen"),
    "toilet": ("bathroom",),
    "bathtub": ("bathroom",),
    "shower": ("bathroom",),
    "refrigerator": ("kitchen",),
    "stove": ("kitchen",),
    "oven": ("kitchen",),
    "microwave": ("kitchen",),
    "cabinet": ("kitchen", "bathroom", "bedroom"),
    "vase": ("living room", "dining room", "bedroom"),
    "lamp": ("living room", "bedroom", "office room"),
}


class SGNavSceneGraphAdapter:
    """Thin adapter around the original SG-Nav SceneGraph when available.

    The original SceneGraph has heavy model dependencies, so this wrapper falls
    back to a deterministic lightweight scorer if importing/initializing it is
    not possible.
    """

    def __init__(self, sgnav_repo: str = "/home/echo/SG-Nav", use_original: bool = False):
        self.sgnav_repo = sgnav_repo
        self.use_original = bool(use_original)
        self.scenegraph = None
        self.obj_goal_sg = ""
        self.object_memory: Optional[ObjectMemory] = None
        self.room_map = None
        self.room_names = list(SGNAV_ROOM_NAMES)
        self.last_score_debug: Dict[str, object] = {}
        if self.use_original:
            self._try_init_original()

    @contextmanager
    def _sgnav_cwd(self):
        old_cwd = os.getcwd()
        try:
            if self.sgnav_repo and os.path.isdir(self.sgnav_repo):
                os.chdir(self.sgnav_repo)
            yield
        finally:
            os.chdir(old_cwd)

    def _try_init_original(self) -> None:
        if self.sgnav_repo and self.sgnav_repo not in sys.path:
            sys.path.insert(0, self.sgnav_repo)
        try:
            with self._sgnav_cwd():
                from scenegraph import SceneGraph  # type: ignore

                camera_matrix = np.eye(3, dtype=np.float32)
                agent_stub = SimpleNamespace(
                    args=SimpleNamespace(
                        sgnav_score_mode="paper_object",
                        frontier_score_norm="weighted_mean",
                        disable_llm_edges=True,
                    )
                )
                self.scenegraph = SceneGraph(
                    map_resolution=5,
                    map_size_cm=4000,
                    map_size=800,
                    camera_matrix=camera_matrix,
                    is_navigation=True,
                    agent=agent_stub,
                )
        except Exception as exc:
            print("[sgnav-adapter] original SceneGraph unavailable, using fallback: %s" % exc, flush=True)
            self.scenegraph = None

    def reset(self, goal_category: str) -> None:
        self.obj_goal_sg = normalize_category(str(goal_category))
        if self.scenegraph is not None:
            try:
                with self._sgnav_cwd():
                    self.scenegraph.reset()
                    self.scenegraph.set_obj_goal(self.obj_goal_sg, self.obj_goal_sg)
            except Exception:
                pass

    def update(self, object_memory: ObjectMemory, room_map=None) -> None:
        self.object_memory = object_memory
        self.room_map = room_map
        if self.scenegraph is not None:
            try:
                with self._sgnav_cwd():
                    self.scenegraph.set_room_map(room_map)
                    self._sync_original_nodes()
            except Exception:
                pass

    def score(self, frontier_locations: np.ndarray, num_frontiers: int) -> np.ndarray:
        if num_frontiers <= 0:
            return np.zeros((0,), dtype=np.float32)
        if self.scenegraph is not None:
            try:
                with self._sgnav_cwd():
                    scores = np.asarray(self.scenegraph.score(frontier_locations, num_frontiers), dtype=np.float32)
                    self.last_score_debug = dict(getattr(self.scenegraph, "last_score_debug", {}) or {})
                    return scores
            except Exception as exc:
                print("[sgnav-adapter] original score failed, using fallback: %s" % exc, flush=True)
        return self._fallback_score(frontier_locations, num_frontiers)

    def _sync_original_nodes(self) -> None:
        if self.scenegraph is None or self.object_memory is None:
            return
        try:
            from scenegraph import ObjectNode  # type: ignore
        except Exception:
            return
        nodes = []
        for mem_node in self.object_memory.nodes:
            node = ObjectNode()
            node.set_caption(normalize_category(mem_node.category))
            node.set_center([int(mem_node.center_grid[0]), int(mem_node.center_grid[1])])
            node.is_new_node = False
            node.is_goal_node = self._category_matches_goal(mem_node.category)
            node.score = float(mem_node.confidence)
            node.object = {
                "conf": [float(mem_node.confidence)],
                "image_idx": [],
                "mask_idx": [],
                "num_detections": int(mem_node.observed_count),
                "center_world": list(mem_node.center_world),
            }
            room_name = self.room_name_at_grid(mem_node.center_grid)
            room_node = self._original_room_node(room_name)
            if room_node is not None:
                self.scenegraph.set_node_room(node, room_node, 1.0, 1.0)
            nodes.append(node)
        self.scenegraph.nodes = nodes
        self.scenegraph._subgraph_score_cache_by_key.clear()
        self.scenegraph._subgraph_score_cache = []
        self.scenegraph._subgraph_score_cache_key = None

    def _original_room_node(self, room_name: Optional[str]):
        if not room_name or self.scenegraph is None:
            return None
        target = normalize_category(room_name).replace("_", " ")
        for room_node in getattr(self.scenegraph, "room_nodes", []):
            if normalize_category(getattr(room_node, "caption", "")) == target:
                return room_node
        return None

    def _fallback_score(self, frontier_locations: np.ndarray, num_frontiers: int) -> np.ndarray:
        scores = np.zeros((num_frontiers,), dtype=np.float32)
        if self.object_memory is None or len(self.object_memory.nodes) == 0:
            self.last_score_debug = {"mode": "fallback", "num_objects": 0}
            return scores
        goal_nodes = [node for node in self.object_memory.nodes if self._category_matches_goal(node.category)]
        context_nodes = goal_nodes or self.object_memory.nodes
        for idx, frontier in enumerate(frontier_locations[:num_frontiers]):
            best = 0.0
            for node in context_nodes:
                center = np.asarray(node.center_grid, dtype=np.float32)
                dist = float(np.linalg.norm(np.asarray(frontier, dtype=np.float32) - center))
                related = self._node_relevance(node)
                best = max(best, related * float(node.confidence) * math.log1p(node.observed_count) / max(dist, 1.0))
            best += self._room_prior_score(tuple(int(v) for v in frontier))
            scores[idx] = best
        self.last_score_debug = {
            "mode": "fallback",
            "num_objects": len(self.object_memory.nodes),
            "num_goal_nodes": len(goal_nodes),
            "frontier_score_min": float(np.min(scores)) if len(scores) else 0.0,
            "frontier_score_max": float(np.max(scores)) if len(scores) else 0.0,
            "frontier_score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        }
        return scores

    def _category_matches_goal(self, category: str) -> bool:
        cat = normalize_category(category)
        goal = normalize_category(self.obj_goal_sg)
        return cat == goal or (goal and (goal in cat or cat in goal))

    def _node_relevance(self, node) -> float:
        if self._category_matches_goal(node.category):
            return 8.0
        cat = normalize_category(node.category)
        goal = normalize_category(self.obj_goal_sg)
        if tuple(sorted((cat, goal))) in RELATED_CATEGORY_PAIRS:
            return 2.0
        room_name = self.room_name_at_grid(node.center_grid)
        if room_name and normalize_category(room_name).replace("_", " ") in GOAL_ROOM_PRIORS.get(goal, ()):
            return 1.25
        return 0.35

    def room_name_at_grid(self, grid: Sequence[int]) -> Optional[str]:
        if self.room_map is None:
            return None
        arr = np.asarray(self.room_map)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[0] == 0:
            return None
        r, c = int(grid[0]), int(grid[1])
        if not (0 <= r < arr.shape[1] and 0 <= c < arr.shape[2]):
            return None
        idx = int(np.argmax(arr[:, r, c]))
        if float(arr[idx, r, c]) <= 0.0 or idx >= len(self.room_names):
            return None
        return self.room_names[idx]

    def _room_prior_score(self, frontier: Tuple[int, int]) -> float:
        goal = normalize_category(self.obj_goal_sg)
        priors = GOAL_ROOM_PRIORS.get(goal, ())
        if not priors:
            return 0.0
        room_name = self.room_name_at_grid(frontier)
        if room_name is None:
            return 0.0
        return 0.6 if normalize_category(room_name).replace("_", " ") in priors else 0.0
