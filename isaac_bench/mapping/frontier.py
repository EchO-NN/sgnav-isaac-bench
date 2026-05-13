from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.navigation.astar import GridAStarPlanner

GridCell = Tuple[int, int]


@dataclass
class FrontierCluster:
    center_grid: GridCell
    center_world: Tuple[float, float]
    members: List[GridCell]
    size: int
    path_distance_from_agent: float


def frontier_cells(free: np.ndarray, observed: np.ndarray) -> np.ndarray:
    free = free.astype(bool)
    observed = observed.astype(bool)
    unknown = ~observed.astype(bool)
    out = np.zeros_like(free, dtype=bool)
    h, w = free.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.zeros_like(free, dtype=bool)
        r0s, r1s = max(0, dr), h + min(0, dr)
        c0s, c1s = max(0, dc), w + min(0, dc)
        r0d, r1d = max(0, -dr), h + min(0, -dr)
        c0d, c1d = max(0, -dc), w + min(0, -dc)
        shifted[r0d:r1d, c0d:c1d] = unknown[r0s:r1s, c0s:c1s]
        out |= free & observed & shifted
    return out


def connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[GridCell]] = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        comp: List[GridCell] = []
        while q:
            cell = q.popleft()
            comp.append(cell)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = cell[0] + dr, cell[1] + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def extract_frontiers(
    free: np.ndarray,
    observed: np.ndarray,
    traversible: np.ndarray,
    map_info: MapInfo,
    agent_grid: GridCell,
    min_cluster_size: int = 3,
    min_distance_m: float = 1.0,
    max_count: int = 64,
) -> List[FrontierCluster]:
    cells = frontier_cells(free, observed)
    planner = GridAStarPlanner(traversible, map_info.resolution_m, allow_diagonal=True)
    clusters: List[FrontierCluster] = []
    for comp in connected_components(cells):
        if len(comp) < min_cluster_size:
            continue
        arr = np.asarray(comp, dtype=np.float32)
        center = tuple(int(round(v)) for v in arr.mean(axis=0))
        dist = planner.distance(agent_grid, comp)
        if not np.isfinite(dist) or dist < min_distance_m:
            continue
        wx, wy = grid_to_world_xy(center[0], center[1], map_info)
        clusters.append(FrontierCluster(center, (wx, wy), comp, len(comp), float(dist)))
    clusters.sort(key=lambda f: (-f.size, f.path_distance_from_agent))
    return clusters[:max_count]
