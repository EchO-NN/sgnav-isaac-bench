from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy
from isaac_bench.navigation.astar import astar_distance_map

GridCell = Tuple[int, int]


@dataclass
class FrontierCluster:
    center_grid: GridCell
    center_world: Tuple[float, float]
    members: List[GridCell]
    size: int
    path_distance_from_agent: float


def _disk_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(mask_bool):
        return mask_bool
    try:
        import skimage.morphology

        return skimage.morphology.binary_dilation(mask_bool, skimage.morphology.disk(radius)).astype(bool)
    except Exception:
        out = np.array(mask_bool, copy=True)
        rows, cols = np.nonzero(mask_bool)
        h, w = mask_bool.shape
        offsets = [
            (dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius
        ]
        for row, col in zip(rows, cols):
            for dr, dc in offsets:
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
        return out


def frontier_cells(
    free: np.ndarray,
    observed: Optional[np.ndarray] = None,
    occupancy: Optional[np.ndarray] = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """SG-Nav FBE frontier map.

    Mirrors /home/echo/SG-Nav/SG_Nav.py::fbe:
      free cells are 1, obstacle-dilated cells are 3, unknown cells are 0;
      frontier cells are free cells intersecting the 1-cell dilation of unknown.
    `observed` is retained for older callers; the SG-Nav definition uses
    `free` and `occupancy`, where cells not in either are unknown.
    """
    free_bool = np.asarray(free).astype(bool)
    occ_bool = np.zeros_like(free_bool, dtype=bool) if occupancy is None else np.asarray(occupancy).astype(bool)
    if occ_bool.shape != free_bool.shape:
        raise ValueError("occupancy and free must have the same shape")

    fbe_map = np.zeros_like(free_bool, dtype=np.int8)
    fbe_map[free_bool] = 1
    dilated_obstacles = _disk_dilate(occ_bool, obstacle_dilation_radius_cells)
    fbe_map[dilated_obstacles] = 3

    unknown = fbe_map == 0
    unknown_dilated = _disk_dilate(unknown, unknown_dilation_radius_cells)
    fbe_cpp = np.array(fbe_map, copy=True)
    fbe_cpp[unknown_dilated] = 0
    frontiers = (fbe_map - fbe_cpp) == 1
    if exclude_mask is not None:
        excluded = np.asarray(exclude_mask).astype(bool)
        if excluded.shape != free_bool.shape:
            raise ValueError("exclude_mask and free must have the same shape")
        frontiers &= ~excluded
    return frontiers


def _connected_components(mask: np.ndarray) -> List[List[GridCell]]:
    src = np.asarray(mask).astype(bool)
    if not np.any(src):
        return []
    visited = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    components: List[List[GridCell]] = []
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for start_row, start_col in zip(*np.nonzero(src)):
        start = (int(start_row), int(start_col))
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[GridCell] = deque([start])
        members: List[GridCell] = []
        while queue:
            row, col = queue.popleft()
            members.append((row, col))
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if visited[rr, cc] or not src[rr, cc]:
                    continue
                visited[rr, cc] = True
                queue.append((rr, cc))
        components.append(members)
    return components


def extract_frontiers(
    free: np.ndarray,
    observed: np.ndarray,
    traversible: np.ndarray,
    map_info: MapInfo,
    agent_grid: GridCell,
    min_cluster_size: int = 3,
    min_distance_m: float = 1.0,
    max_count: int = 64,
    occupancy: Optional[np.ndarray] = None,
    obstacle_dilation_radius_cells: int = 4,
    unknown_dilation_radius_cells: int = 1,
    exclude_mask: Optional[np.ndarray] = None,
) -> List[FrontierCluster]:
    _ = observed
    cells = frontier_cells(
        free,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=obstacle_dilation_radius_cells,
        unknown_dilation_radius_cells=unknown_dilation_radius_cells,
        exclude_mask=exclude_mask,
    )
    dist_map = astar_distance_map(traversible, agent_grid, map_info.resolution_m, allow_diagonal=True)
    clusters: List[FrontierCluster] = []
    near_clusters: List[FrontierCluster] = []
    for members in _connected_components(cells):
        finite_members = [(row, col) for row, col in members if np.isfinite(float(dist_map[row, col]))]
        if len(finite_members) < max(1, int(min_cluster_size)):
            continue
        member_arr = np.asarray(finite_members, dtype=np.float32)
        centroid = np.mean(member_arr, axis=0)
        center_idx = int(np.argmin(np.sum((member_arr - centroid) ** 2, axis=1)))
        center = tuple(int(v) for v in member_arr[center_idx])
        dist = float(dist_map[center])
        wx, wy = grid_to_world_xy(center[0], center[1], map_info)
        cluster = FrontierCluster(center, (wx, wy), finite_members, len(finite_members), dist)
        if dist < min_distance_m:
            near_clusters.append(cluster)
            continue
        clusters.append(cluster)
    if not clusters and near_clusters:
        clusters = near_clusters
    clusters.sort(key=lambda cluster: (-int(cluster.size), float(cluster.path_distance_from_agent)))
    if int(max_count) > 0:
        return clusters[: int(max_count)]
    return clusters
