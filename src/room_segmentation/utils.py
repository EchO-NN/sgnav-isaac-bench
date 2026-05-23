from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Iterable, Sequence

import numpy as np

from .data_types import GridSpec


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(float(hi), max(float(lo), float(value))))


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(value, dtype=np.float64)
    out = 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))
    if np.isscalar(value):
        return float(out)
    return out


def radius_cells(radius_m: float, resolution_m: float) -> int:
    return int(max(0, math.ceil(float(radius_m) / max(float(resolution_m), 1e-9))))


def grid_to_world(row: int, col: int, grid: GridSpec) -> tuple[float, float]:
    return (
        float(grid.origin_xy[0]) + (float(col) + 0.5) * float(grid.resolution_m),
        float(grid.origin_xy[1]) + (float(row) + 0.5) * float(grid.resolution_m),
    )


def world_to_grid(x: float, y: float, grid: GridSpec) -> tuple[int, int]:
    col = int(math.floor((float(x) - float(grid.origin_xy[0])) / max(float(grid.resolution_m), 1e-9)))
    row = int(math.floor((float(y) - float(grid.origin_xy[1])) / max(float(grid.resolution_m), 1e-9)))
    return row, col


def in_bounds(row: int, col: int, shape: tuple[int, int]) -> bool:
    return 0 <= int(row) < int(shape[0]) and 0 <= int(col) < int(shape[1])


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    r = max(0, int(radius))
    return [
        (dr, dc)
        for dr in range(-r, r + 1)
        for dc in range(-r, r + 1)
        if dr * dr + dc * dc <= r * r
    ]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    r = max(0, int(radius))
    if r <= 0:
        return src.copy()
    out = np.zeros_like(src, dtype=bool)
    h, w = src.shape
    for dr, dc in disk_offsets(r):
        sr0 = max(0, -dr)
        sr1 = min(h, h - dr)
        sc0 = max(0, -dc)
        sc1 = min(w, w - dc)
        tr0 = sr0 + dr
        tr1 = sr1 + dr
        tc0 = sc0 + dc
        tc1 = sc1 + dc
        if sr0 < sr1 and sc0 < sc1:
            out[tr0:tr1, tc0:tc1] |= src[sr0:sr1, sc0:sc1]
    return out


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    r = max(0, int(radius))
    if r <= 0:
        return src.copy()
    return ~dilate(~src, r)


def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


def binary_open(mask: np.ndarray, radius: int) -> np.ndarray:
    return dilate(erode(mask, radius), radius)


def neighbor_offsets(connectivity: int = 4) -> tuple[tuple[int, int], ...]:
    if int(connectivity) == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    return ((-1, 0), (0, -1), (0, 1), (1, 0))


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    src = np.asarray(mask, dtype=bool)
    labels = np.zeros(src.shape, dtype=np.int32)
    rows, cols = np.nonzero(src)
    next_label = 0
    offsets = neighbor_offsets(connectivity)
    for start_r, start_c in zip(rows.tolist(), cols.tolist()):
        if labels[start_r, start_c] != 0:
            continue
        next_label += 1
        labels[start_r, start_c] = next_label
        q: deque[tuple[int, int]] = deque([(int(start_r), int(start_c))])
        while q:
            row, col = q.popleft()
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc, src.shape) and src[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = next_label
                    q.append((nr, nc))
    return labels, int(next_label)


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    next_label = 1
    for label in sorted(int(v) for v in np.unique(arr) if int(v) > 0):
        out[arr == label] = next_label
        next_label += 1
    return out


def remove_small_components(mask: np.ndarray, min_cells: int, connectivity: int = 4) -> np.ndarray:
    labels, count = label_components(mask, connectivity)
    out = np.zeros_like(labels, dtype=bool)
    for label in range(1, int(count) + 1):
        comp = labels == label
        if int(np.count_nonzero(comp)) >= int(min_cells):
            out |= comp
    return out


def fill_small_holes(mask: np.ndarray, max_area_cells: int, connectivity: int = 4) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    labels, count = label_components(~src, connectivity)
    out = src.copy()
    h, w = src.shape
    for label in range(1, int(count) + 1):
        comp = labels == label
        rows, cols = np.nonzero(comp)
        if rows.size == 0:
            continue
        if np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)):
            continue
        if int(rows.size) <= int(max_area_cells):
            out[comp] = True
    return out


def rasterize_line(p0: Sequence[float] | Sequence[int], p1: Sequence[float] | Sequence[int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = float(p0[0]), float(p0[1])
    r1, c1 = float(p1[0]), float(p1[1])
    steps = max(int(math.ceil(max(abs(r1 - r0), abs(c1 - c0)))) + 1, 1)
    rows = np.rint(np.linspace(r0, r1, steps)).astype(np.int32)
    cols = np.rint(np.linspace(c0, c1, steps)).astype(np.int32)
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row, col in zip(rows.tolist(), cols.tolist()):
        rc = (int(row), int(col))
        if rc in seen:
            continue
        seen.add(rc)
        if in_bounds(rc[0], rc[1], shape):
            cells.append(rc)
    return cells


def line_mask(p0: Sequence[float] | Sequence[int], p1: Sequence[float] | Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for row, col in rasterize_line(p0, p1, shape):
        out[row, col] = True
    return out


def path_crosses(mask: np.ndarray, p0: Sequence[float], p1: Sequence[float]) -> bool:
    src = np.asarray(mask, dtype=bool)
    return any(src[row, col] for row, col in rasterize_line(p0, p1, src.shape))


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance to the nearest non-mask cell in pixels."""

    passable = np.asarray(mask, dtype=bool)
    h, w = passable.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    heap: list[tuple[float, int, int]] = []
    boundary = ~passable
    if not np.any(boundary):
        boundary = np.zeros_like(passable, dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
    rows, cols = np.nonzero(boundary)
    for row, col in zip(rows.tolist(), cols.tolist()):
        dist[row, col] = 0.0
        heapq.heappush(heap, (0.0, int(row), int(col)))
    offsets = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while heap:
        cur, row, col = heapq.heappop(heap)
        if cur != float(dist[row, col]):
            continue
        for dr, dc, step in offsets:
            nr, nc = row + dr, col + dc
            if not in_bounds(nr, nc, passable.shape):
                continue
            new = cur + step
            if new < float(dist[nr, nc]):
                dist[nr, nc] = new
                heapq.heappush(heap, (new, nr, nc))
    dist[~passable] = 0.0
    return dist


def flood_fill(mask: np.ndarray, seeds: Iterable[tuple[int, int]], connectivity: int = 4) -> np.ndarray:
    passable = np.asarray(mask, dtype=bool)
    out = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for row, col in seeds:
        if in_bounds(row, col, passable.shape) and passable[row, col] and not out[row, col]:
            out[row, col] = True
            q.append((int(row), int(col)))
    offsets = neighbor_offsets(connectivity)
    while q:
        row, col = q.popleft()
        for dr, dc in offsets:
            nr, nc = row + dr, col + dc
            if in_bounds(nr, nc, passable.shape) and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def dijkstra_cost(passable: np.ndarray, sources: Iterable[tuple[int, int]], cost_map: np.ndarray | None = None) -> np.ndarray:
    mask = np.asarray(passable, dtype=bool)
    h, w = mask.shape
    cell_cost = np.ones((h, w), dtype=np.float32) if cost_map is None else np.asarray(cost_map, dtype=np.float32)
    dist = np.full((h, w), np.inf, dtype=np.float32)
    heap: list[tuple[float, int, int]] = []
    for row, col in sources:
        if in_bounds(row, col, mask.shape) and mask[row, col]:
            dist[row, col] = 0.0
            heapq.heappush(heap, (0.0, int(row), int(col)))
    offsets = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    while heap:
        cur, row, col = heapq.heappop(heap)
        if cur != float(dist[row, col]):
            continue
        for dr, dc, step in offsets:
            nr, nc = row + dr, col + dc
            if not in_bounds(nr, nc, mask.shape) or not mask[nr, nc]:
                continue
            new = cur + step * float(0.5 * (cell_cost[row, col] + cell_cost[nr, nc]))
            if new < float(dist[nr, nc]):
                dist[nr, nc] = new
                heapq.heappush(heap, (new, nr, nc))
    return dist


def boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    src = np.asarray(mask, dtype=bool)
    if not np.any(src):
        return []
    inner = erode(src, 1)
    rows, cols = np.nonzero(src & ~inner)
    return [(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]


def centroid_xy(mask: np.ndarray, grid: GridSpec) -> tuple[float, float]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return grid_to_world(0, 0, grid)
    return grid_to_world(int(round(float(np.mean(rows)))), int(round(float(np.mean(cols)))), grid)


def pca_direction_rc(cells: Sequence[tuple[int, int]], fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    if len(cells) < 2:
        return fallback
    coords = np.asarray(cells, dtype=np.float32)
    coords = coords - np.mean(coords, axis=0, keepdims=True)
    cov = np.cov(coords.T)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return fallback
    vec = vecs[:, int(np.argmax(vals))]
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return fallback
    return float(vec[0] / norm), float(vec[1] / norm)


def adjacency_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    arr = np.asarray(labels, dtype=np.int32)
    pairs: set[tuple[int, int]] = set()
    for dr, dc in ((1, 0), (0, 1)):
        a = arr[: arr.shape[0] - dr or None, : arr.shape[1] - dc or None]
        b = arr[dr:, dc:]
        mask = (a > 0) & (b > 0) & (a != b)
        for x, y in zip(a[mask].tolist(), b[mask].tolist()):
            p = tuple(sorted((int(x), int(y))))
            pairs.add(p)
    return pairs

