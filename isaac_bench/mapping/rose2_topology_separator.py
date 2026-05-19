from __future__ import annotations

import numpy as np

from isaac_bench.mapping.rose2_partition_graph import select_topology_effective_separators


def select_topology_effective_separator_boundaries(
    *,
    observed_free: np.ndarray,
    unknown: np.ndarray,
    base_boundary: np.ndarray,
    candidate_lines: list[dict],
    resolution_m: float,
    min_room_area_m2: float = 1.0,
    wall_raster_radius_cells: int = 1,
    max_candidates: int = 256,
    min_largest_component_drop: float = 0.08,
) -> tuple[np.ndarray, list[dict], dict]:
    """Compatibility wrapper for the source-faithful topology test."""

    return select_topology_effective_separators(
        free=observed_free,
        unknown=unknown,
        base_boundary=base_boundary,
        candidate_lines=candidate_lines,
        resolution_m=float(resolution_m),
        min_room_area_m2=float(min_room_area_m2),
        wall_raster_radius_cells=int(wall_raster_radius_cells),
        max_candidates=int(max_candidates),
        min_largest_component_drop=float(min_largest_component_drop),
    )
