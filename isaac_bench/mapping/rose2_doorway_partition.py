from __future__ import annotations

import numpy as np

from isaac_bench.mapping.rose2_partition_graph import generate_doorway_partition_cuts as _generate


def generate_doorway_partition_cuts(
    *,
    observed_free: np.ndarray,
    wall_boundary: np.ndarray,
    selected_separator_boundary: np.ndarray,
    candidate_lines: list[dict],
    resolution_m: float,
    width_min_m: float = 0.45,
    width_max_m: float = 1.60,
    min_wall_support_on_sides_m: float = 0.35,
    max_unknown_ratio: float = 0.20,
) -> tuple[list[dict], dict]:
    """Generate virtual room-label cuts across navigable doorway gaps.

    These cuts are intentionally not obstacles. They are room partition
    evidence only, so navigation can still pass through the opening.
    """

    cuts, debug = _generate(
        free=observed_free,
        wall_boundary=wall_boundary,
        selected_separator_boundary=selected_separator_boundary,
        candidate_lines=candidate_lines,
        resolution_m=float(resolution_m),
        doorway_width_min_m=float(width_min_m),
        doorway_width_max_m=float(width_max_m),
        min_wall_support_on_sides_m=float(min_wall_support_on_sides_m),
    )
    converted = []
    for cut in cuts:
        row = dict(cut)
        row["source"] = "doorway_partition_cut"
        row["is_virtual_room_boundary"] = True
        row["is_navigation_obstacle"] = False
        converted.append(row)
    debug = dict(debug)
    debug["source"] = "doorway_partition_cut"
    debug["max_unknown_ratio"] = float(max_unknown_ratio)
    debug["doorway_partition_cuts"] = converted[:128]
    debug["doorway_partition_cut_count"] = int(len(converted))
    return converted, debug
