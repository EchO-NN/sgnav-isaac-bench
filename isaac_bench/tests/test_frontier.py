import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier import extract_frontiers, frontier_cells


def test_frontier_cells_match_sgnav_unknown_boundary():
    free = np.zeros((7, 7), dtype=bool)
    occupancy = np.zeros_like(free)
    free[2:5, 2:5] = True

    frontiers = frontier_cells(
        free,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
    )

    assert frontiers[2, 3]
    assert frontiers[3, 2]
    assert frontiers[3, 4]
    assert frontiers[4, 3]
    assert not frontiers[3, 3]


def test_obstacle_dilation_suppresses_nearby_free_frontiers():
    free = np.zeros((7, 7), dtype=bool)
    occupancy = np.zeros_like(free)
    free[3, 2:5] = True
    occupancy[3, 5] = True

    frontiers = frontier_cells(
        free,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=1,
        unknown_dilation_radius_cells=1,
    )

    assert not frontiers[3, 4]
    assert frontiers[3, 2]


def test_extract_frontiers_returns_sgnav_fbe_cells_without_projection():
    free = np.zeros((7, 7), dtype=bool)
    observed = np.zeros_like(free)
    occupancy = np.zeros_like(free)
    free[2:5, 2:5] = True
    observed[2:5, 2:5] = True
    traversible = free.copy()

    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=7.0, min_y=0.0, max_y=7.0, width=7, height=7)
    clusters = extract_frontiers(
        free=free,
        observed=observed,
        traversible=traversible,
        map_info=info,
        agent_grid=(3, 3),
        min_cluster_size=1,
        min_distance_m=0.0,
        max_count=0,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
    )
    expected = frontier_cells(
        free,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
    )
    expected_cells = {(int(row), int(col)) for row, col in zip(*np.nonzero(expected))}
    actual_cells = {cluster.center_grid for cluster in clusters}

    assert len(clusters) == 1
    assert set(clusters[0].members) == expected_cells
    assert clusters[0].size == len(expected_cells)
    assert clusters[0].center_grid in expected_cells
    assert actual_cells <= expected_cells


def test_extract_frontiers_filters_small_components():
    free = np.zeros((7, 7), dtype=bool)
    observed = np.zeros_like(free)
    occupancy = np.zeros_like(free)
    free[3, 3] = True
    observed[3, 3] = True
    traversible = free.copy()

    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=7.0, min_y=0.0, max_y=7.0, width=7, height=7)
    clusters = extract_frontiers(
        free=free,
        observed=observed,
        traversible=traversible,
        map_info=info,
        agent_grid=(3, 3),
        min_cluster_size=2,
        min_distance_m=0.0,
        max_count=0,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
    )

    assert clusters == []


def test_static_nearfield_exclude_mask_suppresses_local_frontier_ring():
    free = np.zeros((9, 9), dtype=bool)
    occupancy = np.zeros_like(free)
    free[3:6, 3:6] = True
    exclude = np.zeros_like(free)
    exclude[3:6, 3:6] = True

    frontiers = frontier_cells(
        free,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
        exclude_mask=exclude,
    )

    assert not np.any(frontiers)


def test_extract_frontiers_keeps_near_frontiers_when_all_are_close():
    free = np.zeros((7, 7), dtype=bool)
    observed = np.zeros_like(free)
    occupancy = np.zeros_like(free)
    free[2:5, 2:5] = True
    observed[2:5, 2:5] = True
    traversible = free.copy()

    info = MapInfo(resolution_m=0.05, min_x=0.0, max_x=0.35, min_y=0.0, max_y=0.35, width=7, height=7)
    clusters = extract_frontiers(
        free=free,
        observed=observed,
        traversible=traversible,
        map_info=info,
        agent_grid=(3, 3),
        min_distance_m=10.0,
        max_count=0,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
    )

    assert clusters
