import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.frontier import extract_frontiers, frontier_cells, frontier_debug_layers
from isaac_bench.mapping.frontier_debug import save_frontier_debug_snapshot
from isaac_bench.navigation.astar import astar_distance_map


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


def test_observed_nonfree_nonoccupied_cell_is_not_unknown():
    free = np.zeros((7, 7), dtype=bool)
    observed = np.zeros_like(free)
    occupancy = np.zeros_like(free)
    free[3, 3] = True
    observed[2:5, 2:5] = True

    observed_frontiers = frontier_cells(
        free,
        observed=observed,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
        unknown_source="observed",
    )
    implicit_frontiers = frontier_cells(
        free,
        observed=observed,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
        unknown_source="implicit",
    )

    assert not observed_frontiers[3, 3]
    assert implicit_frontiers[3, 3]


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
    assert clusters[0].center_grid in clusters[0].members
    assert np.isfinite(clusters[0].path_distance_from_agent)
    assert clusters[0].min_path_distance <= clusters[0].mean_path_distance <= clusters[0].max_path_distance
    assert 0.0 <= clusters[0].distance_inverse <= 1.0
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


def test_extract_frontiers_drops_near_frontiers_unless_fallback_enabled():
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

    assert clusters == []

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
        allow_near_frontier_fallback=True,
    )

    assert clusters


def test_distance_map_does_not_cross_unobserved_unknown_cells():
    free = np.zeros((20, 20), dtype=bool)
    free[10, 2:8] = True
    static_navigable = np.ones_like(free, dtype=bool)
    dynamic_traversible = free.copy()

    dist_static = astar_distance_map(static_navigable, (10, 2), 0.05)
    dist_dynamic = astar_distance_map(dynamic_traversible, (10, 2), 0.05)

    assert np.isfinite(dist_static[10, 15])
    assert not np.isfinite(dist_dynamic[10, 15])


def test_frontier_debug_snapshot_saves_expected_layers(tmp_path):
    free = np.zeros((7, 7), dtype=bool)
    observed = np.zeros_like(free)
    occupancy = np.zeros_like(free)
    free[2:5, 2:5] = True
    observed[2:5, 2:5] = True
    traversible = free.copy()
    info = MapInfo(resolution_m=1.0, min_x=0.0, max_x=7.0, min_y=0.0, max_y=7.0, width=7, height=7)
    layers = frontier_debug_layers(
        free,
        observed=observed,
        occupancy=occupancy,
        obstacle_dilation_radius_cells=0,
        unknown_dilation_radius_cells=1,
    )
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

    npz_path, png_path = save_frontier_debug_snapshot(
        tmp_path,
        4,
        free=free,
        occupied=occupancy,
        observed=observed,
        unknown=layers["unknown"],
        unknown_dilated=layers["unknown_dilated"],
        frontier=layers["frontier"],
        traversible=traversible,
        dist_map=astar_distance_map(traversible, (3, 3), info.resolution_m),
        agent_grid=(3, 3),
        clusters=clusters,
        selected_frontier=clusters[0].center_grid,
    )

    assert npz_path.exists()
    assert png_path is None or png_path.exists()
    data = np.load(npz_path)
    assert "frontier_cells" in data
    assert "unknown_dilated" in data
