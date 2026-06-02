from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_grid import (
    VOXEL_FREE,
    VOXEL_OCCUPIED,
    VoxelOccupancyGrid3D,
    VoxelOccupancyGridConfig,
)


def _grid(shape: tuple[int, int] = (12, 12)) -> VoxelOccupancyGrid3D:
    info = MapInfo(resolution_m=0.10, min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, width=shape[1], height=shape[0])
    return VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=1.0, z_resolution_m=0.10, active_z_min_m=0.0),
    )


def test_incremental_navigation_projection_matches_full_without_morphology() -> None:
    grid = _grid()
    grid.state[2, 4, 4] = int(VOXEL_FREE)
    grid.state[3, 7, 8] = int(VOXEL_OCCUPIED)

    full0 = grid.project_navigation(
        force_full=True,
        projection_step=0,
        occupied_close_radius_cells=0,
        occupied_fill_small_holes_max_area_cells=0,
    )
    assert full0.free[4, 4]
    assert full0.occupied[7, 8]

    grid.state[2, 4, 4] = 0
    grid.state[3, 4, 4] = int(VOXEL_OCCUPIED)
    grid.state[2, 6, 6] = int(VOXEL_FREE)
    dirty = np.zeros(grid.shape[0] * grid.shape[1], dtype=np.uint8)
    dirty[4 * grid.shape[1] + 4] = 1
    dirty[6 * grid.shape[1] + 6] = 1

    inc = grid.project_navigation(
        incremental=True,
        force_full=False,
        dirty_rc_flags=dirty,
        projection_step=1,
        occupied_close_radius_cells=0,
        occupied_fill_small_holes_max_area_cells=0,
        dirty_dilation_radius_cells=0,
    )
    full1 = grid.project_navigation(
        incremental=False,
        force_full=True,
        projection_step=2,
        occupied_close_radius_cells=0,
        occupied_fill_small_holes_max_area_cells=0,
    )

    assert inc.debug["voxel_project_navigation_mode"] == "incremental"
    assert int(inc.debug["voxel_project_navigation_dirty_rc_count"]) == 2
    assert np.array_equal(inc.free, full1.free)
    assert np.array_equal(inc.occupied, full1.occupied)
    assert np.array_equal(inc.observed, full1.observed)
