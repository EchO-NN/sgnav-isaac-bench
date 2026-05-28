from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_FREE


def test_initial_blind_zone_forces_navigation_and_voxel_low_z_free() -> None:
    mapper = OnlineMapper(
        size_m=2.0,
        resolution_m=0.10,
        voxel_grid_enabled=True,
        voxel_grid_config={
            "enabled": True,
            "z_min_m": 0.0,
            "z_max_m": 1.0,
            "z_resolution_m": 0.10,
            "active_z_min_m": 0.0,
            "active_z_max_fallback_m": 1.0,
            "integration_backend": "cpu_vectorized",
            "voxel_grid_drives_navigation": True,
        },
        voxel_navigation_blind_zone_config={
            "enabled": True,
            "initial_blind_zone_radius_m": 0.30,
            "initial_blind_zone_steps": 20,
            "current_footprint_radius_m": 0.20,
            "write_to_voxel_grid": True,
            "free_z_min_m": 0.10,
            "free_z_max_m": 0.35,
        },
    )
    mapper.reset((0.0, 0.0))

    debug = mapper._apply_voxel_navigation_blind_zone((0.0, 0.0, 0.0, 0.0), write_to_voxel=True, write_to_grid=True)
    mapper.voxel_grid.refresh_state()

    rows, cols = mapper._disk_cells_for_world_xy((0.0, 0.0), 0.20)
    assert rows.size > 0
    assert np.all(mapper.grid.free[rows, cols] == 1)
    assert not np.any(mapper.grid.occupied[rows, cols])
    z_idx = mapper.voxel_grid.active_z_indices(z_min_m=0.10, z_max_m=0.35)
    assert z_idx.size > 0
    assert np.any(mapper.voxel_grid.state[np.ix_(z_idx, rows, cols)] == int(VOXEL_FREE))
    assert int(debug["voxel_blind_zone_written_voxels"]) > 0
