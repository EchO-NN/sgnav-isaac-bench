from __future__ import annotations

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import VoxelOccupancyDoorWallRoomSegConfig, run_voxel_occupancy_door_wall_roomseg
from isaac_bench.mapping.voxel_occupancy_grid import VOXEL_OCCUPIED, VoxelOccupancyGrid3D, VoxelOccupancyGridConfig


def test_main_voxel_roomseg_pipeline_ignores_occupied_any_furniture_for_projection() -> None:
    shape = (32, 32)
    info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=3.2, min_y=0.0, max_y=3.2, width=shape[1], height=shape[0])
    grid = VoxelOccupancyGrid3D.zeros(
        shape,
        info,
        VoxelOccupancyGridConfig(z_min_m=0.0, z_max_m=2.40, z_resolution_m=0.10, active_z_min_m=0.0, active_z_max_fallback_m=2.40),
    )
    grid.active_z_min_m = 0.0
    grid.active_z_max_m = 2.40
    jitter_rows = [10, 11, 10, 12, 11]
    for idx, col in enumerate(range(5, 25)):
        grid.state[8, jitter_rows[idx % len(jitter_rows)], col] = int(VOXEL_OCCUPIED)

    cfg = VoxelOccupancyDoorWallRoomSegConfig.from_mapping(
        {
            "voxel_roomseg_evidence": {"active_z_min_m": 0.0},
            "voxel_wall_projection": {"min_projected_line_length_m": 0.45, "min_projected_support_ratio": 0.30},
            "voxel_door": {"enabled": False},
            "voxel_step1": {"gap_fill_enabled": False},
            "voxel_step2": {"enabled": False},
        },
        resolution_m=0.10,
        map_info=info,
    )
    result = run_voxel_occupancy_door_wall_roomseg(
        occupancy_map=np.zeros(shape, dtype=bool),
        observed_free_mask=np.zeros(shape, dtype=bool),
        obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=np.ones(shape, dtype=bool),
        voxel_grid=grid,
        navigation_free_mask=np.zeros(shape, dtype=bool),
        navigation_obstacle_mask=np.zeros(shape, dtype=bool),
        resolution_m=0.10,
        config=cfg,
    )

    assert np.any(result.layers["voxel_raw_occupied_wall_support_xy"])
    assert not np.any(result.layers["voxel_wall_xy"])
    assert not np.any(result.layers["voxel_projected_structural_wall_map"])
    assert not np.any(result.layers["voxel_partition_real_wall_map"])
    assert int(result.debug["voxel_wall_projection_line_count"]) == 0
    assert result.debug["voxel_wall_ratio_used_for_final"] is True


def test_parallel_wall_projection_keeps_nearby_walls_split() -> None:
    shape = (28, 32)
    raw = np.zeros(shape, dtype=bool)
    raw[10, 5:25] = True
    raw[15, 5:25] = True
    free = np.zeros(shape, dtype=bool)

    from isaac_bench.mapping.wall_projection import WallProjectionConfig, project_wall_evidence_to_lines

    result = project_wall_evidence_to_lines(
        wall_raw=raw,
        free_map=free,
        occupied_ratio=raw.astype(np.float32),
        resolution_m=0.10,
        config=WallProjectionConfig(min_projected_line_length_m=0.45, separate_parallel_wall_min_cells=4, parallel_peak_min_support_cells=4, side_validation_enabled=False),
    )

    lines = [line for line in result.projected_lines if line.reject_reason is None]
    assert len(lines) >= 2
    horizontal_rows = {int(line.line) for line in lines if line.axis == "h"}
    assert 10 in horizontal_rows
    assert 15 in horizontal_rows
    assert not any(line.axis == "h" and line.projected_cell_count >= 35 for line in lines)
