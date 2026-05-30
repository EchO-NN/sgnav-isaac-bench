from __future__ import annotations

import numpy as np

from isaac_bench.mapping.voxel_occupancy_door_wall_roomseg import merge_small_enclosed_single_neighbor_regions


def test_small_enclosed_single_neighbor_region_merges() -> None:
    labels = np.ones((24, 24), dtype=np.int32)
    labels[:2, :] = 0
    labels[-2:, :] = 0
    labels[:, :2] = 0
    labels[:, -2:] = 0
    labels[11:13, 11:13] = 2
    free = labels > 0

    merged, debug = merge_small_enclosed_single_neighbor_regions(
        labels,
        partition_free=free,
        partition_unknown=np.zeros_like(free),
        final_separator_map=np.zeros_like(free),
        resolution_m=0.10,
        max_area_m2=1.50,
    )

    assert not np.any(merged == 2)
    assert np.all(merged[11:13, 11:13] == 1)
    assert debug["voxel_small_enclosed_merge_event_count"] == 1
    assert np.count_nonzero(debug["voxel_small_enclosed_merged_region_map"]) == 4


def test_small_region_touching_unknown_does_not_merge() -> None:
    labels = np.ones((24, 24), dtype=np.int32)
    labels[:2, :] = 0
    labels[-2:, :] = 0
    labels[:, :2] = 0
    labels[:, -2:] = 0
    labels[11:13, 11:13] = 2
    free = labels > 0
    unknown = np.zeros_like(free)
    unknown[10, 11] = True

    merged, debug = merge_small_enclosed_single_neighbor_regions(
        labels,
        partition_free=free,
        partition_unknown=unknown,
        final_separator_map=np.zeros_like(free),
        resolution_m=0.10,
        max_area_m2=1.50,
    )

    assert np.any(merged == 2)
    assert debug["voxel_small_enclosed_merge_event_count"] == 0


def test_small_region_with_two_neighbors_does_not_merge() -> None:
    labels = np.ones((10, 14), dtype=np.int32)
    labels[:, 7:] = 3
    labels[:2, :] = 0
    labels[-2:, :] = 0
    labels[:, :2] = 0
    labels[:, -2:] = 0
    labels[4:6, 6:8] = 2
    free = labels > 0

    merged, debug = merge_small_enclosed_single_neighbor_regions(
        labels,
        partition_free=free,
        partition_unknown=np.zeros_like(free),
        final_separator_map=np.zeros_like(free),
        resolution_m=0.10,
        max_area_m2=1.50,
    )

    assert np.any(merged == 2)
    assert debug["voxel_small_enclosed_merge_event_count"] == 0


def test_large_region_under_no_condition_merge() -> None:
    labels = np.ones((24, 24), dtype=np.int32)
    labels[:2, :] = 0
    labels[-2:, :] = 0
    labels[:, :2] = 0
    labels[:, -2:] = 0
    labels[5:18, 5:18] = 2
    free = labels > 0

    merged, debug = merge_small_enclosed_single_neighbor_regions(
        labels,
        partition_free=free,
        partition_unknown=np.zeros_like(free),
        final_separator_map=np.zeros_like(free),
        resolution_m=0.10,
        max_area_m2=1.50,
    )

    assert np.any(merged == 2)
    assert debug["voxel_small_enclosed_merge_event_count"] == 0
