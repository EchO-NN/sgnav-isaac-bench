import numpy as np

from isaac_bench.mapping.rose2_partition_graph import should_block_room_merge
from isaac_bench.mapping.rose2_source_form import (
    ROSE2SourceFormConfig,
    run_rose2_source_faithful_v1,
)
from isaac_bench.mapping.structure_extraction import StructureExtractionConfig, connected_components


def _cfg(resolution=0.10):
    return StructureExtractionConfig(
        resolution_m=resolution,
        min_room_area_m2=0.5,
        hough_min_line_length_m=0.35,
        hough_line_gap_m=0.25,
        wall_min_support_ratio=0.10,
    ), ROSE2SourceFormConfig(
        resolution_m=resolution,
        min_room_area_m2=0.5,
        min_wall_line_length_m=0.35,
        thin_wall_min_length_m=0.35,
        thin_wall_max_width_m=0.30,
        thin_wall_min_free_support_ratio=0.20,
        topology_effective_max_candidates=256,
        doorway_width_min_m=0.45,
        doorway_width_max_m=1.60,
    )


def _run(occupied, free, unknown=None, vertical_observed=None, vertical_free=None):
    if unknown is None:
        unknown = ~(occupied | free)
    if vertical_observed is None:
        vertical_observed = occupied | free
    if vertical_free is None:
        vertical_free = free
    structure_cfg, source_cfg = _cfg()
    return run_rose2_source_faithful_v1(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        vertical_observed=vertical_observed,
        vertical_free=vertical_free,
        wall_confidence_map=occupied.astype(np.float32),
        structure_config=structure_cfg,
        source_config=source_cfg,
    )


def test_full_thin_wall_splits_two_rooms():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[5:45, 30] = True
    free[occupied] = False

    result = _run(occupied, free)

    assert result.backend == "rose2_source_faithful_v1"
    assert result.debug["source_room_count"] == 2
    assert result.debug["accepted_topology_separator_count"] >= 1


def test_thin_wall_with_doorway_keeps_navigation_free_connected_but_splits_room_labels():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[5:45, 30] = True
    occupied[21:29, 30] = False
    free[occupied] = False

    result = _run(occupied, free)

    assert len(connected_components(free)) == 1
    assert result.debug["doorway_partition_cut_count"] >= 1
    assert result.debug["source_room_count"] == 2


def test_unknown_black_line_is_not_forced_into_a_room_split():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    unknown = np.zeros(shape, dtype=bool)
    unknown[5:45, 30] = True
    free[unknown] = False
    occupied = np.zeros(shape, dtype=bool)
    vertical_observed = free.copy()

    result = _run(occupied, free, unknown=unknown, vertical_observed=vertical_observed, vertical_free=free)

    assert result.debug["thin_wall_separator_count"] == 0
    assert result.debug["source_room_count"] == 1


def test_nonfree_observed_thin_wall_without_occupied_support_is_promoted():
    shape = (50, 60)
    vertical_free = np.zeros(shape, dtype=bool)
    vertical_free[5:45, 5:55] = True
    vertical_observed = vertical_free.copy()
    vertical_observed[5:45, 30] = True
    vertical_free[5:45, 30] = False
    occupied = np.zeros(shape, dtype=bool)
    unknown = np.zeros(shape, dtype=bool)

    result = _run(
        occupied,
        vertical_free,
        unknown=unknown,
        vertical_observed=vertical_observed,
        vertical_free=vertical_free,
    )

    assert result.debug["thin_wall_separator_count"] >= 1
    assert result.debug["source_room_count"] == 2


def test_open_plan_has_one_room():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)

    result = _run(occupied, free)

    assert result.debug["thin_wall_separator_count"] == 0
    assert result.debug["accepted_topology_separator_count"] == 0
    assert result.debug["source_room_count"] == 1


def test_merge_guard_blocks_strong_separator_edges():
    class _SourceResult:
        debug = {
            "accepted_separators": [
                {
                    "source": "thin_wall_from_nonfree_observed",
                    "topology_effective": True,
                }
            ]
        }

    blocked, reason = should_block_room_merge(
        room_a=1,
        room_b=2,
        adjacency_edge={"edge_wall_weight": 0.0},
        source_result=_SourceResult(),
    )

    assert blocked is True
    assert "thin_wall_from_nonfree_observed" in reason
