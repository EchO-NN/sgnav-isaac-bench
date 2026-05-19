import json
import numpy as np

from isaac_bench.mapping.rose2_source_external_runner import run_rose2_source_external_runner
from isaac_bench.mapping.rose2_source_form import ROSE2SourceFormConfig, run_rose2_source_form_v2
from isaac_bench.mapping.structure_extraction import StructureExtractionConfig
from isaac_bench.mapping.structure_extraction import connected_components


def _cfg(resolution=0.10):
    return StructureExtractionConfig(
        resolution_m=resolution,
        min_room_area_m2=0.5,
        hough_min_line_length_m=0.45,
        hough_line_gap_m=0.25,
        wall_min_support_ratio=0.10,
    ), ROSE2SourceFormConfig(
        resolution_m=resolution,
        min_room_area_m2=0.5,
        min_wall_line_length_m=0.45,
        thin_wall_min_length_m=0.45,
        thin_wall_max_width_m=0.25,
        thin_wall_free_support_band_m=0.30,
        doorway_width_min_m=0.45,
        doorway_width_max_m=1.60,
    )


def _run(occupied, free, unknown=None):
    if unknown is None:
        unknown = ~(occupied | free)
    structure_cfg, source_cfg = _cfg()
    vertical_observed = occupied | free
    return run_rose2_source_form_v2(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        vertical_observed=vertical_observed,
        vertical_free=free,
        wall_confidence_map=occupied.astype(np.float32),
        structure_config=structure_cfg,
        source_config=source_cfg,
    )


def test_thin_one_cell_wall_splits_two_rooms():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[5:45, 30] = True
    free[occupied] = False

    result = _run(occupied, free)

    assert result.backend == "rose2_source_form_v2"
    assert result.debug["thin_wall_separator_count"] >= 1
    assert result.debug["accepted_topology_separator_count"] >= 1
    assert result.debug["source_room_count"] >= 2


def test_wall_with_doorway_keeps_navigation_free_connected_but_room_labels_split():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[5:45, 30] = True
    occupied[21:29, 30] = False
    free[occupied] = False

    result = _run(occupied, free)

    assert _component_count(free) == 1
    assert result.debug["doorway_partition_cut_count"] >= 1
    assert result.debug["accepted_topology_separator_count"] >= 1
    assert result.debug["source_room_count"] >= 2
    assert result.cell_edges or result.debug["doorway_partition_cuts"]


def test_open_plan_remains_one_room():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)

    result = _run(occupied, free)

    assert result.debug["thin_wall_separator_count"] == 0
    assert result.debug["accepted_topology_separator_count"] == 0
    assert result.debug["source_room_count"] == 1


def test_furniture_like_blob_does_not_split_room():
    shape = (50, 60)
    free = np.zeros(shape, dtype=bool)
    free[5:45, 5:55] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[22:27, 28:33] = True
    free[occupied] = False

    result = _run(occupied, free)

    assert result.debug["accepted_topology_separator_count"] == 0
    assert result.debug["source_room_count"] == 1


def test_screenshot_like_thin_wall_layout_produces_multiple_rooms():
    shape = (80, 100)
    free = np.zeros(shape, dtype=bool)
    free[10:60, 8:92] = True
    free[52:72, 42:58] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[10:54, 58] = True
    occupied[10:54, 59] = True
    occupied[35:43, 58:60] = False
    free[occupied] = False

    result = _run(occupied, free)

    assert result.debug["thin_wall_separator_count"] >= 1
    assert result.debug["accepted_topology_separator_count"] >= 1
    assert result.debug["proposal_room_count"] >= 2
    assert result.debug["final_room_count_before_policy_merge"] >= 2


def test_external_runner_smoke_writes_reproducible_failure_report(tmp_path):
    shape = (20, 20)
    free = np.zeros(shape, dtype=bool)
    free[3:17, 3:17] = True
    occupied = np.zeros(shape, dtype=bool)
    unknown = ~(free | occupied)

    result = run_rose2_source_external_runner(
        source_root=tmp_path / "missing_source",
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        work_dir=tmp_path / "external",
        timeout_s=1.0,
    )

    summary_path = tmp_path / "external" / "external_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["failure_reason"] == "missing_rose2_source_root"
    assert (tmp_path / "external" / "external_room_label_map.png").exists()
    assert result.debug["source_backend"] == "rose2_source_external_runner"


def _component_count(mask):
    return len(connected_components(mask))
