from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.rose2_source_form import (
    LEGACY_STYLE_BACKEND,
    ROSE2SourceFormConfig,
    SOURCE_EXTERNAL_BACKEND,
    SOURCE_FORM_BACKEND,
    export_rose2_source_input,
    parse_rose2_source_label_image,
    run_rose2_source_form,
    save_rose2_source_debug,
)
from isaac_bench.mapping.structure_extraction import StructureExtractionConfig
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import UpstreamROSE2Config, UpstreamROSE2PurePythonSegmenter
from isaac_bench.metrics.result_schema import BenchmarkAssetError, validate_strict_benchmark_assets


def _structure_cfg() -> StructureExtractionConfig:
    return StructureExtractionConfig(
        resolution_m=0.10,
        min_room_area_m2=0.5,
        hough_min_line_length_m=0.5,
        hough_line_gap_m=0.15,
        wall_min_support_ratio=0.10,
    )


def _source_cfg() -> ROSE2SourceFormConfig:
    return ROSE2SourceFormConfig(
        resolution_m=0.10,
        min_room_area_m2=0.5,
        min_cell_area_m2=0.30,
        min_cut_spacing_m=0.40,
        min_wall_line_length_m=0.5,
    )


def _room_count(labels: np.ndarray) -> int:
    return len([v for v in np.unique(np.asarray(labels, dtype=np.int32)) if int(v) > 0])


def _fake_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "declutter-reconstruct"
    code = root / "code"
    code.mkdir(parents=True)
    for name in ("FFT_MQ.py", "minibatch.py", "parameters.py"):
        (code / name).write_text("# source-form backend test placeholder\n", encoding="utf-8")
    return root


def _map_info(shape: tuple[int, int]) -> MapInfo:
    h, w = shape
    return MapInfo(resolution_m=0.10, min_x=0.0, max_x=w * 0.10, min_y=0.0, max_y=h * 0.10, width=w, height=h)


def _two_rooms_with_door() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (70, 90)
    free = np.zeros(shape, dtype=bool)
    free[10:60, 8:82] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 44:46] = True
    free[:, 44:46] = False
    occupied[32:42, 44:46] = False
    free[32:42, 44:46] = True
    return occupied, free, ~(occupied | free)


def _three_rooms_with_corridor() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (90, 110)
    free = np.zeros(shape, dtype=bool)
    free[8:82, 8:102] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[8:82, 52:54] = True
    free[8:82, 52:54] = False
    occupied[38:52, 52:54] = False
    free[38:52, 52:54] = True
    occupied[44:46, 54:102] = True
    free[44:46, 54:102] = False
    occupied[44:46, 75:85] = False
    free[44:46, 75:85] = True
    return occupied, free, ~(occupied | free)


def _open_plan_with_clutter() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (60, 80)
    free = np.zeros(shape, dtype=bool)
    free[8:52, 8:72] = True
    occupied = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in [(18, 20, 23, 28), (34, 42, 42, 50), (20, 58, 25, 65)]:
        occupied[r0:r1, c0:c1] = True
        free[r0:r1, c0:c1] = False
    return occupied, free, ~(occupied | free)


def test_source_form_two_rooms_with_door_produces_two_source_cells():
    occupied, free, unknown = _two_rooms_with_door()
    result = run_rose2_source_form(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        structure_config=_structure_cfg(),
        source_config=_source_cfg(),
    )

    assert _room_count(result.room_label_map) >= 2
    assert result.backend == SOURCE_FORM_BACKEND
    assert result.debug["legacy_connected_component_rooms_used"] is False
    assert result.debug["extended_wall_line_count"] > 0
    assert result.debug["labels_outside_vertical_free_cells"] == 0


def test_source_form_three_rooms_and_corridor_produces_multiple_planar_cells():
    occupied, free, unknown = _three_rooms_with_corridor()
    result = run_rose2_source_form(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        structure_config=_structure_cfg(),
        source_config=_source_cfg(),
    )

    assert _room_count(result.room_label_map) >= 3
    assert result.debug["source_cell_count"] >= 3
    assert result.debug["cell_edge_count"] >= 1


def test_source_form_open_plan_does_not_turn_clutter_into_rooms():
    occupied, free, unknown = _open_plan_with_clutter()
    result = run_rose2_source_form(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        structure_config=_structure_cfg(),
        source_config=_source_cfg(),
    )

    assert _room_count(result.room_label_map) == 1
    assert result.debug["source_cell_count"] == 1


def test_adapter_strict_default_uses_source_form_without_external_source_root(tmp_path):
    occupied, free, unknown = _two_rooms_with_door()
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(tmp_path / "missing-source"),
            backend=SOURCE_FORM_BACKEND,
            resolution_m=0.10,
            min_room_area_m2=0.5,
            hough_min_line_length_m=0.5,
            hough_line_gap_m=0.15,
            fail_on_missing_source=True,
            debug_dump=False,
        ),
        _map_info(occupied.shape),
    )

    rooms = segmenter.update(occupied, free, occupied, unknown, step=3)

    assert len(rooms) >= 2
    assert segmenter.last_debug["source_backend"] == SOURCE_FORM_BACKEND
    assert segmenter.last_debug["source_result_summary"]["legacy_connected_component_rooms_used"] is False


def test_legacy_style_backend_is_rejected_when_strict_disallow_is_set(tmp_path):
    occupied, free, unknown = _two_rooms_with_door()
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(_fake_source_root(tmp_path)),
            backend=LEGACY_STYLE_BACKEND,
            strict_disallow_legacy_fallback=True,
            resolution_m=0.10,
            min_room_area_m2=0.5,
            hough_min_line_length_m=0.5,
            debug_dump=False,
        ),
        _map_info(occupied.shape),
    )

    with pytest.raises(ValueError, match="debug/ablation-only"):
        segmenter.update(occupied, free, occupied, unknown, step=3)


def test_strict_asset_contract_requires_external_source_only_for_external_backend(tmp_path):
    base = dict(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode="upstream_rose2_vertical_or_free",
        ablation_name=None,
    )
    validate_strict_benchmark_assets(
        Namespace(**base, room_segmentation_config={"backend": SOURCE_FORM_BACKEND, "source_root": str(tmp_path / "missing")})
    )
    with pytest.raises(BenchmarkAssetError, match="Missing upstream ROSE2"):
        validate_strict_benchmark_assets(
            Namespace(
                **base,
                room_segmentation_config={
                    "backend": SOURCE_EXTERNAL_BACKEND,
                    "source_root": str(tmp_path / "missing"),
                    "require_upstream_source_for_strict": True,
                },
            )
        )


def test_source_form_debug_artifacts_and_source_io_roundtrip(tmp_path):
    occupied, free, unknown = _two_rooms_with_door()
    result = run_rose2_source_form(
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
        structure_config=_structure_cfg(),
        source_config=_source_cfg(),
    )
    dump = save_rose2_source_debug(
        out_dir=tmp_path / "debug",
        step=7,
        result=result,
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
    )
    export = export_rose2_source_input(
        out_dir=tmp_path / "source_io",
        observed_occupied=occupied,
        observed_free=free,
        unknown=unknown,
    )
    label_img = np.zeros((*occupied.shape, 3), dtype=np.uint8)
    label_img[result.room_label_map == 1] = (255, 0, 0)
    label_img[result.room_label_map == 2] = (0, 255, 0)
    label_path = tmp_path / "labels.png"
    Image.fromarray(label_img).save(label_path)
    parsed = parse_rose2_source_label_image(label_path)

    assert Path(dump["paths"]["summary_json"]).exists()
    assert Path(dump["paths"]["overlay_png"]).exists()
    assert Path(dump["paths"]["lines_png"]).exists()
    assert Path(dump["paths"]["cells_png"]).exists()
    assert Path(export["metric_map_png"]).exists()
    assert _room_count(parsed) >= 2
