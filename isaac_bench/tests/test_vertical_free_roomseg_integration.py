from __future__ import annotations

from argparse import Namespace

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import (
    UpstreamROSE2Config,
    UpstreamROSE2PurePythonSegmenter,
)
from isaac_bench.mapping.vertical_free_roomseg import (
    VERTICAL_FREE_ROOMSEG_ALGORITHM,
    VERTICAL_FREE_ROOMSEG_BACKEND,
    VERTICAL_FREE_ROOMSEG_CONTEXT,
)
from isaac_bench.mapping.vertical_free_gap_closure_roomseg import (
    VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
    VERTICAL_FREE_GAP_CLOSURE_BACKEND,
    VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
)
from isaac_bench.metrics.result_schema import validate_strict_benchmark_assets


def _map_info(shape: tuple[int, int], resolution: float = 0.10) -> MapInfo:
    h, w = shape
    return MapInfo(
        resolution_m=resolution,
        min_x=0.0,
        max_x=float(w) * resolution,
        min_y=0.0,
        max_y=float(h) * resolution,
        width=w,
        height=h,
    )


def _two_rooms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:50] = True
    free[10:70, 70:110] = True
    free[35:45, 50:70] = True
    unknown = np.zeros_like(free, dtype=bool)
    occupied = ~free
    return occupied, free, unknown


def test_adapter_vertical_free_backend_returns_source_result_without_rose2_root(monkeypatch):
    monkeypatch.delenv("ROSE2_SOURCE_ROOT", raising=False)
    occupied, free, unknown = _two_rooms()
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=None,
            backend=VERTICAL_FREE_ROOMSEG_BACKEND,
            fail_on_missing_source=False,
            resolution_m=0.10,
            min_room_area_m2=0.20,
            debug_dump=False,
            finalization_mode="no_merge_until_geometry_verified",
            vertical_free_roomseg={
                "min_free_component_area_m2": 0.01,
                "seed_min_clearance_m": 0.20,
                "seed_min_distance_m": 1.0,
                "seed_min_area_m2": 0.05,
                "merge_small_area_m2": 0.20,
            },
        ),
        _map_info(free.shape),
    )

    rooms = segmenter.update(occupied, free, occupied, unknown, step=5)

    assert segmenter.last_debug["source_result_summary"]["backend"] == VERTICAL_FREE_ROOMSEG_BACKEND
    assert segmenter.last_debug["algorithm"] == VERTICAL_FREE_ROOMSEG_ALGORITHM
    assert segmenter.last_debug["source_backend"] == VERTICAL_FREE_ROOMSEG_BACKEND
    assert segmenter.last_debug["strict_fallback_used"] is False
    assert len(rooms) == 2
    labels = np.asarray(segmenter.last_debug["final_room_label_map"], dtype=np.int32)
    assert np.count_nonzero((labels > 0) & ~free) == 0


def test_strict_vertical_free_geodesic_backend_requires_named_ablation(monkeypatch):
    monkeypatch.delenv("ROSE2_SOURCE_ROOT", raising=False)
    args = Namespace(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode=VERTICAL_FREE_ROOMSEG_CONTEXT,
        room_segmentation_config={
            "backend": VERTICAL_FREE_ROOMSEG_BACKEND,
            "source_root": "/definitely/missing/rose2",
            "require_upstream_source_for_strict": False,
        },
        ablation_name=None,
    )

    try:
        validate_strict_benchmark_assets(args)
    except Exception as exc:
        assert "debug/ablation-only" in str(exc)
    else:
        raise AssertionError("strict metric path accepted legacy vertical-free geodesic roomseg")


def test_strict_vertical_free_geodesic_named_ablation_does_not_require_rose2_source(monkeypatch):
    monkeypatch.delenv("ROSE2_SOURCE_ROOT", raising=False)
    args = Namespace(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode=VERTICAL_FREE_ROOMSEG_CONTEXT,
        room_segmentation_config={
            "backend": VERTICAL_FREE_ROOMSEG_BACKEND,
            "source_root": "/definitely/missing/rose2",
            "require_upstream_source_for_strict": False,
        },
        ablation_name="vertical_free_geodesic_room_ablation",
    )

    validate_strict_benchmark_assets(args)


def test_strict_vertical_free_rejects_rose2_backend_mismatch():
    args = Namespace(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode=VERTICAL_FREE_ROOMSEG_CONTEXT,
        room_segmentation_config={"backend": "rose2_source_external_runner", "require_upstream_source_for_strict": False},
        ablation_name=None,
    )

    try:
        validate_strict_benchmark_assets(args)
    except Exception as exc:
        assert "debug/ablation-only" in str(exc)
    else:
        raise AssertionError("strict vertical-free mode accepted a ROSE2 backend")


def test_adapter_vertical_free_gap_closure_backend_splits_short_gap_without_rose2_root(monkeypatch):
    monkeypatch.delenv("ROSE2_SOURCE_ROOT", raising=False)
    free = np.zeros((80, 120), dtype=bool)
    free[10:70, 10:110] = True
    wall = np.zeros_like(free)
    wall[10:70, 59:61] = True
    free[wall] = False
    wall[35:45, 59:61] = False
    free[35:45, 59:61] = True
    unknown = np.zeros_like(free)
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=None,
            backend=VERTICAL_FREE_GAP_CLOSURE_BACKEND,
            fail_on_missing_source=False,
            resolution_m=0.10,
            min_room_area_m2=0.50,
            debug_dump=False,
            finalization_mode="no_merge_until_geometry_verified",
            vertical_free_gap_closure={
                "wall_min_component_cells": 1,
                "free_min_component_cells": 1,
                "wall_micro_close_radius_cells": 0,
                "side_support_min_free_cells": 3,
                "side_support_min_ratio": 0.10,
                "side_support_balance_min": 0.10,
                "line_free_ratio_min": 0.50,
                "candidate_score_min": 0.05,
                "small_component_area_m2": 0.10,
            },
        ),
        _map_info(free.shape),
    )

    rooms = segmenter.update(wall, free, wall, unknown, step=5)

    assert segmenter.last_debug["source_result_summary"]["backend"] == VERTICAL_FREE_GAP_CLOSURE_BACKEND
    assert segmenter.last_debug["algorithm"] == VERTICAL_FREE_GAP_CLOSURE_ALGORITHM
    assert segmenter.last_debug["source_backend"] == VERTICAL_FREE_GAP_CLOSURE_BACKEND
    assert segmenter.context_source == VERTICAL_FREE_GAP_CLOSURE_CONTEXT
    assert segmenter.last_debug["strict_fallback_used"] is False
    assert segmenter.last_debug["num_accepted_closures"] >= 1
    assert len(rooms) == 2


def test_strict_vertical_free_gap_closure_backend_does_not_require_rose2_source(monkeypatch):
    monkeypatch.delenv("ROSE2_SOURCE_ROOT", raising=False)
    args = Namespace(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode=VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
        room_segmentation_config={
            "backend": VERTICAL_FREE_GAP_CLOSURE_BACKEND,
            "source_root": "/definitely/missing/rose2",
            "require_upstream_source_for_strict": False,
        },
        ablation_name=None,
    )

    validate_strict_benchmark_assets(args)
