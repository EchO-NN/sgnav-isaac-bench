from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import RoomProposalState
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import (
    UpstreamROSE2Config,
    UpstreamROSE2PurePythonSegmenter,
    occupancy_to_rose_image,
    rose_polygons_to_room_masks,
)
from isaac_bench.metrics.result_schema import BenchmarkAssetError, validate_strict_benchmark_assets


FORBIDDEN_ROS_MODULES = ("rospy", "nav_msgs", "jsk_recognition_msgs", "roslaunch")


def _fake_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "declutter-reconstruct"
    code = root / "code"
    code.mkdir(parents=True)
    for name in ("FFT_MQ.py", "minibatch.py", "parameters.py"):
        (code / name).write_text("# MIT upstream placeholder for adapter tests\n", encoding="utf-8")
    return root


def _map_info(shape: tuple[int, int], resolution: float = 0.10) -> MapInfo:
    h, w = shape
    return MapInfo(resolution_m=resolution, min_x=0.0, max_x=float(w) * resolution, min_y=0.0, max_y=float(h) * resolution, width=w, height=h)


def test_upstream_rose2_adapter_imports_without_ros():
    for name in FORBIDDEN_ROS_MODULES:
        sys.modules.pop(name, None)

    module = importlib.import_module("isaac_bench.mapping.upstream_rose2_pure_python_adapter")

    assert module.UpstreamROSE2PurePythonSegmenter is not None
    assert all(name not in sys.modules for name in FORBIDDEN_ROS_MODULES)


def test_upstream_rose2_missing_source_strict_fails(tmp_path):
    args = Namespace(
        strict_benchmark=True,
        allow_debug_fallbacks=False,
        detector="none",
        segmenter="none",
        sim_backend="isaac",
        sgnav_mode="legacy",
        llm_enabled=True,
        room_map_mode="upstream_rose2_vertical_or_free",
        room_segmentation_config={
            "backend": "rose2_source_external",
            "source_root": str(tmp_path / "missing"),
            "upstream_repo_env": "ROSE2_SOURCE_ROOT",
            "require_upstream_source_for_strict": True,
        },
        ablation_name=None,
    )

    with pytest.raises(BenchmarkAssetError, match="Missing upstream ROSE2 pure-Python source root"):
        validate_strict_benchmark_assets(args)


def test_upstream_rose2_synthetic_two_rooms(tmp_path):
    shape = (70, 90)
    free = np.zeros(shape, dtype=bool)
    free[10:60, 8:82] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 44:46] = True
    free[:, 44:46] = False
    free[32:42, 44:46] = True
    occupied[32:42, 44:46] = False
    unknown = ~(free | occupied)
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(_fake_source_root(tmp_path)),
            resolution_m=0.10,
            min_room_area_m2=0.5,
            hough_min_line_length_m=0.5,
            hough_line_gap_m=0.15,
            wall_min_support_ratio=0.10,
            debug_dump=False,
        ),
        _map_info(shape),
    )

    rooms = segmenter.update(occupied, free, occupied, unknown, step=1)

    assert len([room for room in rooms if not room.stale]) == 2
    assert segmenter.last_debug["algorithm"] == "rose2_source_form"
    assert segmenter.last_debug["source_mode"] == "source_form_no_ros"
    assert segmenter.last_debug["strict_fallback_used"] is False


def test_upstream_rose2_open_living_not_oversplit(tmp_path):
    shape = (60, 80)
    free = np.zeros(shape, dtype=bool)
    free[8:52, 8:72] = True
    occupied = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in [(18, 20, 23, 28), (34, 42, 42, 50), (20, 58, 25, 65)]:
        occupied[r0:r1, c0:c1] = True
        free[r0:r1, c0:c1] = False
    unknown = ~(free | occupied)
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(_fake_source_root(tmp_path)),
            resolution_m=0.10,
            min_room_area_m2=0.5,
            hough_min_line_length_m=0.5,
            debug_dump=False,
        ),
        _map_info(shape),
    )

    rooms = segmenter.update(occupied, free, occupied, unknown, step=1)

    assert len([room for room in rooms if not room.stale]) == 1


def test_final_room_masks_absorb_rose_free_cells_left_unlabeled_by_boundaries(tmp_path):
    shape = (40, 48)
    labels = np.zeros(shape, dtype=np.int32)
    labels[8:32, 8:22] = 1
    labels[8:32, 26:40] = 2
    rose_free = np.zeros(shape, dtype=bool)
    rose_free[8:32, 8:40] = True
    unknown = ~rose_free
    boundary_sliver = np.zeros(shape, dtype=bool)
    boundary_sliver[8:32, 22:26] = True
    assert np.all(labels[boundary_sliver] == 0)

    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(_fake_source_root(tmp_path)),
            resolution_m=0.10,
            min_room_area_m2=0.5,
            finalization_mode="no_merge",
            debug_dump=False,
        ),
        _map_info(shape),
    )
    state = RoomProposalState(
        proposal_labels=labels,
        structural_free_mask=rose_free,
        structural_obstacle_mask=np.zeros(shape, dtype=bool),
        unknown_mask=unknown,
        distance_m=np.ones(shape, dtype=np.float32),
        step=3,
    )

    rooms = segmenter.finalize_proposals(state)
    room_union = np.zeros(shape, dtype=bool)
    for room in rooms:
        room_union |= np.asarray(room.mask, dtype=bool)

    assert np.all(room_union[boundary_sliver])
    assert np.all(segmenter.last_debug["final_room_label_map"][boundary_sliver] > 0)
    assert segmenter.last_debug["room_free_absorption"]["before_unlabeled_free_cells"] == int(np.count_nonzero(boundary_sliver))
    assert segmenter.last_debug["room_free_absorption"]["after_unlabeled_free_cells"] == 0


def test_upstream_rose2_debug_artifact_contract(tmp_path):
    shape = (35, 45)
    free = np.zeros(shape, dtype=bool)
    free[5:30, 5:40] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[:, 22:23] = True
    free[:, 22:23] = False
    free[16:20, 22:23] = True
    occupied[16:20, 22:23] = False
    unknown = ~(free | occupied)
    out_dir = tmp_path / "roomseg"
    segmenter = UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(_fake_source_root(tmp_path)),
            resolution_m=0.10,
            min_room_area_m2=0.5,
            hough_min_line_length_m=0.5,
            wall_min_support_ratio=0.10,
            debug_dump=True,
            debug_dir=str(out_dir),
        ),
        _map_info(shape),
    )

    segmenter.update(occupied, free, occupied, unknown, step=7)
    png = out_dir / "episode" / "000007_rose2_rooms.png"
    layers = out_dir / "episode" / "000007_rose2_layers.json"
    payload = json.loads(layers.read_text(encoding="utf-8"))

    assert png.exists()
    assert payload["algorithm"] == "rose2_source_form"
    assert payload["source_mode"] == "source_form_no_ros"
    assert "num_rooms" in payload
    assert "num_wall_lines" in payload
    assert "main_directions" in payload
    assert "room_masks" in payload
    assert "room_labels" in payload
    assert "vertical_or_free_z_min_m" in payload
    assert "vertical_free_overrides_occupied" in payload
    assert "vertical_free_overridden_occupied_cells" in payload
    assert payload["strict_fallback_used"] is False
    assert payload["source_repository"] == "https://github.com/goldleaf3i/declutter-reconstruct"
    assert payload["source_provenance"]["available"] is True
    assert segmenter.last_debug["source_provenance"]["available"] is True
    assert all(item["sha256"] for item in segmenter.last_debug["source_provenance"]["files"])


def test_upstream_rose2_room_mask_coordinate_roundtrip():
    map_info = MapInfo(resolution_m=0.5, min_x=10.0, max_x=15.0, min_y=20.0, max_y=25.0, width=10, height=10)
    polygon = np.asarray(
        [
            [11.0, 21.0],
            [13.0, 21.0],
            [13.0, 23.0],
            [11.0, 23.0],
        ],
        dtype=np.float32,
    )

    rooms = rose_polygons_to_room_masks([polygon], (10, 10), map_info)
    image = occupancy_to_rose_image(np.zeros((10, 10), dtype=bool), np.ones((10, 10), dtype=bool))

    assert image.dtype == np.uint8
    assert len(rooms) == 1
    rr, cc = np.nonzero(rooms[0].mask)
    assert int(rr.min()) == 4
    assert int(cc.min()) == 2
    assert int(rr.max()) == 8
    assert int(cc.max()) == 6
