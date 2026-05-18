from __future__ import annotations

from pathlib import Path

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import UpstreamROSE2Config, UpstreamROSE2PurePythonSegmenter
from isaac_bench.mapping.vertical_profile import VerticalProfileMap, band_index
from isaac_bench.perception.object_memory import ObjectNode


def _fake_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "declutter-reconstruct"
    code = root / "code"
    code.mkdir(parents=True)
    for name in ("FFT_MQ.py", "minibatch.py", "parameters.py"):
        (code / name).write_text("# MIT upstream placeholder for vertical profile tests\n", encoding="utf-8")
    return root


def _map_info(shape: tuple[int, int]) -> MapInfo:
    h, w = shape
    return MapInfo(resolution_m=0.10, min_x=0.0, max_x=w * 0.10, min_y=0.0, max_y=h * 0.10, width=w, height=h)


def _segmenter(
    tmp_path: Path,
    shape: tuple[int, int],
    debug_dump: bool = False,
    exterior_margin_cells: int = 2,
) -> UpstreamROSE2PurePythonSegmenter:
    return UpstreamROSE2PurePythonSegmenter(
        UpstreamROSE2Config(
            source_root=str(_fake_source_root(tmp_path)),
            resolution_m=0.10,
            min_room_area_m2=0.5,
            hough_min_line_length_m=0.5,
            hough_line_gap_m=0.15,
            wall_min_support_ratio=0.10,
            wall_confidence_threshold=0.55,
            exterior_margin_cells=exterior_margin_cells,
            debug_dump=debug_dump,
            debug_dir=str(tmp_path / "debug" / "roomseg"),
        ),
        _map_info(shape),
    )


def _two_room_wall(*, doorway: bool = False, window: bool = False, exterior: bool = False):
    shape = (70, 90)
    free = np.zeros(shape, dtype=bool)
    free[10:60, 8:82] = True
    occupied = np.zeros(shape, dtype=bool)
    col = 10 if exterior else 44
    occupied[:, col : col + 2] = True
    free[:, col : col + 2] = False
    if doorway:
        occupied[32:42, col : col + 2] = False
        free[32:42, col : col + 2] = True
    if window:
        occupied[30:38, col : col + 2] = False
        free[30:38, col : col + 2] = False
    unknown = ~(free | occupied)
    profile = _profile_from_maps(occupied, free)
    if doorway:
        _mark_free(profile, (32, col), (42, col + 2), bands=("low", "robot_body"))
    if window:
        _mark_free(profile, (30, col), (38, col + 2), bands=("mid", "upper"))
    return occupied, free, unknown, profile, col


def _profile_from_occupied(occupied: np.ndarray) -> VerticalProfileMap:
    occ = np.zeros((4, *occupied.shape), dtype=np.uint16)
    observed = np.zeros_like(occ)
    for idx in range(4):
        occ[idx, occupied] = 4
        observed[idx, occupied] = 4
    return VerticalProfileMap.from_counts(occupied_count=occ, observed_count=observed)


def _profile_from_maps(occupied: np.ndarray, free: np.ndarray) -> VerticalProfileMap:
    profile = _profile_from_occupied(occupied)
    for name in ("low", "robot_body", "mid", "upper"):
        idx = band_index(name)
        profile.free_ray_count[idx, free] = 4
        profile.observed_count[idx, free] = 4
        profile.unknown_count[idx, free] = 0
    return profile


def _mark_free(profile: VerticalProfileMap, start: tuple[int, int], end: tuple[int, int], bands: tuple[str, ...]) -> None:
    r0, c0 = start
    r1, c1 = end
    for name in bands:
        idx = band_index(name)
        profile.free_ray_count[idx, r0:r1, c0:c1] = 5
        profile.observed_count[idx, r0:r1, c0:c1] = 5
        profile.unknown_count[idx, r0:r1, c0:c1] = 0


def _mark_partial_furniture(profile: VerticalProfileMap, mask: np.ndarray) -> None:
    for name in ("low", "robot_body"):
        idx = band_index(name)
        profile.occupied_count[idx, mask] = 3
        profile.observed_count[idx, mask] = 3
    for name in ("mid", "upper"):
        idx = band_index(name)
        profile.free_ray_count[idx, mask] = 5
        profile.observed_count[idx, mask] = 5


def test_table_chair_clutter_removed_by_vertical_carved_evidence(tmp_path):
    shape = (60, 80)
    free = np.zeros(shape, dtype=bool)
    free[8:52, 8:72] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[8:52, 8] = True
    occupied[8:52, 71] = True
    occupied[8, 8:72] = True
    occupied[51, 8:72] = True
    clutter = np.zeros(shape, dtype=bool)
    clutter[26:34, 34:46] = True
    occupied[clutter] = True
    free[occupied] = False
    unknown = ~(free | occupied)
    profile = _profile_from_occupied(occupied)
    _mark_partial_furniture(profile, clutter)

    segmenter = _segmenter(tmp_path, shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][clutter]) == 0
    assert np.count_nonzero(segmenter.last_debug["vertical_carved_map"][clutter]) == 0


def test_tall_cabinet_with_vertical_free_is_carved_from_roomseg_wall_map(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall()
    cabinet = np.zeros_like(occupied, dtype=bool)
    cabinet[20:45, 26:34] = True
    occupied[cabinet] = True
    free[cabinet] = False
    for idx in range(4):
        profile.occupied_count[idx, cabinet] = 5
        profile.observed_count[idx, cabinet] = 5
    object_memory = [
        ObjectNode(
            node_id=1,
            category="cabinet",
            center_world=(3.0, 3.5, 0.5),
            center_grid=(32, 30),
            confidence=0.9,
            observed_count=3,
            last_seen_step=1,
        )
    ]

    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert np.count_nonzero(segmenter.last_debug["furniture_suppression_mask"][cabinet]) == 0
    assert np.count_nonzero(segmenter.last_debug["initial_roomseg_occupied"][cabinet]) == 0
    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][cabinet]) == 0
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_free"][cabinet]) > 0


def test_object_memory_does_not_change_roomseg_vertical_free_or_wall_decisions(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall()
    object_memory = [
        ObjectNode(
            node_id=9,
            category="sofa",
            center_world=(4.4, 3.5, 0.5),
            center_grid=(35, col),
            confidence=0.9,
            observed_count=4,
            last_seen_step=1,
        )
    ]

    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert len([room for room in rooms if not room.stale]) >= 2
    assert np.count_nonzero(segmenter.last_debug["furniture_suppression_mask"][:, col : col + 2]) == 0
    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][:, col : col + 2]) > 0


def test_undetected_thick_interior_furniture_is_not_rose_structural_wall(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall()
    cabinet = np.zeros_like(occupied, dtype=bool)
    cabinet[20:45, 26:34] = True
    occupied[cabinet] = True
    free[cabinet] = False
    for idx in range(4):
        profile.occupied_count[idx, cabinet] = 5
        profile.observed_count[idx, cabinet] = 5
    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert np.count_nonzero(segmenter.last_debug["vertical_or_free_map"][cabinet]) > 0
    assert np.count_nonzero(segmenter.last_debug["structural_wall_mask"][cabinet]) == 0
    assert np.count_nonzero(segmenter.last_debug["vertical_carved_map"][cabinet]) == 0


def test_curtain_window_does_not_create_room_connection(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True)
    object_memory = [
        ObjectNode(
            node_id=2,
            category="curtain",
            center_world=(4.4, 3.4, 1.2),
            center_grid=(34, col),
            confidence=0.8,
            observed_count=2,
            last_seen_step=1,
        )
    ]

    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert len([room for room in rooms if not room.stale]) == 2
    assert not segmenter.last_debug["repaired_window_gaps"]
    assert not segmenter.last_debug["verified_doorway_gaps"]


def test_actual_doorway_remains_open_and_records_portal(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall(doorway=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert len([room for room in rooms if not room.stale]) == 2
    assert not segmenter.last_debug["verified_doorway_gaps"]
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_free"][:, _col : _col + 2]) > 0


def test_short_structural_wall_fragment_extends_to_split_observed_free_domain(tmp_path):
    shape = (70, 90)
    free = np.zeros(shape, dtype=bool)
    free[10:60, 8:82] = True
    occupied = np.zeros(shape, dtype=bool)
    occupied[28:42, 44:46] = True
    free[28:42, 44:46] = False
    unknown = ~(free | occupied)
    profile = _profile_from_maps(occupied, free)

    segmenter = _segmenter(tmp_path, shape)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)
    extended = [line for line in segmenter.last_debug["representative_lines"] if line.get("extended_to_observed_free_boundary")]

    assert len([room for room in rooms if not room.stale]) == 2
    assert extended
    assert max(float(line["length_m"]) for line in extended) > 3.0
    assert segmenter.last_debug["proposal_room_count"] == 2


def test_high_band_gap_but_floor_blocked_is_window_wall(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall(window=True)
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert not segmenter.last_debug["repaired_window_gaps"]
    assert segmenter.last_debug["vertical_free_overrides_occupied"] is True


def test_floor_to_ceiling_exterior_glass_does_not_merge_rooms(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall(window=True, exterior=True)
    object_memory = [
        ObjectNode(
            node_id=3,
            category="window",
            center_world=(1.0, 3.4, 1.0),
            center_grid=(34, col),
            confidence=0.9,
            observed_count=2,
            last_seen_step=1,
        )
    ]
    segmenter = _segmenter(tmp_path, occupied.shape, exterior_margin_cells=12)
    segmenter.update(occupied, free, occupied, unknown, step=1, object_memory=object_memory, vertical_profile=profile)

    assert not segmenter.last_debug["repaired_window_gaps"]
    assert not segmenter.last_debug["verified_doorway_gaps"]


def test_vertical_free_evidence_alone_cannot_mark_doorway(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall(window=True)
    _mark_free(profile, (30, 44), (38, 46), bands=("low", "robot_body", "mid", "upper"))
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)

    assert not segmenter.last_debug["repaired_window_gaps"]
    assert not segmenter.last_debug["verified_doorway_gaps"]
    assert np.count_nonzero(segmenter.last_debug["repaired_roomseg_free"][30:38, 44:46]) > 0


def test_rose2_receives_clean_wall_confidence_structural_map(tmp_path):
    occupied, free, unknown, profile, col = _two_room_wall()
    segmenter = _segmenter(tmp_path, occupied.shape)
    segmenter.update(occupied, free, occupied, unknown, step=1, vertical_profile=profile)
    structural = segmenter.last_debug["structural_wall_mask"]
    confidence = segmenter.last_debug["wall_confidence_map"]

    assert structural.dtype == bool
    assert confidence.shape == occupied.shape
    assert float(np.max(confidence[:, col : col + 2])) >= 0.55
    assert np.count_nonzero(structural[:, col : col + 2]) > 0


def test_final_room_masks_do_not_merge_through_windows(tmp_path):
    occupied, free, unknown, profile, _col = _two_room_wall(window=True)
    segmenter = _segmenter(tmp_path, occupied.shape, debug_dump=True)
    rooms = segmenter.update(occupied, free, occupied, unknown, step=7, vertical_profile=profile)
    layers = tmp_path / "debug" / "roomseg" / "episode" / "000007_rose2_layers.json"

    assert len([room for room in rooms if not room.stale]) >= 2
    assert layers.exists()
    assert not segmenter.last_debug["repaired_window_gaps"]
