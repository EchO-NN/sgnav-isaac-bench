import json
from pathlib import Path

import numpy as np
import pytest

from isaac_bench.graph.room_context import RoomContextCache, prepare_room_context_for_frontier_scoring
from isaac_bench.mapping.coordinate_transform import MapInfo
from isaac_bench.mapping.room_segmentation import OnlineRoomSegmenter, RoomMask, RoomProposalState, RoomSegmentationConfig
from isaac_bench.mapping.room_segmentation_debug import save_rose2_roomseg_debug
from isaac_bench.mapping.rose2_room_segmentation import OnlineROSE2RoomSegmenter
from isaac_bench.mapping.vertical_profile import VerticalProfileMap, band_index
from isaac_bench.metrics.result_schema import BenchmarkAssetError, validate_strict_benchmark_assets
from isaac_bench.perception.object_memory import ObjectMemory


def _config(tmp_path=None):
    return RoomSegmentationConfig(
        resolution_m=0.10,
        min_room_area_m2=0.5,
        min_wall_line_length_m=0.5,
        debug_dump=False,
        debug_dir=str(tmp_path or "debug/roomseg_rose2"),
        rose2={
            "hough_min_line_length_m": 0.5,
            "hough_line_gap_m": 0.10,
            "wall_min_support_ratio": 0.10,
            "clutter_component_max_area_m2": 0.20,
        },
    )


def _open_map(shape=(40, 40)):
    free = np.ones(shape, dtype=bool)
    occ = np.zeros(shape, dtype=bool)
    unknown = np.zeros(shape, dtype=bool)
    return occ, free, unknown


def _split_map():
    occ, free, unknown = _open_map()
    occ[:, 20] = True
    free[:, 20] = False
    return occ, free, unknown


def test_rose2_open_living_room_not_oversplit():
    occ, free, unknown = _open_map()
    segmenter = OnlineROSE2RoomSegmenter(_config())

    rooms = segmenter.update(occ, free, occ, unknown, step=1, object_memory=[])

    assert len([room for room in rooms if not room.stale]) == 1
    assert segmenter.last_debug["algorithm"] == "rose2_structure"
    assert segmenter.last_debug["source"] == "rose2_structure"


def test_rose2_kitchen_living_structural_boundary_preserved():
    occ, free, unknown = _split_map()
    segmenter = OnlineROSE2RoomSegmenter(_config())

    rooms = segmenter.update(occ, free, occ, unknown, step=1, object_memory=[])

    assert len([room for room in rooms if not room.stale]) == 2
    assert segmenter.last_debug["num_representative_lines"] >= 1
    assert all(room.source == "rose2_structure" for room in rooms)


def test_rose2_no_merge_finalization_keeps_proposals_separate():
    labels = np.zeros((12, 20), dtype=np.int32)
    labels[2:10, 2:10] = 1
    labels[2:10, 10:18] = 2
    free = labels > 0
    segmenter = OnlineROSE2RoomSegmenter(
        RoomSegmentationConfig(
            resolution_m=0.10,
            min_room_area_m2=0.1,
            finalization_mode="no_merge",
            open_boundary_merge=False,
            use_premerge_labels_for_open_plan_merge=False,
        )
    )
    state = RoomProposalState(
        proposal_labels=labels,
        structural_free_mask=free,
        structural_obstacle_mask=np.zeros_like(free),
        unknown_mask=~free,
        distance_m=np.ones_like(labels, dtype=np.float32),
        step=3,
        debug={"algorithm": "rose2_structure", "source": "rose2_structure"},
    )

    rooms = segmenter.finalize_proposals(
        state,
        proposal_semantic_labels={
            1: {"category": "living_room", "label_reliability": 0.9},
            2: {"category": "living_room", "label_reliability": 0.9},
        },
    )

    assert len([room for room in rooms if not room.stale]) == 2
    assert segmenter.last_debug["finalization_mode"] == "no_merge"
    assert segmenter.last_debug["merge_disabled"] is True
    assert segmenter.last_debug["proposal_room_count"] == 2
    assert segmenter.last_debug["final_room_count"] == 2
    assert segmenter.last_debug["merge_operations"] == []
    assert all(room.metadata["proposal_labels"] == [idx] for idx, room in enumerate(rooms, start=1))


def test_rose2_visualization_outputs_layer_json(tmp_path):
    occ, free, unknown = _split_map()
    segmenter = OnlineROSE2RoomSegmenter(_config(tmp_path))
    rooms = segmenter.update(occ, free, occ, unknown, step=7, object_memory=[])

    png, layers = save_rose2_roomseg_debug(
        out_dir=tmp_path,
        episode_id="synthetic",
        step=7,
        occupancy=occ,
        room_masks=rooms,
        debug=segmenter.last_debug,
    )

    payload = json.loads(Path(layers).read_text(encoding="utf-8"))
    assert Path(png).exists()
    assert payload["algorithm"] == "rose2_structure"
    assert payload["num_representative_lines"] >= 1
    assert payload["num_final_rooms"] == len(rooms)


def test_watershed_vertical_free_uses_0p2_2p0_profile_not_navigation_free():
    shape = (30, 40)
    nav_free = np.zeros(shape, dtype=bool)
    nav_free[4:26, 4:36] = True
    nav_occ = np.zeros(shape, dtype=bool)
    unknown = ~nav_free
    profile = VerticalProfileMap.zeros(shape)
    robot_band = band_index("robot_body")
    mid_band = band_index("mid")
    profile.observed_count[robot_band, 4:26, 4:36] = 1
    profile.observed_count[mid_band, 4:26, 4:36] = 1
    profile.free_ray_count[robot_band, 4:26, 4:18] = 1
    profile.free_ray_count[robot_band, 4:26, 22:36] = 1
    profile.free_ray_count[mid_band, 4:26, 4:18] = 1
    profile.free_ray_count[mid_band, 4:26, 22:36] = 1
    profile.occupied_count[robot_band, 4:26, 18:22] = 1
    profile.occupied_count[mid_band, 4:26, 18:22] = 1
    profile.unknown_count[:, :, :] = (profile.observed_count == 0).astype(np.uint16)
    cfg = RoomSegmentationConfig(
        algorithm="legacy_watershed_vertical_free_ablation",
        source_grid="vertical_profile_free_0p2_2p0",
        resolution_m=0.10,
        min_room_area_m2=0.25,
        min_observed_free_cells=5,
        use_structural_obstacle_mask=False,
        morphology_close_radius_m=0.0,
        morphology_open_radius_m=0.0,
    )
    segmenter = OnlineRoomSegmenter(cfg)

    proposals, state = segmenter.build_proposals(
        nav_occ,
        nav_free,
        nav_occ,
        unknown,
        step=1,
        object_memory=[],
        vertical_profile=profile,
    )

    assert state.debug["roomseg_input_source"] == "vertical_profile_free_0p2_2p0_watershed"
    assert state.debug["vertical_or_free_cells"] == int(np.count_nonzero(state.structural_free_mask))
    assert len(proposals) == 2
    assert np.all(~state.structural_free_mask[4:26, 18:22])


def test_room_segmentation_called_only_before_frontier_scoring():
    occ, free, unknown = _split_map()
    map_info = MapInfo(resolution_m=0.10, min_x=0.0, max_x=4.0, min_y=0.0, max_y=4.0, width=40, height=40)
    segmenter = OnlineROSE2RoomSegmenter(_config())

    result = prepare_room_context_for_frontier_scoring(
        step_idx=42,
        object_memory=ObjectMemory(min_valid_confidence=0.0),
        room_segmenter=segmenter,
        room_labeler=None,
        map_info=map_info,
        previous_room_context=RoomContextCache(),
        strict_benchmark=False,
        occupancy=occ,
        observed_free_mask=free,
        obstacle_mask=occ,
        unknown_mask=unknown,
    )

    assert result.segmentation_ran is True
    assert result.room_segmentation_debug["algorithm"] == "rose2_structure"
    metadata = result.metadata()
    assert metadata["room_segmentation_called_for"] == "frontier_scoring_pre_hook"
    assert metadata["frontier_scoring_after_room_context"] is True


def test_rose2_no_oracle_rooms_in_strict_mode():
    class Args:
        strict_benchmark = True
        room_map_mode = "online_geometry_watershed"
        ablation_name = ""
        detector = "none"
        segmenter = "none"
        llm_enabled = False
        sim_backend = "map"

    with pytest.raises(BenchmarkAssetError, match="rose2_source_faithful_v1"):
        validate_strict_benchmark_assets(Args())
