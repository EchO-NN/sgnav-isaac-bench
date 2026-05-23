from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.corridor import CorridorMergeConfig, merge_false_parallel_door_corridor_regions
from isaac_bench.mapping.online_roomseg.separator_candidates import SeparatorCandidate


def _door(cid: int, col: int, *, r0: int = 8, r1: int = 13) -> SeparatorCandidate:
    return SeparatorCandidate(
        candidate_id=cid,
        kind="line_extension_door_neck",
        p0_rc=np.asarray([r0, col], dtype=np.float32),
        p1_rc=np.asarray([r1, col], dtype=np.float32),
        theta=float(np.pi / 2.0),
        length_m=float((int(r1) - int(r0) + 1) * 0.1),
        confidence=0.9,
        source_segment_ids=[cid],
    )


def _horizontal_door(cid: int, row: int, *, c0: int = 20, c1: int = 37) -> SeparatorCandidate:
    return SeparatorCandidate(
        candidate_id=cid,
        kind="line_extension_door_neck",
        p0_rc=np.asarray([row, c0], dtype=np.float32),
        p1_rc=np.asarray([row, c1], dtype=np.float32),
        theta=0.0,
        length_m=float((int(c1) - int(c0) + 1) * 0.1),
        confidence=0.9,
        source_segment_ids=[cid],
    )


def test_strict_parallel_door_neck_edges_merge_only_when_opposite_edges_match():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:15] = 1
    labels[8:14, 16:30] = 2
    labels[8:14, 31:45] = 3
    labels[8:14, 46:62] = 4
    free = labels > 0
    wall = np.zeros_like(free)
    candidates = [_door(1, 15), _door(2, 30), _door(3, 45)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=wall,
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            isolated_chunk_max_area_m2=2.0,
            isolated_chunk_max_length_m=1.5,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["merge_events"]) == 1
    assert debug["merge_events"][0]["reason"] == "adjacent_corridor_like_merge"
    assert debug["merge_events"][0]["shared_edge_candidate"] == 2
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert merged[10, 20] == merged[10, 36]
    assert merged[10, 10] != merged[10, 20]
    assert merged[10, 36] != merged[10, 52]
    assert candidates[1].debug.get("rejected_after_corridor_merge")


def test_strict_parallel_door_neck_edges_find_labels_across_thick_separator_gap():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:14] = 1
    labels[8:14, 17:29] = 2
    labels[8:14, 32:44] = 3
    labels[8:14, 47:62] = 4
    free = labels > 0
    candidates = [_door(1, 15), _door(2, 30), _door(3, 45)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            door_neck_edge_touch_search_cells=2,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["strict_parallel_door_neck_edges"]) == 3
    assert len(debug["merge_events"]) == 1
    assert merged[10, 20] == merged[10, 36]
    assert candidates[1].debug.get("rejected_after_corridor_merge")


def test_corridor_like_merge_uses_label_side_completed_edge_length():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:15] = 1
    labels[8:14, 16:30] = 2
    labels[8:14, 31:45] = 3
    labels[8:14, 46:62] = 4
    free = labels > 0
    wall = np.zeros_like(free)
    wall[4:8, 15] = True
    wall[14:19, 15] = True
    candidates = [_door(1, 15), _door(2, 30), _door(3, 45)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=wall,
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["merge_events"]) == 1
    assert debug["merge_events"][0]["reason"] == "adjacent_corridor_like_merge"
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert merged[10, 20] == merged[10, 36]
    assert debug["corridor_like_labels"] == [2, 3]
    pair = next(item for item in debug["corridor_like_edge_pairs"] if item["label"] == 2)
    assert np.isclose(pair["edge_a_total_length_m"], 1.0)
    assert np.isclose(pair["edge_b_total_length_m"], 1.0)


def test_corridor_like_merge_uses_region_side_length_when_candidate_lengths_differ():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:15] = 1
    labels[8:14, 16:30] = 2
    labels[8:17, 31:45] = 3
    labels[8:17, 46:62] = 4
    free = labels > 0
    candidates = [_door(1, 15), _door(2, 30), _door(3, 45, r0=8, r1=16)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["merge_events"]) == 1
    assert debug["merge_events"][0]["reason"] == "adjacent_corridor_like_merge"
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert merged[10, 20] == merged[10, 36]


def test_strict_parallel_edge_merge_requires_complete_door_neck_edge_coverage():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:15] = 1
    labels[8:14, 16:30] = 2
    labels[8:14, 31:45] = 3
    labels[8:14, 46:62] = 4
    free = labels > 0
    candidates = [_door(1, 15), _door(2, 30), _door(3, 45, r0=6, r1=13)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["merge_events"]) == 0
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 4


def test_strict_parallel_edge_merge_rejects_second_edge_between_same_two_regions():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:30] = 1
    labels[8:14, 31:62] = 2
    free = labels > 0
    candidates = [_door(1, 30), _door(2, 45)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["merge_events"]) == 0
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 2


def test_strict_parallel_edge_merge_rejects_far_parallel_other_edges():
    labels = np.zeros((25, 120), dtype=np.int32)
    labels[8:14, 3:15] = 1
    labels[8:14, 16:30] = 2
    labels[8:14, 31:45] = 3
    labels[8:14, 90:110] = 4
    free = labels > 0
    candidates = [_door(1, 15), _door(2, 30), _door(3, 90), _door(4, 110)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len(debug["merge_events"]) == 0
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 4


def test_parallel_door_pair_without_complete_edge_rule_does_not_merge_small_middle_touching_two_rooms():
    labels = np.zeros((70, 120), dtype=np.int32)
    labels[8:58, 5:45] = 1
    labels[8:14, 46:58] = 2
    labels[8:58, 59:108] = 3
    free = labels > 0
    candidates = [_door(1, 45), _door(2, 58)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            isolated_chunk_max_area_m2=2.0,
            isolated_chunk_max_length_m=1.5,
            width_similarity_tol_m=2.0,
        ),
    )

    assert len(debug["merge_events"]) == 0
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert merged[10, 50] != merged[20, 20]
    assert merged[10, 50] != merged[20, 80]
    assert merged[20, 20] != merged[20, 80]
    assert sum(1 for c in candidates if c.debug.get("rejected_after_corridor_merge")) == 0
    assert debug["corridor_like_labels"] == [2]
    assert debug["sliver_merge_events"] == []


def test_shared_mask_edge_length_can_merge_adjacent_region_when_enabled():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[4:20, 3:15] = 1
    labels[4:20, 16:30] = 2
    labels[4:20, 31:45] = 3
    labels[4:20, 46:62] = 4
    free = labels > 0
    candidates = [_door(1, 15, r0=6, r1=17), _door(2, 30, r0=6, r1=17), _door(3, 45, r0=6, r1=17)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.15,
            parallel_edge_coverage_min_ratio=0.95,
            door_neck_edge_touch_search_cells=1,
            shared_mask_edge_length_match_enabled=True,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert len([v for v in np.unique(merged) if int(v) > 0]) == 1
    assert {event["adjacent_matching_edge_kind"] for event in debug["merge_events"] if "adjacent_matching_edge_kind" in event} == {"region_mask_parallel_edge"}
    assert merged[10, 10] == merged[10, 20] == merged[10, 36] == merged[10, 52]


def test_shared_mask_edge_match_uses_corridor_side_not_large_room_wall_run():
    labels = np.zeros((50, 100), dtype=np.int32)
    labels[5:25, 5:85] = 1
    labels[26:31, 40:61] = 5
    labels[32:45, 40:61] = 8
    free = labels > 0
    wall = np.zeros_like(free)
    wall[25, 5:86] = True
    candidates = [
        _horizontal_door(1, 25, c0=40, c1=60),
        _horizontal_door(2, 31, c0=40, c1=60),
    ]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=wall,
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.15,
            parallel_edge_coverage_min_ratio=0.95,
            door_neck_edge_touch_search_cells=2,
            shared_mask_edge_length_match_enabled=True,
            post_corridor_small_region_merge_enabled=False,
        ),
    )

    assert debug["corridor_like_labels"]
    assert debug["merge_events"] == []
    assert merged[10, 10] != merged[28, 50]
    rejection = next(item for item in debug["rejected_merge_events"] if item["shared_edge_candidate"] == 1)
    assert rejection["shared_edge_corridor_side"]["total_length_m"] < 3.0
    assert rejection["shared_edge_adjacent_side"]["total_length_m"] > 7.0


def test_regions_without_completed_parallel_edges_do_not_merge_by_area_only():
    labels = np.zeros((70, 120), dtype=np.int32)
    labels[8:58, 5:45] = 1
    labels[8:14, 45:58] = 2
    labels[8:58, 58:108] = 3
    free = labels > 0

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=[],
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            min_region_area_m2=2.0,
            isolated_chunk_max_area_m2=2.0,
            isolated_chunk_max_length_m=2.0,
            width_similarity_tol_m=2.0,
        ),
    )

    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert merged[10, 50] != merged[20, 20]
    assert merged[10, 50] != merged[20, 80]
    assert merged[20, 20] != merged[20, 80]
    assert debug["corridor_like_labels"] == []
    assert debug["merge_events"] == []
    assert debug["sliver_merge_events"] == []


def test_free_reachable_neighbors_alone_do_not_trigger_corridor_merge():
    labels = np.zeros((45, 45), dtype=np.int32)
    labels[18:22, 18:22] = 5
    labels[10:18, 18:22] = 1
    labels[26:34, 18:22] = 2
    labels[18:22, 10:14] = 3
    labels[18:22, 26:34] = 4
    free = labels > 0
    free[18:26, 18:22] = True
    free[18:22, 14:26] = True

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=[],
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(),
    )

    assert len([v for v in np.unique(merged) if int(v) > 0]) == 5
    assert debug["corridor_like_labels"] == []
    assert debug["merge_events"] == []
    assert debug["sliver_merge_events"] == []


def test_single_completed_edge_does_not_merge_without_corridor_like_pair():
    labels = np.zeros((70, 80), dtype=np.int32)
    labels[8:58, 5:45] = 1
    labels[20:44, 45:55] = 2
    free = labels > 0
    candidate = _door(1, 44, r0=20, r1=43)

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=[candidate],
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(),
    )

    assert len([v for v in np.unique(merged) if int(v) > 0]) == 2
    assert merged[25, 50] != merged[25, 20]
    assert debug["corridor_like_labels"] == []


def test_unknown_ratio_does_not_drive_corridor_merge_without_edge_evidence():
    labels = np.zeros((70, 120), dtype=np.int32)
    labels[8:58, 5:45] = 1
    labels[8:14, 45:58] = 2
    labels[8:58, 58:108] = 3
    free = labels > 0
    unknown = np.zeros_like(free)
    small_region = labels == 2
    unknown[6:16, 43:60] = True
    unknown &= ~small_region

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=[],
        free_clean=free,
        unknown_clean=unknown,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            post_corridor_small_region_max_area_m2=2.0,
            post_corridor_small_region_max_unknown_ratio=0.20,
        ),
    )

    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert debug["corridor_like_labels"] == []
    assert debug["sliver_merge_events"] == []


def test_just_merged_corridor_region_is_not_absorbed_by_post_small_region_merge():
    labels = np.zeros((25, 70), dtype=np.int32)
    labels[8:14, 3:15] = 1
    labels[8:14, 16:20] = 2
    labels[8:14, 21:25] = 3
    labels[8:14, 26:50] = 4
    free = labels > 0
    candidates = [_door(1, 15), _door(2, 20), _door(3, 25)]

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=candidates,
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(
            parallel_door_pair_max_distance_m=2.0,
            parallel_door_min_overlap_m=0.2,
            parallel_edge_length_tolerance_ratio=0.05,
            parallel_edge_coverage_min_ratio=0.95,
            post_corridor_small_region_max_area_m2=2.0,
        ),
    )

    assert len(debug["merge_events"]) == 1
    assert debug["merge_events"][0]["reason"] == "adjacent_corridor_like_merge"
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 3
    assert merged[10, 17] == merged[10, 22]
    assert merged[10, 10] != merged[10, 17]
    assert merged[10, 17] != merged[10, 30]
    assert debug["sliver_merge_events"] == []


def test_single_small_room_region_merges_to_adjacent_larger_region_without_parallel_pair():
    labels = np.zeros((35, 50), dtype=np.int32)
    labels[4:28, 4:24] = 1
    labels[14:20, 25:45] = 2
    free = labels > 0
    candidate = _door(1, 24)

    merged, debug = merge_false_parallel_door_corridor_regions(
        labels,
        accepted_candidates=[candidate],
        free_clean=free,
        wall_candidate_clean=np.zeros_like(free),
        filtered_lines=[],
        resolution_m=0.1,
        config=CorridorMergeConfig(),
    )

    assert len(debug["merge_events"]) == 0
    assert len([v for v in np.unique(merged) if int(v) > 0]) == 2
    assert merged[16, 30] != merged[16, 12]
    assert not candidate.debug.get("rejected_after_corridor_merge", False)
    assert debug["sliver_merge_events"] == []
