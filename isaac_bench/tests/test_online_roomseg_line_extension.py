from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.separator_candidates import (
    DoorNeckConfig,
    LineExtensionConfig,
    build_door_neck_candidates_from_extensions,
    extend_wall_lines_once,
)
from isaac_bench.mapping.online_roomseg.wall_lines import FilteredWallLine


def _line(line_id: int, p0, p1) -> FilteredWallLine:
    return FilteredWallLine(
        line_id=line_id,
        p0_rc=np.asarray(p0, dtype=np.float32),
        p1_rc=np.asarray(p1, dtype=np.float32),
        theta=0.0,
        normal_theta=float(np.pi / 2.0),
        length_m=0.5,
        support_ratio=1.0,
        mean_wall_score=1.0,
        thickness_m=0.1,
        source_segment_ids=[line_id],
        endpoint_quality={"p0_support_m": 0.2, "p1_support_m": 0.2},
        confidence=0.95,
    )


def test_pass1_extension_hits_real_wall_and_builds_door_neck_candidate():
    free = np.zeros((30, 40), dtype=bool)
    free[10, 11:20] = True
    wall = np.zeros_like(free)
    wall[10, 5:11] = True
    wall[10, 20:26] = True
    unknown = ~(free | wall)

    hits, debug = extend_wall_lines_once(
        [_line(1, [10, 5], [10, 10])],
        free_clean=free,
        wall_target_mask=wall,
        virtual_target_mask=None,
        unknown_clean=unknown,
        resolution_m=0.1,
        pass_id=1,
        config=LineExtensionConfig(min_extension_m=0.4, max_extension_m=1.6, free_ratio_min=0.65, unknown_ratio_max=0.1),
    )
    accepted = [hit for hit in hits if hit.reject_reason is None]
    candidates, cand_debug = build_door_neck_candidates_from_extensions(
        accepted,
        accepted_virtual_targets=None,
        resolution_m=0.1,
        config=DoorNeckConfig(min_confidence=0.4),
    )

    assert debug["accepted_extension_count"] == 1
    assert accepted[0].hit_type == "real_wall"
    assert 0.8 <= accepted[0].length_m <= 1.1
    assert len(candidates) == 1
    assert candidates[0].kind == "line_extension_door_neck"
    assert cand_debug["candidate_count"] == 1


def test_extension_rejects_unknown_only_gap():
    free = np.zeros((30, 40), dtype=bool)
    free[10, 11:14] = True
    wall = np.zeros_like(free)
    wall[10, 5:11] = True
    unknown = ~(free | wall)

    hits, _ = extend_wall_lines_once(
        [_line(1, [10, 5], [10, 10])],
        free_clean=free,
        wall_target_mask=wall,
        virtual_target_mask=None,
        unknown_clean=unknown,
        resolution_m=0.1,
        pass_id=1,
        config=LineExtensionConfig(min_extension_m=0.4, max_extension_m=1.6, unknown_ratio_max=0.1),
    )

    assert any(hit.reject_reason in {"reject_extension_hit_unknown", "reject_extension_no_wall_or_virtual_door_hit"} for hit in hits)
    assert all(not np.array_equal(hit.p_start_rc, hit.p_hit_rc) for hit in hits)
    assert all(hit.debug["sampled_cell_count"] > 0 for hit in hits)


def test_extension_hit_snaps_from_dilated_halo_to_actual_wall_cell():
    free = np.zeros((30, 40), dtype=bool)
    free[10, 11:20] = True
    wall = np.zeros_like(free)
    wall[10, 5:11] = True
    wall[11, 20] = True
    unknown = ~(free | wall)

    hits, debug = extend_wall_lines_once(
        [_line(1, [10, 5], [10, 10])],
        free_clean=free,
        wall_target_mask=wall,
        virtual_target_mask=None,
        unknown_clean=unknown,
        resolution_m=0.1,
        pass_id=1,
        config=LineExtensionConfig(
            min_extension_m=0.4,
            max_extension_m=1.6,
            hit_radius_m=0.2,
            free_ratio_min=0.65,
            unknown_ratio_max=0.1,
        ),
    )
    accepted = [hit for hit in hits if hit.reject_reason is None]

    assert debug["accepted_extension_count"] == 1
    assert accepted[0].hit_type == "real_wall"
    assert accepted[0].p_hit_rc.tolist() == [11.0, 20.0]
    assert accepted[0].debug["hit_cell_snap_target"] == "real_wall"
