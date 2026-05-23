from .test_helpers import make_segmenter, open_room, run_masks, two_rooms_one_door


def test_same_room_visibility_drop_low():
    free, wall, unknown = open_room()
    seg = make_segmenter(free.shape)
    state = seg.vertical_map_builder.from_masks(observed_free_mask=free, obstacle_mask=wall, unknown_mask=unknown, frame_id=1)
    structural = seg.structural_wall_estimator.update(state)
    drop = seg.visibility.visibility_drop([(30, 30)], [(32, 34)], structural)
    assert drop < 0.6


def test_across_door_candidate_has_visibility_drop_signal():
    free, wall, unknown = two_rooms_one_door()
    out = run_masks(free, wall, unknown)
    assert out.cut_candidates
    assert max(c.visibility_drop_score for c in out.cut_candidates) >= 0.2


def test_unknown_limited_visibility_does_not_become_certain():
    free, wall, unknown = two_rooms_one_door()
    unknown[:, 45:55] = True
    free[:, 45:55] = False
    wall[:, 45:55] = False
    out = run_masks(free, wall, unknown)
    if out.cut_candidates:
        assert max(c.visibility_drop_score for c in out.cut_candidates) <= 0.75

