import numpy as np

from src.room_segmentation import CutCandidate, GridSpec
from src.room_segmentation.temporal import TemporalSmoother

from .test_helpers import make_segmenter, two_rooms_one_door


def _candidate(score):
    return CutCandidate(
        candidate_id=1,
        center_uv=(10, 10),
        center_xy=(0.5, 0.5),
        normal_xy=(1.0, 0.0),
        tangent_xy=(0.0, 1.0),
        width_m=0.8,
        cut_cells=[(10, 10), (10, 11)],
        left_seed_cells=[(9, 10)],
        right_seed_cells=[(11, 10)],
        final_score=score,
        is_soft_separator=score >= 0.55,
    )


def test_cut_needs_three_observations_for_hard_split():
    smoother = TemporalSmoother(grid_spec=GridSpec(0.05, (0, 0), 30, 30))
    c1 = _candidate(0.9)
    smoother.update_cuts([c1])
    assert not c1.is_hard_separator
    c2 = _candidate(0.9)
    smoother.update_cuts([c2])
    assert not c2.is_hard_separator
    c3 = _candidate(0.9)
    smoother.update_cuts([c3])
    assert c3.is_hard_separator


def test_room_id_stable_under_small_mask_change():
    free, wall, unknown = two_rooms_one_door()
    seg = make_segmenter(free.shape)
    out1 = seg.update_from_masks(free, wall, unknown, frame_id=1)
    free2 = free.copy()
    free2[12, 12] = False
    unknown2 = unknown.copy()
    unknown2[12, 12] = True
    out2 = seg.update_from_masks(free2, wall, unknown2, frame_id=2)
    assert len(np.unique(out1.room_id_map[out1.room_id_map > 0])) == len(np.unique(out2.room_id_map[out2.room_id_map > 0]))
