import numpy as np

from .test_helpers import run_masks, two_rooms_one_door


def test_mask_assignment_confidence_and_soft_masks_are_valid():
    free, wall, unknown = two_rooms_one_door()
    out = run_masks(free, wall, unknown)
    assert out.room_id_map.shape == free.shape
    assert out.room_confidence_map.shape == free.shape
    assert not np.isnan(out.room_confidence_map).any()
    assert float(out.room_confidence_map.min()) >= 0.0
    assert float(out.room_confidence_map.max()) <= 1.0
    assert out.room_soft_masks
    assert np.all(out.room_id_map[unknown] == -1)

