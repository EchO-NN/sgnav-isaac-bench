import numpy as np

from isaac_bench.mapping.room_segmentation import OnlineRoomSegmenter, RoomSegmentationConfig


def test_room_ids_stay_stable_after_small_observation_change():
    cfg = RoomSegmentationConfig(
        resolution_m=0.1,
        min_observed_free_cells=20,
        min_room_area_m2=0.2,
        morphology_close_radius_m=0.0,
        morphology_open_radius_m=0.0,
        seed_min_clearance_m=0.2,
        seed_min_distance_m=2.0,
        doorway_width_min_m=0.4,
        doorway_width_max_m=1.4,
        doorway_clearance_max_m=2.5,
    )
    segmenter = OnlineRoomSegmenter(cfg)
    free = np.zeros((60, 60), dtype=bool)
    free[8:52, 6:25] = True
    free[8:52, 35:54] = True
    obstacle = np.zeros_like(free)
    unknown = ~free

    first = [room.room_id for room in segmenter.update(obstacle, free, obstacle, unknown, step=0) if not room.stale]
    free2 = free.copy()
    free2[8:52, 25] = True
    unknown2 = ~free2
    second = [room.room_id for room in segmenter.update(obstacle, free2, obstacle, unknown2, step=5) if not room.stale]

    assert first
    assert set(first).issubset(set(second))
