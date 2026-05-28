import numpy as np

from isaac_bench.mapping.height_column_profile import HeightColumnProfileConfig, HeightColumnProfileMap
from isaac_bench.mapping.height_profile_door_detector import (
    HeightProfileDoorConfig,
    classify_door_seed_at_xy,
    detect_height_profile_doors,
)


def _classification_for_states(states):
    z_max = 0.10 + 0.05 * max(38, len(states))
    cfg = HeightColumnProfileConfig(
        z_max_m=z_max,
        storage_z_max_m=z_max,
        active_z_max_m=z_max,
        use_navigation_free_gate=False,
    )
    profile = HeightColumnProfileMap.zeros((1, 1), cfg)
    states = list(states)
    for idx, state in enumerate(states):
        if state == "F":
            profile.free_ray_count[idx, 0, 0] = 1
            profile.observed_count[idx, 0, 0] = 1
        elif state == "O":
            profile.occupied_count[idx, 0, 0] = 1
            profile.observed_count[idx, 0, 0] = 1
    return profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)


def _door_states():
    states = ["F"] * 34 + ["O", "O", "O", "U"]
    return states


def _door_classification(seed_cells, shape=(30, 40)):
    cfg = HeightColumnProfileConfig(use_navigation_free_gate=False)
    profile = HeightColumnProfileMap.zeros(shape, cfg)
    for r, c in seed_cells:
        profile.free_ray_count[0:34, r, c] = 1
        profile.observed_count[0:34, r, c] = 1
        profile.occupied_count[34:37, r, c] = 1
        profile.observed_count[34:37, r, c] = 1
    return profile.classify_columns(navigation_free_mask=np.ones(shape, dtype=bool), cfg=cfg)


def _masks(shape=(30, 40), walls=((15, 12), (15, 28))):
    free = np.zeros(shape, dtype=bool)
    free[4:26, 4:36] = True
    wall = np.zeros(shape, dtype=bool)
    for r, c in walls:
        wall[r, c] = True
        free[r, c] = False
    unknown = ~(free | wall)
    return free, wall, unknown


def test_door_seed_accepts_free_from_0p1_then_occupied_above_1p8_then_unknown():
    cls = _classification_for_states(_door_states())
    ev = classify_door_seed_at_xy(cls.z_state[:, 0, 0], np.arange(cls.z_state.shape[0]) * 0.05 + 0.125, HeightProfileDoorConfig())

    assert ev.accepted
    assert ev.first_occupied_z_m >= 1.8
    assert ev.top_occupied_bins == 3


def test_door_seed_rejects_occupied_below_1p8():
    cls = _classification_for_states(["F"] * 10 + ["O", "O", "O"] + ["U"] * 25)
    ev = classify_door_seed_at_xy(cls.z_state[:, 0, 0], np.arange(cls.z_state.shape[0]) * 0.05 + 0.125, HeightProfileDoorConfig())

    assert not ev.accepted
    assert ev.reject_reason == "top_occupied_too_low"


def test_door_seed_rejects_unknown_below_top_occupied():
    states = ["F"] * 10 + ["U"] + ["F"] * 23 + ["O", "O", "O", "U"]
    cls = _classification_for_states(states)
    ev = classify_door_seed_at_xy(cls.z_state[:, 0, 0], np.arange(cls.z_state.shape[0]) * 0.05 + 0.125, HeightProfileDoorConfig())

    assert not ev.accepted
    assert ev.reject_reason == "unknown_before_top_occupied"


def test_door_seed_rejects_free_after_top_occupied():
    states = ["F"] * 34 + ["O", "O", "O", "U", "F"]
    cls = _classification_for_states(states)
    ev = classify_door_seed_at_xy(cls.z_state[:, 0, 0], np.arange(cls.z_state.shape[0]) * 0.05 + 0.125, HeightProfileDoorConfig())

    assert not ev.accepted
    assert ev.reject_reason == "non_unknown_after_unknown_tail"


def test_door_seed_ignores_non_unknown_bins_above_active_range():
    states = ["F"] * 34 + ["O", "O", "O", "U", "F"]
    cls = _classification_for_states(states)
    active = np.zeros(cls.z_state.shape[0], dtype=bool)
    active[:38] = True

    ev = classify_door_seed_at_xy(
        cls.z_state[:, 0, 0],
        np.arange(cls.z_state.shape[0]) * 0.05 + 0.125,
        HeightProfileDoorConfig(),
        active_z_bin_mask=active,
    )

    assert ev.accepted


def test_door_seed_rejects_conflict_below_top_occupied():
    cls = _classification_for_states(["F"] * 10 + ["O"] * 28)
    cls.z_state[10, 0, 0] = 3
    ev = classify_door_seed_at_xy(cls.z_state[:, 0, 0], np.arange(cls.z_state.shape[0]) * 0.05 + 0.125, HeightProfileDoorConfig())

    assert not ev.accepted
    assert ev.reject_reason == "conflict_before_top_occupied"


def test_door_component_projects_thick_seed_to_single_centerline():
    shape = (30, 40)
    seed_cells = [(14, c) for c in range(18, 22)] + [(15, c) for c in range(18, 22)]
    cls = _door_classification(seed_cells, shape)
    free, wall, unknown = _masks(shape, walls=((14, 12), (14, 28), (15, 12), (15, 28)))

    result = detect_height_profile_doors(
        classification=cls,
        free_clean=free,
        wall_clean=wall,
        unknown_clean=unknown,
        resolution_m=0.05,
        config=HeightProfileDoorConfig(),
    )

    assert result.candidates
    candidate = result.candidates[0]
    rows = {r for r, _c in candidate.seed_projected_centerline_cells}
    assert len(rows) == 1


def test_door_line_extends_to_two_walls():
    seed_cells = [(15, c) for c in range(18, 22)]
    cls = _door_classification(seed_cells)
    free, wall, unknown = _masks(walls=((15, 12), (15, 28)))

    result = detect_height_profile_doors(
        classification=cls,
        free_clean=free,
        wall_clean=wall,
        unknown_clean=unknown,
        resolution_m=0.05,
        config=HeightProfileDoorConfig(),
    )

    assert result.candidates[0].accepted
    assert result.candidates[0].wall_anchor_a is not None
    assert result.candidates[0].wall_anchor_b is not None
    assert np.count_nonzero(result.door_cut_mask) > 0


def test_door_line_rejects_if_width_out_of_range():
    seed_cells = [(15, c) for c in range(18, 22)]
    cls = _door_classification(seed_cells)
    free, wall, unknown = _masks(walls=((15, 5), (15, 35)))

    result = detect_height_profile_doors(
        classification=cls,
        free_clean=free,
        wall_clean=wall,
        unknown_clean=unknown,
        resolution_m=0.05,
        config=HeightProfileDoorConfig(door_width_max_m=0.80),
    )

    assert not result.candidates[0].accepted
    assert result.candidates[0].reject_reason == "door_width_out_of_range"


def test_door_line_batch_rejects_intersection_with_other_door():
    horizontal = [(15, c) for c in range(18, 20)]
    vertical = [(r, 20) for r in range(11, 14)]
    cls = _door_classification(horizontal + vertical)
    free, wall, unknown = _masks(
        walls=((15, 12), (15, 28), (8, 20), (23, 20)),
    )

    result = detect_height_profile_doors(
        classification=cls,
        free_clean=free,
        wall_clean=wall,
        unknown_clean=unknown,
        resolution_m=0.05,
        config=HeightProfileDoorConfig(),
    )

    assert result.candidates
    assert any(not c.accepted and c.reject_reason == "extension_intersects_other_door_candidate" for c in result.candidates)


def test_single_seed_can_infer_direction_from_two_walls():
    cls = _door_classification([(15, 20)])
    free, wall, unknown = _masks(walls=((15, 12), (15, 28)))

    result = detect_height_profile_doors(
        classification=cls,
        free_clean=free,
        wall_clean=wall,
        unknown_clean=unknown,
        resolution_m=0.05,
        config=HeightProfileDoorConfig(),
    )

    assert result.candidates[0].accepted
    assert result.candidates[0].wall_anchor_a is not None
    assert result.candidates[0].wall_anchor_b is not None
