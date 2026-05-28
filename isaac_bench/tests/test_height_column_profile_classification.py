import numpy as np

from isaac_bench.mapping.height_column_profile import HeightColumnProfileConfig, HeightColumnProfileMap


def _profile(shape=(5, 5), **kwargs):
    params = dict(
        z_min_m=0.10,
        z_max_m=3.20,
        storage_z_max_m=3.20,
        z_bin_size_m=0.05,
        active_z_max_m=2.00,
    )
    params.update(kwargs)
    cfg = HeightColumnProfileConfig(**params)
    return HeightColumnProfileMap.zeros(shape, cfg), cfg


def test_xy_free_requires_at_least_three_free_z_bins():
    profile, cfg = _profile()
    profile.free_ray_count[0:3, 2, 2] = 1
    profile.observed_count[0:3, 2, 2] = 1

    cls = profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)

    assert cls.vertical_free_xy[2, 2]
    assert not cls.unknown_xy[2, 2]


def test_xy_with_two_free_z_bins_is_unknown_not_free():
    profile, cfg = _profile()
    profile.free_ray_count[0:2, 2, 2] = 1
    profile.observed_count[0:2, 2, 2] = 1

    cls = profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)

    assert not cls.vertical_free_xy[2, 2]
    assert not cls.wall_xy[2, 2]
    assert cls.unknown_xy[2, 2]


def test_wall_requires_95_percent_occupied_among_active_bins():
    profile, cfg = _profile()
    active_bins = int(np.count_nonzero((profile.bin_centers_m >= cfg.active_z_min_m) & (profile.bin_centers_m <= cfg.active_z_max_m)))
    required = int(np.ceil(active_bins * cfg.wall_occupied_ratio_min))
    profile.occupied_count[0:required, 2, 2] = 1
    profile.observed_count[0:required, 2, 2] = 1

    cls = profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)

    assert cls.wall_xy[2, 2]
    assert np.isclose(cls.occupied_ratio_active_xy[2, 2], required / active_bins)


def test_unknown_active_bins_prevent_furniture_from_becoming_wall():
    profile, cfg = _profile()
    profile.occupied_count[0:8, 2, 2] = 1
    profile.observed_count[0:8, 2, 2] = 1

    cls = profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)

    assert not cls.wall_xy[2, 2]
    assert cls.observed_z_bin_count_xy[2, 2] == 8
    assert cls.occupied_ratio_observed_xy[2, 2] == 1.0
    assert cls.occupied_ratio_active_xy[2, 2] < cfg.wall_occupied_ratio_min
    assert cls.unknown_xy[2, 2]


def test_conflict_bins_are_not_counted_as_occupied_wall_bins():
    profile, cfg = _profile()
    active_bins = int(np.count_nonzero((profile.bin_centers_m >= cfg.active_z_min_m) & (profile.bin_centers_m <= cfg.active_z_max_m)))
    profile.occupied_count[0:active_bins, 2, 2] = 1
    profile.free_ray_count[0:2, 2, 2] = 1
    profile.observed_count[0:active_bins, 2, 2] = 1

    cls = profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)

    assert cls.conflict_bin_mask[0, 2, 2]
    assert cls.conflict_bin_mask[1, 2, 2]
    assert cls.conflict_z_bin_count_xy[2, 2] == 2
    assert not cls.wall_xy[2, 2]


def test_navigation_gate_removes_vertical_free_outside_navigation():
    profile, cfg = _profile(navigation_free_gate_dilation_cells=0)
    profile.free_ray_count[0:3, 2, 2] = 1
    profile.observed_count[0:3, 2, 2] = 1
    nav = np.zeros(profile.shape, dtype=bool)

    cls = profile.classify_columns(navigation_free_mask=nav, cfg=cfg)

    assert cls.vertical_free_raw_xy[2, 2]
    assert cls.vertical_free_outside_navigation_xy[2, 2]
    assert not cls.vertical_free_xy[2, 2]
    assert cls.unknown_xy[2, 2]


def test_hard_wall_overrides_vertical_free_conflict():
    profile, cfg = _profile(wall_occupied_ratio_min=0.70)
    profile.free_ray_count[0:3, 2, 2] = 1
    profile.occupied_count[3:30, 2, 2] = 1
    profile.observed_count[0:30, 2, 2] = 1

    cls = profile.classify_columns(navigation_free_mask=np.ones(profile.shape, dtype=bool), cfg=cfg)

    assert cls.wall_xy[2, 2]
    assert cls.free_wall_conflict_xy[2, 2]
    assert not cls.vertical_free_xy[2, 2]
