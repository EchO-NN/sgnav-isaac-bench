import numpy as np

from isaac_bench.mapping.ceiling_height_estimator import CeilingHeightEstimator, CeilingHeightEstimatorConfig


def test_ceiling_height_estimator_sets_active_z_to_90_percent_ceiling():
    cfg = CeilingHeightEstimatorConfig(
        min_points_per_frame=10,
        min_stable_frames=2,
        smooth_bins=0,
        lock_after_stable=True,
    )
    estimator = CeilingHeightEstimator(
        cfg,
        active_z_min_m=0.10,
        storage_z_max_m=3.20,
        active_z_max_fallback_m=2.00,
        active_z_max_ceiling_ratio=0.90,
    )
    ceiling = np.full((120,), 2.60, dtype=np.float32)
    furniture = np.full((80,), 1.20, dtype=np.float32)
    rel_z = np.concatenate([ceiling, furniture])

    first = estimator.update(rel_z)
    second = estimator.update(rel_z)

    assert first.height_m is not None
    assert 2.55 <= second.height_m <= 2.65
    assert second.stable
    assert second.locked
    assert np.isclose(second.active_z_max_m, 0.90 * second.height_m, atol=0.05)
    assert 2.30 <= second.debug["height_profile_active_z_max_m"] <= 2.40


def test_ceiling_height_estimator_uses_fallback_without_enough_points():
    estimator = CeilingHeightEstimator(
        CeilingHeightEstimatorConfig(min_points_per_frame=80),
        active_z_min_m=0.10,
        storage_z_max_m=3.20,
        active_z_max_fallback_m=2.00,
    )

    result = estimator.update(np.asarray([2.60, 2.62], dtype=np.float32))

    assert result.height_m is None
    assert result.active_z_max_m == 2.00
    assert result.debug["ceiling_height_reason"] == "insufficient_candidate_points"
