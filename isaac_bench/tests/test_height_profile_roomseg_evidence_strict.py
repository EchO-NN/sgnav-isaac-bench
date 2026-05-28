import numpy as np
import pytest

from isaac_bench.mapping.height_column_profile import HeightColumnProfileConfig, HeightColumnProfileMap
from isaac_bench.mapping.height_profile_roomseg_evidence import (
    HeightProfileRoomsegEvidenceConfig,
    build_height_profile_roomseg_evidence,
)


def _height_profile(shape=(6, 6)):
    cfg = HeightColumnProfileConfig(
        z_min_m=0.10,
        z_max_m=3.20,
        storage_z_max_m=3.20,
        z_bin_size_m=0.05,
        active_z_max_m=2.00,
        use_navigation_free_gate=False,
    )
    return HeightColumnProfileMap.zeros(shape, cfg), cfg


def test_navigation_obstacle_suppresses_free_but_does_not_become_wall():
    profile, cfg = _height_profile()
    profile.free_ray_count[0:3, 3, 3] = 1
    profile.observed_count[0:3, 3, 3] = 1
    nav_free = np.ones(profile.shape, dtype=bool)
    nav_obstacle = np.zeros(profile.shape, dtype=bool)
    nav_obstacle[3, 3] = True

    evidence = build_height_profile_roomseg_evidence(
        height_profile=profile,
        navigation_free_mask=nav_free,
        navigation_obstacle_mask=nav_obstacle,
        unknown_mask_from_navigation=~nav_free,
        resolution_m=0.05,
        config=HeightProfileRoomsegEvidenceConfig(height_profile=cfg),
    )

    assert not evidence.free_clean[3, 3]
    assert not evidence.wall_clean[3, 3]
    assert evidence.unknown_clean[3, 3]
    assert evidence.debug["height_evidence_debug_summary"]["nav_obstacle_added_wall_cells"] == 0


def test_evidence_forbids_unknown_fill_modes():
    profile, cfg = _height_profile()
    nav_free = np.ones(profile.shape, dtype=bool)

    with pytest.raises(ValueError, match="preserve unknown"):
        build_height_profile_roomseg_evidence(
            height_profile=profile,
            navigation_free_mask=nav_free,
            navigation_obstacle_mask=np.zeros(profile.shape, dtype=bool),
            unknown_mask_from_navigation=~nav_free,
            resolution_m=0.05,
            config=HeightProfileRoomsegEvidenceConfig(height_profile=cfg, fill_unknown_as_wall=True),
        )
