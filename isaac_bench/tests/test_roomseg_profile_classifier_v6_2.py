from __future__ import annotations

import numpy as np

from isaac_bench.mapping.online_roomseg.height_profile_v6_2 import FloorCeilingEstimate, HeightProfileState
from isaac_bench.mapping.online_roomseg.profile_classifier_v6_2 import classify_height_profile_v6_2


def test_wall_requires_95_percent_endpoint_bins():
    state = _state(40, width=2)
    state.occupied_supported[:38, 0, 0] = True
    state.endpoint_supported[:38, 0, 0] = True
    state.occupied_supported[:37, 0, 1] = True
    state.endpoint_supported[:37, 0, 1] = True

    result = classify_height_profile_v6_2(state)

    assert bool(result.wall_mask_raw[0, 0])
    assert not bool(result.wall_mask_raw[0, 1])
    assert np.isclose(result.wall_endpoint_ratio[0, 0], 0.95)


def test_door_transition_requires_lower_free_upper_occupied_and_min_height():
    state = _state(60, width=3, height_m=3.0)
    state.free_supported[:39, 0, 0] = True
    state.occupied_supported[40:59, 0, 0] = True
    state.endpoint_supported[40:59, 0, 0] = True

    state.free_supported[:37, 0, 1] = True
    state.occupied_supported[40:60, 0, 1] = True
    state.endpoint_supported[40:60, 0, 1] = True

    state.free_supported[:34, 0, 2] = True
    state.occupied_supported[34:60, 0, 2] = True
    state.endpoint_supported[34:60, 0, 2] = True

    result = classify_height_profile_v6_2(state)

    assert bool(result.door_mask_raw[0, 0])
    assert result.best_door_transition_z[0, 0] >= 1.8
    assert not bool(result.door_mask_raw[0, 1])
    assert not bool(result.door_mask_raw[0, 2])


def test_wall_priority_over_door():
    state = _state(60, width=1, height_m=3.0)
    state.free_supported[:40, 0, 0] = True
    state.occupied_supported[:, 0, 0] = True
    state.endpoint_supported[:, 0, 0] = True

    result = classify_height_profile_v6_2(state)

    assert bool(result.wall_mask_raw[0, 0])
    assert not bool(result.door_mask_raw[0, 0])


def _state(k: int, *, width: int, height_m: float | None = None) -> HeightProfileState:
    total_height = float(height_m if height_m is not None else float(k) * 0.05)
    z_edges = np.linspace(0.0, total_height, int(k) + 1, dtype=np.float32)
    z_centers = (z_edges[:-1] + z_edges[1:]) * 0.5
    shape = (int(k), 1, int(width))
    zeros_u16 = np.zeros(shape, dtype=np.uint16)
    zeros_bool = np.zeros(shape, dtype=bool)
    fc = FloorCeilingEstimate(
        floor_z=0.0,
        ceiling_z=total_height,
        room_height_m=total_height,
        inner_low_z=0.0,
        inner_high_z=total_height,
        confidence=1.0,
    )
    return HeightProfileState(
        z_edges_m=z_edges,
        z_centers_m=z_centers,
        free_count=zeros_u16.copy(),
        endpoint_count=zeros_u16.copy(),
        observed_count=zeros_u16.copy(),
        unknown_count=zeros_u16.copy(),
        free_supported=zeros_bool.copy(),
        endpoint_supported=zeros_bool.copy(),
        occupied_supported=zeros_bool.copy(),
        unknown_supported=zeros_bool.copy(),
        floor_ceiling=fc,
    )
