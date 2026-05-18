import numpy as np

from isaac_bench.mapping.frontier_room_context import assign_frontier_room_context


def test_frontier_room_context_uses_known_free_side_not_center():
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[3:6, 2:4] = 2
    free = np.zeros_like(labels, dtype=bool)
    free[3:6, 2:6] = True
    unknown = np.zeros_like(free)
    unknown[:, 6:] = True
    frontier = np.asarray([[4, 5], [3, 5], [5, 5]], dtype=np.int32)

    context = assign_frontier_room_context(
        frontier,
        labels,
        free,
        unknown,
        agent_rc=(4, 2),
        resolution_m=0.1,
        local_radius_m=0.25,
    )

    assert context["room_id"] == 2
    assert context["method"] == "known_free_side"
    assert context["confidence"] >= 0.2


def test_frontier_room_context_nearest_fallback():
    labels = np.zeros((11, 11), dtype=np.int32)
    labels[5, 2] = 4
    free = np.ones_like(labels, dtype=bool)
    unknown = np.zeros_like(free)
    frontier = np.asarray([[5, 7]], dtype=np.int32)

    context = assign_frontier_room_context(
        frontier,
        labels,
        free,
        unknown,
        agent_rc=(5, 0),
        resolution_m=0.2,
        local_radius_m=0.1,
        nearest_fallback_radius_m=1.2,
    )

    assert context["room_id"] == 4
    assert context["method"] == "nearest_labeled_free"
    assert 0.0 < context["confidence"] < 0.5

