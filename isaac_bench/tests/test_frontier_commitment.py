import numpy as np

from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.navigation.frontier_commitment import FrontierCommitmentManager


def _frontier(center, score_dist=5.0):
    r, c = center
    members = [(r, c), (r, c + 1), (r + 1, c)]
    return FrontierCluster(center, (float(c), float(r)), members, len(members), score_dist)


def test_frontier_commitment_keeps_sticky_frontier_until_hysteresis_passes():
    planner = GridAStarPlanner(np.ones((40, 40), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(resolution_m=0.1, min_commit_steps=12, no_progress_steps=999, allow_score_switch=True)
    current = (0, 0)
    a = _frontier((10, 10))
    b = _frontier((25, 25))

    first = manager.select([a, b], a, 1.0, current, 0, planner=planner, scores_by_index=[1.0, 0.5])
    assert not first.keep_existing
    assert first.selected_stable_id == 1

    second = manager.select([a, b], b, 1.12, current, 5, planner=planner, scores_by_index=[1.0, 1.12])
    assert second.keep_existing
    assert second.selected_stable_id == 1
    assert second.frontier is a

    third = manager.select([a, b], b, 1.6, current, 20, planner=planner, scores_by_index=[1.0, 1.6])
    assert not third.keep_existing
    assert third.frontier is b
    assert third.selected_stable_id != 1


def test_frontier_commitment_reached_clears_active_and_selects_next():
    planner = GridAStarPlanner(np.ones((40, 40), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(resolution_m=0.1, reached_radius_m=0.25, no_progress_steps=999)
    a = _frontier((10, 10))
    b = _frontier((25, 25))

    manager.select([a, b], a, 1.0, (0, 0), 0, planner=planner, scores_by_index=[1.0, 0.5])
    reached = manager.select([a, b], b, 0.8, (10, 10), 1, planner=planner, scores_by_index=[1.0, 0.8])

    assert not reached.keep_existing
    assert reached.frontier is b
    assert reached.metadata["active_frontier_id"] == manager.active.stable_id


def test_frontier_commitment_does_not_reach_on_non_target_cluster_member():
    planner = GridAStarPlanner(np.ones((40, 40), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(resolution_m=0.1, reached_radius_m=0.25, no_progress_steps=999)
    a = FrontierCluster((10, 10), (1.0, 1.0), [(10, 10), (10, 11), (20, 20)], 3, 5.0)
    b = _frontier((25, 25))

    manager.select([a, b], a, 1.0, (0, 0), 0, planner=planner, target_cells=[(10, 10)], scores_by_index=[1.0, 0.5])
    out = manager.select([a, b], b, 0.8, (20, 20), 1, planner=planner, scores_by_index=[1.0, 0.8])

    assert out.keep_existing
    assert out.frontier is a
    assert out.reason == "continue_committed_frontier"


def test_frontier_commitment_reached_uses_frontier_center_not_fallback_target():
    planner = GridAStarPlanner(np.ones((40, 40), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(resolution_m=0.1, reached_radius_m=0.25, no_progress_steps=999)
    a = FrontierCluster((10, 10), (1.0, 1.0), [(10, 10), (10, 11), (20, 20)], 3, 5.0)
    b = _frontier((25, 25))

    manager.select([a, b], a, 1.0, (0, 0), 0, planner=planner, target_cells=[(20, 20)], scores_by_index=[1.0, 0.5])
    out = manager.select([a, b], b, 0.8, (20, 20), 1, planner=planner, scores_by_index=[1.0, 0.8])

    assert out.keep_existing
    assert out.frontier is a
    assert out.reason == "continue_committed_frontier"


def test_frontier_commitment_no_progress_blacklists_and_selects_new():
    planner = GridAStarPlanner(np.ones((40, 40), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(
        resolution_m=0.1,
        no_progress_steps=2,
        progress_min_delta_m=0.05,
        blacklist_ttl_steps=50,
    )
    current = (0, 0)
    a = _frontier((10, 10))
    b = _frontier((25, 25))

    manager.select([a, b], a, 1.0, current, 0, planner=planner, scores_by_index=[1.0, 0.2])
    manager.select([a, b], a, 1.0, current, 1, planner=planner, scores_by_index=[1.0, 0.2])
    out = manager.select([a, b], a, 1.0, current, 2, planner=planner, scores_by_index=[1.0, 0.2])

    assert not out.keep_existing
    assert out.frontier is b
    assert manager.blacklist


def test_frontier_commitment_default_holds_until_reached_without_score_switch():
    planner = GridAStarPlanner(np.ones((60, 60), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(resolution_m=0.1, min_commit_steps=1, no_progress_steps=999)
    a = _frontier((10, 10))
    b = _frontier((40, 40))

    manager.select([a, b], a, 1.0, (0, 0), 0, planner=planner, scores_by_index=[1.0, 0.1])
    out = manager.select([a, b], b, 10.0, (1, 1), 10, planner=planner, scores_by_index=[1.0, 10.0])

    assert out.keep_existing
    assert out.frontier is a
    assert out.reason == "continue_committed_frontier"


def test_frontier_commitment_unmatched_frontier_keeps_saved_target_cells():
    planner = GridAStarPlanner(np.ones((60, 60), dtype=bool), resolution_m=0.1)
    manager = FrontierCommitmentManager(resolution_m=0.1, no_progress_steps=999)
    a = _frontier((10, 10))
    b = _frontier((40, 40))

    first = manager.select([a, b], a, 1.0, (0, 0), 0, planner=planner, scores_by_index=[1.0, 0.5])
    out = manager.select([b], b, 10.0, (1, 1), 1, planner=planner, scores_by_index=[10.0])

    assert out.keep_existing
    assert out.frontier is None
    assert out.target_cells == first.target_cells
    assert out.reason == "continue_committed_frontier"
