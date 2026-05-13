import numpy as np

from isaac_bench.mapping.frontier import frontier_cells


def test_frontier_cells_are_reachable_observed_free_boundary():
    free = np.zeros((5, 5), dtype=bool)
    observed = np.zeros((5, 5), dtype=bool)
    free[2, 1:4] = True
    observed[2, 1:3] = True

    frontiers = frontier_cells(free, observed)

    assert frontiers[2, 2]
    assert not frontiers[2, 3]
