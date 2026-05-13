from isaac_bench.metrics.spl import compute_softspl, compute_spl


def test_spl():
    assert compute_spl(False, 5.0, 5.0) == 0.0
    assert compute_spl(True, 5.0, 5.0) == 1.0
    assert compute_spl(True, 5.0, 10.0) == 0.5


def test_softspl():
    assert compute_softspl(10.0, 0.0, 10.0, 10.0) == 1.0
    assert compute_softspl(10.0, 5.0, 10.0, 10.0) == 0.5

