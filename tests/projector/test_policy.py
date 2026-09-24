import numpy as np

from k2cascade.projector.policy import auroc, curve, probe_scores


def test_curve_endpoints_and_oracle_monotone():
    rng = np.random.RandomState(0)
    small = rng.rand(50); big = np.clip(small + 0.3, 0, 1)
    wrong = (small < 0.5).astype(float)
    pts = curve(wrong, small, big, 0.02)
    assert abs(pts[0]["f1"] - small.mean()) < 1e-9 and abs(pts[-1]["f1"] - big.mean()) < 1e-9
    assert all(pts[i + 1]["f1"] >= pts[i]["f1"] - 1e-9 for i in range(len(pts) - 1))
    assert auroc(wrong, wrong.astype(int)) == 1.0


def test_probe_out_of_fold_learns_separable():
    rng = np.random.RandomState(1)
    y = (rng.rand(200) < 0.5).astype(int)
    x = rng.randn(200, 8); x[:, 0] += 3 * y
    s = probe_scores(x, y)
    assert auroc(s, y) > 0.9
