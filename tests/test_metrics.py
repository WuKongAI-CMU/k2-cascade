import numpy as np
import pytest

from k2cascade.metrics import auroc, bootstrap_auroc, leave_one_group_out, spearman


def test_auroc_perfect_reversed_and_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    # one tie between a positive and a negative counts one half
    assert auroc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]) == pytest.approx(0.875)
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))


def test_auroc_matches_sklearn_on_random_data():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        s = np.round(rng.normal(size=60) + y, 1)  # rounding creates ties
        assert auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    s = rng.normal(size=80) + 0.8 * y
    a, lo, hi = bootstrap_auroc(s, y, n_boot=500, seed=3)
    assert lo <= a <= hi and 0.0 <= lo and hi <= 1.0
    assert bootstrap_auroc(s, y, n_boot=500, seed=3) == (a, lo, hi)


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)  # monotone, not linear
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_leave_one_group_out_is_a_partition_by_run():
    groups = ["r1", "r1", "r2", "r3", "r3", "r3", "r2"]
    folds = leave_one_group_out(groups)
    assert len(folds) == 3
    covered = []
    for train, test in folds:
        test_groups = {groups[i] for i in test}
        assert len(test_groups) == 1  # exactly one run held out
        assert test_groups.isdisjoint({groups[i] for i in train})  # never seen in training
        assert set(train) | set(test) == set(range(len(groups)))
        covered += list(test)
    assert sorted(covered) == list(range(len(groups)))  # every attempt held out exactly once
    assert [groups[f[1][0]] for f in folds] == ["r1", "r2", "r3"]
