import numpy as np

from k2cascade.metrics import auroc
from k2cascade.probe import chunk_bounds, fit_all, fit_oof, permutation_null, permute_within_runs


def test_permute_within_runs_keeps_each_runs_rejection_count():
    rng = np.random.default_rng(0)
    groups = ["a"] * 5 + ["b"] * 7 + ["c"] * 3
    y = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    seen_different = False
    for _ in range(20):
        yp = permute_within_runs(y, groups, rng)
        for g in ("a", "b", "c"):
            idx = [i for i, x in enumerate(groups) if x == g]
            assert yp[idx].sum() == y[idx].sum()
        seen_different |= bool((yp != y).any())
    assert seen_different


def test_permutation_null_is_near_chance_on_signal_data():
    X, y, groups = _synthetic(d=6)
    last = X[:, None, :]  # one layer
    null = permutation_null(last, y, groups, C=1.0, n_perm=5, seed=1)
    assert len(null) == 5 and (null < 0.75).all()  # the real signal (AUROC > 0.9) is destroyed by within-run shuffling


def test_chunk_bounds_keep_tail_in_last_chunk():
    assert chunk_bounds(0, 512, 64) == []
    assert chunk_bounds(10, 512, 64) == [(0, 10)]
    assert chunk_bounds(512, 512, 64) == [(0, 512)]
    assert chunk_bounds(520, 512, 64) == [(0, 520)]  # 8-token remainder merged into the previous chunk
    assert chunk_bounds(1080, 512, 64) == [(0, 512), (512, 1080)]  # 56-token remainder merged
    assert chunk_bounds(1100, 512, 64) == [(0, 512), (512, 1024), (1024, 1100)]  # 76 >= 64 stays its own chunk
    for n in (1, 63, 64, 65, 511, 513, 1023, 1025, 4096, 4100):
        b = chunk_bounds(n, 512, 64)
        assert b[0][0] == 0 and b[-1][1] == n and all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
        assert len(b) == 1 or b[-1][1] - b[-1][0] >= 64


def _synthetic(n_per_run=10, n_runs=6, d=20, signal=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_per_run * n_runs).astype(bool)
    X = rng.normal(size=(len(y), d))
    X[:, 0] += signal * y
    groups = [f"run{i // n_per_run}" for i in range(len(y))]
    return X, y, groups


def test_fit_oof_finds_a_linear_signal_out_of_fold():
    X, y, groups = _synthetic()
    oof = fit_oof(X, y, groups, C=1.0)
    assert not np.isnan(oof).any() and (0 <= oof).all() and (oof <= 1).all()
    assert auroc(oof, y) > 0.9


def test_fit_oof_is_chance_when_the_label_is_only_in_the_held_out_run():
    # a feature that is a per-run constant carries no out-of-run information; OOF AUROC should be near chance
    rng = np.random.default_rng(1)
    groups = [f"run{i // 10}" for i in range(80)]
    y = rng.integers(0, 2, 80).astype(bool)
    X = rng.normal(size=(80, 5))
    oof = fit_oof(X, y, groups, C=1.0)
    assert 0.3 < auroc(oof, y) < 0.7


def test_fit_all_reports_best_layer_and_baselines():
    X, y, groups = _synthetic(d=8)
    n = len(y)
    last = np.stack([np.random.default_rng(2).normal(size=(n, 8)), X, X], axis=1)  # layer 0 noise, layers 1-2 signal
    meta = [{"run_id": g, "step": i % 10 + 1, "ok": not yy, "n_prompt_tokens": 100 + i} for i, (g, yy) in enumerate(zip(groups, y))]
    summary, oof = fit_all(last, last, meta, C=1.0, n_boot=50)
    assert oof.shape == (3, n) and summary["best_layer"] in (1, 2)
    assert summary["per_layer"][0]["auroc"] < summary["best"]["auroc"]
    assert summary["best"]["lo"] <= summary["best"]["auroc"] <= summary["best"]["hi"]
    assert set(summary["baselines"]) == {"tail_mean_best_layer", "prompt_length", "step_index"}
    assert set(summary["C_sensitivity_at_best_layer"]) == {"0.01", "0.1", "1.0", "10.0"}
