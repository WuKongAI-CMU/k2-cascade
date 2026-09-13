"""Rank-based AUROC, bootstrap CIs, Spearman, and the leave-one-run-out split. numpy only."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[bool | int]) -> float:
    """P(score of a positive > score of a negative), ties count 1/2 (Mann-Whitney U / (n_pos * n_neg)).
    Returns nan when either class is empty."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auroc(scores: Sequence[float], labels: Sequence[bool | int], n_boot: int = 2000, seed: int = 0,
                    alpha: float = 0.05) -> tuple[float, float, float]:
    """(auroc, lo, hi): percentile bootstrap over attempts. Resamples that lose a class are skipped."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(s)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].all() or not y[idx].any():
            continue
        vals.append(auroc(s[idx], y[idx]))
    if not vals:
        return auroc(s, y), float("nan"), float("nan")
    v = np.asarray(vals)
    return auroc(s, y), float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _rank(np.asarray(a, dtype=np.float64)), _rank(np.asarray(b, dtype=np.float64))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_spearman(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        r = spearman(a[idx], b[idx])
        if not np.isnan(r):
            vals.append(r)
    v = np.asarray(vals)
    return spearman(a, b), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def leave_one_group_out(groups: Sequence[Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold per distinct group (here: run_id); test = that group, train = everything else.
    Folds are ordered by first appearance of the group."""
    g = list(groups)
    seen: list[Any] = []
    for x in g:
        if x not in seen:
            seen.append(x)
    arr = np.arange(len(g))
    folds = []
    for grp in seen:
        test = np.array([i for i in arr if g[i] == grp], dtype=int)
        train = np.array([i for i in arr if g[i] != grp], dtype=int)
        folds.append((train, test))
    return folds
