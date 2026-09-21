"""Properly nested test of whether the pre-action probe adds information over bookkeeping.

The earlier one-dimensional test in shortcut.py reused out-of-fold probe scores that were
computed once, globally. For an outer held-out run R, the training rows' probe scores came
from probes that had been fit on R. That leaks R's labels into the second stage and makes the
"probe adds +x over metadata" number optimistic.

Here, for every outer held-out run R:
  * the probe feature for R's rows comes from a probe fit on all runs except R;
  * the probe feature for the TRAINING rows comes from an inner leave-one-run-out pass
    that never sees R;
  * the second stage [metadata + probe logit] is fit on the training rows only.
The layer is fixed a priori (final norm, and the middle layer), not selected on the data.
Confidence intervals resample whole runs (cluster bootstrap), not individual steps.

Usage: uv run python -m k2cascade.nested --size 3.7b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import auroc
from .shortcut import features


def _fit_predict(Xtr, ytr, Xte, C=1.0):
    if ytr.all() or not ytr.any():
        return np.full(len(Xte), ytr.mean())
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit((Xtr - mu) / sd, ytr)
    return clf.predict_proba((Xte - mu) / sd)[:, 1]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def nested(meta, H, y, groups, C=1.0):
    runs = np.unique(groups)
    s_meta = np.zeros(len(y)); s_probe = np.zeros(len(y)); s_both = np.zeros(len(y))
    for R in runs:
        te = groups == R
        tr = ~te
        s_meta[te] = _fit_predict(meta[tr], y[tr], meta[te])
        s_probe[te] = _fit_predict(H[tr], y[tr], H[te], C)
        # inner OOF probe scores for the training rows, never touching R
        inner = np.zeros(tr.sum())
        g_tr, H_tr, y_tr = groups[tr], H[tr], y[tr]
        for r in np.unique(g_tr):
            ite = g_tr == r
            inner[ite] = _fit_predict(H_tr[~ite], y_tr[~ite], H_tr[ite], C)
        Xtr2 = np.c_[meta[tr], _logit(inner)]
        Xte2 = np.c_[meta[te], _logit(s_probe[te])]
        s_both[te] = _fit_predict(Xtr2, y[tr], Xte2)
    return s_meta, s_probe, s_both


def cluster_boot(y, scores: dict, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    runs = np.unique(groups)
    idx_by_run = {r: np.where(groups == r)[0] for r in runs}
    out = {k: [] for k in scores}
    out["delta_both_minus_meta"] = []
    for _ in range(n):
        pick = rng.choice(runs, size=len(runs), replace=True)
        ix = np.concatenate([idx_by_run[r] for r in pick])
        yy = y[ix]
        if yy.all() or not yy.any():
            continue
        for k, s in scores.items():
            out[k].append(auroc(s[ix], yy))
        out["delta_both_minus_meta"].append(auroc(scores["both"][ix], yy) - auroc(scores["meta"][ix], yy))
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=["0.9b", "3.7b"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    model = f"k2-{a.size}"
    probe_rows = [json.loads(l) for l in Path(f"analysis/hidden/probe_{a.size}.jsonl").read_text().splitlines() if l.strip()]
    key = [(r["run_id"], r["step"], r["attempt"]) for r in probe_rows]
    last = np.load(f"analysis/hidden/hidden_{a.size}.npz", allow_pickle=True)["last"]
    assert len(last) == len(key), (len(last), len(key))

    # metadata in trace order, then aligned to the probe's row order by (run, step, attempt)
    X, y_meta, g_meta, names = features(sorted(Path("traces").glob("*.jsonl")), model)
    meta_keys = []
    for f in sorted(Path("traces").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("step") == 0 or r.get("model") != model:
                continue
            meta_keys.append((r["run_id"], r["step"], r["attempt"]))
    pos = {k: i for i, k in enumerate(meta_keys)}
    missing = [k for k in key if k not in pos]
    assert not missing, f"{len(missing)} probe rows have no metadata row"
    order = np.array([pos[k] for k in key])
    meta, y, groups = X[order], y_meta[order], g_meta[order]
    assert (y == np.array([0 if r["ok"] else 1 for r in probe_rows])).all(), "label mismatch after alignment"

    L = last.shape[1]
    layers = {"final_norm": L - 1, "middle": L // 2}
    res = {"size": a.size, "n": int(len(y)), "n_rejected": int(y.sum()), "n_runs": int(len(np.unique(groups))),
           "note": "nested leave-one-run-out; layer fixed a priori; CIs are run-level cluster bootstrap", "layers": {}}
    for name, l in layers.items():
        sm, sp, sb = nested(meta, last[:, l, :].astype(float), y, groups, a.C)
        pts = {"meta": auroc(sm, y), "probe": auroc(sp, y), "both": auroc(sb, y)}
        ci = cluster_boot(y, {"meta": sm, "probe": sp, "both": sb}, groups, a.n_boot)
        res["layers"][name] = {
            "layer": int(l),
            **{k: {"auroc": round(v, 3), "ci": [round(ci[k][0], 3), round(ci[k][1], 3)]} for k, v in pts.items()},
            "delta_both_minus_meta": {"value": round(pts["both"] - pts["meta"], 3),
                                      "ci": [round(ci["delta_both_minus_meta"][0], 3), round(ci["delta_both_minus_meta"][1], 3)]},
        }
        print(name, json.dumps(res["layers"][name]), flush=True)
    Path(f"analysis/nested_{a.size}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
