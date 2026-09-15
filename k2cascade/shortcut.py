"""Nuisance/metadata shortcut baseline for the pre-action probe.

If a handful of cheap scalars known before the step (step index, how many prior
attempts were rejected in this run, what the last tool was, how long the prompt is)
predicts rejection as well as the hidden-state probe, then the probe is reading
bookkeeping, not competence. Motivated by arXiv 2606.22864.

Usage: uv run python -m k2cascade.shortcut --trace-model k2-0.9b --probe analysis/hidden/probe_0.9b.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import auroc, bootstrap_auroc, spearman

TOOLS = ["shell", "read_file", "write_file", "none"]


def features(traces: list[Path], trace_model: str):
    """One row per attempt, in the same order probe.py enumerates them."""
    rows, labels, groups = [], [], []
    for f in sorted(traces):
        prior_rej = defaultdict(int)  # run_id -> rejected attempts so far
        last_tool: dict[str, str] = {}
        n_steps = defaultdict(int)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("step") == 0 or r.get("model") != trace_model:
                continue
            run = r["run_id"]
            msgs = r.get("messages")
            prompt_len = sum(len(str(m.get("content") or "")) for m in msgs) if msgs else 0
            lt = last_tool.get(run, "none")
            rows.append([
                r["step"],
                r["attempt"],
                prior_rej[run],
                n_steps[run],
                prompt_len / 1000.0,
                float(lt == "shell"), float(lt == "read_file"), float(lt == "write_file"),
                float(prior_rej[run] > 0),
            ])
            labels.append(0 if r["ok"] else 1)
            groups.append(run)
            if not r["ok"]:
                prior_rej[run] += 1
            else:
                n_steps[run] += 1
                calls = r.get("tool_calls") or []
                last_tool[run] = calls[0]["name"] if calls else "none"
    names = ["step", "attempt", "prior_rejects", "accepted_so_far", "prompt_kchars",
             "last=shell", "last=read", "last=write", "any_prior_reject"]
    return np.array(rows, float), np.array(labels, int), np.array(groups), names


def loro_logistic(X, y, groups, C=1.0, iters=400):
    """Leave-one-run-out logistic regression, same protocol as the probe."""
    out = np.zeros(len(y), float)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            out[te] = y[tr].mean() if tr.any() else 0.5
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        A = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
        B = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
        w = np.zeros(A.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-A @ w))
            grad = A.T @ (p - y[tr]) + w / C
            H = A.T @ (A * (p * (1 - p))[:, None]) + np.eye(len(w)) / C
            w -= np.linalg.solve(H + 1e-6 * np.eye(len(w)), grad)
        out[te] = B @ w
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-model", required=True)
    ap.add_argument("--traces", nargs="*", type=Path, default=None)
    ap.add_argument("--probe", type=Path, default=None, help="probe json to compare against")
    ap.add_argument("--probe-scores", type=Path, default=None, help="probe_<size>.jsonl with out-of-fold p_fail per attempt")
    ap.add_argument("--hidden", type=Path, default=None, help="hidden_<size>.npz from probe.py")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    traces = a.traces or sorted(Path("traces").glob("*.jsonl"))
    X, y, g, names = features(traces, a.trace_model)
    res = {"trace_model": a.trace_model, "n": int(len(y)), "n_rejected": int(y.sum()),
           "n_runs": int(len(np.unique(g))), "features": names}
    s = loro_logistic(X, y, g)
    a_, lo, hi = bootstrap_auroc(s, y)
    res["all_metadata"] = {"auroc": a_, "lo": lo, "hi": hi}
    singles = {}
    for j, nm in enumerate(names):
        sc = X[:, j]
        v = auroc(sc, y)
        singles[nm] = round(max(v, 1 - v), 3)
    res["single_feature_best_direction"] = singles
    # The right incremental test at this n: collapse the probe to ONE feature (its
    # out-of-fold p_fail) and refit alongside the 9 metadata scalars. Concatenating
    # 1536 raw dims to 9 scalars over ~130 rows measures overfitting, not information.
    if a.probe_scores and a.probe_scores.exists():
        rows = [json.loads(l) for l in a.probe_scores.read_text().splitlines() if l.strip()]
        if len(rows) != len(y):
            res["incremental_1d"] = {"error": f"probe jsonl has {len(rows)} rows, metadata has {len(y)}"}
        else:
            pf = np.array([r["p_fail_best_layer"] for r in rows], float)
            logit = np.log(np.clip(pf, 1e-6, 1 - 1e-6) / np.clip(1 - pf, 1e-6, 1 - 1e-6))
            s_meta = loro_logistic(X, y, g)
            s_both = loro_logistic(np.c_[X, logit], y, g)
            res["incremental_1d"] = {
                "probe_alone": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(logit, y))),
                "metadata_only": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(s_meta, y))),
                "metadata_plus_probe": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(s_both, y))),
                "delta_over_metadata": float(auroc(s_both, y) - auroc(s_meta, y)),
                "spearman_metadata_vs_probe": spearman(s_meta, logit),
            }
    # raw-dim version, kept for the record: expect it to overfit at this n
    if a.hidden and a.hidden.exists():
        H = np.load(a.hidden, allow_pickle=True)["last"]          # (n_attempts, n_layers, d)
        if len(H) != len(y):
            res["incremental"] = {"error": f"hidden has {len(H)} rows, metadata has {len(y)}"}
        else:
            p_json = json.loads(a.probe.read_text()) if a.probe and a.probe.exists() else {}
            L = p_json.get("best_layer", H.shape[1] // 2)
            h = H[:, L, :].astype(float)
            s_h = loro_logistic(h, y, g, C=1.0)
            s_c = loro_logistic(np.c_[X, h], y, g, C=1.0)
            res["incremental"] = {
                "layer": int(L),
                "metadata_only": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(s, y))),
                "hidden_only": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(s_h, y))),
                "metadata_plus_hidden": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(s_c, y))),
                "spearman_metadata_vs_hidden": spearman(s, s_h),
            }
    if a.probe and a.probe.exists():
        p = json.loads(a.probe.read_text())
        res["probe_best_layer"] = {k: p["best"][k] for k in ("layer", "auroc", "lo", "hi")}
        res["probe_median_layer"] = p["median_layer_auroc"]
    print(json.dumps(res, indent=1))
    out = a.out or Path(f"analysis/shortcut_{a.trace_model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
