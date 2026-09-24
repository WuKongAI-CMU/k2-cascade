"""Milestone 4: accuracy-vs-cost curves for handoff policies.

Inputs: the sender npz from cascade.py and qa.py per-item files for the receiver arms (text, project, verbal).
Triggers: a logistic probe on the sender's hidden state (nested 5-fold), max-prob, semantic entropy, sample
agreement, random, and the oracle (hand off exactly the items the sender gets wrong).
For each trigger and handoff rate r, the policy hands off the r fraction of items the trigger ranks most likely
to fail; accuracy = mean F1 of the chosen answers; cost = c_small + r * c_handoff (per item, in the chosen unit).

uv run python -m k2cascade.projector.policy --sender analysis/cascade/sender.npz --items analysis/cascade --out analysis/cascade/curves.json
"""
from __future__ import annotations

import json

import numpy as np

RATES = [i / 10 for i in range(11)]
# per-item handoff cost in seconds on one A100 (latency.py); the sender's own read is paid in every arm
COST = {"small": 0.045, "text": 0.048, "project": 0.024, "verbal": 0.004}


def probe_scores(feats: np.ndarray, wrong: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold probability of failure from a logistic probe with standardised features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(wrong)); out = np.zeros(len(wrong))
    for f in range(folds):
        test = idx[f::folds]; train = np.setdiff1d(idx, test)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(feats[train], wrong[train]); out[test] = clf.predict_proba(feats[test])[:, 1]
    return out


def auroc(scores, labels) -> float:
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def curve(score: np.ndarray, f1_small: np.ndarray, f1_big: np.ndarray, unit_cost: float) -> list[dict]:
    order = np.argsort(-score)  # most likely to fail first
    n = len(score); pts = []
    for r in RATES:
        k = int(round(r * n)); hand = np.zeros(n, bool); hand[order[:k]] = True
        f1 = np.where(hand, f1_big, f1_small).mean()
        pts.append({"rate": r, "f1": float(f1), "cost": COST["small"] + r * unit_cost})
    return pts


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True); ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args(argv)
    s = np.load(a.sender, allow_pickle=False)
    n = len(s["f1"]); wrong = (s["f1"] < 0.5).astype(int)
    arms = {}
    for arm in ("text", "project", "verbal"):
        p = Path(a.items) / f"qa_items_{arm}.jsonl"
        if p.exists():
            rows = {r["i"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
            arms[arm] = np.array([rows[i][arm]["f1"] if i in rows and arm in rows[i] else np.nan for i in range(n)])
    triggers = {"probe": probe_scores(s["feats"].astype(np.float32), wrong),
                "maxprob": -s["p_max"], "entropy": s["entropy"], "sem_entropy": s["se"], "disagree": -s["agree"],
                "random": np.random.RandomState(0).rand(n), "oracle": wrong.astype(float)}
    res = {"n": int(n), "sender_f1": float(s["f1"].mean()), "frac_wrong": float(wrong.mean()),
           "trigger_auroc": {k: auroc(v, wrong) for k, v in triggers.items()}, "arms_f1": {k: float(np.nanmean(v)) for k, v in arms.items()},
           "curves": {}}
    for arm, f1_big in arms.items():
        ok = ~np.isnan(f1_big)
        for trig, sc in triggers.items():
            res["curves"][f"{arm}/{trig}"] = curve(sc[ok], s["f1"][ok], f1_big[ok], COST[arm])
    # headline: best F1 per cost for each arm using the probe trigger, plus equal-cost comparison state vs text
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    summ = {k: {"auroc": round(v, 3)} for k, v in res["trigger_auroc"].items()}
    print(json.dumps({"sender_f1": round(res["sender_f1"], 3), "arms_f1": {k: round(v, 3) for k, v in res["arms_f1"].items()}, "triggers": summ}, indent=1))
    for arm in arms:
        c = res["curves"][f"{arm}/probe"]; print(arm, " ".join(f"r{p['rate']:.1f}:{p['f1']:.3f}@{p['cost']*1000:.0f}ms" for p in c[::2]))


if __name__ == "__main__":
    main()
