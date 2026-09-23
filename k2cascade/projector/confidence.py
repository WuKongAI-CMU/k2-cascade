"""Confidence-transfer metrics (docs/design-confidence-2026-09-22.md) from qa.py per-item files and
sender_entropy.py files.

Inputs: for each variant v in (clean, contradicted, removed), a per-item jsonl from qa.py (arms x measurements)
and optionally a sender-entropy jsonl. Items are joined on `i`.

Reports, per variant and arm: mean P(gold), P(counter), entropy, JSD to text, F1 (paired bootstrap CIs vs the
text arm); across items: AUROC of each arm's entropy at predicting the binarised sender semantic entropy,
Spearman between text-arm and cache-arm entropy, ECE (10 equal-mass bins) and Brier of P(gold) vs correctness,
Wasserstein-1 between the text-arm and each arm's entropy distribution, and the clean->removed confidence drop
ratio (cache drop / text drop).

uv run python -m k2cascade.projector.confidence --items analysis/conf --se analysis/conf --out analysis/confidence.json
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

VARIANTS = ("clean", "contradicted", "removed")


def load_items(path: str) -> dict[int, dict]:
    return {r["i"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (probability a positive outranks a negative; ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def ece(conf: list[float], correct: list[int], bins: int = 10) -> float:
    order = sorted(range(len(conf)), key=lambda i: conf[i])
    n, tot = len(order), 0.0
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if not idx:
            continue
        c = sum(conf[i] for i in idx) / len(idx)
        a = sum(correct[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(c - a)
    return tot


def brier(conf: list[float], correct: list[int]) -> float:
    return sum((c - y) ** 2 for c, y in zip(conf, correct)) / len(conf)


def wasserstein1(a: list[float], b: list[float]) -> float:
    a, b = sorted(a), sorted(b)
    n = max(len(a), len(b))
    qa = [a[min(len(a) - 1, int(k * len(a) / n))] for k in range(n)]
    qb = [b[min(len(b) - 1, int(k * len(b) / n))] for k in range(n)]
    return sum(abs(x - y) for x, y in zip(qa, qb)) / n


def paired_boot(diffs: list[float], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def mean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def analyse(items: dict[str, dict[int, dict]], se: dict[str, dict[int, dict]] | None, filter_arm: str = "text",
            f1_min: float = 0.5) -> dict:
    """items[variant][i] = qa row; se[variant][i] = sender-entropy row. Keeps items the receiver gets right on
    clean text (F1 >= f1_min in the `filter_arm`)."""
    keep = sorted(i for i, r in items["clean"].items() if r.get(filter_arm, {}).get("f1", 0) >= f1_min)
    keep = [i for i in keep if all(i in items[v] for v in items)]
    arms = [a for a in items["clean"][keep[0]] if isinstance(items["clean"][keep[0]][a], dict)] if keep else []
    out = {"n_kept": len(keep), "n_total": len(items["clean"]), "filter": f"{filter_arm} f1>={f1_min}", "arms": arms,
           "per_variant": {}, "sender_tracking": {}, "drop_ratio": {}}
    for v in items:
        rows = [items[v][i] for i in keep]
        pv = {}
        for a in arms:
            g = lambda k: [r[a].get(k) for r in rows]
            d = {k: mean(g(k)) for k in ("p_gold", "p_counter", "entropy", "jsd_text", "f1", "em", "logp")}
            if a != "text":
                for k in ("p_gold", "entropy", "f1"):
                    diffs = [r[a][k] - r["text"][k] for r in rows if r[a].get(k) is not None]
                    if diffs:
                        lo, hi = paired_boot(diffs)
                        d[f"{k}_minus_text_ci"] = [lo, hi]
                d["w1_entropy_vs_text"] = wasserstein1(g("entropy"), [r["text"]["entropy"] for r in rows])
                d["spearman_entropy_vs_text"] = spearman(g("entropy"), [r["text"]["entropy"] for r in rows])
            corr = [int(r[a]["f1"] >= f1_min) for r in rows]
            d["ece_pgold"] = ece(g("p_gold"), corr); d["brier_pgold"] = brier(g("p_gold"), corr)
            if any(r[a].get("p_counter") is not None for r in rows):
                po, pc = mean(g("p_gold")), mean(g("p_counter"))
                d["memorisation_ratio"] = po / (po + pc) if (po + pc) else float("nan")
            pv[a] = d
        out["per_variant"][v] = pv
        if se and v in se:
            ses = [se[v][i]["se"] for i in keep if i in se[v]]
            idx = [i for i in keep if i in se[v]]
            if ses:
                thr = sorted(ses)[len(ses) // 2]  # median split (MSE-optimal split is a refinement; keep simple)
                lab = [int(s > thr) for s in ses]
                st = {"n": len(idx), "threshold": thr, "positive_rate": mean(lab)}
                for a in arms:
                    ent = [items[v][i][a]["entropy"] for i in idx]
                    st[f"auroc_{a}"] = auroc(ent, lab)
                    st[f"spearman_{a}"] = spearman(ent, ses)
                out["sender_tracking"][v] = st
    if "removed" in items and "clean" in items:
        for a in arms:
            if a == "text":
                continue
            dt = out["per_variant"]["clean"]["text"]["p_gold"] - out["per_variant"]["removed"]["text"]["p_gold"]
            da = out["per_variant"]["clean"][a]["p_gold"] - out["per_variant"]["removed"][a]["p_gold"]
            out["drop_ratio"][a] = da / dt if dt else float("nan")
    return out


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="dir with qa_items_<variant>.jsonl")
    ap.add_argument("--se", default=None, help="dir with se_<variant>.jsonl")
    ap.add_argument("--out", default="analysis/confidence.json"); ap.add_argument("--f1_min", type=float, default=0.5)
    a = ap.parse_args(argv)
    items = {v: load_items(f"{a.items}/qa_items_{v}.jsonl") for v in VARIANTS if Path(f"{a.items}/qa_items_{v}.jsonl").exists()}
    se = {v: load_items(f"{a.se}/se_{v}.jsonl") for v in VARIANTS if a.se and Path(f"{a.se}/se_{v}.jsonl").exists()} or None
    res = analyse(items, se, f1_min=a.f1_min)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"n_kept": res["n_kept"], "sender_tracking": res["sender_tracking"], "drop_ratio": res["drop_ratio"]}, indent=1))


if __name__ == "__main__":
    main()
