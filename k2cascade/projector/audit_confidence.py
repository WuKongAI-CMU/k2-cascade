"""Audit of the confidence-transfer table, no GPU (fifth external review, 2026-09-25):
  (1) label-only AUROC: the confidence word / numeric label itself against the binary target, no receiver;
  (2) paired bootstrap CIs, grouped by passage, for AUROC differences between arms;
  (3) discretisation of the 10-sample semantic entropy (distinct values, ties).

uv run python -m k2cascade.projector.audit_confidence --items analysis/cloud/verbal/verbal_mlp_seed2 --se analysis/cloud/conf \
    --data data/squad_variants_nli.jsonl --out analysis/cloud/verbal/audit.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .confidence import auroc, load_items


def label_scores(se_rows: dict[int, dict], mode: str) -> dict[int, float]:
    ses = sorted(r["se"] for r in se_rows.values()); lo, hi = ses[len(ses) // 3], ses[2 * len(ses) // 3]
    out = {}
    for i, r in se_rows.items():
        if mode == "words":   # low confidence = high uncertainty score
            out[i] = 2.0 if r["se"] > hi else (1.0 if r["se"] > lo else 0.0)
        else:                 # numeric: 1 - agreement of samples with the greedy answer
            labels = r["labels"]; k = len(labels) - 1
            out[i] = 1.0 - sum(1 for l in labels[:k] if l == labels[-1]) / max(k, 1)
    return out


def grouped_boot_diff(keep, groups, sa, sb, y, reps=2000, seed=0):
    """CI for AUROC(a) - AUROC(b) resampling passages (groups) with replacement."""
    rng = random.Random(seed)
    by_g = {}
    for i in keep:
        by_g.setdefault(groups[i], []).append(i)
    gl = list(by_g); diffs = []
    for _ in range(reps):
        idx = [i for g in (rng.choice(gl) for _ in gl) for i in by_g[g]]
        ya = [y[i] for i in idx]
        d = auroc([sa[i] for i in idx], ya) - auroc([sb[i] for i in idx], ya)
        diffs.append(d)
    diffs.sort(); return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True); ap.add_argument("--se", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--f1_min", type=float, default=0.5)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    groups = {i: r.get("clean", r.get("context", ""))[:200] for i, r in enumerate(rows)}
    res = {}
    items = {v: load_items(f"{a.items}/qa_items_{v}.jsonl") for v in ("clean", "contradicted", "removed")}
    keep = sorted(i for i, r in items["clean"].items() if r.get("text", {}).get("f1", 0) >= a.f1_min)
    for v in ("clean", "contradicted", "removed"):
        se = load_items(f"{a.se}/se_{v}.jsonl"); ks = [i for i in keep if i in se and i in items[v]]
        ses = [se[i]["se"] for i in ks]; thr = sorted(ses)[len(ses) // 2]; y = {i: int(se[i]["se"] > thr) for i in ks}
        yl = [y[i] for i in ks]
        r = {"n": len(ks), "n_passages": len({groups[i] for i in ks}), "distinct_se_values": len(set(ses)),
             "frac_at_threshold": sum(1 for s in ses if s == thr) / len(ses)}
        for mode in ("words", "numeric"):
            ls = label_scores(se, mode); r[f"label_auroc_{mode}"] = auroc([ls[i] for i in ks], yl)
        arms = [x for x in items[v][ks[0]] if isinstance(items[v][ks[0]][x], dict)]
        ent = {x: {i: items[v][i][x]["entropy"] for i in ks} for x in arms}
        r["auroc"] = {x: auroc([ent[x][i] for i in ks], yl) for x in arms}
        for b in ("verbal", "none", "text"):
            if "project" in ent and b in ent:
                lo, hi = grouped_boot_diff(ks, groups, ent["project"], ent[b], y)
                r[f"ci_project_minus_{b}"] = [lo, hi]
        res[v] = r
    # within-item intervention (sixth review): same question, the sender's passage changes clean -> removed /
    # contradicted. Does the receiver's entropy change track the *sender's* semantic-entropy change, per item?
    from .confidence import spearman
    se_c = load_items(f"{a.se}/se_clean.jsonl")
    for v in ("removed", "contradicted"):
        se_v = load_items(f"{a.se}/se_{v}.jsonl")
        ks = [i for i in keep if i in se_c and i in se_v and i in items[v] and i in items["clean"]]
        d_send = [se_v[i]["se"] - se_c[i]["se"] for i in ks]
        arms = [x for x in items[v][ks[0]] if isinstance(items[v][ks[0]][x], dict)]
        w = {"n": len(ks), "sender_mean_delta_se": sum(d_send) / len(d_send)}
        for x in arms:
            d_rec = [items[v][i][x]["entropy"] - items["clean"][i][x]["entropy"] for i in ks]
            same_sign = sum(1 for a_, b_ in zip(d_send, d_rec) if (a_ > 0) == (b_ > 0)) / len(ks)
            w[x] = {"spearman_delta": spearman(d_send, d_rec), "mean_delta_entropy": sum(d_rec) / len(d_rec), "same_sign": same_sign}
        res[f"within_item_{v}"] = w
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for v in ("removed", "contradicted"):
        w = res[f"within_item_{v}"]
        print(f"within-item clean->{v}: n {w['n']} sender dSE {w['sender_mean_delta_se']:.2f} | " + " ".join(
            f"{x}: rho {w[x]['spearman_delta']:.3f} dH {w[x]['mean_delta_entropy']:.2f} sign {w[x]['same_sign']:.2f}" for x in w if isinstance(w[x], dict)))
    for v, r in res.items():
        if v.startswith("within"): continue
        print(v, "n", r["n"], "passages", r["n_passages"], "distinct SE", r["distinct_se_values"],
              "| label AUROC words %.3f numeric %.3f" % (r["label_auroc_words"], r["label_auroc_numeric"]),
              "| arms", {k: round(x, 3) for k, x in r["auroc"].items()},
              "| CI proj-verbal [%.3f, %.3f] proj-none [%.3f, %.3f] proj-text [%.3f, %.3f]" % (
                  *r.get("ci_project_minus_verbal", [0, 0]), *r.get("ci_project_minus_none", [0, 0]), *r.get("ci_project_minus_text", [0, 0])))


if __name__ == "__main__":
    main()
