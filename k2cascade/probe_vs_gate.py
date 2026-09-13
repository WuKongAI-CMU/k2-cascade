"""Part C: put the post-hoc surprise gate (gate.py) and the pre-action probe (probe.py) side by side.

Reads analysis/gate/gate_<trace_model>_by_<scorer>.{json,jsonl} and analysis/hidden/probe_<size>.{json,jsonl},
joins them per attempt on (run_id, step, attempt), and writes analysis/probe_vs_gate.md: one table per size
(AUROC for predicting ok == False with 95% bootstrap CIs) plus the Spearman correlation between the probe's
out-of-fold P(rejected) and the gate's surprise on the same attempts.

    uv run python -m k2cascade.probe_vs_gate --gate-dir analysis/gate --probe-dir analysis/hidden --out analysis/probe_vs_gate.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import bootstrap_auroc, bootstrap_spearman

SIZES = (("0.9b", "k2-0.9b", ("0.9b", "3.7b")), ("3.7b", "k2-3.7b", ("3.7b",)))
GATE_FEATURES = ("mean", "max", "mean_action", "mean_think")


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _key(r: dict[str, Any]) -> tuple[str, int, int]:
    return (r["run_id"], int(r["step"]), int(r.get("attempt", 0)))


def _ci(d: dict[str, float]) -> str:
    if d is None or np.isnan(d.get("auroc", float("nan"))):
        return "n/a"
    return f"{d['auroc']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def joined(gate_rows: list[dict], probe_rows: list[dict]) -> list[tuple[dict, dict]]:
    g = {_key(r): r for r in gate_rows}
    return [(g[_key(p)], p) for p in probe_rows if _key(p) in g]


def size_section(size: str, trace_model: str, scorers: tuple[str, ...], gate_dir: Path, probe_dir: Path, n_boot: int) -> tuple[list[str], dict[str, Any]]:
    probe = json.loads((probe_dir / f"probe_{size}.json").read_text())
    probe_rows = _jsonl(probe_dir / f"probe_{size}.jsonl")
    gates = {s: json.loads((gate_dir / f"gate_{trace_model}_by_{s}.json").read_text()) for s in scorers}
    gate_rows = {s: _jsonl(gate_dir / f"gate_{trace_model}_by_{s}.jsonl") for s in scorers}
    b = probe["best"]
    lines = [f"## {trace_model}: n = {probe['n']} attempts ({probe['n_rejected']} rejected) from {probe['n_runs']} runs", ""]
    lines.append("| signal | when | AUROC for ok == False [95% CI] | sign flipped (1 - AUROC) |")
    lines.append("|---|---|---|---|")
    for s in scorers:
        gs = gates[s]
        tag = "self-surprise" if s == size else f"scored by {s}"
        for f in GATE_FEATURES:
            d = gs["features"][f]
            star = f" **(best gate feature, {gs['best_sign'].replace('_', ' ')})**" if f == gs["best_feature"] else ""
            flipped = f"{d['auroc_flipped']:.3f}" if not np.isnan(d.get("auroc_flipped", float("nan"))) else "n/a"
            lines.append(f"| gate {tag}: {f} NLL (n={d['n']}) | post-hoc | {_ci(d)}{star} | {flipped} |")
    lines.append(f"| probe, best layer {b['layer']} of {probe['n_layers_plus_norm']} (LORO-CV, optimistic: best-of-{probe['n_layers_plus_norm']}) | pre-action | {_ci(b)} | |")
    lines.append(f"| probe, median over layers | pre-action | {probe['median_layer_auroc']:.3f} | |")
    for name, d in probe["fixed_layers"].items():
        lines.append(f"| probe, {name.replace('_', ' ')} (layer {d['layer']}) | pre-action | {_ci(d)} | |")
    lines.append(f"| probe, mean of last {probe['capture']['tail']} prompt tokens at layer {b['layer']} | pre-action | {_ci(probe['baselines']['tail_mean_best_layer'])} | |")
    lines.append(f"| baseline: prompt length | pre-action | {_ci(probe['baselines']['prompt_length'])} | |")
    lines.append(f"| baseline: step index | pre-action | {_ci(probe['baselines']['step_index'])} | |")
    cs = ", ".join(f"C={c}: {a:.3f}" for c, a in probe["C_sensitivity_at_best_layer"].items())
    lines += ["", f"Probe: standardized features + L2 logistic regression, C={probe['C']}; at the best layer, {cs}.",
              f"Per-layer AUROC: " + " ".join(f"{p['layer']}:{p['auroc']:.2f}" for p in probe["per_layer"]) + "."]
    if "permutation_null_within_run" in probe:
        pn = probe["permutation_null_within_run"]
        lines.append(f"Selection check: with labels shuffled within each run ({pn['n_perm']} permutations), the best-of-{probe['n_layers_plus_norm']}-layers "
                     f"AUROC averages {pn['best_of_layers_mean']:.3f} (95th pct {pn['best_of_layers_q95']:.3f}, max {pn['best_of_layers_max']:.3f}); "
                     f"observed {pn['observed_best']:.3f}, permutation p = {pn['p_value']:.3f}.")
    # per-reason view: the gate's self-surprise and the probe's P(rejected), split by the verifier's reason
    pairs_self = joined(gate_rows[size], probe_rows)
    by_reason: dict[str, list[tuple[float, float, float]]] = {}
    for g, p in pairs_self:
        by_reason.setdefault(g["reason"], []).append((g["features"]["mean"], g["features"]["max"], p["p_fail_best_layer"]))
    lines += ["", "By verifier reason (self-surprise gate vs probe, medians):", "",
              "| reason | n | median mean NLL | median max NLL | median probe P(rejected) |", "|---|---|---|---|---|"]
    for reason, vals in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        v = np.array(vals)
        lines.append(f"| {reason} | {len(v)} | {np.median(v[:, 0]):.3f} | {np.median(v[:, 1]):.2f} | {np.median(v[:, 2]):.2f} |")
    # correlation probe vs gate, on the joined attempts
    stats: dict[str, Any] = {"size": size, "n": probe["n"]}
    lines += ["", "Spearman between the probe's out-of-fold P(rejected) (best layer) and the gate's surprise, same attempts:", ""]
    lines.append("| gate signal | n | Spearman rho [95% CI] | rho on accepted attempts only |")
    lines.append("|---|---|---|---|")
    for s in scorers:
        pairs = joined(gate_rows[s], probe_rows)
        for f in ("mean", "max", "mean_action"):
            x = np.array([g["features"][f] for g, _ in pairs], dtype=np.float64)
            p = np.array([pr["p_fail_best_layer"] for _, pr in pairs], dtype=np.float64)
            ok = np.array([g["ok"] for g, _ in pairs])
            keep = ~np.isnan(x)
            rho, lo, hi = bootstrap_spearman(p[keep], x[keep], n_boot=n_boot)
            rho_ok = bootstrap_spearman(p[keep & ok], x[keep & ok], n_boot=n_boot)[0]
            tag = "self" if s == size else f"by {s}"
            lines.append(f"| {tag} {f} NLL | {int(keep.sum())} | {rho:+.3f} [{lo:+.3f}, {hi:+.3f}] | {rho_ok:+.3f} |")
            stats[f"rho_{s}_{f}"] = rho
    # the gate's own AUROC recomputed on the joined set, to confirm same attempts
    pairs = joined(gate_rows[size], probe_rows)
    y = np.array([not g["ok"] for g, _ in pairs])
    bf = gates[size]["best_feature"]
    x = np.array([g["features"][bf] for g, _ in pairs])
    keep = ~np.isnan(x)
    a_g = bootstrap_auroc(x[keep], y[keep], n_boot=n_boot)
    a_p = bootstrap_auroc(np.array([p["p_fail_best_layer"] for _, p in pairs]), y, n_boot=n_boot)
    lines += ["", f"Same-attempt check (joined n={len(pairs)}): gate best feature `{bf}` AUROC {a_g[0]:.3f} [{a_g[1]:.3f}, {a_g[2]:.3f}] vs "
                  f"probe best layer AUROC {a_p[0]:.3f} [{a_p[1]:.3f}, {a_p[2]:.3f}]."]
    thr = {}
    for s in scorers:
        thr[s] = f"{gates[s]['tok_per_s']:.0f} tok/s ({gates[s]['tokens_seen']} prompt+target tokens, peak {gates[s]['peak_mem_gb']:.1f} GB)"
    cap = probe["capture"]
    lines += ["", "Throughput: gate " + "; ".join(f"{s}: {t}" for s, t in thr.items()) +
              f". Probe capture {cap['tok_per_s']:.0f} tok/s ({cap['tokens_seen']} prompt tokens, peak {cap['peak_mem_gb']:.1f} GB); "
              f"probe fit {probe['fit_seconds']:.0f} s."]
    rejected = {}
    for g, _ in pairs:
        rejected[g["reason"]] = rejected.get(g["reason"], 0) + 1
    lines += ["", "Reasons (joined attempts): " + ", ".join(f"{k} x{v}" for k, v in sorted(rejected.items(), key=lambda kv: -kv[1])) + "."]
    return lines, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", type=Path, default=Path("analysis/gate"))
    ap.add_argument("--probe-dir", type=Path, default=Path("analysis/hidden"))
    ap.add_argument("--out", type=Path, default=Path("analysis/probe_vs_gate.md"))
    ap.add_argument("--reading", type=Path, default=None, help="markdown file with the plain-language reading to append")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    lines = ["# Pre-action probe vs post-hoc surprise gate on K2 Horizon", "",
             "Same attempts, per model size. Label: ok == False (the verifier rejected the attempt). AUROC is rank-based;",
             "CIs are percentile bootstraps over attempts (2000 resamples). The gate scores each attempt's own output by",
             "teacher forcing (post-hoc, arXiv 2609.05274 style); the probe reads the residual stream at the last prompt",
             "position before anything is generated (pre-action), with leave-one-run-out cross-validation (folds = run_id;",
             "runs of the same task appear on both sides of a fold, by design).", ""]
    for size, trace_model, scorers in SIZES:
        sec, _ = size_section(size, trace_model, scorers, args.gate_dir, args.probe_dir, args.n_boot)
        lines += sec + [""]
    if args.reading and args.reading.exists():
        lines += ["## Reading", "", args.reading.read_text().strip(), ""]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
