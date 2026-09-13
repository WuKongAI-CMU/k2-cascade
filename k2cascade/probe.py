"""Pre-action probe: can the model's hidden state on the PROMPT ALONE predict that its step will be rejected?

Capture: for every attempt of `--trace-model`, render the prompt it saw (reasoning_effort=low, so it ends in
`<|ifm|im_start|>assistant\\n<ifm|think_faster>\\n`), run `--model` (the SAME size that generated the attempt)
over the prompt only, and keep the residual stream at the last prompt position after every layer plus the
final-norm output (n_attempts x (n_layers + 1) x hidden), and the mean over the last `--tail` prompt positions.

Fit: per layer, standardize + L2 logistic regression predicting ok == False, leave-one-run-out
cross-validation (folds = run_id). Out-of-fold probabilities are pooled and scored with a rank AUROC and
a bootstrap CI over attempts. Runs of the same task sit in both train and test folds; that is by design
(the question is "does this prompt lead to a rejected step", not "generalize to unseen tasks").

    uv run python -m k2cascade.probe --model ~/models/k2/0.9b-mlx-8bit --size-name 0.9b --trace-model k2-0.9b --out analysis/hidden
    uv run python -m k2cascade.probe --size-name 0.9b --trace-model k2-0.9b --out analysis/hidden --skip-capture   # refit only
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import auroc, bootstrap_auroc, leave_one_group_out
from .prompt import render
from .surprise import load_attempts
from .tools import TOOLS

C_GRID = (0.01, 0.1, 1.0, 10.0)


# --------------------------------------------------------------------------- capture

def chunk_bounds(n: int, chunk: int, tail: int) -> list[tuple[int, int]]:
    """Split [0, n) into consecutive chunks of `chunk` positions; the last chunk keeps >= tail positions
    (merged with the previous one if needed) so the tail mean can be read from a single forward call."""
    if n <= 0:
        return []
    starts = list(range(0, n, chunk))
    bounds = [(s, min(s + chunk, n)) for s in starts]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < tail:
        s, _ = bounds[-2]
        bounds = bounds[:-2] + [(s, n)]
    return bounds


def hidden_states(model, ids: list[int], chunk: int = 512, tail: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """(last, tail_mean), each [n_layers + 1, hidden] float32: residual stream after every layer at the last
    prompt position, plus the final-norm output as the extra row. Mirrors K2HorizonModel.__call__ with a KV cache."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache

    inner = model.model
    cache = make_prompt_cache(model)
    bounds = chunk_bounds(len(ids), chunk, tail)
    last = tail_mean = None
    for i, (s, e) in enumerate(bounds):
        h = inner.embed_tokens(mx.array(ids[s:e])[None])
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        swa_mask = None
        if inner.swa_idx is not None:
            swa_mask = create_attention_mask(h, cache[inner.swa_idx], window_size=inner.sliding_window)
        hs = []
        for layer, c in zip(inner.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, c)
            hs.append(h)
        hs.append(inner.norm(h))
        if i == len(bounds) - 1:
            k = min(tail, e - s)
            last = mx.stack([x[0, -1] for x in hs]).astype(mx.float32)
            tail_mean = mx.stack([x[0, -k:].astype(mx.float32).mean(axis=0) for x in hs])
            mx.eval(last, tail_mean)
        else:
            mx.eval([c.state for c in cache])
        del hs, h
    mx.clear_cache()
    return np.array(last), np.array(tail_mean)


def capture(model_path: str, attempts: list[tuple[dict[str, Any], list[dict[str, Any]]]], effort: str, chunk: int, tail: int,
            out_npz: Path) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(Path(model_path).expanduser()), tokenizer_config={"trust_remote_code": True})
    lasts, tails, meta = [], [], []
    tokens, secs = 0, 0.0
    for r, msgs in attempts:
        prompt = render(msgs, TOOLS, reasoning_effort=effort)
        ids = tok._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t0 = time.perf_counter()
        last, tm = hidden_states(model, ids, chunk=chunk, tail=tail)
        dt = time.perf_counter() - t0
        tokens += len(ids)
        secs += dt
        lasts.append(last)
        tails.append(tm)
        meta.append({"run_id": r["run_id"], "step": r["step"], "attempt": r.get("attempt", 0), "task": r.get("task"),
                     "model": r["model"], "ok": bool(r["ok"]), "reason": r.get("reason"), "n_prompt_tokens": len(ids)})
        print(f"  {r['run_id']} step {r['step']} attempt {r.get('attempt', 0)} ok={r['ok']} {r.get('reason')}: {len(ids)} tok, "
              f"{dt:.1f}s, peak {mx.get_peak_memory() / 2**30:.1f} GB")
    last_arr, tail_arr = np.stack(lasts), np.stack(tails)
    info = {"model_path": model_path, "effort": effort, "chunk": chunk, "tail": tail, "tokens_seen": tokens,
            "forward_seconds": secs, "tok_per_s": tokens / secs if secs else 0.0, "peak_mem_gb": mx.get_peak_memory() / 2**30,
            "shape": list(last_arr.shape)}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, last=last_arr, tail_mean=tail_arr, meta=json.dumps(meta), info=json.dumps(info))
    return info


# --------------------------------------------------------------------------- fit

def fit_oof(X: np.ndarray, y: np.ndarray, groups: list[Any], C: float = 1.0) -> np.ndarray:
    """Out-of-fold P(rejected) with leave-one-group-out folds; standardization stats come from the train fold only."""
    from sklearn.linear_model import LogisticRegression

    oof = np.full(len(y), np.nan)
    for train, test in leave_one_group_out(groups):
        if len(train) == 0 or y[train].all() or not y[train].any():
            oof[test] = float(y[train].mean()) if len(train) else 0.5
            continue
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=5000)  # default penalty is L2
        clf.fit((X[train] - mu) / sd, y[train])
        oof[test] = clf.predict_proba((X[test] - mu) / sd)[:, 1]
    return oof


def permute_within_runs(y: np.ndarray, groups: list[Any], rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels inside each run, keeping every run's number of rejected attempts."""
    yp = y.copy()
    for g in set(groups):
        idx = np.array([i for i, x in enumerate(groups) if x == g], dtype=int)
        yp[idx] = y[idx][rng.permutation(len(idx))]
    return yp


def permutation_null(last: np.ndarray, y: np.ndarray, groups: list[Any], C: float, n_perm: int, seed: int = 0) -> np.ndarray:
    """Null distribution of the best-over-layers LORO AUROC when labels are shuffled within runs.
    Answers: how high does 'best of L layers' get by chance on this n, with the run structure kept?"""
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(n_perm):
        yp = permute_within_runs(y, groups, rng)
        if yp.all() or not yp.any():
            continue
        best.append(max(auroc(fit_oof(last[:, l, :], yp, groups, C=C), yp) for l in range(last.shape[1])))
    return np.array(best)


def fit_all(last: np.ndarray, tail_mean: np.ndarray, meta: list[dict[str, Any]], C: float, n_boot: int, n_perm: int = 0) -> tuple[dict[str, Any], np.ndarray]:
    y = np.array([not m["ok"] for m in meta])
    groups = [m["run_id"] for m in meta]
    n, L1, _ = last.shape
    per_layer = []
    oof_by_layer = np.zeros((L1, n))
    for l in range(L1):
        oof = fit_oof(last[:, l, :], y, groups, C=C)
        oof_by_layer[l] = oof
        a, lo, hi = bootstrap_auroc(oof, y, n_boot=n_boot)
        per_layer.append({"layer": l, "auroc": a, "lo": lo, "hi": hi})
    best = int(np.argmax([p["auroc"] for p in per_layer]))
    aurocs = np.array([p["auroc"] for p in per_layer])
    # sensitivity to C at the best layer, plus the fixed-layer / baseline probes
    c_sens = {str(c): auroc(fit_oof(last[:, best, :], y, groups, C=c), y) for c in C_GRID}
    tail_oof = fit_oof(tail_mean[:, best, :], y, groups, C=C)
    tail_a = bootstrap_auroc(tail_oof, y, n_boot=n_boot)
    n_prompt = np.array([m["n_prompt_tokens"] for m in meta], dtype=np.float64)
    steps = np.array([m["step"] for m in meta], dtype=np.float64)
    mid = L1 // 2
    summary = {
        "n": int(n), "n_rejected": int(y.sum()), "n_runs": len(set(groups)), "n_layers_plus_norm": int(L1), "hidden": int(last.shape[2]),
        "C": C, "per_layer": per_layer, "best_layer": best,
        "best": {"layer": best, "auroc": per_layer[best]["auroc"], "lo": per_layer[best]["lo"], "hi": per_layer[best]["hi"],
                 "note": "best-of-%d layers on the same CV; selection is optimistic" % L1},
        "median_layer_auroc": float(np.median(aurocs)), "min_layer_auroc": float(aurocs.min()),
        "fixed_layers": {"middle": {"layer": mid, **{k: per_layer[mid][k] for k in ("auroc", "lo", "hi")}},
                         "last_block": {"layer": L1 - 2, **{k: per_layer[L1 - 2][k] for k in ("auroc", "lo", "hi")}},
                         "final_norm": {"layer": L1 - 1, **{k: per_layer[L1 - 1][k] for k in ("auroc", "lo", "hi")}}},
        "C_sensitivity_at_best_layer": c_sens,
        "baselines": {
            "tail_mean_best_layer": {"auroc": tail_a[0], "lo": tail_a[1], "hi": tail_a[2]},
            "prompt_length": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(n_prompt, y, n_boot=n_boot))),
            "step_index": dict(zip(("auroc", "lo", "hi"), bootstrap_auroc(steps, y, n_boot=n_boot))),
        },
    }
    if n_perm:
        null = permutation_null(last, y, groups, C, n_perm)
        observed = per_layer[best]["auroc"]
        summary["permutation_null_within_run"] = {
            "n_perm": int(len(null)), "best_of_layers_mean": float(null.mean()), "best_of_layers_q95": float(np.quantile(null, 0.95)),
            "best_of_layers_max": float(null.max()), "observed_best": observed,
            "p_value": float((1 + (null >= observed).sum()) / (len(null) + 1)),
        }
    return summary, oof_by_layer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to the MLX model directory (the size that produced the attempts)")
    ap.add_argument("--size-name", required=True, help="label, e.g. 0.9b or 3.7b")
    ap.add_argument("--trace-model", required=True, help="attempts whose `model` field matches, e.g. k2-0.9b")
    ap.add_argument("--traces", nargs="*", type=Path, default=None, help="default: traces/*.jsonl")
    ap.add_argument("--out", type=Path, default=Path("analysis/hidden"))
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength for the probe")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=0, help="within-run label permutations for the best-of-layers null (slow: one full sweep each)")
    ap.add_argument("--skip-capture", action="store_true", help="reuse <out>/hidden_<size>.npz and only refit")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    npz = args.out / f"hidden_{args.size_name}.npz"
    if not args.skip_capture:
        if not args.model:
            ap.error("--model is required unless --skip-capture")
        traces = args.traces or [Path(p) for p in sorted(glob.glob("traces/*.jsonl"))]
        attempts = load_attempts(traces, args.trace_model, ok_only=False)
        if args.limit:
            attempts = attempts[:args.limit]
        print(f"probe: capturing {len(attempts)} {args.trace_model} prompts with {args.size_name}")
        info = capture(args.model, attempts, args.reasoning_effort, args.chunk, args.tail, npz)
        print(json.dumps(info, indent=2))
    data = np.load(npz)
    last, tail_mean = data["last"], data["tail_mean"]
    meta, info = json.loads(str(data["meta"])), json.loads(str(data["info"]))
    t0 = time.perf_counter()
    summary, oof_by_layer = fit_all(last, tail_mean, meta, args.C, args.n_boot, n_perm=args.n_perm)
    summary.update({"size": args.size_name, "trace_model": args.trace_model, "capture": info, "fit_seconds": time.perf_counter() - t0})
    with (args.out / f"probe_{args.size_name}.jsonl").open("w") as fh:
        for i, m in enumerate(meta):
            row = dict(m)
            row["p_fail_best_layer"] = float(oof_by_layer[summary["best_layer"], i])
            row["p_fail_by_layer"] = [round(float(x), 5) for x in oof_by_layer[:, i]]
            fh.write(json.dumps(row) + "\n")
    (args.out / f"probe_{args.size_name}.json").write_text(json.dumps(summary, indent=2))
    b = summary["best"]
    print(f"probe {args.size_name}: n={summary['n']} rejected={summary['n_rejected']} runs={summary['n_runs']} "
          f"best layer {b['layer']} AUROC {b['auroc']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] "
          f"median over layers {summary['median_layer_auroc']:.3f}; tail-mean {summary['baselines']['tail_mean_best_layer']['auroc']:.3f}; "
          f"prompt-length {summary['baselines']['prompt_length']['auroc']:.3f}")
    if "permutation_null_within_run" in summary:
        pn = summary["permutation_null_within_run"]
        print(f"  within-run permutation null for best-of-layers ({pn['n_perm']} perms): mean {pn['best_of_layers_mean']:.3f}, "
              f"q95 {pn['best_of_layers_q95']:.3f}, max {pn['best_of_layers_max']:.3f}; observed {pn['observed_best']:.3f}, p = {pn['p_value']:.3f}")


if __name__ == "__main__":
    main()
