"""Post-hoc surprise gate (arXiv 2609.05274 on K2): score each attempt's OWN output by teacher forcing.

For every attempt of `--trace-model` in the traces, rebuild its prompt (prompt.render, reasoning_effort=low,
so the prompt ends in `<ifm|think_faster>\\n` exactly as at generation time) and score the attempt's `raw`
output token by token with the local model given as `--model`. Per attempt we keep the per-token NLL and
role, and a few summary features (mean / max NLL, mean over action tokens, mean over think tokens);
each feature is then evaluated as a predictor of ok == False with a rank AUROC + bootstrap CI.

    uv run python -m k2cascade.gate --model ~/models/k2/0.9b-mlx-8bit --size-name 0.9b --trace-model k2-0.9b --out analysis/gate
    uv run python -m k2cascade.gate --model ~/models/k2/3.7b-mlx-8bit --size-name 3.7b --trace-model k2-0.9b --out analysis/gate
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import bootstrap_auroc
from .prompt import render
from .surprise import ARG_ROLES, Scorer, load_attempts, role_at
from .tools import TOOLS

STOP_TOKENS = ("<|ifm|im_end|>", "<|ifm|endoftext|>")
THINK_CLOSE_RE = re.compile(r"</ifm\|(?:think|think_fast|think_faster)>")
THINK_OPEN_RE = re.compile(r"^<ifm\|(?:think|think_fast|think_faster)>\n?")
CALLS_RE = re.compile(r"<ifm\|tool_calls>.*?</ifm\|tool_calls>", re.S)
TAG_RE = re.compile(r"</?ifm\|(?:tool_calls|tool_call|arg_key|arg_value|arg_type)>")
ACTION_ROLES = ("tool_name", "arg_key", "arg_value", "arg_path", "arg_cmd", "arg_content")
FEATURES = ("mean", "max", "mean_action", "mean_think", "mean_prose", "mean_markup", "sum", "n_tokens")


def strip_stop(raw: str) -> str:
    for s in STOP_TOKENS:
        raw = raw.split(s)[0]
    return raw


def _calls_segments(body: str) -> list[tuple[str, str]]:
    """Label the inside of a <ifm|tool_calls>...</ifm|tool_calls> block. Tags are markup; the line after
    <ifm|tool_call> is the tool name; arg keys/values get surprise.py's roles."""
    seg: list[tuple[str, str]] = []
    pos, state, key = 0, "markup", ""
    for m in TAG_RE.finditer(body):
        text = body[pos:m.start()]
        if text:
            if state == "tool_name":
                name, nl, rest = text.partition("\n")
                seg.append((name, "tool_name"))
                if nl or rest:
                    seg.append((nl + rest, "markup"))
            elif state == "arg_key":
                key = text.strip()
                seg.append((text, "arg_key"))
            elif state == "arg_value":
                seg.append((text, ARG_ROLES.get(key, "arg_value")))
            else:
                seg.append((text, "markup"))
        seg.append((m.group(0), "markup"))
        tag = m.group(0)
        state = {"<ifm|tool_call>": "tool_name", "<ifm|arg_key>": "arg_key", "<ifm|arg_value>": "arg_value"}.get(tag, "markup")
        pos = m.end()
    tail = body[pos:]
    if tail:
        seg.append((tail, "markup"))
    return seg


def raw_segments(raw: str) -> list[tuple[str, str]]:
    """Split a raw K2 completion (as generated AFTER `<ifm|think_faster>\\n`) into (text, role) segments.
    Handles `</ifm|think>` and `</ifm|think_faster>` closers alike; an unclosed think block is all think."""
    text = strip_stop(raw)
    seg: list[tuple[str, str]] = []
    m_open = THINK_OPEN_RE.match(text)
    if m_open:  # the model re-emitted the opening tag (rare); it is markup, not thought
        seg.append((m_open.group(0), "markup"))
        text = text[m_open.end():]
    m = THINK_CLOSE_RE.search(text)
    if not m:
        if text:
            seg.append((text, "think"))
        return seg
    if m.start():
        seg.append((text[:m.start()], "think"))
    seg.append((m.group(0), "markup"))
    rest = text[m.end():]
    pos = 0
    for cm in CALLS_RE.finditer(rest):
        if cm.start() > pos:
            seg.append((rest[pos:cm.start()], "prose"))
        seg += _calls_segments(cm.group(0))
        pos = cm.end()
    if pos < len(rest):
        seg.append((rest[pos:], "prose"))
    return seg


def features(nll: np.ndarray, roles: list[str]) -> dict[str, float]:
    def mean_over(pred) -> float:
        v = [x for x, r in zip(nll, roles) if pred(r)]
        return float(statistics.fmean(v)) if v else float("nan")

    return {
        "mean": float(nll.mean()) if len(nll) else float("nan"),
        "max": float(nll.max()) if len(nll) else float("nan"),
        "mean_action": mean_over(lambda r: r in ACTION_ROLES),
        "mean_think": mean_over(lambda r: r == "think"),
        "mean_prose": mean_over(lambda r: r == "prose"),
        "mean_markup": mean_over(lambda r: r == "markup"),
        "sum": float(nll.sum()),
        "n_tokens": float(len(nll)),
    }


def score_attempt(scorer: Scorer, rec: dict[str, Any], messages: list[dict[str, Any]], effort: str = "low") -> dict[str, Any]:
    prompt = render(messages, TOOLS, reasoning_effort=effort)
    segments = raw_segments(rec.get("raw") or "")
    target = "".join(t for t, _ in segments)
    prompt_ids, _ = scorer.encode(prompt)
    target_ids, offsets = scorer.encode(target)
    if not target_ids:
        nll, roles = np.zeros(0), []
    else:
        nll = -scorer.logprobs(prompt_ids, target_ids)
        roles = [role_at(segments, s, e) for s, e in offsets]
    return {
        "run_id": rec["run_id"], "step": rec["step"], "attempt": rec.get("attempt", 0), "task": rec.get("task"),
        "model": rec["model"], "ok": bool(rec["ok"]), "reason": rec.get("reason"),
        "n_prompt_tokens": len(prompt_ids), "n_target_tokens": len(target_ids),
        "features": features(nll, roles),
        "nll": [round(float(x), 4) for x in nll], "roles": roles,
    }


def evaluate(rows: list[dict[str, Any]], n_boot: int = 2000, min_n: int = 20) -> dict[str, Any]:
    """AUROC of each feature for predicting ok == False with the paper's sign (higher surprise -> reject).
    `auroc_flipped` = 1 - AUROC is the same feature read the other way (low surprise -> reject); a value far
    below 0.5 means the signal exists with the opposite sign. Attempts where a feature is nan (e.g. no think
    tokens) are dropped for that feature and counted. `best_feature` maximises |AUROC - 0.5| among features
    defined on at least `min_n` attempts; `best_sign` says which direction that is (a post-hoc choice)."""
    y_all = np.array([not r["ok"] for r in rows])
    out: dict[str, Any] = {"n": len(rows), "n_rejected": int(y_all.sum()), "features": {}}
    for f in FEATURES:
        x = np.array([r["features"][f] for r in rows], dtype=np.float64)
        keep = ~np.isnan(x)
        if keep.sum() < 3 or y_all[keep].all() or not y_all[keep].any():
            out["features"][f] = {"auroc": float("nan"), "lo": float("nan"), "hi": float("nan"), "auroc_flipped": float("nan"), "n": int(keep.sum())}
            continue
        a, lo, hi = bootstrap_auroc(x[keep], y_all[keep], n_boot=n_boot)
        out["features"][f] = {"auroc": a, "lo": lo, "hi": hi, "auroc_flipped": 1.0 - a, "n": int(keep.sum())}
    valid = [f for f in FEATURES if not np.isnan(out["features"][f]["auroc"]) and out["features"][f]["n"] >= min(min_n, len(rows))]
    if valid:
        best = max(valid, key=lambda f: abs(out["features"][f]["auroc"] - 0.5))
        out["best_feature"], out["best_sign"] = best, ("high_surprise_rejects" if out["features"][best]["auroc"] >= 0.5 else "low_surprise_rejects")
    else:
        out["best_feature"], out["best_sign"] = None, None
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the MLX scorer model directory")
    ap.add_argument("--size-name", required=True, help="label for the scorer, e.g. 0.9b or 3.7b")
    ap.add_argument("--trace-model", required=True, help="score attempts whose `model` field matches, e.g. k2-0.9b")
    ap.add_argument("--traces", nargs="*", type=Path, default=None, help="default: traces/*.jsonl")
    ap.add_argument("--out", type=Path, default=Path("analysis/gate"))
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="debug: only score the first N attempts")
    ap.add_argument("--reevaluate", action="store_true", help="skip scoring; recompute the summary from the existing per-attempt JSONL")
    args = ap.parse_args(argv)

    stem = f"gate_{args.trace_model}_by_{args.size_name}"
    if args.reevaluate:
        old = json.loads((args.out / f"{stem}.json").read_text())
        rows = [json.loads(l) for l in (args.out / f"{stem}.jsonl").read_text().splitlines() if l.strip()]
        summary = evaluate(rows)
        old.update(summary)
        (args.out / f"{stem}.json").write_text(json.dumps(old, indent=2))
        print(json.dumps({k: v for k, v in old.items() if k != "traces"}, indent=2))
        return

    import mlx.core as mx

    traces = args.traces or [Path(p) for p in sorted(glob.glob("traces/*.jsonl"))]
    attempts = load_attempts(traces, args.trace_model, ok_only=False)
    if args.limit:
        attempts = attempts[:args.limit]
    scorer = Scorer(args.model, chunk=args.chunk)
    print(f"gate: scoring {len(attempts)} {args.trace_model} attempts with {args.size_name}")
    rows = []
    t_all = time.perf_counter()
    for r, msgs in attempts:
        t0 = time.perf_counter()
        row = score_attempt(scorer, r, msgs, args.reasoning_effort)
        rows.append(row)
        f = row["features"]
        print(f"  {row['run_id']} step {row['step']} attempt {row['attempt']} ok={row['ok']} {row['reason']}: "
              f"{row['n_prompt_tokens']}+{row['n_target_tokens']} tok, mean nll {f['mean']:.3f} max {f['max']:.2f}, "
              f"{time.perf_counter() - t0:.1f}s, peak {mx.get_peak_memory() / 2**30:.1f} GB")
    wall = time.perf_counter() - t_all
    summary = evaluate(rows)
    summary.update({
        "scorer": args.size_name, "model_path": str(args.model), "trace_model": args.trace_model, "effort": args.reasoning_effort,
        "traces": [str(p) for p in traces], "tokens_seen": scorer.tokens_seen, "forward_seconds": scorer.seconds,
        "tok_per_s": scorer.tokens_seen / scorer.seconds if scorer.seconds else 0.0, "wall_seconds": wall,
        "peak_mem_gb": mx.get_peak_memory() / 2**30,
    })
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"{stem}.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "traces"}, indent=2))


if __name__ == "__main__":
    main()
