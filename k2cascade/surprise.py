"""Surprise map: score one model's agent-step outputs token by token with a local K2 (teacher forcing).

For every attempt in the given traces we rebuild the exact prompt the local models see
(prompt.render) and the assistant turn in K2's native XML format, then ask the local
model for log p(token | prefix) at every target token. Each token gets a role label from
the character span it came from (think / prose / markup / tool_name / arg_key / arg_*).

    uv run python -m k2cascade.surprise --model ~/models/k2/3.7b-mlx-8bit \\
        --traces traces/large-3.jsonl traces/large-4.jsonl --out analysis/
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from .prompt import render
from .tools import TOOLS

IM_END = "<|ifm|im_end|>"
ARG_ROLES = {"path": "arg_path", "cmd": "arg_cmd", "content": "arg_content"}
ROLES = ["think", "prose", "markup", "tool_name", "arg_key", "arg_value", "arg_path", "arg_cmd", "arg_content"]


# --------------------------------------------------------------------------- target text

def _value_text(v: Any) -> str:
    """XML tool-call format: scalars as plain text, dict/list as JSON (matches the chat template)."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def clean_thinking(thinking: str) -> tuple[str, bool]:
    """Drop literal think tags that the 375B endpoint's reasoning parser sometimes leaks into the
    reasoning field (e.g. '</ifm|think>\\n' or '<ifm|think>\\n</ifm|think>\\n'). Returns (text, changed)."""
    out = thinking
    for tag in ("<ifm|think>\n", "<ifm|think>", "</ifm|think>\n", "</ifm|think>"):
        out = out.replace(tag, "")
    return out, out != thinking


def build_target(thinking: str, content: str, tool_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return the assistant turn as (text, role) segments, WITHOUT the opening `<ifm|think>\\n`
    (the generation prompt already opened the think block)."""
    seg: list[tuple[str, str]] = []
    if thinking:
        seg.append((thinking, "think"))
    seg.append(("</ifm|think>", "markup"))
    if content:
        seg.append((content, "prose"))
    if tool_calls:
        seg.append(("<ifm|tool_calls>", "markup"))
        for call in tool_calls:
            seg.append(("\n<ifm|tool_call>", "markup"))
            seg.append((call["name"], "tool_name"))
            seg.append(("\n", "markup"))
            for k, v in call["arguments"].items():
                seg.append(("<ifm|arg_key>", "markup"))
                seg.append((k, "arg_key"))
                seg.append(("</ifm|arg_key>\n<ifm|arg_value>", "markup"))
                seg.append((_value_text(v), ARG_ROLES.get(k, "arg_value")))
                seg.append(("</ifm|arg_value>\n", "markup"))
            seg.append(("</ifm|tool_call>", "markup"))
        seg.append(("\n</ifm|tool_calls>", "markup"))
    seg.append((IM_END, "markup"))
    return seg


def role_at(segments: list[tuple[str, str]], start: int, end: int) -> str:
    """Role of the segment with the largest overlap with the character span [start, end)."""
    best, best_ov, pos = "prose", -1, 0
    for text, role in segments:
        s, e = pos, pos + len(text)
        ov = min(e, end) - max(s, start)
        if ov > best_ov:
            best, best_ov = role, ov
        pos = e
        if s >= end:
            break
    return best


# --------------------------------------------------------------------------- scoring

class Scorer:
    def __init__(self, model_path: str, chunk: int = 512):
        from mlx_lm import load

        # config.json's `model_file` makes mlx-lm import k2_horizon_mlx.py; the tokenizer flag stops transformers' y/N prompt.
        self.model, self.tok = load(str(Path(model_path).expanduser()), tokenizer_config={"trust_remote_code": True})
        self.chunk = chunk
        self.tokens_seen = 0
        self.seconds = 0.0

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def token_str(self, tid: int) -> str:
        return self.tok.decode([tid])

    def logprobs(self, prompt_ids: list[int], target_ids: list[int]) -> np.ndarray:
        """log p(target_ids[j] | prompt_ids + target_ids[:j]) for every j."""
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = prompt_ids + target_ids
        n_prompt = len(prompt_ids)
        cache = make_prompt_cache(self.model)
        out: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, len(ids) - 1, self.chunk):  # the last token predicts nothing
            end = min(start + self.chunk, len(ids) - 1)
            logits = self.model(mx.array(ids[start:end])[None], cache=cache)
            first_needed = max(start, n_prompt - 1)  # position p predicts ids[p + 1]
            if first_needed < end:
                lg = logits[0, first_needed - start:end - start].astype(mx.float32)
                targets = mx.array(ids[first_needed + 1:end + 1])
                lp = mx.take_along_axis(lg, targets[:, None], axis=-1)[:, 0] - mx.logsumexp(lg, axis=-1)
                mx.eval(lp)
                out.append(np.array(lp, dtype=np.float64))
            else:
                mx.eval([c.state for c in cache])  # prompt-only chunk: skip the lm_head
            del logits
        mx.clear_cache()
        self.seconds += time.perf_counter() - t0
        self.tokens_seen += len(ids)
        return np.concatenate(out)


def score_attempt(scorer: Scorer, rec: dict[str, Any], messages: list[dict[str, Any]], effort: str, strip_ws: bool = False) -> list[dict[str, Any]]:
    prompt = render(messages, TOOLS, reasoning_effort=effort)
    thinking, content = clean_thinking(rec.get("thinking") or "")[0], rec.get("content") or ""
    if strip_ws:  # same normalisation parse.py applies to the local models' raw output
        thinking, content = thinking.strip(), content.strip()
    segments = build_target(thinking, content, rec.get("tool_calls") or [])
    target = "".join(t for t, _ in segments)
    prompt_ids, _ = scorer.encode(prompt)
    target_ids, offsets = scorer.encode(target)
    lps = scorer.logprobs(prompt_ids, target_ids)
    return [
        {
            "run": rec["run_id"], "step": rec["step"], "attempt": rec.get("attempt", 0), "pos": j,
            "role": role_at(segments, s, e), "token": scorer.token_str(tid), "logprob": float(lps[j]),
        }
        for j, (tid, (s, e)) in enumerate(zip(target_ids, offsets))
    ]


# --------------------------------------------------------------------------- traces

def load_attempts(paths: list[Path], model: str | None, ok_only: bool) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Return (record, messages) for every scoreable attempt. Attempts >0 carry no messages; reuse attempt 0's."""
    out = []
    for p in paths:
        seen_msgs: dict[tuple[str, int], list] = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("step") == 0:  # run summary line
                continue
            if r.get("messages"):
                seen_msgs[(r["run_id"], r["step"])] = r["messages"]
            if model and r.get("model") != model:
                continue
            if ok_only and not r.get("ok"):
                continue
            msgs = r.get("messages") or seen_msgs.get((r["run_id"], r["step"]))
            if msgs:
                out.append((r, msgs))
    return out


# --------------------------------------------------------------------------- summary

def per_role(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["role"], []).append(-r["logprob"])
    by["ALL"] = [-r["logprob"] for r in rows]
    return {role: {"n": len(v), "mean": statistics.fmean(v), "median": statistics.median(v)} for role, v in by.items() if v}


def _q(s: str) -> str:
    return s.replace("\n", "\\n").replace("|", "\\|")


def summary_md(main: list[dict], base: list[dict], main_label: str, base_label: str, meta: dict[str, Any]) -> str:
    ms, bs = per_role(main), per_role(base)
    lines = [f"# Surprise map: {main_label} outputs scored by {meta['scorer']}", ""]
    lines.append(f"- scorer: `{meta['model_path']}` (mlx-lm {meta['mlx_lm_version']}, mlx {meta['mlx_version']}), "
                 f"teacher forcing, prompts rendered with reasoning_effort={meta['effort']}")
    lines.append(f"- {main_label}: {meta['n_main']} attempts, {len(main)} target tokens from {', '.join(meta['main_traces'])}")
    lines.append(f"- {base_label}: {meta['n_base']} attempts, {len(base)} target tokens from {', '.join(meta['base_traces'])}")
    lines.append(f"- throughput: {meta['tok_per_s']:.0f} tokens/s ({meta['tokens_seen']} prompt+target tokens in {meta['seconds']:.1f} s, forward passes only)")
    lines.append(f"- normalisation: literal think tags removed from the `thinking` field of {meta['n_leaked']} {main_label} attempts; "
                 f"whitespace around thinking/content {'stripped like parse.py does for local models' if meta['strip_ws'] else 'kept as recorded (--strip-ws to strip)'}")
    lines += ["", "## Negative log-prob per role (nats per token)", ""]
    lines.append(f"| role | n ({main_label}) | mean | median | n ({base_label}) | mean | median | diff mean | diff median |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for role in ROLES + ["ALL"]:
        m, b = ms.get(role), bs.get(role)
        if not m and not b:
            continue
        cells = [role]
        for st in (m, b):
            cells += [str(st["n"]), f"{st['mean']:.3f}", f"{st['median']:.3f}"] if st else ["0", "-", "-"]
        cells += [f"{m['mean'] - b['mean']:+.3f}", f"{m['median'] - b['median']:+.3f}"] if m and b else ["-", "-"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f"diff = {main_label} minus {base_label} (positive: the scorer finds the {main_label} output more surprising)."]
    for label, rows in ((main_label, main), (base_label, base)):
        lines += ["", f"## Top 15 most surprising tokens ({label})", ""]
        lines.append("| nll | role | token | run/step | context (±5 tokens, target token in **bold**) |")
        lines.append("|---:|---|---|---|---|")
        for i in sorted(range(len(rows)), key=lambda i: rows[i]["logprob"])[:15]:
            r = rows[i]
            same = lambda k: rows[k]["run"] == r["run"] and rows[k]["step"] == r["step"] and rows[k]["attempt"] == r["attempt"]  # noqa: E731
            ctx = [k for k in range(i - 5, i + 6) if 0 <= k < len(rows) and same(k)]
            text = "".join(f"**{_q(rows[k]['token'])}**" if k == i else _q(rows[k]["token"]) for k in ctx)
            lines.append(f"| {-r['logprob']:.2f} | {r['role']} | `{_q(r['token'])}` | {r['run']}/{r['step']} | {text} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to the MLX model directory")
    ap.add_argument("--traces", nargs="+", type=Path, required=True, help="traces with the outputs to score")
    ap.add_argument("--trace-model", default="k2-375b", help="only score attempts whose `model` field matches")
    ap.add_argument("--self-traces", nargs="*", type=Path, default=[Path("traces/smoke-3.7b.jsonl")], help="self-baseline traces (ok attempts only)")
    ap.add_argument("--self-model", default="k2-3.7b")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--reasoning-effort", default="high", help="effort used to render BOTH prompts (high opens <ifm|think>)")
    ap.add_argument("--chunk", type=int, default=512, help="positions per forward chunk (vocab is 250k; keeps logits small)")
    ap.add_argument("--strip-ws", action="store_true", help="strip whitespace around thinking/content the way parse.py does for local outputs")
    args = ap.parse_args(argv)

    import mlx.core as mx
    import mlx_lm

    scorer = Scorer(args.model, chunk=args.chunk)
    main_attempts = load_attempts(args.traces, args.trace_model, ok_only=False)
    base_attempts = load_attempts(args.self_traces, args.self_model, ok_only=True)
    print(f"scoring {len(main_attempts)} {args.trace_model} attempts + {len(base_attempts)} {args.self_model} self attempts")

    def run(attempts):
        rows = []
        for r, msgs in attempts:
            t0 = time.perf_counter()
            new = score_attempt(scorer, r, msgs, args.reasoning_effort, strip_ws=args.strip_ws)
            rows += new
            print(f"  {r['run_id']} step {r['step']} attempt {r.get('attempt', 0)}: {len(new)} target tokens, "
                  f"mean nll {-statistics.fmean(x['logprob'] for x in new):.3f}, {time.perf_counter() - t0:.1f}s, "
                  f"peak mem {mx.get_peak_memory() / 2**30:.1f} GB")
        return rows

    main_rows = run(main_attempts)
    base_rows = run(base_attempts)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("surprise_375b_by_3.7b.jsonl", main_rows), ("surprise_self_3.7b.jsonl", base_rows)):
        with (args.out / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "scorer": Path(args.model).expanduser().name, "model_path": str(args.model), "effort": args.reasoning_effort,
        "mlx_lm_version": mlx_lm.__version__, "mlx_version": mx.__version__,
        "n_main": len(main_attempts), "n_base": len(base_attempts),
        "main_traces": [str(p) for p in args.traces], "base_traces": [str(p) for p in args.self_traces],
        "tokens_seen": scorer.tokens_seen, "seconds": scorer.seconds,
        "tok_per_s": scorer.tokens_seen / scorer.seconds if scorer.seconds else 0.0,
        "n_leaked": sum(clean_thinking(r.get("thinking") or "")[1] for r, _ in main_attempts), "strip_ws": args.strip_ws,
    }
    md = summary_md(main_rows, base_rows, args.trace_model, f"{args.self_model} self", meta)
    (args.out / "surprise_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
