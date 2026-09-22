"""Passage QA transfer: the sender reads a SQuAD passage, the receiver reads only the question.

Arms (same as Noma): none / text / self / raw / project / derange, where derange hands the receiver the mapped
cache of the *previous* example's passage. We report the mean per-token log-prob of the gold answer and greedy
exact-match / token-F1 over up to `max_new` generated tokens (stopped at a newline).

uv run python -m k2cascade.projector.qa --projector runs/mlp/step_1500 --n 300 --out analysis/qa_mlp.json
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter

import torch
import torch.nn as nn

from .extract import extract, make_cache

ARMS = ("none", "text", "self", "raw", "project", "derange")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = normalize(pred).split()
    for g in golds:
        g = normalize(g).split()
        common = Counter(p) & Counter(g)
        n = sum(common.values())
        if n == 0:
            continue
        pr, rc = n / len(p), n / len(g)
        best = max(best, 2 * pr * rc / (pr + rc))
    return best


def em(pred: str, golds: list[str]) -> float:
    return float(any(normalize(pred) == normalize(g) for g in golds))


class QAEncoder:
    def __init__(self, tokenizer, max_passage: int = 512):
        self.tok, self.max_passage = tokenizer, max_passage
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    def passage(self, text: str) -> list[int]:
        ids = self.tok("Passage: " + text.strip(), add_special_tokens=False)["input_ids"][: self.max_passage]
        return self.bos + ids

    def question(self, q: str) -> list[int]:
        return self.tok(f"\n\nQuestion: {q.strip()}\nAnswer:", add_special_tokens=False)["input_ids"]

    def answer(self, a: str) -> list[int]:
        return self.tok(" " + a.strip(), add_special_tokens=False)["input_ids"]


def _cache(model, keys, values):
    return make_cache(model, [k.to(model.dtype) for k in keys], [v.to(model.dtype) for v in values])


@torch.no_grad()
def prefix_cache(arm: str, src: nn.Module, tgt: nn.Module, projector, passage: torch.Tensor, other: torch.Tensor):
    """Cache the receiver starts from, and the number of positions it holds. `none` gives (None, 0)."""
    if arm == "none":
        return None, 0
    if arm == "text":
        return tgt(input_ids=passage, use_cache=True).past_key_values, passage.shape[1]
    if arm == "self":
        b = extract(tgt, passage, with_hidden=False)
        return _cache(tgt, b.keys, b.values), passage.shape[1]
    if arm == "raw":
        b = extract(src, passage, with_hidden=False)
        return _cache(tgt, b.keys, b.values), passage.shape[1]
    if arm in ("project", "derange"):
        p = passage if arm == "project" else other
        k, v = projector(extract(src, p, with_hidden=False))
        return _cache(tgt, k, v), p.shape[1]
    raise ValueError(arm)


@torch.no_grad()
def score_and_generate(tgt: nn.Module, cache, prefix_len: int, question: torch.Tensor, answer: torch.Tensor,
                       bos: list[int], max_new: int, newline_ids: set[int]) -> tuple[float, list[int]]:
    """Mean log-prob per gold-answer token (teacher forced), then greedy generation from the question."""
    dev = question.device
    q = torch.cat([torch.tensor([bos], device=dev), question], 1) if (cache is None and bos) else question
    # teacher-forced scoring: run question + answer in one pass on a copy of the cache
    import copy
    c1 = copy.deepcopy(cache) if cache is not None else None
    qa = torch.cat([q, answer], 1)
    pos = torch.arange(prefix_len, prefix_len + qa.shape[1], device=dev)
    out = tgt(input_ids=qa, past_key_values=c1, position_ids=pos[None], cache_position=pos, use_cache=c1 is not None)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    na = answer.shape[1]
    gold = answer[0]
    score = lp[q.shape[1] - 1: q.shape[1] - 1 + na].gather(1, gold[:, None]).mean().item()
    # greedy generation
    c2 = copy.deepcopy(cache) if cache is not None else None
    pos = torch.arange(prefix_len, prefix_len + q.shape[1], device=dev)
    out = tgt(input_ids=q, past_key_values=c2, position_ids=pos[None], cache_position=pos, use_cache=True)
    c2 = out.past_key_values
    cur = prefix_len + q.shape[1]
    toks = []
    nxt = int(out.logits[0, -1].argmax())
    for _ in range(max_new):
        if nxt in newline_ids:
            break
        toks.append(nxt)
        p = torch.tensor([cur], device=dev)
        out = tgt(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=c2, position_ids=p[None],
                  cache_position=p, use_cache=True)
        c2 = out.past_key_values
        cur += 1
        nxt = int(out.logits[0, -1].argmax())
    return score, toks


def run(src, tgt, projector, tok, examples: list[dict], arms=ARMS, max_new: int = 8, max_passage: int = 512) -> dict:
    """examples: dicts with context, question, answers (list[str])."""
    dev = next(tgt.parameters()).device
    enc = QAEncoder(tok, max_passage)
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    newline_ids = set(nl) | ({tok.eos_token_id} if getattr(tok, "eos_token_id", None) is not None else set())
    stats = {a: {"logp": 0.0, "em": 0.0, "f1": 0.0} for a in arms}
    prev = None
    for ex in examples:
        pa = torch.tensor([enc.passage(ex["context"])], device=dev)
        other = prev if prev is not None else pa
        prev = pa
        q = torch.tensor([enc.question(ex["question"])], device=dev)
        ans = torch.tensor([enc.answer(ex["answers"][0])], device=dev)
        for a in arms:
            if a in ("project", "derange") and projector is None:
                continue
            cache, plen = prefix_cache(a, src, tgt, projector, pa, other)
            score, toks = score_and_generate(tgt, cache, plen, q, ans, enc.bos, max_new, newline_ids)
            pred = tok.decode(toks)
            s = stats[a]
            s["logp"] += score; s["em"] += em(pred, ex["answers"]); s["f1"] += f1(pred, ex["answers"])
    n = len(examples)
    out = {a: {k: v / n for k, v in s.items()} for a, s in stats.items()
           if not (a in ("project", "derange") and projector is None)}
    if "project" in out and "derange" in out:
        out["content_f1"] = out["project"]["f1"] - out["derange"]["f1"]
    out.update(n=n, max_new=max_new)
    return out


def load_squad(n: int, seed: int = 0, split: str = "validation") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split=split).shuffle(seed=seed).select(range(n))
    return [{"context": r["context"], "question": r["question"], "answers": list(r["answers"]["text"])} for r in ds]


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", default="none"); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="analysis/qa.json")
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--data", default=None, help="jsonl with context/question/answers; default: SQuAD validation")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    if a.projector == "none":
        proj = None
    elif (Path(a.projector) / "mlp.safetensors").exists():
        from .mlp import MLPProjector
        proj = MLPProjector.load(a.projector, dev)
    else:
        proj = RidgeProjector.load(a.projector, dev)
    if a.data:
        examples = [json.loads(l) for l in open(a.data)][: a.n]
    else:
        examples = load_squad(a.n, a.seed)
    res = run(src, tgt, proj, tok, examples, tuple(a.arms.split(",")), a.max_new)
    res.update(source=a.source, target=a.target, projector=a.projector, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
