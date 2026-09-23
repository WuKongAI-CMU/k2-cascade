"""Passage QA transfer: the sender reads a passage, the receiver reads only the question.

Arms: none / text / self / raw / project / derange (mapped cache of the previous item's passage) /
zero (all-zero cache of the same shape) / random (Gaussian cache with the projected cache's per-layer moments).
Per item we record, for each arm: mean gold log-prob per token, first-token log-prob, greedy EM/F1, and at the
first answer position P(gold), P(counter) (if the item has a counter-answer), entropy, and JSD to the text arm.

  uv run python -m k2cascade.projector.qa --projector runs/mlp/step_1500 --n 300 --out analysis/qa_mlp.json
  ... --data data/squad_variants.jsonl --variant contradicted --per_item analysis/qa_items.jsonl
"""
from __future__ import annotations

import copy
import json
import re
import string
from collections import Counter

import torch
import torch.nn as nn

from .extract import extract, make_cache

ARMS = ("none", "text", "self", "raw", "project", "derange")
CONTROL_ARMS = ("zero", "random")


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

    def question(self, q: str, receiver_ctx: str | None = None) -> list[int]:
        """Receiver-side text: optionally its own passage (the different-context setting), then the question."""
        own = f"\n\nPassage: {receiver_ctx.strip()}" if receiver_ctx else ""
        return self.tok(f"{own}\n\nQuestion: {q.strip()}\nAnswer:", add_special_tokens=False)["input_ids"]

    def answer(self, a: str) -> list[int]:
        return self.tok(" " + a.strip(), add_special_tokens=False)["input_ids"]


def _cache(model, keys, values):
    return make_cache(model, [k.to(model.dtype) for k in keys], [v.to(model.dtype) for v in values])


@torch.no_grad()
def prefix_cache(arm: str, src: nn.Module, tgt: nn.Module, projector, passage: torch.Tensor, other: torch.Tensor,
                 seed: int = 0):
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
    if arm in ("zero", "random"):
        k, v = projector(extract(src, passage, with_hidden=False))
        if arm == "zero":
            k, v = [torch.zeros_like(a) for a in k], [torch.zeros_like(a) for a in v]
        else:  # moment-matched: per-layer mean and std of the projected cache, fresh Gaussian noise
            g = torch.Generator(device=k[0].device).manual_seed(seed)
            k = [a.mean() + a.std() * torch.randn(a.shape, generator=g, device=a.device, dtype=a.dtype) for a in k]
            v = [a.mean() + a.std() * torch.randn(a.shape, generator=g, device=a.device, dtype=a.dtype) for a in v]
        return _cache(tgt, k, v), passage.shape[1]
    raise ValueError(arm)


@torch.no_grad()
def score_item(tgt: nn.Module, cache, prefix_len: int, question: torch.Tensor, answer: torch.Tensor,
               bos: list[int], max_new: int, newline_ids: set[int], counter: torch.Tensor | None = None) -> dict:
    """Teacher-forced gold scoring, first-answer-token distribution stats, then greedy generation."""
    dev = question.device
    q = torch.cat([torch.tensor([bos], device=dev), question], 1) if (cache is None and bos) else question
    c1 = copy.deepcopy(cache) if cache is not None else None
    qa = torch.cat([q, answer], 1)
    pos = torch.arange(prefix_len, prefix_len + qa.shape[1], device=dev)
    out = tgt(input_ids=qa, past_key_values=c1, position_ids=pos[None], cache_position=pos, use_cache=c1 is not None)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    na, gold = answer.shape[1], answer[0]
    span = lp[q.shape[1] - 1: q.shape[1] - 1 + na].gather(1, gold[:, None]).squeeze(1)
    first = lp[q.shape[1] - 1]                                  # distribution over the first answer token
    p = first.exp()
    res = {"logp": span.mean().item(), "logp_first": span[0].item(), "p_gold": p[gold[0]].item(),
           "entropy": float(-(p * first).sum()), "first_dist": first}
    if counter is not None:
        res["p_counter"] = p[counter[0, 0]].item()
    c2 = copy.deepcopy(cache) if cache is not None else None
    pos = torch.arange(prefix_len, prefix_len + q.shape[1], device=dev)
    out = tgt(input_ids=q, past_key_values=c2, position_ids=pos[None], cache_position=pos, use_cache=True)
    c2, cur, toks = out.past_key_values, prefix_len + q.shape[1], []
    nxt = int(out.logits[0, -1].argmax())
    for _ in range(max_new):
        if nxt in newline_ids:
            break
        toks.append(nxt)
        pp = torch.tensor([cur], device=dev)
        out = tgt(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=c2, position_ids=pp[None],
                  cache_position=pp, use_cache=True)
        c2, cur, nxt = out.past_key_values, cur + 1, int(out.logits[0, -1].argmax())
    res["tokens"] = toks
    return res


def jsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jensen-Shannon divergence (nats) between two log-prob vectors."""
    pa, pb = a.exp(), b.exp()
    m = 0.5 * (pa + pb)
    lm = m.clamp_min(1e-30).log()
    return float(0.5 * (pa * (a - lm)).sum() + 0.5 * (pb * (b - lm)).sum())


def run(src, tgt, projector, tok, examples: list[dict], arms=ARMS, max_new: int = 8, max_passage: int = 512,
        variant: str = "clean", per_item=None) -> dict:
    """examples: dicts with context (or the `variant` field), question, answers (list[str]), optional counter_answer.
    per_item: optional open file; one json line per item with every arm's measurements."""
    dev = next(tgt.parameters()).device
    enc = QAEncoder(tok, max_passage)
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    newline_ids = set(nl) | ({tok.eos_token_id} if getattr(tok, "eos_token_id", None) is not None else set())
    keys = ("logp", "logp_first", "p_gold", "p_counter", "entropy", "jsd_text", "em", "f1")
    stats = {a: {k: 0.0 for k in keys} for a in arms}
    counts = {a: {k: 0 for k in keys} for a in arms}
    prev = None
    for i, ex in enumerate(examples):
        ctx = ex.get(variant, ex.get("context")) if variant != "clean" else ex.get("clean", ex["context"])
        pa = torch.tensor([enc.passage(ctx)], device=dev)
        other = prev if prev is not None else pa
        prev = pa
        q = torch.tensor([enc.question(ex["question"], ex.get("context_receiver"))], device=dev)
        ans = torch.tensor([enc.answer(ex["answers"][0])], device=dev)
        counter = torch.tensor([enc.answer(ex["counter_answer"])], device=dev) if ex.get("counter_answer") else None
        row, text_dist = {"i": i, "id": ex.get("id"), "variant": variant}, None
        for a in arms:
            if a in ("project", "derange", "zero", "random") and projector is None:
                continue
            cache, plen = prefix_cache(a, src, tgt, projector, pa, other, seed=i)
            r = score_item(tgt, cache, plen, q, ans, enc.bos, max_new, newline_ids, counter)
            pred = tok.decode(r.pop("tokens"))
            dist = r.pop("first_dist")
            if a == "text":
                text_dist = dist
            r["jsd_text"] = jsd(text_dist, dist) if text_dist is not None else None
            r["em"], r["f1"], r["pred"] = em(pred, ex["answers"]), f1(pred, ex["answers"]), pred
            row[a] = r
            for k in keys:
                if r.get(k) is not None:
                    stats[a][k] += r[k]; counts[a][k] += 1
        if per_item is not None:
            per_item.write(json.dumps(row) + "\n")
    out = {a: {k: stats[a][k] / counts[a][k] for k in keys if counts[a][k]} for a in arms if counts[a]["f1"]}
    if "project" in out and "derange" in out:
        out["content_f1"] = out["project"]["f1"] - out["derange"]["f1"]
    out.update(n=len(examples), max_new=max_new, variant=variant)
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
    ap.add_argument("--variant", default="clean", choices=("clean", "contradicted", "removed"))
    ap.add_argument("--per_item", default=None, help="jsonl path for per-item measurements")
    ap.add_argument("--compress", default=None, help="message compression spec, see compress.py")
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
    if a.compress and proj is not None:
        from .compress import Compressed
        proj = Compressed(proj, a.compress)
    examples = [json.loads(l) for l in open(a.data)][: a.n] if a.data else load_squad(a.n, a.seed)
    pi = open(a.per_item, "w") if a.per_item else None
    res = run(src, tgt, proj, tok, examples, tuple(a.arms.split(",")), a.max_new, variant=a.variant, per_item=pi)
    if pi:
        pi.close()
    res.update(source=a.source, target=a.target, projector=a.projector, seed=a.seed, compress=getattr(proj, "info", None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
