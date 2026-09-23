"""Sender-side uncertainty: semantic entropy of the small model's answers (Farquhar et al. 2024).

For each item and variant, the sender reads the passage and question, samples `k` short answers at T=1
(plus one greedy), clusters them by bidirectional entailment with an NLI model (falls back to normalised string
equality when the NLI model is unavailable), and reports the discrete semantic entropy over clusters.

uv run python -m k2cascade.projector.sender_entropy --data data/squad_variants.jsonl --variant clean --out analysis/se_clean.jsonl
"""
from __future__ import annotations

import json
import math
from collections import Counter

import torch

from .qa import QAEncoder, normalize


class NLI:
    """Bidirectional entailment with a DeBERTa MNLI model; `same(q, a, b)` is True when each entails the other."""

    def __init__(self, name: str = "microsoft/deberta-large-mnli", device=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device or "cpu").eval()
        self.dev = device or "cpu"
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.ent = labels.get("entailment", 2)

    @torch.no_grad()
    def entails(self, premise: str, hypothesis: str) -> bool:
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True).to(self.dev)
        return int(self.model(**x).logits.argmax(-1)) == self.ent

    def same(self, q: str, a: str, b: str) -> bool:
        pa, pb = f"{q} {a}", f"{q} {b}"
        return self.entails(pa, pb) and self.entails(pb, pa)


def cluster(question: str, answers: list[str], nli=None) -> list[int]:
    """Greedy clustering: each answer joins the first cluster whose representative it is equivalent to."""
    reps, labels = [], []
    for a in answers:
        for j, r in enumerate(reps):
            if (normalize(a) == normalize(r)) or (nli is not None and nli.same(question, a, r)):
                labels.append(j); break
        else:
            reps.append(a); labels.append(len(reps) - 1)
    return labels


def semantic_entropy(labels: list[int]) -> float:
    n = len(labels)
    return -sum(c / n * math.log(c / n) for c in Counter(labels).values())


@torch.no_grad()
def sample_answers(model, tok, enc: QAEncoder, context: str, question: str, k: int, max_new: int, seed: int) -> list[str]:
    """k sampled answers followed by one greedy answer (last element)."""
    dev = next(model.parameters()).device
    ids = torch.tensor([enc.passage(context) + enc.question(question)], device=dev)
    nl = set(tok("\n", add_special_tokens=False)["input_ids"])
    torch.manual_seed(seed)
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    gen = model.generate(ids, do_sample=True, temperature=1.0, top_p=0.9, top_k=50, max_new_tokens=max_new,
                         num_return_sequences=k, pad_token_id=pad)
    greedy = model.generate(ids, do_sample=False, max_new_tokens=max_new, pad_token_id=pad)
    outs = []
    for seq in list(gen) + list(greedy):
        new = seq[ids.shape[1]:].tolist()
        cut = next((i for i, t in enumerate(new) if t in nl), len(new))
        outs.append(tok.decode(new[:cut]).strip())
    return outs


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="clean"); ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--nli", default="microsoft/deberta-large-mnli", help="'none' for string matching only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, tok = load_models(TrainConfig(source=a.model, target=a.model), dev)
    freeze(model)
    nli = NLI(a.nli, dev) if a.nli != "none" else None
    enc = QAEncoder(tok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i, line in enumerate(open(a.data)):
            if i >= a.n:
                break
            ex = json.loads(line)
            ctx = ex.get(a.variant, ex.get("context"))
            ans = sample_answers(model, tok, enc, ctx, ex["question"], a.k, a.max_new, seed=i)
            labels = cluster(ex["question"], ans, nli)
            f.write(json.dumps({"i": i, "id": ex.get("id"), "variant": a.variant, "answers": ans, "labels": labels,
                                "se": semantic_entropy(labels[: a.k]), "n_clusters": len(set(labels)),
                                "greedy": ans[-1]}) + "\n")


if __name__ == "__main__":
    main()
