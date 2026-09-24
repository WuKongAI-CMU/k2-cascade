"""Milestone 4, sender side: for each QA item the small model's answer, its confidence signals, and the
pre-action hidden state at chosen layers (last prompt token), so a policy can decide when to hand off.

uv run python -m k2cascade.projector.cascade --data data/squad_600.jsonl --out analysis/cascade/sender.npz
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .extract import extract
from .qa import QAEncoder, f1
from .sender_entropy import NLI, cluster, sample_answers, semantic_entropy


@torch.no_grad()
def sender_features(model, tok, enc: QAEncoder, ex: dict, layers: list[int], k: int, max_new: int, nli, seed: int) -> dict:
    dev = next(model.parameters()).device
    ids = torch.tensor([enc.passage(ex["context"]) + enc.question(ex["question"])], device=dev)
    b = extract(model, ids, with_hidden=True)
    feats = np.concatenate([b.hidden[j][0, -1].float().cpu().numpy() for j in layers])  # hidden[0] = embeddings
    logits = model(input_ids=ids).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    p_max = float(lp.max().exp()); ent = float(-(lp.exp() * lp).sum())
    answers = sample_answers(model, tok, enc, ex["context"], ex["question"], k, max_new, seed)
    labels = cluster(ex["question"], answers, nli)
    greedy = answers[-1]
    return {"id": ex.get("id"), "greedy": greedy, "f1": f1(greedy, ex["answers"]), "p_max": p_max, "entropy": ent,
            "se": semantic_entropy(labels[:k]), "agree": sum(1 for l in labels[:k] if l == labels[-1]) / k, "feats": feats}


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100000); ap.add_argument("--layers", default="12,18,24")
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--nli", default="microsoft/deberta-large-mnli"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, tok = load_models(TrainConfig(source=a.model, target=a.model), dev)
    freeze(model)
    nli = NLI(a.nli, dev) if a.nli != "none" else None
    enc = QAEncoder(tok)
    layers = [int(x) for x in a.layers.split(",")]
    rows = []
    for i, line in enumerate(open(a.data)):
        if i >= a.n:
            break
        rows.append(sender_features(model, tok, enc, json.loads(line), layers, a.k, a.max_new, nli, seed=i))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, feats=np.stack([r["feats"] for r in rows]).astype(np.float16),
                        f1=np.array([r["f1"] for r in rows]), p_max=np.array([r["p_max"] for r in rows]),
                        entropy=np.array([r["entropy"] for r in rows]), se=np.array([r["se"] for r in rows]),
                        agree=np.array([r["agree"] for r in rows]), ids=np.array([str(r["id"]) for r in rows]),
                        greedy=np.array([r["greedy"] for r in rows]), layers=np.array(layers))
    print(json.dumps({"n": len(rows), "sender_f1": float(np.mean([r["f1"] for r in rows])),
                      "frac_wrong": float(np.mean([r["f1"] < 0.5 for r in rows]))}))


if __name__ == "__main__":
    main()
