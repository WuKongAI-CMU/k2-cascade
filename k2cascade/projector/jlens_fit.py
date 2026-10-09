"""Fit a Jacobian lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in Language
Models"; reference code github.com/anthropics/jacobian-lens) on a K2-Horizon model.

The lens is fitted in parts of `--part` prompts each, so a part that finishes is usable even if the VM dies;
`merge` combines parts weighted by prompt count. Prompts are the projector's FineWeb-Edu token ids.

uv run python -m k2cascade.projector.jlens_fit --model $TGT --data data/fineweb_1024.jsonl --n 96 --out runs/jlens_7b
uv run python -m k2cascade.projector.jlens_fit --merge runs/jlens_7b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from k2cascade.projector.extract import decoder_layers


class K2LensModel:
    """`jlens.protocol.LensModel` over a K2-Horizon model. `encode` takes space-separated token ids rather than
    text, so the fit sees exactly the token sequences the projector was trained on."""

    def __init__(self, hf, tokenizer=None):
        self.hf, self.tokenizer = hf, tokenizer
        hf.eval()
        for p in hf.parameters():
            p.requires_grad_(False)
        self.layers = decoder_layers(hf)
        self.n_layers, self.d_model = hf.config.num_hidden_layers, hf.config.hidden_size

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        ids = [int(x) for x in text.split()][:max_length]
        return torch.tensor([ids], device=self.hf.device)

    def forward(self, input_ids: torch.Tensor):
        return self.hf.model(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        w = self.hf.lm_head.weight
        return self.hf.lm_head(self.hf.model.norm(residual.to(w.device, w.dtype)))


def load_prompts(path: str, n: int, seq: int) -> list[str]:
    out = []
    for line in open(path):
        ids = json.loads(line)["ids"]
        if len(ids) >= seq:
            out.append(" ".join(map(str, ids[:seq])))
        if len(out) == n:
            break
    return out


def merge(folder: str):
    import jlens

    parts = [jlens.JacobianLens.load(p) for p in sorted(glob.glob(f"{folder}/part_*.pt"))]
    lens = jlens.JacobianLens.merge(parts)
    lens.save(f"{folder}/lens.pt")
    return lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--data", default="data/fineweb_1024.jsonl")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--part", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dim_batch", type=int, default=16)
    ap.add_argument("--out")
    ap.add_argument("--merge")
    a = ap.parse_args()
    if a.merge:
        print(merge(a.merge))
        return
    import jlens
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    model = K2LensModel(hf)
    prompts = load_prompts(a.data, a.n, a.seq)
    os.makedirs(a.out, exist_ok=True)
    for i in range(0, len(prompts), a.part):
        path = f"{a.out}/part_{i // a.part:03d}.pt"
        if os.path.exists(path):
            continue
        t = time.time()
        lens = jlens.fit(model, prompts[i : i + a.part], dim_batch=a.dim_batch, max_seq_len=a.seq)
        lens.save(path)
        print(f"part {i // a.part}: {len(prompts[i:i + a.part])} prompts in {time.time() - t:.0f}s", flush=True)
    print(merge(a.out))


if __name__ == "__main__":
    main()
