"""Per-layer diagnostics for a fitted projector on held-out text: how well does it reconstruct the receiver's
pre-RoPE K and V (R^2, cosine), and how close are the receiver's hidden states when it reads the mapped cache
instead of its own? Separates 'the map did not fit' from 'the map fits but the receiver cannot use it'.

uv run python -m k2cascade.projector.diag --projector runs/ridge --data data/fineweb_1024.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch

from .extract import extract, make_cache


@torch.no_grad()
def layer_fit(src, tgt, proj, batches):
    L = tgt.config.num_hidden_layers
    sse_k = torch.zeros(L); sst_k = torch.zeros(L); sse_v = torch.zeros(L); sst_v = torch.zeros(L)
    cos_k = torch.zeros(L); cos_v = torch.zeros(L); n = 0
    for x in batches:
        s, t = extract(src, x, with_hidden=False), extract(tgt, x, with_hidden=False)
        pk, pv = proj(s)
        for j in range(L):
            for pred, true, sse, sst, cos in ((pk[j], t.keys[j], sse_k, sst_k, cos_k), (pv[j], t.values[j], sse_v, sst_v, cos_v)):
                p, y = pred.float(), true.float()
                sse[j] += ((p - y) ** 2).sum().cpu()
                sst[j] += ((y - y.mean(dim=(0, 2), keepdim=True)) ** 2).sum().cpu()
                cos[j] += torch.nn.functional.cosine_similarity(p, y, dim=-1).mean().cpu()
        n += 1
    return {"r2_k": (1 - sse_k / sst_k).tolist(), "r2_v": (1 - sse_v / sst_v).tolist(),
            "cos_k": (cos_k / n).tolist(), "cos_v": (cos_v / n).tolist()}


@torch.no_grad()
def hidden_cosine(src, tgt, proj, batches, prefix_len: int):
    """Cosine of the receiver's last hidden state on continuation tokens: mapped cache vs its own prefill."""
    vals = []
    dt = next(tgt.parameters()).dtype
    for x in batches:
        pre, cont = x[:, :prefix_len], x[:, prefix_len:]
        pos = torch.arange(prefix_len, x.shape[1], device=x.device)
        own = tgt(input_ids=pre, use_cache=True).past_key_values
        k, v = proj(extract(src, pre, with_hidden=False))
        mapped = make_cache(tgt, [a.to(dt) for a in k], [b.to(dt) for b in v])
        kw = dict(input_ids=cont, position_ids=pos[None].expand(x.shape[0], -1), cache_position=pos, use_cache=True)
        h_own = tgt.model(past_key_values=own, **kw).last_hidden_state.float()
        h_map = tgt.model(past_key_values=mapped, **kw).last_hidden_state.float()
        vals.append(torch.nn.functional.cosine_similarity(h_own, h_map, dim=-1).mean().item())
    return sum(vals) / len(vals)


def main(argv=None) -> None:
    from pathlib import Path
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, jsonl_sequences, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--eval_seqs", type=int, default=16); ap.add_argument("--prefix_len", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    proj = RidgeProjector.load(a.projector, dev)
    held = jsonl_sequences(a.data, tok, 1024)[:a.eval_seqs]
    batches = [torch.tensor([r], device=dev) for r in held]
    res = layer_fit(src, tgt, proj, [b[:, :a.prefix_len] for b in batches])
    res["hidden_cos_mapped_vs_own"] = hidden_cosine(src, tgt, proj, batches, a.prefix_len)
    summ = {k: round(sum(v) / len(v), 3) for k, v in res.items() if isinstance(v, list)}
    summ["hidden_cos_mapped_vs_own"] = round(res["hidden_cos_mapped_vs_own"], 3)
    res["summary"] = summ
    out = a.out or f"analysis/diag_{Path(a.projector).name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(summ))
    for key in ("r2_k", "r2_v"):
        print(key, " ".join(f"{x:.2f}" for x in res[key]))


if __name__ == "__main__":
    main()
