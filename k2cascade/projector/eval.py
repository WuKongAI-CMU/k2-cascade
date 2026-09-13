"""Retention evaluation: how much of the (no-prefix -> oracle-prefix) loss gap a projected cache recovers.

Three arms, same continuation tokens `cont` (B, T_c); loss = mean CE of cont[1:] given cont[:-1]:
  none    : target reads only `cont` (no prefix)
  oracle  : target reads its own KV cache of `prefix` (own prefill) then `cont`
  project : target reads the projector's cache built from the *source* model's KV of `prefix`
retention = (none - project) / (none - oracle)   (arXiv 2608.03893: 1.0 = oracle, 0 = no help).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .extract import extract, make_cache


def _chunked_ce(lm_head: nn.Module, hidden: torch.Tensor, targets: torch.Tensor, chunk: int) -> torch.Tensor:
    """Per-token CE (N,) computing logits `chunk` tokens at a time (vocab 250k: never hold full logits)."""
    def piece(h, y):
        return F.cross_entropy(lm_head(h).float(), y, reduction="none")
    out = []
    for i in range(0, hidden.shape[0], chunk):
        h, y = hidden[i:i + chunk], targets[i:i + chunk]
        out.append(checkpoint(piece, h, y, use_reentrant=False) if torch.is_grad_enabled() else piece(h, y))
    return torch.cat(out)


def continuation_loss(model: nn.Module, cont: torch.Tensor, cache=None, prefix_len: int = 0,
                      reduce: bool = True, chunk: int = 1024) -> torch.Tensor:
    """CE of cont[:, 1:] predicted from cont[:, :-1] with `cache` (a fresh DynamicCache holding
    `prefix_len` positions) or none. Positions start at prefix_len so RoPE matches the cache.
    The LM head is applied only to continuation positions, in chunks."""
    b, t = cont.shape
    pos = torch.arange(prefix_len, prefix_len + t, device=cont.device)
    base, head = getattr(model, "model", None), getattr(model, "lm_head", None)
    if base is None or head is None:  # generic fallback: full logits
        logits = model(input_ids=cont, past_key_values=cache, position_ids=pos[None].expand(b, -1),
                       cache_position=pos, use_cache=cache is not None).logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), cont[:, 1:].reshape(-1), reduction="none")
    else:
        h = base(input_ids=cont, past_key_values=cache, position_ids=pos[None].expand(b, -1),
                 cache_position=pos, use_cache=cache is not None).last_hidden_state[:, :-1]
        loss = _chunked_ce(head, h.reshape(-1, h.shape[-1]), cont[:, 1:].reshape(-1), chunk)
    return loss.mean() if reduce else loss.view(b, t - 1)


@torch.no_grad()
def oracle_cache(model: nn.Module, prefix: torch.Tensor):
    return model(input_ids=prefix, use_cache=True).past_key_values


def projected_cache(src_model: nn.Module, tgt_model: nn.Module, projector: nn.Module, prefix: torch.Tensor,
                    grad: bool = False):
    """Source KV of `prefix` (no grad) -> projector -> DynamicCache for the target (grad if requested)."""
    with torch.no_grad():
        s = extract(src_model, prefix, with_hidden=False)
    with torch.set_grad_enabled(grad):
        keys, values = projector(s)
    return make_cache(tgt_model, keys, values)


def retention(none: float, oracle: float, project: float) -> float:
    gap = none - oracle
    return float((none - project) / gap) if abs(gap) > 1e-8 else float("nan")


@torch.no_grad()
def evaluate(src_model: nn.Module, tgt_model: nn.Module, projector: nn.Module, batches, prefix_len: int,
             with_source_oracle: bool = False) -> dict[str, float]:
    """batches: iterable of input_ids (B, T), T > prefix_len. Mean losses per arm + retention.
    with_source_oracle adds the source model reading its own prefix (what the small model alone gets)."""
    sums: dict[str, float] = {"none": 0.0, "oracle": 0.0, "project": 0.0}
    n = 0
    for ids in batches:
        prefix, cont = ids[:, :prefix_len], ids[:, prefix_len:]
        sums["none"] += continuation_loss(tgt_model, cont).item()
        sums["oracle"] += continuation_loss(tgt_model, cont, oracle_cache(tgt_model, prefix), prefix_len).item()
        sums["project"] += continuation_loss(
            tgt_model, cont, projected_cache(src_model, tgt_model, projector, prefix), prefix_len).item()
        if with_source_oracle:
            sums["source_oracle"] = sums.get("source_oracle", 0.0) + continuation_loss(
                src_model, cont, oracle_cache(src_model, prefix), prefix_len).item()
        n += 1
    res = {k: v / n for k, v in sums.items()}
    res["retention"] = retention(res["none"], res["oracle"], res["project"])
    res["batches"] = n
    return res


@torch.no_grad()
def messages_continuation_loss(tokenizer, src_model, tgt_model, projector, messages: list[dict],
                               continuation: str) -> dict[str, float]:
    """Hook for our own agent traces: chat-templated `messages` are the prefix, `continuation` is the
    assistant text whose loss we measure under none / oracle / project. Same dict as `evaluate`."""
    dev = next(tgt_model.parameters()).device
    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")["input_ids"].to(dev)
    cont = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    return evaluate(src_model, tgt_model, projector, [torch.cat([prefix, cont], 1)], prefix.shape[1])


def main(argv=None) -> None:
    """uv run python -m k2cascade.projector.eval --projector runs/mlp/step_300 --data data/fineweb_1024.jsonl"""
    import argparse, json
    from .mlp import MLPProjector
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, jsonl_sequences, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", required=True, help="dir with ridge.safetensors or mlp.safetensors")
    ap.add_argument("--data", required=True); ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--prefix_len", type=int, default=512)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    from pathlib import Path
    proj = (MLPProjector.load if (Path(a.projector) / "mlp.safetensors").exists() else RidgeProjector.load)(a.projector, dev)
    held = jsonl_sequences(a.data, tok, a.seq_len)[:a.eval_seqs]
    batches = [torch.tensor(held[i:i + a.batch], device=dev) for i in range(0, len(held) - a.batch + 1, a.batch)]
    print(json.dumps(evaluate(src, tgt, proj, batches, a.prefix_len, with_source_oracle=True)))


if __name__ == "__main__":
    main()
