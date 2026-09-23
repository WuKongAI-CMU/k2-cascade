"""Train the MLP projector: target next-token CE on the continuation with the projected prefix cache.

  uv run python -m k2cascade.projector.train --source IFM/K2-Horizon-3.7B --target IFM/K2-Horizon-7B \
      --data data/fineweb_1024.jsonl --ridge runs/ridge --out runs/mlp --steps 300

Both models frozen (bf16 on CUDA); only the projector trains (fp32 params, bf16 autocast).
`train()` is the testable core; `main()` handles loading and CLI. Checkpoints: <out>/last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from .eval import continuation_loss, evaluate, projected_cache
from .mlp import MLPProjector
from .ridge import RidgeProjector, fit_ridge


@dataclass
class TrainConfig:
    source: str = "IFM/K2-Horizon-3.7B"
    target: str = "IFM/K2-Horizon-7B"
    data: str = "data/fineweb_1024.jsonl"
    out: str = "runs/mlp"
    ridge: str = ""  # dir with ridge.safetensors; if empty, fit ridge on the first `ridge_seqs` sequences
    ridge_seqs: int = 64
    ridge_lam: float = 1e-2
    per_head: bool = True
    steps: int = 300
    batch: int = 2
    accum: int = 8
    seq_len: int = 1024
    prefix_len: int = 512
    lr: float = 1e-4
    wd: float = 0.01
    warmup: float = 0.1
    bottleneck: int = 128  # LCF uses 128-256; per-head inputs are only 256-d so 128 is enough
    freeze_ridge: bool = False
    grad_ckpt: bool = False  # HF checkpointing drops past_key_values in train mode, so the projected cache would be ignored
    bf16: bool = True
    attn: str = "sdpa"  # or flash_attention_2
    save_every: int = 50
    log_every: int = 1
    eval_seqs: int = 32
    resume: bool = False
    wandb: bool = False
    seed: int = 0
    derange_source: bool = False  # control: sender reads the previous micro-batch's prefix (no content to carry)


def parse_args(argv=None) -> TrainConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(TrainConfig):
        if f.type == "bool" or f.type is bool:
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(ap.parse_args(argv)))


def load_models(cfg: TrainConfig, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if cfg.bf16 and device.type == "cuda" else torch.float32
    kw = dict(trust_remote_code=True, dtype=dtype, attn_implementation=cfg.attn)
    src = AutoModelForCausalLM.from_pretrained(cfg.source, **kw).to(device)
    tgt = AutoModelForCausalLM.from_pretrained(cfg.target, **kw).to(device)
    tok = AutoTokenizer.from_pretrained(cfg.target, trust_remote_code=True)
    src_tok = AutoTokenizer.from_pretrained(cfg.source, trust_remote_code=True)
    probe = "The quick brown fox, 1932."
    if src_tok(probe, add_special_tokens=False)["input_ids"] != tok(probe, add_special_tokens=False)["input_ids"]:
        from .align import attach_aligner  # different vocabularies: the sender reads its own tokens
        attach_aligner(src, src_tok, tok)
        print(f"cross-tokenizer sender: {cfg.source} aligned to {cfg.target} by character offsets")
    return src, tgt, tok


def freeze(model: nn.Module, grad_ckpt: bool = False) -> nn.Module:
    model.requires_grad_(False).eval()
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF checkpointing only fires in train mode; dropout is 0 for K2 Horizon
    return model


def jsonl_sequences(path: str, tok, seq_len: int) -> list[list[int]]:
    """Each line: {"ids": [...]} or {"text": "..."}; keeps sequences with >= seq_len tokens, truncated."""
    out = []
    for line in open(path):
        row = json.loads(line)
        ids = row.get("ids") or tok(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
    return out


def batches(seqs: list[list[int]], batch: int, device, shuffle: bool = True, seed: int = 0):
    """Infinite generator of (B, T) LongTensors."""
    rng = random.Random(seed)
    while True:
        order = list(range(len(seqs)))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order) - batch + 1, batch):
            yield torch.tensor([seqs[j] for j in order[i:i + batch]], device=device)


def lr_at(step: int, cfg: TrainConfig) -> float:
    warm = max(1, int(cfg.warmup * cfg.steps))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, cfg.steps - warm)))


def save_checkpoint(path: Path, proj: MLPProjector, opt, step: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"projector": proj.state_dict(), "meta": proj.meta, "optim": opt.state_dict(),
                "step": step, "cfg": asdict(cfg)}, path)


def load_checkpoint(path: Path, proj: MLPProjector, opt) -> int:
    ck = torch.load(path, map_location=next(proj.parameters()).device, weights_only=False)
    proj.load_state_dict(ck["projector"]), opt.load_state_dict(ck["optim"])
    return ck["step"]


def make_optimizer(proj: MLPProjector, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; weight decay on matrices only (not on biases / gates)."""
    params = [p for p in proj.parameters() if p.requires_grad]
    return torch.optim.AdamW([{"params": [p for p in params if p.dim() > 1], "weight_decay": cfg.wd},
                              {"params": [p for p in params if p.dim() <= 1], "weight_decay": 0.0}], lr=cfg.lr)


def train(cfg: TrainConfig, src: nn.Module, tgt: nn.Module, proj: MLPProjector, data, log=print) -> list[float]:
    """Run up to cfg.steps optimizer steps over `data` (generator of (B, T) ids). Returns per-step losses."""
    out = Path(cfg.out)
    params = [p for p in proj.parameters() if p.requires_grad]
    opt = make_optimizer(proj, cfg)
    step = load_checkpoint(out / "last.pt", proj, opt) if cfg.resume and (out / "last.pt").exists() else 0
    run = None
    if cfg.wandb:
        import wandb
        run = wandb.init(project="k2-kv-projector", config=asdict(cfg), resume="allow")
    dev = next(tgt.parameters()).device
    use_bf16 = cfg.bf16 and dev.type == "cuda"
    losses, t0, prev_prefix = [], time.time(), None
    while step < cfg.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg.accum):
            ids = next(data)
            prefix, cont = ids[:, :cfg.prefix_len], ids[:, cfg.prefix_len:]
            src_prefix = prev_prefix if (cfg.derange_source and prev_prefix is not None) else None
            prev_prefix = prefix
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                cache = projected_cache(src, tgt, proj, prefix, grad=True, source_prefix=src_prefix)
                loss = continuation_loss(tgt, cont, cache, cfg.prefix_len) / cfg.accum
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        losses.append(total)
        if step % cfg.log_every == 0:
            msg = {"step": step, "loss": round(total, 4), "lr": lr_at(step - 1, cfg),
                   "gate_mean": round(proj.gates().mean().item(), 4), "s_per_step": round((time.time() - t0) / len(losses), 2)}
            log(json.dumps(msg))
            if run:
                run.log(msg, step=step)
        if step % cfg.save_every == 0 or step == cfg.steps:
            save_checkpoint(out / "last.pt", proj, opt, step, cfg)
            proj.save(out / f"step_{step}")
    if run:
        run.finish()
    return losses


def main(argv=None) -> None:
    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(cfg, dev)
    freeze(src), freeze(tgt, cfg.grad_ckpt)
    seqs = jsonl_sequences(cfg.data, tok, cfg.seq_len)
    held, seqs = seqs[:cfg.eval_seqs], seqs[cfg.eval_seqs:]
    print(f"{len(seqs)} train sequences, {len(held)} held-out, seq_len {cfg.seq_len}, prefix {cfg.prefix_len}")
    if cfg.ridge:
        ridge = RidgeProjector.load(cfg.ridge, dev)
    else:
        fit_data = batches(seqs[:cfg.ridge_seqs], cfg.batch, dev, shuffle=False)
        ridge = fit_ridge(src, tgt, [next(fit_data) for _ in range(cfg.ridge_seqs // cfg.batch)],
                          lam=cfg.ridge_lam, per_head=cfg.per_head, log=print)
        ridge.save(Path(cfg.out) / "ridge")
    proj = MLPProjector.from_ridge(ridge, cfg.bottleneck, freeze_ridge=cfg.freeze_ridge)
    print(f"projector params: {sum(p.numel() for p in proj.parameters() if p.requires_grad) / 1e6:.1f}M trainable")
    held_batches = [torch.tensor(held[i:i + cfg.batch], device=dev) for i in range(0, len(held) - cfg.batch + 1, cfg.batch)]
    print("ridge eval:", json.dumps(evaluate(src, tgt, ridge, held_batches, cfg.prefix_len)))
    train(cfg, src, tgt, proj, batches(seqs, cfg.batch, dev, seed=cfg.seed))
    print("mlp eval:", json.dumps(evaluate(src, tgt, proj, held_batches, cfg.prefix_len)))


if __name__ == "__main__":
    main()
