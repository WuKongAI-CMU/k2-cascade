"""Message compression for the projected cache: what happens to transfer when the sender ships fewer bytes.

spec strings (comma-separated, applied in order):
  int8 | int4         absmax fake quantisation per (layer, head) tensor
  layers=18-35        zero every receiver layer outside the band (the receiver still attends, to nothing)
  heads=4             keep the first k KV heads per layer, zero the rest
  rank=16             truncated SVD of each (T, d) per-head matrix
  fill=mean           untransmitted layers/heads carry the per-(layer, head) position-mean of the projected cache
                      (a one-vector summary, 1/T of the bytes) instead of zeros
  fill=filler         untransmitted layers/heads carry the receiver's own cache of a neutral filler text of the same
                      length: a structurally valid, episode-free null state (zeros dilute the softmax; see the
                      attention-sink literature), so this separates "content missing" from "attention broken"

`bytes_per_token(keys, values, spec)` reports the message size the spec implies (bf16 = 2 bytes per element).
"""
from __future__ import annotations

import torch


def _band(s: str) -> set[int]:
    out = set()
    for part in s.split("+"):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def _low_rank(x: torch.Tensor, r: int) -> torch.Tensor:
    # x: (B, H, T, d); SVD on each (T, d) matrix in float32
    u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    r = min(r, s.shape[-1])
    return (u[..., :r] * s[..., None, :r]) @ vh[..., :r, :]


def apply(keys: list[torch.Tensor], values: list[torch.Tensor], spec: str | None, null=None):
    """Return compressed (keys, values) lists and a dict describing the message size.
    `null(T) -> (keys, values)` supplies the receiver's own filler cache for fill=filler."""
    L = len(keys)
    keep_layers, keep_heads, bits, rank, fill = set(range(L)), None, 16, None, "zero"
    if spec:
        for op in spec.split(","):
            op = op.strip()
            if op == "int8":
                bits = 8
            elif op == "int4":
                bits = 4
            elif op.startswith("layers="):
                keep_layers = _band(op[7:])
            elif op.startswith("heads="):
                keep_heads = int(op[6:])
            elif op.startswith("rank="):
                rank = int(op[5:])
            elif op.startswith("fill="):
                fill = op[5:]
            elif op in ("", "none"):
                continue
            else:
                raise ValueError(op)
    ks, vs = [], []
    nk = nv = None
    if fill == "filler":
        if null is None:
            raise ValueError("fill=filler needs a null(T) provider")
        nk, nv = null(keys[0].shape[2])
    def blank(x, j, which):  # what an untransmitted layer/head carries
        if fill == "mean":
            return x.mean(dim=2, keepdim=True).expand_as(x).contiguous()
        if fill == "filler":
            return (nk if which == "k" else nv)[j].to(x.dtype)
        return torch.zeros_like(x)
    for j, (k, v) in enumerate(zip(keys, values)):
        if j not in keep_layers:
            ks.append(blank(k, j, "k")); vs.append(blank(v, j, "v")); continue
        if keep_heads is not None:
            m = torch.zeros(k.shape[1], device=k.device, dtype=k.dtype); m[:keep_heads] = 1
            mm = m[None, :, None, None]
            k, v = k * mm + blank(k, j, "k") * (1 - mm), v * mm + blank(v, j, "v") * (1 - mm)
        if rank is not None:
            k, v = _low_rank(k, rank).to(k.dtype), _low_rank(v, rank).to(v.dtype)
        if bits < 16:
            k, v = _fake_quant(k, bits), _fake_quant(v, bits)
        ks.append(k); vs.append(v)
    # message size per token: transmitted layers x heads x (K+V) x d x bits; low rank ships U*S and V^T
    H, d = keys[0].shape[1], keys[0].shape[-1]
    T = keys[0].shape[2]
    heads = H if keep_heads is None else min(keep_heads, H)
    per_layer_elems = 2 * heads * (d if rank is None else (rank + rank * d / max(T, 1)))
    bytes_per_tok = len(keep_layers) * per_layer_elems * bits / 8
    if fill == "mean":  # summaries for the rest: one vector per untransmitted (layer, head) for K and V
        bytes_per_tok += ((L - len(keep_layers)) * H + len(keep_layers) * (H - heads)) * 2 * d * 2 / max(T, 1)
    return ks, vs, {"spec": spec or "none", "layers": len(keep_layers), "heads": heads, "bits": bits, "rank": rank, "fill": fill,
                    "bytes_per_token": bytes_per_tok, "bytes_per_token_full": L * 2 * H * d * 2}


class Compressed:
    """Wraps a projector so its output passes through `apply`. `null` is a null(T) provider for fill=filler."""

    def __init__(self, projector, spec: str | None, null=None):
        self.projector, self.spec, self.info, self.null = projector, spec, None, null

    def __call__(self, bundle):
        k, v = self.projector(bundle)
        k, v, self.info = apply(k, v, self.spec, self.null)
        return k, v


def filler_null(model, tokenizer, text: str):
    """null(T): the receiver's own pre-RoPE K/V for `text` tokens (repeated) truncated to T positions, cached by T."""
    from .extract import extract
    ids = tokenizer(" " + text * 16, add_special_tokens=False)["input_ids"]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    dev = next(model.parameters()).device
    memo = {}
    def null(T):
        if T not in memo:
            b = extract(model, torch.tensor([(bos + ids)[:T]], device=dev), with_hidden=False)
            memo[T] = (b.keys, b.values)
        return memo[T]
    return null
