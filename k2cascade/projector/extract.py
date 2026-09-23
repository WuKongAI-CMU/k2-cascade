"""Per-layer KV / hidden-state extraction for K2 Horizon (HF remote code) via forward hooks.

What is captured: PRE-RoPE keys and values. In `modeling_k2_horizon.K2HorizonAttention.forward`
the keys are `k_proj(h)` (no q/k norm: `query_key_norm` is false for 3.7B and 7B), then RoPE is
applied, then the cache is updated. Hooking the `k_proj` / `v_proj` nn.Linear outputs therefore gives
the pre-RoPE tensors without patching anything. If a model ever sets `query_key_norm=True`, hook
`k_norm` instead (see `_kv_hook_points`). Injection into the target re-applies RoPE at the prefix
positions (`make_cache`), because the HF DynamicCache stores post-RoPE keys.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.cache_utils import DynamicCache


def attention_modules(model: nn.Module) -> list[nn.Module]:
    """Expected hook points: `model.model.layers[i].self_attn` (K2HorizonAttention). Fallback: any
    submodule that owns both `k_proj` and `v_proj`, in registration order."""
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is not None and all(hasattr(l, "self_attn") for l in layers):
        return [l.self_attn for l in layers]
    return [m for m in model.modules() if hasattr(m, "k_proj") and hasattr(m, "v_proj")]


def _kv_hook_points(attn: nn.Module) -> tuple[nn.Module, nn.Module]:
    if getattr(getattr(attn, "config", None), "query_key_norm", False) and hasattr(attn, "k_norm"):
        return attn.k_norm, attn.v_proj
    return attn.k_proj, attn.v_proj


def kv_geometry(config) -> tuple[int, int]:
    """(num_key_value_heads, head_dim)."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return config.num_key_value_heads, head_dim


def rope_theta(config) -> float:
    rp = getattr(config, "rope_parameters", None) or {}
    return float(rp.get("rope_theta", getattr(config, "rope_theta", 1e7)))


@dataclass
class KVBundle:
    keys: list[torch.Tensor]  # per layer, (B, H_kv, T, D), pre-RoPE
    values: list[torch.Tensor]  # per layer, (B, H_kv, T, D)
    hidden: list[torch.Tensor] = field(default_factory=list)  # [embeddings, out of layer 0..L-1], pre final norm

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def flat(self, layer: int) -> torch.Tensor:
        """(B, T, 2*H_kv*D): [K ‖ V] per token, heads flattened."""
        k, v = self.keys[layer], self.values[layer]
        b, h, t, d = k.shape
        return torch.cat([k.transpose(1, 2).reshape(b, t, h * d), v.transpose(1, 2).reshape(b, t, h * d)], -1)


def unflat(x: torch.Tensor, n_kv: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `KVBundle.flat`: (B, T, 2*H*D) -> K, V each (B, H, T, D)."""
    b, t, _ = x.shape
    k, v = x.split(n_kv * head_dim, dim=-1)
    return k.reshape(b, t, n_kv, head_dim).transpose(1, 2), v.reshape(b, t, n_kv, head_dim).transpose(1, 2)


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Expected: `model.model.layers` (K2HorizonDecoderLayer); each returns the residual-stream tensor."""
    return list(getattr(getattr(model, "model", model), "layers", []))


@contextlib.contextmanager
def capture_kv(model: nn.Module, with_hidden: bool = False):
    """Context manager: forward hooks on every attention's k_proj/v_proj (and, optionally, on the decoder
    layers for the residual stream; the remote code ignores `output_hidden_states`). Yields dict of lists."""
    n_kv, head_dim = kv_geometry(model.config)
    store: dict[str, list] = {"keys": [], "values": [], "hidden": []}
    handles = []

    def hook(name):
        def fn(_mod, _inp, out):
            b, t, _ = out.shape
            store[name].append(out.view(b, t, n_kv, head_dim).transpose(1, 2))
        return fn

    for attn in attention_modules(model):
        kp, vp = _kv_hook_points(attn)
        handles.append(kp.register_forward_hook(hook("keys")))
        handles.append(vp.register_forward_hook(hook("values")))
    layers = decoder_layers(model) if with_hidden else []
    for i, layer in enumerate(layers):
        if i == 0:
            handles.append(layer.register_forward_pre_hook(lambda _m, args: store["hidden"].append(args[0])))
        handles.append(layer.register_forward_hook(
            lambda _m, _i, out: store["hidden"].append(out[0] if isinstance(out, tuple) else out)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _extract(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
             with_hidden: bool = True) -> KVBundle:
    with capture_kv(model, with_hidden) as store:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return KVBundle(keys=store["keys"], values=store["values"], hidden=store["hidden"])


def extract(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
            with_hidden: bool = True) -> KVBundle:
    """Run `model` on `input_ids` (B, T) and return pre-RoPE K/V for every layer and position, plus the
    residual stream (embeddings and every layer output) if `with_hidden`. A sender with a different tokenizer
    (see align.attach_aligner) is run on its own tokens and gathered back to these positions."""
    if getattr(model, "_k2_aligner", None) is not None:
        from .align import extract_aligned
        return extract_aligned(model, input_ids, with_hidden)
    return _extract(model, input_ids, attention_mask, with_hidden)


extract.__wrapped__ = _extract


# ---- RoPE helpers (default rope_type; theta 1e7 for K2 Horizon) ----

def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype=torch.float32,
                 device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape (B, T, head_dim), same layout as K2HorizonRotaryEmbedding (concat(freqs, freqs))."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = position_ids.to(device=device, dtype=torch.float32)[..., None] * inv
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (B, T, D). Same formula as the remote code."""
    return x * cos[:, None] + _rotate_half(x) * sin[:, None]


def strip_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse rotation: rotate by -theta, i.e. cos(-t)=cos, sin(-t)=-sin."""
    return x * cos[:, None] - _rotate_half(x) * sin[:, None]


def make_cache(model: nn.Module, keys: list[torch.Tensor], values: list[torch.Tensor],
               position_ids: torch.Tensor | None = None) -> DynamicCache:
    """Build a DynamicCache for `model` from PRE-RoPE keys and values (per layer, (B,H,T,D)).
    RoPE is applied here at `position_ids` (default 0..T-1). Autograd flows through."""
    b, _, t, d = keys[0].shape
    if position_ids is None:
        position_ids = torch.arange(t, device=keys[0].device)[None].expand(b, -1)
    cos, sin = rope_cos_sin(position_ids, d, rope_theta(model.config), keys[0].dtype, keys[0].device)
    cache = DynamicCache(config=getattr(model, "config", None))
    for i, (k, v) in enumerate(zip(keys, values)):
        cache.update(apply_rope(k, cos, sin), v.to(k.dtype), i)
    return cache
