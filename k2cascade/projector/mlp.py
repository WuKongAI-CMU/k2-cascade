"""Latent-Cache-Flow style projector (arXiv 2605.22863): per target layer j and group g (kv head, or
one flattened group), y = ridge_j,g(x) + sigmoid(alpha_j,g) * up(SwiGLU(down(x))) with bottleneck d.
The affine part is initialised from the fitted ridge map and (by default) stays trainable; `up` is
zero-initialised and the gate starts near 0 (sigmoid(-4) ~ 0.018), so step 0 equals the ridge baseline.
Weights are grouped tensors used with einsum so per-head and flattened layouts share one code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file, save_file

from .extract import KVBundle
from .ridge import LayerMap, RidgeProjector, feats_to_kv, gather_inputs


class LayerProjector(nn.Module):
    def __init__(self, groups: int, d_in: int, d_out: int, bottleneck: int = 256, hidden_mult: int = 4,
                 gate_init: float = -4.0):
        super().__init__()
        hid = hidden_mult * bottleneck
        self.W = nn.Parameter(torch.zeros(groups, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(groups, d_out))
        self.down = nn.Parameter(torch.randn(groups, d_in, bottleneck) / d_in ** 0.5)
        self.w1 = nn.Parameter(torch.randn(groups, bottleneck, hid) / bottleneck ** 0.5)
        self.w3 = nn.Parameter(torch.randn(groups, bottleneck, hid) / bottleneck ** 0.5)
        self.w2 = nn.Parameter(torch.randn(groups, hid, bottleneck) / hid ** 0.5)
        self.up = nn.Parameter(torch.zeros(groups, bottleneck, d_out))
        self.gate = nn.Parameter(torch.full((groups,), gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,T,G,d_in)
        y = torch.einsum("btgi,gio->btgo", x, self.W) + self.b
        h = torch.einsum("btgi,gio->btgo", x, self.down)
        h = F.silu(torch.einsum("btgi,gio->btgo", h, self.w1)) * torch.einsum("btgi,gio->btgo", h, self.w3)
        h = torch.einsum("btgi,gio->btgo", h, self.w2)
        return y + torch.sigmoid(self.gate)[:, None] * torch.einsum("btgi,gio->btgo", h, self.up)


class MLPProjector(nn.Module):
    def __init__(self, layer_map: LayerMap, groups: int, d_in: int, d_out: int, n_kv: int, head_dim: int,
                 bottleneck: int = 256, hidden_mult: int = 4, gate_init: float = -4.0):
        super().__init__()
        self.meta = dict(layer_map=layer_map, groups=groups, d_in=d_in, d_out=d_out, n_kv=n_kv,
                         head_dim=head_dim, bottleneck=bottleneck, hidden_mult=hidden_mult, gate_init=gate_init)
        self.layer_map, self.n_kv, self.head_dim, self.per_head = layer_map, n_kv, head_dim, groups > 1
        self.layers = nn.ModuleList(
            [LayerProjector(groups, d_in, d_out, bottleneck, hidden_mult, gate_init) for _ in layer_map])

    @classmethod
    def from_ridge(cls, ridge: RidgeProjector, bottleneck: int = 256, hidden_mult: int = 4,
                   gate_init: float = -4.0, freeze_ridge: bool = False) -> "MLPProjector":
        L, G, d_in, d_out = ridge.W.shape
        p = cls(ridge.layer_map, G, d_in, d_out, ridge.n_kv, ridge.head_dim, bottleneck, hidden_mult, gate_init)
        with torch.no_grad():
            for j, lp in enumerate(p.layers):
                lp.W.copy_(ridge.W[j]), lp.b.copy_(ridge.b[j])
                lp.W.requires_grad_(not freeze_ridge), lp.b.requires_grad_(not freeze_ridge)
        return p.to(ridge.W.device)

    def forward(self, src: KVBundle) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        keys, values = [], []
        for j, (layers, lp) in enumerate(zip(self.layer_map, self.layers)):
            y = lp(gather_inputs(src, layers, self.per_head).to(lp.W.dtype))
            k, v = feats_to_kv(y, self.n_kv, self.head_dim)
            keys.append(k), values.append(v)
        return keys, values

    def gates(self) -> torch.Tensor:
        return torch.stack([torch.sigmoid(lp.gate.detach()) for lp in self.layers])

    def save(self, path: str | Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous().cpu() for k, v in self.state_dict().items()}, path / "mlp.safetensors")
        (path / "mlp.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, path: str | Path, device=None) -> "MLPProjector":
        path = Path(path)
        p = cls(**json.loads((path / "mlp.json").read_text()))
        p.load_state_dict(load_file(path / "mlp.safetensors"))
        return p.to(device) if device else p
