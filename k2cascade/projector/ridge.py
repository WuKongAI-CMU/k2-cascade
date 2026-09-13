"""Closed-form ridge KV projector (baseline in the style of arXiv 2608.03893).

For every target layer j: x_t = concat_{l in map[j]} [K_l(t) ‖ V_l(t)] of the source (pre-RoPE) ->
y_t = [K_j(t) ‖ V_j(t)] of the target, fitted by ridge regression over all tokens of the fitting set.
Gram matrices are accumulated batch by batch, so the fit is streaming.

Two feature layouts (3.7B and 7B both have 8 kv-heads x 128, so both are well defined):
  per_head=True  (default): one 2D->2D map per kv head, G = H groups (small, 256x256 per layer/head)
  per_head=False: heads flattened, one (k*2HD)->(2HD) map per layer (mixes heads; 2048x2048)
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from .extract import KVBundle, extract, kv_geometry

LayerMap = list[list[int]]


def last_aligned_map(n_src: int, n_tgt: int, k: int = 1) -> LayerMap:
    """Align the last layers and walk backward: target j <- source n_src-1-(n_tgt-1-j), plus k-1
    neighbours below it (clamped at 0). Identity when n_src == n_tgt (3.7B -> 7B: both 36)."""
    out = []
    for j in range(n_tgt):
        top = max(0, n_src - 1 - (n_tgt - 1 - j))
        out.append(sorted({max(0, top - i) for i in range(k)}))
    return out


def kv_feats(k: torch.Tensor, v: torch.Tensor, per_head: bool) -> torch.Tensor:
    """K, V (B,H,T,D) -> (B,T,G,d): G=H, d=2D per head, or G=1, d=2HD flattened."""
    x = torch.cat([k, v], -1).transpose(1, 2)  # (B,T,H,2D)
    return x if per_head else x.reshape(x.shape[0], x.shape[1], 1, -1)


def feats_to_kv(y: torch.Tensor, n_kv: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `kv_feats`: (B,T,G,d_out) -> K, V each (B,H,T,D)."""
    y = y.reshape(y.shape[0], y.shape[1], n_kv, 2 * head_dim).transpose(1, 2)
    return y[..., :head_dim], y[..., head_dim:]


def gather_inputs(src: KVBundle, layers: list[int], per_head: bool) -> torch.Tensor:
    """(B, T, G, len(layers) * d)."""
    return torch.cat([kv_feats(src.keys[l], src.values[l], per_head) for l in layers], -1)


@torch.no_grad()
def layer_correlation(src: KVBundle, tgt: KVBundle, max_tokens: int = 4096) -> torch.Tensor:
    """Linear CKA between flattened source-layer KV and target-layer KV; (n_src, n_tgt)."""
    def feats(b: KVBundle):
        xs = []
        for l in range(b.num_layers):
            x = kv_feats(b.keys[l], b.values[l], False).reshape(-1, 2 * b.keys[l].shape[1] * b.keys[l].shape[3])
            x = x[:max_tokens].float()
            xs.append(x - x.mean(0, keepdim=True))
        return xs
    S, T = feats(src), feats(tgt)
    corr = torch.zeros(len(S), len(T))
    for i, x in enumerate(S):
        xx = (x.T @ x).norm()
        for j, y in enumerate(T):
            corr[i, j] = (y.T @ x).norm() ** 2 / (xx * (y.T @ y).norm() + 1e-12)
    return corr


def topk_map(corr: torch.Tensor, k: int = 1) -> LayerMap:
    return [sorted(corr[:, j].topk(min(k, corr.shape[0])).indices.tolist()) for j in range(corr.shape[1])]


class RidgeProjector(nn.Module):
    """Per target layer j and group g: y = x @ W[j,g] + b[j,g]. Buffers only (no grad)."""

    def __init__(self, layer_map: LayerMap, groups: int, d_in: int, d_out: int, n_kv: int, head_dim: int):
        super().__init__()
        self.layer_map, self.n_kv, self.head_dim = layer_map, n_kv, head_dim
        self.per_head = groups > 1
        self.register_buffer("W", torch.zeros(len(layer_map), groups, d_in, d_out))
        self.register_buffer("b", torch.zeros(len(layer_map), groups, d_out))

    def forward(self, src: KVBundle) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        keys, values = [], []
        for j, layers in enumerate(self.layer_map):
            x = gather_inputs(src, layers, self.per_head).to(self.W.dtype)
            y = torch.einsum("btgi,gio->btgo", x, self.W[j]) + self.b[j]
            k, v = feats_to_kv(y, self.n_kv, self.head_dim)
            keys.append(k), values.append(v)
        return keys, values

    def save(self, path: str | Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        save_file({"W": self.W.contiguous().cpu(), "b": self.b.contiguous().cpu()}, path / "ridge.safetensors")
        (path / "ridge.json").write_text(json.dumps(
            {"layer_map": self.layer_map, "n_kv": self.n_kv, "head_dim": self.head_dim}))

    @classmethod
    def load(cls, path: str | Path, device=None) -> "RidgeProjector":
        path = Path(path)
        meta = json.loads((path / "ridge.json").read_text())
        t = load_file(path / "ridge.safetensors")
        L, G, d_in, d_out = t["W"].shape
        p = cls(meta["layer_map"], G, d_in, d_out, meta["n_kv"], meta["head_dim"])
        p.W.copy_(t["W"]), p.b.copy_(t["b"])
        return p.to(device) if device else p


@torch.no_grad()
def fit_ridge(src_model: nn.Module, tgt_model: nn.Module, batches, layer_map: LayerMap | None = None,
              lam: float = 1e-2, k: int = 1, map_mode: str = "last_aligned", per_head: bool = True,
              log=None) -> RidgeProjector:
    """batches: iterable of input_ids (B, T) on the models' device. `lam` is relative to the mean
    diagonal of X^T X (scale-free). map_mode: 'last_aligned' or 'topk' (CKA on the first batch)."""
    n_kv, head_dim = kv_geometry(tgt_model.config)
    XtX = XtY = None
    n_tokens = 0
    for ids in batches:
        s, t = extract(src_model, ids, with_hidden=False), extract(tgt_model, ids, with_hidden=False)
        if layer_map is None:
            layer_map = (topk_map(layer_correlation(s, t), k) if map_mode == "topk"
                         else last_aligned_map(s.num_layers, t.num_layers, k))
        if XtX is None:
            x0 = gather_inputs(s, layer_map[0], per_head)
            G, d_in, d_out = x0.shape[2], x0.shape[3], (2 * head_dim if per_head else 2 * n_kv * head_dim)
            XtX = torch.zeros(len(layer_map), G, d_in + 1, d_in + 1, device=ids.device)
            XtY = torch.zeros(len(layer_map), G, d_in + 1, d_out, device=ids.device)
        for j, layers in enumerate(layer_map):
            x = gather_inputs(s, layers, per_head).reshape(-1, G, d_in).float()
            x = torch.cat([x, torch.ones(x.shape[0], G, 1, device=x.device)], -1)
            y = kv_feats(t.keys[j], t.values[j], per_head).reshape(-1, G, d_out).float()
            XtX[j] += torch.einsum("ngi,ngj->gij", x, x)
            XtY[j] += torch.einsum("ngi,ngo->gio", x, y)
        n_tokens += ids.numel()
        if log:
            log(f"ridge: accumulated {n_tokens} tokens")
    proj = RidgeProjector(layer_map, G, d_in, d_out, n_kv, head_dim).to(XtX.device)
    eye = torch.eye(d_in + 1, dtype=torch.float64, device=XtX.device)
    eye[-1, -1] = 0  # do not shrink the bias
    for j in range(len(layer_map)):
        for g in range(G):
            A, B = XtX[j, g].double(), XtY[j, g].double()
            A = A + lam * A.diagonal()[:-1].mean() * eye
            Wb = torch.linalg.solve(A, B)
            proj.W[j, g].copy_(Wb[:-1].float()), proj.b[j, g].copy_(Wb[-1].float())
    proj.n_tokens = n_tokens
    return proj


def main(argv=None) -> None:
    """uv run python -m k2cascade.projector.ridge --data data/fineweb_1024.jsonl --out runs/ridge --seqs 64"""
    import argparse
    from .eval import evaluate
    from .train import TrainConfig, freeze, jsonl_sequences, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="runs/ridge")
    ap.add_argument("--seqs", type=int, default=64); ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--prefix_len", type=int, default=512); ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--map", default="last_aligned")
    ap.add_argument("--flat", action="store_true", help="flatten heads (default: per kv head)")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(source=a.source, target=a.target)
    src, tgt, tok = load_models(cfg, dev)
    freeze(src), freeze(tgt)
    seqs = jsonl_sequences(a.data, tok, a.seq_len)
    held, fit = seqs[:a.eval_seqs], seqs[a.eval_seqs:a.eval_seqs + a.seqs]
    mk = lambda rows: [torch.tensor(rows[i:i + a.batch], device=dev) for i in range(0, len(rows) - a.batch + 1, a.batch)]
    proj = fit_ridge(src, tgt, mk(fit), lam=a.lam, k=a.k, map_mode=a.map, per_head=not a.flat, log=print)
    proj.save(a.out)
    print("layer_map:", proj.layer_map)
    print("eval:", json.dumps(evaluate(src, tgt, proj, mk(held), a.prefix_len, with_source_oracle=True)))


if __name__ == "__main__":
    main()
