"""The shared workspace: what crosses when one model reads another's memory.

J-space (Gurnee et al. 2026) is the part of a model's residual stream that the Jacobian lens reads out as words: a
sparse non-negative combination of J-lens vectors a_w = J_l^T (gamma * W_U[w]), where a_w . h is the lens logit of
word w at layer l (up to the final norm's per-position scale). Its coordinates are named by the vocabulary, so two
models with one tokenizer share a coordinate system.

  coords   one page read by sender and receiver: for every (sender layer, receiver layer), the mean overlap of the
           top-k J-lens words per position, against the same with positions shuffled. Logit lens alongside.
  swap     at the queried value position, in the sender's workspace layers, replace with the partner episode's:
           the whole state; only its J-space part; only the rest; a random edit as large as the J-space swap; only
           the two colour coordinates. The receiver reads the mapped memory; the sender answers the question too.

uv run python -m k2cascade.projector.workspace --projector runs/mlp_seed0 --src_lens runs/jlens_src/lens.pt \
    --tgt_lens runs/jlens_tgt/lens.pt --out analysis/workspace
"""
from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import torch

from .extract import decoder_layers
from .lens import _cache, fact_positions, forward_with_lens, mapped
from .noma import Encoder, make_episodes


class Workspace:
    """J-lens atoms of one model. `J` maps layer -> (d, d); `WU` is the unembedding with the final norm's gain
    folded in, so the lens logit of word w for residual h is (J h) . WU[w]."""

    def __init__(self, model, jacobians: dict[int, torch.Tensor]):
        dev = model.lm_head.weight.device
        self.WU = (model.lm_head.weight.float() * model.model.norm.weight.float()).to(torch.bfloat16)
        self.J = {l: J.to(dev, torch.float32) for l, J in jacobians.items()}
        self.layers = sorted(self.J)
        self._norms: dict[int, torch.Tensor] = {}

    @classmethod
    def load(cls, model, path: str):
        import jlens
        return cls(model, jlens.JacobianLens.load(path).jacobians)

    def atoms(self, layer: int, ids: torch.Tensor) -> torch.Tensor:
        """a_w for the given word ids: (n, d) float32."""
        return self.WU[ids].float() @ self.J[layer]

    def norms(self, layer: int, chunk: int = 16384) -> torch.Tensor:
        if layer not in self._norms:
            J = self.J[layer].to(torch.bfloat16)
            self._norms[layer] = torch.cat([(self.WU[i:i + chunk] @ J).float().norm(dim=1)
                                            for i in range(0, self.WU.shape[0], chunk)]).clamp_min(1e-6)
        return self._norms[layer]

    def scores(self, layer: int, h: torch.Tensor, normalised: bool = True) -> torch.Tensor:
        """Lens logits (or cosines with the unit atoms) of every word for residuals h: (T, d) -> (T, V)."""
        s = ((h.float() @ self.J[layer].T).to(torch.bfloat16) @ self.WU.T).float()
        return s / self.norms(layer) if normalised else s

    def decompose(self, layer: int, h: torch.Tensor, k: int = 25):
        """Greedy non-negative matching pursuit over the unit J-lens atoms: h = J-part + rest, J-part a sum of at
        most k atoms with positive weights. Returns (ids (T, k), weights (T, k), J-part (T, d)); a weight of 0
        marks a step at which no atom had positive overlap with what was left."""
        r = h.float().clone()
        ids = torch.zeros(h.shape[0], k, dtype=torch.long, device=h.device)
        ws = torch.zeros(h.shape[0], k, device=h.device)
        nrm = self.norms(layer)
        for i in range(k):
            s = self.scores(layer, r)
            c, w = s.max(-1)
            c = c.clamp_min(0)
            r = r - c[:, None] * self.atoms(layer, w) / nrm[w][:, None]
            ids[:, i], ws[:, i] = w, c
        return ids, ws, h.float() - r


@contextlib.contextmanager
def edit_layers(model, edits: dict[int, callable]):
    """Forward hooks that replace each listed layer's output with fn(output) (tensor (B, T, d))."""
    def hook(fn):
        def f(_m, _i, out):
            if isinstance(out, tuple):
                return (fn(out[0]),) + tuple(out[1:])
            return fn(out)
        return f
    layers = decoder_layers(model)
    handles = [layers[l].register_forward_hook(hook(fn)) for l, fn in edits.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def record_layers(model, layers: list[int], pos: int):
    """Records each listed layer's output at one position (batch 0)."""
    store: dict[int, torch.Tensor] = {}
    def hook(l):
        def f(_m, _i, out):
            store[l] = (out[0] if isinstance(out, tuple) else out)[0, pos].detach().float().clone()
        return f
    ls = decoder_layers(model)
    handles = [ls[l].register_forward_hook(hook(l)) for l in layers]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def band(n_layers: int, lo: float = 0.38, hi: float = 0.92) -> list[int]:
    """Layers in the relative-depth band where the J-space paper finds its workspace (layer 38 to 92 of 100)."""
    return [l for l in range(n_layers - 1) if lo <= l / (n_layers - 1) <= hi]


ARMS = ("none", "full", "jspace", "rest", "random", "colour")


def _edit(arm: str, ws: Workspace, l: int, k: int, own_id: int, par_id: int, hp: torch.Tensor, jp: torch.Tensor,
          gen: torch.Generator, stats: dict):
    """The replacement for the value-position state h at layer l under one arm."""
    def fn(h: torch.Tensor) -> torch.Tensor:
        if arm == "full":
            return hp
        _, _, jh = ws.decompose(l, h[None], k)
        jh = jh[0]
        stats.setdefault("j_share", []).append((jh.norm() ** 2 / h.norm() ** 2).item())
        if arm == "jspace":
            return h - jh + jp
        if arm == "rest":
            return jh + (hp - jp)
        if arm == "random":
            d = torch.randn(h.shape, generator=gen, device="cpu").to(h.device)
            return h + d / d.norm() * (jp - jh).norm()
        if arm == "colour":
            A = ws.atoms(l, torch.tensor([own_id, par_id], device=h.device)).T  # (d, 2)
            a = torch.linalg.lstsq(A, h[:, None]).solution[:, 0]
            return h + A @ (a.flip(0) - a)
        raise ValueError(arm)
    return fn


@torch.no_grad()
def swap(src, tgt, projector, enc: Encoder, eps, ws: Workspace, layers: list[int], k: int = 25, seed: int = 0,
         arms=ARMS) -> dict:
    """See the module docstring. Per arm: receiver P(own), P(partner), accuracy, follow; the sender's own
    answer to the question (accuracy, follow); and the J-space share of the edited states' squared norm."""
    dev = next(tgt.parameters()).device
    vid = torch.tensor(enc.colour_ids, device=dev)
    gen = torch.Generator().manual_seed(seed)
    res = {a: dict(p_own=0.0, p_partner=0.0, acc=0, follow=0, src_acc=0, src_follow=0) for a in arms}
    stats: dict = {}
    for e in eps:
        f = torch.tensor([enc.facts(e)], device=dev); pf = torch.tensor([enc.facts(eps[e.partner])], device=dev)
        q = torch.tensor([enc.question(e)], device=dev)
        vpos = fact_positions(enc, e)[0][e.query]
        own, par = e.answer, eps[e.partner].colours[e.query]
        with record_layers(src, layers, vpos) as hp:
            src(input_ids=pf, use_cache=False)
        jp = {l: ws.decompose(l, hp[l][None], k)[2][0] for l in layers}
        for arm in arms:
            edits = {} if arm == "none" else {
                l: _at(vpos, _edit(arm, ws, l, k, vid[own].item(), vid[par].item(), hp[l], jp[l], gen, stats))
                for l in layers}
            with edit_layers(src, edits):
                kk, vv = mapped(src, projector, f)
                sl = src(input_ids=torch.cat([f, q], 1), use_cache=False).logits[0, -1, vid].float()
            lp, _ = forward_with_lens(tgt, q, _cache(tgt, kk, vv), f.shape[1], vid)
            d = torch.softmax(lp[vid], -1); r = res[arm]
            r["p_own"] += d[own].item(); r["p_partner"] += d[par].item()
            r["acc"] += int(d.argmax().item() == own); r["follow"] += int(d.argmax().item() == par)
            r["src_acc"] += int(sl.argmax().item() == own); r["src_follow"] += int(sl.argmax().item() == par)
    n = len(eps)
    out = {a: {kk: v / n for kk, v in r.items()} for a, r in res.items()}
    js = stats.get("j_share", [])
    out["j_share_mean"] = sum(js) / max(len(js), 1)
    out["layers"], out["k"], out["n"] = layers, k, n
    return out


def _at(pos: int, fn):
    def g(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[0, pos] = fn(x[0, pos].float()).to(x.dtype)
        return x
    return g


@contextlib.contextmanager
def record_all(model):
    """Records every layer's output at every position (batch 0): layer -> (T, d) float32."""
    store: dict[int, torch.Tensor] = {}
    def hook(l):
        def f(_m, _i, out):
            store[l] = (out[0] if isinstance(out, tuple) else out)[0].detach().float()
        return f
    handles = [m.register_forward_hook(hook(l)) for l, m in enumerate(decoder_layers(model))]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean over positions of |top-k(a) ∩ top-k(b)| / k for id tensors (T, k)."""
    return (a[:, :, None] == b[:, None, :]).any(-1).float().mean().item()


@torch.no_grad()
def coords(src, tgt, enc: Encoder, eps, ws_s: Workspace, ws_t: Workspace, k: int = 25, n_pages: int = 16,
           seed: int = 0) -> dict:
    """Top-k word overlap for every (sender layer, receiver layer) on the same pages, by J-lens and by logit lens,
    at matched positions and with receiver positions shuffled. Position 0 (BOS) is skipped."""
    dev = next(tgt.parameters()).device
    rng = random.Random(seed)
    Ls, Lt = ws_s.layers, ws_t.layers
    out = {m: [[0.0] * len(Lt) for _ in Ls] for m in ("j", "j_shuffled", "logit", "logit_shuffled")}
    for e in eps[:n_pages]:
        f = torch.tensor([enc.facts(e)], device=dev)
        with record_all(src) as hs:
            src(input_ids=f, use_cache=False)
        with record_all(tgt) as ht:
            tgt(input_ids=f, use_cache=False)
        pos = list(range(1, f.shape[1])); perm = pos[:]; rng.shuffle(perm)
        tops = {}
        for name, ws, H, L in (("s", ws_s, hs, Ls), ("t", ws_t, ht, Lt)):
            for l in L:
                tops[name, "j", l] = ws.scores(l, H[l][pos]).topk(k, -1).indices
                tops[name, "logit", l] = (H[l][pos].to(torch.bfloat16) @ ws.WU.T).float().topk(k, -1).indices
        for i, ls in enumerate(Ls):
            for j, lt in enumerate(Lt):
                for lens in ("j", "logit"):
                    a, b = tops["s", lens, ls], tops["t", lens, lt]
                    out[lens][i][j] += _overlap(a, b) / n_pages
                    out[lens + "_shuffled"][i][j] += _overlap(a, b[[p - 1 for p in perm]]) / n_pages
    out["sender_layers"], out["receiver_layers"], out["k"], out["n_pages"] = Ls, Lt, k, n_pages
    return out


def main(argv=None) -> None:
    import argparse
    from .mlp import MLPProjector
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", required=True); ap.add_argument("--out", default="analysis/workspace")
    ap.add_argument("--src_lens", required=True); ap.add_argument("--tgt_lens", required=True)
    ap.add_argument("--episodes", type=int, default=200); ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--pages", type=int, default=16)
    ap.add_argument("--skip_coords", action="store_true"); ap.add_argument("--skip_swap", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    proj = (MLPProjector.load if (Path(a.projector) / "mlp.safetensors").exists() else RidgeProjector.load)(a.projector, dev)
    enc = Encoder(tok, 8, 8)
    eps = make_episodes(a.episodes, 8, 8, seed=7)
    ws_s, ws_t = Workspace.load(src, a.src_lens), Workspace.load(tgt, a.tgt_lens)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    if not a.skip_coords:
        (o / "coords.json").write_text(json.dumps(coords(src, tgt, enc, eps, ws_s, ws_t, a.k, a.pages)))
        print("coords done", flush=True)
    if not a.skip_swap:
        layers = band(src.config.num_hidden_layers)
        r = {"jlens": swap(src, tgt, proj, enc, eps, ws_s, layers, a.k)}
        print("swap jlens", json.dumps({m: r["jlens"][m] for m in ARMS}), flush=True)
        eye = torch.eye(src.config.hidden_size)
        ws_id = Workspace(src, {l: eye for l in ws_s.layers})
        r["logit"] = swap(src, tgt, proj, enc, eps, ws_id, layers, a.k, arms=("jspace", "rest", "colour"))
        (o / "swap.json").write_text(json.dumps(r, indent=1))
    print("wrote", sorted(p.name for p in o.iterdir()))


if __name__ == "__main__":
    main()
