"""Where does the sender's uncertainty get lost: in the map, or in the receiver's read-out?

For each item (passage + question, one variant) we collect three feature vectors and one label:
  raw      the sender's own last-prompt-token hidden state (mean over chosen layers)
  mapped   the mapped cache at the last prefix position (K and V, mean over layers and heads)
  receiver the receiver's last hidden state after reading the question through the mapped cache
  text     the receiver's last hidden state after reading the passage + question as text (reference)
  label    sender semantic entropy above the median (from sender_entropy.py)
Equal-capacity logistic probes (nested 5-fold) give one AUROC per stage. raw high / mapped low = map loss;
mapped high / receiver low = read-out loss.

uv run python -m k2cascade.projector.diag_uncertainty --projector runs/mlp --data data/conf.jsonl --se analysis/conf/se_clean.jsonl --out analysis/diag_unc_clean.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .extract import extract
from .qa import QAEncoder, _cache


@torch.no_grad()
def features(src, tgt, proj, enc: QAEncoder, ex: dict, variant: str, layers: list[int]) -> dict:
    dev = next(tgt.parameters()).device
    ctx = ex.get(variant, ex.get("context")) if variant != "clean" else ex.get("clean", ex["context"])
    pa = torch.tensor([enc.passage(ctx)], device=dev)
    q = torch.tensor([enc.question(ex["question"])], device=dev)
    full = torch.cat([pa, q], 1)
    b = extract(src, full, with_hidden=True)
    raw = np.mean([b.hidden[j][0, -1].float().cpu().numpy() for j in layers], 0)
    bp = extract(src, pa, with_hidden=False)
    k, v = proj(bp)
    mapped = np.concatenate([np.mean([x[0, :, -1, :].float().cpu().numpy() for x in k], 0).reshape(-1),
                             np.mean([x[0, :, -1, :].float().cpu().numpy() for x in v], 0).reshape(-1)])
    cache = _cache(tgt, k, v)
    pos = torch.arange(pa.shape[1], pa.shape[1] + q.shape[1], device=dev)
    h_rec = tgt.model(input_ids=q, past_key_values=cache, position_ids=pos[None], cache_position=pos,
                      use_cache=True).last_hidden_state[0, -1].float().cpu().numpy()
    h_text = tgt.model(input_ids=full).last_hidden_state[0, -1].float().cpu().numpy()
    bos = torch.tensor([enc.bos], device=dev) if enc.bos else None
    qo = torch.cat([bos, q], 1) if bos is not None else q
    h_q = tgt.model(input_ids=qo).last_hidden_state[0, -1].float().cpu().numpy()  # question only, no passage
    return {"raw": raw, "mapped": mapped, "mapped+q": np.concatenate([mapped, h_q]), "receiver": h_rec,
            "text": h_text, "question": h_q}


def probe_auroc(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, groups=None) -> float:
    """Out-of-fold AUROC of a standardised logistic probe; folds are split by `groups` (e.g. passage) when given."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed); out = np.zeros(len(y))
    if groups is None:
        groups = np.arange(len(y))
    ug = rng.permutation(np.unique(groups)); gfold = {g: i % folds for i, g in enumerate(ug)}
    fold_of = np.array([gfold[g] for g in groups])
    for f in range(folds):
        te = np.where(fold_of == f)[0]; tr = np.where(fold_of != f)[0]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        clf.fit(X[tr], y[tr]); out[te] = clf.predict_proba(X[te])[:, 1]
    pos, neg = out[y == 1], out[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    from .mlp import MLPProjector
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", required=True); ap.add_argument("--data", required=True); ap.add_argument("--se", required=True)
    ap.add_argument("--variant", default="clean"); ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--layers", default="12,18,24"); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    proj = (MLPProjector.load if (Path(a.projector) / "mlp.safetensors").exists() else RidgeProjector.load)(a.projector, dev)
    enc = QAEncoder(tok); layers = [int(x) for x in a.layers.split(",")]
    se = {r["i"]: r["se"] for r in (json.loads(l) for l in open(a.se) if l.strip())}
    feats = {k: [] for k in ("raw", "mapped", "mapped+q", "receiver", "text", "question")}; ses = []; groups = []
    for i, line in enumerate(open(a.data)):
        if i >= a.n or i not in se:
            continue
        ex = json.loads(line)
        f = features(src, tgt, proj, enc, ex, a.variant, layers)
        for k in feats:
            feats[k].append(f[k])
        ses.append(se[i]); groups.append(hash(ex.get("clean", ex.get("context", ""))[:200]))
    y = (np.array(ses) > np.median(ses)).astype(int); groups = np.array(groups)
    res = {"variant": a.variant, "n": int(len(y)), "positive_rate": float(y.mean()), "n_groups": int(len(set(groups.tolist()))),
           "auroc": {k: probe_auroc(np.stack(v).astype(np.float32), y, groups=groups) for k, v in feats.items()},
           "dims": {k: int(np.stack(v).shape[1]) for k, v in feats.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == "__main__":
    main()
