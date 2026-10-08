"""Memory lens and interventions on transplanted memories, recorded per episode for the web article.

  episodes   per binding episode: the facts, question, answer, partner answer; the receiver's distribution over
             the eight values for the arms none / text / project / derange; and a logit lens: at every receiver
             layer, the distribution over the eight values read from the residual stream at the last position
             (final norm, then the unembedding), for text / project / derange.
  blend      interpolate the receiver-side memory between this episode's and the partner's,
             (1 - a) * own + a * partner per layer, for a in 0..1: P(own value), P(partner value), accuracy.
  onset      own memory in layers >= l (partner below) and in layers < l (partner above), for every l.
  squad      SQuAD cases: per arm the top-5 first answer tokens with probabilities and the greedy answer.

uv run python -m k2cascade.projector.lens --projector runs/mlp/step_1500 --out analysis/lens
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .extract import decoder_layers, extract, make_cache
from .noma import COLOURS, NAMES, Encoder, make_episodes


def _cache(model, keys, values):
    return make_cache(model, [k.to(model.dtype) for k in keys], [v.to(model.dtype) for v in values])


@torch.no_grad()
def forward_with_lens(tgt, question: torch.Tensor, cache, prefix_len: int, value_ids: torch.Tensor):
    """Next-token log-probs over the full vocabulary, and per-layer distributions over `value_ids`."""
    hidden = []
    hooks = [layer.register_forward_hook(lambda _m, _i, out: hidden.append((out[0] if isinstance(out, tuple) else out)[0, -1]))
             for layer in decoder_layers(tgt)]
    try:
        t = question.shape[1]
        pos = torch.arange(prefix_len, prefix_len + t, device=question.device)
        out = tgt(input_ids=question, past_key_values=cache, position_ids=pos[None], cache_position=pos,
                  use_cache=cache is not None)
    finally:
        for h in hooks:
            h.remove()
    norm = tgt.model.norm
    lens = []
    for h in hidden:
        logits = tgt.lm_head(norm(h[None, None]))[0, 0].float()
        lens.append(torch.softmax(logits[value_ids], -1).tolist())
    return torch.log_softmax(out.logits[0, -1].float(), -1), lens


def mapped(src, projector, facts):
    return projector(extract(src, facts, with_hidden=False))


def dist8(lp, value_ids):
    return torch.softmax(lp[value_ids], -1).tolist()


@torch.no_grad()
def record_episodes(src, tgt, projector, enc: Encoder, eps, n_lens: int):
    dev = next(tgt.parameters()).device
    vid = torch.tensor(enc.colour_ids, device=dev)
    out = []
    for i, e in enumerate(eps):
        f = torch.tensor([enc.facts(e)], device=dev)
        pf = torch.tensor([enc.facts(eps[e.partner])], device=dev)
        q = torch.tensor([enc.question(e)], device=dev)
        p = f.shape[1]
        row = {"i": i, "names": enc.names, "values": enc.colours, "colours": e.colours, "query": e.query,
               "answer": e.answer, "partner_answer": eps[e.partner].colours[e.query],
               "facts_text": " ".join(f"{n} is {enc.colours[c]}." for n, c in zip(enc.names, e.colours)),
               "question_text": f"What colour is {enc.names[e.query]}?"}
        lp_none, _ = forward_with_lens(tgt, torch.cat([f[:, :1], q], 1), None, 0, vid)
        row["none"] = dist8(lp_none, vid)
        arms = {"text": (torch.cat([f, q], 1), None, 0)}
        k, v = mapped(src, projector, f); arms["project"] = (q, _cache(tgt, k, v), p)
        k, v = mapped(src, projector, pf); arms["derange"] = (q, _cache(tgt, k, v), p)
        for a, (qq, c, pl) in arms.items():
            lp, lens = forward_with_lens(tgt, qq, c, pl, vid)
            row[a] = dist8(lp, vid)
            if i < n_lens:
                row[f"lens_{a}"] = lens
        out.append(row)
    return out


@torch.no_grad()
def blend(src, tgt, projector, enc, eps, alphas):
    dev = next(tgt.parameters()).device
    vid = torch.tensor(enc.colour_ids, device=dev)
    res = {a: {"p_own": 0.0, "p_partner": 0.0, "acc": 0, "follow": 0} for a in alphas}
    for e in eps:
        f = torch.tensor([enc.facts(e)], device=dev); pf = torch.tensor([enc.facts(eps[e.partner])], device=dev)
        q = torch.tensor([enc.question(e)], device=dev)
        ko, vo = mapped(src, projector, f); kp, vp = mapped(src, projector, pf)
        own, par = e.answer, eps[e.partner].colours[e.query]
        for a in alphas:
            k = [(1 - a) * x + a * y for x, y in zip(ko, kp)]; v = [(1 - a) * x + a * y for x, y in zip(vo, vp)]
            lp, _ = forward_with_lens(tgt, q, _cache(tgt, k, v), f.shape[1], vid)
            d = torch.softmax(lp[vid], -1)
            r = res[a]; r["p_own"] += d[own].item(); r["p_partner"] += d[par].item()
            r["acc"] += int(d.argmax().item() == own); r["follow"] += int(d.argmax().item() == par)
    n = len(eps)
    return [{"alpha": a, **{k: v / n for k, v in r.items()}} for a, r in res.items()]


@torch.no_grad()
def onset(src, tgt, projector, enc, eps):
    dev = next(tgt.parameters()).device
    vid = torch.tensor(enc.colour_ids, device=dev)
    L = tgt.config.num_hidden_layers
    res = {"from": [[0, 0] for _ in range(L + 1)], "until": [[0, 0] for _ in range(L + 1)]}
    for e in eps:
        f = torch.tensor([enc.facts(e)], device=dev); pf = torch.tensor([enc.facts(eps[e.partner])], device=dev)
        q = torch.tensor([enc.question(e)], device=dev)
        ko, vo = mapped(src, projector, f); kp, vp = mapped(src, projector, pf)
        own, par = e.answer, eps[e.partner].colours[e.query]
        for l in range(L + 1):
            for mode in ("from", "until"):
                take = (lambda j: j >= l) if mode == "from" else (lambda j: j < l)
                k = [x if take(j) else y for j, (x, y) in enumerate(zip(ko, kp))]
                v = [x if take(j) else y for j, (x, y) in enumerate(zip(vo, vp))]
                lp, _ = forward_with_lens(tgt, q, _cache(tgt, k, v), f.shape[1], vid)
                am = torch.softmax(lp[vid], -1).argmax().item()
                res[mode][l][0] += int(am == own); res[mode][l][1] += int(am == par)
    n = len(eps)
    return {m: [{"layer": l, "acc": a / n, "follow": b / n} for l, (a, b) in enumerate(v)] for m, v in res.items()}


@torch.no_grad()
def squad_cases(src, tgt, projector, tok, rows, k_top: int = 5):
    from .qa import QAEncoder
    dev = next(tgt.parameters()).device
    enc = QAEncoder(tok)
    out = []
    for i, ex in enumerate(rows):
        case = {"question": ex["question"], "answers": ex["answers"], "sender_cands": ex.get("sender_cands"),
                "sender_conf": ex.get("sender_conf"), "variants": {}}
        prev = rows[(i + 1) % len(rows)]
        for var in ("clean", "removed"):
            ctx = ex.get(var, ex.get("context"))
            pa = torch.tensor([enc.passage(ctx)], device=dev)
            other = torch.tensor([enc.passage(prev.get(var, prev.get("context")))], device=dev)
            q = torch.tensor([enc.question(ex["question"])], device=dev)
            qv = torch.tensor([enc.question(ex["question"], None, (ex.get("sender_answer", ""), ex.get("sender_conf", "")))], device=dev) if ex.get("sender_answer") else None
            arms = {"text": (torch.cat([pa, q], 1), None, 0)}
            kk, vv = mapped(src, projector, pa); arms["memory"] = (q, _cache(tgt, kk, vv), pa.shape[1])
            kk, vv = mapped(src, projector, other); arms["wrong_memory"] = (q, _cache(tgt, kk, vv), other.shape[1])
            if qv is not None:
                arms["word"] = (torch.cat([torch.tensor([enc.bos], device=dev), qv], 1) if enc.bos else qv, None, 0)
            arms["question_only"] = (torch.cat([torch.tensor([enc.bos], device=dev), q], 1) if enc.bos else q, None, 0)
            vr = {"passage": ctx[:1200]}
            for a, (qq, c, pl) in arms.items():
                t = qq.shape[1]; pos = torch.arange(pl, pl + t, device=dev)
                logits = tgt(input_ids=qq, past_key_values=c, position_ids=pos[None], cache_position=pos, use_cache=c is not None).logits[0, -1].float()
                pr = torch.softmax(logits, -1); top = pr.topk(k_top)
                ent = float(-(pr * torch.log(pr.clamp_min(1e-30))).sum())
                vr[a] = {"top": [[tok.decode([int(t_)]), round(float(p_), 4)] for p_, t_ in zip(top.values, top.indices)], "entropy": ent}
            case["variants"][var] = vr
        out.append(case)
    return out


def main(argv=None) -> None:
    import argparse
    from .mlp import MLPProjector
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", required=True); ap.add_argument("--out", default="analysis/lens")
    ap.add_argument("--episodes", type=int, default=200); ap.add_argument("--n_lens", type=int, default=40)
    ap.add_argument("--squad", default=None, help="jsonl with clean/removed variants and sender_* fields")
    ap.add_argument("--n_squad", type=int, default=24); ap.add_argument("--skip_binding", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    proj = (MLPProjector.load if (Path(a.projector) / "mlp.safetensors").exists() else RidgeProjector.load)(a.projector, dev)
    enc = Encoder(tok, 8, 8)
    eps = make_episodes(a.episodes, 8, 8, seed=7)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    if not a.skip_binding:
        (o / "episodes.json").write_text(json.dumps(record_episodes(src, tgt, proj, enc, eps, a.n_lens)))
        (o / "blend.json").write_text(json.dumps(blend(src, tgt, proj, enc, eps, [round(x * 0.1, 1) for x in range(11)]), indent=1))
        (o / "onset.json").write_text(json.dumps(onset(src, tgt, proj, enc, eps), indent=1))
    if a.squad:
        rows = [json.loads(l) for l in open(a.squad) if l.strip()][: a.n_squad]
        (o / "squad_cases.json").write_text(json.dumps(squad_cases(src, tgt, proj, tok, rows), ensure_ascii=False, indent=1))
    print("wrote", sorted(p.name for p in o.iterdir()))


if __name__ == "__main__":
    main()
