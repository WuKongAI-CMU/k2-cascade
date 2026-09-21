"""Noma binding-transfer test: does the translated sender cache carry a fact only the sender saw?

Each episode assigns random colours to a fixed list of nonce names ("Noma is amber. Vela is indigo. ...").
Only the sender reads that text. The receiver is asked "What colour is <name>?" and we score the next-token
log-probabilities of every colour. Arms (all use the same question and the same positions):

  none      receiver sees only the question                       floor
  text      receiver reads the facts itself                       ceiling
  self      receiver's own pre-RoPE KV of the facts, re-injected  checks the injection path (should equal text)
  raw       sender KV injected without translation                does matching shape alone work?
  project   sender KV -> projector -> receiver cache              the test
  derange   projected KV from another episode with the same layout the control: content, or just compute?

Content-specific transfer = acc(project) - acc(derange). For `derange` we also report how often the answer
follows the OTHER episode's colour: if content moves, the answer should track the message it was given.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from .extract import extract, make_cache

NAMES = ["Noma", "Vela", "Tarn", "Quill", "Brisa", "Oskel", "Mirel", "Dunra"]
COLOURS = ["amber", "indigo", "crimson", "teal", "violet", "olive", "coral", "silver"]
ARMS = ("none", "text", "self", "raw", "project", "derange")


@dataclass
class Episode:
    colours: list[int]   # colour index for each name, in NAMES order
    query: int           # index of the queried name
    partner: int = -1    # index of the derangement episode

    @property
    def answer(self) -> int:
        return self.colours[self.query]


def make_episodes(n: int, n_names: int, n_colours: int, seed: int = 0) -> list[Episode]:
    """Random colour assignments; each episode is paired with another whose colour for the queried name differs
    (same names, same order, so the token layout is identical)."""
    rng = random.Random(seed)
    eps = [Episode([rng.randrange(n_colours) for _ in range(n_names)], rng.randrange(n_names)) for _ in range(n)]
    for i, e in enumerate(eps):
        cands = [j for j, o in enumerate(eps) if j != i and o.colours[e.query] != e.answer]
        e.partner = rng.choice(cands) if cands else (i + 1) % n
    return eps


class Encoder:
    """Turns an episode into token ids. `facts` (with BOS) goes to the sender; `question` ends right before the colour."""

    def __init__(self, tokenizer, n_names: int, n_colours: int):
        self.tok, self.names, self.colours = tokenizer, NAMES[:n_names], COLOURS[:n_colours]
        self.colour_ids = []
        for c in self.colours:
            ids = tokenizer(" " + c, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"colour {c!r} is not a single token: {ids}")
            self.colour_ids.append(ids[0])
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    def facts(self, e: Episode) -> list[int]:
        text = " ".join(f"{n} is {self.colours[c]}." for n, c in zip(self.names, e.colours))
        return self.bos + self.tok(text, add_special_tokens=False)["input_ids"]

    def question(self, e: Episode) -> list[int]:
        q = f"\nQuestion: What colour is {self.names[e.query]}?\nAnswer: {self.names[e.query]} is"
        return self.tok(q, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def next_token_logprobs(model: nn.Module, question: torch.Tensor, cache=None, prefix_len: int = 0) -> torch.Tensor:
    """Log-probs of the token after `question` (1, T), given an optional cache holding `prefix_len` positions."""
    t = question.shape[1]
    pos = torch.arange(prefix_len, prefix_len + t, device=question.device)
    out = model(input_ids=question, past_key_values=cache, position_ids=pos[None], cache_position=pos,
                use_cache=cache is not None)
    return torch.log_softmax(out.logits[0, -1].float(), -1)


def _cache_from_bundle(model, keys, values):
    return make_cache(model, [k.to(model.dtype) for k in keys], [v.to(model.dtype) for v in values])


@torch.no_grad()
def arm_logprobs(arm: str, src: nn.Module, tgt: nn.Module, projector: nn.Module | None,
                 facts: torch.Tensor, partner_facts: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary next-token log-probs for one arm. `facts`/`partner_facts` are (1, P) with identical P."""
    p = facts.shape[1]
    if arm == "none":
        q = torch.cat([facts[:, :1], question], 1) if facts.shape[1] else question  # keep BOS
        return next_token_logprobs(tgt, q)
    if arm == "text":
        return next_token_logprobs(tgt, torch.cat([facts, question], 1))
    if arm == "self":
        b = extract(tgt, facts, with_hidden=False)
        return next_token_logprobs(tgt, question, _cache_from_bundle(tgt, b.keys, b.values), p)
    if arm == "raw":
        b = extract(src, facts, with_hidden=False)
        return next_token_logprobs(tgt, question, _cache_from_bundle(tgt, b.keys, b.values), p)
    if arm in ("project", "derange"):
        b = extract(src, facts if arm == "project" else partner_facts, with_hidden=False)
        keys, values = projector(b)
        return next_token_logprobs(tgt, question, _cache_from_bundle(tgt, keys, values), p)
    raise ValueError(arm)


def run(src, tgt, projector, enc: Encoder, episodes: list[Episode], arms=ARMS) -> dict:
    """Accuracy, mean log-prob of the true colour, and (derange) follow-rate over the colour set."""
    dev = next(tgt.parameters()).device
    cid = torch.tensor(enc.colour_ids, device=dev)
    stats = {a: {"correct": 0, "logp": 0.0, "follow": 0} for a in arms}
    for e in episodes:
        f = torch.tensor([enc.facts(e)], device=dev)
        pf = torch.tensor([enc.facts(episodes[e.partner])], device=dev)
        assert f.shape == pf.shape, "derangement partner must have the same token layout"
        q = torch.tensor([enc.question(e)], device=dev)
        for a in arms:
            if a in ("project", "derange") and projector is None:
                continue
            lp = arm_logprobs(a, src, tgt, projector, f, pf, q)[cid]
            pred = int(lp.argmax())
            s = stats[a]
            s["correct"] += pred == e.answer
            s["logp"] += float(lp[e.answer] - torch.logsumexp(lp, 0))
            if a == "derange":
                s["follow"] += pred == episodes[e.partner].colours[e.query]
    n = len(episodes)
    out = {a: {"acc": s["correct"] / n, "logp_true": s["logp"] / n} for a, s in stats.items()
           if not (a in ("project", "derange") and projector is None)}
    if "derange" in out:
        out["derange"]["follow_rate"] = stats["derange"]["follow"] / n
    if "project" in out and "derange" in out:
        out["content_transfer"] = out["project"]["acc"] - out["derange"]["acc"]
    if all(k in out for k in ("none", "text", "project")):
        gap = out["text"]["acc"] - out["none"]["acc"]
        out["retention"] = (out["project"]["acc"] - out["none"]["acc"]) / gap if gap else float("nan")
    out.update(n=n, chance=1 / len(enc.colour_ids))
    return out


def main(argv=None) -> None:
    """uv run python -m k2cascade.projector.noma --projector runs/ridge --episodes 300 --out analysis/noma_ridge.json
    --projector none runs only none/text/self/raw (a pre-check that needs no fitted map)."""
    import argparse, json
    from pathlib import Path
    from .ridge import RidgeProjector
    from .train import TrainConfig, freeze, load_models
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="IFM/K2-Horizon-3.7B"); ap.add_argument("--target", default="IFM/K2-Horizon-7B")
    ap.add_argument("--projector", default="none"); ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--names", type=int, default=6); ap.add_argument("--colours", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="analysis/noma.json")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src, tgt, tok = load_models(TrainConfig(source=a.source, target=a.target), dev)
    freeze(src), freeze(tgt)
    if a.projector == "none":
        proj = None
    elif (Path(a.projector) / "mlp.safetensors").exists():
        from .mlp import MLPProjector
        proj = MLPProjector.load(a.projector, dev)
    else:
        proj = RidgeProjector.load(a.projector, dev)
    enc = Encoder(tok, a.names, a.colours)
    res = run(src, tgt, proj, enc, make_episodes(a.episodes, a.names, a.colours, a.seed))
    res.update(source=a.source, target=a.target, projector=a.projector, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
