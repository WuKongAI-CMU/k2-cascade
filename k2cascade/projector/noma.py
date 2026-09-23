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

NAMES = ["Noma", "Vela", "Tarn", "Quill", "Brisa", "Oskel", "Mirel", "Dunra",
         "Fenwick", "Lisbet", "Corvin", "Adaline", "Thurlow", "Isolde", "Percival", "Marisol"]
COLOURS = ["amber", "indigo", "crimson", "teal", "violet", "olive", "coral", "silver"]
# attribute type -> (value pool, fact template, question, answer stem). Values must be single tokens.
ATTRS = {
    "colour": (COLOURS, "{n} is {v}.", "What colour is {n}?", "{n} is"),
    "city": (["Paris", "Rome", "Tokyo", "Berlin", "London", "Madrid", "Cairo", "Lima"],
             "{n} lives in {v}.", "Where does {n} live?", "{n} lives in"),
    "animal": (["dog", "cat", "fox", "owl", "bear", "wolf", "deer", "frog"],
               "{n} is a {v}.", "What animal is {n}?", "{n} is a"),
}
# neutral filler that mentions no name or value; repeated to pad the facts to a target length
FILLER = ("The afternoon light moved slowly across the wooden floor of the reading room. Outside, a delivery van "
          "idled at the corner while its driver checked an address on a folded sheet of paper. The library "
          "had been built in 1911 and extended twice since, most recently with a glass wing that faced the park. "
          "A notice near the door announced revised opening hours for the autumn, and a second notice, older and "
          "slightly torn, advertised a lecture series on regional geology that had ended the previous spring. ")
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
    """Turns an episode into token ids. `facts` (with BOS) goes to the sender; `question` ends right before the value.
    `attr` picks the attribute type; `pad` appends neutral filler after the facts up to `pad` tokens, so the
    queried fact sits further from the question (every episode gets the same filler: layouts stay identical)."""

    def __init__(self, tokenizer, n_names: int, n_colours: int, attr: str = "colour", pad: int = 0):
        pool, self.fact_t, self.q_t, self.stem_t = ATTRS[attr]
        self.tok, self.names, self.colours, self.pad = tokenizer, NAMES[:n_names], pool[:n_colours], pad
        self.colour_ids = []
        for c in self.colours:
            ids = tokenizer(" " + c, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"value {c!r} is not a single token: {ids}")
            self.colour_ids.append(ids[0])
        self.bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.filler = self.tok(" " + FILLER * 8, add_special_tokens=False)["input_ids"] if pad else []

    def facts(self, e: Episode) -> list[int]:
        text = " ".join(self.fact_t.format(n=n, v=self.colours[c]) for n, c in zip(self.names, e.colours))
        ids = self.bos + self.tok(text, add_special_tokens=False)["input_ids"]
        if self.pad and len(ids) < self.pad:
            ids = ids + self.filler[:self.pad - len(ids)]
        return ids

    def question(self, e: Episode) -> list[int]:
        n = self.names[e.query]
        q = f"\nQuestion: {self.q_t.format(n=n)}\nAnswer: {self.stem_t.format(n=n)}"
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
                 facts: torch.Tensor, partner_facts: torch.Tensor, question: torch.Tensor,
                 layers: set[int] | None = None) -> torch.Tensor:
    """Full-vocabulary next-token log-probs for one arm. `facts`/`partner_facts` are (1, P) with identical P.
    `layers` restricts the `project` arm: only those receiver layers get this episode's projected cache, the
    rest get the partner's (so any content can only have entered through the listed layers)."""
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
        if arm == "project" and layers is not None:
            pk, pv = projector(extract(src, partner_facts, with_hidden=False))
            keys = [k if j in layers else o for j, (k, o) in enumerate(zip(keys, pk))]
            values = [v if j in layers else o for j, (v, o) in enumerate(zip(values, pv))]
        return next_token_logprobs(tgt, question, _cache_from_bundle(tgt, keys, values), p)
    raise ValueError(arm)


def run(src, tgt, projector, enc: Encoder, episodes: list[Episode], arms=ARMS, layers: set[int] | None = None) -> dict:
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
            lp = arm_logprobs(a, src, tgt, projector, f, pf, q, layers)[cid]
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
    out.update(n=n, chance=1 / len(enc.colour_ids), facts_tokens=len(enc.facts(episodes[0])),
               layers=sorted(layers) if layers is not None else None)
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
    ap.add_argument("--attr", default="colour", choices=list(ATTRS)); ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--layers", default=None, help="receiver layers that get this episode's cache, e.g. 0-11 or 3,7")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--compress", default=None, help="message compression spec, see compress.py (e.g. int8,layers=18-35)")
    a = ap.parse_args(argv)
    layers = None
    if a.layers:
        layers = set()
        for part in a.layers.split(","):
            lo, _, hi = part.partition("-")
            layers.update(range(int(lo), int(hi or lo) + 1))
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
    if a.compress and proj is not None:
        from .compress import Compressed
        proj = Compressed(proj, a.compress)
    enc = Encoder(tok, a.names, a.colours, a.attr, a.pad)
    res = run(src, tgt, proj, enc, make_episodes(a.episodes, a.names, a.colours, a.seed), tuple(a.arms.split(",")), layers)
    res.update(source=a.source, target=a.target, projector=a.projector, seed=a.seed, attr=a.attr, pad=a.pad,
               compress=getattr(proj, "info", None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
