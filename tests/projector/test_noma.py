import torch

from k2cascade.projector.extract import kv_geometry
from k2cascade.projector.noma import Encoder, arm_logprobs, make_episodes, run
from k2cascade.projector.ridge import RidgeProjector, last_aligned_map
from tests.projector.toy import VOCAB


class FakeTok:
    """Word-level tokenizer into the tiny vocab: every whitespace/punctuation-split piece is one id."""
    bos_token_id = 1

    def __init__(self):
        self.ids = {}

    def _id(self, w):
        if w not in self.ids:
            self.ids[w] = 2 + len(self.ids) % (VOCAB - 2)
        return self.ids[w]

    def __call__(self, text, add_special_tokens=False):
        for p in ".?\n":
            text = text.replace(p, f" {p} ")
        return {"input_ids": [self._id(w) for w in text.split()]}


def identity_projector(model):
    n_kv, d = kv_geometry(model.config)
    L = model.config.num_hidden_layers
    p = RidgeProjector(last_aligned_map(L, L), n_kv, 2 * d, 2 * d, n_kv, d)
    p.W[:] = torch.eye(2 * d)
    return p


def test_episodes_partner_same_layout_different_answer():
    eps = make_episodes(50, n_names=4, n_colours=5, seed=0)
    for i, e in enumerate(eps):
        o = eps[e.partner]
        assert e.partner != i and len(o.colours) == len(e.colours)
        assert o.colours[e.query] != e.answer


def test_self_injection_matches_reading_the_text(target):
    enc = Encoder(FakeTok(), 3, 4)
    e = make_episodes(1, 3, 4, seed=1)[0]
    f, q = torch.tensor([enc.facts(e)]), torch.tensor([enc.question(e)])
    text = arm_logprobs("text", target, target, None, f, f, q)
    self_ = arm_logprobs("self", target, target, None, f, f, q)
    assert torch.allclose(text, self_, atol=1e-4)


def test_identity_projector_reproduces_text(target):
    enc = Encoder(FakeTok(), 3, 4)
    e = make_episodes(1, 3, 4, seed=2)[0]
    f, q = torch.tensor([enc.facts(e)]), torch.tensor([enc.question(e)])
    proj = identity_projector(target)
    assert torch.allclose(arm_logprobs("project", target, target, proj, f, f, q),
                          arm_logprobs("text", target, target, None, f, f, q), atol=1e-4)


def test_run_reports_all_arms(source, target):
    enc = Encoder(FakeTok(), 3, 4)
    eps = make_episodes(12, 3, 4, seed=3)
    out = run(source, target, identity_projector(target), enc, eps)
    for a in ("none", "text", "self", "raw", "project", "derange"):
        assert 0.0 <= out[a]["acc"] <= 1.0 and out[a]["logp_true"] <= 0.0
    assert out["n"] == 12 and 0.0 <= out["derange"]["follow_rate"] <= 1.0
    assert "content_transfer" in out and "retention" in out
