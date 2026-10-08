import torch

from k2cascade.projector.lens import blend, onset, record_episodes, squad_cases
from k2cascade.projector.noma import Encoder, make_episodes
from tests.projector.test_noma import FakeTok, identity_projector
from tests.projector.test_qa import EX, QATok


def test_record_blend_onset(source, target):
    enc = Encoder(FakeTok(), 3, 4)
    eps = make_episodes(6, 3, 4, seed=1)
    proj = identity_projector(target)
    rows = record_episodes(target, target, proj, enc, eps, n_lens=2)
    L = target.config.num_hidden_layers
    assert len(rows) == 6 and len(rows[0]["lens_project"]) == L and len(rows[0]["lens_project"][0]) == 4
    assert abs(sum(rows[0]["project"]) - 1) < 1e-4 and "lens_text" not in rows[3]
    # identity map on the same model: the mapped memory equals the text path
    assert max(abs(x - y) for x, y in zip(rows[0]["project"], rows[0]["text"])) < 1e-3
    b = blend(target, target, proj, enc, eps, [0.0, 1.0])
    # alpha 0 is this episode's memory, alpha 1 the partner's: they must equal the project / derange arms
    p_own_project = sum(r["project"][r["answer"]] for r in rows) / len(rows)
    p_own_derange = sum(r["derange"][r["answer"]] for r in rows) / len(rows)
    assert abs(b[0]["p_own"] - p_own_project) < 1e-4 and abs(b[1]["p_own"] - p_own_derange) < 1e-4
    o = onset(target, target, proj, enc, eps)
    assert len(o["from"]) == L + 1 and o["from"][0]["acc"] == b[0]["acc"] and o["until"][0]["acc"] == b[1]["acc"]


def test_squad_cases(source, target):
    ex = [dict(e, clean=e["context"], removed=e["context"], sender_answer=e["answers"][0], sender_conf="high",
               sender_cands=[[e["answers"][0], 1.0]]) for e in EX]
    out = squad_cases(source, target, identity_projector(target), QATok(), ex, k_top=3)
    v = out[0]["variants"]["clean"]
    assert set(v) >= {"passage", "text", "memory", "wrong_memory", "word", "question_only"} and len(v["memory"]["top"]) == 3


def test_surgery_and_heads(source, target):
    from k2cascade.projector.lens import fact_positions, head_patch, surgery
    enc = Encoder(FakeTok(), 3, 4)
    eps = make_episodes(5, 3, 4, seed=2)
    vpos, npos = fact_positions(enc, eps[0])
    assert len(vpos) == 3 and all(n == v - 2 for v, n in zip(vpos, npos))
    s = surgery(target, target, identity_projector(target), enc, eps)
    assert set(s) == {"value", "name", "other_value", "value_and_name"} and 0 <= s["value"]["p_own"] <= 1
    h = head_patch(target, target, identity_projector(target), enc, eps, 2)
    assert len(h["drop"]) == target.config.num_hidden_layers and len(h["drop"][0]) == target.config.num_key_value_heads
