import torch

from k2cascade.projector.qa import QAEncoder, prefix_cache, run, score_item, f1, em
from tests.projector.test_noma import FakeTok, identity_projector


class QATok(FakeTok):
    eos_token_id = None

    def decode(self, ids):
        inv = {v: k for k, v in self.ids.items()}
        return " ".join(inv.get(i, "?") for i in ids)


EX = [{"context": "The river runs north past the old mill.", "question": "Where does the river run?", "answers": ["north"]},
      {"context": "A green door faces the square.", "question": "What colour is the door?", "answers": ["green"]},
      {"context": "The bakery opens at dawn every day.", "question": "When does the bakery open?", "answers": ["at dawn"]}]


def test_metrics():
    assert em("The North", ["north"]) == 1.0 and f1("green door", ["green"]) > 0.6 and f1("x", ["y"]) == 0.0


def test_self_equals_text_on_gold_logprob(target):
    tok = QATok(); enc = QAEncoder(tok)
    pa = torch.tensor([enc.passage(EX[0]["context"])]); q = torch.tensor([enc.question(EX[0]["question"])])
    ans = torch.tensor([enc.answer("north")])
    nl = set(tok("\n", add_special_tokens=False)["input_ids"])
    ct, lt = prefix_cache("text", target, target, None, pa, pa)
    cs, ls = prefix_cache("self", target, target, None, pa, pa)
    st = score_item(target, ct, lt, q, ans, enc.bos, 4, nl)
    ss = score_item(target, cs, ls, q, ans, enc.bos, 4, nl)
    assert abs(st["logp"] - ss["logp"]) < 1e-3 and abs(st["entropy"] - ss["entropy"]) < 1e-3


def test_run_all_arms_and_controls(source, target, tmp_path):
    import io, json
    buf = io.StringIO()
    arms = ("none", "text", "self", "raw", "project", "derange", "zero", "random")
    ex = [dict(e, counter_answer="south") for e in EX]
    out = run(source, target, identity_projector(target), QATok(), ex, arms=arms, max_new=3, per_item=buf)
    for a in arms:
        assert out[a]["logp"] <= 0.0 and 0.0 <= out[a]["f1"] <= 1.0 and out[a]["entropy"] >= 0.0
    assert out["text"]["jsd_text"] < 1e-6 and out["zero"]["jsd_text"] >= 0.0
    assert "content_f1" in out and out["n"] == 3 and "p_counter" in out["project"]
    rows = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert len(rows) == 3 and set(arms) <= set(rows[0])


def test_verbal_arm_uses_sender_answer_without_passage(source, target):
    ex = [dict(e, sender_answer=e["answers"][0], sender_conf="high") for e in EX]
    out = run(source, target, identity_projector(target), QATok(), ex, arms=("none", "verbal", "text"), max_new=3)
    assert "verbal" in out and out["verbal"]["logp"] <= 0.0
