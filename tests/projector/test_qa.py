import torch

from k2cascade.projector.qa import QAEncoder, prefix_cache, run, score_and_generate, f1, em
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
    st, _ = score_and_generate(target, ct, lt, q, ans, enc.bos, 4, nl)
    ss, _ = score_and_generate(target, cs, ls, q, ans, enc.bos, 4, nl)
    assert abs(st - ss) < 1e-3


def test_run_all_arms(source, target):
    out = run(source, target, identity_projector(target), QATok(), EX, max_new=3)
    for a in ("none", "text", "self", "raw", "project", "derange"):
        assert out[a]["logp"] <= 0.0 and 0.0 <= out[a]["f1"] <= 1.0
    assert "content_f1" in out and out["n"] == 3
