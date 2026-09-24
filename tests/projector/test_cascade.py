import numpy as np

from k2cascade.projector.cascade import sender_features
from k2cascade.projector.qa import QAEncoder
from tests.projector.test_qa import EX, QATok


def test_sender_features_shapes(source):
    tok = QATok(); enc = QAEncoder(tok)
    r = sender_features(source, tok, enc, EX[0], layers=[1, 2], k=3, max_new=3, nli=None, seed=0)
    d = source.config.hidden_size
    assert r["feats"].shape == (2 * d,) and 0 <= r["p_max"] <= 1 and r["entropy"] >= 0 and 0 <= r["agree"] <= 1
    assert isinstance(r["greedy"], str) and 0 <= r["f1"] <= 1
