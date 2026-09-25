import numpy as np

from k2cascade.projector.diag_uncertainty import features, probe_auroc
from k2cascade.projector.qa import QAEncoder
from tests.projector.test_noma import identity_projector
from tests.projector.test_qa import EX, QATok


def test_features_and_probe(source, target):
    enc = QAEncoder(QATok())
    f = features(source, target, identity_projector(target), enc, EX[0], "clean", [1, 2])
    assert all(np.isfinite(v).all() for v in f.values()) and f["receiver"].shape == f["text"].shape
    rng = np.random.RandomState(0); y = (rng.rand(120) < 0.5).astype(int); x = rng.randn(120, 6); x[:, 1] += 2 * y
    assert probe_auroc(x, y) > 0.85
