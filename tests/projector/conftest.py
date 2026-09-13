import pytest
import torch

from tests.projector.toy import VOCAB, tiny_k2


@pytest.fixture
def source():
    return tiny_k2(hidden=24, layers=2, groups=2, seed=1)  # like 3.7B: smaller hidden, same KV shape


@pytest.fixture
def target():
    return tiny_k2(hidden=32, layers=2, groups=4, seed=2)  # like 7B: bigger hidden, 4 norm groups


@pytest.fixture
def memorized():
    """A target trained to memorize one 48-token sequence, plus that sequence (prefix 32, cont 16)."""
    torch.manual_seed(3)
    seq = torch.randint(0, VOCAB, (1, 48))
    tgt = tiny_k2(hidden=32, layers=2, groups=4, seed=2)
    opt = torch.optim.Adam(tgt.parameters(), lr=3e-3)
    for _ in range(300):
        out = tgt(input_ids=seq[:, :-1], use_cache=False)
        loss = torch.nn.functional.cross_entropy(out.logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1))
        opt.zero_grad(), loss.backward(), opt.step()
    return tgt.eval().requires_grad_(False), seq
