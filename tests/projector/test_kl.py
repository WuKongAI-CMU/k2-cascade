import torch

from k2cascade.projector.eval import _chunked_kl, continuation_hidden, oracle_cache


def test_kl_zero_for_identical_hidden_and_positive_otherwise(target):
    torch.manual_seed(0)
    ids = torch.randint(2, 30, (1, 16))
    h = continuation_hidden(target, ids[:, 8:], oracle_cache(target, ids[:, :8]), 8)
    d = h.shape[-1]
    assert _chunked_kl(target.lm_head, h.reshape(-1, d), h.reshape(-1, d), 4).abs().max() < 1e-5
    h2 = continuation_hidden(target, ids[:, 8:], None, 0)
    assert _chunked_kl(target.lm_head, h2.reshape(-1, d), h.reshape(-1, d), 4).mean() > 0
