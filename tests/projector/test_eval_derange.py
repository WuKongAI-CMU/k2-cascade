import torch

from k2cascade.projector.eval import evaluate
from tests.projector.test_noma import identity_projector


def test_evaluate_derange_arm_present_and_identity_matches_oracle(target):
    torch.manual_seed(0)
    batches = [torch.randint(2, 30, (1, 24)) for _ in range(3)]
    res = evaluate(target, target, identity_projector(target), batches, prefix_len=12, with_derange=True)
    assert abs(res["project"] - res["oracle"]) < 1e-3   # identity map on the same model == own prefill
    assert "derange" in res and "retention_derange" in res
    assert res["derange"] != res["project"]
