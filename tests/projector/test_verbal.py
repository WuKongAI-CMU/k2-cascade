from k2cascade.projector.verbal import attach


def test_attach_terciles():
    rows = [{"question": f"q{i}"} for i in range(6)]
    se = {i: {"i": i, "se": float(i), "greedy": f"a{i}"} for i in range(6)}
    out = attach(rows, se)
    assert [r["sender_conf"] for r in out] == ["high", "high", "high", "medium", "medium", "low"]
    assert out[0]["sender_answer"] == "a0"


def test_attach_numeric_uses_agreement_with_greedy():
    rows = [{"question": "q"}]
    se = {0: {"i": 0, "se": 0.5, "greedy": "a", "labels": [0, 0, 1, 0, 0]}}  # 4 samples, 3 agree with greedy (label 0)
    assert attach(rows, se, "numeric")[0]["sender_conf"] == "0.8"
