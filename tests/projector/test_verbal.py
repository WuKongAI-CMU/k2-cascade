from k2cascade.projector.verbal import attach


def test_attach_terciles():
    rows = [{"question": f"q{i}"} for i in range(6)]
    se = {i: {"i": i, "se": float(i), "greedy": f"a{i}"} for i in range(6)}
    out = attach(rows, se)
    assert [r["sender_conf"] for r in out] == ["high", "high", "high", "medium", "medium", "low"]
    assert out[0]["sender_answer"] == "a0"
