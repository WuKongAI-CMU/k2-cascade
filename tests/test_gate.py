import numpy as np

from k2cascade.gate import evaluate, features, raw_segments, strip_stop
from k2cascade.surprise import role_at

RAW_FASTER = ("</ifm|think_faster>\n<ifm|tool_calls>\n<ifm|tool_call>read_file\n<ifm|arg_key>path</ifm|arg_key>\n"
              "<ifm|arg_value>todo.py</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>")


def _roles(seg):
    return [(t, r) for t, r in seg]


def test_raw_segments_think_faster_tool_call():
    seg = raw_segments(RAW_FASTER)
    assert "".join(t for t, _ in seg) == strip_stop(RAW_FASTER)
    assert seg[0] == ("</ifm|think_faster>", "markup")  # empty thinking: no think segment
    assert ("read_file", "tool_name") in seg
    assert ("path", "arg_key") in seg
    assert ("todo.py", "arg_path") in seg
    assert all(r == "markup" for t, r in seg if t.startswith("<ifm|") or t.startswith("</ifm|"))


def test_raw_segments_think_close_variant_with_thinking_and_prose():
    raw = ("Plan: run tests</ifm|think>\nRunning now.<ifm|tool_calls>\n<ifm|tool_call>shell\n<ifm|arg_key>cmd</ifm|arg_key>\n"
           "<ifm|arg_value>pytest -q</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>")
    seg = raw_segments(raw)
    assert seg[0] == ("Plan: run tests", "think")
    assert seg[1] == ("</ifm|think>", "markup")
    assert seg[2] == ("\nRunning now.", "prose")
    assert ("shell", "tool_name") in seg and ("cmd", "arg_key") in seg and ("pytest -q", "arg_cmd") in seg
    assert "".join(t for t, _ in seg) == raw
    text = raw
    assert role_at(seg, text.index("pytest"), text.index("pytest") + 3) == "arg_cmd"


def test_raw_segments_unclosed_think_and_final_answer():
    assert raw_segments("I never close the block") == [("I never close the block", "think")]
    seg = raw_segments("</ifm|think_faster>\nAll tests pass.<|ifm|im_end|>")
    assert seg == [("</ifm|think_faster>", "markup"), ("\nAll tests pass.", "prose")]
    assert raw_segments("") == []


def test_features_split_action_and_think():
    nll = np.array([1.0, 3.0, 0.5, 0.1, 2.0])
    roles = ["think", "think", "markup", "tool_name", "arg_path"]
    f = features(nll, roles)
    assert f["mean"] == 6.6 / 5 and f["max"] == 3.0 and f["mean_think"] == 2.0
    assert f["mean_action"] == 1.05 and f["mean_markup"] == 0.5 and np.isnan(f["mean_prose"])
    assert f["n_tokens"] == 5 and f["sum"] == 6.6


def test_evaluate_picks_best_feature_and_drops_nan():
    rows = []
    for i in range(20):
        ok = i % 2 == 0
        rows.append({"ok": ok, "features": {"mean": 0.0 if ok else 1.0, "max": float(i), "mean_action": 1.0 if ok else 0.0,
                                            "mean_think": float("nan") if i < 10 else (0.0 if ok else 1.0), "mean_prose": float("nan"),
                                            "mean_markup": 0.5, "sum": 0.0, "n_tokens": 1.0}})
    out = evaluate(rows, n_boot=50)
    assert out["n"] == 20 and out["n_rejected"] == 10
    assert out["features"]["mean"]["auroc"] == 1.0 and out["features"]["mean_action"]["auroc"] == 0.0
    assert out["features"]["mean_action"]["auroc_flipped"] == 1.0
    assert out["features"]["mean_think"]["n"] == 10 and out["features"]["mean_think"]["auroc"] == 1.0
    assert np.isnan(out["features"]["mean_prose"]["auroc"])
    # mean_think is defined on only 10 of 20 attempts, so it cannot be "best"; the two full-coverage features tie on |AUROC - 0.5|
    assert out["best_feature"] in ("mean", "mean_action")
    assert out["best_sign"] == ("high_surprise_rejects" if out["best_feature"] == "mean" else "low_surprise_rejects")
