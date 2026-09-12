import json
from pathlib import Path

from k2cascade.parse import parse
from k2cascade.surprise import build_target, clean_thinking, load_attempts, per_role, role_at

CALLS = [{"name": "write_file", "arguments": {"path": "a.py", "content": "x = 1\nprint(x)"}},
         {"name": "shell", "arguments": {"cmd": "ls", "opts": ["-l"]}}]


def test_build_target_matches_native_xml_and_roundtrips_through_parse():
    seg = build_target("plan\n", "\n", CALLS)
    text = "".join(t for t, _ in seg)
    assert text == (
        "plan\n</ifm|think>\n<ifm|tool_calls>"
        "\n<ifm|tool_call>write_file\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>a.py</ifm|arg_value>\n"
        "<ifm|arg_key>content</ifm|arg_key>\n<ifm|arg_value>x = 1\nprint(x)</ifm|arg_value>\n</ifm|tool_call>"
        "\n<ifm|tool_call>shell\n<ifm|arg_key>cmd</ifm|arg_key>\n<ifm|arg_value>ls</ifm|arg_value>\n"
        "<ifm|arg_key>opts</ifm|arg_key>\n<ifm|arg_value>[\"-l\"]</ifm|arg_value>\n</ifm|tool_call>"
        "\n</ifm|tool_calls><|ifm|im_end|>"
    )
    out = parse(text)
    assert out.ok and out.thinking == "plan" and out.content == ""
    assert out.tool_calls == CALLS


def test_clean_thinking_drops_leaked_tags_only():
    assert clean_thinking("</ifm|think>\n") == ("", True)
    assert clean_thinking("<ifm|think>\n</ifm|think>\n") == ("", True)
    assert clean_thinking("plan\n") == ("plan\n", False)


def test_build_target_empty_thinking_and_final_answer():
    assert "".join(t for t, _ in build_target("", "Done.", [])) == "</ifm|think>Done.<|ifm|im_end|>"


def test_roles_by_char_span():
    seg = build_target("plan", "Hello", CALLS)
    text = "".join(t for t, _ in seg)
    assert role_at(seg, 0, 2) == "think"
    assert role_at(seg, text.index("</ifm|think>"), text.index("</ifm|think>") + 3) == "markup"
    assert role_at(seg, text.index("Hello"), text.index("Hello") + 5) == "prose"
    assert role_at(seg, text.index("write_file"), text.index("write_file") + 5) == "tool_name"
    assert role_at(seg, text.index("path</ifm"), text.index("path</ifm") + 4) == "arg_key"
    assert role_at(seg, text.index("a.py"), text.index("a.py") + 4) == "arg_path"
    assert role_at(seg, text.index("x = 1"), text.index("x = 1") + 1) == "arg_content"
    assert role_at(seg, text.index("ls<"), text.index("ls<") + 2) == "arg_cmd"
    assert role_at(seg, text.index('["-l"]'), text.index('["-l"]') + 6) == "arg_value"
    # a token straddling two segments takes the one with more overlap
    i = text.index("a.py</ifm")
    assert role_at(seg, i + 3, i + 3 + 5) == "markup"


def test_load_attempts_reuses_attempt0_messages_and_skips_summaries(tmp_path: Path):
    msgs = [{"role": "user", "content": "hi"}]
    rows = [
        {"run_id": "r", "step": 1, "attempt": 0, "model": "k2-3.7b", "ok": False, "messages": msgs},
        {"run_id": "r", "step": 1, "attempt": 1, "model": "k2-3.7b", "ok": True, "messages": None},
        {"run_id": "r", "step": 2, "attempt": 0, "model": "k2-375b", "ok": True, "messages": msgs},
        {"run_id": "r", "step": 0, "summary": {}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = load_attempts([p], "k2-3.7b", ok_only=True)
    assert [(r["step"], r["attempt"]) for r, _ in got] == [(1, 1)] and got[0][1] == msgs
    assert [r["step"] for r, _ in load_attempts([p], "k2-375b", ok_only=False)] == [2]


def test_per_role_stats():
    rows = [{"role": "think", "logprob": -1.0}, {"role": "think", "logprob": -3.0}, {"role": "markup", "logprob": -0.5}]
    st = per_role(rows)
    assert st["think"] == {"n": 2, "mean": 2.0, "median": 2.0}
    assert st["ALL"]["n"] == 3 and abs(st["ALL"]["mean"] - 1.5) < 1e-9
