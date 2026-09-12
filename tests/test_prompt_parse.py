from k2cascade.parse import parse
from k2cascade.prompt import render

TOOLS = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command in the repo.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    },
}]


def test_render_has_tools_and_generation_prompt():
    p = render([{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "List files."}], TOOLS)
    assert p.startswith("<|ifm|begin_of_text|>")
    assert "# Tools" in p and "shell" in p
    assert "<ifm|tool_calls>" in p
    assert p.endswith("<|ifm|im_start|>assistant\n<ifm|think_faster>\n")


def test_render_roundtrip_assistant_tool_turn():
    msgs = [
        {"role": "user", "content": "List files."},
        {"role": "assistant", "content": "", "reasoning_content": "need ls",
         "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": {"cmd": "ls"}}}]},
        {"role": "tool", "content": "a.py\nb.py"},
    ]
    p = render(msgs, TOOLS)
    assert "<ifm|tool_call>shell\n<ifm|arg_key>cmd</ifm|arg_key>\n<ifm|arg_value>ls</ifm|arg_value>\n</ifm|tool_call>" in p
    assert "<|ifm|im_start|>tool\na.py\nb.py<|ifm|im_end|>" in p


def test_parse_tool_call():
    raw = 'I should list.\n</ifm|think_faster>Let me look.<ifm|tool_calls>\n<ifm|tool_call>{"name": "shell", "arguments": {"cmd": "ls"}}</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>'
    out = parse(raw)
    assert out.ok
    assert out.thinking == "I should list."
    assert out.content == "Let me look."
    assert out.tool_calls == [{"name": "shell", "arguments": {"cmd": "ls"}}]


def test_parse_final_answer():
    out = parse("done thinking\n</ifm|think>All good.<|ifm|im_end|>")
    assert out.ok and out.tool_calls == [] and out.content == "All good."


def test_parse_bad_json_is_flagged():
    out = parse('x</ifm|think><ifm|tool_calls><ifm|tool_call>{"name": "shell", "arguments": {cmd: ls}}</ifm|tool_call></ifm|tool_calls>')
    assert not out.ok and "bad_tool_call" in out.parse_errors[0]


def test_parse_xml_tool_call_as_emitted_by_0_9b():
    raw = ("Let me run tests.\n</ifm|think>\n<ifm|tool_calls>\n<ifm|tool_call>shell\n<ifm|arg_key>cmd</ifm|arg_key>\n"
           "<ifm|arg_value>python -m pytest -q</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>")
    out = parse(raw)
    assert out.ok, out.parse_errors
    assert out.tool_calls == [{"name": "shell", "arguments": {"cmd": "python -m pytest -q"}}]


def test_parse_xml_write_file_multiline_content():
    raw = ("</ifm|think><ifm|tool_calls><ifm|tool_call>write_file\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>a.py</ifm|arg_value>\n"
           "<ifm|arg_key>content</ifm|arg_key>\n<ifm|arg_value>x = 1\nprint(x)\n</ifm|arg_value>\n</ifm|tool_call></ifm|tool_calls>")
    out = parse(raw)
    assert out.ok and out.tool_calls[0]["arguments"] == {"path": "a.py", "content": "x = 1\nprint(x)"}


def test_render_default_is_xml_format():
    p = render([{"role": "user", "content": "hi"}], TOOLS)
    assert "<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>" in p
