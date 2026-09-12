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
    assert '<ifm|tool_call>{"name": <function-name>' in p  # json format instructions
    assert p.endswith("<|ifm|im_start|>assistant\n<ifm|think_faster>\n")


def test_render_roundtrip_assistant_tool_turn():
    msgs = [
        {"role": "user", "content": "List files."},
        {"role": "assistant", "content": "", "reasoning_content": "need ls",
         "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": {"cmd": "ls"}}}]},
        {"role": "tool", "content": "a.py\nb.py"},
    ]
    p = render(msgs, TOOLS)
    assert '<ifm|tool_call>{"name": "shell", "arguments": {"cmd": "ls"}}</ifm|tool_call>' in p
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
    assert not out.ok and "bad_tool_call_json" in out.parse_errors[0]
