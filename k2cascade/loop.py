"""Agent loop with three modes: small, large, cascade."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import TOOLS, execute
from .trace import Trace
from .verify import check

SYSTEM = (
    "You are a coding agent working inside a project directory. Use the tools to inspect and change files "
    "and to run commands. The project directory is the current working directory: always use paths relative to it "
    "(for example todo.py, tests/test_todo.py). Take one small step at a time. When the task is fully done and verified, reply "
    "with a short summary and no tool call."
)


@dataclass
class Totals:
    steps: int = 0
    escalations: int = 0
    tokens: dict[str, int] = field(default_factory=dict)
    ms: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, usage) -> None:
        self.tokens[model] = self.tokens.get(model, 0) + usage.prompt_tokens + usage.completion_tokens
        self.ms[model] = self.ms.get(model, 0) + usage.latency_ms


def run(task: str, cwd: Path, *, ladder: list, mode: str = "cascade", trace_path: Path, max_steps: int = 30, retries: int = 1, judge: bool = True, log=print) -> dict[str, Any]:
    """ladder: models ordered smallest to largest. small = ladder[0] only, large = ladder[-1] only, cascade = try each in order."""
    small, large = ladder[0], ladder[-1]
    run_id = uuid.uuid4().hex[:8]
    trace = Trace(trace_path, {"run_id": run_id, "mode": mode, "task": task})
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    history_calls: list[Any] = []
    totals = Totals()
    done, final = False, ""
    sticky_level, sticky_left = 0, 0  # after an escalation, keep that level for STICKY more steps before handing back
    STICKY = 2

    for step_no in range(1, max_steps + 1):
        chosen = None
        attempts = []
        if mode == "large":
            order = [(large, i) for i in range(retries + 1)]
        elif mode == "small":
            order = [(small, i) for i in range(retries + 1)]
        else:
            start = sticky_level if sticky_left > 0 else 0
            order = [(m, i) for m in ladder[start:] for i in range((retries + 1) if m is not large else 1)]

        for model, attempt in order:
            parsed, usage, prompt = model.step(messages, TOOLS)
            totals.add(model.name, usage)
            ok, reason = check(parsed, history_calls)
            judge_raw = None
            if ok and judge and mode == "cascade" and model is not large and hasattr(large, "judge"):
                proposed = parsed.tool_calls or [{"final_answer": parsed.content[:300]}]
                yes, judge_raw, ju = large.judge(messages, proposed, task)
                totals.add(large.name + "-judge", ju)
                if not yes:
                    ok, reason = False, "judge_no"
            rec = {"step": step_no, "model": model.name, "attempt": attempt, "ok": ok, "reason": reason,
                   "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "latency_ms": usage.latency_ms,
                   "thinking": parsed.thinking, "content": parsed.content, "tool_calls": parsed.tool_calls, "raw": parsed.raw,
                   "judge": judge_raw, "escalated": mode == "cascade" and model is not small, "messages": messages if attempt == 0 and model is (order[0][0]) else None}
            attempts.append(rec)
            log(f"[{step_no}] {model.name} attempt {attempt}: {reason}  ({usage.completion_tokens} tok, {usage.latency_ms} ms)")
            if ok:
                chosen = (model, parsed)
                break
            if reason == "repeat_call" and mode == "cascade":
                # a repeated call means this level is looping; do not retry at the same level
                pass
        for rec in attempts:
            trace.write(**rec)
        if chosen is None:
            log(f"[{step_no}] no acceptable step; stopping")
            break
        model, parsed = chosen
        totals.steps += 1
        if mode == "cascade":
            level = ladder.index(model)
            if level > 0 and level != sticky_level:
                totals.escalations += 1
                sticky_level, sticky_left = level, STICKY
            elif level == sticky_level and sticky_left > 0:
                sticky_left -= 1
                if sticky_left == 0:
                    sticky_level = 0
            elif level > 0:
                totals.escalations += 1

        if not parsed.tool_calls:
            done, final = True, parsed.content
            messages.append({"role": "assistant", "content": parsed.content, "reasoning_content": parsed.thinking})
            log(f"[{step_no}] final: {parsed.content[:200]}")
            break

        messages.append({"role": "assistant", "content": parsed.content, "reasoning_content": parsed.thinking,
                         "tool_calls": [{"id": c.get("id") or f"call_{step_no}_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(parsed.tool_calls)]})
        history_calls.append([{"name": c["name"], "arguments": c["arguments"]} for c in parsed.tool_calls])
        outs = []
        for c in parsed.tool_calls:
            out = execute(c["name"], c["arguments"], cwd)
            outs.append(out)
            log(f"      -> {c['name']} {str(c['arguments'])[:100]}\n         {out[:160].rstrip()}")
        messages.append({"role": "tool", "content": "\n\n".join(outs), "outputs": outs})

    summary = {"run_id": run_id, "mode": mode, "done": done, "final": final, "steps": totals.steps,
               "escalations": totals.escalations, "tokens": totals.tokens, "ms": totals.ms}
    trace.write(step=0, summary=summary)
    return summary
