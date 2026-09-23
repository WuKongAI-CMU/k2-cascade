"""Build the three passage variants for the confidence test (docs/design-confidence-2026-09-22.md).

  clean        original passage
  contradicted one fluent sentence asserting a counter-answer, inserted right after the gold sentence
  removed      the gold sentence deleted (unanswerable)

The counter-answer and the sentence come from the 375B judge (as in Adaptive Chameleon / ConflictBank); the
NLI filter runs later on the GPU box (confidence job). Runs on the laptop: only API calls, no GPU.

uv run python -m k2cascade.projector.variants --n 1000 --out data/squad_variants.jsonl
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def gold_sentence_index(sentences: list[str], answer_start: int) -> int:
    pos = 0
    for i, s in enumerate(sentences):
        end = pos + len(s)
        if answer_start < end + 1:
            return i
        pos = end + 1  # the split consumed at least one whitespace char
    return len(sentences) - 1


JUDGE_PROMPT = """You are helping build a reading-comprehension test. Given a passage, a question and its correct answer,
invent a plausible WRONG answer of the same type and format (a date for a date, a person for a person, a number of similar
magnitude for a number, a place for a place), then write ONE fluent sentence, in the passage's own style, that asserts this
wrong answer to the question as if it were a fact. The sentence must stand on its own, must not mention or deny the correct
answer, and must read naturally if placed right after the sentence that currently contains the correct answer.

Passage: {context}
Question: {question}
Correct answer: {answer}

Reply with JSON only, no explanation: {{"counter_answer": "...", "sentence": "..."}}"""


def ask_judge(client, context: str, question: str, answer: str, tries: int = 2) -> dict | None:
    """The judge sometimes writes its reasoning into `content`; allow room for it and retry a failed parse once."""
    from ..local import _post_with_retry
    q = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    import time
    d = None
    for attempt in range(tries):
        try:
            r = _post_with_retry(client.client, "/chat/completions", json={
                "model": client.model, "max_tokens": 300, "temperature": 0, "reasoning_effort": "low",
                "messages": [{"role": "system", "content": "Reply with a single JSON object and nothing else."},
                             {"role": "user", "content": q}]})
            body = r.json()
            txt = body["choices"][0]["message"]["content"] or ""
        except Exception as e:  # rate limit / server error bodies have no "choices"; back off and retry
            time.sleep(5 * (attempt + 1))
            continue
        blocks = re.findall(r"\{[^{}]*\}", txt, re.S)
        for b in reversed(blocks):
            try:
                d = json.loads(b); break
            except json.JSONDecodeError:
                continue
        if d:
            break
    if not d:
        return None
    if not d.get("counter_answer") or not d.get("sentence"):
        return None
    if answer.lower() in d["sentence"].lower() or d["counter_answer"].lower() == answer.lower():
        return None
    return {"counter_answer": d["counter_answer"].strip(), "sentence": d["sentence"].strip()}


def build_variants(context: str, answer_start: int, judged: dict) -> dict:
    sents = split_sentences(context)
    gi = gold_sentence_index(sents, answer_start)
    contradicted = " ".join(sents[: gi + 1] + [judged["sentence"]] + sents[gi + 1:])
    removed = " ".join(sents[:gi] + sents[gi + 1:])
    return {"clean": context, "contradicted": contradicted, "removed": removed, "gold_sentence": sents[gi]}


def build(examples: list[dict], judge_fn, workers: int = 8) -> list[dict]:
    """examples need context, question, answers, answer_start. judge_fn(context, question, answer) -> dict|None."""
    def one(ex):
        try:
            j = judge_fn(ex["context"], ex["question"], ex["answers"][0])
        except Exception:
            return None
        if j is None:
            return None
        v = build_variants(ex["context"], ex["answer_start"], j)
        return {**ex, **j, **v}
    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, examples))
    return [o for o in out if o]


def load_squad_with_offsets(n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation").shuffle(seed=seed).select(range(n))
    return [{"id": r["id"], "context": r["context"], "question": r["question"],
             "answers": list(r["answers"]["text"]), "answer_start": int(r["answers"]["answer_start"][0])} for r in ds]


def main(argv=None) -> None:
    import argparse
    from ..cloud import CloudK2
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/squad_variants.jsonl"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    import os
    client = CloudK2()
    ex = load_squad_with_offsets(a.n, a.seed)
    done = set()
    if os.path.exists(a.out):  # resume: skip items already written
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
    todo = [e for e in ex if e["id"] not in done]
    written = len(done)
    with open(a.out, "a") as f:
        for i in range(0, len(todo), 40):  # small chunks so progress lands on disk as it goes
            rows = build(todo[i:i + 40], lambda c, q, ans: ask_judge(client, c, q, ans), a.workers)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush(); written += len(rows)
            print(f"{written} written, {min(i + 40, len(todo))}/{len(todo)} attempted", flush=True)
    print(f"{written}/{len(ex)} items with variants -> {a.out}")


if __name__ == "__main__":
    main()
