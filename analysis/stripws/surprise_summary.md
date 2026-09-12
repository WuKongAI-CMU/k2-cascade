# Surprise map: k2-375b outputs scored by 3.7b-mlx-8bit

- scorer: `/Users/peter/models/k2/3.7b-mlx-8bit` (mlx-lm 0.31.3, mlx 0.32.2), teacher forcing, prompts rendered with reasoning_effort=high
- k2-375b: 20 attempts, 2315 target tokens from traces/large-3.jsonl, traces/large-4.jsonl, traces/large-6.jsonl, traces/large-7.jsonl
- k2-3.7b self: 6 attempts, 731 target tokens from traces/smoke-3.7b.jsonl
- throughput: 591 tokens/s (34898 prompt+target tokens in 59.0 s, forward passes only)
- normalisation: literal think tags removed from the `thinking` field of 3 k2-375b attempts; whitespace around thinking/content stripped like parse.py does for local models

## Negative log-prob per role (nats per token)

| role | n (k2-375b) | mean | median | n (k2-3.7b self) | mean | median | diff mean | diff median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| think | 56 | 0.633 | 0.251 | 108 | 0.426 | 0.032 | +0.207 | +0.220 |
| prose | 249 | 0.484 | 0.050 | 66 | 0.199 | 0.012 | +0.285 | +0.037 |
| markup | 302 | 0.137 | -0.000 | 93 | 0.079 | -0.000 | +0.058 | +0.000 |
| tool_name | 30 | 0.102 | 0.001 | 7 | 0.160 | 0.026 | -0.058 | -0.025 |
| arg_key | 23 | 0.105 | 0.000 | 7 | 0.061 | 0.000 | +0.044 | -0.000 |
| arg_path | 31 | 0.009 | 0.000 | 2 | 0.018 | 0.018 | -0.010 | -0.018 |
| arg_cmd | 76 | 0.157 | 0.000 | 62 | 0.121 | 0.000 | +0.036 | -0.000 |
| arg_content | 1548 | 0.012 | 0.000 | 386 | 0.033 | 0.000 | -0.021 | +0.000 |
| ALL | 2315 | 0.101 | 0.000 | 731 | 0.121 | 0.000 | -0.020 | -0.000 |

diff = k2-375b minus k2-3.7b self (positive: the scorer finds the k2-375b output more surprising).

## Top 15 most surprising tokens (k2-375b)

| nll | role | token | run/step | context (±5 tokens, target token in **bold**) |
|---:|---|---|---|---|
| 7.10 | markup | `</ifm\|think>` | 7e6f46f9/2 | **</ifm\|think>**<ifm\|tool_calls>\n<ifm\|tool_call>read_file |
| 5.64 | markup | `</ifm\|think>` | 0c7570fd/1 | **</ifm\|think>**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 5.64 | markup | `</ifm\|think>` | bafbe4d3/1 | **</ifm\|think>**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 5.51 | prose | ` it` | ffa1ea3c/5 | --pending` is passed** it** returns only incomplete items, |
| 5.31 | prose | ` flags` | ffa1ea3c/5 |  only completed ones. The** flags** flow through from the CLI |
| 5.08 | think | `Implement` | bafbe4d3/3 | **Implement** the filter flags in list |
| 4.53 | prose | ` flow` | ffa1ea3c/5 |  completed ones. The flags** flow** through from the CLI's |
| 4.38 | think | `Simple` | ffa1ea3c/3 | **Simple** fix: filter in ` |
| 3.99 | markup | `</ifm\|think>` | 7e6f46f9/3 | **</ifm\|think>**<ifm\|tool_calls>\n<ifm\|tool_call>write_file |
| 3.80 | prose | ` incomplete` | ffa1ea3c/5 |  is passed it returns only** incomplete** items, and when ` |
| 3.72 | prose | ` filtering` | bafbe4d3/5 |  and `--done`** filtering** in `list_items()` |
| 3.72 | think | ` to` | ffa1ea3c/1 |  start by exploring the repository** to** understand the structure and see |
| 3.54 | arg_content | ` args` | bafbe4d3/3 |  items = load()\n   ** args** = args or []\n    |
| 3.49 | prose | ` no` | 7e6f46f9/5 |  only completed items, and** no** flags returns everything as before |
| 3.23 | arg_content | ` args` | ffa1ea3c/3 | _items(args=None):\n   ** args** = args or []\n    |

## Top 15 most surprising tokens (k2-3.7b self)

| nll | role | token | run/step | context (±5 tokens, target token in **bold**) |
|---:|---|---|---|---|
| 8.75 | arg_content | `]))` | eab118a0/4 | (sys.argv[1:**]))**</ifm\|arg_value>\n</ifm\|tool_call>\n</ifm\|tool_calls> |
| 4.06 | think | ` update` | eab118a0/3 |  done items. Let me** update** todo.py.</ifm\|think><ifm\|tool_calls> |
| 3.70 | think | ` failing` | eab118a0/4 | The** failing** tests expect `list_items |
| 3.53 | markup | `</ifm\|tool_calls>` | eab118a0/1 |  ls</ifm\|arg_value>\n</ifm\|tool_call>\n**</ifm\|tool_calls>**<\|ifm\|im_end\|> |
| 2.93 | think | ` without` | eab118a0/4 |  implement that in todo.py** without** modifying the tests.</ifm\|think> |
| 2.78 | think | `The` | eab118a0/4 | **The** failing tests expect `list |
| 2.52 | think | ` or` | eab118a0/4 | ["--pending"]`** or** `["--done"] |
| 2.42 | think | ` expect` | eab118a0/4 | The failing tests** expect** `list_items` to |
| 1.92 | arg_cmd | `find` | eab118a0/1 | <ifm\|arg_key>cmd</ifm\|arg_key>\n<ifm\|arg_value>**find** . -type f - |
| 1.74 | think | `["` | eab118a0/4 | _items` to accept `**["**--pending"]` or |
| 1.72 | think | ` Done` | eab118a0/6 |  4 tests pass.** Done**.</ifm\|think>All 4 |
| 1.68 | arg_cmd | `15` | eab118a0/3 | >&1 \| tail -**15**</ifm\|arg_value>\n</ifm\|tool_call>\n</ifm\|tool_calls> |
| 1.56 | prose | ` done` | eab118a0/6 | ` is passed, only** done** items are returned. No |
| 1.54 | prose | ` when` | eab118a0/6 | ()` in todo.py:** when** `--pending` is |
| 1.53 | think | ` I` | eab118a0/4 |  args and filter accordingly.** I**'ll implement that in todo |
