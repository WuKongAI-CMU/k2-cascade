# Surprise map: k2-375b outputs scored by 3.7b-mlx-8bit

- scorer: `/Users/peter/models/k2/3.7b-mlx-8bit` (mlx-lm 0.31.3, mlx 0.32.2), teacher forcing, prompts rendered with reasoning_effort=high
- k2-375b: 20 attempts, 2328 target tokens from traces/large-3.jsonl, traces/large-4.jsonl, traces/large-6.jsonl, traces/large-7.jsonl
- k2-3.7b self: 6 attempts, 731 target tokens from traces/smoke-3.7b.jsonl
- throughput: 599 tokens/s (34911 prompt+target tokens in 58.3 s, forward passes only)
- normalisation: literal think tags removed from the `thinking` field of 3 k2-375b attempts; whitespace around thinking/content kept as recorded (--strip-ws to strip)

## Negative log-prob per role (nats per token)

| role | n (k2-375b) | mean | median | n (k2-3.7b self) | mean | median | diff mean | diff median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| think | 56 | 1.436 | 0.255 | 108 | 0.426 | 0.032 | +1.009 | +0.223 |
| prose | 262 | 0.912 | 0.063 | 66 | 0.199 | 0.012 | +0.713 | +0.051 |
| markup | 302 | 0.170 | -0.000 | 93 | 0.079 | -0.000 | +0.091 | +0.000 |
| tool_name | 30 | 0.101 | 0.001 | 7 | 0.160 | 0.026 | -0.059 | -0.025 |
| arg_key | 23 | 0.100 | 0.000 | 7 | 0.061 | 0.000 | +0.039 | +0.000 |
| arg_path | 31 | 0.009 | 0.000 | 2 | 0.018 | 0.018 | -0.009 | -0.018 |
| arg_cmd | 76 | 0.167 | 0.000 | 62 | 0.121 | 0.000 | +0.046 | -0.000 |
| arg_content | 1548 | 0.012 | 0.000 | 386 | 0.033 | 0.000 | -0.020 | +0.000 |
| ALL | 2328 | 0.175 | 0.000 | 731 | 0.121 | 0.000 | +0.055 | -0.000 |

diff = k2-375b minus k2-3.7b self (positive: the scorer finds the k2-375b output more surprising).

## Top 15 most surprising tokens (k2-375b)

| nll | role | token | run/step | context (±5 tokens, target token in **bold**) |
|---:|---|---|---|---|
| 16.44 | think | `.\n` | 7e6f46f9/1 |  to see what's failing**.\n**</ifm\|think>\n<ifm\|tool_calls>\n<ifm\|tool_call> |
| 14.56 | think | `.\n` | ffa1ea3c/1 |  and see the failing tests**.\n**</ifm\|think>\n<ifm\|tool_calls>\n<ifm\|tool_call> |
| 13.78 | prose | `\n` | 0c7570fd/1 | </ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 13.78 | prose | `\n` | bafbe4d3/1 | </ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 12.82 | prose | `\n` | 7e6f46f9/1 |  what's failing.\n</ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 12.13 | prose | `\n` | ffa1ea3c/1 |  the failing tests.\n</ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 11.45 | prose | `\n` | 7e6f46f9/4 | </ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 9.64 | prose | `\n` | 0c7570fd/4 | </ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>shell\n |
| 7.87 | think | `.\n` | bafbe4d3/3 |  filter flags in list_items**.\n**</ifm\|think>\n<ifm\|tool_calls>\n<ifm\|tool_call> |
| 7.57 | prose | `\n` | ffa1ea3c/5 | </ifm\|think>**\n**All 4 tests pass |
| 7.11 | think | `.\n` | ffa1ea3c/3 | _items` based on args**.\n**</ifm\|think>\n<ifm\|tool_calls>\n<ifm\|tool_call> |
| 7.10 | markup | `</ifm\|think>` | 7e6f46f9/2 | **</ifm\|think>**<ifm\|tool_calls>\n<ifm\|tool_call>read_file |
| 7.05 | prose | `\n` | 0c7570fd/5 | </ifm\|think>**\n**All 4 tests pass |
| 7.00 | prose | `\n` | ffa1ea3c/3 |  based on args.\n</ifm\|think>**\n**<ifm\|tool_calls>\n<ifm\|tool_call>write_file |
| 6.83 | prose | `\n` | 7e6f46f9/5 | </ifm\|think>**\n**All 4 tests pass |

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
