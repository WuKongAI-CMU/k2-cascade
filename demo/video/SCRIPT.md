# K2 Cascade — IFM demo clip, narration script

Read along with `k2-cascade-ifm-silent.mp4` to record your own voice. Each block starts at the time shown.

**0:00** — slide_01

Hi, I'm Peter Qin from Carnegie Mellon. This is K2 Cascade, built at HackCMU on the K2 Horizon family. Today, every agent step goes to the biggest model, because no model can tell when a step is beyond it. We think the next capability is a model that knows its own limits.

**0:18** — slide_03

So we asked a concrete question. In a coding agent, which steps actually need a three hundred and seventy five billion parameter model? We built a ladder. The point nine B and the three point seven B run on this laptop, and the three seventy five B runs through the IFM API.

**0:35** — slide_05

K2 Horizon makes this measurable. Six sizes share one training recipe, one chat template and one tool call format, so model size is close to the only variable. Each step goes to the smallest model first. A verifier checks it, and on failure, the next size takes over.

**0:53** — live run

Here is a live run on this laptop, sped up. The point nine B runs the tests, then goes looking for the source, at a path it made up. The judge rejects its next move twice, and the three point seven B takes over. It reads the file, writes the fix, and runs the tests. They pass, and control goes back to the point nine B to write the summary.

**1:18** — slide_06

Across four tasks and thirty seven runs, ninety percent of the accepted agent steps ran on the laptop. And different tasks needed different sizes. One the point nine B solved alone, one needed the three point seven B, and one really needed the three seventy five B.

**1:34** — slide_07

When we scored the big model's steps token by token with the three point seven B, its edge sat in its prose, not in its tool calls.

**1:42** — slide_08

So we asked whether the small model already knows. Before it writes anything, we read its hidden state, and fit a linear probe to predict whether the step will be rejected. On the three point seven B it reaches point eight eight, and it adds real information beyond simple bookkeeping, like the step count and earlier failures.

**2:01** — slide_09

That signal is exactly what models never pass to each other. They hand off in English, or copy a cache that carries what the sender computed, but not how sure it was. The three point seven B and the seven B share a key value layout and a tokenizer. So our next step is to pass that state directly, with the uncertainty travelling alongside it.

**2:22** — slide_15

Thank you to IFM for opening the whole ladder. The code, traces and results are on GitHub, at WuKong AI CMU, slash K2 Cascade.
