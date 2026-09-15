# K2 Cascade — 2–3 minute clip for IFM's K2 Horizon Demo Series

Format: screen recording (QuickTime, 1920×1080), you on camera optional. Show `demo/slides-v2.html` full screen,
switch to a terminal for the live run, back to slides. IFM adds their own opening and closing; start on the first slide.
No AUROC numbers in this cut (results go to IFM later, in person). Target 2:30.

(The deck is now 12 slides and doubles as a research pitch deck. For the 2.5-minute IFM cut use slides
1, 3, 4, then the terminal, then 5, 6, 7, 8, 12 — skip slide 2 (why now), slide 9 (what we refuse to claim)
and slide 11 (roadmap), which are for a research or investor audience. Slide order: 1 cover, 2 why now,
3 the question, 4 why K2 Horizon, 5 results, 6 traces, 7 the probe, 8 latent communication,
9 what we refuse to claim, 10 architecture, 11 roadmap, 12 close.)

## 0:00–0:20 · Slide 1
Hi, I'm Peter Qin from CMU. This is K2 Cascade, built at HackCMU on the K2 Horizon family.
Today every agent step goes to the biggest model, because no model can tell when a step is beyond it.
K2 Cascade is a first measurement of which steps actually need a big model.

## 0:20–0:40 · Slides 2–3
K2 Horizon gives six sizes trained with one recipe, one chat template, one tool-call format.
So we built a ladder: 0.9B and 3.7B run on this laptop, the 375B runs through the IFM API.
Each step goes to the smallest model first; a verifier checks it; on failure the next size takes over.
Every attempt is logged, so the trace file is a dataset of which size each step needed.

## 0:40–1:20 · Terminal, live run
```
cd ~/projects/k2-cascade && cp -R demo/todo-cli /tmp/k2demo && uv run python -m k2cascade.run --mode cascade --cwd /tmp/k2demo --task "$(sed -n '3,6p' demo/TASK.md)"
```
Narrate: two failing tests, implement two flags. The 0.9B runs the tests, reads the source, reads the tests.
At the code-writing step it's rejected and the 3.7B takes over. The 375B only answers yes or no.
(If a 429 appears, wait; the client retries.)

## 1:20–1:40 · Slides 4–5
Across four tasks and 37 runs, ninety percent of the accepted agent steps ran on the laptop.
Different tasks needed different rungs: one the 0.9B solved alone, one needed the 3.7B, one really needed the 375B.
And when we scored the 375B's steps token by token with the 3.7B, the surprise sat in the prose, not in the tool calls.

## 1:40–2:00 · Slide 6 (the result)
So we asked whether the small model already knows. We read its hidden state at the last prompt position, before it
writes anything, and fit a linear probe to predict whether the verifier will reject the step. It works at both sizes,
at almost every layer, and it beats scoring the output after the fact.

## 2:00–2:25 · Slides 7 and 9 (where this goes)
That signal is the piece nobody sends. Models today hand off to each other in English, or by copying a KV cache
that carries what the sender computed but not how sure the sender was. The 3.7B and the 7B share a key-value
shape and a tokenizer, so the cache can be mapped between them. Our next step is to make the probe's answer
part of what travels.

## 2:25–2:40 · Slide 10
Thanks to IFM for opening the whole ladder. Code, traces and results: github.com/WuKongAI-CMU/k2-cascade.

## Recording checklist
- Both local servers up (`curl localhost:8081/health`, `localhost:8082/health`), fresh /tmp/k2demo, terminal font 18+.
- Do the live run once before recording so the prompt cache is warm.
- Export 1080p, upload to Google Drive, share link with Jane; also send the LinkedIn URL and X handle.
