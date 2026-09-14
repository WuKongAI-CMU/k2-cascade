# K2 Cascade — 2–3 minute clip for IFM's K2 Horizon Demo Series

Format: screen recording (QuickTime, 1920×1080), you on camera optional. Show `demo/slides-v2.html` full screen,
switch to a terminal for the live run, back to slides. IFM adds their own opening and closing; start on the first slide.
No AUROC numbers in this cut (results go to IFM later, in person). Target 2:30.

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

## 1:20–1:50 · Slides 4–5
On this task the small models carried the whole loop; the 375B was never needed for an action.
We then scored the 375B's steps token by token with the 3.7B: the surprise sits in the prose, not in the tool calls.
With more tasks, three tasks landed on three different rungs: one the 0.9B solved alone, one needed the 3.7B, one needed the 375B.

## 1:50–2:15 · Slide 6
What we're building next: read the small model's own state before it acts, so it knows when a step is beyond it,
and later hand that state to the larger model directly instead of re-reading text. Latent communication inside one family.

## 2:15–2:30 · Slide 8
Thanks to IFM for opening the whole ladder. Code, traces and results: github.com/WuKongAI-CMU/k2-cascade.

## Recording checklist
- Both local servers up (`curl localhost:8081/health`, `localhost:8082/health`), fresh /tmp/k2demo, terminal font 18+.
- Do the live run once before recording so the prompt cache is warm.
- Export 1080p, upload to Google Drive, share link with Jane; also send the LinkedIn URL and X handle.
