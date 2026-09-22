# Cross-Size KV Handoff: A Small Model's Cache Can Be Read by a Larger One

*Draft for CoRL 2026 EEAI workshop (4 pages, non-archival). Numbers from runs program-20260922b and
design2-20260922-022708; brackets mark results still running.*

## Abstract

Agents built from several language models hand work to each other through text, which drops everything the
sender computed while reading. We ask whether a smaller model's key–value (KV) cache can be handed directly to a
larger model of the same family. Between K2-Horizon-3.7B and K2-Horizon-7B, a per-head linear map fit in closed
form recovers almost nothing (22% on a binding test, chance 12.5%), but a gated residual on top of it, trained
for 1,500 steps on the larger model's continuation loss with both models frozen, lets the 7B answer 63–73% of
questions about facts only the 3.7B read. Three controls separate content from compute: a cache from the wrong
episode drops accuracy to 5% and the 7B follows the wrong cache 64% of the time; a projector trained with a
deranged sender learns a soft prompt that recovers 77% of the continuation-loss gap yet answers at 23%; and
injecting the cache into only the first half of the receiver's layers gives chance, while the second half alone
gives 52%. Transfer does not degrade when the fact is 512 tokens away from the question and holds across three
attribute types. We release the maps and the test.

## 1. Introduction

Edge–cloud agent stacks run a small model on device and escalate to a large one when the small model is
unsure. Escalation today means re-sending the transcript: the large model re-reads everything and the small
model's work is discarded. If the large model could instead continue from the small model's internal state, the
handoff would carry what the small model had already computed — including, in the longer run, how sure it was.

This paper asks the narrowest version of that question. Given two frozen models of different size that share a
tokenizer and KV geometry (K2-Horizon-3.7B and 7B: 36 layers, 8 KV heads, head dimension 128), can a learned map
from the small model's cache to the large model's cache carry a *specific fact* that only the small model saw?
We call this binding transfer, and we test it with a control for the standard objection that "latent gains" are
extra compute rather than communication.

Contributions. (1) A binding-transfer test (Noma) with six arms that isolates content from shape and compute.
(2) A two-stage projector — closed-form ridge over correlation-selected source layers, plus a gated residual
trained on the receiver's continuation loss — that reaches 63–73% binding transfer where the ridge alone reaches
22%. (3) Three controls that bound what the receiver actually gets from the cache: a deranged-episode cache, a
deranged-sender projector, and layer-band injection.

## 2. Setup

**Models.** Sender K2-Horizon-3.7B, receiver K2-Horizon-7B, both frozen, bf16, RoPE θ = 10⁷. We capture the
sender's pre-RoPE K and raw V at every layer, map them, apply RoPE at the receiver, and insert the result as the
receiver's cache for the prefix positions. The receiver then reads only the question.

**Projector.** Stage 1: for each receiver layer ℓ and KV head h, a ridge regression from the concatenated
[K;V] of the sender (256-d per head) to the receiver's [K;V]; source layers are the three sender layers whose
K/V correlate best with receiver layer ℓ ("top-3"), which beat the aligned-layer map in a first pass. Stage 2: a
gated SwiGLU residual (bottleneck 128, ~150M parameters) on top of the ridge output, trained with AdamW for 1,500
steps, effective batch 16 sequences of 1,024 FineWeb-Edu tokens, prefix 512, on the receiver's cross-entropy
over the continuation. Nothing in training mentions the test.

**Noma binding test.** Each episode assigns one of eight single-token values to eight nonce names ("Noma is
amber. Vela is indigo. …"). Only the sender reads the facts. The receiver is asked "What colour is Vela?" and
we score the eight values. Arms: *none* (question only), *text* (receiver reads the facts itself), *self*
(receiver's own cache re-injected; equals *text* to 1e-4), *raw* (sender cache injected without a map),
*project* (mapped cache), *derange* (mapped cache from another episode with the same layout but a different
value for the queried name). Content transfer = project − derange. For *derange* we also report the follow rate:
how often the answer is the *other* episode's value.

## 3. Results

**Table 1. Binding transfer, 8 names × 8 values, n = 1000 episodes, chance 12.5%.**

| projector | none | text | raw | project | derange | follow | content |
|---|---|---|---|---|---|---|---|
| ridge top-3 (closed form) | 10.6 | 100 | 14 | 21.7 | 11.4 | 18.2 | 10.3 |
| + residual, 1500 steps | 10.6 | 100 | 14 | **63.0** | 4.6 | **63.7** | 58.4 |
| residual trained with deranged sender | 10.6 | 100 | 14 | 23.4 | 11.3 | 22.7 | 12.1 |

On the original 6-name variant (n = 300) three training seeds give 71.3 / 73.0 / 75.3% (mean 73.2, sd 2.0).
Two ablations on the same variant: freezing the ridge and training only the residual gives 59.0%, so
letting the linear map move during training is worth ~14 points; replacing the correlation-selected top-3
source layers with the aligned-layer map gives 32.7%, so which sender layers feed each receiver layer is the
single largest design choice. Matching shape alone (*raw*) does nothing. The follow rate is the decisive number: when the
receiver is handed the wrong episode's cache it answers with the wrong episode's value 64% of the time, and its
accuracy falls below chance. The receiver believes the cache.

With the full 8-name pool and 500 episodes, the seed-2 projector gives 65.8% and 67.6% on two episode seeds
(ridge: 20.4%, 20.6%). On 256 fresh held-out sequences deduplicated against training text, continuation
retention is 1.19 for the trained projector and −0.38 for the ridge.

**Other attributes.** Same protocol with cities ("Vela lives in Paris") and animals ("Vela is a fox"), n = 500:
trained projector 61.8% and 67.2%; ridge 19.8% and 12.8%; deranged-sender control 15.4% and 12.4%.

**Distance.** Padding the facts with neutral filler so the question is 64 / 128 / 256 / 512 tokens after them:
61.0 / 57.0 / 62.4 / 68.8%. No decay; the best value is at the training prefix length.

**Where the binding lives (Table 2).** Inject this episode's mapped cache only into a band of receiver layers,
the deranged episode's elsewhere. n = 500, 6-name variant.

| layers | 0–5 | 0–11 | 0–17 | 12–23 | 24–35 | 30–35 | 18–35 | all |
|---|---|---|---|---|---|---|---|---|
| accuracy | 7.2 | 8.0 | 10.4 | 28.6 | 30.8 | 7.6 | 51.8 | 63.0 |

The first half of the network carries nothing usable on its own; the second half alone carries most of it, and
no six-layer band suffices. The binding is read out from a distributed set of later layers.

**Real passages (Table 2b).** The same projectors, never trained on questions, on SQuAD v1.1 validation:
the sender reads the passage (≤ 512 tokens), the receiver reads only the question and answers greedily
(≤ 8 tokens, stopped at newline). Token-F1 against the gold spans, n = 300 (600 for the trained projector).

| receiver sees | none | text | raw | ridge | trained | deranged-sender ctrl | trained, derange arm |
|---|---|---|---|---|---|---|---|
| F1 | 16.6 | 54.7 | 0.6 | 14.3 | **53.0** (51.8 at n=600) | 15.1 | 13.8 |
| gold log-prob / token | −3.57 | −0.90 | −11.2 | −3.88 | −1.74 | −3.77 | −4.18 |

On F1 the trained projector recovers 96% of the gap between question-only and reading the passage; on gold
log-probability it recovers 69%. The wrong passage's cache (derange), the ridge map, and the content-free
projector all sit at or below question-only. The channel carries the passage, not a bias toward answering.

**Continuation retention is the wrong headline (Table 3).** Following the closed-form-map literature we also
report retention = (loss_none − loss_project)/(loss_none − loss_oracle) on 128 fresh held-out sequences.

| projector | retention | retention with deranged sender text |
|---|---|---|
| ridge top-3 | −0.30 | −0.66 |
| + residual | 1.23 | 0.31 |
| residual trained with deranged sender | 0.77 | 0.72 |

A projector trained with the sender reading the *wrong* text — a learned soft prompt with no content channel —
recovers 77% of the oracle gap, and almost all of it (0.72) survives when its input is deranged at test time.
The trained projector's 1.23 decomposes into ~0.3 of the same prompt-like effect and ~0.9 that disappears when
the sender's text is swapped. Retention above 1 is therefore not evidence of communication, and retention alone
cannot distinguish a channel from a prompt; the binding test with its derange arm can.

## 4. What this does and does not show

It shows that a specific fact can cross from a 3.7B cache into a 7B without text, that the crossing depends on
the cache's content, and that the receiver reads it out from its later layers. It does not show that the channel
is cheaper than text (a 512-token cache is ~75 MB against ~1 KB of text; compression is future work), that it
carries anything richer than a single-token binding, or that it works across model families. The reverse
direction (7B → 3.7B) is running [pending].

## 5. Related work (contemporaneous)

Cache-to-cache communication (C2C, 2510.03215) and Latent Cache Flow (2605.22863) train fusers between models
of similar size; a closed-form KV map (2608.03893) reports 44–98% continuation retention across sizes; a causal
audit (2608.04893) finds that deranged caches lose only 0.4 points in such setups, which is the objection our
derange and deranged-sender arms answer. The Replay Gap (2608.08239) and Confidence Laundering (2606.20662)
argue that handoffs must be evaluated on branched rollouts and must carry uncertainty; both are next steps for
this line. These works are concurrent with ours; we treat them as the surrounding conversation, not as prior
results this paper improves on.

## 6. Conclusion

A learned cache map turns a size-mismatched pair of frozen models into a sender and a receiver. The controls
matter more than the headline: the wrong cache is followed, the content-free projector fails the test, and the
signal is read out late in the receiver. The next question is whether the same channel can carry how sure the
sender was.
