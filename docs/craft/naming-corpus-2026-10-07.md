# How memorable ML terms got coined, and what it says about ours (2026-10-07)

Reader studying craft, not content. Sources read: Maynez et al. 2020 (abs), Mirsky-style "AI Hallucinations: A Misnomer Worth Clarifying" 2401.06796 (PDF), Karpathy 2015 blog, Smith/Greaves/Panch 2023 (via summary), Askell et al. 2021 (PDF), Power et al. 2022 (PDF), Wei et al. 2022 emergent (PDF), Schaeffer et al. 2023 (abs), Wei et al. 2022 CoT (PDF), Kaplan et al. 2020 (abs), Hestness et al. 2017 (abs), Lambert RLHF book ch.2, Ziegler et al. 2019 (abs), Ouyang et al. 2022 (abs), Shazeer 2019 (abs), Pope et al. 2022 (PDF), Shi et al. 2026 "Confidence Laundering" (PDF), Du et al. Interlat 2511.09149 (abs), Zheng et al. "Thought Communication" 2510.20733 (abs). Our texts: docs/paper/main.tex (v5), docs/thesis/thesis.tex, goal/GOAL.md.

## Term by term

| Term | Who / where | Coining sentence (<=15 words) | Why it stuck | What lost |
|---|---|---|---|---|
| hallucination | Karpathy blog 2015 casual; Maynez et al. 2020 formal; CV use since Baker & Kanade 2000 | "the model just hallucinated it" (Karpathy 2015) | one word, a verb, a little clinical-dangerous; names the behaviour from outside | confabulation (Smith et al. 2023), fabrication / fact fabrication, delusion (2401.06796 table) |
| emergent abilities | Wei et al. 2022, via Steinhardt 2022 and Anderson 1972 | "not present in smaller models but is present in larger models" (Wei 2022) | definition and a prediction in one sentence ("cannot be predicted by extrapolating") | phase transition (used in the same paper, too physics-exact); then attacked by "mirage" (Schaeffer 2023) |
| alignment tax | Askell et al. 2021, in scare quotes, with "Alignment Tax/Bonus" in a figure title | "impose little or no 'alignment tax' on large models" (Askell 2021) | a payer, a direction, an antonym (bonus); measurable as a difference | "capability degradation", "performance cost" (no payer, no antonym) |
| scaling laws | Kaplan et al. 2020 title; the finding is Hestness et al. 2017 | "The loss scales as a power-law with model size" (Kaplan 2020) | "laws" over-claims lawfulness, which is exactly the dangerous part; a title noun phrase became a field | "Deep Learning Scaling is Predictable, Empirically" (Hestness 2017): a descriptive sentence, no noun to cite |
| grokking | Power et al. 2022, OpenAI; Heinlein's verb | "We call this phenomenon 'grokking'." (Power 2022) | coined word in title, descriptive gloss as subtitle ("Generalization Beyond Overfitting"); christened in a contribution bullet after the observation | "delayed generalization", "grok-like learning curve" (Neelakantan 2015, cited by Power) |
| chain of thought | Wei et al. 2022; idea from Ling et al. 2017, Nye et al. 2021 | "a chain of thought—a series of intermediate reasoning steps" (Wei 2022) | names the artifact as the model's own cognition (anthropomorphic); method = name + "prompting" | rationale (Ling 2017, lawyerly), scratchpad (Nye 2021, clerical) |
| RLHF | nobody in an abstract; Christiano 2017 "human preferences", Ziegler 2019 "reward learning", Ouyang 2022 spells it out | "using reinforcement learning from human feedback" (Ouyang 2022) | the community acronymised a phrase the authors had only spelled out; four letters, a procedure | "learning from human preferences", "reward learning", TAMER (Knox & Stone 2008) |
| KV cache | Pope et al. 2022, in a subordinate clause; artifact existed since 2017 | "which we refer to as the KV cache" (Pope 2022) | christening an implementation detail made it a research object; "cache" implies it can be moved, which licensed cache transfer (C2C etc.) | "keys and values tensors" (Shazeer 2019), "incremental state" (fairseq), past_key_values (HF API) |
| confidence laundering | Shi, Zhang, Bao, Nelson, Ye 2026 | "We call the downstream effect confidence laundering" (Shi 2026) | moral vocabulary (a crime), colon, one-clause definition, then five concrete instances ("A complete API call appears executable...") | "interface collapse" (their own neutral term, same paper, not adopted) |
| latent telepathy | not a title term anywhere found; "telepathy" is a simile in Zheng 2025 and Du 2025 | "interact directly mind-to-mind, akin to telepathy" (Zheng 2025) | the dangerous word lives in the motivation sentence; the sober noun (thought communication, latent communication, Interlat) is the term | n/a: the phrase "latent telepathy" does not appear in our paper, thesis or GOAL.md either (grep: 0 hits) |

## Rules derived

Phenomenon: one ordinary-life verb-noun with a slight moral or clinical charge (hallucinate, launder, grok, tax), named from the outside; christened in one sentence of the form "We call this X:" immediately after the observation; definition carries its own test or prediction; metaphor for the motivation sentence, sober noun for the term.

Method: a plain compound noun that is also the instruction (chain-of-thought prompting, KV cache); christen in a subordinate clause if the artifact already exists; do not coin the acronym, supply the acronym-ready phrase.

Audit: take the opponent's noun and change where it lives (mirage, misnomer; our "message versus reader"); state the audit as an imperative of one line ("score the message, then the reader"); name the quantity it produces as a difference (tax, gap).

## Our terms, checked against the rules

Paper v5 names one contribution three ways: "two-level report", "label-only audit", "read-out gap"; the thesis names the catalogue two ways: "price sheet", "priced catalogue". Pick one each. The paper's title already follows coin + gloss. The relocation claim ("the reader, not the message") lacks a noun of its own; candidate: reader-side laundering vs channel-side laundering, keeping Shi et al.'s word and marking the move. "Uncertainty transfer" as a noun implies the thing happens; the paper's own verdict is SUGGESTED not SHOWN, so do not nominalise it.
