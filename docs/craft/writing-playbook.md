# 写作手册 / Writing playbook

*2026-10-08. Synthesised from the six craft studies in `docs/craft/` dated 2026-10-07: `corpus-agent-economy-essays`, `craft-long-reports`, `measurement-papers-craft`, `named-things-corpus`, `naming-corpus`, `program-launchers`. Our three documents: `docs/paper/main.tex` (paper v5), `docs/thesis/thesis.tex` (45 pp.), and the manifesto `docs/vision/manifesto-2026-10-07.md` (with `goal/GOAL.md` north star as its source). Numbers about our work come only from main.tex or thesis.tex. Quotes from others are at most 15 words and attributed to the author the craft study cites; this file did not re-check them against the originals. Line numbers refer to the files as of commit aa47171 plus the uncommitted `docs/vision/` drafts.*

Sections 1–8 are rules, each with one example of 15 words or fewer; 9 lists anti-patterns; 10 is the per-paragraph worklist.

Two of our own terms recur below. **The handle** is the thesis's name (thesis.tex line 134) for the pair **.552 against .532**: the receiver's AUROC after reading the confidence word, against the same receiver given no message. **SHOWN / SUGGESTED / BET** are the thesis's three evidence tags (thesis.tex lines 88–132): SHOWN means a controlled measurement, SUGGESTED means a consistent direction where one interval includes zero, and BET means a dated prediction with no evidence yet. The manifesto uses 【已证明】 and 【赌注】 for the same tags.

---

## 1. 开头：承诺的第一句 / Opening: the first sentence is a promise

None of the nine papers in the named-things study opens its abstract with a question. Each one opens with a flat statement about the field or with the finding. The reader decides from that first sentence whether to keep reading, so it has to state a fact or a result. It should not describe the setup.

**Rule 1.1. Open with a declarative about the field, or with the measured result. Never open with a question.**
Example: "Deeper neural networks are more difficult to train" (He et al., ResNet, 2016).
The thesis abstract already does this: its first sentence carries .955, .552 and .532. The paper abstract (main.tex line 58) opens with a question.

**Rule 1.2. The first sentence names its scope and the kind of evidence behind it.**
Example: "The biggest lesson that can be read from 70 years of AI research is" (Sutton, 2019).
Sutton's sentence sets the scope (70 years) and the kind of evidence (history) before it states the claim. Ours should say "377 SQuAD items, one frozen pair" in the first or second sentence.

**Rule 1.3. Name the belief you are displacing in the abstract itself, in one clause.**
Example: "The claim is that text launders confidence. It does not; this reader does." (thesis.tex line 141).
This pivot sentence is our best sentence, and it appears in neither abstract.

**Rule 1.4. When the essay is about money, the first concrete number is a unit cost on a mechanism, not a market size.**
Example: "fees as high as $0.30 per transaction, microtransactions become impractical" (Coinbase, x402 whitepaper, 2025).
Our unit costs are 144 KB per token, 72 MiB per 512-token prefix, and about 604 ms on a 1 Gbps link against 24 ms of receiver compute saved (main.tex line 300). In all three documents these numbers appear late.

## 2. 一句话主张的分层 / The one-line claim, in layers

All six studies describe the same structure. There is one sentence a reader can repeat. Below it are one or two numbers that make the sentence true. Below those are the controls and intervals. The layers go in that order, and each layer stays in its own place.

**Rule 2.1. Layer 1 is one sentence with no numbers. Layer 2 is the handle. Layer 3 is the full table, which stays in the body.**
Example: "Performance depends strongly on scale, weakly on model shape" (Kaplan et al., 2020).
Seven papers in the measurement-papers corpus put zero numbers in their abstracts. Our paper abstract (main.tex line 58) carries several dozen, mostly as "a / b / c" triples. The thesis abstract stops at three numbers.

**Rule 2.2. Write the handle as a pair, and say what each side means before printing either number.**
Example: "The receiver handed the label does no better than the receiver handed nothing" (thesis.tex line 134).
Every number needs its comparison beside it. The paper should say ".552 after reading the word, .532 with no message" before it prints any other AUROC.

**Rule 2.3. Keep the hedge one sentence away from the claim it qualifies. Never put it inside the same clause.**
Example: "We find that current large language models are significantly undertrained" (Hoffmann et al., 2022).
"Cache−word excludes zero on one variant of three" (main.tex line 58) belongs in the sentence after the claim, or in the limitations under the table. It should not sit inside the abstract's ordering clause. This is not a softer claim: every unrun experiment is still stated, just in its own sentence.

**Rule 2.4. Phrase the one-line recommendation as a rule, and print it word for word at three points: end of abstract, end of introduction, start of conclusion.**
Example: "for every doubling of model size the number of training tokens should also be doubled" (Hoffmann et al., 2022).
Ours is "Score the message, then score the reader." (main.tex line 317). Its reporting form is "report the label-only AUROC next to the receiver AUROC."

## 3. 命名：现象、方法、审计 / Naming: phenomenon, method, audit

The naming study (`naming-corpus`) splits coined terms into three kinds. Each kind follows its own rule.

**Rule 3.1. Phenomenon: one everyday noun or verb with a slight moral or clinical charge, coined in a sentence of the form "We call this X" right after the observation, with a test built into the definition.**
Example: "We call this phenomenon 'grokking'." (Power et al., 2022).
"Confidence laundering" passes this rule, but it is Shi et al.'s term: "We call the downstream effect confidence laundering" (Shi et al., 2026). Our relocation (the loss is at the reader) has no noun of its own yet. Candidate: **reader-side laundering**. It keeps their word and marks our move.

**Rule 3.2. Method: a plain compound noun that is also the instruction, or one named by what it removes.**
Example: "dispensing with recurrence and convolutions entirely" (Vaswani et al., 2017).
"Label-only audit" already follows this rule, because it names the method by removing the receiver.

**Rule 3.3. Audit: keep the opponent's noun, change where it applies, and name the quantity the audit produces as a difference.**
Example: "impose little or no 'alignment tax' on large models" (Askell et al., 2021).
"Read-out gap" is our difference-noun: message-level AUROC minus receiver-level AUROC. The paper currently names this one contribution three ways: "two-level report", "label-only audit", "read-out gap" (main.tex lines 66 and 74–75). Use one name for the procedure (label-only audit) and one for the quantity (read-out gap), and drop "two-level report".

**Rule 3.4. Introduce the name in the abstract, defined once and never justified. Never coin it first in paragraph 2 of the introduction.**
Example: "which we refer to as the KV cache" (Pope et al., 2022).
Neither of our abstracts contains "read-out gap".

**Rule 3.5. Name any mechanism the text describes more than twice, even while it is still a hypothesis.**
Example: "fluently make claims that are both wrong and arbitrary" (Farquhar et al., 2024).
"Evidence crosses and the receiver recomputes" appears unnamed in several places across main.tex and thesis.tex. Name the pair once, at main.tex line 70: **relay** (the sender's spread itself crosses) versus **recompute** (the evidence crosses and the receiver re-derives its own spread). Also name the thesis conditional (thesis.tex line 937): the **state-return hypothesis**.

**Rule 3.6. Borrow one classical frame, declare the borrowing once with its source, and do not add a second.**
Counter-example: "Nature hates a vacuum." (Grady, Sequoia, 2025). This frame is borrowed and never paid off with evidence.
We borrow "laundering" from finance, and "audit" is its natural partner. Do not add "telepathy" or any third metaphor. The craft study grepped all three files for "latent telepathy" and found zero hits. Keep it that way.

## 4. 一张标志性图的规则 / One iconic figure

In every document the named-things study looked at, Figure 1 is the finding drawn so simply that a reader could sketch it from memory: ResNet's two training curves, the U-shaped curve in *Lost in the Middle*, Chinchilla's fit drawn over Kaplan's. The one exception is the Transformer architecture diagram, and it works only because the architecture itself was the contribution.

**Rule 4.1. Figure 1 shows the finding. The apparatus goes in Figure 2.**
Example: "Cluster sizes in a t-SNE plot mean nothing" (Wattenberg, Viégas, Johnson, Distill, 2016).
The paper's first main-body figure is `fig:arms` (main.tex line 129): Noma accuracy by arm, which is the instrument's calibration. The finding figure, `fig:audit` (line 246), comes later and has two panels. The thesis already has the single-panel version, `fig:handle` (thesis.tex line 256).

**Rule 4.2. One visual encoding carries the claim, and the caption names it in one clause.**
Example: "the vertical distance from marker to bar is the read-out gap" (main.tex `fig:audit` caption).
Keep this clause, move it to the start of the caption, and delete the right-hand panel from Figure 1. The within-item test becomes its own figure.

**Rule 4.3. The figure states its floor and ceiling inside the plot.**
Example: "designed to be a minimal testbed for the basic ability" (Liu et al., 2023).
Question-only (.532) is the floor and text (.597) is the ceiling. Draw both as dashed lines so the reader sees where the cache (.585) and the word (.552) fall between them.

**Rule 4.4. One recurring example object across the whole document.**
Example: "Noma is amber. Vela is indigo." (main.tex line 86; thesis.tex line 340).
In *Scaling Monosemanticity*, the Golden Gate Bridge feature does this job, and in *Concrete Problems* the cleaning robot does. In our documents Noma appears only in the binding test. Reuse the same cast of names in the thesis toy example (thesis.tex line 232): a sender unsure whether Vela is indigo or amber.

## 5. 证据与赌注的排版分离 / Typesetting evidence apart from bets

The long-report study found that the best long documents separate "what we did" from "what we believe" by section and by typography, not by hedge words. The belief sections are short and come last. Our SHOWN / SUGGESTED / BET boxes are stronger than anything in that corpus. The rule is to carry them everywhere.

**Rule 5.1. Every claim carries exactly one tag, and the tag stays the same wherever the claim recurs.**
Example: "A claim appears in exactly one of them." (thesis.tex line 91).
The paper has no tags. Its Contributions (1)–(4) (main.tex lines 72–76) should each end with SHOWN or SUGGESTED in small caps.

**Rule 5.2. Use belief verbs for beliefs and measurement verbs for measurements. Let the section, not an adverb, carry the uncertainty.**
Example: "deliberately speculative" (Olah et al., *Zoom In*, 2020), said once about the claims as a group.
Words like "nearly", "barely" and "about as well" do work that the interval should do. Print the interval instead. In main.tex line 70, "nearly as well through the mapped cache" should become ".585 vs .597, interval includes zero".

**Rule 5.3. Every bet carries a date or a condition that would kill it, in the same typeset unit as the bet.**
Example: "this plan is super-ambitious" (Buterin, 2024), followed by a viability tag on each branch.
The manifesto ledger (`manifesto-2026-10-07.md` lines 53–64) does this. Our counter-example is Aschenbrenner's "trust the trendlines" (2024): inevitability with no kill condition.

**Rule 5.4. Write a "Nuance" note under each result, not one long Limitations paragraph at the end.**
Example: "we expect that these are on the higher end of real world gains" (TypeSafe, 2026).
The paper's Limitations section (main.tex line 311) is one dense paragraph. Split it into short notes placed under the tables they qualify. Then give the remaining items headings phrased the way a referee would say them.

**Rule 5.5. State one non-claim early, in a sentence that costs nothing.**
Example: "nothing in this paper should be interpreted as claiming" (Schaeffer et al., 2023).
The thesis has a section for this, "What this document does not claim" (line 293). The paper's introduction has no such sentence.

## 6. 负结果写成发现 / Negatives written as findings

**Rule 6.1. Give each negative its own heading, phrased as what it established.**
Example: "None of the six is an apology; each is tagged with what it established." (thesis.tex line 828).
Thesis Part III does this. The paper's §6 heading, "Two Objectives That Did Not Move the Loss" (main.tex line 270), is already phrased as a finding.

**Rule 6.2. End each negative with what it rules out, then add the untested fix in one clause.**
Example: "Thus we conclude that we only find partial generalization on this task." (Kadavath et al., 2022).
The last sentence of main.tex §6 already has this order: "Neither target contains the sender's spread", then the sender-side reader as the untested fix. Keep it.

**Rule 6.3. A missed pre-registered criterion is stated once, with both numbers, and placed where the reader looks for results.**
Example: "appeared to disagree with the commonly held belief" (Sutskever, thesis, 2013).
Ours: the design note asked for cache AUROC ≥ .70. We measured .585 / .605 / .624. The drop-ratio condition passed at .54. This belongs in a single sentence in the §4 results paragraph, not only in Limitations.

**Rule 6.4. Do not undersell the surprise. When a nearly perfect label meets a reader who ignores it, say so plainly.**
Example: "we discover with great surprise that modern neural networks are no longer well-calibrated" (Guo et al., 2017).
"The audit is a sanity check, not a finding" (main.tex line 262) buries the handle in the very sentence that should carry it. Keep the caveat about how the label was built (tercile against median), but put it after the contrast.

**Rule 6.5. File a retraction as a two-sentence correction. Never put it in a box before the first result.**
Example: "Updates and Corrections" (Distill convention, dated footer).
The "Capacity, stated correctly" box (main.tex line 96) sits in §2, before any finding. Its first sentence should be the one saying that no measurement in the paper tests it.

## 7. 长文档的呼吸 / Pacing a long document

"Pacing" here means three concrete devices that let a reader stop and restart a 45-page document without losing the thread.

**Rule 7.1. Put a results list before any method: five or more bold one-line claims, each followed by a pointer to its section.**
Example: "Continuation retention is the wrong headline." (main.tex line 191, a caption).
The thesis's SHOWN box does this. The paper needs a five-line version between the abstract and §1.

**Rule 7.2. Open each chapter with one paragraph that maps it: what this chapter assesses, and what it leaves to the next.**
Example: "The Importance of Knowing What We Don't Know" (Gal, thesis, 2016), followed by a map paragraph.
Thesis Part II chapters open with a figure and no map sentence. `ch:where` (line 766) is the model to follow.

**Rule 7.3. Name the handle in the first paragraph of every results chapter, not only in the front matter.**
Example: "One pair of numbers runs through the document: .552 against .532." (thesis.tex line 134).

## 8. 愿景文章：把赌注写得像必然又不撒谎 / Vision essays: bets that read as inevitable, without lying

**Rule 8.1. Make inevitability come from a count the reader can redo, never from adjectives.**
Example: "Counting the OOMs" (Aschenbrenner, 2024), a section title that is also arithmetic.
Our count is 72 MiB against about 1 KB, and 604 ms against 24 ms.

**Rule 8.2. Justify the program under several futures, and say what still gets published in each one.**
Example: "help us succeed across a range of different scenarios" (Anthropic, *Core Views*, 2023).

**Rule 8.3. Mark the least certain claim as the bet, in the text itself, and say what would show it wrong.**
Example: "one of which turned out to be horribly wrong" (Olah et al., *Zoom In*, 2020, on Schwann).

**Rule 8.4. End by giving the reader one cheap job.**
Example: "Who will build the first Software 2.0 IDE?" (Karpathy, 2017).

## 9. 反模式 / Anti-patterns

Each was seen in the corpus; the pointer says where it appears in our files, if anywhere.

- Opening with "Imagine" in place of a measurement: "Imagine an agent streaming funds to a compute provider." (Broner, a16z, 2026). Not present in our files.
- Fictional prices printed under a "usage" heading, as in x402 §11. Every unmeasured price we have stays in the BET column.
- Taste offered in place of a benchmark: "good model smell" (Almeida, Latent Space, 2026). Not present in our files.
- An abstract built like a table. Present at main.tex line 58.
- A title that uses the opponent's word for our own finding. Present at main.tex line 41.
- Contributions written as chains of method clauses. Present at main.tex lines 73–76.
- The same hedge repeated four times. "Established on one variant of three" and "intervals include zero" both recur in main.tex. State each once in Results and once in Limitations.
- A weakened baseline admitted only in Limitations. Present at main.tex line 311 ("not the strongest a competent engineer could build"). Move it under `tab:confidence`.
- A balanced benefits/limitations list that never says which side wins. Present at the end of main.tex Limitations.

## 10. 三份文档的逐段改法 / Paragraph-by-paragraph worklists

Each row: location, change, rule applied. No new numbers.

### 10a. 论文 / Paper, `docs/paper/main.tex`

| Where | Change | Rule |
|---|---|---|
| l.41 title | Subtitle → "The Label Carries the Signal, This Reader Discards It" (the thesis's subtitle) | 3.3 |
| l.58 abstract, sentence 1 | Replace the question with the pivot: the label scores .955, the reader of it .552, no message .532 | 1.1, 1.3 |
| l.58 abstract, body | Drop the triples and keep one cache sentence (63.0% binding, 53.0 vs 54.7 F1, .585); add "read-out gap" and "label-only audit" | 2.1, 3.4 |
| l.58 last sentence | Keep "Score the message, then the reader" | 2.4 |
| between l.59 and l.61 | Five-line results summary, each line ending with its tag | 7.1, 5.1 |
| l.64 intro ¶1 | Second sentence: 144 KB per token against about 1 KB of text | 1.4 |
| l.70 intro ¶4 | Name relay vs recompute; "nearly as well" → number plus interval; add the non-claim sentence | 3.5, 5.2, 5.5 |
| l.72–76 Contributions | Four bold sentences, with cross-references after them | 2.1 |
| l.96 capacity box | Two-sentence correction at the end of Related Work | 6.5 |
| l.129 / l.246 figures | `fig:audit` left panel becomes Figure 1, with floor and ceiling lines | 4.1, 4.3 |
| l.212 §4 heading | "The Reader Handed the Word Does No Better Than the Reader Handed Nothing" | 2.2 |
| l.262 §5(a) | Contrast first, construction caveat second; delete "sanity check, not a finding" | 6.4 |
| §4 results ¶ | Missed ≥ .70 criterion stated once, with .585 / .605 / .624 | 6.3 |
| l.311 Limitations | Split into notes under each table, plus three referee-phrased headings | 5.4 |
| l.317 Conclusion | Open with a numbered 1–4 lesson; close with the reporting rule | 2.4 |

### 10b. 论文长文 / Thesis, `docs/thesis/thesis.tex`

| Where | Change | Rule |
|---|---|---|
| l.82 abstract | Add the pivot sentence from l.141 and the term "read-out gap" | 1.3, 3.4 |
| l.88–132 claims | Call the BET list "deliberately speculative" and say one item is expected to fail | 8.3 |
| l.232 toy example | Recast with Noma and Vela as the cast | 4.4 |
| l.256 `fig:handle` | Add floor and ceiling dashed lines; make it the first figure | 4.1, 4.3 |
| l.305, 405, 515, 576, 657 | Add a one-paragraph map under each opening figure, naming the handle | 7.2, 7.3 |
| l.937 conditional | Name it the state-return hypothesis and put it in a box | 3.5 |
| l.1091 Horizon | Three futures, one sentence each, saying what still gets published in each | 8.2 |

### 10c. 宣言 / Manifesto, `docs/vision/manifesto-2026-10-07.md` (source: `goal/GOAL.md`)

| Where | Change | Rule |
|---|---|---|
| l.1 title | A statement, not a question (e.g. 状态是下一个协议) | 1.1 |
| l.7 opening | Keep; it promises one measurement and then bets | 1.2 |
| l.9 | Add 72 MiB / 604 ms here, before §一 | 1.4, 8.1 |
| l.25 | Put 读者侧洗白 / reader-side laundering in bold once | 3.1 |
| l.29 一句话 | Name relay vs recompute (转述 vs 重算) | 3.5 |
| l.39 | Add what would show this sentence wrong | 8.3 |
| l.41 §四 | Move to directly after the opening; it is the count | 8.1 |
| l.49 | Keep the one-line request as the closing sentence, after the ledger | 8.4 |
| GOAL.md north star 1–3 | Tag each sentence; sentence 4 already reads (bet) | 5.1 |
