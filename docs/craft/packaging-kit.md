# Packaging kit: names, title, pitches, figure

Date: 2026-10-08. Paper: `docs/paper/main.tex` (v5). Every number below was checked against `docs/paper/main.tex` or `analysis/cloud/verbal/audit.json`; line numbers refer to main.tex. Nothing here changes the paper; this file is input for the next edit pass.

## 0. Corrections found while checking the picks and kits

These change what the pitches may say. Each one points to the line that settles it.

1. **The .955 is a construction, not a discovery.** The confidence word is an oracle label built from the sender's own semantic-entropy terciles (line 92), and main.tex calls scoring it "a sanity check, not a finding" (line 262). Any wording like "the label was right", "nearly perfect label" or "the sender told it exactly how sure it was" overclaims. The claim that survives: even a label built from the sender's own uncertainty goes unread by this receiver.
2. **The sender did not say the word.** Line 92: "the sender does not verbalise its own confidence." Pitches that say "agents attach the right confidence word" or "a small AI told a bigger one" describe a setup we did not run. Say "a confidence word built from the sender's uncertainty".
3. **The contradicted swap is reported.** Pick 1 says audit.json has a contradicted within-item swap "that the paper does not report", and kit 2 says the swap ran on one variant only. Both are wrong: line 264 reports it (cache .208, text .325, word .179, another item's cache .004). The headline swap is clean to removed (.280 / .322 / .146 / .004); the contradicted one is the weaker second run.
4. **The cache does not reliably beat no message on clean passages.** Cache minus question-only on clean is [-.005, .111] (line 253, audit.json `ci_project_minus_none`). Only the contradicted variant excludes zero ([.065, .180]). So "evidence pull" and any cache-advantage wording cannot be a headline.
5. **"Text bought 3.5 F1" is not a text-handoff gain.** The 3.5 F1 is the receiver reading the passage minus the sender (55.7 vs 52.2, line 312), used to explain why the handoff policy is negative.
6. **"One line of code" was never measured.** main.tex says "a one-line audit" in the abstract (line 58); keep that phrase or "one extra score", and drop "one line of code".
7. **The current title uses the opponent's word.** Line 41: "Where Confidence Gets Laundered". The writing playbook (`docs/craft/writing-playbook.md`, section 9) lists "a title that uses the opponent's word for our own finding" as an anti-pattern and names line 41.

## 1. The three picks side by side

| Slot | Pick 1 | Pick 2 | Pick 3 |
|---|---|---|---|
| Phenomenon (headline) | the unread label | the unread label (as gloss) | the unread label |
| Phenomenon (body coinage) | "reader-side laundering", once, credit 2606.20662 | "We call this reader-side laundering", once, credit 2606.20662 | none; credit Shi et al.'s "confidence laundering" as the claim we narrowed |
| Scope wording | "for this receiver" every time | one 7B base reader, one completion prompt, no hedge instruction | same as pick 2, stated at first use |
| Procedure | label-only audit | label-only audit | label-only audit |
| Quantity | read-out gap = message AUROC minus receiver AUROC, within-arm only | same, defined in the abstract | same, defined in the abstract |
| Printed rule | "Score the message, then the reader." | (title) | "Score the message, then score the reader." (main.tex line 317 already ends on it) |
| Within-item control | described, not branded | "passage swap, with another item's cache as null (.004)" | passage swap, question held fixed, null .004 |
| Line name | state handoff | state handoff | state handoff (状态交接) |
| "State, Not Text" | stays in GOAL.md, labelled BET | stays, labelled BET everywhere | stays on GOAL.md and manifesto, labelled BET |
| Title | The Unread Label: Scoring the Message and the Reader in a Handoff between Frozen Models | Score the Message, Then the Reader: Where Uncertainty Is Lost in a Handoff between Frozen Models | The Label Was Right: Message versus Reader in a Model Handoff (fallback: current title) |
| Figure 1 | clean only, y .5 to 1, marker .955 over word .552, bracket "read-out gap", floor .532, ceiling .597, cache dot .585, paired intervals drawn | clean only, bars .532 / .552 / .585 / .597 with 95% intervals, marker .955, caption opens with the read-out-gap sentence | clean only, floor .532 and ceiling .597, bars, marker .955, bracket, intervals so cache minus text visibly includes zero |
| Figure 2 | within-item Spearman .280 / .322 / .146 / .004 | same | same |
| Rejected | label deafness, dropped label, handed-nothing parity, recompute-not-relay; state channels, Statewire, Handoff Ledger, priced state; "The Label Was Right" | label deafness, handed-nothing parity, recompute-not-relay as finding; state channels, Statewire, Handoff Ledger, priced state, K2 Cascade; "The Label Was Right" | label deafness, handed-nothing parity, recompute-not-relay as name; state channels, Handoff Ledger |

Agreement: all three put "the unread label" in the headline, "label-only audit" and "read-out gap" as the method pair, and "state handoff" as the line name, and all three keep the within-item control as a description. They split on two points: whether to coin "reader-side laundering" in the body (picks 1 and 2 yes, pick 3 no) and the title (three different ones).

## 2. Final recommendation

Rule used: majority where the picks agree, best-argued where they split. A pick counts as better argued when its claim survives the checks in section 0.

**Phenomenon: "the unread label" in the headline and the figure; "reader-side laundering" coined once in the body.** On the coinage this goes with the majority (picks 1 and 2), and it is also what the playbook asks for (Rule 3.1 and Rule 3.3 in `docs/craft/writing-playbook.md`: keep the opponent's noun and change where it applies). Pick 3's worry, that coining our own "laundering" term borrows Shi et al.'s moral charge, is handled by crediting 2606.20662 in the same sentence and never putting "laundering" in a title. Sentence to use, right after the .955 / .552 / .532 triple: "We call this reader-side laundering, after Shi et al. (2606.20662): the label carries the signal and this reader discards it." Every use carries the scope: one 7B base receiver, one completion prompt, no instruction to hedge.

**Method: "label-only audit" (the procedure) and "read-out gap" (the quantity).** All three picks agree. Define the read-out gap in the abstract as "message-level AUROC minus receiver-level AUROC, a difference of two AUROCs, comparable within an arm". The cache arm has no label, so its message level needs a linear probe, which is a different instrument (line 58: .557 / .579 one-position probe). Print the rule as main.tex line 317 already has it: "Score the message, then score the reader."

**Within-item control: describe it, do not brand it.** All three agree. Wording: "a passage swap with the question held fixed, and another item's cache as the null (Spearman .004)". Both runs are in main.tex line 264: clean to removed (.280 / .322 / .146 / .004) and clean to contradicted (.208 / .325 / .179 / .004).

**Line: "state handoff" (状态交接).** All three agree. "State, Not Text" stays the GOAL.md north star (`goal/GOAL.md` line 5) and carries a BET tag wherever it appears outside GOAL.md, because cache minus text includes zero on clean and contradicted passages and text is above the cache on removed ([-.091, -.002], line 253).

**Title: pick 1's.**

> The Unread Label: Scoring the Message and the Reader in a Handoff between Frozen Models

Why this one over the other two:
- Pick 3's "The Label Was Right" fails correction 1: the label's rightness is how it was built, and main.tex line 262 says so. Two of three picks reject it.
- Pick 2's "Where Uncertainty Is Lost" promises a location for the loss in both arms. main.tex locates it only for the label arm; for the cache arm "linear probes at four points of the chain do not localise the loss" (line 70). The title would promise what section `sec:where` does not deliver.
- Pick 1's title puts the agreed headline noun first and names the procedure in the subtitle. It makes no claim about the cache arm and no claim that the label was good.

Risk of the chosen title: "The Unread Label" reads as general if the scope is missing. The abstract's first sentence that uses the phrase must carry "for this receiver". The running head (line 36) becomes "The Unread Label".

Fallback title, if a reviewer finds the headline noun too cute: pick 2's main clause with pick 1's subtitle, "Score the Message, Then the Reader: Auditing Uncertainty in a Handoff between Frozen Models" (this is kit 1's second title).

Edits this implies for main.tex (not made here; this file only records them):
- line 36 and line 41: the new title and running head.
- the sentence after the .955 / .552 / .532 triple in the introduction (line 70): add the "We call this reader-side laundering" coinage.
- the abstract (line 58): define read-out gap as a difference of two AUROCs.
- `fig:audit` (caption at line 249): promote the clean-passage left panel to Figure 1 (section 6 below).

## 3. All candidates with risks

Merged from the three kits. Status: CHOSEN, KEEP (usable in a stated role), or REJECTED.

### 3.1 Phenomenon names

| Name | What it says | Risk | Status |
|---|---|---|---|
| the unread label (没被读的标签) | Word scored as a label .955; receiver reading it .552; no message .532 (clean). | Only about the label arm. One base model, one completion prompt, no hedge instruction; must not stretch to "LLMs ignore labels". | CHOSEN (headline, figure) |
| reader-side laundering (读者侧洗白) | Keeps Shi et al.'s noun, moves the loss from the channel to the reader. | Borrowed word that implies intent; must be credited each time; can read as "Shi et al. were wrong in general". | CHOSEN (body, once) |
| label deafness (标签失聪) | The label arrives and the reader acts as if it had not. | Clinical word that sounds like a trait of all models; sets up a "cure" paper someone else writes. | REJECTED |
| the dropped label (被丢下的标签) | The signal was in the message; this reader threw it away. | "Dropped" reads as loss in the channel, the opposite of the finding; too ordinary to cite. | REJECTED |
| handed-nothing parity | The receiver handed the word does no better than the receiver handed nothing. | Word minus no-message is +.020 / +.063 / -.011 (line 312). "Parity" fits only clean, and on removed the question alone beats the word (.603 vs .592). | REJECTED |
| recompute, not relay (重算而非转述) | Evidence crosses in the cache and the receiver re-derives its own spread. | SUGGESTED, not shown: the four-point probes do not localise the loss and no experiment isolates relay (line 70). Allowed only as "the parsimonious reading". | KEEP (discussion only, never a title) |
| evidence pull / evidence-following (随证据而动) | Within an item, receiver entropy moves with the sender's: cache .280, text .322, word .146, another item's cache .004. | Modest effect; text pulls at least as hard; cache minus question-only on clean includes zero ([-.005, .111]). Reads as a cache advantage. | REJECTED as a name; the numbers stay in Figure 2 |
| evidence moves, labels don't | Both halves in one slogan. | Not a noun, hard to cite; the word does move the receiver a little (.146). | REJECTED |
| the label was right | The message carried the signal. | The .955 is a construction (line 262). | REJECTED |

### 3.2 Method names

| Name | What it says | Risk | Status |
|---|---|---|---|
| label-only audit (仅标签审计) | Score the message against the receiver's target with no reader, then score the reader. | Covers the label arm only; the cache needs a linear probe, a different instrument. Reviewers may call it obvious. | CHOSEN (procedure) |
| read-out gap (读出差) | Message-level AUROC minus receiver-level AUROC; about .40 for the word on clean (.955 minus .552). | A difference of two AUROCs, not an information measure; not comparable across arms; the .955 depends on terciles over 587 items vs a median over 377. | CHOSEN (quantity) |
| Score the message, then the reader (先评消息，再评读者) | The rule, printed word for word in the abstract and conclusion. | Imperative, not a noun; easy to cite without running it; needs the .955 / .552 pair beside it. | KEEP (printed rule) |
| score twice | Short imperative form. | Reads as advice, not a method. | REJECTED |
| two-level report | Report both levels. | A second name for the same procedure. Use one name. | REJECTED |
| held-question swap / same-question swap (同题换文) | Hold the question, change only the sender's passage. | "Swap" collides with activation-patching vocabulary; same-sign rates are capped by the 91.5% ceiling, so report the Spearman. | KEEP as a description, not a brand |
| derange control | Another item's cache as the null (.004). | Standard in relay-audit work (line 307 cites 2608.04893 and 2607.26773); branding it overclaims novelty. | KEEP as a description |
| two scores and a swap (两评一换) | The whole kit in one phrase. | Reads as a gimmick. | Talks only |

### 3.3 Line names

| Name | What it says | Risk | Status |
|---|---|---|---|
| state handoff (状态交接) | Names the event measured: frozen models pass computed state, and the reader is audited. | Generic; may read as "just cache transfer"; "handoff" also appears in robotics (Helix, GR00T). | CHOSEN |
| State, Not Text (要状态，不要文本) | The GOAL.md north star. | A BET. Text matches or beats the cache in our own Table (`tab:confidence`), and the cascade's handoff policy was negative (line 312). | KEEP in GOAL.md, labelled BET |
| state channels (状态通道) | Models exchange state instead of words. | Collides with blockchain payment "state channels", worse next to "agent economy". | REJECTED unless written "model-to-model state channels" |
| cache relay | Concrete artifact plus verb. | Narrows the line to KV and puts the 72 MiB cost in the name. | REJECTED |
| handoff audit(s) (交接审计) | The line measures every handoff: text, label or state. | Smaller ambition than a protocol program; reads as a benchmark shop. | KEEP as the near-term product framing (BET) |
| priced state (带价的状态) | State with a cost, a value and an audit. | No price measured beyond bytes (72 MiB per 512-token prefix) and latency (~604 ms on 1 Gbps). | REJECTED |
| Statewire, Handoff Ledger, K2 Cascade | Brand names. | Statewire is unrelated to anything measured; Handoff Ledger sounds like crypto; K2 Cascade is the repo, and the cascade policy result was negative. | REJECTED |

### 3.4 Titles

| Title | Risk | Status |
|---|---|---|
| The Unread Label: Scoring the Message and the Reader in a Handoff between Frozen Models | Headline reads as general unless the abstract scopes it. | CHOSEN |
| Score the Message, Then the Reader: Auditing Uncertainty in a Handoff between Frozen Models | Imperative title; gets cited as a phrase. | Fallback |
| Score the Message, Then the Reader: Where Uncertainty Is Lost in a Handoff between Frozen Models | Promises a location for the cache arm's loss that the probes do not give. | REJECTED |
| The Label Was Right: Message versus Reader in a Model Handoff | Treats the .955 as a finding (line 262). | REJECTED |
| Where Confidence Gets Laundered: Message versus Reader in a Model Handoff (current, line 41) | Opponent's word in the main title; playbook section 9 anti-pattern. | REJECTED |
| Handed the Label, It Ignored It: Reader-Side Laundering between Frozen Language Models | "Laundering" in the title; "ignored" overstates the +.020. | REJECTED |
| Reader-Side Laundering: A Confidence Label at .955, Its Reader at .552 | Numbers in a title invite "on one model". | REJECTED |
| The Reader Handed the Word Does No Better Than the Reader Handed Nothing | Too long; true only on clean. | REJECTED |
| Evidence Crosses, the Reader Recomputes: Uncertainty in KV-Cache Handoffs | SUGGESTED mechanism in the title. | REJECTED until an isolating experiment exists |

## 4. The four pitches

Built from the three picks' pitches with the section 0 corrections applied. Each pitch says "built from the sender's uncertainty" (correction 2) and "by construction" or equivalent (correction 1). Founder and journalist pitches carry a BET tag where they say what is sellable.

### 4.1 Reviewer / 审稿人

EN: Handoff studies usually score only the receiver. On 377 SQuAD items, a confidence word built from the 3.7B sender's semantic entropy scores .955 AUROC as a label, which holds largely by construction. The K2-Horizon 7B receiver reading that word reaches .552, against .532 with no message. So for this receiver, under one completion prompt, the text arm's loss is at read-out, not in what the message carries. Verbal-confidence baselines should report both levels.

中文：交接研究通常只给接收方打分。在 377 道 SQuAD 题上，用 3.7B 发送方语义熵构造的置信词，单独当标签打分是 AUROC .955，这个数主要来自构造方式。K2-Horizon 7B 接收方读了这个词之后是 .552，不给任何消息是 .532。所以对这个接收方、在这一种补全提示下，文本这一路的损失发生在读取环节，不在消息本身。报告口头置信度基线的工作应该把消息层和接收方层两个数都报出来。

### 4.2 Lab lead / 实验室负责人

EN: Add one row to your eval: score the message against the same target you score the receiver on. That separates a handoff that fails in what it sends from one that fails in who reads it. In our 3.7B-to-7B handoff, for this reader, it failed at the reader. A mapped KV cache moved the receiver with the sender's evidence within an item (Spearman .280, against .004 for another item's cache), but reading the passage did as well (.322), and the cache costs 72 MiB per 512-token prefix.

中文：评测里加一行：用给接收方打分的同一个目标，给消息本身打分。这样能分清交接是败在发出去的内容上，还是败在读它的模型上。我们 3.7B 到 7B 的交接里，对这个接收方来说，败在读者。映射后的 KV 缓存让接收方在同一道题内随发送方的证据变化（Spearman .280，换成别的题的缓存是 .004），但直接读原文也一样好（.322），而缓存每 512 token 的前缀要 72 MiB。

### 4.3 Founder / 创始人

EN: A handoff can carry a confidence word built from the sender's own uncertainty, and the next model can still act almost as if it had been told nothing. We measured that on one model pair. Moving state instead of words makes the receiver follow the sender's evidence, but at 72 MiB per message (about 604 ms on a 1 Gbps link, against 24 ms of compute saved) the mapped-cache channel is a measurement instrument today. BET: what is ready now is the label-only audit, not the channel.

中文：交接时可以附上一个按发送方自身不确定性构造的置信词，下一个模型却可能表现得和什么都没收到差不多。这是我们在一对模型上测到的。改传状态能让接收方跟着发送方的证据走，但每条消息 72 MiB（1 Gbps 链路上约 604 ms，而省下的计算只有 24 ms），所以映射缓存这条通道眼下只是测量工具。押注（BET）：现在能拿出去用的是"只给标签"的审计，不是这条通道。

### 4.4 Journalist / 记者

EN: We handed a larger AI a note saying how unsure a smaller AI was, worked out from the smaller one's own answers. The note was accurate because we built it that way. The larger AI, reading it, did barely better than when it got no note at all. In this test the problem was the reader, not the note.

中文：我们根据小模型自己的多次回答，算出它有多不确定，写成一张条子交给大模型。条子是准的，因为它就是这样算出来的。大模型读了条子，表现只比没拿到条子时好一点点。在这次测试里，问题出在读条子的那一方，不在条子。

Pitches from the kits that were not used, and why:
- "the label was fine / nearly perfect (.955)" (kit 1 founder, pick 3 founder): correction 1.
- "finding it takes one line of code" (kit 1 founder): correction 6.
- "A small AI told a bigger one exactly how sure it was" (kit 2 journalist): correction 2; the sender did not say it.
- "Before building a state channel, run the label-only audit" (kit 1 lab lead): uses the rejected line name.
- "the audit is the thing you can sell today" (pick 1 founder): no sale or customer has been measured; kept only as a BET line.

## 5. Iconic-figure spec

All three picks agree on the design; this spec merges them and fixes one gap (per-bar intervals do not exist yet; see the note under the main panel).

### Figure 1: the unread label (clean passages only)

Source: `analysis/cloud/verbal/audit.json`, key `clean`; the same numbers are in `tab:confidence` (main.tex lines 226 to 237).

Main panel:
- y axis: AUROC of receiver first-token entropy against sender semantic entropy (median split), from .5 to 1.0. Do not truncate below .5: the marker at .955 has to fit, and a truncated axis makes the .532 to .597 spread look large.
- Chance: thin solid line at .5.
- Dashed floor at question-only .532, labelled "no message".
- Dashed ceiling at text .597, labelled "receiver reads the passage".
- Bar: confidence word, receiver level, .552.
- Hollow marker at .955 directly above the word bar. Legend: "word scored as a label (tercile of sender SE; no receiver)".
- Vertical bracket from .552 to .955, labelled "read-out gap".
- Cache: a filled dot at .585, between floor and ceiling, with no hollow marker (it has no label).
- Optional: numeric-confidence bar .548 with its own hollow marker at .870. Leave it out if the panel gets crowded; the word arm alone carries the point.

Right-hand strip (same figure, narrow): the paired, passage-grouped bootstrap intervals that exist (344 passages, 2,000 resamples; line 253), drawn as horizontal intervals around a zero line:
- cache minus question-only [-.005, .111]
- cache minus word [-.027, .094]
- cache minus text [-.066, .047]
All three include zero on clean passages. This strip is what stops a reader from taking the dot's position as a cache win.

Note on intervals: pick 2 asks for 95% intervals on each bar. Per-arm intervals are not in audit.json or main.tex. Either compute them from the saved per-item scores (local, no cloud job) before drawing, or draw only the paired-difference strip above. Do not draw bar intervals that have not been computed.

Caption, first sentence: "The vertical distance from marker to bar is the read-out gap."
Caption, second sentence (scope): "One K2-Horizon-7B base receiver, one completion prompt, no instruction to hedge; the marker's height reflects how the label was built."

What moves to the appendix: contradicted and removed passages (word .542 / .592 under labels .942 / 1.000). The 1.000 on removed passages is a tie artefact: 22% of items sit at the threshold and the tercile boundary coincides with it (line 262).

Sketch test: someone who saw the figure once should be able to redraw it as "a short bar near the floor, a hollow dot near the top, a bracket between them". If a draft fails that test, remove elements until it passes.

### Figure 2: within-item swap

Source: main.tex line 264 and audit.json `within_item_removed`.
- Four bars, Spearman between the receiver's entropy change and the sender's SE change, clean to removed, question held fixed: cache .280, text .322, word .146, another item's cache .004.
- Same-sign rates (88 / 90 / 49 / 47%) as small annotations only, with a reference line at 91.5% (the share of items where the sender's SE rises), since that caps them.
- Second, lighter row or inset for clean to contradicted: cache .208, text .325, word .179, another item's cache .004.
- Caption says the text path tracks at least as well as the cache, so the figure shows that the cache's content moves the receiver, not that the cache beats text.

### What the figures must not show

- No bar or marker that implies the cache beats text or no message on clean passages.
- No "recompute" label on any figure element (SUGGESTED, line 70).
- No cost panel in Figure 1. The 72 MiB, 604 ms and 24 ms belong in `sec:cost`.
