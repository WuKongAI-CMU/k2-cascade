# Craft notes: documents that launched research programs

Date: 2026-10-07. Corpus read (seven, each from the primary text or its PDF where
reachable, otherwise two independent summaries): Sutton, *The Bitter Lesson* (2019);
Amodei et al., *Concrete Problems in AI Safety* (2016); Anthropic, *Core Views on AI
Safety* (2023); Karpathy, *Software 2.0* (2017); Huh et al., *The Platonic Representation
Hypothesis* (2024); Aschenbrenner, *Situational Awareness* (2024, intro, ch. I, "The
Project"); LeCun, *A Path Towards Autonomous Machine Intelligence* (2022, via two
summaries; the OpenReview PDF is behind a browser check). Our documents: `docs/paper/
main.tex` (v5), `docs/thesis/thesis.tex`, `goal/GOAL.md`. This note is about craft, not
content; every "apply" line names a place in one of our three files.

## What these seven do that we do not yet do

### 1. The thesis is the first sentence, and it is already a measurement.
Sutton: "The biggest lesson that can be read from 70 years of AI research is..." The
sentence contains the claim, its scope (70 years) and its evidence class (history). The
Platonic paper puts its one-sentence hypothesis in a box on page 1 before the abstract is
finished. Our paper's abstract opens with a question ("How much of a sender's uncertainty
survives..."), which spends the first sentence on setup. The thesis abstract does it right:
".955 ... .552 ... within .02 of a receiver given no message at all (.532)."
*Apply:* move the paper abstract's first sentence to the thesis pattern: "A confidence word
scored as a label predicts the sender's semantic entropy at AUROC .955; the receiver that
reads it lands at .552, .02 above a receiver given nothing. The loss is at the reader."

### 2. One repeated frame per example, so the reader learns the pattern and predicts it.
Sutton runs chess, Go, speech, vision through the identical frame: human-knowledge
approach first, dismay, then "search and learning" win. By the third case the reader is
completing the sentence. Karpathy's domain list does the same with "used to ... now ..."
*Apply:* the paper's §3–§5 each introduce controls in a different order. Give every arm one
fixed four-beat sentence: arm / number / control / difference. The thesis "SHOWN" box already
has the beat ("number, then control"); carry that exact beat into the paper's tables'
captions and the first sentence of each results paragraph.

### 3. Name the lesson by its emotional cost, then say whose cost.
"Bitter" is a one-word confession about the audience: the lesson hurts the researchers who
hold it. Karpathy's "2.0" tells programmers their craft is being versioned. Our title,
"Where Confidence Gets Laundered", borrows an opponent's metaphor (2606.20662) and then
spends a paragraph relocating it. The thesis's one-line thesis "It does not; this reader
does" is the sharper title material.
*Apply:* paper subtitle candidate: "The Reader Launders, Not the Message". Keep the
borrowed word, flip its object in the title itself, so the correction is visible before
the abstract.

### 4. A four-point numbered statement of the lesson, late, after the stories.
Sutton: "1) AI researchers have often tried to build knowledge into their agents, 2) this
always helps in the short term... 3) in the long run it plateaus... 4) breakthrough
progress eventually arrives by an opposing approach." Narrative first, enumerated law
second. The Platonic paper likewise: evidence (§2), then named mechanisms (§3–4), then
implications (§5).
*Apply:* the paper has no such paragraph. Add one at the top of §Conclusion: "1) a label
can carry the sender's uncertainty almost perfectly; 2) a base receiver under a completion
prompt does not read it; 3) a state message with no label is read, and the receiver
recomputes; 4) therefore score the message before scoring the reader." Four clauses, the
third carrying the honest limit.

### 5. A running toy that every section returns to.
Concrete Problems introduces the office cleaning robot once and then shows each of five
problems through it: the vase, disabling its vision, the mop in the outlet. The reader has
one scene and five variations. Our paper's "Noma is amber. Vela is indigo." is that
scene, but it appears only in the binding test.
*Apply:* in the thesis ch. "Text by inertia" there is a "toy example" of the read-out gap.
Reuse the Noma names there and in the uncertainty chapter (a sender unsure whether Vela
is indigo or amber; what the word carries vs what the cache carries). One cast, every
chapter.

### 6. Each problem gets a "Potential Experiments" paragraph that a stranger could run.
Every Concrete Problems section ends with experiments stated at toy scale ("a toy
environment with some simple goal (like moving a block)"), with the next step after the
toy. The audience is given work, not opinions. Our thesis program chapters have
Hypothesis / Done / Next / Kill, which is better than most; the paper has only "the
first experiment we would run."
*Apply:* add a one-paragraph "Experiments this audit enables" to the paper's conclusion
or limitations: (a) any verbal-confidence baseline: report label-only AUROC next to receiver
AUROC; (b) instruction-tuned receiver on the same 377 items; (c) three projector seeds
through the protocol. Each with the file that holds the items (`analysis/cloud/verbal/`).

### 7. Scenarios, not hedges: the plan is justified under three futures.
Anthropic's "Core Views" refuses to pick between optimistic / intermediate / pessimistic
and says what the program is worth in each; then "a 'portfolio' of safety work that can
help us succeed across a range of different scenarios." Uncertainty is carried by the
structure, not by adverbs.
*Apply:* GOAL.md's north star is four sentences "from most to least certain", which is
the same device unlabelled. Label it. In the thesis ch. "Horizon", the four kills are
listed; add the three-scenario reading next to them: (i) bytes solved and uncertainty
crosses: economy chapter is the thesis; (ii) bytes unsolved, uncertainty recomputed: the
thesis is instrument + trigger + audit; (iii) instruction-tuned receiver reads the word:
the thesis is the audit alone. One sentence each, what we still publish in each.

### 8. Pre-empt the strongest objection in its own subsection, by its own name.
Aschenbrenner gives "the data wall" a heading; the Platonic paper gives "Counterexamples
and limitations" and "Lots left to explain" their own headings and leads with the hardest
one ("Different modalities may contain different information"). Naming the objection in
a heading signals that the author got there first.
*Apply:* our paper's Limitations is one dense paragraph. Split it into named sub-heads
that are the referee's sentences: "The receiver is a base model under one prompt";
"The effect is .03–.06 and excludes zero on one variant"; "72 MiB is not a deployment".
Same words, headed.

### 9. Count the thing, then let the count carry the inevitability.
"Counting the OOMs" converts a prediction into arithmetic the reader can redo. Sutton's
engine is Moore's law. We have the equivalent and bury it: 144 KB per token, 72 MiB per
512 tokens, 604 ms vs 24 ms saved.
*Apply:* the thesis price sheet (`tab:pricesheet`) should appear once in the paper too,
reduced to three rows, under the heading "What the message costs, in units the reader
can recompute". GOAL.md sentence 3 ("cost, value, fraud → price, reputation, audit")
should cite the row that gives each its number.

### 10. Give the reader a job in one line, and make it cheap.
Sutton: build "methods that scale with computation". Karpathy: "Who will build the first
Software 2.0 IDE?" Concrete Problems: five problems "ready for experimentation today".
Our thesis closes with "The request to the field is one line"; the paper's abstract ends
with "a one-line audit for any verbal-confidence baseline". This is our best move and it
is already there.
*Apply:* make it typographically unmissable in the paper: a boxed single sentence at the
end of the introduction, repeated verbatim as the last sentence of the conclusion. The
Platonic paper boxes its hypothesis; we box the request.

### 11. Say what you are not, early, in a sentence that costs nothing.
LeCun's prologue labels the document a position paper with no results; Concrete Problems
spends a paragraph declining the "superintelligent agents" framing; Anthropic says "we may
be wrong". Each buys the right to speak confidently afterwards. The thesis's "What this
document does not claim" section and the SHOWN / SUGGESTED / BET tags do this better than
any document in the corpus.
*Apply:* the paper lacks the sentence. Add to the end of the introduction: "This is a
measurement paper on one model pair; the position it tests is someone else's and the
program it suggests is in the companion thesis."

### 12. Turn the document's own retraction into evidence of method.
The paper withdraws the "log 3 nats" deduction inside a box titled "Capacity, stated
correctly." No corpus document does this; Aschenbrenner's lack of it is his weakness.
Keep it, and move the box's last sentence ("no measurement in this paper tests it") to the
first position so the reader sees the method before the mathematics.

## How the corpus names things
- Coin a noun phrase that is also a verdict: "bitter lesson", "Software 2.0", "confidence
  laundering" (the opponent's; we inherited it). Ours that qualify: **read-out gap**,
  **label-only audit**, **deranged-sender projector**. Ours that do not travel: "two-level
  report", "message-level signal / receiver-level signal" (descriptive, not memorable).
- Name every hypothesis and box it (Platonic: Multitask Scaling, Capacity, Simplicity
  Bias Hypotheses). Our thesis conditional in ch. "The thesis sentence" is unnamed; call
  it the **state-return hypothesis** and box it.
- Name the controls as characters: "derange", "follow rate", "Noma". Keep.
- Borrow one classical frame and use it once (Plato's cave, Kahneman's Mode-1/Mode-2).
  We have "laundering" from finance; the matching second term is **audit**, already in
  use. Do not add a third metaphor.
- Number the law after the stories (Sutton's 1–4); never before.

## Anti-patterns seen in the corpus, to avoid
- Inevitability by extrapolation without a kill (Aschenbrenner: "trust the trendlines").
  Our thesis already pairs every bet with a kill; keep the pairing in the paper too.
- A balanced "benefits / limitations" list that never says which side wins (Karpathy).
  Our Limitations paragraph should end with the ranking, not the list.
- Borrowed authority over named evidence ("you can see the future first in San
  Francisco"). Every number in our abstract has a file; keep it that way.
- Hedging adverbs doing the work structure should do. Where we write "nearly", "barely",
  "about as well", the interval is the hedge; print it.
- A question as the first sentence of an abstract (our paper v5). Statement first.
- A position paper with no experiment a stranger can run (LeCun's weakness; also the
  reason Concrete Problems aged better).
