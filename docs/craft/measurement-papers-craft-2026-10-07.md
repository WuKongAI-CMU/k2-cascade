# Craft notes: how measurement papers turn a number into a story (2026-10-07)

Corpus read in full text (pdftotext of the arXiv PDFs; Nature paper via PMC):
Schaeffer et al. 2023 "Are Emergent Abilities of LLMs a Mirage?"; Henderson et al. 2018 "Deep RL that Matters";
Guo et al. 2017 "On Calibration of Modern Neural Networks"; Liu et al. 2023 "Lost in the Middle";
Dziri et al. 2023 "Faith and Fate"; Kadavath et al. 2022 "Language Models (Mostly) Know What They Know";
Farquhar et al. 2024 "Detecting hallucinations ... semantic entropy" (+ Kuhn et al. 2023 ICLR abstract).
Our side: docs/paper/main.tex (v5), docs/thesis/thesis.tex, goal/GOAL.md. Quotes from others are <= 15 words.

## The one fact that matters most

Count the numbers in the abstracts. Mirage: 0. Deep RL: 0. Calibration: 0. Lost in the Middle: 0. Faith and Fate: 0.
Mostly Know: 0. Semantic entropy: 0. Ours (main.tex v5): about forty-five, formatted as "a / b / c" triples.
Every paper in the corpus puts the finding in words and one figure, and leaves the numbers to the body.
Our thesis already knows this: its front matter says "One pair of numbers runs through the document: .552 against .532",
and the paper's abstract does not use it.

## Moves (each with where it goes in our text)

1. Title names the question or the verdict, not the opponent's metaphor.
   Mirage asks a question the abstract answers; "Lost in the Middle" names the curve; "(Mostly) Know" puts the hedge in the title.
   Ours, "Where Confidence Gets Laundered", is Shi et al.'s word. Subtitle should carry our verdict:
   "Where Confidence Gets Laundered: The Reader, Not the Message". The thesis chapter title can stay.

2. Quote the standing belief in its own words, then "we call into question".
   Mirage reproduces Wei et al.'s definition verbatim before doubting it. Thesis ch.1 does this already
   ("The field's standing belief, in its own words"). Paper intro para 1 should quote 2606.20662's title clause
   ("Why Uncertainty Needs a Latent Carrier") rather than paraphrase.

3. Turn the diagnosis into two predictions and say which one the data confirmed.
   Mirage: "make, test and confirm three predictions"; Faith and Fate: "We propose two hypotheses."
   Paper intro para 2 currently lists "two possible causes". Rewrite as: if the loss is in the message, the label scored
   alone is low; if at the reader, the label is high and the receiver low. Then one sentence: the second is what we found
   (.955 vs .552). The cache arm gets its own pair, named "relay" vs "recompute" (see Naming), and the honest line is
   "the probes do not decide between them".

4. One handle, repeated until the reader can say it back.
   Guo's Figure 1 (LeNet vs ResNet reliability diagram); Liu's U-curve; Kadavath's Figure 1.
   Our handle is ".955 / .552 / .532": label alone, reader handed the label, reader handed nothing.
   Abstract rewrite (replace all triples): "Scored on its own, the confidence word predicts the sender's uncertainty almost
   perfectly; read by the receiver it does no better than no message at all. A mapped KV cache with no label leaves the
   receiver tracking the sender about as well as reading the passage. The text arm's loss is at the reader, not in the message."
   Then the three numbers once, then the one-line recommendation. Everything else moves to Table 3.

5. Headings as findings or questions.
   Henderson: "Can random seeds drastically alter performance?"; Liu: "Is More Context Always Better?";
   Kadavath: "Replacing an Option with 'None of the Above' Harms Performance and Calibration".
   Paper Sec. 4 "How Much Uncertainty Survives Each Handoff" -> "The Reader Handed the Word Does No Better Than the Reader Handed Nothing".
   Sec. 5 "Where the Loss Occurs" is already a question; Sec. 6 "Two Objectives That Did Not Move the Loss" is already a finding.

6. The surprise frame, stated as surprise.
   Guo: "we discover with great surprise that modern neural networks are no longer well-calibrated" (Guo et al. 2017).
   Our Sec. 5(a) says "The audit is a sanity check, not a finding". That sentence buries the handle. Keep the caveat about
   the construction (tercile vs median) but lead with the contrast: a label that is nearly a perfect detector, read by a
   7B model, becomes a coin flip.

7. Negatives filed as findings, each with what it rules out.
   Kadavath lists a negative as a contribution bullet and closes a section with "Thus we conclude that we only find partial
   generalization on this task." (Kadavath et al. 2022). Faith and Fate is a negative paper whose mechanism has a name.
   Thesis Part III ("Six negatives, filed as findings", each with "What it taught:") is this move done well.
   Paper Sec. 6 should end with the rule-out sentence, not the fix: "neither the receiver's own text-path distribution nor
   its per-token entropy contains the sender's spread", then the untested fix in one clause.

8. Name the mechanism even while it is a hypothesis.
   Dziri: "reducing multi-step compositional reasoning into linearized subgraph matching" (Dziri et al. 2023);
   Liu borrows "primacy bias"/"recency bias"; Farquhar carves "confabulations" out of "hallucinations" with an operational
   definition: "fluently make claims that are both wrong and arbitrary" (Farquhar et al. 2024).
   Our cache-arm reading, "evidence crosses and the receiver recomputes", is unnamed in six places. Name it once in Sec. 2:
   "relay (the sender's spread crosses as such) versus recompute (the evidence crosses, the receiver re-derives its own spread)".

9. The recommendation costs one line and is phrased as a rule.
   Guo: temperature scaling "surprisingly is often the most effective" (Guo et al. 2017), one parameter.
   Henderson: report all hyperparameters, run more seeds. Mirage: include controls. Liu: new evaluation protocols.
   Ours already is one line ("Score the message, then the reader"). Make it a reporting rule and put it in three places:
   last sentence of the abstract, a one-line box at the end of the intro, last line of the conclusion (thesis already does the last):
   "Any table with a verbal-confidence baseline reports the label-only AUROC next to the receiver AUROC."

10. One explicit non-claim sentence, early.
    Mirage: "nothing in this paper should be interpreted as claiming" LLMs cannot show emergence (Schaeffer et al. 2023).
    Guo: "Though we cannot claim causality" (Guo et al. 2017). Farquhar: the method "explicitly does not directly address
    situations in which LLMs are confidently wrong" systematically (Farquhar et al. 2024).
    Paper intro, end of para 3: "Nothing here says text handoffs cannot launder confidence; it says that in the one handoff
    measured here, the signal was lost at the reader, not in the message." Thesis has "What this document does not claim" already.

11. Floor and ceiling named in the table, and a minimal synthetic testbed beside the real task.
    Liu contextualises with closed-book (floor) and oracle (ceiling); the key-value task is "designed to be a minimal
    testbed for the basic ability" (Liu et al. 2023). Dziri pairs multiplication/puzzles/DP with GPT-4 runs.
    Our Noma test and question-only / text arms are the same pair. Table 3 caption: "question-only is the floor, text the
    ceiling"; Sec. 2 one sentence: "Noma is the minimal testbed for whether anything crosses at all."

12. Close with the overturning experiment and an invitation.
    Dziri: "We invite the broader research community" with more compute (Dziri et al. 2023). Henderson closes on
    "in what setting would this work be useful?" (Henderson et al. 2018). Kadavath: "We hope these observations lay the groundwork".
    Paper conclusion already names the instruction-tuned receiver as the experiment that would overturn the claim; add the
    field-facing line from the thesis conclusion. GOAL.md's kill criteria are the same move at program scale.

## Naming: how the corpus coined or packaged its terms

- Mirage: no coinage; a plain noun set against the field's own term, in a question title.
- Lost in the Middle: an idiom becomes the phenomenon's name; mechanism words borrowed from psychology (primacy, recency).
- Calibration: reused old words (calibration, reliability diagram); the method is named by its single parameter (temperature).
- Deep RL that Matters: a slogan title; the structure is carried by question headings, not by new terms.
- Faith and Fate: literary pairing plus epigraph; the mechanism named as a compound noun phrase; metrics named operationally
  (relative information gain, partial computation accuracy).
- Mostly Know: hedge in parentheses in the title; quantities named as pronounceable probabilities P(True), P(IK); a Glossary of observables.
- Semantic entropy: method named by what it respects (meaning); a sub-category (confabulation) carved from a popular word with a seed-sensitivity definition.
- Ours: "confidence laundering" is the opponent's; "read-out gap", "label-only audit", "message-level / receiver-level signal"
  are ours and are the reusable artefact; thesis tags SHOWN / SUGGESTED / BET are a naming device the corpus does not have and should keep.
  Missing name: relay vs recompute.

## Anti-patterns (ours, against the corpus)

- Abstract as a table: ~45 numbers against 0 in every corpus abstract.
- Title that uses the position paper's word for our own finding.
- "The audit is a sanity check, not a finding": underselling the handle in the sentence that should carry it.
- The same hedge ("established on one variant of three", "intervals include zero") repeated four times in the body; say it once in Results, once in Limitations.
- A full boxed retraction in the paper; Mirage handles disagreement in one sentence. Paper: two sentences; thesis keeps the full account.
- Contributions that re-describe method ("a two-level report ... with controls ...") instead of stating findings, as Kadavath's bullets do.
- Non-claims scattered through the conclusion instead of one explicit sentence in the intro.
