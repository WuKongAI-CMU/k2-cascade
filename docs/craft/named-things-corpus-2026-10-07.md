# Craft notes: papers that named a thing and the field kept the name (2026-10-07)

Reader's brief: study craft, not content. Nine papers, each read twice (arXiv abstract page and the ar5iv full text): Attention Is All You Need (1706.03762), Deep Residual Learning (1512.03385), BERT (1810.04805), Scaling Laws for Neural Language Models (2001.08361), Training Compute-Optimal LLMs / Chinchilla (2203.15556), Emergent Abilities (2206.07682), Lost in the Middle (2307.03172), Textbooks Are All You Need (2306.11644), DPO (2305.18290). Quotes are at most 15 words and attributed. Everything said about our own writing comes from docs/paper/main.tex (v5), docs/thesis/thesis.tex (2026-10-07) and goal/GOAL.md.

## What the nine have in common

1. **First sentence: a flat declarative about the field, never a question, never about the authors.** Transformer: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks" (Vaswani et al.). ResNet: "Deeper neural networks are more difficult to train" (He et al.). Chinchilla: "We investigate the optimal model size and number of tokens" (Hoffmann et al.) is the only "we" opener and it still names the question in the field's terms. None of nine opens with a question. Our paper abstract opens with a question ("How much of a sender's uncertainty survives..."); our intro opens correctly ("When one language model hands work to another it usually writes a message").

2. **The abstract stages one claim and spends one or two numbers on it.** Transformer: 28.4 BLEU. ResNet: 3.57% error. phi-1: 50.6% HumanEval. Chinchilla: 67.5% MMLU. Scaling Laws and Emergent Abilities and Lost in the Middle: no number at all, one shape in words ("power-law", "not present in smaller models but is present in larger models", "highest when relevant information occurs at the beginning or end"). Our paper abstract carries roughly thirty numbers in triplets. The thesis abstract is closer to the corpus: .955 against .552 against .532 and stops.

3. **The pivot clause sits inside the abstract.** "This paper instead discusses an unpredictable phenomenon" (Wei et al.). "Unlike recent language representation models" (Devlin et al.). "We find that current large language models are significantly undertrained" (Hoffmann et al.). One clause that says which belief is being displaced. Our thesis has this sentence already, as the last line of the Part I opener: "The claim is that text launders confidence. It does not; this reader does." It is not in either abstract.

4. **The name is introduced in the abstract, with a definite article or a one-sentence criterion, and never justified.** "a new simple network architecture, the Transformer" (Vaswani et al.) — the full text never explains the word. "BERT, which stands for Bidirectional Encoder Representations from Transformers" (Devlin et al.) — the expansion is the claim. "We consider an ability to be emergent if it is not present in smaller models" (Wei et al.). Our names (message-level signal, receiver-level signal, read-out gap, label-only audit) first appear in paper intro paragraph 2 and in thesis ch. 1 "Contributions"; neither abstract contains any of them.

5. **The iconic figure is the finding, drawn so that a reader can redraw it from memory; it is not the apparatus.** ResNet Fig. 1: two training-error curves, the 56-layer plain net above the 20-layer. Lost in the Middle Fig. 1: a U. Emergent Fig. 2: flat, then a jump, eight times. Scaling Laws Fig. 1: three straight lines on log-log axes. Chinchilla Fig. 1: three predictions overlaid on Kaplan's. The exception is Transformer Fig. 1, the architecture diagram, and that is because the artefact is the contribution. Our paper's first figure (fig:arms) is Noma accuracy by arm: the instrument's calibration. The finding figure (fig:audit) comes third, is two panels, and encodes the read-out gap as "the vertical distance from marker to bar". The thesis's fig:handle is the single-panel version.

6. **Bold finding headers that are themselves sentences.** Scaling Laws, Summary: "Performance depends strongly on scale, weakly on model shape" (Kaplan et al.), then seven more. Each is quotable alone. Our thesis SHOWN box does this ("Content crosses." "The label carries the signal; this reader discards it."). Our paper's Contributions (1)-(4) are clause-chains with parenthetical cross-references.

7. **Subtitle is a sentence a reader will repeat, not a topic.** "Your Language Model is Secretly a Reward Model" (Rafailov et al.). "How Language Models Use Long Contexts" (Liu et al.). Our paper subtitle, "Message versus Reader in a Model Handoff", is a topic. Our thesis subtitle, "The Label Carries the Signal, This Reader Discards It", is a sentence.

8. **Borrowed vocabulary is borrowed once, with the source named.** Emergent Abilities quotes Anderson's "More is different" and adds "phase transition". Lost in the Middle names the serial-position effect, primacy and recency. The borrowed term supplies a shape and a precedent. GOAL.md borrows market vocabulary (cost, value, fraud; price, reputation, audit); the thesis uses it across Part I and Part IV without one paragraph that declares the borrowing.

9. **The method is named by what it removes.** "Direct" Preference Optimization: no reward model, no RL. "dispensing with recurrence and convolutions entirely" (Vaswani et al.). "residual learning" names a reformulation, and "degradation problem" names the enemy in the same paper (He et al.). Our "label-only audit" is already this kind of name: the receiver is removed.

10. **The close is a rule the practitioner can apply without the paper.** Chinchilla's abstract: "for every doubling of model size the number of training tokens should also be doubled" (Hoffmann et al.). Our conclusion's first sentence, "Score the message, then score the reader", is that rule. It appears last in the paper and last in the thesis; in Chinchilla the rule is in the abstract.

11. **Titles that take a side get adopted as templates.** "Textbooks Are All You Need" echoes the 2017 title deliberately; "Secretly" in DPO. The corpus never does this for a result it hedges in the same breath. Our result is narrow (one reader, one prompt, 377 items); the assertive title must therefore name the thing that is SHOWN, not the thing that is SUGGESTED.

12. **Nothing stands between the abstract and the first figure except the problem.** No corpus paper puts a correction of its own earlier draft before its results. Our paper's "Capacity, stated correctly" box, which withdraws the v4 log 3 deduction, sits in Section 2 before any finding.

## How the corpus coined or packaged its terms

- Transformer: bare noun with the definite article in the abstract's third sentence; no etymology anywhere.
- BERT: acronym whose expansion carries the claim ("Bidirectional"); first sentence of the abstract.
- ResNet: the mechanism is the name (residual, shortcut connections) and the enemy gets a name too (degradation problem); both in the intro's first page, Fig. 1 shows the enemy.
- Scaling Laws: a regularity borrowed from physics; the abstract says "power-law" in its second sentence.
- Chinchilla: a mascot name for the artefact, introduced only as the test of a hypothesis; the finding's name is the adjective "compute-optimal".
- Emergent abilities: a one-sentence criterion in the abstract, then Anderson, then "phase transition".
- Lost in the Middle: the title is the finding as an idiom; the abstract never uses the phrase; Fig. 1 is the phrase.
- Textbooks / phi-1: template-echo title; "textbook quality" in scare quotes in the abstract; four adjectives define it in the body (clear, self-contained, instructive, balanced).
- DPO: named by subtraction; the subtitle states the insight as a secret.

## Rewrites for our files (facts unchanged)

- Paper abstract, first sentence: replace the question with the declarative plus pivot. "When one model hands work to another it writes text, and a confidence word attached to that text is said to be laundered. We measure where." Then one pair of numbers: .955 as a label, .552 after the reader (clean passages), .532 with no message. Then the cache in one sentence (63% binding, 53.0 vs 54.7 F1, .585). Triplets move to Table confidence.
- Paper abstract, name: add one sentence. "We call the difference between the two the read-out gap, and the measurement the label-only audit."
- Paper subtitle: take the thesis's. "Where Confidence Gets Laundered: The Label Carries the Signal, This Reader Discards It." The topic subtitle moves to the running title if needed.
- Figure 1: the thesis fig:handle, single panel, one vertical segment per arm from message-level to receiver-level AUROC; the word arm's segment is long, the cache has one point. fig:arms (Noma) becomes Figure 2 in the calibration section.
- Contributions: four bold sentences in the thesis SHOWN style, cross-references after the sentence, not inside it.
- Conclusion rule: "Score the message, then score the reader" also as the abstract's last sentence (it is already the thesis abstract's second-to-last sentence).
- Capacity box: move to Limitations or the appendix with the retracted deduction; keep the withdrawal, lose its position before Figure 1.
- Thesis ch. "What an economy of state would need": one paragraph that declares the market borrowing (cost, value, fraud; price, reputation, audit) with its source (goal/GOAL.md), the way Emergent Abilities declares Anderson, so the rest of Part IV can use the words without re-explaining.
- GOAL.md north star: already a four-sentence declarative ladder; the craft match is Scaling Laws' summary (bold sentence, then evidence). No change needed beyond keeping sentence 4 marked (bet).

## Anti-patterns (things we do that none of the nine does)

- Opening the abstract with a question.
- Thirty numbers in the abstract, in triplets.
- Taking the title's key term from the paper being corrected ("confidence laundering" is 2606.20662's; the thesis says so: "The claim is in the title").
- Coining names in intro paragraph 2 and never letting them reach the abstract.
- A first figure that shows the apparatus (Noma accuracy) rather than the finding.
- A self-correction box before the first result.
- Hedge and claim in the same sentence of the abstract ("cache−word excludes zero on one variant of three"): the corpus puts the claim alone in the abstract and the limits in a Limitations section. Our rule from memory (confident positioning is not exaggeration; unrun things are said plainly) is compatible: the hedge stays, one sentence later.
- Contributions as clause-chains with parenthetical cross-references.
