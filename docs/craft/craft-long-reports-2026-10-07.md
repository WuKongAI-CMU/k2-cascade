# Craft notes: long-form reports that keep readers (2026-10-07)

Reader studying craft, not content. Corpus read (full text, not abstracts): Anthropic "Towards Monosemanticity" (2023) and "Scaling Monosemanticity" (2024); DeepMind "A Generalist Agent" (Gato, TMLR 2022) and "Gemini: A Family of Highly Capable Multimodal Models" (v1 report); OpenAI "GPT-4 Technical Report" (arXiv 2303.08774, with system card); Distill "Zoom In: An Introduction to Circuits" (2020), "How to Use t-SNE Effectively" (2016), "Research Debt" (2017); AlphaFold 2 (Nature 2021, via PMC summary); Yarin Gal, *Uncertainty in Deep Learning* (Cambridge 2016, 170 pp.); Ilya Sutskever, *Training Recurrent Neural Networks* (Toronto 2013, 100 pp.).

Our documents: `docs/paper/main.tex` (v5, "Where Confidence Gets Laundered"), `docs/thesis/thesis.tex` (45 pp., "What this document claims" + program Part IV), `goal/GOAL.md`. Facts about our work come only from those three files.

## What the corpus does that 40+ pages survive on

1. **A results list before any method.** Towards Monosemanticity opens with "Summary of Results": seven bold one-sentence claims, each followed by one sentence of evidence and a pointer to the section. Scaling Monosemanticity repeats it as "Key Results" (six bullets). GPT-4 puts the predictable-scaling result in the abstract and Figure 1 on page 2. Reader can stop after page 1 and leave with the claims.
2. **Figure 1 is the paper.** Gato's Figure 1 (one network, many embodiments) sits above the introduction; GPT-4's Figure 1 is predicted-vs-actual loss; AlphaFold's Figure 1 answers "what was built, how well, why the architecture matters" in five panels. Every one is a one-picture statement of the headline, not a method diagram.
3. **Claims tagged by epistemic status.** Zoom In calls its three claims "deliberately speculative" and models them on Schwann's three cell claims, one of which "turned out to be horribly wrong". Gal's chapter openings say "we assess", "we turn to a more theoretical analysis". Towards Monosemanticity's Discussion opens "This work has persuaded us"; belief verbs for belief, measurement verbs for measurement.
4. **Scope-and-withholding section up front.** GPT-4 §2 "Scope and Limitations of this Technical Report" says in four sentences what the report will not disclose and why. Gato's abstract ends "document the current capabilities"; the Conclusions section is the only place with "we can build".
5. **Section headings that are sentences or verdicts.** t-SNE: "Cluster sizes in a t-SNE plot mean nothing". Towards Monosemanticity: "The feature is not a neuron", "Retention is the wrong headline" is already our idiom. Gal: "The Importance of Knowing What We Don't Know".
6. **One recurring example object.** Scaling Monosemanticity's Golden Gate Bridge feature (named ~20 times in the page); Zoom In's curve detectors run through all three claims; Gal's regression toy appears in ch. 4 and ch. 6.
7. **Separate "what we did" from "what we believe" by section, not by hedge words.** Gato: §4 Capabilities, §5 Analysis, §7 Broader Impact, §8 Limitations, §9 Conclusions. Gemini: §5 Evaluation, §7 Responsible Deployment, §8 Discussion. GPT-4: body vs. system card. The belief sections are short and come last.
8. **Chapter-opening paragraphs that are maps.** Gal ch. 4 and ch. 6 each start with one paragraph: what this chapter assesses, in what order, what is deferred to the next. Gemini's introduction ends "In the following sections, we first ... then ... Next ... Finally". Sutskever §0.1 maps chapters to papers.
9. **Theses state the contrarian claim in the abstract.** Sutskever: "directly contradicts widespread beliefs" (about first-order methods); the Summary of Contributions repeats "was considered completely unsolvable ... prior to our work". Gal's abstract ends with the theoretical part as a second act, not an afterthought.
10. **Terms coined by definition-then-use, with a glossary.** Zoom In defines feature, circuit, universality once each in the claim block and has a glossary. Towards Monosemanticity introduces "feature splitting", "finite-state automata" in the summary and uses them unchanged. Gemini names Ultra/Pro/Nano in Table 1 before any result. Gato: full descriptive phrase first ("multi-modal, multi-task, multi-embodiment generalist policy"), then the name.
11. **Negative and surprised results get their own headings.** Towards Monosemanticity: "Features which seemed like Bugs", "Are 'Token in Context' Features Real?"; Sutskever ch. 7 opens with results that "appeared to disagree with the commonly held belief". Our thesis ch. "Six negatives, filed as findings" is already this.
12. **Closing is one reframing paragraph, not a summary.** t-SNE ends by resolving the opening tension; Zoom In ends with "Interpretability as a Natural Science" (a stance, labelled as one); Research Debt ends with a proposal. None re-lists results.

## Anti-patterns the corpus avoids (and we sometimes do)

- Numbers in the abstract without the comparison that makes them mean something (GPT-4 gives "top 10%" and "bottom 10%" of GPT-3.5 side by side). Our paper abstract carries nine AUROCs in slashes before the reader knows what .5 and 1.0 mean here.
- Section headings that name a method instead of a finding ("Four-point probes" vs. "The cache arm's loss is not localised").
- Contributions lists that re-describe the method rather than the claim (Gato has none; Towards Monosemanticity's Summary of Results is the contributions list).
- Mixing the plan into the result chapters. Gato keeps future work in §8; our thesis keeps Part IV separate, but the paper's Limitations paragraph carries the pre-registration history that belongs in a method box.
- Dense first paragraph. Every corpus opener is a plain two-sentence setup (Gato: "There are significant benefits to using a single neural sequence model across all tasks."). Ours opens with a citation and a coined term in sentence two.
- A footnote-sized theory box in the middle of methods (our "Capacity, stated correctly" box). The corpus puts retractions in the Discussion or a dated "Updates and Corrections" footer (Distill convention).

## Where to apply, concretely

Paper (`docs/paper/main.tex`):
- Add a "Summary of results" block of five bold one-liners between the abstract and §1 (ICML allows it as the first paragraph of the intro): label carries signal / reader discards it; cache carries content (63.0 vs 12.5); ordering text ≳ cache > word, one interval excludes zero; probes do not localise; 72 MiB per message. Each with a `\cref`.
- Make Figure 1 the read-out gap picture (the thesis already has it: hollow markers for label scored directly, bars for the receiver). Move it above the introduction.
- Rename §5 "Where the Loss Occurs" to the verdict: "The word's loss is at the reader; the cache's is not localised".
- Move the capacity/retraction box to the end of Related Work as "Correction to v4", two sentences.
- Abstract: lead with the pair .955 vs .552 and say what "no message" gives (.532) before any other number; drop the slash triples from the abstract and keep them in Table 1.

Thesis (`docs/thesis/thesis.tex`):
- "What this document claims" already does move 1 and 3 (SHOWN / SUGGESTED / BET). Add the Zoom In sentence form: say once that the BET list is "deliberately speculative" and that one of the north-star sentences is expected to be wrong.
- Every chapter of Parts II–III: first paragraph is a map (what is assessed, what is deferred), Gal style. Most already start with a figure; add the map sentence under it.
- "The handle" (.552 against .532) is our Golden Gate Bridge; name it in each results chapter's first paragraph, not only in the front matter.
- Program chapters: keep the Hypothesis / Done / Next / Kill paragraph headings; they are the GPT-4 §2 move (scope and withholding) applied per chapter.

Manifesto (`goal/GOAL.md`):
- The four north-star sentences are already ordered most-to-least certain; label sentence 4 "(bet)" is Schwann's third claim. Add the one-line admission that it may be wrong and what would show it (the thesis's "What kills the whole program", item 3).
- The "Three things text cannot carry" list should be the Figure 1 of the document: one picture, three arrows.

## Naming in the corpus

Feature, circuit, universality (Zoom In: defined once in a claim block, glossary at end). Monosemantic / polysemantic, superposition, dictionary features, feature splitting (Towards Monosemanticity: coined in the results summary, reused verbatim). Golden Gate Bridge feature (Scaling: a named specimen stands for the method). Generalist agent, Gato (Gato: descriptive phrase, then proper noun). Ultra / Pro / Nano (Gemini: a size ladder named before results). Predictable scaling (GPT-4: a section title that is also the contribution). pLDDT, Evoformer, structure module (AlphaFold: term + functional definition in the same clause). Research debt, interpretive labor, distillation (Research Debt: economic metaphor carried consistently). Ours already in this register: confidence laundering (borrowed), read-out gap, label-only audit, message-level / receiver-level signal, the handle, SHOWN / SUGGESTED / BET, the order of the kills.
