# Craft notes: agent-economy and machine-to-machine-market essays, 2024–2026

Read 2026-10-07 for craft, not content. Our texts under study: `docs/paper/main.tex` (v5), `docs/thesis/thesis.tex`, `goal/GOAL.md`. Quotes from others are ≤15 words and attributed. Facts about our work come only from those three files.

## Corpus (ten documents, five primary)

1. Vitalik Buterin, "The promise and challenges of crypto + AI applications," vitalik.eth.limo, 2024-01-30.
2. Pat Grady, Sonya Huang, Konstantine Buhler (Sequoia), "AI's Trillion-Dollar Opportunity," AI Ascent keynote write-up, 2025-05-07.
3. Daniel Barabander (Variant Fund), "Agents in a Bazaar," 2025-05-13.
4. Erik Reppel, Ronnie Caspers, Kevin Leffew, Danny Organ, Dan Kim, Nemil Dalal (Coinbase), "x402: An open standard for internet-native payments," whitepaper, 2025-05-06; with Will Allen (Cloudflare), "x402 Foundation" launch post, 2025-09-23.
5. Sam Broner (a16z crypto), "Agents will pay like locals, not tourists," 2026-02 (substack dated 02-28; press 02-19).
6. Diogo Almeida (TypeSafe), "Introducing System One Models and Jev," typesafe.ai launch post, 2026-09-28; typesafe.ai landing page; swyx/Latent Space interview with Almeida, 2026-09-21.
7. nekuda (company byline), "Why AI Agent Payments is the Next Big Category," 2024-11-19. Secondary: no named author.
8. mpp.dev (Stripe/Tempo Machine Payments Protocol docs) and Ben Weiss (Fortune), 2026-03-18. Secondary: docs have no author; Fortune is reportage.

## Where the economic claim becomes concrete

The pattern across all ten: the number that does argumentative work is never the market size. It is a **unit cost on a mechanism** that makes the mechanism bite or break.

- nekuda: Stripe's fixed fee; "$0.35 flat fee almost doubles price-per-call" for a $0.50 agent-to-agent call. One fee, one call size, one consequence.
- Broner: cards clear in the "$20 to $1,000 range"; agents live outside it (sub-cent streams, $50,000 invoices). The range is the argument; the stablecoin conclusion is a corollary.
- x402 whitepaper: "fees as high as $0.30 per transaction, microtransactions become impractical" against "settle in ∼200ms" and "$0.001 cents per request." Fee, latency, floor.
- Cloudflare/Allen: "Over a billion HTTP 402 response codes" a day already exist. Volume on a mechanism nobody uses yet.
- TypeSafe: "193.6x faster" is always printed with both sides, "0.114s vs 8.566s," and a price, "$42 per billion tokens."
- Vitalik: "AIs are willing to work for less than $1 per hour" next to "tens of billions on sports." Wage against stake.

Our equivalent numbers already exist and are of exactly this type: 144 KB per token, 72 MiB per 512-token prefix, 604 ms link against 24 ms compute saved, 3.5 F1 for 48 ms (paper §cost, §limitations; thesis tab:pricesheet). They are printed late. The corpus prints them first.

## Where they hand-wave

Three devices, all easy to spot once named:

1. **"Imagine"** replacing a measurement. Broner: "Imagine an agent streaming funds to a compute provider." Nothing in the essay shows one.
2. **Fictional prices in the present tense.** x402 §11 lists "A trading AI retrieves real-time stock market data for $0.02 per request" under "how AI agents and humans are using x402." The prices are illustrative; the heading says usage.
3. **Market size with no mechanism.** Sequoia's title has "trillion-dollar"; the body has no dollar figure, and the write-up contains no "we believe" or "we think" anywhere. Pat Grady: "Nature hates a vacuum. There is a tremendous sucking sound in the market." Urgency stands in for evidence.

A fourth, from the interview genre: taste replacing a benchmark. Almeida on Latent Space: "there's a je ne sais quoi to it... good model smell."

The honest exceptions are instructive. TypeSafe's launch post follows every demo with a "Nuance" block: "we expect that these are on the higher end of real world gains"; "We can't prove it isn't subsidized." Barabander: "I'm not sure how it will play out in practice." Vitalik tags each branch of his taxonomy with a viability bracket, "[highest viability]" to "[tread very carefully]," and says "this plan is super-ambitious."

## How they name

| Author | Term | Device |
|---|---|---|
| Vitalik | AI as player / interface / rules / objective "of the game" | parallel grammar + bracketed viability tag per branch |
| Broner | tourists vs locals; "pay like locals" | social contrast pair; headline is the claim, imperative mood |
| Barabander | bazaar vs cathedral; "untrusted agents" | borrowed dichotomy (Raymond), then one adjective repeated |
| Coinbase | x402; facilitator; scheme | a dormant status code as brand; IETF-style role nouns |
| Stripe/Tempo | Challenges, Credentials, Intents, Receipts | generic protocol nouns, capitalised, no origin story |
| TypeSafe | System One Model; "Decisions, not strings"; RLCD; Choice/Score/Noulli; Jev | Kahneman borrow; three-word slogan; acronym shaped like RLHF/RLVR; invented primitives with etymology (Bernoulli, Jevons) |
| Sequoia | agent economy; stochastic mindset; autopilots | escalate an existing term (copilot → autopilot) |
| nekuda | last-mile vs resource payments | two-bin taxonomy |

Our own terms by the same table: confidence laundering (borrowed from 2606.20662, then relocated: "this reader does"), read-out gap, message-level / receiver-level signal, the audit ("score the message, then the reader"), SHOWN / SUGGESTED / BET, the price sheet (thesis tab:pricesheet), the order of the kills (thesis §program), the handle (".552 against .532"). We have the devices; what we lack is the corpus's habit of putting the coined pair in the first sentence and the headline.

## Twelve moves, each with a rewrite for us

Listed in the StructuredOutput of this run and summarised here.

1. Unit cost before market size → move 72 MiB / 604 ms / 24 ms into the paper's first paragraph; GOAL north-star sentence 3 names them inline.
2. Both sides of every multiple → already done in the abstract; extend to GOAL ("3.5 F1 for 48 ms" is the model).
3. Headline as counter-claim → paper's first sentence becomes the thesis Part I epigraph: "The claim is that text launders confidence. It does not; this reader does."
4. Contrast pair carries the architecture → section titles in the paper become "The message" / "The reader" pairs where the content allows.
5. Viability tag per branch → paper's Contributions (1)–(4) get the thesis's SHOWN/SUGGESTED tags; GOAL's four north-star sentences get the tag after "most to least certain."
6. Per-step transaction table → thesis ch. economy gets one table: probe fires → cache mapped → receiver reads → audit scores, one measured number per row, blanks marked BET.
7. Old-vs-new two-column table with measured cells → paper intro: text handoff vs cache handoff across bytes, latency, F1, AUROC, binding; every cell from the paper.
8. "Nuance" block under each result, not one Limitations paragraph → paper's 400-word §limitations split into four short notes placed under the tables they qualify (uncertainty table, probes, controls, cost).
9. Dormant-artefact hook → GOAL sentence 1 ("human-readable by inertia, not by design") becomes the paper's second sentence; no new metaphor needed.
10. Scale ladder in one sentence → thesis §"What it costs": "$500 buys about 33 runs at under $15" moves to the chapter's first line.
11. Name the open question and the experiment that answers it → keep paper conclusion's "That is the first experiment we would run"; add the same sentence to the thesis abstract.
12. Declare the price before the run, and say it is declared → price sheet already does this; add one line to the paper §cost pointing at it so the paper's bytes number is read as a declared price, not a complaint.

## Anti-patterns to avoid (seen in the corpus, checked against our texts)

- "Imagine…" as evidence. Not present in our three files.
- Fictional prices under a "usage" heading. Our BET tag prevents this; keep every unmeasured price in the BET column.
- Market size with no mechanism. GOAL says "there is an economy" with cost/value/fraud named; it never gives a market size. Keep it that way.
- No epistemic markers at all (Sequoia). We over-mark if anything; fine.
- Superlatives on an unmeasured property ("near-zero fees," "zero hallucinations," "eliminating chargebacks"). Our risk word is "channel": the paper already says "a measurement instrument, not a deployment."
- Prestige borrow without showing the borrowed property (System One without a System-1 test). Our borrow is "laundering"; the paper shows the loss location, so the borrow is paid for.
- Weakened baseline admitted only in a footnote. TypeSafe admits its LLM wrapper "tends to be slower" in the nuance block; our paper admits the text baselines "are not the strongest a competent engineer could build" in §limitations. Move that sentence next to the table it qualifies.
- Taste in place of a benchmark ("good model smell"). Not present.
