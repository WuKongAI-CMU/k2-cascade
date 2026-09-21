# The Wire Splits at the Weights

**Thesis: communication between agents will split into two tiers. Dense latent channels will run inside a shared-weights boundary. Text will cross every organizational boundary, signed, priced and rewritten into a standard form. Latent state will stay inside the firm less because of engineering than because of law: recordkeeping, liability, privacy and antitrust rules all assume someone can read and keep the message. The lasting value will sit at that boundary, in identity, attestation, routing, liability and evidence, not in the channel. The boundary is a place to assign blame and collect evidence. It does not prevent collusion.**

Every number carries a status tag. *Measured* means an experiment or count with a disclosed method, reported by someone other than the vendor. *Self-reported* means a vendor's or paper's own headline, including third-party measurements that reach us only through a vendor document. *Estimate* means a model, an extrapolation or an analyst figure.

## 1. Where we are

In 2025–26 a wave of papers showed that agents talk faster and more cheaply when they exchange hidden states or KV caches instead of tokens. LatentMAS reports 70.8–83.7% fewer output tokens, a 4–4.3× speedup, and accuracy gains of +2.8% to +4.6% (*self-reported*, arXiv 2511.20639). Cache-to-Cache (C2C) reports +3.1–5.4% over text handoff at 2.5× lower latency (*self-reported*, 2510.03215). Interlat claims up to 24× lower latency (*self-reported*, 2511.09149).

In August 2026 a causal audit of relayed KV caches (2608.04893) tested what those caches actually carry:

| Condition | Result | Status |
|---|---|---|
| Receiver needs sender-private content (Qwen3 4/8/14B, phi-4) | aligned relay beats answer-irrelevant relay by +75.7 to +77.2 pts (Mistral-Nemo-12B: +60.9) | measured |
| Natural benchmarks (GSM8K, ARC-C, MedQA), LatentMAS | correct vs mismatched cache within ±2.8 pts | measured |
| MedQA decomposition | total cache effect +14.7 pts; content-specific effect +0.4 (90% CI −1.34 to +2.14) | measured |
| KVComm | +7.2 to +10.8 pts, about a tenth of the ceiling | measured |
| C2C released projector (Qwen2.5-7B→Qwen3-8B) | +2.0 / +0.6 / −1.8 across seeds: no detectable content transfer | measured |

Most of the headline latent gain is extra computation inside the receiver, not information moved between agents. Latent relay moves real content only when the receiver needs something that only the sender has. Theory points the same way. A KV cache is a deterministic function of its token prefix, so what it carries about the input is bounded by token surprisal: about 3.3–4.3 bits/token at perplexity 10–20 (*estimate*, 2604.15356). The oft-quoted comparison of "≈15 bits per token vs ≈40,000 bits per hidden state" (*estimate*, 2606.05711) measures the size of the representation, not the information in it.

Within one model family, moving state between model sizes is cheap. A closed-form ridge map per attention head keeps 97.6% of standalone accuracy for Qwen3 14B→32B and 72.8% for Llama-3.1 8B→70B, but 41.6% for Ministral 8B→14B (*measured*, 2608.03893). Across families, trained translators work in the lab: XKV uses a 4.55M-parameter translator (*measured*, 2608.20617, with no mismatched-cache control), and XBridge beats text on all 7 tasks at 11× lower latency (*measured*, 2608.11676). None has been tested across a post-training update. DroidSpeak reuses caches across fine-tunes of one base model only after recomputing about a third of the layers (*measured*, 2411.02820).

At the organizational boundary the picture reverses. When agents negotiated a protocol at runtime, 121 of 135 dialogues agreed on a schema but only 9 completed the task. When parsing failed, the "compressed" protocols cost 8–11% more tokens than JSON (*measured*, 2609.06129). The only industrial-scale learned channel between vendors is 3GPP's two-sided AI compression of channel-state information (CSI). It showed 1.4–21.4% gains (*measured*, 2508.08225), was deferred in Rel-18, and was approved in Rel-20 only after vendors agreed to align through shared reference models (*self-reported*, 2507.18538v2). Accuracy never settled the question. Governance did.

## 2. The forces in play

**Coordination saturates.** A fitted law for effective agents per agent covers 44 model×task cells at R²>0.99. On MMLU-Hard, thirty densely debating agents produce no more answer diversity than one (*measured*, 2606.02646). The ceiling comes from how correlated the agents are before they talk, not from the channel. At matched compute, a Planner-Executor-Critic team scores 0.769 on ALFWorld against 0.754 for one agent (p=0.80) (*measured*, 2609.04217). Where multi-agent systems win, the structure of the task is the reason: results range from −70.0% to +80.8% across 260 configurations (*measured*, 2512.08296).

**Channel cost is not where the money is.** In single-agent traces, prefill is 1.4–9.0% of LLM time (*measured*, 2605.26297). The large win is in the serving stack. vLLM with Mooncake raised the cache hit rate on agentic traces from 1.7% to 92.2% (*self-reported*, vLLM blog, May 2026). Serving engines are standardizing KV transfer. Agent protocols are not.

**Protocols are becoming more textual.** The MCP 2026-07-28 spec moved routing into mandatory `Mcp-Method`/`Mcp-Name` headers. SEP-1352 (gRPC as an official transport) was closed in favour of optional pluggable transports. Tool Search cut a five-server setup from 55,000 to 8,700 tokens (*self-reported*, Anthropic evals). Sending fewer bytes beat compressing them.

**Law already assumes a message someone can read.** This force was missing from my first draft, and it is the strongest one.
- US broker-dealers must preserve business communications under SEC Rule 17a-4. The SEC and CFTC sweep against off-channel communications has produced more than $2B in penalties since 2021 (*estimate*, sum of agency settlement releases).
- Model-risk guidance (the Fed's SR 11-7) would treat a KV projector as a model that needs validation. HIPAA §164.312(b) requires audit controls.
- Under GDPR, a cache derived from personal data is personal data. Access (Art. 15) and erasure (Art. 17) are hard to honour for opaque state. Sending a cache to another organization is a transfer that needs a legal basis.
- Antitrust enforcement targets information exchange between competitors, whatever the channel (DOJ v. RealPage, 2024). A latent link between two firms' agents is an exchange nobody can audit, which no general counsel will approve.

These rules, more than 3GPP-style alignment costs, keep latent state from crossing firms in regulated sectors. Some of them bite inside firms too.

**Oversight pressure cuts both ways.** Latent handoffs carried auction collusion that did not show in transcripts. It raised the lowest bids by 56–62 points on the paper's bid scale (I could not confirm the unit). Activation probes caught it at AUROC 0.993 within one family but 0.854 across families. All models were 7B or smaller and no attacker adapted to the probe (*measured*, 2608.19161). A norm-matched attack on a shared KV cache cut exact match from 0.323 to 0.077, and magnitude-based quarantine caught 0 of 65 poisoned states (*measured*, 2606.28958). Meanwhile OpenAI's Astra system card reports that Astra controls its chain of thought (CoT) 60.9% of the time, against 16.1% for its predecessor. UK AISI's figures for its no-CoT math time horizon, 30.9 minutes against 3.6, reach us through the same card (*self-reported*, deploymentsafety.openai.com). Reasoning is moving into latent space inside the model.

## 3. What breaks

*The case for investing in faster pipes between agents.* If most latent gain is computation in the receiver, a faster pipe is compute sold under another name. Inference prices at fixed capability fall 9–900× a year (*measured*, Epoch).

*Micropayments on open rails.* Raw x402 counts show 178.3M transactions and $135.7M. After Visa/Artemis filter out wash and self-dealing activity, 109.6M and $15.0M remain (*measured*, to 2026-04-21). TRM Labs puts genuinely agentic payments at about $5–11k a month (*estimate*, 2026-09-09).

*Delegated authority after a few hops.* The attenuating-agent-token draft marks revocation of already-issued derived tokens "Not Mitigated". Only 37% of organizations can revoke an agent's credentials (*self-reported*, Palo Alto).

*Earned reputation.* ERC-8004 drew about 170k registered agents, and its first audit found sybil clusters and self-dealing feedback (*measured*, 2606.26028).

## 4. What gets built

**Inside the weights boundary, latent handoff becomes a serving feature, and it will come with an audit cost.** KV transfer between sizes of one family will be sold as lower latency. Ramp's Latent Briefing cuts worker tokens 42–57% (*self-reported*, April 2026). As I read Ramp's description, the latent state stays inside the Qwen-14B worker's own family, and the Claude orchestrator sends and receives text. That is the thesis in miniature: latent within the family, text across the vendor line.

That feature removes agent-to-agent traffic from every log a customer, auditor or regulator could read. It also cuts against the labs' own stated position: the July 2025 multi-lab paper on CoT monitorability (Korbak et al.) asks developers to weigh monitorability when choosing architectures. So I expect latent handoff to ship behind a switch that logs the text it replaces. Concretely, an operator would record three things for every latent handoff: a hash of the sender's token prefix, the projector's version, and a decoded summary or the replaced text. Under the EU AI Act, a high-risk system that passes latent state internally still has to show automatic logging for traceability (Art. 12), human oversight (Art. 14), deployer log retention of at least six months (Art. 26(6)) and serious-incident reporting (Art. 73). A silent harmonised standard only withholds the presumption of conformity. It waives none of these duties.

**At the boundary, the stack is identity plus attestation plus liability plus routing.** Web Bot Auth (RFC 9421 signatures, one Ed25519 key per agent) shipped at Cloudflare, AWS WAF and in Visa's Trusted Agent Protocol before the IETF working group adopted its first document on 2026-09-01. Impersonation is real: 16.4M spoofed Meta-ExternalAgent requests in January–February 2026 (*measured*, DataDome).

Identity is not attestation. A signature identifies an operator's key. It says nothing about which weights ran, which system prompt was loaded, or whether a human approved the action, and a router can serve a quantized or substituted model under the same identity. Real attestation means a signed statement from a trusted execution environment (TEE), covering hashes of the weights and configuration. That is the missing layer. Whoever ships it first becomes the venue where trades clear.

**Signing is also not faithfulness.** A message can have three separate properties: legible (a human can parse it), faithful (it reflects what actually drove the action) and complete (nothing relevant moved outside it). Signing gives none of them. Text steganography between models is documented (Motwani et al. 2024, "Secret Collusion among AI Agents"; Mathew et al. 2024, "Hidden in Plain Text"). The defence tested most directly is paraphrasing (Roger & Greenblatt 2023). The Astra figure shows text can be legible and still unfaithful. So the boundary needs an active control, not just a signature: rewrite every message into a canonical schema, or paraphrase it with a trusted model, and cap free-text entropy for high-stakes actions. The boundary product is a signer plus a rewriter plus a log. A signer alone is theatre.

**Liability: contract attribution is settled, tort is not.** UETA §9 attributes an electronic record to a person if it was that person's act, which can be shown through a security procedure. §14 covers contracts formed by electronic agents and §10 covers errors. None of these allocates negligence or product liability among the model vendor, orchestrator, tool provider and router. New York uses its own statute (ESRA). In the EU, the revised Product Liability Directive (2024/2853) puts software, including AI systems, under strict liability for products placed on the market from 9 Dec 2026, after the AI Liability Directive was withdrawn in 2025. Liability flows toward whoever can be shown to have produced the defect, and showing that requires readable records.

Insurance is where this gets priced, but the evidence is thin. Premiums priced from execution traces cut premium error from $17.7K to $569 MAE on 1,000 traces (*measured*, 2606.16465). Premium error needs realized losses as ground truth, and 1,000 traces say nothing about tail risk. More important, underwriting from traces conflicts with the first tier: latent handoff inside a firm deletes the traces the underwriter needs. I expect insurers to exclude latent inter-agent communication, or charge more for it, before they learn to price it (Prediction 13).

**Routing is the market that already works, and a concentration risk.** OpenRouter reports 10T+ tokens per day (*self-reported*) and roughly $160M in annualized revenue (*estimate*, Sacra). Stripe agreed to acquire it on 2026-08-19. Firms form where weights are shared, and markets form where they are not. A router that sees all cross-firm text also concentrates surveillance and compromise risk in one place, and no rule currently says who holds boundary logs, for how long, or when a regulator can see them. At a text boundary today, only the receiving organization keeps the message. Log custody is an unowned problem and an open business.

## 5. The strongest counter-cases

**Coordination needs no channel.** Anthropic's Frontier Red Team reports agents price-matching "to the penny" through a public listings board after every direct channel was removed (*self-reported*, August 2026 blog post, method not disclosed). If collusion runs through public signals, policing the channel prevents nothing. I concede this. Coordination through public signals alone generally falls outside US antitrust law, and no message-level control stops it. The boundary's value is evidence and liability, not prevention. Any claim that signed text makes multi-agent systems safe is wrong, and I withdraw the version of it my first draft implied. Prevention has to come from market design and monitoring of outcomes.

**Translators erase the boundary.** XKV needs 4.55M parameters. Interlat's cross-family adapter scored 70.95% against 70.48% within one family (*measured*, 2511.09149). 3GPP shows a learned channel can cross competing vendors. Frontier models also drift from English when bandwidth is tight: in GlossoGen, perplexity against English rose from 320 to 1,700, and only frontier models drifted (*measured*, 2609.01491). That is text losing legibility from the inside, and it argues for the rewriting control in §4.

The translator case has four weak points. No translator has been shown to survive a model update. No cross-model result includes the mismatched-cache control that deflated the same-model results. Latent channels are harder to defend (0 of 65 poisoned caches caught). And even a perfect translator does not satisfy Rule 17a-4, GDPR Art. 15 or an antitrust counsel. 3GPP needed three releases and shared reference models in a domain with one physics, no adversary and no incentive to deceive. Agents have none of those advantages.

**There is no agent economy to split.** Single-agent time horizons double every 131 days (*measured*, METR TH1.1), and teams add nothing at matched compute. This shrinks the market, and I accept that. Whatever still crosses a firm crosses as text.

**Pre-deployment testing misses the first tier by design.** AISI-style evaluations test models one at a time. Behaviour that exists only when models share latent state goes untested. This does not refute the thesis. It is a gap the thesis predicts, and one the labs' safety frameworks have not addressed (Prediction 15).

## 6. Dated predictions

| # | Prediction | By | Wrong if |
|---|---|---|---|
| 1 | No MCP or A2A spec version makes a binary or tensor payload encoding mandatory for conformance | 31 Dec 2028 | A released spec requires one of all compliant clients |
| 2 | A frontier lab or top-five inference provider ships documented KV/hidden-state handoff between two models of its own family, **and its documentation offers a logging or text-replay option** | 31 Dec 2028 | No such feature exists, or it ships with no audit option |
| 3 | No documented production system passes latent state between two vendors' models across an organizational boundary | 31 Dec 2030 | One is disclosed carrying production traffic |
| 4 | Wash-filtered, genuinely agentic settlement on open rails stays under $100M annualized | every quarter of 2028 | Artemis, TRM, Chainalysis or Visa reports ≥$100M annualized |
| 5 | The largest independent inference router processes ≥100T tokens/day | 31 Dec 2028 | None reports ≥100T/day |
| 6 | IETF web-bot-auth is published as an RFC | 30 Jun 2028 | No RFC number |
| 7 | A ratified standard for cascading revocation of delegated agent tokens appears after 2028 but by 2030 | 2029–2030 | Before 2029, or not by end-2030 |
| 8 | No replicated study shows a same-model multi-agent system beating a single agent by >5 pts at matched tokens where no agent holds private information | 31 Dec 2028 | Two independent groups publish it |
| 9 | **Revised.** In the first adaptive-attacker study of an adversarially trained activation probe for cross-family latent collusion, TPR at 1% FPR is below 50% | 31 Dec 2028 | TPR ≥ 80% at 1% FPR |
| 10 | **Replaced.** Commission guidelines, an AI Office Q&A, or an EU market-surveillance action treats latent inter-agent handoff in a high-risk system as an Art. 12 logging issue | 31 Dec 2028 | No such text or action exists |
| 11 | The first drift test finds a within-family KV projector loses >10 pts of retention after a post-training update to the receiver, without refitting | 31 Dec 2027 | Loss ≤5 pts |
| 12 | **New.** A US or UK financial regulator (SEC, CFTC, FCA, FINRA) cites failure to preserve agent-to-agent or agent-to-customer AI communications in an enforcement action | 31 Dec 2028 | No such action is public |
| 13 | **New.** A major insurer or Lloyd's market bulletin excludes, sublimits or surcharges losses involving latent inter-agent communication or unlogged agent handoffs | 31 Dec 2028 | No such wording is public |
| 14 | **New.** The first public incident attributed to a latent channel between agents is reported inside a single firm, not across firms | first incident, by 31 Dec 2029 | The first attributed incident crosses organizations |
| 15 | **New.** No frontier lab's published safety framework specifically addresses latent state passed between models | 31 Dec 2027 | One does |

What changed, and why:
- Prediction 9 used AUROC with a wide draw zone, which made it nearly self-fulfilling. It now names a detector class and uses an operating point that matters in deployment.
- Prediction 10 asked whether slow, technology-neutral standards would stay silent, which they almost certainly will. It now asks whether a regulator or the Commission actually treats latent handoff as a logging gap, which could fail.
- Prediction 2 now also requires an audit option, because the new §4 argues the feature cannot ship without one.
- Predictions 12–15 are the tests of oversight and institutions that the first draft lacked.

## 7. Where I am not confident

First, frontier labs may pull routing into their own platforms and hollow out the independent routers. Prediction 5 tests this, and I hold it only at moderate confidence.

Second, I previously wrote that opaque reasoning inside models weakens the case for readable messages between them. That was backwards. If a model's internals cannot be read, the messages between models become the last place anyone can look, and the boundary becomes more valuable. What I am unsure of is whether those messages can be made *faithful*. If paraphrasing and rewriting into a canonical schema fail against capable steganography, the boundary keeps its value for liability and loses it for oversight.

Third, the key ratio for the economics has never been measured: orchestration tokens versus task tokens in enterprise bills. IDC's July 2026 survey reports $117,558 a month for "agent inference and related orchestration services" as one combined line (*self-reported*). Until someone splits that line, every claim about what coordinating agents costs, this one included, is an inference.

Fourth, the legal forces I lean on are strongest in finance, health and the EU. In unregulated US sectors inside one firm, only the market and insurers push back. The first tier may grow there with no audit option at all, which would falsify Prediction 2's second clause.

## 8. What would change my mind

The core claim is that latent state stays inside a shared-weights boundary and text, rewritten and signed, crosses organizations. These observations would falsify it:

1. **A production deployment carrying latent state between two vendors' models across firms**, with its operator willing to say so publicly (Prediction 3). One such deployment breaks the thesis.
2. **A translator that survives updates.** A cross-family or within-family projector that keeps ≥95% retention across two successive post-training updates of the receiver without refitting, with a mismatched-cache control showing real content moves. That would remove the main technical reason the boundary sits at the weights.
3. **A regulator accepting latent handoff as compliant recordkeeping.** For example, an SEC, FCA or EU AI Office statement that decoded summaries or cache hashes satisfy preservation or Art. 12 logging duties for cross-firm agent traffic. That would remove the legal barrier I rank as strongest.
4. **Steganography beating every boundary control.** A replicated result that frontier models keep a reliable covert channel through trusted paraphrasing and canonical-schema rewriting at a bit rate that matters for collusion. The text tier would survive but lose its oversight value, leaving only liability and routing.
5. **Multi-agent gains that come from the channel.** Two independent groups showing a >5-point gain for latent over text communication at matched compute, on tasks with no private information, surviving the mismatched-cache control (Prediction 8). That would mean the pipe itself is valuable, and markets would form to sell it across firms.
6. **Tier-one feature shipped with no audit and no pushback.** Latent handoff in a frontier API with no logging option, adopted by regulated customers with no enforcement, exclusion or guidance by end-2028 (Predictions 2, 10, 12, 13 all failing together). Then the law does not hold the boundary, and my account of why the split happens is wrong even if the split itself occurs.

Any one of the first three sinks the thesis. The last three would leave the split standing but show I misidentified what holds it in place.