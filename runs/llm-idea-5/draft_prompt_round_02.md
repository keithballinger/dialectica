You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.
        Field: general

        Constraints of Paper:
        From: constraints/llm.md

- Research focused on Large Language Model inference
- Very impactful on quality, performance, or agentic workflows
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models


        Draft:
        Major Revisions

Title
Lookahead-Consistency Token Filtering: Decoding-time Micro-Rollouts that Prefer Semantically Stable Continuations

Abstract
Large language models (LLMs) still hallucinate under standard decoding, especially when local token choices steer the model into unstable semantic regions. We introduce Lookahead-Consistency Token Filtering (LCTF), a decoding-time method that prefers next tokens whose short continuations agree semantically under cheap micro-rollouts from a small “draft” model. At each step, we shortlist top-p candidates from the target model, roll out k short continuations with the draft model, score pairwise agreement via an NLI-based entailment metric, and reweight the candidate logits accordingly. Unlike speculative decoding, which checks token-level acceptance by a small model, LCTF explicitly scores semantic agreement across multiple micro-rollouts conditioned on the same candidate token, penalizing locally risky choices. We provide an implementation plan with open-source models (≤2B parameters), falsification experiments on TruthfulQA, BioASQ-lite, and WikiSQL text-to-SQL, and ablations against greedy/top-p, contrastive decoding, and self-consistency. If successful, LCTF improves factuality and exact-match metrics with modest overhead, and can be composed with speculative decoding and verifier-based re-ranking for agentic workflows.

Introduction
- Problem. Decoding from LLMs can commit to locally plausible but globally unstable tokens that amplify errors into hallucinations or invalid plans. Standard heuristics (temperature, top-p) modulate entropy but do not explicitly assess the downstream semantic stability of each candidate token.
- Prior art. Self-consistency samples full solutions and majority-votes, improving reasoning but incurring large compute. Speculative decoding accelerates inference by accepting tokens predicted by a draft model; it does not measure semantic agreement across multiple futures. Contrastive/DoLa methods reduce generic errors via internal contrast but do not assess multi-rollout stability. Verifier-based reranking works post hoc and often requires multiple full generations.
- Key idea. Before committing to a next token, run cheap k-step micro-rollouts with a small draft model and prefer tokens whose continuations agree semantically. Intuition: tokens that lead to mutually entailing short futures are safer; those that lead to divergent futures are risky.
- Contributions.
  1) A decoding-time, agreement-based lookahead filter that is orthogonal to speculative decoding and self-consistency.
  2) A practical instantiation using small, open-source models for both draft rollouts and NLI scoring.
  3) A falsification plan across factuality (TruthfulQA, BioASQ-lite) and structured generation (WikiSQL) with ablations and efficiency analysis.

Method
Notation and setup
- Target model MT: the main LLM used for final generation (e.g., Pythia-1.4B or Mistral-7B).
- Draft model MD: a cheaper model used only for micro-rollouts (e.g., TinyLlama-1.1B, Pythia-410M).
- NLI scorer S: a lightweight entailment classifier (e.g., deberta-v3-base-mnli) that returns entailment probabilities between text pairs.

At decoding step t with context C:
1) Candidate shortlist. Compute MT logits over next tokens; form a set A of M candidates via top-p or top-M.
2) Micro-rollouts. For each a in A:
   - Condition MD on C ⊕ a.
   - Sample k rollouts R(a) = {r1, …, rk}, each of length L tokens, with low temperature (e.g., T=0.7) and top-p (e.g., 0.9).
3) Agreement scoring. Convert each rollout into a sentence string (detokenize).
   - Pairwise NLI: For all pairs (ri, rj), compute entailment in both directions and take max (or average of bi-directional logits).
   - Define consistency score s(a) = average pairwise entailment probability across all i < j.
   - Optional: combine NLI with cosine similarity in an embedding space (e.g., e5-small-v2) to stabilize scores: s(a) ← α·NLI + (1−α)·cosine.
4) Logit reweighting/filtering.
   - Base logit z(a) from MT. Compute bonus b(a) = β·log(s(a) + ε) − β·log(1 − s(a) + ε) (logit transform; ε≈1e−5).
   - Adjusted logit: z′(a) = z(a) + b(a).
   - Optional hard filter: discard tokens with s(a) < τ (e.g., τ=0.55).
5) Selection. Sample or argmax from adjusted distribution over A; commit the chosen token and proceed.

Pseudocode (high-level)
- Input: context C, MT, MD, S, hyperparams (M, p, k, L, β, τ).
- Obtain top-p set A of size M from MT(C).
- For each a in A:
  - R ← {sample MD(C ⊕ a, length=L) repeated k times}
  - s(a) ← mean_pairwise(max_bidir_entailment(S, R))
- For each a in A:
  - z′(a) ← z(a) + β·logit(s(a))
- Optionally filter {a | s(a) < τ}.
- Choose next token from softmax(z′) restricted to remaining candidates.

Design choices and efficiency
- Compute cost. Each step adds O(M·k·L) tokens from MD and O(M·k^2) NLI calls on short strings. With M=6, k=3, L=8, this is 144 draft tokens and 27 NLI pairs per step. If MD is ≥10× cheaper than MT per token and NLI is small, wall-clock overhead is often ≤1.5–2.5× for single-token steps. Batching and kv-caching for MD amortize further.
- Stability vs diversity. Larger k or L increases sensitivity to instability but raises cost; start small (k=3, L=6–8).
- Agreement metric. NLI is robust and directional; embeddings-only can act as a fallback when latency is constrained. A hybrid reduces miscalibration.
- Compatibility. LCTF composes with:
  - Speculative decoding: reuse MD; interleave lookahead consistency with acceptance checks.
  - Verifier reranking: LCTF reduces bad early tokens; verifiers clean up end-stage errors.
  - Contrastive decoding/DoLa: apply logit bonuses after contrastive adjustments.
- Practical defaults.
  - MT: Pythia-1.4B or Mistral-7B-Instruct
  - MD: TinyLlama-1.1B or Pythia-410M
  - S: deberta-v3-base-mnli (or -small for faster)
  - M=6, top-p=0.9, k=3, L=8, β=1.0, τ=0.55, T=0.7

Experiments (falsification plan)
Goals
- Test whether agreement-based micro-lookahead improves factuality and structured accuracy with acceptable overhead.
- Falsify by showing no improvement vs strong baselines or by showing regressions on creative/diverse tasks.

Datasets and metrics
- TruthfulQA (generation): Report %True, %Truthful+Informative using the official evaluation script (multiple-choice subset for objectivity; report MC1, MC2).
- BioASQ-lite factoid QA: Exact match and F1 on gold answers (normalized).
- WikiSQL text-to-SQL: Exact match of SQL string and execution accuracy on provided tables/queries.
- Optional sanity checks: GSM8K short-form final-answer accuracy (no chain-of-thought), to assess planning side-effects; XSum summary factual consistency via QAGS-lite or SummaC-ZS.

Models
- Target MT:
  - Pythia-1.4B (primary small-open baseline)
  - Mistral-7B-Instruct (optional mid-size)
- Draft MD:
  - TinyLlama-1.1B (primary)
  - Pythia-410M (ablation)
- NLI S:
  - deberta-v3-base-mnli (primary)
  - deberta-v3-small-mnli (speed ablation)

Baselines
- Greedy, top-p (p=0.9), temperature T∈{0.7,1.0}.
- Beam search (width 4).
- Contrastive decoding (e.g., DoLa or similar, if open implementation available).
- Self-consistency (k=5 full generations, majority vote) where applicable.
- Draft-only speculative decoding acceleration without consistency filtering (where applicable).

LCTF variants
- Agreement metric: NLI-only vs NLI+embedding vs embedding-only.
- Hyperparameters: k∈{2,3,5}, L∈{4,8,16}, β∈{0.5,1.0,2.0}, τ∈{0.5,0.6,0.7}.
- Hard filter vs soft reweighting.

Protocol
- Prompting: Use published prompts for each dataset; no chain-of-thought in final outputs.
- Decoding length caps: TruthfulQA 64 tokens; BioASQ 32; WikiSQL 128.
- Compute: Single 24–48GB GPU; batch NLI pairs; cache MD KV states across k rollouts per a when possible.
- Statistics: 3 seeds; report mean±95% CI.
- Efficiency: Tokens/sec for MT, end-to-end latency per generated token, and overhead breakdown (MD tokens, NLI time).
- Falsification criteria:
  - If LCTF fails to improve at least one of TruthfulQA true rate (+2% absolute) or WikiSQL exact match (+1% absolute) vs top-p baseline at matched MT temperature, we consider the method falsified for those settings.
  - If overhead exceeds 2.5× on Pythia-1.4B without measurable gains, we consider it impractical under small-model constraints.

Expected analyses
- Calibration: Plot accuracy vs average s(a) across steps; check if higher agreement correlates with correctness.
- Early-step influence: Measure how often LCTF blocks the top-1 base token and whether that correlates with final correctness.
- Error taxonomy: For TruthfulQA, categorize reductions in confident falsehoods vs hedged true statements.
- Structured validity: On WikiSQL, measure parse validity and execution errors; does LCTF reduce invalid SQL?

Discussion
- Why it can work. Hallucinations often arise when local choices push the model into divergent semantic basins. Agreement across short futures indicates local robustness to sampling noise; penalizing disagreement acts as a risk-sensitive prior at decoding time.
- Relation to prior methods.
  - Self-consistency operates at the solution level; LCTF is token-local, cheaper, and can prevent early commitment to unstable paths.
  - Speculative decoding accelerates but does not judge semantic stability; LCTF supplies a complementary stability prior, potentially using the same draft model.
  - Verifier reranking corrects after the fact; LCTF is preventive.
  - Contrastive/DoLa temper generic errors using internal signals, while LCTF uses external agreement across futures.
- Agentic workflows. In tools/agents, early unstable tokens can derail tool calls or schema usage. LCTF is naturally aligned with reliability-first decoding and can be turned on selectively for high-stakes spans (e.g., entity mentions, function names).
- Practical guidance.
  - Use small k and L for low-latency QA; increase for structured tasks (SQL) or high-stakes segments.
  - Combine with retrieval or schema constraints: restrict candidate tokens to schema vocabulary and reweight by agreement.
  - Adaptive β, τ via uncertainty: tighten thresholds when MT entropy is high.

Limitations
- Compute overhead. Although MD is cheap, NLI calls add latency. Embedding-only variants are faster but noisier.
- Metric brittleness. NLI on short, fragmentary rollouts can be noisy; calibration is required. Domain-specific entailment (e.g., biomedical) may need fine-tuned NLI models.
- Diversity suppression. The method may bias toward generic continuations, harming creative writing or tasks requiring exploration.
- Token-level myopia. Short lookaheads may miss long-range contradictions or promote locally stable but globally incorrect directions.
- Draft-model mismatch. If MD’s local dynamics are poorly aligned with MT, agreement scores may mislead.

Conclusion
We propose Lookahead-Consistency Token Filtering, a decoding-time method that prefers next tokens whose short futures agree semantically under cheap micro-rollouts from a small draft model. This simple, model-agnostic mechanism targets a key failure mode—locally risky token choices—without retraining or heavy verifier pipelines. Our falsification plan with small open-source models assesses factuality and structured accuracy under strict efficiency constraints. If validated, LCTF offers a practical, composable primitive for more reliable LLM inference and agentic workflows.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
