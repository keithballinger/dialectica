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

Brief critique
- Novelty/positioning: The core idea is promising but overlaps with existing families: self-consistency and best-of-N with verifiers, SelfCheckGPT/consistency-based hallucination detection, Tree-of-Thought/verification-guided decoding, and value/reranker-guided search. You need a sharper comparison, citations, and ablations showing that LCD’s token-local, micro-rollout agreement beats: (a) best-of-N with the same NLI budget, (b) verifier-reranked beams, and (c) SelfCheck-style contradiction scoring without per-token lookahead.
- Efficiency realism: The stated per-step overhead (e.g., 144 draft tokens + 18 NLI calls) is likely >2.5x latency for the proposed model sizes. TinyLlama-1.1B is not >10x cheaper than Pythia-1.4B. NLI cross-encoder calls dominate latency unless aggressively batched. Add dynamic gating (uncertainty-triggered), hierarchical scoring (cheap embedding prefilter → selective NLI), and costed, batched KV-caching details. Provide a rigorous compute accounting (FLOPs/tokens) and a Pareto plot versus quality.
- Agreement metric robustness: NLI on short fragments is noisy and biased toward “neutral/generic” continuations, risking degeneration. Add safeguards: specificity/informativeness regularizer, length calibration, domain-aware validators (e.g., SQL parse/execution checks), and ablations with embedding-only and contradiction-only metrics. For structured code/SQL, replace NLI with exact-structure checks.
- Degenerate incentives: Penalizing divergence can over-prefer bland continuations (“I don’t know”) or suppress minority-correct futures. Incorporate a specificity prior and demonstrate no loss on creative/diverse tasks (or scope the method to reliability-critical domains and gate usage).
- Experimental coverage: Add stronger baselines (SelfCheckGPT-style consistency scoring at output level, best-of-N with the same NLI/token budget, verifier-reranked beam search). Include significance testing, length control, and compute-normalized comparisons. Add agentic benchmarks where single-token errors are costly (e.g., function-call accuracy on ToolBench-like subsets) to substantiate the agentic-workflow claim.
- Clarity and notation: Simplify notation (avoid overloading M for both model and candidate count), define all hyperparameters, and clearly describe caching/batching and memory requirements for branching rollouts.
- Claims calibration: Temper “modest overhead (<2x)” to a hypothesis contingent on gating and hierarchical scoring. Explicitly state failure modes and where LCD should not be used.

Revised Draft

# Lookahead-Consistency Decoding: Risk-Aware Token Selection via Semantic Agreement of Micro-Rollouts

## Abstract
We propose Lookahead-Consistency Decoding (LCD), a decoding-time algorithm that reduces unforced errors by preferring next tokens whose short futures exhibit high semantic agreement. LCD uses a small draft model to generate brief micro-rollouts conditioned on each candidate token and scores their agreement using a lightweight hierarchy: embedding-based consensus followed by selective NLI checks. The resulting consistency score modulates the target model’s logits, penalizing locally risky choices. Unlike speculative decoding (speed) and self-consistency or verifier reranking (solution-level), LCD provides a token-local, preventive signal. We detail an efficiency-focused design with uncertainty gating, hierarchical scoring, and batched KV reuse, and a rigorous falsification plan on TruthfulQA, BioASQ, WikiSQL, and function-call accuracy, all with small open models. We test whether LCD improves factuality and structured accuracy under compute-normalized budgets, and we provide ablations and negative controls to probe robustness and degeneracies.

## 1. Introduction
- Problem: Standard decoding commits to locally likely tokens that induce semantically unstable states, yielding hallucinations or brittle structured outputs.
- Limitations of prior art: Self-consistency and verifier reranking operate post hoc and are compute-heavy; speculative decoding accelerates but does not assess semantic stability; contrastive methods (e.g., DoLa) are internal and token-local but do not probe multi-token consequences; SelfCheck-like methods assess output-level consistency, not token-local risk.
- Key idea: Estimate local semantic stability by probing candidate tokens with cheap micro-rollouts and favoring tokens whose futures agree semantically.

Contributions
1) LCD: a token-level lookahead scoring mechanism that measures the agreement of micro-rollouts and modulates logits accordingly.  
2) Efficiency-first implementation: uncertainty gating, hierarchical scoring (embedding prefilter → selective NLI), and batched KV-cached rollouts.  
3) Rigorous falsification: compute-normalized evaluations across factuality and structured tasks, with strong baselines and ablations that isolate where agreement helps or hurts.  
4) Open, small-model validation: end-to-end code with Pythia-1.4B/TinyLlama-1.1B and DeBERTa-MNLI variants.

## 2. Method

2.1 Overview and notation
- Target model: T (e.g., Pythia-1.4B, Mistral-7B).  
- Draft model: D (e.g., TinyLlama-1.1B, Pythia-410M).  
- Agreement scorers: E (sentence-embedding encoder, e.g., all-MiniLM) and optional NLI S (e.g., deberta-v3-(small|base)-mnli).  
- At step t with context C, let P_T be T’s next-token distribution. Let m be candidate set size.

2.2 Uncertainty-triggered gating
Run LCD only when the step is “risky,” e.g., if any of the following hold:  
- High entropy: H(P_T) ≥ h.  
- Low margin: p_top − p_2 ≤ δ.  
- Rapid drift: cosine distance between current and previous hidden states ≥ ρ.  
Otherwise, decode normally. Gating targets ≤20–30% of steps.

2.3 Candidate selection and micro-rollouts
- Candidates A: take m tokens via top-p/top-k with temperature τ_T.  
- For each a in A, spawn k rollouts of length L with D using moderate temperature τ_D and top-p (stochastic but not excessively diverse).  
- Reuse the KV cache for C across all branches; branch-specific caches start from C ⊕ a.

2.4 Hierarchical agreement scoring
Goal: robust, low-latency agreement that avoids bias toward generic text.

Step 1: Embedding consensus (cheap, all candidates)
- Encode each rollout r_i with E (mean pooling).  
- Compute pairwise cosine similarities; let s_emb be the mean of the upper-triangular entries.  
- Specificity prior: penalize generic/neutral rollouts via a length- and IDF-adjusted specificity score spec (e.g., sum of IDF-weighted content tokens normalized by length).  
- Preliminary score: s0 = s_emb − λ(1 − spec).

Step 2: Selective NLI (expensive, few candidates)
- Take the top u candidates by s0 (e.g., u ≤ 2).  
- Compute NLI entailment and contradiction for each rollout pair; let s_nli be mean entailment minus mean contradiction (clipped to [0,1]).  
- Final score: s = α s0 + (1 − α) s_nli for candidates with NLI; for others, s = s0.

Structured-output specialization
- For SQL/code, replace NLI with structural checks: parse validity, execution success on dev DB, and normalized edit distance between ASTs. Define s_struct as 1 − mean normalized distance; use s = α s0 + (1 − α) s_struct.

2.5 Logit modulation and selection
- Stabilize probabilities: s ← clip(s, ε, 1 − ε).  
- Consistency bonus: b(a) = β · logit(s(a)), with β tuned per task.  
- Adjusted logits: z′(a) = z(a) + b(a) for a ∈ A, leaving others unchanged.  
- Optional hard filter: discard a if s(a) < τ.  
- Sample next token from softmax(z′) with original temperature.

2.6 Pseudocode (single step)
def lcd_step(C, T, D, E, S, params):
    if not is_risky_step(T, C, params):
        return sample_from_T(T, C, params)
    A, logits = candidate_set(T, C, params)
    rollouts = sample_rollouts(D, C, A, k, L, kv_share=True, batch=True)
    s0 = embedding_consensus(rollouts, E, specificity_weight=lambda)
    A_sel = top_u_by_score(A, s0, u)
    s = s0.copy()
    if use_nli:
        s_nli = nli_consensus(rollouts[A_sel], S, batch=True)
        for a in A_sel:
            s[a] = alpha*s0[a] + (1-alpha)*s_nli[a]
    z_prime = {}
    for a in A:
        sa = clip(s[a], eps, 1-eps)
        if sa >= tau:
            z_prime[a] = logits[a] + beta*logit(sa)
    return sample_from_logits(z_prime)

## 3. Efficiency and engineering

- Compute accounting: per risky step cost ≈ m·k·L tokens on D + c_embed·(m·k) + c_nli·(u·k·(k−1)/2). Report FLOPs and latency, not just tokens.  
- Batching: batch all rollouts across candidates; reuse KV of C; interleave greedy steps of T with buffered D/S batches to hide latency.  
- Memory: rolling-window rollout with small L; free branch KV after scoring; micro-batching to fit GPU memory.  
- Typical settings: m=4, k=2, L=6, u=1–2, α≈0.5, β tuned per task; gating rate 20–30%. Target ≤2x median latency on A100 for T=1.4B.

## 4. Experimental validation

4.1 Hypotheses and falsification
- H1 (quality): LCD improves primary metrics by ≥2–3% absolute over tuned top-p at matched or lower end-to-end compute on small open models.  
- H2 (efficiency): With gating and hierarchical scoring, median latency overhead ≤2.0x and ≤1.2x compute-normalized degradation versus standard decoding.  
- H3 (mechanism): Higher average s(a) correlates with correctness; improvements concentrate at risky steps.

Falsification criteria (any suffices)
- No statistically significant improvement (bootstrap p<0.05) on primary metrics at compute-matched budgets.  
- Latency >2.5x without offsetting quality gains.  
- No positive correlation between s(a) and outcome correctness.

4.2 Tasks and metrics
- TruthfulQA (generation): %True, %Truthful+Informative.  
- BioASQ factoid: EM and F1.  
- WikiSQL: exact match and execution accuracy.  
- Function-call accuracy: exact tool name and argument match on a subset of ToolBench-style prompts (open-source subset), to probe agentic reliability.

4.3 Models
- T: Pythia-1.4B (primary), Mistral-7B-Instruct (scale-up).  
- D: TinyLlama-1.1B (primary), Pythia-410M (speed ablation).  
- E: all-MiniLM-L6-v2 (primary), MPNet (ablation).  
- S: deberta-v3-small/base-mnli (speed/quality).

4.4 Baselines
- Greedy; tuned top-p/top-k; beam search (w=4).  
- Self-consistency (k=5 full generations) with majority vote.  
- Verifier-reranked beam search (NLI or task-specific), compute-matched to LCD.  
- Best-of-N with verifier under the same total D/S budget as LCD.  
- SelfCheckGPT-style contradiction scoring at the output level.  
- Contrastive decoding (open implementation).

4.5 Protocol
- Compute-normalized comparisons: equalize total extra FLOPs/time across methods.  
- Significance: 1k bootstrap replicates; report mean±95% CI.  
- Controls: length-matched evaluation; temperature sweep; seed robustness (≥5 seeds).  
- Ablations: remove gating; embedding-only vs NLI-only; vary m,k,L,u; β, τ; replace NLI with structural checks on WikiSQL; replace D with a weaker model; remove specificity prior.  
- Mechanistic probes: correlate s(a) with final correctness; analyze where LCD alters the chosen token; categorize error reductions (e.g., invalid SQL, confident falsehoods).

## 5. Relation to prior work
- Self-consistency/best-of-N: solution-level voting; LCD operates per token with micro-rollouts.  
- Verifier reranking and value-guided search: typically rerank beams or full generations; LCD supplies a lightweight, local risk signal.  
- Speculative decoding: accelerates acceptance of tokens; LCD aims at quality via semantic stability and is composable post-speculative acceptance.  
- SelfCheck/consistency-based hallucination detection: scores contradictions post hoc; LCD injects a preventive, token-local agreement prior.  
- Contrastive decoding (e.g., DoLa): internal contrast without external lookahead; LCD explicitly probes future branches.

## 6. Limitations and safeguards
- Overhead remains nontrivial; practicality hinges on gating and batching.  
- NLI fragility on short text; mitigated via embedding prefilter, specificity prior, and domain-specific structural checks.  
- Diversity suppression and “generic” degeneration; mitigated via specificity prior and scope restriction to reliability-critical spans (e.g., function calls, numeric facts, SQL).  
- Draft–target mismatch; ablate with multiple D and analyze sensitivity.  
- Myopia: short L may miss long-range contradictions; future work: adaptive L, periodic re-checks.

## 7. Conclusion
LCD introduces a token-local, preventive stability prior by scoring semantic agreement among micro-rollouts and modulating logits accordingly. With uncertainty gating, hierarchical scoring, and careful batching, LCD targets reliability gains at acceptable overhead. Our falsification-oriented evaluation will determine whether agreement signals at lookahead scale translate into measurable improvements in factuality and structured accuracy on small open models. If validated, LCD offers a composable primitive for reliability-first inference, especially in agentic workflows where single-token errors are costly.

## 8. Reproducibility checklist
- Open-source code with deterministic seeds; HF model IDs and commit hashes.  
- Scripts for batching/KV reuse; FLOPs/latency logging per component (T, D, E/S).  
- Configs for all hyperparameters and gating thresholds; evaluation scripts and prompts.  
- Results with 5 seeds and bootstrap CIs; ablation notebooks.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
