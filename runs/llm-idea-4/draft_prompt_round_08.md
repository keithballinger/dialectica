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
        Minor Revisions

Brief critique
- Strong, timely contribution to LLM inference with a clear, verifier-free algorithm and solid empirical evidence under matched budgets. Novelty is plausible but should be positioned more sharply against early-consensus/self-consistency variants and MBR/beam-style reranking.
- Provide a more explicit and auditable budget-matching protocol for self-consistency and beam/diverse beam (what counts toward the budget, handling of terminated vs truncated branches).
- Clarify latency methodology (p50/p95, end-to-end vs generation-only), and report dispersion for latency, not only means.
- Tighten the definition of “lattice” and specify the α schedule; fix minor pseudocode issues (iteration index, guess-overhead cap).
- Briefly discuss failure modes relative to “confidently wrong consensus” and how α-scheduling/late probing mitigates them.

Revised Draft
# Lattice-of-Thought Decoding: Budget-Aware Online Pruning for Efficient Reasoning in LLMs

## Abstract
We present Lattice-of-Thought (LoT) decoding, a budget-aware inference algorithm for Large Language Models (LLMs) that improves accuracy per token via shallow parallel exploration with online pruning. LoT expands multiple short reasoning branches in parallel and scores them using only in-model signals: per-token log-probability and cheap cross-branch answer-consistency probes. Unlike linear Chain-of-Thought (CoT) and post-hoc self-consistency, LoT reallocates tokens during generation to promising branches without external verifiers or training. On GSM8K, SVAMP, and ARC-Challenge with Llama-3-8B-Instruct and Mistral-7B-Instruct, LoT outperforms greedy CoT and self-consistency under matched total token budgets and offers a superior accuracy–latency trade-off versus beam/diverse beam search. These results indicate that budget-aware shallow exploration with online pruning is a practical decoding strategy for constrained inference.

## 1. Introduction
LLM reasoning is often token-inefficient. Linear Chain-of-Thought (CoT) commits early to a single path; self-consistency improves robustness but spends entire budgets on many full traces before voting. Search-based methods introduce structure but frequently rely on external verifiers or reward models that add overhead and complicate deployment.

We hypothesize that uncertainty is highest early in reasoning and that low-cost, in-model signals suffice to triage branches online. LoT operationalizes this with: (i) shallow parallel expansion; (ii) cheap answer probes that capture cross-branch agreement; and (iii) likelihood-aware pruning under a fixed token budget. We use “lattice” to denote the partially ordered set of partial traces whose states are periodically compared and pruned based on agreement signals, not to imply explicit state merging.

Contributions:
- A verifier-free, budget-aware decoding algorithm that combines per-token likelihood and cross-branch answer consistency to reallocate tokens online.
- An implementation compatible with open-source models via standard log-prob APIs; no fine-tuning required.
- Evidence that LoT improves accuracy at matched token budgets over greedy CoT and self-consistency, and surpasses beam/diverse beam in accuracy at comparable latency, on GSM8K, SVAMP, and ARC-Challenge with 7–8B models.
- A budget-matching and latency measurement protocol enabling reproducible and auditable comparisons.

## 2. Related Work
- Chain-of-Thought and Self-Consistency: CoT yields single-trace reasoning; self-consistency samples multiple traces and votes post-hoc, increasing token usage. Early-consensus variants accelerate SC by partial voting but typically do not reallocate tokens within a unified budget.
- Tree/Graph-of-Thought and Search: These methods introduce search and often rely on external evaluators or learned rewards. LoT retains search benefits while using only in-model signals and explicit budget accounting.
- Beam and Diverse Beam Search: Beam maximizes sequence likelihood; diverse beam promotes exploration. LoT instead optimizes answer accuracy under a token budget using task-aligned agreement signals and online width/depth scheduling.
- Early-Exit and Reranking; MBR: Best-of-N with truncation/reranking and minimum Bayes risk decoding leverage partial generations or final answers for selection. LoT integrates early consensus signals into the generation loop to guide budget reallocation.

## 3. Lattice-of-Thought (LoT)

### 3.1 Overview
Given a question x and total token budget B_total, LoT maintains a set of active branches. Each iteration expands branches by short chunks, obtains a cheap answer guess per branch, scores branches by combined likelihood and cross-branch agreement, prunes to a target width, and reallocates remaining budget to the top branches. Width and chunk-length schedules shift from exploration to exploitation.

### 3.2 Scoring
- Likelihood s_lm(b): Average per-token log-probability of reasoning tokens in branch b to mitigate length bias.
- Consistency s_cons(b): Each branch produces a cheap guess ŷ_b (≤5 tokens) via a minimal prompt suffix; branches are clustered by normalized guesses. A branch’s consistency score is log(1 + cluster_size).
- Total score: Per-iteration Z-normalization across active branches; s(b) = z(s_lm(b)) + α · z(s_cons(b)), with optional α scheduling that increases over iterations to de-emphasize early noise and amplify late agreement.

Guess normalization:
- Numeric: parse to float/rational; strip formatting; round to task-specific precision.
- Multiple-choice: map to option IDs; resolve aliases.
- Short-form text: lowercase; strip punctuation/whitespace.

### 3.3 Budget Accounting and Schedules
- All generated tokens (reasoning, guesses, and final answer) count toward B_total.
- A typical schedule starts wide/shallow (e.g., W0 = 6, chunk C0 = 8 tokens), then narrows/deepens (e.g., W2 = 3, C2 = 20–24).
- Guess overhead is capped (e.g., ≤15% of B_total) by reducing guess frequency or length late in decoding.
- Temperature τ and top-p are applied consistently across branches.
- Optional α schedule: α_i increases linearly with iteration i to reduce premature pruning from spurious agreement.

### 3.4 Termination and Output
- Branches emitting an explicit final-answer marker (e.g., “Final Answer:”) move to a finished set.
- Decoding stops when budget is exhausted or the active set is empty.
- Final answer: majority over finished branches’ normalized guesses; ties broken by highest s(b). If no finished branches exist, vote over active guesses.

### 3.5 Pseudocode
```
Input: x, budget B_total, width schedule {W_i}, chunk schedule {C_i}, α schedule {α_i}, temperature τ
Initialize: A ← {root branch with prompt(x)}; Finished ← ∅; tokens_used ← 0; i ← 0

while tokens_used < B_total and |A| > 0:
    B_new ← ∅
    for b in A:
        y_chunk, t_chunk ← sample_chunk(b, C_i, τ)   # exactly C_i tokens
        tokens_used += t_chunk
        b' ← append(b, y_chunk)
        if emits_final_answer(b'):
            Finished ← Finished ∪ {b'}
        else:
            if guess_budget_remaining(B_total, tokens_used):
                y_guess, t_guess ← cheap_guess(b')
                tokens_used += t_guess
                b'.guess ← normalize(y_guess)
            B_new ← B_new ∪ {b'}
        if tokens_used ≥ B_total: break

    if |B_new| == 0: break
    s_lm ← avg_logprob(B_new)
    s_cons ← consensus_logcounts(B_new.guess)        # missing guesses → 0
    s_total ← z(s_lm) + α_i · z(s_cons)
    A ← top_k(B_new, k = W_i) by s_total
    i ← i + 1

if |Finished| > 0:
    return majority(Finished.guess, tie_break = max s_total)
else:
    return majority(A.guess, tie_break = max s_total)
```

### 3.6 Complexity and Implementation Notes
- Token complexity: O(Σ_i W_i · C_i). Scoring and clustering are O(|A|) per iteration using hash maps on normalized guesses.
- Batch decoding across branches amortizes overhead; log-probs via standard APIs (e.g., output_scores=True).
- Memory footprint ~ beam search with width max_i W_i.

## 4. Experimental Setup

Tasks
- GSM8K (math word problems), SVAMP (math with structural variation), ARC-Challenge (multiple-choice science).

Models
- Llama-3-8B-Instruct, Mistral-7B-Instruct.
- 8-shot CoT exemplars shared across methods.

Baselines
- Greedy CoT (single trace).
- Self-Consistency (SC): N traces with majority vote; N and per-trace max length chosen to match B_total on average.
- Beam search (fixed beam widths matched to LoT’s early widths) with length-normalized log-prob scoring.
- Diverse beam search (Hamming diversity).
- LoT ablations: likelihood-only (α=0); consistency-only (drop s_lm); no scheduling (fixed width/chunk).

Budget-matching protocol
- For all methods, we count prompt, reasoning, guess, and final-answer tokens toward B_total.
- SC: choose N and per-trace max-length such that expected total tokens per question is within ±1% of B_total, measured over the evaluation set; truncate traces that would exceed B_total.
- Beam/diverse beam: enforce a shared per-step token cap so that the sum of generated tokens across beam hypotheses matches B_total ±1% on average; early-terminated hypotheses stop consuming tokens.
- LoT: guess frequency is reduced late to enforce the ≤15% guess-overhead cap within B_total.

Metrics and hardware
- Primary: accuracy; Secondary: wall-clock latency per question (batch size 16).
- Latency protocol: end-to-end decoding time including model forward passes and LoT scoring; we report mean, p50, and p95 across the evaluation set.
- Hardware: single NVIDIA A100 (80GB), bf16 where available.
- Variance: report mean ± standard deviation over 3 seeds; confidence intervals in appendix.

Hyperparameters
- Default α = 1.0 with optional late ramp; τ = 0.7; width 6 → 3; chunk 8 → 24; guess length ≤5 tokens. Sensitivity sweeps in appendix.

Reproducibility
- Code release includes exact prompts, seeds, decoding configs, token accounting, and normalization rules. We log per-question token usage and latency.

## 5. Results

Main results
At matched budgets, LoT consistently outperforms greedy CoT and SC across datasets and models. Gains are largest under tighter budgets where SC’s post-hoc voting is inefficient.

| Model        | Dataset | Budget | CoT         | Self-Consistency | LoT (Ours)           |
| :----------- | :------ | :----: | :---------- | :--------------- | :------------------- |
| Llama-3-8B-I | GSM8K   |  300   | 49.5 ± 1.1% | 52.3 ± 0.9%      | 58.1 ± 1.2%          |
| Llama-3-8B-I | GSM8K   |  600   | 51.2 ± 1.0% | 56.9 ± 0.8%      | 61.4 ± 0.9%          |
| Llama-3-8B-I | ARC-C   |  300   | 65.1 ± 1.3% | 66.8 ± 1.1%      | 70.2 ± 1.0%          |
| Mistral-7B-I | GSM8K   |  300   | 45.3 ± 1.2% | 47.1 ± 1.0%      | 52.5 ± 1.3%          |
| Mistral-7B-I | SVAMP   |  300   | 68.8 ± 1.5% | 70.4 ± 1.2%      | 74.1 ± 1.1%          |
Table 1: Accuracy (mean ± std dev over 3 seeds) at matched token budgets.

Accuracy–latency trade-off and ablations
Likelihood and consistency are complementary: likelihood-only (α=0) benefits from breadth; consistency contributes task-aligned signal. LoT achieves higher accuracy than beam/diverse beam at comparable latency, and is both more accurate and faster than budget-matched SC due to online pruning.

| Method             | Accuracy | Mean Latency (ms) | p50 (ms) | p95 (ms) |
| :----------------- | :------: | :---------------: | :------: | :------: |
| Greedy CoT         |  49.5%   |       850         |   820    |   980    |
| Beam (W=6)         |  51.1%   |      1950         |  1880    |  2200    |
| Diverse Beam (W=6) |  52.8%   |      2100         |  2010    |  2400    |
| Self-Consistency   |  52.3%   |      2800         |  2700    |  3200    |
| LoT (Ours)         |  58.1%   |     2250          |  2150    |  2550    |
Table 2: GSM8K, Llama-3-8B-Instruct, 300-token budget; batch size 16, A100. Latency includes end-to-end decoding and LoT scoring.

Sensitivity
Performance is robust for α in [0.5, 1.5]; wider initial widths help at small budgets; longer chunks help at larger budgets. Full sweeps in appendix.

Fairness checks
Across datasets and methods, realized average tokens per question were within ±1% of the target B_total. Latency comparisons use identical batching and stopping criteria.

## 6. Discussion

Why LoT helps
Early-stage uncertainty benefits from breadth; cheap in-model agreement signals identify convergence. Online pruning reallocates budget to promising branches before completing full traces, unlike post-hoc SC. The combination of likelihood and consistency is more robust than either signal alone.

Limitations
- Confidently wrong consensus can prune correct minorities; late α ramping and reduced early guess frequency mitigate but do not eliminate this failure mode.
- Guess extraction is task-dependent; open-ended generation may require richer normalization or multi-granular probes.
- Benefits may diminish for tasks solvable with single-step answers or where log-prob correlates poorly with correctness.

Extensions
- Adaptive schedules (bandits) for width/chunk and α.
- Lightweight self-evaluation prompts without external tools.
- Integration with speculative decoding or KV-caching to reduce latency.

## 7. Conclusion
LoT is a simple, verifier-free, budget-aware decoding algorithm that improves accuracy per token by combining shallow exploration with online pruning using in-model signals. Experiments on GSM8K, SVAMP, and ARC-Challenge with 7–8B models show consistent gains over greedy CoT and self-consistency at matched budgets and favorable accuracy–latency trade-offs versus beam variants. LoT is practical for open-source inference and offers a strong baseline for token-efficient reasoning.

## 8. Reproducibility Checklist
- Code release with scripts for all experiments, prompts, and seeds.
- Exact decoding configs (τ, top-p, width/chunk schedules, α schedule).
- Logged per-question token counts and latency (mean, p50, p95).
- Parsing/normalization rules and answer mapping.
- Random-seed variability (std dev and confidence intervals).
- Hardware and batching details.

## 9. Ethical and Societal Impact
Token-efficient reasoning reduces compute cost and energy for common reasoning workloads. However, efficiency without reliability can amplify confidently wrong outputs; practitioners should calibrate α and schedules and monitor failure modes on safety-critical tasks.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
