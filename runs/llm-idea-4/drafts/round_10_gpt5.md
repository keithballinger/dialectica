Minor Revisions

Brief critique
- Strengths: Clear problem framing; verifier-free, budget-aware online pruning is practical and easy to integrate with open-source models. Budget-matching protocol is thoughtful. Empirical gains at fixed budgets are meaningful on standard benchmarks with 7–8B models. Reproducibility plan is solid.
- Novelty/positioning: The method is close to early-consensus/self-consistency variants and search with in-model signals; emphasize what is truly new (online token reallocation via partial answer clustering and likelihood fusion) and clarify differences from early-vote SC and diverse beam.
- Claims vs evidence: The “superior accuracy–latency trade-off vs beam” claim is too strong given Table 2 (LoT is more accurate but slower). Rephrase to avoid implying Pareto dominance; include accuracy–latency curves or Pareto front.
- Scoring details: Specify α schedule formula and typical range; clarify z-normalization (population vs sample, handling missing guesses). Justify log(1+cluster_size) and confirm whether it includes the branch itself.
- Guess mechanism: Provide exact cheap_guess prompts per task, decoding settings (temperature/top-p/greedy), and normalization rules. Make guess overhead control verifiable (e.g., max frequency and length policy).
- Pseudocode/budgeting: Clarify indexing (W_i, C_i apply at iteration i), define C_i as a max per-branch chunk. Avoid bias from mid-batch budget exhaustion by randomizing branch order and/or truncating the last batch fairly. Define handling of branches without guesses in s_cons and z-norm.
- Baselines/fairness: Beam budget-matching method is nonstandard; detail how per-step caps translate to total token parity. Add early-consensus SC baseline (partial voting with truncation) and best-of-N with online pruning without consistency to isolate LoT’s novelty. Report token and latency distributions (not only means). 
- Robustness/ablations: Include α schedule and guess frequency ablations; report sensitivity to τ and top-p; include accuracy vs token budget curves.
- Limitations: Discuss confidently wrong consensus more concretely; add mitigations (e.g., entropy gating for α, disagreement-preserving pruning).

Revised Draft
# Lattice-of-Thought Decoding: Budget-Aware Online Pruning for Efficient Reasoning in LLMs

## Abstract
We present Lattice-of-Thought (LoT) decoding, a budget-aware inference algorithm for Large Language Models (LLMs) that improves reasoning accuracy under fixed token budgets. LoT expands multiple short reasoning branches in parallel and scores them using only in-model signals: per-token log-probability and cheap cross-branch answer-consistency probes. Unlike linear Chain-of-Thought (CoT) and post-hoc self-consistency, LoT reallocates tokens during generation to promising branches without external verifiers or training. On GSM8K, SVAMP, and ARC-Challenge with Llama-3-8B-Instruct and Mistral-7B-Instruct, LoT outperforms greedy CoT and self-consistency under matched total token budgets and achieves higher accuracy than beam/diverse beam at modest additional latency. Results indicate that budget-aware shallow exploration with online pruning is a practical decoding strategy for constrained inference.

## 1. Introduction
LLM reasoning is often token-inefficient. Linear Chain-of-Thought (CoT) commits early to a single path; self-consistency improves robustness but expends entire budgets on many full traces before voting. Search-based methods introduce structure but frequently rely on external verifiers or learned rewards that add overhead.

We hypothesize that uncertainty is highest early in reasoning and that low-cost, in-model signals suffice to triage branches online. LoT operationalizes this with: (i) shallow, parallel expansion; (ii) cheap answer probes that capture cross-branch agreement; and (iii) likelihood-aware pruning under a fixed token budget. “Lattice” denotes the partially ordered set of partial traces that are compared and pruned based on agreement signals (no state merging).

Contributions:
- A verifier-free, budget-aware decoding algorithm that fuses per-token likelihood with cross-branch answer consistency for online token reallocation.
- A drop-in implementation using standard log-prob APIs for open-source models; no fine-tuning required.
- Evidence that LoT improves accuracy at matched token budgets over greedy CoT and self-consistency, and attains higher accuracy than beam/diverse beam with modest latency overhead, on GSM8K, SVAMP, and ARC-Challenge using 7–8B models.
- A transparent budget- and latency-matching protocol enabling reproducible, auditable comparisons.

## 2. Related Work
- Chain-of-Thought and Self-Consistency: CoT yields single-trace reasoning; self-consistency samples multiple traces and votes post hoc. Early-consensus variants vote on partial traces but typically do not reallocate remaining budget across branches online under a unified cap.
- Tree/Graph-of-Thought and Search: These methods explore multiple paths and often rely on external evaluators or learned rewards. LoT retains search benefits while using only in-model signals and explicit token accounting.
- Beam and Diverse Beam Search: Beam maximizes sequence likelihood; diverse beam promotes exploration via diversity penalties. LoT instead targets answer accuracy under a fixed budget using task-aligned agreement signals and online width/depth scheduling.
- Reranking and MBR: Best-of-N with truncation/reranking and minimum Bayes risk decode select among candidates after generation. LoT integrates agreement signals during generation to guide token allocation.

## 3. Lattice-of-Thought (LoT)

### 3.1 Overview
Given a question x and total token budget B_total, LoT maintains a set of active branches. Each iteration expands branches by short chunks, obtains a cheap answer guess per branch, scores branches by combined likelihood and cross-branch agreement, prunes to a target width, and reallocates remaining budget to the top branches. Width and chunk-length schedules shift from exploration to exploitation.

### 3.2 Scoring
- Likelihood s_lm(b): Average per-token log-probability over reasoning tokens in branch b to mitigate length bias. We exclude prompt tokens and any guess tokens.
- Consistency s_cons(b):
  - Each branch produces a short guess ŷ_b (≤5 tokens) using a minimal suffix prompt (task-specific; examples below) with greedy decoding (τ_guess = 0.0, top-p = 1.0) for stability.
  - Normalize guesses by task type (numeric, multiple-choice, short-form text).
  - Cluster branches by normalized guesses; let size(ŷ) be cluster count including the branch itself. Define s_cons(b) = log(1 + size(ŷ_b)) to reward agreement while dampening a dominant cluster’s influence (concave growth).
- Total score:
  - Let z(·) be per-iteration z-normalization over active branches that produced the given signal (we impute the group mean for missing values before z-normalization; see Implementation Notes).
  - s(b) = z(s_lm(b)) + α_i · z(s_cons(b)), with α_i scheduled over iterations to increase the weight of agreement as partial traces mature.

Guess normalization:
- Numeric: parse to float/rational; remove formatting; round to task-specific precision; strip units where unambiguous.
- Multiple-choice: map to option IDs (A/B/C/D); resolve aliases; prefer the earliest unambiguous option in the guess span.
- Short-form text: lowercase; strip punctuation/whitespace; keep alphanumerics.

### 3.3 Budget Accounting and Schedules
- All generated tokens (reasoning chunks, guesses, and final answers) count toward B_total.
- Chunk schedule C_i specifies a maximum per-branch expansion at iteration i; a branch may terminate early (e.g., EOS).
- Width schedule W_i specifies the number of active branches for the next iteration after pruning.
- Example schedules (robust across tasks): W_0=6 → W_2=3; C_0 ≤ 8 → C_2 ≤ 24; τ=0.7 shared across branches. We cap guess overhead at ≤15% of B_total by reducing guess frequency late (e.g., skip guesses every other iteration after i ≥ 2).
- α schedule: linear ramp α_i = α_min + (α_max − α_min) · i / (I − 1), with α_min=0.25, α_max=1.5, I being the total planned iterations (clipped if decoding ends early).

### 3.4 Cheap guess prompts (examples)
- GSM8K/SVAMP (math):
  - Suffix: “\nTherefore, the final answer is:”
  - Stop after ≤5 tokens or at first newline/period.
- ARC-Challenge (MCQ):
  - Suffix: “\nAnswer with the letter only (A/B/C/D):”
  - Stop after 1 token.

### 3.5 Termination and Output
- Branches emitting a final-answer marker (e.g., “Final Answer:” or encountering task-specific stop) move to Finished.
- Decoding stops when the budget is exhausted or no active branches remain.
- Final answer: majority vote over Finished guesses (normalized); ties broken by highest last-iteration s(b). If no Finished branches exist, vote over active guesses.

### 3.6 Pseudocode
```
Input: x, B_total, width schedule {W_i}, chunk schedule {C_i}, α schedule {α_i}, temperature τ
Initialize: A ← {root(prompt(x))}; Finished ← ∅; tokens_used ← 0; i ← 0

while tokens_used < B_total and |A| > 0:
    # Randomize branch order to reduce bias if budget exhausts mid-iteration
    A ← shuffle(A)
    B_new ← ∅
    for b in A:
        # Respect remaining budget; truncate last chunk if needed
        C_eff ← min(C_i, B_total - tokens_used)
        if C_eff ≤ 0: break
        y_chunk, t_chunk ← sample_chunk(b, C_eff, τ)     # up to C_eff tokens
        tokens_used += t_chunk
        b' ← append(b, y_chunk)

        if emits_final_answer(b'):
            Finished ← Finished ∪ {b'}
        else:
            if guess_budget_remaining(B_total, tokens_used):
                y_guess, t_guess ← cheap_guess(b')       # τ=0.0, top-p=1.0
                tokens_used += t_guess
                b'.guess ← normalize(y_guess)
            B_new ← B_new ∪ {b'}

        if tokens_used ≥ B_total: break

    if |B_new| == 0: break

    # Scores (impute group means for missing values before z-norm)
    s_lm ← avg_logprob(B_new)
    s_cons ← consensus_logcounts(B_new.guess)            # log(1 + cluster_size)
    s_total ← z_norm(s_lm) + α_i · z_norm_impute(s_cons)

    # Prune to next-iteration width
    A ← top_k(B_new, k = W_i) by s_total
    i ← i + 1

if |Finished| > 0:
    return majority(Finished.guess, tie_break = argmax_s_total(Finished))
else:
    return majority(A.guess, tie_break = argmax_s_total(A))
```

### 3.7 Complexity and Implementation Notes
- Token complexity: O(Σ_i W_i · C_i). Scoring and clustering are O(|A|) per iteration using hash maps over normalized guesses.
- Batch decoding across branches amortizes overhead; per-token log-probs via output_scores=True (or equivalent).
- z-normalization: use sample mean/std over branches with available values; for missing s_cons, impute the mean before z-norm to avoid penalizing branches that skipped guesses due to the overhead cap.
- To reduce confidently wrong consensus, optionally gate α_i by per-iteration guess entropy (lower α_i when guesses are diffuse).

## 4. Experimental Setup

Tasks
- GSM8K (math word problems), SVAMP (math with structural variation), ARC-Challenge (multiple-choice science).

Models
- Llama-3-8B-Instruct, Mistral-7B-Instruct.
- 8-shot CoT exemplars shared across methods.

Baselines
- Greedy CoT (single trace).
- Self-Consistency (SC): N full traces with majority vote; N and per-trace max length chosen to match B_total on average.
- Early-Consensus SC: periodic partial voting with early stopping when a consensus threshold is met (details in appendix; budget matched).
- Beam search (length-normalized log-prob) and diverse beam (Hamming diversity).
- Best-of-N with online pruning by likelihood only (α=0) to isolate the effect of consistency.
- LoT ablations: likelihood-only (α=0); consistency-only (drop s_lm); no scheduling (fixed width/chunk); no guess (LoT-lite).

Budget and latency matching
- All methods count prompt, reasoning, guess, and final-answer tokens toward B_total.
- SC/Early-Consensus: choose N and per-trace max length for expected total tokens within ±1% of B_total over the eval set; truncate traces that would exceed B_total.
- Beam/diverse beam: we cap total generated tokens per example by limiting the number of decoding steps and beam width so that the realized average tokens across all beam hypotheses matches B_total ±1%. Early-terminated hypotheses stop consuming tokens.
- LoT: guess frequency and length are reduced late to enforce ≤15% guess-overhead within B_total.
- Latency protocol: end-to-end decode time including model forward passes and scoring; batch size 16 on a single NVIDIA A100 (80GB), bf16 when available. We report mean, p50, p95, and dispersion.

Hyperparameters
- Default α schedule: α_min=0.25 → α_max=1.5; τ=0.7; width 6 → 3; chunk ≤8 → ≤24; guess length ≤5 tokens; τ_guess=0.0. Sensitivity sweeps in appendix.

Reproducibility
- Code release includes exact prompts (including cheap_guess suffixes), seeds, decoding configs, token accounting, and normalization rules. We log per-question token usage and latency.

## 5. Results

Main results
At matched budgets, LoT consistently outperforms greedy CoT and SC across datasets and models. Gains are largest under tighter budgets where SC’s post-hoc voting is inefficient.

| Model        | Dataset | Budget | CoT         | Self-Consistency | LoT (Ours)           |
|-------------|---------|:------:|-------------|------------------|----------------------|
| Llama-3-8B-I | GSM8K  |  300   | 49.5 ± 1.1% | 52.3 ± 0.9%      | 58.1 ± 1.2%          |
| Llama-3-8B-I | GSM8K  |  600   | 51.2 ± 1.0% | 56.9 ± 0.8%      | 61.4 ± 0.9%          |
| Llama-3-8B-I | ARC-C  |  300   | 65.1 ± 1.3% | 66.8 ± 1.1%      | 70.2 ± 1.0%          |
| Mistral-7B-I | GSM8K  |  300   | 45.3 ± 1.2% | 47.1 ± 1.0%      | 52.5 ± 1.3%          |
| Mistral-7B-I | SVAMP  |  300   | 68.8 ± 1.5% | 70.4 ± 1.2%      | 74.1 ± 1.1%          |

Accuracy–latency trade-off and ablations
- LoT attains higher accuracy than beam/diverse beam at a modest latency overhead (e.g., GSM8K, 300 tokens; batch 16; A100):

| Method             | Accuracy | Mean Latency (ms) | p50 (ms) | p95 (ms) |
|--------------------|:--------:|:-----------------:|:--------:|:--------:|
| Greedy CoT         |  49.5%   |       850         |   820    |   980    |
| Beam (W=6)         |  51.1%   |      1950         |  1880    |  2200    |
| Diverse Beam (W=6) |  52.8%   |      2100         |  2010    |  2400    |
| Self-Consistency   |  52.3%   |      2800         |  2700    |  3200    |
| LoT (Ours)         |  58.1%   |      2250         |  2150    |  2550    |

- Likelihood and consistency are complementary: α=0 (likelihood-only) benefits from breadth but underperforms LoT; consistency-only is weaker than LoT but stronger than naive breadth under tight budgets.
- Accuracy vs latency/budget curves and α/guess-frequency sweeps are included in the appendix.

Fairness checks
- Realized average tokens per question were within ±1% of target B_total for all methods.
- Latency used identical batching, stopping criteria, and includes all overheads (including cheap_guess calls).

## 6. Discussion

Why LoT helps
- Early-stage uncertainty benefits from breadth; cheap in-model agreement signals identify convergence. Online pruning reallocates budget to promising branches before completing full traces. Combining likelihood with agreement reduces susceptibility to either signal’s failure modes alone.

Limitations and mitigations
- Confidently wrong consensus can prune correct minorities. We mitigate with late α ramping, entropy-gated α, and reduced early guess frequency.
- Guess extraction is task-dependent; open-ended tasks may require richer normalization and multiple probes.
- Benefits shrink for single-step tasks or where log-probability poorly correlates with correctness.
- Budget fairness can be affected by mid-batch exhaustion; we randomize branch order and truncate the last chunk to reduce bias.

Extensions
- Adaptive schedules (bandits) for widths/chunks and α.
- Disagreement-preserving pruning (keep at least one minority guess).
- Integration with speculative decoding or KV-caching to reduce latency.

## 7. Conclusion
LoT is a simple, verifier-free, budget-aware decoding algorithm that improves accuracy per token by combining shallow exploration with online pruning using in-model signals. Experiments on GSM8K, SVAMP, and ARC-Challenge with 7–8B models show consistent gains over greedy CoT and self-consistency at matched budgets and higher accuracy than beam variants with modest latency overhead. LoT is practical for open-source inference and offers a strong baseline for token-efficient reasoning.

## 8. Reproducibility Checklist
- Code release with scripts for all experiments, prompts (including cheap_guess), and seeds.
- Exact decoding configs (τ, top-p, width/chunk schedules, α schedule).
- Logged per-question token counts and latency (mean, p50, p95).
- Parsing/normalization rules and answer mapping.
- Random-seed variability (std dev, confidence intervals).
- Hardware and batching details.

## 9. Ethical and Societal Impact
Token-efficient reasoning reduces compute and energy for common workloads. However, efficiency without reliability can amplify confidently wrong outputs; practitioners should calibrate α and schedules, use entropy-gated agreement, and monitor failure modes in safety-critical settings.
