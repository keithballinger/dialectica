Minor Revisions

The paper presents a novel and well-motivated approach to speculative decoding. The core idea of framing multiple branches as a coding theory problem and fusing them before verification is elegant and promising. The proposed method is detailed, and the experimental plan is rigorous, featuring appropriate baselines, ablations, and clear falsification criteria.

The revisions are minor and focus on improving clarity and precision:
1.  **Clarity of Heuristics:** The connection between the formal coding-theoretic model (which assumes known error probabilities) and the practical reliability model (a linear combination of heuristic features) should be made more explicit. The current model is a sensible engineering choice, but it is not a direct derivation from the theory.
2.  **Conciseness:** The prose can be tightened throughout for greater impact. The title is slightly long, and the abstract can be more direct.
3.  **Motivation:** The justification for certain design choices in the fusion rule, such as the soft-voting term, could be briefly stated.

The concept is strong and, if validated by the proposed experiments, would be a significant contribution.

### Revised Draft

# Error-Correcting Speculative Decoding

## Abstract

Speculative decoding accelerates large language model (LLM) inference by using a small "drafter" model to propose multi-token sequences that are verified in a single pass by a large "verifier" model. Existing schemes typically accept or reject a single drafted sequence, discarding the valuable information present in multiple speculative branches. We propose Error-Correcting Speculative Decoding (ECSD), a method that treats speculative branches as noisy, redundant encodings of the desired token sequence. Before verification, ECSD fuses these branches using a reliability-weighted vote to construct a single, higher-fidelity proposal. Under a fixed verifier compute budget, this error-correction step increases the probability that the fused proposal matches the verifier's output, improving both token acceptance rates and model perplexity. We provide: (i) a coding-theoretic formulation of multi-branch drafting; (ii) a practical, on-the-fly reliability estimator using drafter uncertainty and inter-branch agreement, optionally refined with sparse verifier spot-checks; (iii) a fusion rule that approximates MAP decoding; and (iv) a falsifiable experimental plan using open models to validate that ECSD outperforms standard speculative decoding and unweighted voting baselines.

## Introduction

Speculative decoding amortizes the high computational cost of a large verifier model by drafting several future tokens with a much smaller drafter model and verifying them in a single parallel forward pass. The resulting speedup depends critically on the number of drafted tokens the verifier accepts. Current methods often generate multiple branches (via diverse sampling or tree search) but typically use them only for fallback after a primary branch is rejected, rather than for synergistic improvement.

This approach discards the rich information contained across multiple speculative branches. We view these branches from a coding-theoretic perspective: as noisy, redundant encodings of the verifier's intended sequence. This reframing suggests a new strategy. Instead of verifying and selecting one branch, we can first fuse them to correct errors.

We hypothesize that by generating several low-cost branches, estimating the reliability of each, and performing a weighted vote, we can construct a single, higher-fidelity proposal. This fused sequence should align more closely with the verifier, increasing the expected accepted prefix length and reducing perplexity without increasing the number of verifier forward passes.

Our contributions are:
- A coding-theoretic framing of speculative branches as repetition codes transmitted through noisy channels.
- ECSD, a reliability-weighted branch fusion algorithm that strictly controls the verifier budget.
- A practical reliability estimation method using drafter-side signals and optional, budget-constrained verifier spot checks.
- A falsifiable experimental plan with an open-source implementation path.

## Method

### Setup
- **Verifier (V):** A large LLM (e.g., Mistral-7B) producing a target distribution `p_V(· | x)`.
- **Drafter (D):** A small LLM (e.g., TinyLlama-1.1B) proposing `B` branches. Each branch `b ∈ {1..B}` is a sequence of `k` tokens, `t_{b,1:k}`, with associated drafter distributions `q_{b,j}(·)`.
- **Standard Approach:** Verify a single branch and accept the longest prefix consistent with `V`.
- **ECSD Approach:** Fuse `B` branches into a single proposal `t*_{1:k}`, then verify this fused sequence.

### Coding-Theoretic View
For each position `j`, the verifier's intended token `y_j` is drawn from `p_V(· | x, t*_{1:j-1})`. Each branch `b` provides a noisy observation `t_{b,j}` with an unknown error probability `ε_{b,j} = P(t_{b,j} ≠ y_j)`. If we assume branch errors are independent and we know each `ε_{b,j}`, the Maximum a Posteriori (MAP) decoding of `y_j` reduces to a reliability-weighted vote. Specifically, we would choose the token `y_j` that maximizes `∑_b log(P(t_{b,j} | y_j))`, where `P(t_{b,j} | y_j)` is `1-ε_{b,j}` if `t_{b,j} = y_j` and `ε_{b,j}/(V-1)` otherwise (assuming uniform error distribution). Since `ε_{b,j}` is unknown, we estimate a proxy for reliability to use as a voting weight.

### Reliability Estimation
As true error rates are unknown, we construct a practical reliability score `r_{b,j}` for each token `t_{b,j}` using a linear combination of observable signals:
- **Drafter Uncertainty:** The negative entropy of the drafter's distribution, `u_{b,j} = -H(q_{b,j})`. Lower entropy suggests higher confidence.
- **Inter-branch Agreement:** The fraction of other branches that sampled the same token `t_{b,j}` at position `j`.
- **Drafter Self-Consistency:** The drafter's assigned probability to its own sampled token, `q_{b,j}(t_{b,j})`.
- **Sparse Verifier Spot Checks:** At a small set `S` of highly uncertain positions, we run `V` to get `p_V(·)`. A consistency feature `c_{b,j}` can be computed (e.g., log-probability ratio), providing a highly informative but expensive signal.

The reliability score is `r_{b,j} = α_1·u_{b,j} + α_2·agreement_{b,j} + α_3·log q_{b,j}(t_{b,j}) + α_4·I[j∈S]·c_{b,j}`. A per-branch score `r_b` can be obtained by averaging `r_{b,j}`. These scores are transformed into positive weights `w_{b,j} = exp(γ·r_{b,j})` for voting. The hyperparameters `α` and `γ` are set on a small held-out dataset.

### Fusion Rule
For each position `j = 1..k`:
1.  **Candidates:** Collect all unique tokens proposed across branches at position `j`, `C_j = ⋃_b {t_{b,j}}`.
2.  **Scoring:** Score each candidate token `t ∈ C_j` via a weighted vote:
    `S(t,j) = ∑_b w_{b,j} · 1[t_{b,j} = t] + λ · (∑_b w_{b,j} · q_{b,j}(t))`
    The optional second term (`λ > 0`) is a "soft vote" that helps recover high-probability tokens from the drafter's distributions that may not have been sampled in any branch.
3.  **Selection:** The fused token is `t*_j = argmax_t S(t,j)`. Ties are broken using verifier logits if available from a spot check, or by average drafter probability otherwise.

The final fused sequence `t*_{1:k}` is then passed to the verifier for a standard single-pass acceptance check.

### Budget Control
To ensure a fair comparison, ECSD's verifier budget is matched to that of a standard speculative decoding baseline. Let `L` be the baseline's average number of verified tokens per generated token. ECSD partitions this budget between sparse pre-fusion spot checks (`R` tokens) and post-fusion acceptance verification (`A` tokens), such that the average `R+A` per generated token does not exceed `L`. If the fused sequence is accepted for more than `L-R` tokens, the verification is stopped to bank the savings, maintaining the budget constraint.

### Practical Algorithm (per step)
1.  **Draft:** Generate `B` branches of length `k` from drafter `D`.
2.  **Estimate Reliability:** Compute drafter-side signals (uncertainty, agreement, self-consistency).
3.  **Spot Check (Optional):** Select up to `R` positions with the highest uncertainty or disagreement. Run `V` at these positions to get logits for reliability estimation. This consumes `R` verifier evaluations.
4.  **Fuse:** Compute reliability weights `w_{b,j}` and use the fusion rule to produce `t*_{1:k}`.
5.  **Verify:** Run `V` on `t*_{1:k}` to find the longest matching prefix, using at most `L-R` verifier evaluations.
6.  **Accept & Correct:** Accept the matched prefix. If a mismatch occurs, sample the next token from the verifier's last computed distribution.

## Experiments (Falsification Plan)

**Goal:** Test if ECSD improves verifier-aligned accuracy (perplexity, acceptance length, task accuracy) for the same verifier compute budget.

- **Models:**
    - Verifiers: Mistral-7B-v0.3, Pythia-6.9B.
    - Drafters: TinyLlama-1.1B, Pythia-1.4B, Qwen1.5-0.5B (including 4-bit quantized versions to test robustness).
- **Data:** WikiText-103 (perplexity), TriviaQA (short-form QA), GSM8K (reasoning).
- **Baselines:**
    1.  **SD-1:** Standard speculative decoding (1 branch).
    2.  **SD-multi:** Multi-branch variant that accepts the first valid branch.
    3.  **Vote-unif:** Unweighted majority vote fusion before verification.
    4.  **Oracle-Top1:** An un-budgeted upper bound selecting the best single branch using full verifier knowledge.
- **ECSD Variants:**
    1.  **ECSD-lite:** No spot checks (`R=0`).
    2.  **ECSD-spot:** Uses `R=1,2` spot checks.
- **Budget Matching:** All methods will be tuned to use the same number of average verifier evaluations per output token as the optimized SD-1 baseline.
- **Metrics:**
    - Perplexity (w.r.t. verifier `V`).
    - Mean acceptance length.
    - Effective throughput (tokens/sec) and verifier evaluations per output token.
    - Downstream task accuracy (Exact Match).
- **Ablations:** Vary branch count `B`, depth `k`, and contribution of each reliability signal.
- **Falsification Criteria:**
    - The hypothesis is rejected if ECSD, at a matched verifier budget, fails to improve both perplexity and acceptance length over SD-1 on at least two datasets.
    - If Vote-unif performs as well as ECSD, the reliability weighting component is not contributing meaningfully.
    - If gains are negligible for practical `B` and `k` values (e.g., `B=4, k=4`), the method may lack practical value.

## Discussion

**Relation to Prior Work:** Unlike standard speculative decoding, which selects from branches, ECSD fuses them. While methods like Medusa add specialized prediction heads, ECSD is model-agnostic and exploits redundancy from any off-the-shelf drafter. It applies voting, typically used for full generations (e.g., self-consistency), at the micro-level of a single decoding step under a strict compute budget.

**Why Reliability Weighting Matters:** Branches are not equally reliable. Signals like drafter entropy and inter-branch agreement provide cheap proxies for channel quality. Weighting allows the fusion process to favor more reliable branches, approximating a MAP estimate and protecting against correlated errors from a few overconfident but wrong branches, a failure mode for unweighted voting.

**Limitations:** The theoretical independence of branch errors is violated in practice. The method, like all speculative decoding, prioritizes efficient generation of the verifier's greedy output, not faithful sampling from its distribution. The overhead of generating and storing branches may be non-trivial on memory-constrained systems.

## Conclusion

We introduce Error-Correcting Speculative Decoding (ECSD), a method that treats multiple speculative branches as a noisy code to be corrected. By fusing branches with a reliability-weighted vote before verification, ECSD constructs a more accurate proposal, increasing acceptance lengths and improving perplexity while adhering to a strict verifier compute budget. This coding-theoretic perspective offers a principled and practical path to enhancing the efficiency of modern LLM inference.
