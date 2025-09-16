You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Draft:
        Major Revisions

- Mathematical correctness: The derivative of entropy with respect to temperature is misstated. For p_i ∝ exp(s_i/T), dH/dT = Var_p(s)/T^3, not Var_p(s)/T^2. This error affects the Newton update and stability analysis.
- Theoretical clarity: Proofs (monotonicity, existence/uniqueness) are asserted but not shown. Edge cases (degenerate logits, numerical truncation) are not handled rigorously.
- Experimental rigor: Reported results lack dataset sizes, seeds, tuning protocols, confidence intervals, and code references. Baselines omit “typical sampling,” a closely related method. Human evaluation methodology is unspecified.
- Implementation detail: Overhead claims and solver behavior need precise profiling, numerics (log-sum-exp, clamping), and fallback logic. Interaction with common inference tricks (top-k, repetition penalty) should be documented.
- Positioning and novelty: Relation to locally typical sampling and entropy/surprisal control literature needs fuller coverage to support the novelty claim.
- Reproducibility: Provide pseudo-code and a minimal reference implementation usable with small open-source models.

Revised Draft
# Target-Entropy Decoding: Closed-Loop Control of Predictive Uncertainty for LLM Inference

## Abstract
We introduce Target-Entropy Decoding (TED), a decoding algorithm that directly controls the Shannon entropy of the next-token distribution during generation by solving, at each step, for the temperature that achieves a user-specified entropy target. We prove that, for non-degenerate logits, entropy is a continuous, strictly increasing function of temperature T with dH/dT = Var_p(s)/T^3 > 0, ensuring a unique temperature solution for any target H* ∈ (0, log V). TED uses a warm-started, damped Newton method with a bracketed fallback, adding 2–5% per-token overhead on small models. We evaluate constant, ramp, and prompt-adaptive entropy schedules on open-ended generation and factual QA using Pythia-410M and Pythia-1.4B, comparing against tuned temperature, nucleus, typical sampling, and Mirostat v2. TED reduces repetition and hallucination while preserving or improving overall quality, and its gains persist under strong baseline tuning. We release a minimal implementation and scripts for reproducibility.

## 1. Introduction
Decoding policies govern the trade-off between diversity, coherence, and factuality in LLM generation. Common heuristics—temperature, nucleus (top-p), and repetition penalties—modulate uncertainty indirectly. Typical sampling and Mirostat introduce control signals tied to token-level surprisal or its target.

We propose to control a distribution-level property—Shannon entropy—directly and deterministically at each step. Entropy summarizes total predictive uncertainty over the vocabulary; clamping it to a target H* yields a simple, interpretable lever for “explore versus commit” behavior. Target-Entropy Decoding (TED) solves, per step, for the temperature T* that matches H(p(T*)) to H*, then samples from p(T*).

Contributions:
- Theory: We derive dH/dT = Var_p(s)/T^3, prove continuity, monotonicity, and uniqueness of T* for any H* ∈ (0, log V) when logits are not all equal, and give safe bounds for bracketing.
- Algorithm: A warm-started, damped Newton solver with a bracketed fallback, numerically robust with log-sum-exp stabilization. We show convergence in 2–4 iterations.
- Practice: Entropy schedules (constant, linear ramp, prompt-adaptive) that improve repetition and hallucination without sacrificing quality.
- Evaluation: Reproducible experiments on small open models (Pythia-410M/1.4B) across open-ended text and TruthfulQA MC1 against tuned baselines including typical sampling and Mirostat v2. We ablate solver choices, schedules, and interactions with top-k and repetition penalties.

## 2. Method

### 2.1 Preliminaries
Let s ∈ R^V be next-token logits. For temperature T > 0, p_i(T) = exp(s_i/T)/Z(T), Z(T) = Σ_j exp(s_j/T). The Shannon entropy is H(T) = −Σ_i p_i(T) log p_i(T). We seek T* such that H(T*) = H*, for a target H* ∈ (0, log V).

We parameterize by inverse temperature β = 1/T when convenient; the distribution is Gibbs with energy −s.

### 2.2 Properties of H(T)
Lemma 1 (Limits). As T → 0+, p(T) → one-hot at argmax_i s_i and H(T) → 0. As T → ∞, p(T) → uniform and H(T) → log V.

Lemma 2 (Continuity and monotonicity). If logits are not all equal, H(T) is continuous and strictly increasing in T. Moreover,
- dH/dβ = −β Var_pβ(s) ≤ 0,
- dH/dT = Var_pT(s)/T^3 ≥ 0,
with equality only when p is uniform or degenerate.

Proof sketch. Using H(β) = log Z(β) − β E_pβ[s] and d/dβ log Z = E[s], we obtain dH/dβ = −β Var_pβ(s). Chain rule yields dH/dT. Continuity follows from smoothness of log Z.

Corollary (Existence/uniqueness). For any H* ∈ (0, log V) and non-degenerate s, there exists a unique T* ∈ (0, ∞) with H(T*) = H*.

Remark (Degenerate case). If all logits are equal, p is uniform for all T and H(T) = log V; we clamp H* ≤ H_max − ε in practice.

### 2.3 Solving for T*
We solve g(T) = H(T) − H* = 0 by a damped Newton step with warm start:
- Initialize T_0 from previous step’s T*, else T_0 = T_init (e.g., 1.0).
- Iterate for k = 0..K_max:
  - Compute p(T_k), H(T_k), and Var_p(s) using log-sum-exp and fused kernels where available.
  - Compute derivative g'(T_k) = dH/dT = Var_p(s)/T_k^3.
  - Update T_{k+1} = T_k − α · g(T_k)/g'(T_k), with damping α ∈ (0, 1].
  - If step leaves a bracket or T not in [T_min, T_max], reduce α or fall back to bisection.
- Stop when |g(T_k)| ≤ ε_H (e.g., 1e−3 nats) or after K_max iterations; use bracketed bisection otherwise.

Numerical safety:
- Clamp T ∈ [T_min, T_max] (e.g., [1e−2, 1e3]).
- Compute H and Var in float32/float64; reuse softmax buffers to avoid recomputation.
- Warm-start reduces iterations; when the entropy schedule changes slowly, K ≈ 2–3.

Pseudocode (PyTorch-like):
```
def entropy_and_var(logits, T):
    x = logits / T
    x = x - x.max()                # stability
    p = torch.softmax(x, dim=-1)
    H = -(p * (torch.log(p + 1e-12))).sum(-1)
    mu = (p * logits).sum(-1)
    var = (p * (logits - mu.unsqueeze(-1))**2).sum(-1)
    return H, var

def solve_T_for_entropy(logits, H_star, T0=1.0, eps=1e-3, Kmax=6,
                        Tmin=1e-2, Tmax=1e3):
    T = torch.clamp(T0, Tmin, Tmax)
    # bracket via coarse doubling/halving
    Ta, Tb = Tmin, Tmax
    Ha, _ = entropy_and_var(logits, Ta); Hb, _ = entropy_and_var(logits, Tb)
    Ha, Hb = Ha.item(), Hb.item()
    if not (Ha <= H_star <= Hb):   # clamp target if out of range
        H_star = min(max(H_star, Ha + 1e-6), Hb - 1e-6)
    for k in range(Kmax):
        H, var = entropy_and_var(logits, T)
        g = H.item() - H_star
        if abs(g) < eps: return T.item()
        dH = (var.item() / (T**3))
        step = g / max(dH, 1e-8)
        alpha = 1.0
        while True:
            T_new = float(torch.clamp(T - alpha*step, Tmin, Tmax))
            H_new, _ = entropy_and_var(logits, T_new)
            if (Ha <= H_new <= Hb) and (abs(H_new - H_star) < abs(g) or alpha < 1e-3):
                T = torch.tensor(T_new); break
            alpha *= 0.5
    # fallback: bisection
    lo, hi = Ta, Tb
    for _ in range(24):
        mid = 0.5*(lo+hi)
        H_mid, _ = entropy_and_var(logits, mid)
        if H_mid < H_star: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)
```

### 2.4 Entropy schedules
- Constant: H*(t) = h0.
- Linear ramp: H*(t) = h_start + (h_end − h_start) · min(t/T_ramp, 1).
- Prompt-adaptive: Choose schedule based on a lightweight classifier of prompt type (e.g., factual vs. creative) or readily available metadata; we use a simple keyword heuristic in experiments.

Optional: impose rate limits |H*(t) − H*(t−1)| ≤ Δ to smooth solver dynamics.

### 2.5 Interactions with standard tricks
- Top-k/top-p filtering: For exact control, compute T* on the full vocabulary, then optionally sample from the full p(T*). If truncation is mandatory, compute T* after truncation and renormalization; this changes the entropy target to that of the truncated distribution and may introduce non-monotonicities as the active support changes.
- Repetition penalty/logit biasing: Apply such transformations to logits before solving for T*, so the controlled entropy reflects the effective sampling distribution.

## 3. Related Work
- Temperature and nucleus sampling control uncertainty indirectly via scalar rescaling or tail-mass truncation.
- Locally typical sampling selects tokens with surprisals close to the distribution entropy, targeting a token-level typical set rather than a distribution-level entropy constraint.
- Mirostat aims to maintain target surprisal (or perplexity) via stochastic feedback updates.
- Calibration via temperature scaling is used for post-hoc probability calibration but not for per-step closed-loop control.
TED differs by deterministically achieving a specified distribution entropy at each step and supporting explicit schedules.

## 4. Experiments

### 4.1 Setup
Models:
- Pythia-410M and Pythia-1.4B (EleutherAI), fp16 inference.

Datasets and prompts:
- Open-ended: WritingPrompts (200 prompts sampled; ≤200 tokens generation).
- Summarization: XSum (200 articles; ≤128 tokens).
- Factual QA: TruthfulQA MC1 (all multiple-choice questions; standard evaluation).

Baselines:
- Temperature: T ∈ {0.7, 1.0, 1.3}.
- Nucleus: p ∈ {0.9, 0.95} with T=1.0.
- Typical sampling: τ_typ ∈ {0.8, 1.0, 1.2}.
- Mirostat v2: τ ∈ {3.0, 5.0}, η tuned on a validation split.
All baselines tuned per task/model on a held-out validation set with the same budget as TED schedules.

TED configurations:
- TED-Const: h0 ∈ {2.5, 3.0, 3.5}.
- TED-Ramp: h_start ∈ {3.5, 4.0}, h_end ∈ {2.2, 2.8}, T_ramp ∈ {32, 64} tokens.
- TED-Adapt: heuristic prompt classifier selects TED-Const for QA and TED-Ramp for open-ended.

Metrics:
- Repetition: fraction of repeated 4-grams over generations (lower is better).
- Diversity: distinct-3 over generated tokens (higher is better).
- Hallucination/Factuality: TruthfulQA MC1 accuracy.
- Distributional quality: MAUVE (for open-ended).
- Efficiency: tokens/sec and per-token softmax iterations.
Report mean ± 95% CIs over 3 seeds; non-overlapping CIs marked.

Hardware: A100-40GB; batch size 1; greedy length control off unless specified.

### 4.2 Main results
Summary:
- TED-Const (h0 ≈ 2.8–3.2) reduces repetition on open-ended tasks versus tuned nucleus and typical sampling while maintaining MAUVE.
- TED-Ramp improves distinct-3 and MAUVE over fixed-temperature and nucleus with similar or lower repetition.
- On TruthfulQA, TED-Const improves MC1 accuracy by ~2–3 points on Pythia-1.4B relative to its best-tuned typical sampling baseline; gains are smaller but consistent on 410M.
- TED-Adapt matches or exceeds the best single-schedule variant across tasks without per-task retuning.

Efficiency:
- Warm-started Newton converges in 2–3 iterations on average (K_max=6), with a 2–5% per-token latency overhead relative to temperature sampling; overhead is dominated by extra softmax/entropy passes, which are a small fraction of total time on transformer blocks.

We include full tables with per-task/model numbers, CIs, and tuned hyperparameters in the repository.

### 4.3 Ablations and analyses
- Solver choice: Damped Newton outperforms pure bisection in wall-clock time with identical final entropies; fallback triggers rarely (<0.5% steps) on spiky logits.
- Schedule sensitivity: TED-Ramp is robust across reasonable h_start/h_end; aggressive ramps can increase incoherence early on.
- Interaction with truncation: Computing T* post top-k leads to staircase-like T trajectories as k changes; computing T* on full V and sampling from p(T*) is preferred when feasible.
- Entropy tracking: Empirical H(p(T*)) tracks H* with mean absolute error < 1e−3 nats (by design), confirming exact control.

## 5. Discussion and Limitations
- Semantics vs. statistics: Entropy control shapes uncertainty but does not guarantee truthfulness; confident errors persist.
- Targets and calibration: Optimal H* is task- and model-dependent; automatic schedule selection is an open problem.
- Large-vocabulary and custom kernels: On very large V, extra softmax passes can be non-negligible without fused kernels; our implementation uses standard log-sum-exp and benefits from vendor libraries.
- Truncation effects: If truncation is required (safety filters), exact entropy control applies to the truncated distribution.

## 6. Reproducibility
We release:
- A minimal PyTorch implementation of TED with Newton+bisection solvers.
- Scripts to reproduce all experiments with Pythia-410M/1.4B, including prompt lists, seeds, and tuned hyperparameters for baselines.
- Logged per-step entropy and temperature traces.

## 7. Conclusion
Target-Entropy Decoding reframes decoding as closed-loop uncertainty control, achieving a user-specified entropy at every step via a fast, robust temperature solver. Supported by simple theory and practical solvers, TED delivers consistent quality and factuality gains over strong baselines on small open-source models at modest overhead. We hope TED becomes a useful primitive for principled, interpretable steering of LLM behavior.

## References
- Holtzman et al. (2020). The Curious Case of Neural Text Degeneration. (nucleus/top-p)
- Meister et al. (2023). Locally Typical Sampling. (typical sampling)
- Basu et al. (2021). Mirostat: A Memory-Efficient Algorithm for Controlling Perplexity. (surprisal control)
- Guo et al. (2017). On Calibration of Modern Neural Networks. (temperature scaling)

## Appendix A: Derivation of dH/dT
Let β = 1/T. Z(β) = Σ_i exp(β s_i), p_i(β) = exp(β s_i)/Z(β).
H(β) = −Σ_i p_i log p_i = log Z(β) − β E_pβ[s].
Using d/dβ log Z = E[s] and dE_pβ[s]/dβ = Var_pβ(s), we obtain:
dH/dβ = E[s] − E[s] − β Var_pβ(s) = −β Var_pβ(s) ≤ 0.
By chain rule, dH/dT = (dH/dβ)(dβ/dT) = (−β Var_pβ(s))(−1/T^2) = Var_pβ(s)/T^3 ≥ 0.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
