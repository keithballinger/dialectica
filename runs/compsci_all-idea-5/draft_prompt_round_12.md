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

Brief critique:
- Novelty and scope: The core idea—closed-loop control of next-token entropy via solving for temperature—is interesting and plausibly novel within LLM decoding. The theory is sound and code-validatable on small open models.
- Key issues to fix:
  1) Reproducibility: Repository URL and commit hashes are placeholders; HuggingFace model revisions, dataset versions, and exact prompts need to be fully specified. Without these, results are not independently reproducible.
  2) Efficiency: Overhead claims need a clear, reproducible microbenchmark protocol, coverage across batch sizes, and a breakdown of extra kernels (softmax/moments). The “Tokens/sec ↓” labeling is incorrect; report both absolute throughput and percentage deltas with CIs.
  3) Truncation interaction: The draft implies “solve-then-truncate” maintains exact control; it does not. You must report realized entropy error for both solve-then-truncate and truncate-then-solve, and state the guarantee precisely (exact only without truncation).
  4) Evaluation: Expand and clarify the evaluation protocol (dataset versions, sample sizes, stopping criteria, tuning budgets). Provide full tables and per-metric 95% CIs. Current numbers are sparse and partly illustrative.
  5) Edge cases and stability: Clarify behavior when variance is near zero, all active logits are equal, or heavy masking reduces V_active; ensure solver safeguards and target clamping are specified and tested.
  6) Positioning: Strengthen related work to ensure novelty (e.g., entropy-targeting or entropy-regularized sampling in adjacent literature) and distinguish from Mirostat/typical sampling more precisely.

Revised Draft
# Target-Entropy Decoding: Closed-Loop Control of Predictive Uncertainty for LLM Inference

## Abstract
We introduce Target-Entropy Decoding (TED), a decoding algorithm that controls the next-token Shannon entropy by solving, at each step, for the temperature T achieving a user-specified entropy target H*. We prove H(T) is continuous and strictly increasing in T when active logits are non-degenerate, with dH/dT = Var_p(s)/T^3 > 0 (nats), ensuring a unique solution T*. We implement a warm-started safeguarded Newton solver with bisection fallback that converges in 2–4 iterations and adds 2–5% per-token latency in vectorized GPU inference. On Pythia-410M and Pythia-1.4B, constant, ramp, and prompt-adaptive entropy schedules reduce repetition and hallucination while preserving or improving open-ended quality and TruthfulQA MC1 versus tuned temperature, nucleus, typical sampling, and Mirostat v2. We release code, scripts, and environment lockfiles to reproduce results.

## 1. Introduction
Decoding policies modulate uncertainty to balance diversity, coherence, and factuality. Temperature, nucleus (top-p), and repetition penalties are indirect controls; locally typical sampling and Mirostat provide surprisal- or token-level feedback. We propose Target-Entropy Decoding (TED), a deterministic, per-step controller that enforces a target distributional entropy H* by solving for the temperature T* with H(p(T*)) = H*. Scheduling H* provides an interpretable lever for exploration (high entropy) versus commitment (low entropy).

Contributions:
- Theory: We show H(T) is continuous and strictly increasing in T when active logits are not all equal, with dH/dT = Var_p(s)/T^3, yielding existence/uniqueness of T* for H* ∈ (0, log V_active).
- Algorithm: A numerically stable, batch-vectorized, warm-started Newton solver with bisection safeguards and log-sum-exp stabilization.
- Practice: Simple schedules (constant, ramps, prompt-adaptive heuristics) that improve repetition/diversity trade-offs and factual QA on small open models.
- Evaluation: Reproducible experiments with baselines (temperature, nucleus, typical sampling, Mirostat v2), ablations (solver, schedules, truncation order), and microbenchmarks (throughput, iterations).

## 2. Method

### 2.1 Setup
Let s ∈ R^V be next-token logits after all masking and penalties. Active set M = {i | s_i > −∞}, size V_active. For temperature T > 0:
p_i(T) = exp(s_i/T)/Z(T), with Z(T) = Σ_{j∈M} exp(s_j/T).
Entropy H(T) = −Σ_{i∈M} p_i(T) log p_i(T) (nats). We seek T* s.t. H(T*) = H*, for H* ∈ (0, log V_active).

### 2.2 Properties of H(T)
Limits: T→0+ yields one-hot at argmax s, H→0; T→∞ yields uniform on M, H→log V_active.

Monotonicity: With β=1/T, H(β)=log Z(β) − β E_pβ[s]. Using d/dβ log Z = E[s] and d/dβ E[s] = Var_pβ(s), we obtain:
dH/dβ = −β Var_pβ(s) ≤ 0 and dH/dT = Var_pT(s)/T^3 ≥ 0,
with strict inequality for finite T when active logits are not identical. Thus, for non-degenerate s on M, H(T) is continuous and strictly increasing.

Existence/Uniqueness: By the intermediate value theorem and strict monotonicity, for any H* ∈ (0, log V_active) there exists a unique T* ∈ (0, ∞).

Degeneracy and masking: If all active logits are equal, H(T)=log V_active for all T; any T is valid (we return T=1). All computations respect masking; the maximal achievable entropy equals log V_active.

### 2.3 Solver
We solve g(T)=H(T)−H*=0 via a warm-started safeguarded Newton method with bisection fallback. Safeguards address small variance regimes and schedule jumps.

Numerics and stability:
- Use log-sum-exp; reuse buffers for moments to amortize cost.
- Clamp T ∈ [T_min, T_max] (e.g., [1e−2, 1e3]) and H* ∈ [ε_H, log V_active − ε_H] (ε_H≈1e−4).
- Detect degeneracy (Var ≈ 0 or all active logits equal); return T=1 and skip iterations.
- Warm-start from previous token’s T* (or previous stream element).

Vectorized PyTorch-like pseudocode:

```python
def entropy_and_moments(logits, T, mask=None):
    # logits: [B, V], T: [B], mask True=keep
    if mask is None:
        mask = torch.isfinite(logits)
    s = logits.masked_fill(~mask, float('-inf'))
    vmax = torch.where(mask, s, torch.tensor(-float('inf'), device=s.device)).amax(-1, keepdim=True)
    x = (s - vmax) / T.unsqueeze(-1)
    ex = torch.where(mask, torch.exp(x), torch.zeros_like(s))
    Z = ex.sum(-1, keepdim=True).clamp_min(1e-40)
    p = ex / Z
    logp = torch.where(mask, torch.log(p.clamp_min(1e-40)), torch.zeros_like(p))
    H = -(p * logp).sum(-1)
    mu = (p * s.masked_fill(~mask, 0.0)).sum(-1, keepdim=True)
    var = (p * (s - mu).pow(2)).sum(-1)
    return H, var, p

def solve_T_for_entropy(logits, H_star, T0=None, eps=1e-3, Kmax=8,
                        Tmin=1e-2, Tmax=1e3, mask=None):
    B, V = logits.shape
    T = (torch.full((B,), 1.0, device=logits.device, dtype=logits.dtype)
         if T0 is None else T0.clamp(Tmin, Tmax))
    V_active = (mask.sum(-1) if mask is not None else torch.isfinite(logits).sum(-1))
    H_max = torch.log(V_active.float().clamp_min(1.0))
    H_star = H_star.clamp_min(1e-4)
    H_star = torch.minimum(H_star, H_max - 1e-4)

    lo = torch.full_like(T, Tmin)
    hi = torch.full_like(T, Tmax)
    best_T = T.clone()
    best_err = torch.full_like(T, float('inf'))

    for _ in range(Kmax):
        H, var, _ = entropy_and_moments(logits, T, mask)
        g = H - H_star
        err = g.abs()
        better = err < best_err
        best_T = torch.where(better, T, best_T)
        best_err = torch.minimum(err, best_err)
        if (err <= eps).all(): break

        # Bracket update using monotonicity of H(T)
        inc = g < 0  # H too low -> increase T
        lo = torch.where(inc, T, lo)
        hi = torch.where(~inc, T, hi)

        # Newton step with variance guard
        dH_dT = var / (T.pow(3).clamp_min(1e-20))
        newton = T - g / dH_dT.clamp_min(1e-20)
        invalid = (~torch.isfinite(newton)) | (newton < lo) | (newton > hi)
        T = torch.where(invalid, 0.5 * (lo + hi), newton).clamp(Tmin, Tmax)

    return torch.where(best_err <= eps, T, best_T)
```

Complexity: O(V) per iteration; empirical iterations/token: 2–3.

### 2.4 Entropy schedules
- Constant: H*(t)=h0.
- Linear ramp: H*(t)=h_start + (h_end−h_start)·min(t/T_ramp,1).
- Prompt-adaptive: Heuristics choose schedule by prompt intent (e.g., lower H* for QA; higher early H* for creative writing).
- Smoothness constraint: |H*(t)−H*(t−1)| ≤ Δ to avoid large jumps.

### 2.5 Interactions with truncation and penalties
- Exact guarantee: TED guarantees H(p(T*))=H* only when sampling from the full active vocabulary M (no truncation).
- Truncate-then-solve: Apply top-k/top-p or safety masks to define an effective active set M′; solve for T* on M′ for exact control over the truncated distribution.
- Solve-then-truncate: Solving on M and truncating afterward breaks the exact entropy target; realized entropy will be ≤ H*. Use only if smooth T trajectories are preferred over exact matching.
- Apply repetition penalties, logit biases, and mandatory EOS handling before solving so control reflects the effective sampling distribution.

## 3. Related Work
- Temperature, top-p (Holtzman et al., 2020) adjust uncertainty indirectly.
- Locally typical sampling (Meister et al., 2023) enforces token-level typicality but not global entropy.
- Mirostat (Basu et al., 2021) controls target surprisal via stochastic feedback; TED instead solves a deterministic root problem for distributional entropy.
- Temperature scaling (Guo et al., 2017) targets calibration offline, not per-step generation control.
- Entropy-regularized control and energy-based models provide background on Gibbs measures and entropy targets; to our knowledge, direct per-step entropy matching for LLM decoding has not been formalized and evaluated as here.

## 4. Experiments

### 4.1 Setup and protocol
Models (HuggingFace with revision IDs):
- eleutherai/pythia-410m-deduped (rev: 7f9e7e8)
- eleutherai/pythia-1.4b-deduped (rev: 5c1240b)
Inference: fp16, PyTorch 2.0.1, CUDA 11.8.

Datasets (versions and sample sizes):
- WritingPrompts (HP, 200 prompts; cap 200 tokens).
- XSum v1.1 (200 articles; cap 128 tokens).
- TruthfulQA v1.0 MC1 (817 questions; single-choice).

Decoding stop: EOS or token cap. Batch size 1 unless stated.

Baselines (grid-tuned on validation splits with equal budget):
- Temperature T ∈ {0.7, 1.0, 1.3}.
- Nucleus p ∈ {0.9, 0.95} (T=1).
- Typical sampling τ_typ ∈ {0.8, 1.0, 1.2}.
- Mirostat v2 τ ∈ {3.0, 5.0}, η ∈ {0.05, 0.1}.

TED configs:
- TED-Const h0 ∈ {2.5, 3.0, 3.5}.
- TED-Ramp h_start ∈ {3.5, 4.0}, h_end ∈ {2.2, 2.8}, T_ramp ∈ {32, 64}.
- TED-Adapt: heuristic classifier (regex intent + prompt length) chooses TED-Const for QA and TED-Ramp for open-ended; details in code.

Metrics (mean ± 95% CI over 3 seeds):
- Repetition: fraction of repeated 4-grams (↓).
- Diversity: distinct-3 (↑).
- MAUVE for open-ended (↑).
- TruthfulQA MC1 accuracy (↑).
- Efficiency: tokens/sec (↑), avg iterations/token, extra softmax-equivalents/token.
- Entropy control: mean absolute entropy error |H_real − H*| (nats).

### 4.2 Main results
Summary (best baseline vs. best TED per setting):

| Model       | Task         | Method        | Repetition ↓      | Diversity ↑       | MAUVE ↑           | MC1 ↑             | Tokens/sec ↑      |
|-------------|--------------|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| Pythia-410M | Open-ended   | Typical (best)| 0.12 ± 0.02       | 0.45 ± 0.03       | 0.68 ± 0.04       | —                 | 85 ± 2            |
|             |              | TED-Ramp      | 0.09 ± 0.01       | 0.48 ± 0.02       | 0.70 ± 0.03       | —                 | 82 ± 2 (−3.5%)    |
| Pythia-410M | TruthfulQA   | Mirostat (best)| —                 | —                 | —                 | 0.42 ± 0.03       | 88 ± 1            |
|             |              | TED-Const     | —                 | —                 | —                 | 0.44 ± 0.02       | 85 ± 1 (−3.4%)    |
| Pythia-1.4B | Open-ended   | Nucleus (best)| 0.10 ± 0.01       | 0.50 ± 0.02       | 0.72 ± 0.03       | —                 | 62 ± 3            |
|             |              | TED-Adapt     | 0.08 ± 0.01       | 0.53 ± 0.02       | 0.74 ± 0.02       | —                 | 60 ± 2 (−3.2%)    |
| Pythia-1.4B | TruthfulQA   | Typical (best)| —                 | —                 | —                 | 0.51 ± 0.02       | 65 ± 2            |
|             |              | TED-Const     | —                 | —                 | —                 | 0.54 ± 0.02       | 62 ± 2 (−4.6%)    |

Entropy tracking and efficiency:
- No truncation: mean |H_real − H*| < 1e−3 nats; 2.7 ± 0.9 iterations/token; 2.1 ± 0.4 extra softmax-equivalents/token (vs. one for baseline sampling).
- Truncate-then-solve (top-p=0.95): |H_real − H*| < 2e−3 nats with staircase T trajectories at support changes.
- Solve-then-truncate (top-p=0.95): realized entropy is below H* by 0.05–0.15 nats early; we do not recommend this mode when exact control is required.

### 4.3 Microbenchmarks
Methodology: A100-40GB, PyTorch 2.0.1, batch size ∈ {1, 4, 16, 32}, sequence length 128, vocab 50k, fused attention kernels. We report mean tokens/sec and GPU time/token over 3 × 1k-token runs with warm cache.

Results:
- Overhead vs. temperature sampling: +2.0% (B=32) to +5.0% (B=1) latency on Pythia-1.4B; similar on 410M.
- Bottleneck: extra softmax/moment passes; arithmetic intensity is low relative to attention/MLP compute, so overhead is modest on GPU.
- CPU inference (single-thread MKL): +12–18% (not a target use case).

### 4.4 Ablations
- Solver: Safeguarded Newton converges in fewer iterations than pure bisection with identical accuracy; fallback triggers <0.5% steps on near-degenerate logits.
- Schedules: Ramps reduce early repetition without harming coherence; overly aggressive ramps (ΔH*>0.5 per token) degrade fluency.
- Penalties and masking: With heavy masking, V_active shrinks, reducing achievable H*; clamping avoids infeasible targets. Degenerate-equal logits occur <1% steps; default T=1 is safe.
- Truncation order: Truncate-then-solve is preferred when exact matching is required; solve-then-truncate offers smoother T but incurs predictable entropy deficit.

## 5. Discussion and Limitations
- Entropy vs. truthfulness: Lower entropy can reduce overt hallucinations but does not ensure factual correctness.
- Target selection: H* is task- and model-dependent; learning schedules from feedback or reinforcement is promising.
- Efficiency: Overhead is small on GPUs but noticeable on CPUs or with huge vocabularies without fused kernels.
- Safety filters: Exact control applies to the effective active set after safety masking and truncation.

## 6. Reproducibility
We release:
- Code: MIT-licensed PyTorch implementation with batched TED and kernels; repo: https://github.com/ted-decoding/ted (tag: v0.1.0, commit: 3a7c1d2).
- Models: HF revisions specified above; automatic download via scripts with revision pinning.
- Data: Prompt lists, dataset versions, and preprocessing scripts; eval harness for MAUVE, repetition, distinct-n; TruthfulQA MC1 protocol.
- Configs: Exact hyperparameters, seeds {7,13,29}, environment lockfiles (conda YAML, CUDA/PyTorch versions).
- Logs: Per-step H*, realized H, T, iterations, profiler traces.

## 7. Conclusion
TED provides deterministic, per-step control of next-token entropy with modest overhead and consistent quality gains over strong baselines on small open models. Results support entropy as a practical control knob for LLM decoding. Future work: learned schedules, larger models, and integration with safety filters.

## References
- Holtzman et al. (2020). The Curious Case of Neural Text Degeneration.
- Meister et al. (2023). Locally Typical Sampling.
- Basu et al. (2021). Mirostat: A Memory-Efficient Algorithm for Controlling Perplexity.
- Guo et al. (2017). On Calibration of Modern Neural Networks.

## Appendix A: Derivation of dH/dT
Let β=1/T and Z(β)=Σ_{i∈M} exp(β s_i). Then p_i(β)=exp(β s_i)/Z(β), H(β)=log Z(β)−β E_pβ[s]. Using d/dβ log Z=E[s] and d/dβ E_pβ[s]=Var_pβ(s):
dH/dβ = E[s] − E[s] − β Var_pβ(s) = −β Var_pβ(s) ≤ 0.
By the chain rule, dH/dT = (−β Var_pβ(s))·(−1/T^2) = Var_pT(s)/T^3 ≥ 0, with equality only when p is one-hot (T→0) or all active logits are equal.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
