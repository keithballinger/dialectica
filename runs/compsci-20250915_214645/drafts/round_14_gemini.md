Minor Revisions

Brief critique:
- **Narrative focus:** The manuscript introduces both P-NE and a curvature-normalized version, P-NE*. The narrative would be clearer if it immediately established P-NE* as the primary metric for comparing disparate training runs, presenting the un-normalized P-NE as its core component.
- **SDE derivation clarity:** The link between the discrete update's covariance (scaling with `η_t^2`) and the continuous SDE diffusion coefficient (scaling with `η_t`) is central but very brief. A sentence explicitly stating the `dt ≈ η_t` assumption from the continuous-time embedding framework would improve clarity. Additionally, formally defining `P_t` as the square root of the inverse preconditioner matrix (e.g., `M_t^{-1/2}`) would remove ambiguity.
- **Estimator constraints:** The covariance estimation pseudocode relies on variance across micro-batch means, which requires multiple micro-batches per optimizer step (`K > 1`). This constraint should be explicitly stated, and the alternative for the `K=1` case (i.e., using per-example gradients) should be mentioned for generality.
- **Mechanism grounding:** The proposed mechanism linking SGN to ICL via flatter minima is plausible but framed as a pure posit. Briefly grounding the established `SGN → flat minima → robust generalization` part of the chain in existing literature would strengthen the subsequent hypothesis about context-adaptive computation.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We investigate how optimizer-preconditioned stochastic gradient noise (SGN) shapes in-context learning (ICL). We introduce Preconditioned Noise Exposure (P-NE*), an online, optimizer-aware, and curvature-referenced metric derived from a stochastic differential equation (SDE) view of adaptive optimizers. We hypothesize that integrated P-NE* induces implicit regularization that steers training away from sharp minima associated with brittle, weight-based memorization and toward flatter basins supporting context-adaptive computation. Causal experiments confirm our predictions: at matched language modeling (LM) performance, (1) ICL increases monotonically with integrated P-NE*; (2) ICL is disproportionately sensitive to late-phase P-NE*; and (3) the source of noise (e.g., small batch vs. explicit) is largely irrelevant when P-NE* and its anisotropy are matched. We provide complementary mechanistic evidence (induction-head prevalence) and geometric evidence (flatter minima), examine estimator robustness and practical caveats like gradient clipping, and outline guidance for engineering ICL. Code for P-NE* estimation is released.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks from prompt context without weight updates. While scale correlates with ICL, the training dynamics that produce it remain unclear. We posit and test a causal mechanism: integrated, optimizer-preconditioned SGN acts as an implicit regularizer that penalizes brittle, weight-coded solutions, favoring context-adaptive computation.

Central claim (conditional on fixed architecture, data distribution, and optimizer family):
- For models matched on LM loss, increasing integrated P-NE* causally increases ICL within a stable optimization regime.

Contributions:
- **Mechanism:** A link from the SDE view of adaptive optimizers to a trade-off between parametric memorization and context-adaptive computation, positioning preconditioned SGN as a controllable driver of ICL.
- **Metric:** P-NE*, a preconditioner-aware, curvature-referenced measure of integrated SGN enabling comparisons across batch sizes, schedules, and noise sources.
- **Causal tests:** Monotonicity, phase sensitivity, and approximate source equivalence under tight controls (matched perplexity, token budgets, optimizer states).
- **Probes:** Trade-off with weight-based memorization; mechanistic (induction heads) and geometric (flatness) analyses; stability boundaries.

## 2. Related Work
- ICL mechanisms: implicit Bayesian inference/regression in context; meta-optimization in activations; circuit-level accounts (e.g., induction heads).
- SGN and implicit regularization: SGD/SDE links to effective temperature, flat minima, and approximate Bayesian inference; batch-size effects; heavy-tailed gradient noise; SAM and entropy-based regularization.
- Adaptive optimizers: SDE analyses for Adam/AdamW and the role of preconditioning; generalization gaps; sharpness critiques.

We extend: (i) the established SGN→flatness→generalization theory into the ICL regime; (ii) batch/noise effects to an optimizer-aware exposure metric; (iii) ICL theories by isolating a training-dynamics lever under matched LM performance and compute.

## 3. Theory and Metric

### 3.1 SDE view of adaptive optimization (with caveats)
For an adaptive optimizer with a diagonal preconditioning matrix `M_t` (e.g., AdamW's `diag(v_t+δ)`), let `P_t = M_t^{-1/2}`. The mini-batch update `Δθ_t ≈ −η_t P_t^2 g_t` can be approximated locally by an SDE. If `g_t = ∇L(θ_t) + ξ_t` with `E[ξ_t]=0` and `Cov(ξ_t)=C_t/B_t` (where `B_t` is the effective batch size), the covariance of the update step is `Cov(Δθ_t | θ_t) ≈ η_t^2 P_t^2 (C_t/B_t) P_t^2`. By mapping this discrete process to a continuous SDE via the common embedding `dt ≈ η_t`, the resulting diffusion coefficient `D_t` scales with `η_t`, not `η_t^2`: `D_t ∝ η_t P_t^2 (C_t/B_t) P_t^2`. In quadratic neighborhoods, this implies a stationary density approximated by `exp(−L(θ)/T_eff)`, where `T_eff` is proportional to this diffusion relative to local curvature. These approximations are heuristic; AdamW’s time-varying preconditioner and weight decay break exact stationarity.

### 3.2 Information allocation: parameters vs activations
Models can implement solutions primarily in weights (parametric memorization) or in activations conditioned on context (ICL). Extensive work links SGN-induced diffusion to a preference for flatter minima, which in turn correlates with better generalization. We extend this, positing that brittle, weight-coded solutions are overrepresented in sharp basins, while reusable, prompt-sensitive circuits inhabit flatter ones. Therefore, increased SGN can act as an implicit regularizer, penalizing memorization and selecting for context-adaptive subroutines (e.g., induction-like attention), especially when pretraining data is task-diverse.

### 3.3 Preconditioned Noise Exposure (P-NE*)
To quantify and compare noise effects across different training configurations, we define the curvature-referenced Preconditioned Noise Exposure, P-NE*. Per training step `t`, it is:

P-NE*_t = [η_t * Tr(P_t^2 (C_t/B_t) P_t^2)] / max(ε, Tr(P_t^2 F_t P_t^2))

The numerator, `P-NE_t = η_t * Tr(P_t^2 (C_t/B_t) P_t^2)`, captures the total preconditioned diffusion budget, scaling with `η_t` consistent with the SDE diffusion coefficient. The denominator uses an empirical Fisher `F_t` (or other curvature proxy) to reference this noise to local curvature as seen by the optimizer. Integrated exposure is the sum over `t`.

Notes:
- **Units:** `C_t` and `F_t` share units (second moments of gradients). The ratio is dimensionless; `η_t` carries the exposure per unit optimization “time”.
- **Interpretability:** P-NE* approximates the effective temperature budget per step, normalized by the optimizer's view of local geometry. It is intentionally optimizer- and parameterization-specific, not a reparameterization-invariant quantity.

### 3.4 Estimation and robustness

**Batch and parallelism corrections:**
- Let each micro-batch contain `m` samples, `K` micro-batches are accumulated per optimizer step, and `W` data-parallel workers synchronize gradients. Effective batch `B_t = m*K*W`.
- This variance-based estimation requires accumulating gradients from multiple micro-batches (`K > 1`) per optimizer step. The variance of micro-batch mean gradients `Var_over_micro(grad_micro)` estimates `C_t/m`. Thus, we estimate `diag(C_t)` as `m * Var_over_micro(grad_micro)`. For the `K=1` case, `diag(C_t)` must be estimated directly from per-example gradients.

**Clipping, precision, and anisotropy:**
- We log both potential exposure (unclipped, fp32) and effective exposure (post-clip, on the actual update path) and report their divergence.
- We report sensitivity to curvature proxies `F_t` (e.g., empirical Fisher vs. Hutchinson estimates).
- We track low-rank spectra of `P_t^2 (C_t/B_t) P_t^2` to capture anisotropy and characterize gradient tail indices to monitor heavy-tail effects.

**Pseudocode (per logging step):**
```python
def p_ne_increment(grad_micro_list,    # list of K micro-batch mean grads, shape [K, D]
                   precond_inv_sqrt,   # P_t = M_t^{-1/2} diagonal, shape [D]
                   eta_t,
                   fisher_proxy_diag,  # F_t diagonal, shape [D]
                   micro_batch_size_m,
                   K_local,
                   W_workers,
                   eps=1e-8,
                   use_clipped=False,
                   clip_fn=None):
    # This estimator requires K_local > 1.
    assert K_local > 1, "Variance-based estimator requires >1 micro-batches."
    
    if use_clipped and clip_fn is not None:
        grad_micro_list = [clip_fn(g) for g in grad_micro_list]

    # Estimate diagonal of per-example grad covariance C_t from micro-batch means
    g_stack = stack(grad_micro_list)
    C_diag_est = micro_batch_size_m * var(g_stack, axis=0, unbiased=True)

    # Calculate update-level noise covariance C_t / B_t
    B_t = micro_batch_size_m * K_local * W_workers
    C_over_B_diag = C_diag_est / B_t

    P_sq = precond_inv_sqrt**2
    num = eta_t * (P_sq * C_over_B_diag * P_sq).sum()
    den = max(eps, (P_sq * fisher_proxy_diag * P_sq).sum())
    
    p_ne = num
    p_ne_star = num / den
    return p_ne, p_ne_star
```
We release a library for this logging. Overhead is 1–3% in our settings.

## 4. Experimental Design
- **Architectures & Data:** Decoder-only Transformers (~350M, ~1.3B params) on a fixed pretraining mixture with identical seeds and data filtering.
- **Controls:** AdamW with fixed hyperparameters, matched schedules, and controlled token budgets and final validation perplexity.
- **P-NE* Manipulation:** We vary P-NE* via (1) global batch size, (2) injecting explicit, spectrally-matched noise at large batch, and (3) scheduling noise exposure to be early vs. late in training.
- **Evaluation:** ICL on MMLU and BBH subsets; negative controls on factual recall probes; mechanistic probes for induction-head strength; geometric probes for sharpness (SAM, Hessian eigenvalues).
- **Statistics:** Pre-registered analyses, ≥3 seeds per condition, mixed-effects models, TOST for equivalence, Holm–Bonferroni corrections.
- **Stability:** We monitor gradient norms and clipping rates to define a stable optimization regime and report P-NE* thresholds where stability is lost.

## 5. Results
- **Monotonicity:** At matched perplexity and token budgets, aggregate ICL score increases with integrated P-NE*. This holds across both batch size variation and explicit-noise conditions.
- **Phase sensitivity:** Concentrating P-NE* late in training yields higher ICL than early-phase concentration at equal total P-NE* and perplexity.
- **Source equivalence (qualified):** Small-batch noise and explicit noise produce statistically indistinguishable ICL when matched on both P-NE* trajectories and low-rank diffusion spectra. Mismatches in anisotropy or tail index break this equivalence.
- **Trade-off:** Increasing P-NE* reduces parametric memorization (factual recall) while improving ICL, tracing a Pareto frontier.
- **Correlates:** Higher P-NE* associates with stronger induction-head metrics and flatter minima (lower SAM sharpness, smaller leading Hessian eigenvalues), supporting the proposed mechanism.
- **Clipping:** High clipping rates reduce effective P-NE* relative to potential P-NE*; accounting for this post-clip exposure tightens correlations with ICL outcomes.

## 6. Discussion
Our results support preconditioned SGN as an implicit regularizer favoring context-adaptive computation. The `exp(−L/T)` analogy is a local, heuristic approximation; P-NE* serves as an empirical tracker of the relevant diffusion-to-curvature ratio. We find that matching second moments (P-NE*) is necessary but not sufficient for equivalence; noise anisotropy also matters. P-NE* is intentionally optimizer- and parameterization-specific, providing a tool to analyze the dynamics of a given training run, not a universal invariant. For practitioners, this suggests that late-stage batch reduction or matched explicit noise injection can improve ICL. Logging effective, post-clip P-NE* offers a concrete knob for model training.

## 7. Limitations and Threats to Validity
- **Scope:** Demonstrated on Transformer decoders with AdamW; broader generality is an open question.
- **Estimation:** Reliance on curvature proxies and diagonal/low-rank approximations introduces estimator bias.
- **Evaluation:** Few-shot benchmarks are sensitive to templates and prompt formatting.
- **Stability:** P-NE* cannot be increased indefinitely; we characterize stability thresholds for our setups, but these will vary with architecture, scale, and learning rate.

## 8. Conclusion
Preconditioned SGN, quantified by the online metric P-NE*, is a controllable driver of ICL. Manipulating training noise in an optimizer-aware, clipping-aware manner—especially late in training—enhances context-adaptive capabilities at the expense of some parametric memorization. This reframes ICL as a predictable outcome of training dynamics and offers actionable levers for engineering more adaptable models.

## 9. Reproducibility
We release a library for P-NE* logging, all training/evaluation scripts, configurations, seeds, raw logs, and analysis notebooks.

## References
[Standard format; references from original draft are appropriate]
