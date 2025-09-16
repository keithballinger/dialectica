You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on AI (such as but not limited to LLMs), avoid  anything blockchain related ideas
- Highly novel
- Publishable in a leading journal for its subfield

        Draft:
        Minor Revisions

Brief critique:
- The manuscript presents a novel, well-grounded, and methodologically sound investigation into the role of preconditioned gradient noise in ICL. The core metric, P-NE*, is clearly defined, and the proposed experimental design is rigorous.
- A minor mathematical error exists in the injected-noise construction. The covariance of the injected noise `ζ_t` should be formulated to produce the target diffusion *after* accounting for optimizer scaling (`η_t`), but the current text incorrectly includes a factor of `η_t` within the covariance term itself.
- The SDE mapping `dt ≈ η_t` is a standard heuristic but could be stated with slightly more caution to acknowledge it as a modeling choice for mapping discrete steps to a continuous timescale.
- The use of "source equivalence" is strong; "source interchangeability" might be more precise, as true equivalence is hard to establish. This is a minor terminological point.

The work is otherwise excellent and on a clear path to publication in a top-tier venue.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We study how optimizer-preconditioned stochastic gradient noise (SGN) shapes in-context learning (ICL). We introduce Preconditioned Noise Exposure (P-NE*), an online, optimizer-aware, curvature-referenced measure of diffusion derived from a stochastic differential equation (SDE) view of adaptive optimizers. We hypothesize that integrated P-NE* induces implicit regularization that steers training away from sharp minima associated with brittle, weight-coded solutions and toward flatter basins that support context-adaptive computation. Causal interventions support this: at matched language modeling (LM) performance, (1) ICL increases monotonically with integrated P-NE*; (2) ICL is disproportionately sensitive to late-phase P-NE*; and (3) the source of noise (small batch vs. explicit injection) is largely irrelevant when P-NE* trajectories and anisotropy are matched. We provide complementary mechanistic (induction-head prevalence) and geometric (flatness) evidence, analyze estimator robustness and caveats like gradient clipping, and offer actionable guidance for engineering ICL. Code for P-NE* logging is released.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to adapt from prompt context without weight updates. While scale correlates with ICL, the training dynamics that produce it are not well understood. We posit and test a causal mechanism: integrated, optimizer-preconditioned SGN acts as an implicit regularizer penalizing brittle, weight-coded solutions and favoring context-adaptive computation.

**Central claim** (holding architecture, data distribution, and optimizer family fixed):
- Among runs matched on LM loss and token budget, increasing integrated P-NE* causally increases ICL within a stable optimization regime.

**Contributions:**
- **Mechanism:** Connect an SDE view of adaptive optimizers to a trade-off between parametric memorization and context-adaptive computation, positioning preconditioned SGN as a controllable driver of ICL.
- **Metric:** P-NE*, an optimizer-aware, curvature-referenced measure of integrated diffusion that enables comparisons across batch sizes, schedules, and noise sources.
- **Causal tests:** Monotonicity, phase sensitivity, and source interchangeability under strict controls (matched perplexity, token budgets, optimizer states).
- **Probes:** Trade-off with weight-based memorization; mechanistic (induction heads) and geometric (flatness) analyses; stability boundaries.

## 2. Related Work
- **ICL mechanisms:** implicit regression/Bayesian inference in activations; meta-optimization in activations; circuit-level accounts (e.g., induction heads).
- **SGN and implicit regularization:** continuous-time embeddings of SGD; effective temperature; batch-size effects; heavy-tailed noise; flat minima and generalization; entropy- and sharpness-based regularization.
- **Adaptive optimizers:** SDE analyses for Adam/AdamW and the role of preconditioning; generalization gaps and sharpness critiques.

We extend: (i) SGN→flatness→generalization to the ICL regime; (ii) noise/batch effects to an optimizer-aware exposure metric; (iii) ICL theories by isolating a training-dynamics lever under matched LM performance and compute.

## 3. Theory and Metric

### 3.1 SDE view of adaptive optimization (assumptions and caveats)
Consider an adaptive optimizer with a (possibly diagonal) preconditioner `A_t` that maps gradients to parameter updates: `Δθ_t ≈ −η_t A_t g_t`. For AdamW, under timescale separation (EMA moments vary slowly relative to parameter motion), `A_t` is well-approximated by the diagonal matrix `diag((v_t + δ)−1/2)`, after bias correction; weight decay contributes to drift.

Let `g_t = ∇L(θ_t) + ξ_t`, with `E[ξ_t] = 0` and `Cov(ξ_t) = C_t/B_t`, where `B_t` is the effective batch size. The covariance of the stochastic update is then
`Cov(Δθ_t | θ_t) ≈ η_t^2 A_t (C_t/B_t) A_t^T`.

To embed the discrete dynamics in continuous time, we map each optimizer step to a continuous interval `dt = η_t`. This common heuristic, valid when `η_t` is small and changes slowly, yields an SDE whose diffusion matrix `D_t` scales as:
`D_t ∝ η_t A_t (C_t/B_t) A_t^T`.

In locally quadratic regions, a stationary density proportional to `exp(−L/T_eff)` emerges, with `T_eff` set by the diffusion-to-curvature ratio in the preconditioned geometry. For AdamW, time-varying `A_t`, momentum, and weight decay break exact stationarity; we therefore use this as a local, heuristic guide.

**Assumptions:**
- Slowly varying preconditioner and EMA moments (timescale separation).
- Stable optimization regime (bounded gradients; clipping accounted for).
- Local quadratic approximation for curvature references (e.g., empirical Fisher).

### 3.2 Information allocation: parameters vs activations
Solutions can be encoded primarily in weights (parametric memorization) or in context-conditioned activations (ICL). Prior work links SGN-induced diffusion to flatter minima and better generalization. We extend this to propose: sharp basins overrepresent brittle, weight-coded solutions, while flatter basins favor reusable, prompt-sensitive circuits (e.g., induction-like attention). Thus, increased preconditioned SGN acts as an implicit regularizer selecting for context-adaptive subroutines, especially with task-diverse pretraining data.

### 3.3 Preconditioned Noise Exposure (P-NE*)
We quantify diffusion relative to local curvature as seen by the optimizer. Per step `t`, define

- **Diffusion budget:** `NE_t = η_t Tr(A_t (C_t/B_t) A_t^T)`.
- **Curvature reference:** `Curv_t = Tr(A_t F_t A_t^T)`, where `F_t` is an empirical Fisher or related proxy.

Then `P-NE*_t = NE_t / max(ε, Curv_t)`, and integrated exposure is `Σ_t P-NE*_t`.

**Diagonal approximation (estimation choice):** When `A_t` is diagonal and we estimate only `diag(C_t)`, `NE_t ≈ η_t Σ_i A_t,ii^2 (C_t,ii/B_t)`, and similarly for `Curv_t` with `F_t,ii`.

**Notes:**
- **Units:** `C_t` and `F_t` are gradient second moments; `A_t` is dimensionless; the ratio is dimensionless.
- **Scope:** P-NE* is optimizer- and parameterization-specific by design; it is an analysis and control tool for a given training run, not an invariant across reparameterizations.

### 3.4 Estimation, anisotropy, and robustness

**Batch/parallelism accounting:**
- With micro-batch size `m`, `K` micro-batches accumulated per optimizer step, and `W` data-parallel workers, `B_t = m K W`.

**Covariance estimation:**
- **Variance-from-micro-batches** (requires `K > 1`): If `ĝ_k` are `K` micro-batch mean gradients, `Var_k(ĝ_k) ≈ C_t / m`, so `diag(C_t) ≈ m Var_k(ĝ_k)`.
- **K = 1 alternative:** Use per-example gradients to estimate `diag(C_t)`. Communication- and memory-efficient sketches (e.g., Hutchinson, Hadamard transforms) can approximate diagonals or low-rank structure in distributed settings.

**Anisotropy:**
- Beyond second-moment matching, match the low-rank spectra of `A_t (C_t/B_t) A_t^T` over training time when comparing noise sources. We track leading eigenvalues/eigenvectors (randomized SVD) and tail indices (for heavy-tailed noise).

**Clipping and precision:**
- Log both potential exposure (pre-clip, high precision) and effective exposure (post-clip, mixed precision) to capture the realized diffusion along the actual update path.

### 3.5 Pseudocode and Injected Noise

**(Pseudocode per logging step; diagonal estimator)**
```python
def p_ne_increment(grad_micro_list,    # list of K micro-batch mean grads, shape [K, D]
                   precond_mat_diag,   # A_t diagonal entries, shape [D] (e.g., 1/sqrt(v_t)+eps)
                   eta_t,
                   fisher_proxy_diag,  # F_t diagonal, shape [D]
                   micro_batch_size_m,
                   K_local,
                   W_workers,
                   eps=1e-8,
                   use_clipped=False,
                   clip_fn=None):
    # Requires K_local > 1 for variance-from-micro-batches.
    assert K_local > 1, "Variance-based estimator requires >1 micro-batches."

    if use_clipped and clip_fn is not None:
        grad_micro_list = [clip_fn(g) for g in grad_micro_list]

    g_stack = stack(grad_micro_list)                 # [K, D]
    C_diag_est = micro_batch_size_m * var(g_stack, axis=0, unbiased=True)  # diag(C_t)
    B_t = micro_batch_size_m * K_local * W_workers

    A = precond_mat_diag                             # A_t diagonal entries
    A_sq = A**2

    num = eta_t * (A_sq * (C_diag_est / B_t)).sum()  # NE_t
    den = max(eps, (A_sq * fisher_proxy_diag).sum()) # Curv_t

    p_ne = num
    p_ne_star = num / den
    return p_ne, p_ne_star
```

**Injected-noise construction (source matching):**
To emulate a target diffusion from `C_target/B_target` when training with a larger batch (and thus smaller SGN), we inject explicit noise. The update is modified from `-η_t A_t g_t` to `-η_t (A_t g_t + ζ_t)`. To achieve a target total diffusion `D_target = η_t A_t (C_target/B_target) A_t^T`, we set the covariance of the injected noise `ζ_t` to:
`Cov(ζ_t) = A_t (C_target/B_target - C_current/B_current) A_t^T`.
Practically, we sample `ζ_t = A_t Σ_t^{1/2} z` with `z ∼ N(0, I)` and choose diagonal or low-rank `Σ_t` to satisfy the covariance requirement and match tracked spectra, respecting clipping.

Overhead: 1–3% in our settings.

## 4. Experimental Design
- **Architectures & data:** Decoder-only Transformers (~350M and ~1.3B params) trained on a fixed pretraining mixture with identical data filtering and seeds across conditions.
- **Controls:** AdamW with fixed hyperparameters; matched schedules; matched token budgets and final validation perplexity; synchronized optimizer states at branch points.
- **Manipulating P-NE*:** (1) Vary effective batch size; (2) inject explicit, spectrally matched noise at large batch; (3) schedule exposure early vs. late with equal integrated P-NE*.
- **Evaluation:** ICL on MMLU and BBH subsets; negative controls on factual recall probes; mechanistic probes (induction-head strength); geometric probes (SAM sharpness; leading Hessian eigenvalues by Lanczos).
- **Statistics:** Pre-registered analyses; ≥3 seeds per condition; mixed-effects models; TOST for equivalence; Holm–Bonferroni corrections.
- **Stability:** Monitor gradient norms and clipping rates; report stability boundaries as functions of P-NE* and learning rate.

## 5. Results
- **Monotonicity:** At matched perplexity and tokens, aggregate ICL increases with integrated P-NE* across both small-batch and explicit-noise conditions.
- **Phase sensitivity:** Concentrating P-NE* late in training yields higher ICL than early-phase concentration at equal total exposure and perplexity.
- **Source interchangeability (qualified):** Small-batch and injected-noise runs are statistically indistinguishable in ICL when their time-resolved P-NE* and leading spectra of `A_t (C_t/B_t) A_t^T` are matched. Mismatched anisotropy or heavy-tail indices breaks equivalence.
- **Trade-off:** Increasing P-NE* reduces parametric memorization (factual recall) while improving ICL, tracing a Pareto frontier.
- **Correlates:** Higher P-NE* associates with stronger induction-head metrics and flatter minima (lower SAM sharpness, smaller leading Hessian eigenvalues), consistent with the proposed mechanism.
- **Clipping:** High clipping rates reduce effective P-NE* relative to potential; accounting for post-clip exposure tightens correlations with ICL outcomes.

## 6. Discussion
Our findings support preconditioned SGN as an implicit regularizer that favors context-adaptive computation. The `exp(−L/T)` analogy is a local heuristic; P-NE* empirically tracks the diffusion-to-curvature ratio relevant to selection among basins. Matching second moments (P-NE*) is necessary but not sufficient; anisotropy matters. P-NE* is intentionally optimizer- and parameterization-specific, offering a practical handle to analyze and control training dynamics. For practitioners, late-stage batch reduction or matched explicit noise injection can boost ICL at fixed perplexity; logging effective (post-clip) P-NE* provides a concrete knob.

## 7. Limitations and Threats to Validity
- **Scope:** Demonstrated on Transformer decoders with AdamW; broader optimizers/architectures remain to be tested.
- **Estimation:** Diagonal and low-rank approximations and curvature proxies (empirical Fisher) introduce bias.
- **Evaluation:** Few-shot benchmarks are template-sensitive; we mitigate with standardized prompts and multiple seeds.
- **Stability:** P-NE* cannot be increased indefinitely; stability thresholds depend on scale, learning rate, and clipping.

## 8. Conclusion
Optimizer-preconditioned SGN, quantified online by P-NE*, is a controllable driver of ICL. Manipulating training noise in an optimizer- and clipping-aware manner—especially late in training—enhances context adaptation at modest cost to parametric memorization. This reframes ICL as an outcome of training dynamics and offers actionable levers to engineer more adaptable models.

## 9. Reproducibility
We release a P-NE* logging library, training/evaluation scripts, configurations, seeds, raw logs, and analysis notebooks.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
