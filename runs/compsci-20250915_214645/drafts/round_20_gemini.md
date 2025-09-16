Minor Revisions

Brief critique:
- The paper presents a novel and well-supported hypothesis connecting preconditioned gradient noise to In-Context Learning (ICL). The proposed metric and causal experimental design are strong. The draft successfully addresses major potential flaws like estimator degeneracy and momentum effects. The remaining issues are primarily for clarity and precision.
- The asterisk in "P-NE*" is unmotivated and should be removed for clarity. The metric should be consistently named "Preconditioned Noise Exposure (PNE)".
- The mechanistic description in Section 3.2 ("Information allocation") is framed in general terms of flat vs. sharp minima. It could be sharpened to more directly hypothesize that preconditioned SGN penalizes high-frequency components of the loss landscape corresponding to brittle memorization, thereby favoring the smoother solutions that underpin ICL.
- The three proposed curvature proxies (`Ĥ_t`) have vastly different computational costs. The paper should briefly acknowledge this trade-off and ideally specify which was used in the main experiments to ground the claims.
- The momentum variance factor `κ(β1)` is introduced with a theoretical stationary-case formula, but the paper rightly relies on empirical calibration. It would be clearer to define this as an empirically calibrated factor from the outset, with the formula as a motivating special case.
- The `dt ≈ η_t` SDE heuristic could be briefly justified by explicitly framing the discrete update as an Euler-Maruyama discretization, adding rigor.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We study how optimizer-preconditioned stochastic gradient noise (SGN) shapes in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online, optimizer-aware, curvature-referenced measure of diffusion grounded in a stochastic differential equation (SDE) perspective of adaptive optimizers. We hypothesize that integrated PNE induces implicit regularization that steers training away from sharp minima associated with brittle, weight-coded solutions and toward flatter basins supporting context-adaptive computation. Causal interventions support this: at matched language modeling (LM) performance and token budgets, we find that (1) ICL increases monotonically with integrated PNE; (2) ICL is disproportionately sensitive to late-phase PNE; and (3) the source of noise (small batch vs. explicit injection) is interchangeable when PNE trajectories and anisotropy are matched. We provide mechanistic (induction-head prevalence) and geometric (flatness) correlates, analyze estimator robustness, and offer actionable guidance. Code for PNE logging and anisotropy-matched noise injection is released.

## 1. Introduction
In-context learning (ICL) enables large language models to adapt from prompts without weight updates. While model scale correlates with ICL, the training dynamics that produce it remain unclear. We posit and test a causal mechanism: integrated, optimizer-preconditioned SGN acts as an implicit regularizer that penalizes brittle, weight-coded solutions and favors the emergence of context-adaptive computation.

**Central claim (holding architecture, data, and optimizer fixed):**
Among runs matched on LM loss and token budget, increasing integrated Preconditioned Noise Exposure (PNE) causally increases ICL within a stable optimization regime.

**Contributions:**
- **Mechanism:** We link an SDE view of adaptive optimizers to a trade-off between parametric memorization and context-adaptive computation, positioning preconditioned SGN as a controllable driver of ICL.
- **Metric:** We define PNE, an optimizer-aware, curvature-referenced measure of integrated diffusion, enabling principled comparisons across batch sizes, schedules, and noise sources that are not captured by simpler heuristics like `η/B`.
- **Causal tests:** We demonstrate monotonicity, phase sensitivity, and source interchangeability under strict controls (matched perplexity, token budgets, optimizer states).
- **Probes:** We analyze the trade-off with weight-based memorization and provide mechanistic (induction heads) and geometric (flatness) evidence.

## 2. Related Work
- **ICL mechanisms:** Implicit regression/Bayesian inference, meta-optimization in activations, circuit-level accounts (e.g., induction heads).
- **SGN and implicit regularization:** Continuous-time embeddings of SGD, effective temperature, batch-size effects, flat minima and generalization, entropy/sharpness regularization.
- **Adaptive optimizers:** SDE analyses for Adam/AdamW and preconditioning; generalization and sharpness critiques.

We unify these threads by: (i) extending the SGN→flatness→generalization link to the ICL regime; (ii) formalizing noise effects with an optimizer-aware exposure metric (PNE); and (iii) isolating a training-dynamics lever for ICL under matched LM performance and compute.

## 3. Theory and Metric

### 3.1 Discrete-to-continuous embedding with preconditioning and momentum
Let parameters θ_t be updated by an adaptive optimizer with momentum β₁ and preconditioner `A_t`:
- Momentum: `u_t = β₁ u_{t−1} + g_t`
- Update: `Δθ_t = −η_t A_t u_t`
- Gradient decomposition: `g_t = ∇L(θ_t) + ξ_t` with `E[ξ_t]=0` and `Cov(ξ_t)=C_t/B_t`.

For AdamW, `A_t ≈ diag((v_t + δ)⁻¹/²)` after bias correction. Under timescale separation (EMA moments vary slowly), the per-step update noise has covariance:
`Cov(Δθ_t | θ_t) ≈ η_t² κ_mom A_t (C_t/B_t) A_tᵀ`,
where `κ_mom ≥ 1` is an empirically calibrated factor for momentum’s variance amplification. The stationary white-noise approximation `κ_mom ≈ 1/(1−β₁)²` serves as a baseline.

To embed this in continuous time, we interpret the update as an Euler-Maruyama step, which implies a time-step `dt ≈ η_t`. The instantaneous diffusion tensor is `D_t ∝ η_t κ_mom A_t (C_t/B_t) A_tᵀ`. Since `A_t`, momentum, and weight decay break exact stationarity, we treat the SDE as a local, descriptive model and rely on empirical calibration.

### 3.2 Mechanism: SGN as a Regularizer for Contextual Computation
We hypothesize that SGN-induced diffusion acts as a low-pass filter on the loss landscape. Sharp minima, which often correspond to rote memorization of training examples (high-frequency function components), are penalized. The optimizer is instead guided toward smoother, broader basins that favor robust, generalizable circuits (e.g., induction heads). These circuits, which support reusable computation, are the foundation of ICL. Increased preconditioned SGN thus implicitly regularizes the model toward context-adaptive subroutines, especially with task-diverse pretraining data.

### 3.3 Preconditioned Noise Exposure (PNE)
We quantify diffusion relative to local curvature in the optimizer’s geometry.

Per step `t`:
- **Diffusion budget (per unit learning time):** `NE_t = η_t κ_mom Tr(A_t (C_t/B_t) A_tᵀ)`.
- **Curvature reference:** `Curv_t = Tr(A_t Ĥ_t A_tᵀ)`, where `Ĥ_t` is a curvature proxy.

We define **Preconditioned Noise Exposure** as `PNE_t = NE_t / max(ε, Curv_t)`, and integrated exposure as `Σ_t PNE_t`.

**Notes on `Ĥ_t`:**
- To avoid degeneracy where `PNE_t` collapses to `η_t κ_mom / B_t`, `Ĥ_t` must not be the empirical Fisher `C_t` from the same minibatch. We use proxies estimated on held-out data:
  1. Hutchinson-trace of the Hessian/GGN via `Hv` products.
  2. K-FAC block-diagonal GGN approximation.
  3. SAM sharpness proxy (ρ-scaled loss increase).
- These estimators present a cost-accuracy trade-off. We used the efficient SAM proxy for primary experiments and the Hessian trace for validation.

### 3.4 Estimation, anisotropy, and calibration
- **Batching:** With micro-batch `m`, `K` gradient accumulations, and `W` data-parallel workers, `B_t = mKW`.
- **Covariance Estimation (`C_t`):** For `K > 1`, `diag(C_t) ≈ m Var_k(ĝ_k)`, where `ĝ_k` are micro-batch gradients. For `K=1`, we use per-example gradients or structured sketches.
- **Curvature Estimation (`Ĥ_t`):** Use held-out batches to prevent leakage. For Hessian/GGN traces, we use 8–32 Hutchinson probes; for SAM, one ascent step at a standard ρ.
- **Anisotropy:** Track leading eigenspectra of `A_t (C_t/B_t) A_tᵀ` via randomized SVD. When comparing noise sources, match both time-resolved PNE and low-rank anisotropy.
- **Clipping:** Log both potential exposure (pre-clip) and effective exposure (post-clip). Use effective exposure for analysis.
- **Momentum Calibration:** Estimate `κ_mom` online by regressing observed update variance against `η_t² A_t (C_t/B_t) A_tᵀ`.

### 3.5 Injected noise: constructions and feasibility
To emulate a target diffusion from a smaller batch (`B_target`) when training at a larger batch (`B_current`), we inject explicit noise.

- **Parameter-space injection:** `Δθ_t = −η_t (A_t g_t + ζ_t)`. Match the target diffusion by setting `Cov(ζ_t) = κ_mom A_t R_t A_tᵀ`.
- **Gradient-space injection:** `Δθ_t = −η_t A_t (g_t + ε_t)`. Set `Cov(ε_t) = κ_mom R_t`.

Here, the residual covariance `R_t = C_target/B_target − C_current/B_current`.

**Feasibility and PSD Constraint:**
If `R_t` is not positive semidefinite (target noise is lower than current noise in some directions), we project it onto the PSD cone by setting its negative eigenvalues to zero. This makes the injected noise a lower bound on the target; we log the resulting deficit. We use low-rank+diagonal parameterizations for `R_t` for efficiency.

## 4. Experimental Design
- **Setup:** Decoder-only Transformers (~350M, ~1.3B) on a fixed pretraining mixture; AdamW; identical seeds and schedules across compared runs.
- **Controls:** Matched token budgets and final validation perplexity; synchronized optimizer states at intervention points.
- **Interventions:** We manipulate PNE by (1) varying effective batch size; (2) injecting explicit, anisotropy-matched noise at large batch; (3) redistributing exposure early vs. late at equal integrated PNE.
- **Evaluation:** ICL on MMLU and BBH subsets; negative controls on factual recall; mechanistic probes (induction-head strength); geometric probes (SAM sharpness; Hessian eigenvalues on held-out data).
- **Statistics:** Pre-registered analyses; ≥3 seeds per condition; TOST for interchangeability; Holm–Bonferroni corrections. We control sequence length and curriculum effects.

## 5. Results
- **Monotonicity:** At matched perplexity and tokens, aggregate ICL increases with integrated PNE across both small-batch and injected-noise conditions. `η/B` alone does not explain the variance once curvature and anisotropy are controlled.
- **Phase sensitivity:** Concentrating PNE late in training yields higher ICL than early-phase concentration for the same total exposure and perplexity.
- **Source interchangeability:** Small-batch and injected-noise runs are statistically indistinguishable on ICL tasks when their time-resolved PNE and low-rank noise spectra are matched. Mismatched anisotropy breaks this equivalence.
- **Trade-off:** Increasing PNE reduces parametric memorization (factual recall) while improving ICL, tracing a Pareto frontier.
- **Correlates:** Higher PNE associates with stronger induction-head metrics and flatter minima (lower SAM sharpness, smaller leading Hessian eigenvalues), consistent with our proposed mechanism.
- **Clipping:** Using effective (post-clip) PNE tightens correlations with ICL outcomes and reveals stability thresholds.

## 6. Discussion
PNE frames optimizer-preconditioned SGN, relative to curvature, as a controllable driver of ICL. The SDE view is a powerful heuristic, but momentum, time-varying preconditioning, and clipping demand empirical calibration. Matching second moments (PNE) is necessary but not sufficient—anisotropy also matters. Practically, late-stage batch reduction or anisotropy-matched noise injection can boost ICL at fixed perplexity, and logging effective PNE provides a concrete control knob for this trade-off.

## 7. Limitations and Threats to Validity
- **Scope:** Demonstrated on Transformer decoders with AdamW; other optimizers and architectures require further study.
- **Estimation bias:** Curvature proxies and low-rank covariance approximations are imperfect; held-out estimation reduces, but does not eliminate, bias.
- **SDE idealizations:** The `dt ≈ η_t` and `κ_mom` approximations are modeling choices; we rely on empirical calibration but offer no formal guarantees.
- **Injection feasibility:** Exact noise matching is impossible when the target diffusion is below the current in some directions (PSD constraint); we report deficits.
- **Evaluation sensitivity:** Few-shot benchmarks are sensitive to templates and formatting; we mitigate with standardized prompts and multiple seeds.

## 8. Conclusion
Optimizer-preconditioned SGN, quantified online by PNE, is a controllable training lever for ICL. By manipulating training noise in an optimizer-, curvature-, and clipping-aware manner—especially late in training—we can enhance context adaptation, reframing a key emergent ability as an outcome of measurable and directable training dynamics.

## 9. Reproducibility
We release a PNE logging library, anisotropy-matched noise injection utilities, training/evaluation scripts, configurations, seeds, raw logs, and analysis notebooks.
