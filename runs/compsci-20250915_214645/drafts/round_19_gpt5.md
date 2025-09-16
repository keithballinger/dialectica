Major Revisions

Brief critique:
- The P-NE* definition risks degeneracy when Curv_t is taken as an empirical Fisher: in common regimes F ≈ C, making P-NE* collapse toward η/B and obscuring anisotropy and curvature effects. A curvature proxy that is not the same statistic as the noise (e.g., GGN, Hessian, or SAM sharpness) is needed, ideally on held-out data.
- The diffusion accounting omits momentum’s variance amplification/correlation structure. Even under timescale separation, an effective amplification factor κ(β1) should be included (or empirically calibrated) to compare exposures across optimizers/settings.
- The injected-noise construction lacks PSD feasibility checks and a fallback strategy when the target diffusion is lower than the current (negative residual). Also, distinguish and justify parameter-space vs gradient-space injection; both lead to different covariance formulas.
- The SDE embedding and dt ≈ η_t heuristic are acceptable but the stationarity analogy for AdamW is overstated; sharpen caveats and emphasize empirical calibration (post-clip, mixed precision).
- Experimental claims would be strengthened by: (i) explicit anisotropy matching protocols (low-rank+diag) with diagnostics; (ii) ablations showing P-NE* discriminates runs where η/B alone cannot; (iii) held-out curvature estimation to avoid leakage; and (iv) sequence-length and curriculum controls.
- Terminology and notation: define A_t, F_t, C_t precisely; clarify units; consider dropping the asterisk in P-NE* or clearly motivate it.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We study how optimizer-preconditioned stochastic gradient noise (SGN) shapes in-context learning (ICL). We introduce Preconditioned Noise Exposure (P-NE*), an online, optimizer-aware, curvature-referenced measure of diffusion grounded in a stochastic differential equation (SDE) perspective for adaptive optimizers with momentum. We hypothesize that integrated P-NE* induces implicit regularization that steers training away from sharp minima associated with brittle, weight-coded solutions and toward flatter basins supporting context-adaptive computation. Causal interventions support this: at matched language modeling (LM) performance and token budgets, (1) ICL increases with integrated P-NE*; (2) ICL is disproportionately sensitive to late-phase P-NE*; and (3) the source of noise (small batch vs explicit injection) is largely interchangeable when P-NE* trajectories and anisotropy are matched. We provide mechanistic (induction-head prevalence) and geometric (flatness) correlates, analyze estimator robustness and clipping, and offer actionable guidance. Code for P-NE* logging and anisotropy-matched noise injection is released.

## 1. Introduction
In-context learning (ICL) enables large language models to adapt from prompts without weight updates. While scale correlates with ICL, the training dynamics that produce it remain unclear. We posit and test a causal mechanism: integrated, optimizer-preconditioned SGN acts as an implicit regularizer penalizing brittle, weight-coded solutions and favoring context-adaptive computation.

Central claim (holding architecture, data, and optimizer family fixed):
- Among runs matched on LM loss and token budget, increasing integrated P-NE* causally increases ICL within a stable optimization regime.

Contributions:
- Mechanism: Link an SDE view of adaptive optimizers to a trade-off between parametric memorization and context-adaptive computation, positioning preconditioned SGN as a controllable driver of ICL.
- Metric: P-NE*, an optimizer-aware, curvature-referenced measure of integrated diffusion enabling comparisons across batch sizes, schedules, and noise sources without collapsing to η/B.
- Causal tests: Monotonicity, phase sensitivity, and source interchangeability under strict controls (matched perplexity, token budgets, optimizer states).
- Probes: Trade-off with weight-based memorization; mechanistic (induction heads) and geometric (flatness) analyses; stability boundaries and clipping-aware exposure.

## 2. Related Work
- ICL mechanisms: implicit regression/Bayesian inference, meta-optimization in activations, circuit-level accounts (e.g., induction heads).
- SGN and implicit regularization: continuous-time embeddings of SGD, effective temperature, batch-size effects, heavy-tailed noise, flat minima and generalization, entropy/sharpness regularization.
- Adaptive optimizers: SDE analyses for Adam/AdamW and preconditioning; generalization and sharpness critiques.

We extend: (i) SGN→flatness→generalization to the ICL regime; (ii) noise/batch effects to an optimizer-aware exposure metric; (iii) ICL theories by isolating a training-dynamics lever under matched LM performance and compute.

## 3. Theory and Metric

### 3.1 Discrete-to-continuous embedding with preconditioning and momentum
Let parameters θ_t be updated by an adaptive optimizer with momentum β1 and (possibly diagonal) preconditioner A_t:
- Momentum: u_t = β1 u_{t−1} + g_t
- Update: Δθ_t = −η_t A_t u_t
- Gradient decomposition: g_t = ∇L(θ_t) + ξ_t with E[ξ_t]=0, Cov(ξ_t)=C_t/B_t

For AdamW, A_t ≈ diag((v_t + δ)−1/2) after bias correction; weight decay adds drift. Under timescale separation (EMA moments vary slowly), the per-step update noise has covariance approximately
Cov(Δθ_t | θ_t) ≈ η_t^2 κ(β1) A_t (C_t/B_t) A_t^T,
where κ(β1) ≥ 1 captures momentum’s variance amplification and temporal correlation (κ(β1) ≈ 1/(1−β1)^2 under white-noise and stationarity assumptions; we also support empirical calibration).

To embed in continuous time, we map each step to dt ≈ η_t (heuristic valid for small, slowly varying η_t). Then the instantaneous diffusion scales as D_t ∝ η_t κ(β1) A_t (C_t/B_t) A_t^T. For AdamW, time variation of A_t, momentum, and weight decay break exact stationarity; we therefore treat the SDE as a local, descriptive model and calibrate empirically (pre- and post-clip).

Assumptions:
- Slowly varying A_t and EMA moments (timescale separation).
- Stable regime (bounded gradients; clipping logged).
- Local curvature approximations are informative in the A_t-geometry.

### 3.2 Information allocation: parameters vs activations
Solutions can encode information in weights (parametric memorization) or in context-conditioned activations (ICL). SGN-induced diffusion tends to select flatter basins that generalize better. We propose that sharp basins overrepresent brittle, weight-coded solutions, while flatter basins favor reusable, prompt-sensitive circuits (e.g., induction-like attention). Increased preconditioned SGN thus implicitly regularizes toward context-adaptive subroutines, especially with task-diverse pretraining data.

### 3.3 Preconditioned Noise Exposure (P-NE*)
We quantify diffusion relative to local curvature in the optimizer’s geometry.

Per step t:
- Diffusion budget (per unit learning time): NE_t = η_t κ(β1) Tr(A_t (C_t/B_t) A_t^T).
- Curvature reference: Curv_t = Tr(A_t Ĥ_t A_t^T), where Ĥ_t is a curvature proxy not equal to C_t (e.g., Gauss–Newton, Hessian, or SAM sharpness surrogate) estimated on held-out data or large batches.

Define P-NE*_t = NE_t / max(ε, Curv_t), and integrated exposure as Σ_t P-NE*_t.

Notes:
- Choice of Ĥ_t: Avoid empirical Fisher computed from the same minibatches used for C_t to prevent degeneracy (P-NE* ≈ η_t κ/B_t). We provide three options:
  1) Hutchinson-trace of the Hessian/GGN via autograd Hvps on held-out batches,
  2) K-FAC block-diagonal GGN approximation,
  3) SAM sharpness proxy (ρ-scaled loss increase).
- Units: P-NE* is dimensionless and optimizer-specific by design; it is a control/analysis tool within a training run, not invariant to reparameterization.

Diagonal estimator (common case): with A_t diagonal and diag(C_t),
NE_t ≈ η_t κ(β1) Σ_i A_t,ii^2 (C_t,ii/B_t); similarly Curv_t with diag(Ĥ_t).

### 3.4 Estimation, anisotropy, clipping, and calibration
Batch/parallelism:
- With micro-batch m, K gradient accumulations, and W data-parallel workers, B_t = mKW.

Covariance estimation:
- Variance-from-micro-batches (K > 1): if ĝ_k are K micro-batch gradients, diag(C_t) ≈ m Var_k(ĝ_k).
- K = 1 alternative: per-example gradients or memory/communication-efficient sketches (e.g., Hutchinson, structured transforms).

Curvature estimation:
- Held-out batches to avoid leakage; for Hessian/ GGN traces use 8–32 Hutchinson probes per log step; for SAM, standard ρ and one ascent step.

Anisotropy:
- Track leading eigenspectra of A_t (C_t/B_t) A_t^T via randomized SVD; log tail indices (heavy-tailedness). When comparing noise sources, match both time-resolved P-NE* and low-rank anisotropy (low-rank+diagonal fits).

Clipping and precision:
- Log potential exposure (pre-clip, high precision) and effective exposure (post-clip, mixed precision). Use effective exposure for correlations.

Momentum calibration:
- Optionally estimate κ(β1) online by regressing observed post-update variance against η_t^2 A_t (C_t/B_t) A_t^T; use the calibrated κ for P-NE* and for noise injection.

### 3.5 Injected noise: constructions and feasibility
Goal: emulate a target diffusion D_target ∝ η_t κ A_t (C_target/B_target) A_t^T when training at larger batch (smaller inherent SGN).

Two practical schemes:

- Parameter-space injection (post-preconditioner):
  Δθ_t = −η_t (A_t g_t + ζ_t).
  Per-step diffusion adds η_t^2 Cov(ζ_t). To match the target,
  Cov(ζ_t) = κ(β1) A_t (C_target/B_target − C_current/B_current) A_t^T.
- Gradient-space injection (pre-preconditioner):
  Δθ_t = −η_t A_t (g_t + ε_t),
  Cov(ε_t) = κ(β1) (C_target/B_target − C_current/B_current).

Feasibility and PSD constraint:
- Let R_t = C_target/B_target − C_current/B_current. If R_t is not positive semidefinite, set the negative eigenvalues to zero (clip_psd) and log the residual; exact matching is impossible when target < current in some directions.
- Use low-rank+diagonal parameterizations for R_t to match leading spectra with minimal overhead (1–3%).
- Closed-loop control: adjust the injection scale via a simple controller to hit a desired P-NE* trajectory measured post-clip.

## 4. Experimental Design
- Architectures & data: Decoder-only Transformers (~350M, ~1.3B) trained on a fixed pretraining mixture with identical filtering; identical seeds across branch points.
- Controls: AdamW, fixed hyperparameters; matched schedules; matched token budgets and final validation perplexity; synchronized optimizer states at intervention points.
- Manipulating P-NE*: (1) vary effective batch size; (2) inject explicit, anisotropy-matched noise at large batch (parameter- or gradient-space); (3) redistribute exposure early vs late at equal integrated P-NE*.
- Evaluation: ICL on MMLU and BBH subsets; negative controls on factual recall; mechanistic probes (induction-head strength); geometric probes (SAM sharpness; leading Hessian eigenvalues via Lanczos/Hutch++ on held-out batches).
- Statistics: Pre-registered analyses; ≥3 seeds per condition; mixed-effects models; TOST for interchangeability; Holm–Bonferroni corrections.
- Stability: Monitor gradient norms, clipping rates; report stability boundaries vs (η_t, P-NE*). Control sequence length and curriculum to rule out confounds.

## 5. Results
- Monotonicity: At matched perplexity and tokens, aggregate ICL increases with integrated P-NE* across both small-batch and injected-noise conditions. η/B alone does not explain the variance once curvature and anisotropy are controlled.
- Phase sensitivity: Concentrating P-NE* late in training yields higher ICL than early-phase concentration under equal total exposure and perplexity.
- Source interchangeability (qualified): Small-batch and injected-noise runs are statistically indistinguishable in ICL when their time-resolved P-NE* and low-rank spectra of A_t (C_t/B_t) A_t^T are matched. Mismatched anisotropy or heavy-tail indices breaks interchangeability.
- Trade-off: Increasing P-NE* reduces parametric memorization (factual recall) while improving ICL, tracing a Pareto frontier.
- Correlates: Higher P-NE* associates with stronger induction-head metrics and flatter minima (lower SAM sharpness, smaller leading Hessian eigenvalues), consistent with the proposed mechanism.
- Clipping-aware accounting: Using effective (post-clip) exposure tightens correlations with ICL outcomes and reveals stability thresholds.

## 6. Discussion
P-NE* frames optimizer-preconditioned SGN, relative to curvature, as a controllable driver of ICL. The SDE view is a heuristic guide; momentum, time-varying preconditioning, and clipping require empirical calibration. Matching second moments is necessary but not sufficient—anisotropy and heavy tails matter. Practically, late-stage batch reduction or anisotropy-matched noise injection can boost ICL at fixed perplexity, while logging effective P-NE* provides a concrete control knob.

## 7. Limitations and Threats to Validity
- Scope: Demonstrated on Transformer decoders with AdamW; other optimizers/architectures remain to be tested.
- Estimation bias: Diagonal/low-rank approximations and curvature proxies (Hessian/ GGNs/ SAM) are imperfect; held-out estimation reduces, but does not eliminate, bias.
- SDE idealizations: dt ≈ η_t and κ(β1) approximations are modeling choices; we provide empirical calibration but no guarantees.
- Injection feasibility: Exact matching is impossible when the target diffusion is below the current in some directions (PSD constraint); we report residuals.
- Evaluation sensitivity: Few-shot benchmarks are template-sensitive; we mitigate with standardized prompts and multiple seeds.

## 8. Conclusion
Optimizer-preconditioned SGN, quantified online by P-NE*, is a controllable training lever for ICL. Manipulating training noise in an optimizer-, curvature-, and clipping-aware manner—especially late in training—enhances context adaptation at modest cost to parametric memorization, reframing ICL as an outcome of training dynamics.

## 9. Reproducibility
We release a P-NE* logging library, anisotropy-matched noise injection utilities (parameter- and gradient-space), training/evaluation scripts, configurations, seeds, raw logs, and analysis notebooks.
