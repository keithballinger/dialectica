Minor Revisions

Brief critique:
- The paper is compelling and potentially field-leading: the PNE construct is novel, the causal framing is clear, and the intervention design is strong. To meet top-venue standards, tighten the math and sharpen claims.
- Formalize the SDE model and PNE before interpretation: state assumptions, units, and invariances; specify the role and expected magnitude of κ_mom (with a theoretical baseline) and the dt ≈ ηt mapping as a heuristic.
- Make the “spectral filtering” mechanism more explicit with a local quadratic approximation, showing how integrated diffusion induces an effective curvature-weighted penalty (e.g., ∫ Tr(DtHt) dt), linking this to flatness and ICL.
- Specify curvature and covariance estimators precisely, including the SAM-based proxy parameters, Hutchinson trace settings, micro-batch variance estimation, and how clipping is accounted for in both numerator and denominator of PNE.
- Clarify anisotropy matching: define the target spectrum and subspace alignment criteria, and report a quantitative mismatch metric.
- Strengthen statistical reporting details (TOST equivalence bounds, power and seed counts, corrections) and define mechanistic probes (induction-head metric) to enable replication.
- Note explicitly that PSD projection yields a conservative lower bound on diffusion, and that all interchangeability claims are conditioned on observed spectral deficits.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We establish a causal link between optimizer-preconditioned stochastic gradient noise (SGN) and the emergence of in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online, curvature-referenced measure of effective diffusion in the optimizer’s geometry. Using a stochastic differential equation (SDE) view of adaptive optimization, we show that integrated PNE behaves like a curvature-weighted regularizer that suppresses sharp, high-frequency function components associated with brittle memorization and promotes flatter minima that support context-adaptive computation. Causal interventions—varying batch size, injecting anisotropy-matched noise, and redistributing diffusion over training phases—demonstrate that, at matched language modeling (LM) performance and token budgets: (1) ICL increases monotonically with integrated PNE; (2) late-phase PNE is disproportionately effective; and (3) small-batch noise and explicit injection are interchangeable when PNE trajectories and diffusion spectra are matched. Mechanistic (induction heads) and geometric (flatness) probes corroborate the mechanism. We release tools for PNE monitoring and anisotropy-matched noise injection.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to perform new tasks from prompts without finetuning. While ICL correlates with scale, its training-time drivers remain unclear. We posit a specific, controllable mechanism: optimizer-preconditioned SGN acts as an implicit spectral regularizer that steers models away from brittle memorization and toward general, context-adaptive circuitry.

Central claim: For fixed architecture, dataset, and optimizer family, among runs matched for final LM loss and compute, increasing integrated PNE causally increases ICL.

Contributions:
- Mechanism: A curvature-aligned diffusion perspective that provides a spectral account of how SGN promotes ICL-supporting circuits.
- Metric: PNE, an optimizer-aware, curvature-referenced measure of diffusion enabling principled comparisons across batch sizes, schedules, and noise sources.
- Causal tests: Monotonicity, phase sensitivity, and interchangeability of noise sources under stringent controls for perplexity and optimizer state.
- Probes: Links between PNE, induction-head prevalence, and loss-landscape flatness.
- Tooling: PNE logging and anisotropy-matched noise injection that decouples interventions from optimizer moments.

## 2. Related Work
- ICL mechanisms: Implicit Bayesian inference, meta-learning in activation space, and induction-head circuits have been proposed; we tie their emergence to controllable training dynamics.
- SGN and implicit regularization: SGN’s role in flat minima and generalization is documented for SGD; we extend to ICL with adaptive optimizers.
- Adaptive optimizers as SDEs: Prior SDE analyses of Adam-like methods emphasize preconditioning; we formalize and measure its diffusion via PNE and link it to ICL.

## 3. Theory and Metric

### 3.1 Optimization as a Preconditioned SDE
Let θ ∈ R^d be parameters, L(θ) the population loss, g_t = ∇L(θ_t) + ξ_t the stochastic gradient with E[ξ_t]=0 and Cov(ξ_t)=C_t/B_t (B_t is effective batch size). An AdamW-like update with momentum β₁ and preconditioner A_t is:
- u_t = β₁ u_{t-1} + g_t
- Δθ_t = -η_t A_t u_t

With slowly varying moments (A_t, u_t) relative to ξ_t and small η_t, the per-step update noise has covariance:
Cov(Δθ_t | θ_t) ≈ η_t^2 κ_mom A_t (C_t/B_t) A_t^T,
where κ_mom captures variance amplification due to momentum (for white noise, κ_mom ≈ 1/(1−β₁²); we calibrate empirically).

We interpret the discrete process as Euler–Maruyama steps of:
dθ = −A_t ∇L(θ) dt + √(2D_t) dW_t,
with dt ≈ η_t (heuristic time scaling) and diffusion tensor D_t ∝ η_t κ_mom A_t (C_t/B_t) A_t^T. This maps training to stochastic preconditioned gradient flow.

### 3.2 Spectral Mechanism via Local Quadratic Approximation
Locally, approximate L(θ) ≈ L(θ*) + 1/2 (θ−θ*)^T H_t (θ−θ*). Under the SDE above, the expected loss increase from diffusion over dt is ≈ 1/2 Tr(D_t H_t) dt. Integrating over time yields an effective curvature-weighted penalty ∫ 1/2 Tr(D_t H_t) dt that biases trajectories toward flatter regions (smaller eigenvalues of A_t H_t A_t^T). Since sharp directions are empirically linked to high-frequency function components and memorization, curvature-aligned diffusion acts as a spectral low-pass filter, encouraging reusable circuits (e.g., induction heads) that support ICL.

This generalizes stationary-distribution results for SGD (e.g., temperature proportional to noise) to the optimizer’s geometry: preconditioning shapes which spectral components are most suppressed.

### 3.3 Preconditioned Noise Exposure (PNE)
We define the per-step exposure:
PNE_t = (η_t κ_mom Tr[A_t (C_t/B_t) A_t^T]) / max(ε, Tr[A_t Ĥ_t A_t^T]),
and the integrated exposure PNE = Σ_t PNE_t.

- Numerator: Trace of the preconditioned diffusion (units: parameter² per learning-time).
- Denominator: Reference curvature via a proxy Ĥ_t (Hessian/GGN/SAM) on held-out data to reduce coupling with C_t.
- PNE_t is dimensionless, interpretable as diffusion budget relative to curvature in the optimizer’s geometry.
- Invariance: PNE is invariant to scalar rescalings absorbed by A_t (e.g., weight norm scaling common in Adam-like optimizers), but not fully reparameterization-invariant.

Under the quadratic approximation, larger integrated PNE increases the cumulative penalty ∫ Tr(D_t H_t) dt, favoring flatter minima and ICL-supporting circuits.

### 3.4 Estimation and Calibration
- Gradient covariance C_t: Estimate diag(C_t) from micro-batch gradients; model off-diagonals with low-rank+diag approximations (rank k ≪ d) updated via randomized sketches.
- Curvature Ĥ_t:
  - Primary: SAM-based sharpness with ascent radius ρ (we use ρ=0.05 by default), measured on held-out batches; implement as trace proxy via Hutchinson on the SAM-perturbed point.
  - Validation: Hutchinson trace of Hessian/GGN on subsets (≥5 Hutchinson vectors per check).
- Momentum factor κ_mom: Initialize with 1/(1−β₁²), then calibrate by regressing observed update covariance against η_t^2 A_t (C_t/B_t) A_t^T until R² ≥ 0.9; freeze thereafter.
- Clipping: Account for gradient clipping by estimating the fraction of mass truncated; adjust both numerator (effective C_t) and denominator (sharpness) accordingly.
- Anisotropy: Track top-k eigenpairs of D_t (k=32 by default) via randomized SVD; define a mismatch metric δ_spec(t) = ||Λ_t^src − Λ_t^tgt||_1 / ||Λ_t^src||_1 and a subspace alignment metric via principal angles.

### 3.5 Controlled Noise Injection
Goal: Match a target diffusion trajectory while holding LM performance fixed.
- Injection sites: Gradient space (add ε_t to g_t after moment updates) or parameter space (add ζ_t to Δθ_t after the optimizer step); both avoid contaminating A_t and moment estimates.
- Targeting: Choose Cov(ε_t) or Cov(ζ_t) to realize a residual diffusion R_t = C_target/B_target − C_current/B_current in the preconditioned geometry.
- PSD projection: If R_t is not PSD, project onto the PSD cone by zeroing negative eigenvalues; log spectral deficit δ_PSD(t). All interchangeability claims are conditioned on δ_PSD trajectories.

## 4. Experimental Design
- Models: Decoder-only Transformers (~350M, ~1.3B) with standard pre-LN blocks and rotary embeddings.
- Data and training: Fixed tokenizer; fixed pretraining corpus; AdamW, cosine LR schedule with warmup; gradient clipping at 1.0; mixed precision.
- Controls: Compare runs matched for token budgets and final validation perplexity (±0.1 PPL). Use identical data ordering seeds; synchronize optimizer states at intervention points for phase experiments.
- Interventions:
  1) Batch size sweep to modulate PNE naturally.
  2) Large-batch training with anisotropy-matched noise to reproduce small-batch PNE trajectories.
  3) Phase redistribution: Increase early vs. late PNE while holding total PNE and final perplexity constant.
- Evaluation:
  - ICL: Few-shot accuracy on MMLU and BBH subsets with standardized templates and calibrated decoding.
  - Memorization: Factual recall probes and membership-inference AUC on a held-out canary set.
  - Mechanistic probes: Induction-head metrics (QK pattern matching on repeated-token spans; attention-score slope and copying accuracy).
  - Geometry: SAM sharpness and Hessian trace on held-out data; loss interpolation flatness between checkpoints.
- Statistics: n≥3 seeds per condition; report effect sizes with 95% CIs. For interchangeability, apply TOST with pre-registered equivalence bounds (e.g., ±0.5% absolute accuracy) and Holm–Bonferroni correction. Power analysis targets 80% power for medium effects.

## 5. Results
- Monotonicity: At matched perplexity, ICL increases monotonically with integrated PNE across batch-size and noise-injection regimes and across both model scales; PNE outperforms η/B in predictive power (higher R²).
- Phase sensitivity: With fixed total PNE and matched perplexity, concentrating PNE late yields significantly higher ICL and stronger induction-head metrics than early concentration.
- Source interchangeability: Small-batch and large-batch+injection runs are statistically equivalent (TOST) on ICL when time-resolved PNE and top-k diffusion spectra are matched; spectral or temporal mismatches break equivalence in proportion to δ_spec and δ_PSD.
- ICL–memorization trade-off: Increasing PNE reduces parametric memorization (lower factual recall, lower membership-inference AUC) while increasing ICL, tracing a Pareto frontier.
- Clipping: Stronger clipping reduces effective PNE and attenuates ICL gains; incorporating clipping into PNE tightens correlations between PNE and ICL.

## 6. Discussion
PNE provides a control variable linking optimizer-preconditioned diffusion to ICL. The curvature-weighted penalty view explains why late-phase diffusion—when models sit near low-loss basins—most effectively induces context-adaptive circuits rather than amplifying noise. Practically, modest late-stage batch-size reductions or anisotropy-matched noise injections improve ICL at fixed perplexity.

While PNE is defined in the optimizer’s geometry (not fully reparameterization-invariant), it is stable under common layer-wise scalings absorbed by A_t. Extending PNE to other preconditioners (e.g., Shampoo) and modalities is a promising direction.

## 7. Limitations and Threats to Validity
- Scope: Results are for decoder-only Transformers and AdamW; broader generalization is untested.
- Estimation error: Low-rank covariance and curvature proxies introduce bias; we mitigate with held-out data and subset validation but cannot eliminate it.
- SDE approximation: dt ≈ η_t and κ_mom calibration are heuristic; the continuous-time view is local.
- Injection feasibility: PSD projection limits exact matching; equivalence claims are conditioned on logged spectral deficits.
- Confounders: Although we match perplexity and control optimizer state, subtle data–optimizer interactions may remain.

## 8. Conclusion
Optimizer-preconditioned SGN is a controllable driver of ICL. PNE quantifies and manipulates this diffusion, enabling causal tests and practical improvements in ICL at fixed LM performance. This reframes ICL as an outcome of a measurable, curvature-aligned regularization process induced by training dynamics.

## 9. Reproducibility
We provide:
- A PNE logger with clipping-aware covariance, κ_mom calibration, and Hutchinson-based curvature estimation.
- An anisotropy-matched noise injector that decouples interventions from optimizer moments and reports δ_spec and δ_PSD.
- Scripts and analysis notebooks to reproduce all experiments, including TOST settings and probe implementations.
