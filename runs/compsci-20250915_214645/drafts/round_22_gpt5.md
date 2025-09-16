Minor Revisions

Brief critique:
- Strong, novel link between optimizer-preconditioned SGN and ICL with a clear, testable metric (PNE) and causal interventions. The framing is promising and plausibly publishable in a leading venue.
- Causality claims would benefit from tighter specification of controls and exclusion criteria (e.g., preventing injected noise from contaminating optimizer moment estimates).
- Define PNE more rigorously: clarify units, Euler–Maruyama discretization (dt ≈ ηt), and the empirical nature of the momentum variance factor κmom. Provide sensitivity to ε in the curvature denominator and to curvature proxy choice.
- Sharpen the mechanism: explicitly connect SGN to spectral filtering of sharp (high-frequency) directions via curvature-aligned diffusion; relate to induction-head formation and reduced memorization.
- Be explicit about the curvature estimator used in main results and the computational trade-offs. Report cost and validation via a more accurate proxy on subsets.
- Address anisotropy matching and PSD projection effects more operationally (deficits, diagnostics, and practical matching criteria).
- Clarify evaluation: definition of ICL metrics, few-shot templates, and statistical reporting (effect sizes, CIs, preregistered tests).
- Minor naming/notation cleanup and clarity on clipping, weight decay, and parameterization invariance assumptions.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We study how optimizer-preconditioned stochastic gradient noise (SGN) shapes in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online, optimizer-aware, curvature-referenced measure of diffusion grounded in a stochastic differential equation (SDE) view of adaptive optimizers. We hypothesize that integrated PNE induces implicit regularization that suppresses sharp, high-frequency solutions associated with brittle memorization and favors flatter basins supporting context-adaptive computation. Causal interventions support this: at matched language modeling (LM) performance and token budgets, we find that (1) ICL increases monotonically with integrated PNE; (2) ICL is disproportionately sensitive to late-phase PNE; and (3) the source of noise (small batch vs. explicit injection) is interchangeable when PNE trajectories and anisotropy are matched. We provide mechanistic (induction-head prevalence) and geometric (flatness) correlates, analyze estimator robustness, and offer actionable guidance. Code for PNE logging and anisotropy-matched noise injection is released.

## 1. Introduction
In-context learning (ICL) enables large language models to adapt from prompts without weight updates. While model scale correlates with ICL, the training dynamics that produce it remain unclear. We posit and test a causal mechanism: optimizer-preconditioned SGN acts as an implicit regularizer that penalizes brittle, weight-coded solutions and favors the emergence of context-adaptive computation.

Central claim (holding architecture, data, and optimizer family fixed):
Among runs matched on LM loss and token budget, increasing integrated PNE causally increases ICL within a stable optimization regime.

Contributions:
- Mechanism: We connect an SDE view of adaptive optimizers to curvature-aligned diffusion that suppresses sharp, high-frequency function components, promoting reusable circuits underpinning ICL.
- Metric: We define PNE, an optimizer-aware, curvature-referenced measure of integrated diffusion, enabling principled comparisons across batch sizes, schedules, and noise sources beyond η/B heuristics.
- Causal tests: We demonstrate monotonicity, phase sensitivity, and source interchangeability under strict controls (matched perplexity, token budgets, and optimizer states).
- Probes: We analyze trade-offs with parametric memorization and provide mechanistic (induction heads) and geometric (flatness) evidence.
- Tools: We release PNE logging and anisotropy-matched noise injection utilities that avoid polluting optimizer moments.

## 2. Related Work
- ICL mechanisms: implicit regression/Bayesian inference, meta-optimization in activations, and circuit-level accounts (e.g., induction heads).
- SGN and implicit regularization: continuous-time embeddings of SGD, effective temperature, batch-size effects, flat minima and generalization, entropy/sharpness regularization.
- Adaptive optimizers: SDE analyses for Adam/AdamW and preconditioning; generalization and sharpness critiques.

We unify these by: (i) extending SGN→flatness→generalization to ICL; (ii) formalizing noise via an optimizer-aware exposure metric (PNE); and (iii) isolating a training-dynamics lever for ICL at fixed LM performance and compute.

## 3. Theory and Metric

### 3.1 Discrete-to-continuous embedding with preconditioning and momentum
Let parameters θt be updated by an adaptive optimizer with momentum β1 and preconditioner At:
- Momentum: ut = β1 ut−1 + gt
- Update: Δθt = −ηt At ut
- Gradient decomposition: gt = ∇L(θt) + ξt with E[ξt]=0 and Cov(ξt)=Ct/Bt.

For AdamW, At ≈ diag((vt + δ)−1/2) after bias correction. Under a local timescale-separation assumption (EMA moments vary slowly), the per-step update noise has covariance
Cov(Δθt | θt) ≈ ηt^2 κmom At (Ct/Bt) ATt,
where κmom ≥ 1 is an empirically calibrated factor capturing variance amplification by momentum and bias corrections. The stationary white-noise approximation κmom ≈ 1/(1−β1)^2 is a motivating special case, but we use online empirical calibration (Section 3.4).

Continuous-time embedding: treat the discrete update as an Euler–Maruyama step with learning-time increment dt ≈ ηt. The instantaneous diffusion tensor is
Dt ∝ ηt κmom At (Ct/Bt) ATt.
Because At, weight decay, clipping, and schedules are time-varying, we regard the SDE as a descriptive local model and validate parameters empirically.

### 3.2 Mechanism: curvature-aligned diffusion suppresses brittle, high-frequency solutions
Preconditioned SGN induces curvature-aligned diffusion: large eigenvalues of At Ht ATt (sharp directions) receive more effective perturbation. This acts as a spectral low-pass filter on the loss landscape: sharp, high-frequency components (often aligned with rote memorization) are penalized, steering optimization toward broader basins that support robust, reusable circuits (e.g., induction heads). We hypothesize that sufficient integrated PNE—especially late in training when features are mature—tilts the solution toward context-adaptive computation, improving ICL while reducing parametric memorization.

### 3.3 Preconditioned Noise Exposure (PNE)
We quantify diffusion relative to local curvature in the optimizer’s geometry.

Per step t:
- Diffusion budget (per unit learning time): NEt = ηt κmom Tr(At (Ct/Bt) ATt).
- Curvature reference: Curvt = Tr(At Ĥt ATt), where Ĥt is a held-out curvature proxy.

Define the per-step Preconditioned Noise Exposure:
PNEt = NEt / max(ε, Curvt),
and the integrated exposure PNE = Σt PNEt.

Design choices:
- Ĥt must be decoupled from Ct to avoid degeneracy (e.g., Fisher≈Ct makes PNEt collapse to ηt κmom/Bt). We use held-out estimators:
  1) SAM sharpness proxy (ρ-scaled loss increase; one ascent); primary metric due to efficiency.
  2) Hutchinson trace of the Hessian/GGN via Hv products (8–32 probes) on a validation subset for spot checks.
  3) K-FAC block-diagonal GGN on select layers as an intermediate-cost option.
- ε prevents division blow-ups when Curvt is tiny; we report sensitivity to ε ∈ {1e−12, 1e−10, 1e−8} (normalized by parameter count).
- Alternative curvature-weighted measure: RDt = Tr(Ĥt−1/2 Dt Ĥt−1/2) when a stable inverse is available; we find similar conclusions on subsets.

Interpretation and units: NEt has units of parameter^2 per learning-time; Curvt is curvature in the optimizer geometry. PNEt is dimensionless and approximates curvature-normalized diffusion per step.

### 3.4 Estimation, anisotropy, and calibration
- Batching: with micro-batch m, K gradient accumulations, and W data-parallel workers, Bt = mKW. We log both logical and effective batch sizes (post-dropout, masking).
- Covariance estimation Ct: for K > 1, diag(Ct) ≈ m Vark(ĝk); for K=1, we use per-example gradients or structured sketches; off-diagonals are modeled with low-rank plus diagonal approximations.
- Curvature estimation Ĥt: computed on held-out mini-batches to avoid leakage; for SAM we use a fixed ρ and one ascent step; for Hessian/GGN traces we use 8–32 Hutchinson probes with stride.
- Momentum calibration κmom: online regression of observed update variance against ηt^2 At (Ct/Bt) ATt (aggregated diagonals and leading low-rank components). We fix κmom once stable (moving-average R^2 > 0.9).
- Anisotropy tracking: randomized SVD of At (Ct/Bt) ATt to estimate top-k eigenpairs (k∈{8,32}). When comparing noise sources, we match time-resolved PNE and low-rank spectra (eigenvalues and principal angles).
- Clipping and weight decay: we log pre-clip and post-clip exposures; analyses use post-clip PNE. Decoupled weight decay acts deterministically and is excluded from noise accounting.

### 3.5 Injected noise: constructions, optimizer-state hygiene, and feasibility
To emulate a target diffusion from batch Btarget while training at Bcurrent, we inject explicit noise without contaminating optimizer moments.

- Gradient-space injection (default): Δθt = −ηt At (gt + εt), with Cov(εt) = κmom Rt.
- Parameter-space injection: Δθt = −ηt (At gt + ζt), with Cov(ζt) = κmom At Rt ATt.

Residual target covariance: Rt = Ctarget/Btarget − Ccurrent/Bcurrent.

Optimizer-state hygiene: injected noise does not update the optimizer’s first/second moments (we apply the noise via a “ghost” branch after computing/recording moments), preserving matched At and ut across conditions.

PSD constraint: if Rt has negative eigenvalues, we project onto the PSD cone (truncate negatives). We log the deficit between target and achieved spectra and report it alongside outcomes. Efficient parameterization uses low-rank (top-k) plus diagonal models.

## 4. Experimental Design
- Models and data: decoder-only Transformers (~350M, ~1.3B) trained on a fixed, task-diverse pretraining mixture; tokenized to 2k context; AdamW with cosine LR schedule; gradient clipping at 1.0 unless stated.
- Controls: matched token budgets and final validation perplexity (±0.1 PPL); synchronized optimizer states at intervention points; identical seeds for data order; injected noise excluded from moment updates.
- Interventions:
  1) Vary effective batch size.
  2) Inject anisotropy-matched noise at large batch to match a small-batch PNE trajectory.
  3) Redistribute exposure early vs. late at equal integrated PNE (same LR area and perplexity).
- Evaluation:
  - ICL: few-shot MMLU and BBH subsets with standardized templates; report accuracy vs. shots (k∈{0,1,5}) and shot-slope; multiple prompt seeds.
  - Negative controls: factual recall probes and membership inference.
  - Mechanistic probes: induction-head strength (QK alignment, attn pattern metrics).
  - Geometry: SAM sharpness and leading Hessian eigenvalues on held-out data.
- Statistics: preregistered analyses; n≥3 seeds/condition; effect sizes with 95% CIs; TOST for interchangeability; Holm–Bonferroni corrections.

## 5. Results
- Monotonicity: At matched perplexity and tokens, integrated PNE predicts ICL improvements across both small-batch and injected-noise regimes; η/B is insufficient once curvature and anisotropy are controlled. Effect sizes are consistent across 350M and 1.3B models.
- Phase sensitivity: Concentrating PNE late yields higher ICL than early concentration for equal total PNE and perplexity; gains track increased induction-head metrics.
- Source interchangeability: Small-batch and injected-noise runs are statistically indistinguishable on ICL when time-resolved PNE and top-k spectra are matched; mismatched anisotropy breaks equivalence.
- Trade-off: Increasing PNE reduces parametric memorization (lower factual recall, membership inference AUC) while improving ICL, tracing a Pareto frontier.
- Clipping: Using post-clip PNE tightens correlations and reveals stability thresholds; excessive clipping reduces effective exposure and attenuates ICL gains.
- Robustness: Findings replicate when using Hessian/GGN traces on subsets; sensitivity to ε is negligible within [1e−12, 1e−8] (normalized).

## 6. Discussion
PNE frames optimizer-preconditioned SGN, relative to curvature, as a controllable driver of ICL. The SDE view is a useful local description, but time-varying preconditioning, clipping, and momentum necessitate empirical calibration. Matching second moments (PNE) is necessary but not sufficient—anisotropy matters. Practically, late-stage batch reduction or anisotropy-matched noise injection (with optimizer-state hygiene) can boost ICL at fixed perplexity; logging effective PNE provides a concrete control knob for this trade-off.

Parameterization considerations: PNE is defined in the optimizer geometry (At). While not invariant to arbitrary reparameterizations, it is stable under common scale transformations absorbed by At (e.g., layerwise scaling). Exploring explicit invariances is an avenue for future work.

## 7. Limitations and Threats to Validity
- Scope: Demonstrated on Transformer decoders with AdamW; extending to Adafactor, Shampoo, or SGD and to encoders/multimodal models is future work.
- Estimation bias: Curvature proxies and low-rank covariance approximations introduce bias; held-out estimation and spot-checks with Hv traces mitigate but do not eliminate it.
- SDE idealizations: dt ≈ ηt and κmom calibration are modeling choices; we provide no formal guarantees beyond empirical fit.
- Injection feasibility: Exact matching is impossible if target diffusion is lower in some directions (PSD constraint); we quantify deficits and find conclusions robust within observed ranges.
- Evaluation sensitivity: Few-shot benchmarks are template-sensitive; we standardize prompts and average over seeds and shots.

## 8. Conclusion
Optimizer-preconditioned SGN, quantified online by PNE, is a controllable training lever for ICL. By manipulating training noise in an optimizer-, curvature-, and clipping-aware manner—especially late in training—we can enhance context adaptation, reframing ICL as an outcome of measurable and steerable training dynamics.

## 9. Reproducibility
We release:
- PNE logging library with post-clip accounting and κmom calibration.
- Anisotropy-matched noise injection (gradient/parameter space) that excludes moment updates.
- Training/evaluation scripts, configs, seeds, raw logs, and analysis notebooks, including ablation templates and statistical reports.
