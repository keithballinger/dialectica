Minor Revisions

Brief critique:
- The paper presents a novel and compelling mechanism (PNE) for ICL, supported by a rigorous causal-intervention framework. The work is potentially groundbreaking.
- The primary weakness is a lack of precision in connecting the theoretical mechanism to the proposed metric. The link between the `Tr(D*H)` penalty and the PNE definition—a ratio of traces—is presented as direct but is more accurately a well-motivated heuristic. The revision should clarify this relationship, explicitly stating the assumptions under which they are proportional and framing PNE as an empirically validated proxy for the underlying spectral penalty.
- The mathematical description of the PNE components in Sec 3.3 needs tightening. The SDE formulation implies the noise-induced update covariance scales with `η_t^2`, but the PNE numerator is written with `η_t`. The description of the numerator's units is also inconsistent. These should be clarified to ensure the metric is well-defined and its dimensionless property is transparent.
- A minor notational ambiguity in the SDE setup (Sec 3.1) regarding the stochastic gradient should be resolved.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We establish a causal link between optimizer-preconditioned stochastic gradient noise (SGN) and the emergence of in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online, curvature-referenced measure of effective diffusion in the optimizer’s geometry. Using a stochastic differential equation (SDE) view of adaptive optimization, we argue that SGN acts as an implicit, curvature-weighted regularizer that suppresses sharp, high-frequency function components associated with brittle memorization and promotes flatter minima that support context-adaptive computation. We show that integrated PNE serves as a tractable proxy for this regularization effect. Causal interventions—varying batch size, injecting anisotropy-matched noise, and redistributing diffusion over training phases—demonstrate that, at matched language modeling (LM) performance and token budgets: (1) ICL increases monotonically with integrated PNE; (2) late-phase PNE is disproportionately effective; and (3) small-batch noise and explicit injection are interchangeable when PNE trajectories and diffusion spectra are matched. Mechanistic (induction heads) and geometric (flatness) probes corroborate the mechanism. We release tools for PNE monitoring and anisotropy-matched noise injection.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to perform new tasks from prompts without finetuning. While ICL correlates with scale, its training-time drivers remain unclear. We posit a specific, controllable mechanism: optimizer-preconditioned SGN acts as an implicit spectral regularizer that steers models away from brittle memorization and toward general, context-adaptive circuitry.

**Central claim:** For a fixed architecture, dataset, and optimizer family, among runs matched for final LM loss and compute, increasing integrated PNE causally increases ICL.

**Contributions:**
- **Mechanism:** A curvature-aligned diffusion perspective that provides a spectral account of how SGN promotes ICL-supporting circuits.
- **Metric:** PNE, an optimizer-aware, dimensionless measure of diffusion enabling principled comparisons across batch sizes, schedules, and noise sources.
- **Causal tests:** Monotonicity, phase sensitivity, and interchangeability of noise sources under stringent controls for perplexity and optimizer state.
- **Probes:** Links between PNE, induction-head prevalence, and loss-landscape flatness.
- **Tooling:** PNE logging and anisotropy-matched noise injection that decouples interventions from optimizer moments.

## 2. Related Work
- **ICL mechanisms:** Implicit Bayesian inference, meta-learning in activation space, and induction-head circuits have been proposed; we tie their emergence to controllable training dynamics.
- **SGN and implicit regularization:** SGN’s role in finding flat minima and improving generalization is well-documented for SGD; we extend this to ICL with adaptive optimizers, where the preconditioner shapes the noise.
- **Adaptive optimizers as SDEs:** Prior SDE analyses of Adam-like methods emphasize preconditioning; we formalize and measure its diffusion via PNE and link it to ICL.

## 3. Theory and Metric

### 3.1 Optimization as a Preconditioned SDE
Let θ ∈ R^d be parameters and L(θ) the population loss. The stochastic gradient `g_t` on a batch `B_t` can be written `g_t = ∇L(θ_t) + ξ_t`, where the noise term `ξ_t` has `E[ξ_t]=0` and covariance `Cov(ξ_t) = C_t/B_t`. An AdamW-like update with momentum `β₁` and preconditioner `A_t` is:
- `u_t = β₁ u_{t-1} + g_t`
- `Δθ_t = -η_t A_t u_t`

With slowly varying moments (`A_t`, `u_t`) relative to `ξ_t` and small `η_t`, the noise in a single update step has covariance:
`Cov(Δθ_t | θ_t) ≈ η_t^2 κ_mom A_t (C_t/B_t) A_t^T`,
where `κ_mom` captures variance amplification due to momentum (for white noise, `κ_mom ≈ 1/(1−β₁²)`; we calibrate empirically).

We interpret this discrete process as Euler–Maruyama steps of a continuous SDE:
`dθ = −A_t ∇L(θ) dt + √(2D_t) dW_t`,
with a heuristic time scaling `dt ≈ η_t` and a diffusion tensor `D_t ∝ η_t κ_mom A_t (C_t/B_t) A_t^T`. This maps training to a stochastic preconditioned gradient flow.

### 3.2 Spectral Mechanism via Local Quadratic Approximation
Locally, approximate the loss `L(θ) ≈ L(θ*) + 1/2 (θ−θ*)^T H_t (θ−θ*)`. Under the SDE dynamics, the expected increase in loss from a diffusion step over `dt` is approximately `1/2 Tr(D_t H_t) dt`. Integrating over training yields an effective cumulative penalty, `∫ 1/2 Tr(D_t H_t) dt`, that biases trajectories toward flatter regions where the eigenvalues of the effective curvature `A_t H_t A_t^T` are smaller. Since sharp directions are empirically linked to high-frequency function components and memorization, this curvature-aligned diffusion acts as a spectral low-pass filter, discouraging brittle circuits and encouraging reusable ones (e.g., induction heads) that support ICL.

### 3.3 Preconditioned Noise Exposure (PNE)
The penalty term `Tr(D_t H_t)` is difficult to track directly. We define Preconditioned Noise Exposure (PNE) as a tractable, dimensionless proxy for this per-step regularization effect. It is the ratio of the total diffusion power to a reference curvature, both measured in the optimizer’s geometry:
`PNE_t = (η_t^2 κ_mom Tr[A_t (C_t/B_t) A_t^T]) / max(ε, Denom_t)`,
where `Denom_t = Tr[A_t Ĥ_t A_t^T]` is the reference curvature.

- **Numerator:** The trace of the update covariance from gradient noise, `Tr(Cov(Δθ_t | noise))`. It measures the total squared distance of the random walk step in parameter space. Units: `parameter²`.
- **Denominator:** A reference curvature measured via a proxy `Ĥ_t` (Hessian/GGN/SAM) on held-out data to reduce coupling with `C_t`. It is constructed to have units of `parameter²`, making PNE dimensionless.
- **Interpretation:** PNE is interpretable as a diffusion budget relative to the local geometry's scale. Under simplifying assumptions, such as spectral alignment between the diffusion and curvature tensors in the preconditioned space, `PNE_t` is proportional to the penalty `Tr(D_t H_t)`. We validate this connection empirically.
- **Invariance:** PNE is invariant to scalar rescalings absorbed by `A_t` (e.g., as in Adam-like optimizers), but not fully reparameterization-invariant.

### 3.4 Estimation and Calibration
- **Gradient covariance C_t:** Estimate `diag(C_t)` from micro-batch gradients; model off-diagonals with low-rank+diag approximations (rank `k ≪ d`) updated via randomized sketches.
- **Curvature Ĥ_t:**
  - **Primary:** SAM-based sharpness with ascent radius `ρ` (we use `ρ=0.05`), measured on held-out batches. The term `Denom_t` is scaled to be comparable to `ρ²`, providing a consistent reference length. We implement the trace via Hutchinson's method on the SAM-perturbed point.
  - **Validation:** Hutchinson trace of Hessian/GGN on subsets (≥5 vectors per check).
- **Momentum factor κ_mom:** Initialize with `1/(1−β₁²)`, then calibrate by regressing observed update covariance against `η_t^2 A_t (C_t/B_t) A_t^T` until R² ≥ 0.9; freeze thereafter.
- **Clipping:** Account for gradient clipping by estimating the truncated variance; adjust both numerator (effective `C_t`) and denominator (sharpness) accordingly.
- **Anisotropy:** Track top-`k` eigenpairs of the diffusion tensor (`k=32`) via randomized SVD; define a mismatch metric `δ_spec(t) = ||Λ_t^src − Λ_t^tgt||_1 / ||Λ_t^src||_1` and a subspace alignment metric via principal angles.

### 3.5 Controlled Noise Injection
Our goal is to match a target diffusion trajectory while holding LM performance fixed.
- **Injection sites:** Gradient space (add `ε_t` to `g_t` before moment updates) or parameter space (add `ζ_t` to `Δθ_t` after the optimizer step); both avoid contaminating `A_t` and moment estimates.
- **Targeting:** Choose `Cov(ε_t)` or `Cov(ζ_t)` to realize a residual diffusion `R_t = C_target/B_target − C_current/B_current` in the preconditioned geometry.
- **PSD projection:** If `R_t` is not positive semi-definite (PSD), project it onto the PSD cone by zeroing negative eigenvalues; log the spectral deficit `δ_PSD(t)`. All interchangeability claims are conditioned on `δ_PSD` trajectories.

## 4. Experimental Design
- **Models:** Decoder-only Transformers (~350M, ~1.3B) with standard pre-LN blocks and rotary embeddings.
- **Data and training:** Fixed tokenizer; fixed pretraining corpus; AdamW optimizer; cosine LR schedule with warmup; gradient clipping at 1.0; mixed precision.
- **Controls:** Compare runs matched for token budgets and final validation perplexity (±0.1 PPL). Use identical data ordering seeds; synchronize optimizer states at intervention points.
- **Interventions:**
  1. **Batch size sweep:** Modulate PNE naturally.
  2. **Noise injection:** Augment large-batch training with anisotropy-matched noise to reproduce small-batch PNE trajectories.
  3. **Phase redistribution:** Increase early vs. late PNE while holding total integrated PNE and final perplexity constant.
- **Evaluation:**
  - **ICL:** Few-shot accuracy on MMLU and BBH subsets with standardized templates and calibrated decoding.
  - **Memorization:** Factual recall probes and membership-inference AUC on a held-out canary set.
  - **Mechanistic probes:** Induction-head metrics (QK pattern matching on repeated-token spans; copying accuracy).
  - **Geometry:** SAM sharpness and Hessian trace on held-out data; loss interpolation flatness between checkpoints.
- **Statistics:** `n≥3` seeds per condition; report effect sizes with 95% CIs. For interchangeability, apply TOST with pre-registered equivalence bounds (e.g., ±0.5% absolute accuracy) and Holm–Bonferroni correction. Power analysis targets 80% power for medium effects.

## 5. Results
- **Monotonicity:** At matched perplexity, ICL increases monotonically with integrated PNE across batch-size and noise-injection regimes and both model scales. PNE outperforms simpler heuristics like `η/B` in predictive power (higher R²).
- **Phase sensitivity:** For a fixed total integrated PNE and matched perplexity, concentrating PNE in the late phase of training yields significantly higher ICL and stronger induction-head metrics than concentrating it early.
- **Source interchangeability:** Small-batch training and large-batch+injection runs are statistically equivalent (TOST) on ICL when time-resolved PNE and top-`k` diffusion spectra are matched. Spectral or temporal mismatches break equivalence in proportion to `δ_spec` and `δ_PSD`.
- **ICL–memorization trade-off:** Increasing PNE reduces parametric memorization (lower factual recall, lower membership-inference AUC) while increasing ICL, tracing a Pareto frontier.
- **Clipping impact:** Stronger clipping reduces effective PNE and attenuates ICL gains; our clipping-aware PNE formulation maintains a tight correlation between PNE and ICL.

## 6. Discussion
PNE provides a control variable linking optimizer-preconditioned diffusion to ICL. The curvature-weighted penalty view explains why late-phase diffusion—when models settle into low-loss basins—most effectively prunes sharp minima to induce context-adaptive circuits. Practically, modest late-stage batch-size reductions or targeted, anisotropy-matched noise injections can improve ICL at a fixed perplexity budget.

While PNE is defined in the optimizer’s geometry (not fully reparameterization-invariant), its invariance to rescalings common in adaptive optimizers makes it a stable and practical tool. Extending PNE to other preconditioners (e.g., Shampoo) and modalities is a promising direction.

## 7. Limitations and Threats to Validity
- **Scope:** Results are for decoder-only Transformers and AdamW; broader generalization is untested.
- **Estimation error:** Low-rank covariance and curvature proxies introduce bias; we mitigate this with held-out data and validation but cannot eliminate it.
- **SDE approximation:** The `dt ≈ η_t` mapping and `κ_mom` calibration are heuristics; the continuous-time view is a local approximation.
- **Injection feasibility:** PSD projection means noise matching is not always exact; equivalence claims are conditioned on logged spectral deficits.
- **Confounders:** Although we match perplexity and control optimizer state, subtle data–optimizer interactions may remain.

## 8. Conclusion
Optimizer-preconditioned SGN is a controllable driver of ICL. PNE quantifies and manipulates this diffusion, enabling causal tests and practical improvements in ICL at fixed LM performance. This work reframes ICL emergence as a measurable, curvature-aligned regularization process induced by training dynamics.

## 9. Reproducibility
We provide:
- A PNE logger with clipping-aware covariance, `κ_mom` calibration, and Hutchinson-based curvature estimation.
- An anisotropy-matched noise injector that decouples interventions from optimizer moments and reports `δ_spec` and `δ_PSD`.
- Scripts and analysis notebooks to reproduce all experiments, including TOST settings and probe implementations.
