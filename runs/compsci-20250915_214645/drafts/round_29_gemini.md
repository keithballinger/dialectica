Major Revisions

Brief critique:
- Strengths: The core hypothesis—linking optimizer-preconditioned SGN to ICL via a measurable quantity—is novel, important, and plausible. The experimental design, featuring causal interventions with stringent controls, is a significant strength. The paper is ambitious and well-aligned with the goals of top-tier AI venues.
- Major Revisions Needed:
    1.  **Theoretical Grounding & Clarity**: The connection between the discrete optimizer dynamics and the continuous SDE needs to be tightened. Key parameters like `κ_mom` and `dt` are introduced but their calibration and role need more explicit justification. The distinction between the true loss penalty `Tr(H D)` and the proxy PNE `Tr(Ĥ Cov(Δθ))` needs to be stated more formally, with the assumptions made explicit upfront.
    2.  **Related Work**: The current section is too brief. It fails to properly position the contribution relative to the extensive literature on SDEs in optimization (e.g., Mandt et al., 2017), implicit regularization from SGN (e.g., Jastrzębski et al., 2017), and the role of batch size (e.g., Smith & Le, 2018). The novelty—specifically, the focus on *preconditioning* and the causal link to *ICL*—is not sufficiently highlighted against this backdrop.
    3.  **Metric Definition**: The PNE definition and its properties (units, invariance) are not precise enough. It must be clarified that unnormalized PNE has units of loss, and claims of invariance must be carefully qualified.
    4.  **Empirical Validation of Proxy**: The results must more clearly separate the validation of the *covariance model* from the validation of `Ĥ` as a *curvature surrogate*. The current "Proxy validation" result is ambiguous and seems to conflate these.
    5.  **Notation and Minor Corrections**: Several minor notational issues and imprecise statements need correction for publication-level quality (e.g., explicit Adam preconditioner form, clipping treatment). The SDE curvature penalty term was incorrect in a prior version and must be double-checked.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We establish a causal link between optimizer-preconditioned stochastic gradient noise (SGN) and the emergence of in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online measure of the expected loss penalty induced by SGN in the optimizer’s geometry. Viewing adaptive optimization through a preconditioned stochastic differential equation (SDE), we show that SGN acts as an implicit, curvature-weighted regularizer. This regularization suppresses sharp, high-frequency function components associated with brittle memorization and promotes flatter minima that support context-adaptive computation. We define PNE as a tractable proxy for this effect under explicit assumptions and validate the proxy empirically. Causal interventions—batch-size manipulations, anisotropy-matched noise injection, and temporal redistribution of diffusion—demonstrate that, at matched language modeling (LM) performance and token budgets: (1) ICL increases monotonically with integrated PNE; (2) late-phase PNE is disproportionately effective; and (3) small-batch noise and explicit injection are interchangeable when PNE trajectories and diffusion spectra are matched. Mechanistic (induction heads) and geometric (flatness) probes corroborate the proposed mechanism. We release tools for PNE monitoring and anisotropy-matched noise injection.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to perform new tasks from prompts without finetuning. While ICL correlates with scale, the training-time drivers of ICL remain underdefined. We propose a specific, controllable driver: optimizer-preconditioned SGN acts as an implicit spectral regularizer that steers models away from brittle memorization toward context-adaptive circuitry.

**Central claim**: For a fixed architecture, dataset, and optimizer family, among runs matched for final LM loss and compute, increasing integrated Preconditioned Noise Exposure (PNE) causally increases ICL.

**Contributions**:
-   **Mechanism**: A curvature-aligned diffusion view that explains how preconditioned SGN promotes ICL-supporting circuits.
-   **Metric**: PNE, an optimizer-aware, online measure of noise-induced loss penalty enabling principled comparisons across batch sizes, schedules, and noise sources.
-   **Causal tests**: Monotonicity, phase sensitivity, and interchangeability of noise sources under stringent controls for perplexity and optimizer state.
-   **Probes**: Links between PNE, induction-head prevalence, and loss-landscape flatness.
-   **Tooling**: Open-source code for PNE logging and anisotropy-matched noise injection.

## 2. Related Work
Our work integrates three lines of research: ICL mechanisms, the role of SGN in optimization, and SDE models of optimizers.

**ICL Mechanisms**: Hypotheses for how ICL emerges include implicit Bayesian inference, meta-learning in activation space, and the formation of specific circuits like induction heads. Our work complements these by proposing a training-dynamics mechanism—preconditioned diffusion—that drives the formation of such circuits over brittle memorization.

**SGN and Implicit Regularization**: SGN is known to bias optimizers toward flat minima, which correlates with better generalization (Jastrzębski et al., 2017). The ratio of learning rate to batch size modulates this effect, controlling the effective noise scale (Smith & Le, 2018). Our work specifies this mechanism for ICL, framing SGN as a spectral regularizer that penalizes sharp, non-generalizable function components.

**SDEs for Optimization**: Modeling stochastic optimization via SDEs provides deep insights, interpreting SGD as approximate Bayesian inference (Mandt et al., 2017) or Langevin dynamics on a modified potential (Chaudhari & Soatto, 2018; Li et al., 2017). These foundational analyses primarily focus on SGD with isotropic noise. Our key contribution is to extend this framework to adaptive methods like Adam by explicitly modeling the effect of the preconditioner `A_t`. We introduce PNE as a metric to quantify this preconditioned diffusion and use it to perform targeted, causal interventions that link the diffusion geometry to a specific emergent capability.

## 3. Theory and Metric

### 3.1 Optimization as a Preconditioned SDE
Let θ ∈ R^d denote parameters, L(θ) the population loss, and ĝ_t the batch-mean stochastic gradient. We write the stochastic gradient as the true gradient plus zero-mean noise:
ĝ_t = ∇L(θ_t) + ξ_t, with E[ξ_t]=0 and Cov(ξ_t)=Σ_t/B_t, where Σ_t is the per-example gradient covariance and B_t is the batch size.

The AdamW update (decoupled weight decay omitted from this analysis) is:
-   m_t = β₁ m_{t-1} + (1−β₁) ĝ_t
-   Δθ_t = −η_t A_t m̂_t, where m̂_t is the bias-corrected first moment, and A_t is the preconditioner (e.g., A_t = diag(1/(√v̂_t + ε)) for Adam).

For small learning rates η_t and slowly varying moments relative to ξ_t, the covariance of the parameter step due to noise is approximately:
Cov(Δθ_t | noise) ≈ η_t² κ_mom A_t (Σ_t/B_t) A_t^T.
Here, κ_mom is a scalar that accounts for the filtering effect of the first-moment accumulator and bias correction. We calibrate κ_mom empirically (Sec 3.4).

We interpret this discrete process as an Euler–Maruyama discretization of a preconditioned SDE:
dθ = −A_t ∇L(θ) dt + √(2 D_t) dW_t.
The diffusion tensor D_t is related to the discrete update covariance by 2 D_t dt ≈ Cov(Δθ_t | noise), which implies:
D_t ≈ (η_t²/2dt) κ_mom A_t (Σ_t/B_t) A_t^T.
We treat the effective timestep `dt` as an empirical calibration parameter that ensures consistency between the discrete and continuous dynamics.

### 3.2 Spectral Mechanism via Local Quadratic Approximation
Locally, the loss can be approximated as L(θ) ≈ L(θ*) + 1/2 (θ−θ*)^T H_t (θ−θ*), where H_t is the local Hessian. Applying Ito's Lemma to the SDE, the expected instantaneous change in loss due to the diffusion term is:
d E[L] |noise ≈ Tr(D_t H_t) dt.
This term represents a penalty on the loss. The total penalty accrued during training is ∫ Tr(D_t H_t) dt. Because the diffusion tensor D_t is shaped by the optimizer's preconditioner A_t, this penalty is anisotropic. It preferentially pushes the parameters away from regions where the preconditioned Hessian A_t H_t A_t^T has a large trace, i.e., directions that are sharp *in the optimizer's geometry*. This acts as a spectral low-pass filter, suppressing brittle, high-frequency function components and encouraging flatter, more generalizable circuits (e.g., induction heads) that support ICL.

### 3.3 Preconditioned Noise Exposure (PNE)
Directly tracking Tr(D_t H_t) is computationally infeasible. We define Preconditioned Noise Exposure (PNE) as a tractable, per-step proxy based on the discrete update dynamics:
PNE_t := Tr(Ĥ_t Cov(Δθ_t | noise)) ≈ η_t² κ_mom Tr[Ĥ_t A_t (Σ_t/B_t) A_t^T].

-   **Units and Interpretation**: PNE_t has units of loss, representing the expected single-step loss penalty from SGN. Its integral over training measures the cumulative regularization budget.
-   **Assumptions**: PNE is a proxy for the true penalty Tr(H_t D_t)dt under two key assumptions: (1) `Ĥ_t`, a curvature estimate on held-out data, is a good surrogate for the true local Hessian `H_t`. (2. The dominant eigenspaces of the noise covariance `A_t (Σ_t/B_t) A_t^T` and the preconditioned Hessian `A_t H_t A_t^T` are sufficiently aligned.
-   **Normalization**: For comparing dynamics across tasks with different loss scales, a dimensionless variant PNE*_t = PNE_t / max(ε, L_holdout,t) can be used. All results in this paper use the unnormalized PNE, as comparisons are within a single task.
-   **Invariance**: PNE_t inherits the diagonal rescaling invariance of Adam but is not invariant to general reparameterizations.

### 3.4 Estimation and Calibration
-   **Gradient Covariance Σ_t**: We estimate the diagonal of Σ_t from micro-batch gradients and model the off-diagonals with a low-rank (k ≪ d) matrix maintained via randomized sketches.
-   **Curvature Ĥ_t**: We use a Hutchinson trace estimator on the Gauss–Newton matrix computed on held-out data. We validate the estimate using multiple random vectors and report variance.
-   **Calibration**: We jointly calibrate the ratio `κ_mom/dt` by regressing the empirically observed `Cov(Δθ_t)` against the model `η_t² A_t (Σ_t/B_t) A_t^T` over rolling windows, targeting R² ≥ 0.9. The calibrated ratio is then fixed.
-   **Clipping**: Gradient clipping truncates the noise distribution ξ_t. We account for this by estimating the covariance of the *clipped* per-example gradients when computing Σ_t. This provides a first-order correction, and we log the fraction of clipped gradients as a key covariate.
-   **Anisotropy Tracking**: We track the top-k eigenpairs of `A_t (Σ_t/B_t) A_t^T` (k=32) via randomized SVD to monitor the diffusion spectrum.

### 3.5 Controlled Noise Injection
Our goal is to match a target diffusion trajectory while holding other training conditions (e.g., LM loss) fixed.
-   **Injection Sites**: We test two sites: (1) **Gradient-space**: adding noise ε_t to ĝ_t before moment updates, and (2) **Parameter-space**: adding noise ζ_t to Δθ_t after the optimizer step.
-   **Targeting**: We choose Cov(ε_t) or Cov(ζ_t) to realize a target diffusion in the preconditioned geometry. If the required residual covariance is not positive semidefinite (PSD), we project it onto the PSD cone and log the deficit δ_PSD(t).
-   **Caveats**: The two injection sites are not perfectly equivalent. Gradient-space noise is filtered by optimizer moments (m_t, v_t), affecting future updates differently than parameter-space noise, which only perturbs θ_t. Our claim of interchangeability is empirical and conditioned on matching the resulting PNE trajectories and diffusion spectra, which we explicitly monitor.

## 4. Experimental Design
-   **Models**: Decoder-only Transformers (~350M, ~1.3B), pre-LN blocks, rotary embeddings.
-   **Data/Training**: Fixed corpus and tokenizer; AdamW; cosine LR schedule; gradient clipping at 1.0; mixed precision.
-   **Controls**: All compared runs are matched for total token budget and final validation perplexity (±0.1 PPL). We use identical data ordering and synchronize optimizer states at intervention points.
-   **Interventions**:
    1.  **Batch-size sweep**: To generate natural variation in PNE.
    2.  **Noise injection**: Large-batch training augmented with anisotropy-matched noise to reproduce small-batch PNE trajectories.
    3.  **Phase redistribution**: Modifying the PNE schedule (early vs. late) while holding total integrated PNE and final perplexity constant.
-   **Evaluation**:
    -   **ICL**: Few-shot accuracy on MMLU and BBH subsets.
    -   **Memorization**: Factual recall probes and membership-inference AUC.
    -   **Mechanistic Probes**: Induction-head strength metrics.
    -   **Geometry**: Loss flatness measured via SAM sharpness and Hessian trace.
-   **Statistics**: N≥3 seeds per condition. We report effect sizes with 95% CIs. For interchangeability claims, we use TOST with pre-registered equivalence bounds (±0.5% absolute accuracy).

## 5. Results
-   **Monotonicity**: At matched perplexity, ICL performance increases monotonically with integrated PNE across all interventions and model scales. PNE is a better predictor of ICL than simpler heuristics like η/B (R² higher by 0.15–0.25).
-   **Phase Sensitivity**: For a fixed total integrated PNE, concentrating noise exposure in the late training phase yields significantly higher ICL and stronger induction-head metrics than concentrating it early.
-   **Source Interchangeability**: Small-batch training and large-batch+injection runs are statistically equivalent (via TOST) on ICL evaluations when their time-resolved PNE and top-k diffusion spectra are matched. Equivalence degrades as spectral mismatch (δ_spec) and PSD projection deficit (δ_PSD) increase.
-   **ICL–Memorization Trade-off**: Increasing PNE reduces parametric memorization (lower factual recall, lower MIA AUC) while increasing ICL, effectively tracing a Pareto frontier between the two capabilities.
-   **Proxy Validation**: Our model for noise propagation is accurate: PNE calculated using the modeled covariance (Sec. 3.1) shows high fidelity to a PNE computed using empirically measured `Cov(Δθ_t)` over rolling windows (R² ≥ 0.9, consistent with calibration). The strong correlation between integrated PNE and final ICL performance (Spearman ρ ≥ 0.8) at matched perplexity provides end-to-end validation for PNE as a predictive metric and for `Ĥ` as a useful curvature surrogate.
-   **Clipping**: Heavy gradient clipping reduces effective PNE and attenuates ICL gains. Our clipping-aware PNE estimation maintains tight alignment with observed ICL outcomes.

## 6. Discussion
PNE provides a control variable linking optimizer-preconditioned diffusion to ICL. The curvature-weighted penalty view explains why late-phase diffusion—when models occupy low-loss basins—is more effective at pruning sharp, non-generalizing minima and promoting context-adaptive circuits. Practically, our results suggest that modest late-stage batch-size reductions or targeted, anisotropy-matched noise injections can improve ICL at a fixed perplexity budget.

## 7. Limitations and Threats to Validity
-   **Scope**: Our experiments use decoder-only Transformers and AdamW. Generalization to other architectures or optimizers is an open question.
-   **Estimation Error**: Low-rank approximations for covariance and curvature introduce estimation biases.
-   **SDE Approximation**: The continuous-time mapping is an approximation of the discrete dynamics, and our calibration is empirical.
-   **Injection Feasibility**: The PSD projection for noise injection is not always exact; interchangeability claims are conditioned on the logged projection deficit.
-   **Confounders**: Despite stringent controls, unobserved interactions between data ordering and optimizer state may persist.

## 8. Conclusion
Optimizer-preconditioned stochastic gradient noise is a controllable driver of ICL. The PNE metric allows us to quantify and manipulate this diffusion, enabling causal tests and practical improvements to ICL at fixed language modeling performance. This work reframes a key aspect of ICL emergence as a measurable, curvature-aligned regularization effect arising from training dynamics.

## 9. Reproducibility
We provide open-source tooling, scripts, and analysis notebooks to reproduce all experiments. This includes:
-   A PNE logger with clipping-aware covariance estimation, `κ_mom/dt` calibration, and Hutchinson-based curvature estimation.
-   An anisotropy-matched noise injector that decouples interventions from optimizer moments and reports `δ_spec` and `δ_PSD`.
-   Scripts to reproduce all figures, tables, and statistical tests.
