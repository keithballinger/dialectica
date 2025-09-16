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
- The central hypothesis connecting optimizer-preconditioned SGN to ICL is highly novel and the proposed metric (PNE) provides a principled, testable framework. The experimental design, based on causal interventions, is rigorous and appropriate for a top-tier venue.
- The SDE formalism and the PNE definition, while well-motivated, require more precise mathematical statement. The core definitions should be presented as formal equations before their components are discussed. The `dt ≈ ηt` mapping should be explicitly framed as a heuristic for defining a path integral.
- The mechanism linking diffusion to ICL via spectral filtering can be articulated more sharply. Explicitly state the hypothesis that sharp directions correspond to high-frequency function components associated with memorization, which are then suppressed by curvature-aligned noise.
- The implementation details, particularly the choice of curvature proxy and the noise injection procedure, need to be clarified for the main experiments. State which proxy was used for primary results and justify its use (e.g., efficiency), while noting its validation against more accurate methods. Rephrase "optimizer-state hygiene" with more formal terminology.
- The consequence of the PSD projection for noise injection (i.e., that the intervention is a conservative lower bound on the target diffusion) should be briefly noted as it impacts the interpretation of results.
- While the draft format omits figures, the textual descriptions in the Results section could be made more concrete by referencing the statistical tests used to support claims (e.g., TOST for interchangeability).

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We establish a causal link between optimizer-preconditioned stochastic gradient noise (SGN) and the emergence of in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online, curvature-normalized metric of the effective diffusion induced by an adaptive optimizer. Grounded in a stochastic differential equation (SDE) model of training, PNE quantifies how SGN explores the loss landscape. We hypothesize that integrated PNE acts as an implicit spectral regularizer, suppressing sharp, high-frequency solutions tied to brittle memorization and promoting flatter minima that support context-adaptive computation. We test this via causal interventions. At matched language modeling (LM) performance and token budgets, we find that: (1) ICL increases monotonically with integrated PNE; (2) late-phase PNE has a disproportionately large impact on ICL; and (3) the source of noise (small batch vs. explicit injection) is interchangeable when PNE trajectories and spectral anisotropy are matched. These findings are supported by mechanistic probes (induction-head prevalence) and geometric analysis (flatness). We release code for PNE monitoring and anisotropy-matched noise injection.

## 1. Introduction
In-context learning (ICL) allows large language models (LLMs) to perform new tasks from prompts without explicit finetuning. While ICL capability correlates with model scale, the training dynamics that foster it are not well understood. We propose and test a specific mechanism: optimizer-preconditioned SGN acts as an implicit regularizer that steers the model away from simple memorization and towards the development of general, context-adaptive circuits.

**Central Claim:** For a fixed architecture, dataset, and optimizer family, and among training runs matched for final language modeling loss and compute budget, increasing integrated Preconditioned Noise Exposure (PNE) causally increases ICL performance.

**Contributions:**
- **Mechanism:** We frame preconditioned SGN as a spectral filter that suppresses sharp, high-frequency function components, thereby promoting the reusable circuits underpinning ICL over rote memorization.
- **Metric:** We define PNE, an online, optimizer-aware, and curvature-referenced measure of integrated diffusion, enabling principled comparisons across different batch sizes, schedules, and noise sources that go beyond simple `η/B` heuristics.
- **Causal Tests:** We use controlled interventions to demonstrate PNE's monotonic effect on ICL, its phase sensitivity, and the interchangeability of noise sources, all under strict controls for perplexity and optimizer states.
- **Probes:** We link ICL gains to increased induction-head prevalence and flatter loss basins, providing mechanistic and geometric evidence.
- **Tools:** We release robust utilities for PNE logging and anisotropy-matched noise injection that avoid contaminating optimizer moments.

## 2. Related Work
This work integrates three lines of research:
- **ICL Mechanisms:** Prior work posits ICL arises from implicit Bayesian inference, meta-optimization in activation space, or the formation of specific circuits like induction heads. We provide a mechanism rooted in training dynamics that promotes such circuits.
- **SGN and Implicit Regularization:** The role of SGN in finding flat minima and improving generalization is well-established for SGD. We extend this framework to ICL in the context of adaptive optimizers.
- **Adaptive Optimizer Dynamics:** SDE analyses of optimizers like Adam(W) have explored their preconditioning effects. We leverage this SDE view to define a concrete, measurable quantity (PNE) that links preconditioning to a specific emergent capability (ICL).

Our primary contribution is to unify these threads by formalizing optimizer-preconditioned noise via PNE and using it to causally isolate a training-dynamics lever for ICL at fixed model performance.

## 3. Theory and Metric

### 3.1 SDE Model of Preconditioned Optimization
An adaptive optimizer update with momentum `β₁` and preconditioner `Aₜ` can be written as:
`uₜ = β₁ uₜ₋₁ + gₜ`
`Δθₜ = -ηₜ Aₜ uₜ`
where `gₜ = ∇L(θₜ) + ξₜ` is the stochastic gradient, with `E[ξₜ]=0` and `Cov(ξₜ)=Cₜ/Bₜ`. For AdamW, `Aₜ` is approximately `diag((vₜ + δ)⁻¹/²)` where `vₜ` is the second moment estimate.

Assuming the optimizer moments (`uₜ`, `vₜ`) evolve slowly relative to the noise, the per-step update noise has covariance:
`Cov(Δθₜ | θₜ) ≈ ηₜ² κ_mom Aₜ (Cₜ/Bₜ) Aₜᵀ`
where `κ_mom ≥ 1` is an empirically calibrated factor accounting for variance amplification from momentum. We model the discrete optimization trajectory as an Euler–Maruyama discretization of an SDE, where the learning rate `ηₜ` serves as the time-step `dt`, mapping iterations to a path length in parameter space. The instantaneous diffusion tensor is `Dₜ ∝ ηₜ κ_mom Aₜ (Cₜ/Bₜ) Aₜᵀ`.

### 3.2 Mechanism: Curvature-Aligned Diffusion as a Spectral Filter
Preconditioned SGN induces curvature-aligned diffusion. The effective noise is greatest in directions where the preconditioned Hessian `Aₜ Hₜ Aₜᵀ` has large eigenvalues (i.e., sharp directions in the optimizer's geometry). This process acts as a spectral low-pass filter on the loss function: it preferentially perturbs parameters along sharp eigendirections, penalizing high-frequency components of the loss function often associated with rote memorization. This encourages the optimization to settle in broader, flatter basins, which we hypothesize support the robust, reusable circuits (e.g., induction heads) necessary for ICL.

### 3.3 Preconditioned Noise Exposure (PNE)
We define PNE to quantify the diffusion budget relative to the local curvature, both measured in the optimizer's preconditioned geometry. The per-step PNE is:
`PNEₜ = (ηₜ κ_mom Tr(Aₜ (Cₜ/Bₜ) Aₜᵀ)) / max(ε, Tr(Aₜ Ĥₜ Aₜᵀ))` (1)
Total integrated exposure is `PNE = Σₜ PNEₜ`.

- **Numerator:** The trace of the per-step diffusion in the preconditioned space, with units of `parameter² / learning-time`.
- **Denominator:** A reference curvature, also traced in the preconditioned space. `Ĥₜ` is a curvature proxy (e.g., Hessian, GGN) estimated on held-out data to prevent degeneracy that would occur if `Ĥₜ` were correlated with the gradient covariance `Cₜ`.
- **`ε`:** A small constant to ensure numerical stability. We confirm results are insensitive to `ε` over several orders of magnitude.
- **Interpretation:** PNEₜ is a dimensionless ratio measuring the step's diffusion budget relative to the local curvature.

### 3.4 Estimation and Calibration
- **Covariance `Cₜ`:** We estimate `diag(Cₜ)` from per-micro-batch gradient variance. Off-diagonals are modeled using low-rank plus diagonal approximations.
- **Curvature `Ĥₜ`:** Our primary results use an efficient SAM-based sharpness proxy (loss increase after one gradient ascent step on a held-out batch). We validate its correlation with a more expensive Hutchinson trace of the Hessian/GGN on subsets.
- **Momentum Factor `κ_mom`:** Calibrated via online regression of observed parameter update variance against the theoretical noise term. `κ_mom` is fixed once its estimate stabilizes (`R² > 0.9`).
- **Anisotropy:** We track the top-`k` eigenpairs of the diffusion tensor `Dₜ` via randomized SVD to compare spectral properties across runs.

### 3.5 Controlled Noise Injection
To isolate the effect of PNE, we inject synthetic noise to match a target diffusion trajectory while using a different batch size.
- **Injection:** We add noise `εₜ` in gradient space or `ζₜ` in parameter space, with covariance chosen to match a target residual `Rₜ = C_target/B_target − C_current/B_current`.
- **Decoupling from Optimizer Moments:** The injected noise is applied *after* the optimizer's internal moments (`uₜ`, `vₜ`) are computed and updated. This ensures that interventions on noise do not confound the preconditioning `Aₜ` or momentum `uₜ`, isolating the effect of SGN.
- **PSD Constraint:** If the residual covariance `Rₜ` is not positive semi-definite (PSD), we project it onto the PSD cone by truncating negative eigenvalues. This makes the intervention a conservative lower bound on the target diffusion. We log the resulting spectral deficit.

## 4. Experimental Design
- **Setup:** Decoder-only Transformers (~350M, ~1.3B) on a fixed pretraining dataset. AdamW optimizer with a cosine learning rate schedule and gradient clipping at 1.0.
- **Controls:** All comparisons are made between runs with matched token budgets and final validation perplexity (±0.1 PPL). We use identical data ordering seeds and synchronize optimizer states at intervention points.
- **Interventions:**
    1. **Batch Size:** Vary effective batch size to naturally modulate PNE.
    2. **Noise Injection:** Match the PNE trajectory of a small-batch run using a large-batch run with injected, anisotropy-matched noise.
    3. **Phase Shift:** Redistribute PNE to be higher in early vs. late training, while holding total integrated PNE and final perplexity constant.
- **Evaluation:**
    - **ICL:** Few-shot accuracy on MMLU and BBH subsets using standardized templates.
    - **Memorization:** Factual recall probes and membership inference attacks as negative controls.
    - **Mechanistic Probes:** Induction-head strength metrics.
    - **Geometric Probes:** SAM sharpness on held-out data.
- **Statistics:** Preregistered analyses, `n≥3` seeds per condition, effect sizes with 95% CIs, and two one-sided tests (TOST) for interchangeability claims, with Holm–Bonferroni correction.

## 5. Results
- **Monotonicity:** At matched perplexity, ICL performance increases monotonically with integrated PNE. This holds across both batch size variation and noise injection regimes, for both 350M and 1.3B models. PNE is a better predictor than simpler `η/B` heuristics.
- **Phase Sensitivity:** For a fixed total PNE and final perplexity, concentrating exposure in the late phase of training yields significantly higher ICL performance than concentrating it early. This gain correlates with stronger induction-head formation.
- **Source Interchangeability:** Small-batch runs and large-batch runs with injected noise are statistically indistinguishable on ICL benchmarks (via TOST) when their time-resolved PNE and low-rank spectral anisotropy are matched. Mismatched anisotropy breaks this equivalence.
- **ICL-Memorization Trade-off:** Increasing PNE consistently reduces parametric memorization (lower factual recall and membership inference AUC) while improving ICL, effectively tracing a Pareto frontier between these two capabilities.
- **Clipping Effects:** Excessive gradient clipping reduces effective PNE and attenuates ICL gains. Accounting for clipping in the PNE calculation tightens all observed correlations.

## 6. Discussion
PNE provides a quantitative framework for understanding SGN as a controllable driver of ICL. Our results suggest that late-stage, curvature-aligned exploration is crucial for converting general pattern-matching abilities into robust, context-adaptive computation. Practically, a late-stage reduction in batch size or a targeted, anisotropy-matched noise injection can be used to improve ICL at a fixed perplexity budget. PNE serves as the control variable for this process.

While PNE is defined in the optimizer's geometry and is not fully reparameterization-invariant, it is stable under common scaling transformations that are absorbed by the preconditioner `Aₜ`.

## 7. Limitations and Threats to Validity
- **Scope:** Findings are demonstrated for decoder-only Transformers with AdamW. Generalization to other architectures, optimizers (e.g., Shampoo), and modalities requires further study.
- **Estimation Bias:** Our covariance and curvature proxies are efficient but introduce estimation error. We mitigate this with held-out data and validation against more accurate estimators on subsets, but bias may remain.
- **SDE Approximation:** The continuous-time model is a local approximation. The `dt ≈ ηₜ` mapping and `κ_mom` calibration are empirically motivated choices, not formal derivations.
- **Injection Feasibility:** The PSD constraint means we cannot perfectly match any arbitrary target diffusion. We report deficits and find our conclusions robust within the achievable range.

## 8. Conclusion
Optimizer-preconditioned stochastic gradient noise is not merely a side effect of mini-batch training but a key, controllable ingredient for inducing in-context learning. By measuring and manipulating this noise via the Preconditioned Noise Exposure (PNE) framework, we can predictably steer training dynamics to enhance emergent capabilities, reframing ICL as a direct outcome of a measurable regularization process.

## 9. Reproducibility
We release a library for PNE logging (including post-clip accounting and `κ_mom` calibration) and a module for anisotropy-matched noise injection that correctly decouples noise from optimizer moments. All scripts, configurations, and analysis notebooks are provided to reproduce our findings.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
