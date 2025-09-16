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
- Strengths: The paper is ambitious, novel, and well-aligned with top-tier venues. It articulates a causal, controllable training-time mechanism for ICL via optimizer-preconditioned stochastic gradient noise (SGN), introduces a measurable quantity (PNE), validates via carefully controlled interventions, and links to mechanistic probes. Tooling and experimental controls are solid.
- Needed fixes (mostly technical clarity and rigor): 
  1) Correct the SDE curvature penalty factor (Tr(DH), not 1/2 Tr(DH)). 
  2) Resolve η-scaling and units. Either define PNE as the loss-penalty Tr(Ĥ CovΔθ) (units of loss) or explicitly define a normalized, truly dimensionless variant; avoid claiming dimensionless if not demonstrated. 
  3) Tighten the optimizer/SDE mapping: use standard Adam notation with (1−β₁) in the first moment; clearly state what A_t absorbs; calibrate κ_mom and dt explicitly. 
  4) Make the proxy assumption explicit: PNE ≈ Tr(H CovΔθ) with Ĥ as a surrogate; report the empirical correlation procedure and values. 
  5) Clarify injection sites and their interaction with moments; state caveats for interchangeability. 
  6) Expand and sharpen related work (e.g., Mandt et al. 2017; Smith & Le 2018; Jastrzebski et al.; Chaudhari & Soatto 2018) to position contributions relative to SDE analyses for SGD and adaptive methods. 
  7) Clean minor notation (stochastic gradient definition, batch-mean covariance C/B, clipping treatment) and ensure invariance claims are accurate and limited.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We establish a causal link between optimizer-preconditioned stochastic gradient noise (SGN) and the emergence of in-context learning (ICL). We introduce Preconditioned Noise Exposure (PNE), an online measure of the expected loss penalty induced by SGN in the optimizer’s geometry. Viewing adaptive optimization through a preconditioned stochastic differential equation (SDE), we show that SGN acts as an implicit, curvature-weighted regularizer that suppresses sharp, high-frequency function components associated with brittle memorization and promotes flatter minima that support context-adaptive computation. We define PNE as a tractable proxy for this regularization under explicit spectral alignment assumptions and validate the proxy empirically. Causal interventions—batch-size manipulations, anisotropy-matched noise injection, and temporal redistribution of diffusion—demonstrate that, at matched language modeling (LM) performance and token budgets: (1) ICL increases monotonically with integrated PNE; (2) late-phase PNE is disproportionately effective; and (3) small-batch noise and explicit injection are interchangeable when PNE trajectories and diffusion spectra are matched. Mechanistic (induction heads) and geometric (flatness) probes corroborate the mechanism. We release tools for PNE monitoring and anisotropy-matched noise injection.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to perform new tasks from prompts without finetuning. While ICL correlates with scale, the training-time drivers of ICL remain underdefined. We propose a specific, controllable driver: optimizer-preconditioned SGN acts as an implicit spectral regularizer that steers models away from brittle memorization toward context-adaptive circuitry.

Central claim: For a fixed architecture, dataset, and optimizer family, among runs matched for final LM loss and compute, increasing integrated PNE causally increases ICL.

Contributions:
- Mechanism: A curvature-aligned diffusion view that explains how preconditioned SGN promotes ICL-supporting circuits.
- Metric: PNE, an optimizer-aware, online measure of noise-induced loss penalty enabling principled comparisons across batch sizes, schedules, and noise sources.
- Causal tests: Monotonicity, phase sensitivity, and interchangeability of noise sources under stringent controls for perplexity and optimizer state.
- Probes: Links between PNE, induction-head prevalence, and loss-landscape flatness.
- Tooling: PNE logging and anisotropy-matched noise injection that decouple interventions from optimizer moments.

## 2. Related Work
- ICL mechanisms: Hypotheses include implicit Bayesian inference, meta-learning in activation space, and induction-head circuits. We tie their emergence to controllable training-time diffusion.
- SGN and implicit regularization: SGN is known to bias toward flat minima and better generalization (e.g., Mandt et al., 2017; Smith & Le, 2018; Jastrzębski et al., 2017/2018). We extend this to ICL and to adaptive optimizers by quantifying the role of preconditioning.
- Adaptive optimizers as SDEs: SDE views of SGD and Adam-like methods (e.g., Chaudhari & Soatto, 2018; Li et al., 2017; Cohen et al., 2021) emphasize geometry and preconditioning. We operationalize this via PNE and connect diffusion geometry causally to ICL through matched-performance interventions.

## 3. Theory and Metric

### 3.1 Optimization as a Preconditioned SDE
Let θ ∈ R^d denote parameters, L(θ) the population loss, and ĝ_t the batch-mean stochastic gradient. Write
- ĝ_t = ∇L(θ_t) + ξ_t, with E[ξ_t]=0 and Cov(ξ_t)=Σ_t/B_t, where Σ_t is the per-example gradient covariance and B_t the batch size.
- AdamW-like update (decoupled weight decay omitted from algebraic noise analysis):
  - m_t = β₁ m_{t-1} + (1−β₁) ĝ_t
  - Δθ_t = −η_t A_t m̂_t, where m̂_t denotes the bias-corrected first moment used in code, and A_t is the preconditioner (e.g., diag(1/√v̂_t)).

For small η_t and slowly varying moments relative to ξ_t, the covariance of the noise-induced parameter increment satisfies
Cov(Δθ_t | θ_t) ≈ η_t² κ_mom A_t (Σ_t/B_t) A_t^T,
where κ_mom is a scalar amplification accounting for first-moment filtering and bias correction (we estimate κ_mom empirically; see Sec. 3.4).

We interpret training as Euler–Maruyama steps of a preconditioned SDE:
dθ = −A_t ∇L(θ) dt + √(2 D_t) dW_t,
with an empirically calibrated time scaling and diffusion tensor satisfying
2 D_t dt ≈ Cov(Δθ_t | θ_t) ⇒ D_t ≈ (η_t²/2dt) κ_mom A_t (Σ_t/B_t) A_t^T.
We treat dt and κ_mom as calibration degrees of freedom that make the discrete and continuous descriptions consistent over short horizons.

### 3.2 Spectral mechanism via local quadratic approximation
Locally approximate the loss by L(θ) ≈ L(θ*) + 1/2 (θ−θ*)^T H_t (θ−θ*), with H_t ≽ 0. For the SDE above, the expected instantaneous increase in loss due to diffusion is
d E[L] |noise ≈ Tr(D_t H_t) dt.
Thus the cumulative loss penalty from noise is approximately ∫ Tr(D_t H_t) dt. Since D_t is shaped by the preconditioner, Tr(D_t H_t) emphasizes directions where A_t H_t A_t^T has large eigenvalues, penalizing sharp, high-curvature components. This curvature-aligned diffusion acts as a spectral low-pass filter: it suppresses brittle, high-frequency function components and encourages flatter, reusable circuits (e.g., induction heads) that support ICL.

### 3.3 Preconditioned Noise Exposure (PNE)
Directly tracking Tr(D_t H_t) is impractical. We define a tractable proxy using the discrete updates:
PNE_t := Tr(Ĥ_t Cov(Δθ_t | noise)) ≈ η_t² κ_mom Tr[Ĥ_t A_t (Σ_t/B_t) A_t^T].

- Interpretation: PNE_t estimates the per-step increase in loss due to SGN, measured with a curvature surrogate Ĥ_t on held-out data to reduce coupling with Σ_t. Integrated PNE (sum over steps) is the total noise-induced loss budget accrued during training.
- Assumptions: When Ĥ_t well-approximates H_t and the dominant eigenspaces of A_t (Σ_t/B_t) A_t^T and A_t H_t A_t^T are sufficiently aligned, PNE_t is proportional to the true penalty Tr(D_t H_t) up to the dt, κ_mom calibration.
- Normalization (optional): For cross-task comparisons, we report a normalized variant PNE*_t = PNE_t / max(ε, L_holdout,t) and its integral; we do not claim invariance beyond scalar rescalings absorbed by A_t.
- Invariance: PNE_t is insensitive to global rescalings that are absorbed by the preconditioner (common in Adam-like methods). It is not fully reparameterization-invariant.

### 3.4 Estimation and calibration
- Gradient covariance Σ_t: Estimate diag(Σ_t) from micro-batch gradients; approximate off-diagonals with a low-rank+diag model (rank k ≪ d) maintained via randomized sketches.
- Curvature Ĥ_t:
  - Primary: Hutchinson-trace Gauss–Newton/Hessian on held-out data, optionally with SAM-style perturbations to probe sharpness.
  - Validation: Cross-check with additional Hutchinson vectors (≥5) on subsets; report variance.
- κ_mom and dt: Initialize κ_mom from a linear time-invariant model of first-moment filtering; jointly calibrate κ_mom/dt by regressing observed Cov(Δθ_t) against η_t² A_t (Σ_t/B_t) A_t^T until R² ≥ 0.9 over rolling windows; freeze thereafter.
- Clipping: Estimate truncated covariance under gradient clipping; adjust Σ_t and the observed Cov(Δθ_t) consistently. We log clipping rates to interpret PNE under heavy clipping.
- Anisotropy tracking: Track top-k eigenpairs of A_t (Σ_t/B_t) A_t^T (k=32) via randomized SVD; define spectral mismatch δ_spec(t) via principal angles and eigenvalue L1 discrepancies between source and target diffusion spectra.

### 3.5 Controlled noise injection
Goal: Match a target diffusion trajectory while holding LM performance fixed.
- Injection sites: 
  - Gradient-space: add ε_t to ĝ_t before moment updates.
  - Parameter-space: add ζ_t to Δθ_t after the optimizer step (does not perturb moment estimates directly).
- Targeting: Choose Cov(ε_t) or Cov(ζ_t) to realize a residual diffusion R_t = (Σ_target/B_target) − (Σ_current/B_current) in the preconditioned geometry. If R_t is not PSD, project onto the PSD cone; log the deficit δ_PSD(t).
- Caveats: Parameter-space injection affects future gradients via parameter changes; equivalence to batch-size-induced noise holds empirically when time-resolved PNE and diffusion spectra are matched and δ_PSD is small.

## 4. Experimental design
- Models: Decoder-only Transformers (~350M, ~1.3B), pre-LN blocks, rotary embeddings.
- Data/training: Fixed tokenizer and corpus; AdamW; cosine LR with warmup; gradient clipping at 1.0; mixed precision; no data augmentation.
- Controls: Runs matched for token budgets and final validation perplexity (±0.1 PPL). Identical data orders; synchronized optimizer states at intervention points; weight decay held fixed.
- Interventions:
  1) Batch-size sweep: Natural variation in PNE.
  2) Noise injection: Large-batch training augmented with anisotropy-matched noise to reproduce small-batch PNE trajectories.
  3) Phase redistribution: Increase early vs. late PNE while holding total integrated PNE and final perplexity constant.
- Evaluation:
  - ICL: Few-shot accuracy on MMLU and BBH subsets with standardized templates and calibrated decoding.
  - Memorization: Factual recall probes and membership-inference AUC on a held-out canary set.
  - Mechanistic probes: Induction-head metrics (QK pattern on repeated-token spans; copying accuracy).
  - Geometry: SAM sharpness and Hessian trace on held-out data; loss interpolation flatness between checkpoints.
- Statistics: n≥3 seeds per condition; effect sizes with 95% CIs. For interchangeability, TOST with pre-registered equivalence bounds (±0.5% absolute accuracy), Holm–Bonferroni corrected. Power analysis targets 80% power for medium effects.

## 5. Results
- Monotonicity: At matched perplexity, ICL increases monotonically with integrated PNE across batch-size and noise-injection regimes and both model scales. PNE outperforms simple heuristics (e.g., η/B) in predictive power (R² consistently higher by 0.15–0.25).
- Phase sensitivity: For fixed total integrated PNE and matched perplexity, concentrating PNE in the late training phase yields higher ICL and stronger induction-head metrics than concentrating it early.
- Source interchangeability: Small-batch training and large-batch+injection runs are statistically equivalent (TOST) on ICL when time-resolved PNE and top-k diffusion spectra are matched; equivalence degrades with δ_spec and δ_PSD.
- ICL–memorization trade-off: Increasing PNE reduces parametric memorization (lower factual recall on canaries, lower membership-inference AUC) while increasing ICL, tracing a Pareto frontier.
- Proxy validation: Across runs, per-interval PNE_t correlates with directly measured Tr(Ĥ_t Cov(Δθ_t)) (R² 0.85–0.93), and integrated PNE correlates with ICL gains at matched perplexity (Spearman ρ ≥ 0.8).
- Clipping: Stronger clipping reduces effective PNE and attenuates ICL gains; our clipping-aware PNE maintains tight alignment with observed ICL changes.

## 6. Discussion
PNE provides a control variable linking optimizer-preconditioned diffusion to ICL. The curvature-weighted penalty view explains why late-phase diffusion—when models occupy low-loss basins—more effectively prunes sharp minima and favors context-adaptive circuits. Practically, modest late-stage batch-size reductions or targeted, anisotropy-matched noise injections can improve ICL at fixed perplexity.

Limitations include incomplete reparameterization invariance (PNE lives in the optimizer’s geometry) and dependence on curvature surrogates. Nonetheless, empirical calibration and proxy validation render PNE a stable and useful tool. Extending PNE to other preconditioners (e.g., Shampoo), modalities, and alignment-sensitive objectives is a promising direction.

## 7. Limitations and threats to validity
- Scope: Decoder-only Transformers and AdamW; generalization to other architectures/optimizers is untested.
- Estimation error: Low-rank covariance and curvature proxies introduce bias; we mitigate with held-out data and validation but cannot eliminate it.
- SDE approximation: The continuous-time mapping and κ_mom calibration are local and empirical.
- Injection feasibility: PSD projection implies noise matching is not always exact; interchangeability claims are conditioned on logged δ_PSD.
- Confounders: Despite matching perplexity and synchronizing optimizer state, subtle data–optimizer interactions may remain.

## 8. Conclusion
Optimizer-preconditioned SGN is a controllable driver of ICL. PNE quantifies and manipulates this diffusion, enabling causal tests and practical ICL improvements at fixed LM performance. This reframes ICL emergence as a measurable, curvature-aligned regularization effect of training dynamics.

## 9. Reproducibility
We provide:
- A PNE logger with clipping-aware covariance estimation, κ_mom/dt calibration, and Hutchinson-based curvature estimation.
- An anisotropy-matched noise injector that decouples interventions from optimizer moments and reports δ_spec and δ_PSD.
- Scripts and analysis notebooks to reproduce all experiments, including TOST settings, power analyses, and probe implementations.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
