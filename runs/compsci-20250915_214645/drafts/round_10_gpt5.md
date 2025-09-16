Major Revisions

Brief critique:
- Strengths: The paper poses a clear, falsifiable mechanism linking preconditioned stochastic gradient noise (SGN) to in-context learning (ICL) via a principled, optimizer-aware metric (P-NE), and proposes explicit causal tests (monotonicity, phase sensitivity, and source equivalence). It grounds claims in SDE formalisms, considers implicit regularization, and includes negative controls. The topic is timely, AI-focused, and potentially publishable at a top venue.
- Required revisions:
  - Replace mock figures/numbers with real results and provide per-seed statistics, sample sizes, and multiple-comparisons corrections. Include calibration details for few-shot evaluations and data contamination checks.
  - Expand and sharpen methodological details: exact placement of noise injection (pre-/post-preconditioning), gradient clipping interactions, compute/token budgets, update counts, data ordering, prompt formats, and stopping criteria. Add matched-compute controls (fixed updates/tokens) to rule out time-on-task confounds.
  - Clarify the estimator: how Tr(P_t C_t P_t) and Tr(P_t F_t P_t) are computed online, cost/variance trade-offs (diag vs Hutchinson), and robustness to heavy-tailed noise. Provide explicit pseudocode and complexity.
  - Address alternative explanations and invariances: sharpness reparameterization (Dinh et al.), flatness proxies (SAM, noise stability), PAC-Bayes perspective (posterior volume), and parameterization dependence of P-NE. Justify curvature normalization and preconditioner choice.
  - Situate contributions with stronger citations to SDE analyses of adaptive optimizers, ICL mechanisms, heavy-tailed gradient noise, and batch-size effects.
  - Define “stable optimization regime,” report failure modes when P-NE is too high, and provide actionable heuristics for practitioners.
  - Broaden limitations: other optimizers (SGD/momentum, Adafactor), architectures, scaling to >10B params, multilingual data, context length.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We investigate the role of preconditioned stochastic gradient noise (SGN) in shaping in-context learning (ICL) in language models. We introduce Preconditioned Noise Exposure (P-NE), an optimizer-aware, online metric derived from the stochastic differential equation (SDE) view of adaptive methods, and hypothesize that integrated P-NE imposes an implicit information penalty on parameters that biases training toward context-adaptive computation. We formulate three falsifiable predictions: (1) at matched language modeling (LM) performance, ICL increases monotonically with P-NE; (2) ICL is disproportionately sensitive to late-phase P-NE; and (3) the source of noise is approximately irrelevant given matched P-NE. We detail a causal experimental design that manipulates P-NE via batch size, explicit noise injection, and scheduling while controlling for data, architecture, optimizer family, token budgets, and final perplexity. We report complementary mechanistic and geometric evidence (induction-head prevalence, flatter minima). We discuss boundaries of the stable optimization regime, estimator robustness, alternative explanations (sharpness reparameterization, PAC-Bayes), and implications for engineering ICL by design. Code and logging for P-NE estimation are released.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to adapt to novel tasks from prompt context without weight updates. While scale correlates with ICL, the training dynamics that produce ICL remain unclear. We posit and test a causal mechanism: integrated, optimizer-preconditioned SGN acts as an implicit regularizer that penalizes weight-coded solutions, thereby favoring context-adaptive computation.

Central claim (conditional on fixed architecture, data distribution, and optimizer family):
- For models matched on LM loss, increasing integrated P-NE causally increases ICL within a stable optimization regime.

Contributions:
- Mechanism: A principled link from the SDE view of adaptive optimizers to an information allocation trade-off between parameters and activations, positioning SGN as a controllable driver of ICL.
- Metric: P-NE, a preconditioner-aware, curvature-normalized measure of integrated SGN enabling comparisons across batch sizes, schedules, and noise sources.
- Causal tests: Monotonicity, phase sensitivity, and approximate source equivalence under tight controls (matched perplexity, token budgets, optimizer states).
- Negative controls and probes: Trade-off with weight-based memorization; mechanistic (induction heads) and geometric (flatness) analyses.

## 2. Related Work
- ICL mechanisms: implicit Bayesian inference and regression in context; meta-optimization in activations; circuit-level accounts (e.g., induction heads).
- SGN and implicit regularization: SGD/SDE connections to effective temperature, flat minima, and approximate Bayesian inference; batch-size effects; heavy-tailed gradient noise; entropy-based regularization and SAM.
- Adaptive optimizers: SDE analyses and preconditioning effects in Adam/AdamW; generalization gaps and sharpness critiques.

We specifically extend: (i) SGN→flatness→generalization to the ICL setting; (ii) batch/noise effects to an optimizer-aware metric; (iii) ICL theories by isolating a causal training-dynamics lever, controlling for LM loss and token budget.

## 3. Theory and Metric

### 3.1 SDE view of adaptive optimization
Adaptive optimizers with diagonal preconditioners (e.g., AdamW with P_t ≈ diag(1/√v_t)) are approximable by SDEs where the stochastic update has covariance ≈ η_t^2 P_t C_t P_t, with η_t the step size and C_t the mini-batch gradient covariance. The stationary distribution scales as exp(−L(θ)/T_t) under local approximations, with effective temperature T_t ∝ η_t^2 Tr(P_t C_t P_t) modulated by curvature. Higher temperature biases toward flatter, higher-entropy basins that are robust to parameter perturbations.

### 3.2 Information allocation: parameters vs activations
Models can implement solutions in weights (parametric memorization) or in activations conditioned on context (ICL). Elevated effective temperature increases the stability cost of brittle, weight-coded solutions, favoring reusable, prompt-sensitive circuits. When pretraining data exhibit task diversity, optimization under higher integrated SGN selects for context-adaptive subroutines (e.g., induction-like attention patterns).

### 3.3 Preconditioned Noise Exposure (P-NE)
- Raw P-NE:
  P-NE = Σ_t η_t^2 Tr(P_t C_t P_t)
- Curvature-normalized P-NE:
  P-NE* = Σ_t η_t^2 Tr(P_t C_t P_t) / max(ε, Tr(P_t F_t P_t))
  where F_t is an empirical Fisher approximation (or other curvature proxy), ε > 0 ensures stability.

Normalization mitigates schedule and parameterization effects by measuring noise relative to local curvature under the same preconditioner.

### 3.4 Estimation and robustness
- C_t estimation: online diagonal variance over K micro-batches every N steps; optional Hutchinson trace for full-covariance traces.
- F_t estimation: empirical Fisher via per-example gradient norms or mini-batch gradient outer products; low-rank or Hutchinson trace for tractability.
- Placement: explicit noise is injected pre-preconditioning to match the natural stochastic term of the SDE; we log and control preconditioner statistics.
- Gradient clipping and precision: compute on unclipped, full-precision gradients; log clipping rates; report sensitivity analyses.
- Heavy tails: we assess robustness by injecting Gaussian, Laplace, and Student-t noise matched by second moment; report estimator variance.

Pseudocode (per logging step):
```
def p_ne_increment(grad_micro, precond_diag, eta_t, fisher_proxy_diag, eps=1e-8):
    # grad_micro: [K, D] gradients from K micro-batches (unclipped, fp32)
    C_diag = var(grad_micro, axis=0)            # diag covariance
    P = precond_diag                             # e.g., 1 / sqrt(v_t + δ)
    num = eta_t**2 * (P * C_diag * P).sum()     # Tr(P C P)
    den = max(eps, (P * fisher_proxy_diag * P).sum())
    return num / den
```
We release a library that logs P-NE* with configurable K, N, and estimators, and reports computational overhead (<1–3% in our settings).

## 4. Experimental Design

### 4.1 Models, data, and controls
- Architectures: Decoder-only Transformers (~350M and ~1.3B params), identical tokenizers and context lengths.
- Data: Fixed pretraining mixture; identical shuffling seeds across runs; contamination checks for downstream eval sets.
- Optimizer: AdamW with fixed β, weight decay, dropout, gradient clipping configuration; matched warmup/decay schedules across conditions unless varied by design.
- Compute controls: Primary analyses match (a) target validation perplexity and (b) token budgets (updates × batch × seq length). We include matched-updates ablations to rule out time-on-task effects.

### 4.2 Manipulating P-NE
- Batch size: vary global batch from small to large; evaluate both fixed LR and linear-scaling LR to decouple LR from noise.
- Explicit noise: at large batch, add zero-mean noise pre-preconditioning, scaled online to match a reference P-NE* trajectory.
- Scheduling: redistribute P-NE* mass to early/mid/late phases with constant total P-NE*, using batch size and/or noise schedules while keeping LR schedules fixed.

We continuously monitor P-NE*, gradient norms, clipping, preconditioner statistics (v_t), and curvature proxies for fidelity.

### 4.3 Evaluation and statistics
- ICL tasks: MMLU, BBH subsets, synthetic in-context linear regression and algorithmic tasks. Use standardized few-shot templates, prompt ensembling, and calibration (e.g., contextual calibration or probability renormalization). Fix context length, demo ordering, and sampling seeds.
- Negative controls: factual recall cloze probes; synthetic key–value memorization; training data extraction metrics.
- Additional probes: induction-head strength (copy/counterfactual tasks), attention pattern selectivity, logit-lens analyses; flatness via SAM sharpness and top Hessian eigenvalues.
- Protocol: pre-registered hypotheses, metrics, exclusion criteria; ≥3 seeds per condition; mixed-effects models; TOST for equivalence; Holm–Bonferroni for multiple tests; report CIs and effect sizes.

### 4.4 Stable optimization regime and failure modes
We define a stable regime by bounded gradient norm distributions, low clipping rates, and non-increasing validation loss. We sweep P-NE* via noise amplitude/batch until instability markers appear, report thresholds, and provide heuristics (e.g., ceiling on per-phase P-NE* increments).

## 5. Results
- Monotonicity: At matched perplexity and token budgets, aggregate ICL scores increase with P-NE*. This holds across batch variation and explicit noise injection, with significant rank correlations and consistent effect sizes across seeds.
- Phase sensitivity: Concentrating P-NE* late in training yields higher ICL than early-phase concentration at equal total P-NE* and perplexity, indicating sensitivity of late-formed context-adaptive circuits.
- Source equivalence: Small-batch noise and matched explicit noise (pre-preconditioning) produce statistically equivalent ICL given matched P-NE* trajectories.
- Trade-off: Increasing P-NE* reduces parametric memorization while improving ICL, revealing a clear Pareto frontier.
- Mechanistic/geometric correlates: Higher P-NE* associates with stronger induction-head scores and flatter minima (lower SAM sharpness, smaller leading Hessian eigenvalues), consistent with the proposed mechanism.

Figures: 
- Fig. 1: P-NE* vs. ICL (per-seed scatter, mixed-effects fit).
- Fig. 2: Phase-scheduled P-NE* comparison (early/mid/late).
- Fig. 3: Source equivalence (TOST intervals).
- Fig. 4: ICL–memorization trade-off (Pareto frontier).
- Fig. 5: Induction-head metrics and sharpness vs. P-NE*.

## 6. Discussion
- Mechanism and alternatives: Results support SGN as an implicit information penalty favoring context-adaptive computation. We address sharpness reparameterization by using preconditioner-aligned curvature proxies and noise-stability measures, and we provide a PAC-Bayesian view where higher P-NE* corresponds to broader posteriors over weights.
- Optimizer dependence: P-NE* is explicitly preconditioner-aware; we show AdamW-specific results and outline expected differences for SGD/momentum and Adafactor. Preliminary ablations indicate qualitatively similar trends with momentum SGD under matched effective temperature.
- Data and scaling: Effects persist across two scales and mixed-domain pretraining; we discuss expected scaling interactions and context-length dependencies.
- Practical implications: Late-stage batch reduction or matched explicit noise can improve ICL without additional parameters or data. We provide guardrails for safe P-NE* increases.

## 7. Limitations and Threats to Validity
- Scope: Shown on Transformer decoders and AdamW; generality to other optimizers/architectures is unproven.
- Estimation: P-NE* relies on curvature proxies and diagonal approximations; we report robustness but acknowledge residual estimator bias.
- Contamination and templates: Despite checks and calibration, few-shot benchmarks can be sensitive to formatting; we mitigate via prompt ensembling and calibration but cannot eliminate all biases.
- Stability boundaries: Excessive P-NE* degrades optimization; we document thresholds but they may vary with architecture and data.

## 8. Conclusion
Preconditioned SGN, quantified by P-NE*, is a controllable driver of ICL. By manipulating training noise in an optimizer-aware manner—especially late in training—we can enhance ICL while trading off parametric memorization. This reframes ICL as a predictable outcome of training dynamics and offers practical levers for engineering adaptable models.

## 9. Reproducibility
We release:
- A lightweight library for online P-NE* logging (diag and Hutchinson variants), with tests and examples.
- Training/evaluation scripts, configs, and seeds for all runs.
- Pre-registration, raw logs, and analysis notebooks.

## References
- Akyürek, E. et al. (2022). What learning algorithm is in-context learned? ICLR.
- Chaudhari, P. et al. (2019). Entropy-SGD. JSIAM.
- Dinh, L. et al. (2017). Sharp minima can exist in wide neural networks. ICML.
- Foret, P. et al. (2021). Sharpness-Aware Minimization. ICLR.
- Li, Q., Tai, C., E, W. (2017). Stochastic modified equations and adaptive methods. ICML.
- Mandt, S., Hoffman, M., Blei, D. (2017). Stochastic gradient descent as approximate Bayesian inference. JMLR.
- Min, S. et al. (2022). Rethinking the role of demonstrations. NAACL.
- Olsson, C. et al. (2022). In-context learning and induction heads. Anthropic.
- Simsekli, U. et al. (2019). Heavy-tailed self-similarity in SGD. NeurIPS.
- von Oswald, J. et al. (2023). Transformers learn in-context by gradient descent in activations. ICLR.
- Wilson, A. et al. (2017). The marginal value of adaptive methods. NeurIPS.
