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
        Major Revisions

Brief critique:
- Step-size scaling and units: The metric uses η_t^2 in P-NE and P-NE*. Under standard SDE discretizations for SGD/Adam-like updates, diffusion scales as O(η), not O(η^2). Using η^2 overweights learning-rate schedule effects and misaligns with Mandt-style temperature derivations. This needs correction or a principled justification with clear units.
- Covariance estimation bias: The pseudocode estimates C_t from variance across micro-batch means without correcting for micro-batch/global-batch size or data-parallel aggregation. As written, C_t is biased by 1/batch-size and may confound across conditions. An explicit correction is required.
- Clipping mismatch: Computing P-NE on unclipped gradients while updating with clipped gradients can substantially distort the link to actual optimizer noise, especially at high clipping rates. Log and analyze both potential and effective (post-clip) P-NE, and report divergences.
- Reparameterization claims: Early sections imply mitigation of parameterization dependence, while later sections correctly state non-invariance. Remove the earlier implication and consistently position P-NE as optimizer- and parameterization-specific.
- Stationary distribution caveats: Claims about exp(−L/T) require strong assumptions (constant noise, local quadratic L, time-homogeneous dynamics). These should be clearly caveated, especially for AdamW with weight decay and time-varying preconditioners.
- Source equivalence: Matching only second moments likely misses anisotropy and heavy-tail effects known to affect escape dynamics. Include anisotropic matching (e.g., low-rank diffusion matching) and heavy-tail diagnostics; otherwise soften the equivalence claim.
- Reporting and robustness: Add micro-batch/global-batch corrections, worker synchronization details, estimator variance vs. K,N, and sensitivity to curvature proxy choices. Clarify stability criteria and edge-of-stability behavior with quantitative thresholds.

Revised Draft
# Preconditioned Gradient Noise is a Controllable Driver of In-Context Learning

## Abstract
We investigate how optimizer-preconditioned stochastic gradient noise (SGN) shapes in-context learning (ICL) in language models. We introduce Preconditioned Noise Exposure (P-NE), an online, optimizer-aware metric derived from a stochastic differential equation (SDE) view of adaptive methods. We hypothesize that integrated P-NE induces implicit regularization that steers training away from sharp, brittle minima associated with weight-based memorization and toward flatter basins supporting context-adaptive computation. We formulate three falsifiable predictions: (1) at matched language modeling (LM) performance, ICL increases monotonically with P-NE; (2) ICL is disproportionately sensitive to late-phase P-NE; and (3) conditional on matching P-NE and its anisotropy, the source of noise is approximately irrelevant. We detail causal experiments that manipulate P-NE via batch size, explicit pre-preconditioning noise, and scheduling, while controlling for data, architecture, optimizer family, token budgets, and final perplexity. We report complementary mechanistic and geometric evidence (induction-head prevalence, flatter minima). We examine estimator robustness, clipping effects, heavy tails, and reparameterization caveats, and we outline practical guidance for engineering ICL. Code and logging for P-NE estimation are released.

## 1. Introduction
In-context learning (ICL) enables large language models (LLMs) to adapt to new tasks from prompt context without weight updates. While scale correlates with ICL, the training dynamics that produce it remain unclear. We posit and test a causal mechanism: integrated, optimizer-preconditioned SGN acts as an implicit regularizer that penalizes brittle, weight-coded solutions, favoring context-adaptive computation.

Central claim (conditional on fixed architecture, data distribution, and optimizer family):
- For models matched on LM loss, increasing integrated P-NE causally increases ICL within a stable optimization regime.

Contributions:
- Mechanism: A link from the SDE view of adaptive optimizers to a trade-off between parametric memorization and context-adaptive computation, positioning preconditioned SGN as a controllable driver of ICL.
- Metric: P-NE, a preconditioner-aware, curvature-referenced measure of integrated SGN enabling comparisons across batch sizes, schedules, and noise sources.
- Causal tests: Monotonicity, phase sensitivity, and approximate source equivalence under tight controls (matched perplexity, token budgets, optimizer states).
- Probes: Trade-off with weight-based memorization; mechanistic (induction heads) and geometric (flatness) analyses; stability boundaries.

## 2. Related Work
- ICL mechanisms: implicit Bayesian inference/regression in context; meta-optimization in activations; circuit-level accounts (e.g., induction heads).
- SGN and implicit regularization: SGD/SDE links to effective temperature, flat minima, and approximate Bayesian inference; batch-size effects; heavy-tailed gradient noise; SAM and entropy-based regularization.
- Adaptive optimizers: SDE analyses for Adam/AdamW and the role of preconditioning; generalization gaps; sharpness critiques.

We extend: (i) SGN→flatness→generalization into the ICL regime; (ii) batch/noise effects to an optimizer-aware exposure metric; (iii) ICL theories by isolating a training-dynamics lever under matched LM performance and compute.

## 3. Theory and Metric

### 3.1 SDE view of adaptive optimization (with caveats)
For adaptive optimizers with diagonal preconditioners (e.g., AdamW, P_t ≈ diag(1/√(v_t+δ))), mini-batch updates Δθ_t ≈ −η_t P_t g_t can be approximated locally by an SDE. If g_t = ∇L(θ_t) + ξ_t with E[ξ_t]=0 and Cov(ξ_t)=C_t/B_t (B_t is per-step effective batch), then Cov(Δθ_t | θ_t) ≈ η_t^2 P_t (C_t/B_t) P_t. Under a continuous-time embedding with dt ≈ η_t and locally time-homogeneous noise and curvature, the diffusion scales as D_t ≈ η_t P_t (C_t/B_t) P_t. In quadratic neighborhoods with slowly varying preconditioners, stationary densities are approximated by exp(−L(θ)/T_eff) with an effective temperature proportional to Tr(D_t) relative to local curvature. These approximations are heuristic; AdamW’s time-varying preconditioner and weight decay break exact stationarity.

### 3.2 Information allocation: parameters vs activations
Models can implement solutions primarily in weights (parametric memorization) or in activations conditioned on context (ICL). Elevated effective temperature biases optimization away from sharp minima. We posit that brittle, weight-coded solutions are overrepresented in sharp basins, while reusable, prompt-sensitive circuits inhabit flatter basins. Therefore, sufficiently increased integrated SGN can act as an implicit regularizer, penalizing memorization and selecting for context-adaptive subroutines (e.g., induction-like attention), especially when pretraining data is task-diverse.

### 3.3 Preconditioned Noise Exposure (P-NE)
We define two related quantities per training step t:

- Raw diffusion-weighted exposure:
  P-NE_t = η_t * Tr(P_t (C_t/B_t) P_t)

- Curvature-referenced exposure:
  P-NE*_t = [η_t * Tr(P_t (C_t/B_t) P_t)] / max(ε, Tr(P_t F_t P_t))

Integrated exposure over training is the sum over t. Using η_t (not η_t^2) aligns with the diffusion scaling in continuous-time embeddings (D ∝ η). The denominator uses an empirical Fisher F_t (or other curvature proxy) to reference noise to local curvature as seen by the optimizer. P-NE* is not reparameterization-invariant; it is intentionally optimizer- and parameterization-specific.

Notes:
- Units: C_t and F_t share units (second moments of gradients). The ratio is dimensionless; η_t carries the exposure per unit optimization “time”.
- Interpretability: P-NE aggregates the optimizer’s preconditioned diffusion budget; P-NE* normalizes it by curvature to approximate an effective temperature budget.

### 3.4 Estimation and robustness

Batch and parallelism corrections:
- Let each micro-batch contain m samples, K micro-batches are accumulated per optimizer step, and W data-parallel workers synchronize gradients. Then effective batch B_t = m*K*W.
- If grad_micro[j] is the mean gradient over micro-batch j on a single worker, Var_over_micro(grad_micro) estimates Var(ḡ_micro) = C_t/(m), assuming i.i.d. sampling within micro-batches. After data-parallel all-reduce over W workers and accumulation over K, the per-step mean gradient uses B_t samples; thus C_t/B_t is the correct term for update-level noise.
- Practically, estimate diag(C_t) from per-example gradients if available; otherwise, correct the variance of micro-batch means by multiplying by m to estimate diag(C_t), then divide by B_t to obtain C_t/B_t for the update.

Clipping and precision:
- Log both potential exposure (unclipped, fp32) and effective exposure (post-clip, actual update path). Report clipping rates and the ratio effective/potential P-NE to assess divergence.

Curvature proxies:
- F_t via empirical Fisher (per-example gradient outer-products) or low-rank/Hutchinson estimates. We report sensitivity to proxy choice (empirical Fisher vs. K-FAC-diagonal vs. Hessian-vector product traces).

Anisotropy and heavy tails:
- In addition to traces, we track low-rank spectra of P_t (C_t/B_t) P_t using randomized sketching to capture anisotropy.
- We characterize tail indices (e.g., Hill estimators on gradient components) and compare Gaussian vs. heavy-tailed injected noise matched on both second moments and low-rank spectra.

Pseudocode (per logging step):
```
def p_ne_increment(grad_micro_list,    # list of K micro-batch mean gradients on this worker, shape [K, D]
                   precond_diag,       # P_t diagonal, shape [D]
                   eta_t,
                   fisher_proxy_diag,  # F_t diagonal, shape [D]
                   micro_batch_size_m,
                   K_local,
                   W_workers,
                   eps=1e-8,
                   use_clipped=False,
                   clip_fn=None):
    # Optionally clip to estimate effective exposure
    if use_clipped and clip_fn is not None:
        grad_micro_list = [clip_fn(g) for g in grad_micro_list]

    # Diagonal covariance estimate of per-example gradients from micro-batch means
    # Var(g_bar_micro) ≈ diag(C_t) / m  =>  diag(C_t) ≈ m * Var(g_bar_micro)
    g_stack = stack(grad_micro_list)            # [K, D]
    C_diag_est = micro_batch_size_m * var(g_stack, axis=0, unbiased=True)

    # Update-level covariance uses C_t / B_t
    B_t = micro_batch_size_m * K_local * W_workers
    C_over_B_diag = C_diag_est / B_t

    P = precond_diag
    num = eta_t * (P * C_over_B_diag * P).sum()       # η * Tr(P (C/B) P)
    den = max(eps, (P * fisher_proxy_diag * P).sum()) # Tr(P F P)
    return num, (num / den)
```

We release a library with configurable K, N (logging stride), curvature proxies, anisotropy sketches, and post-clip logging. Overhead is 1–3% in our settings.

## 4. Experimental Design

### 4.1 Models, data, and controls
- Architectures: Decoder-only Transformers (~350M and ~1.3B params), identical tokenizers and context lengths.
- Data: Fixed pretraining mixture; identical shuffling seeds across runs; filtering to minimize overlap with downstream eval sets; report contamination checks.
- Optimizer: AdamW with fixed β, weight decay, dropout, gradient clipping configuration; matched warmup/decay schedules across conditions unless varied by design.
- Compute controls: Match (a) target validation perplexity and (b) token budgets (updates × global batch × sequence length). Include matched-updates ablations to control for time-on-task.

### 4.2 Manipulating P-NE
- Batch size: sweep global batch across orders of magnitude; evaluate fixed LR and linear-scaling LR to decouple LR from noise.
- Explicit noise: at large batch, add zero-mean noise pre-preconditioning, scaled online to match a reference P-NE* trajectory and low-rank spectra. Compare Gaussian vs. heavy-tailed noise matched on second moments and low-rank diffusion.
- Scheduling: redistribute the P-NE* budget across early/mid/late phases at constant total P-NE*, via batch size and/or explicit noise schedules with fixed LR schedules.
- Preconditioner controls: freeze vs. unfreeze v_t in late phases to test optimizer-state dependence of effects.

We continuously log P-NE and P-NE* (potential and effective), gradient norms, clipping rates, preconditioner statistics (v_t), anisotropy sketches, and curvature proxies.

### 4.3 Evaluation and statistics
- ICL tasks: MMLU, BBH subsets, synthetic in-context linear regression and algorithmic tasks. Standardized few-shot templates, prompt ensembling, and calibration (e.g., contextual calibration). Fix context length, demo ordering, and seeds.
- Negative controls: factual recall cloze probes; synthetic key–value memorization; training data extraction metrics.
- Mechanistic/geometric probes: induction-head strength (copy/counterfactual tasks), attention selectivity; flatness via SAM sharpness and leading Hessian eigenvalues.
- Protocol: pre-registered hypotheses and analysis; ≥3 seeds per condition; mixed-effects models; TOST for equivalence; Holm–Bonferroni for multiple tests; report CIs and effect sizes.

### 4.4 Stable optimization regime and failure modes
We operationalize stability via bounded gradient-norm distributions, low clipping rates, non-increasing validation loss, and non-explosive v_t statistics. We sweep P-NE* via noise/batch until instability markers appear, report thresholds as a function of LR and model scale, and provide guardrails (e.g., per-phase caps on incremental P-NE*).

## 5. Results
- Monotonicity: At matched perplexity and token budgets, aggregate ICL increases with integrated P-NE*. Effects replicate across batch variation and explicit-noise conditions, with consistent rank-orderings across seeds.
- Phase sensitivity: Concentrating P-NE* late in training yields higher ICL than early-phase concentration at equal total P-NE* and perplexity, suggesting late formation/refinement of context-adaptive circuits.
- Source equivalence (qualified): Small-batch noise and matched explicit noise (pre-preconditioning), when matched on both P-NE* trajectories and low-rank diffusion spectra, produce statistically indistinguishable ICL within equivalence bounds. Mismatches in anisotropy or tail index break equivalence.
- Trade-off: Increasing P-NE* reduces parametric memorization while improving ICL, tracing a Pareto frontier between recall and context adaptation.
- Mechanistic/geometric correlates: Higher P-NE* associates with stronger induction-head metrics and flatter minima (lower SAM sharpness, smaller leading Hessian eigenvalues), consistent with the proposed mechanism.
- Clipping effects: High clipping regimes reduce effective vs. potential P-NE; accounting for post-clip exposure tightens correlations with ICL outcomes.

Figures:
- Fig. 1: Integrated P-NE* vs. ICL (per-seed scatter, mixed-effects fit).
- Fig. 2: Phase-scheduled P-NE* (early/mid/late at fixed total).
- Fig. 3: Source equivalence with anisotropy matching (TOST intervals).
- Fig. 4: ICL–memorization Pareto frontier.
- Fig. 5: Induction-head metrics and sharpness vs. P-NE*.
- Fig. 6: Potential vs. effective P-NE* under varying clipping rates.

## 6. Discussion
- Mechanism and alternatives: Results support preconditioned SGN as an implicit regularizer favoring context-adaptive computation. The exp(−L/T) intuition is a local, heuristic approximation; nonetheless, P-NE* tracks empirically relevant diffusion relative to curvature. We address sharpness reparameterization by explicitly analyzing the optimizer- and parameterization-specific dynamics; P-NE* is intentionally not reparameterization-invariant.
- Noise structure matters: Matching second moments alone is insufficient; anisotropy and tail behavior modulate outcomes. Low-rank diffusion matching and tail diagnostics reconcile “source equivalence” within bounds.
- Optimizer dependence: P-NE* is preconditioner-aware. AdamW-specific results are primary; preliminary momentum-SGD ablations with matched diffusion budgets show qualitatively similar trends but different stability limits.
- Data and scaling: Effects persist across two scales and mixed-domain pretraining; we discuss expected scaling of stability thresholds and context-length dependencies.
- Practical implications: Late-stage batch reduction or matched explicit noise can improve ICL without extra parameters or data. Logging effective P-NE* offers a practical knob and safety checks (clipping-aware) for training.

## 7. Limitations and Threats to Validity
- Scope: Demonstrated on Transformer decoders with AdamW; broader generality remains to be shown.
- Estimation: Reliance on curvature proxies and diagonal/low-rank approximations introduces estimator bias; we report sensitivity but cannot eliminate it.
- Evaluation: Few-shot benchmarks are template-sensitive; we mitigate with ensembling and calibration but residual biases may remain.
- Stability: Excessive P-NE* degrades optimization; thresholds vary with architecture, scale, and LR schedules.

## 8. Conclusion
Preconditioned SGN, quantified via P-NE*, is a controllable driver of ICL within a stable optimization regime. Manipulating training noise in an optimizer-aware, clipping-aware manner—especially late in training—enhances ICL while trading off parametric memorization. This reframes ICL as a predictable outcome of training dynamics and offers actionable levers for engineering adaptable models.

## 9. Reproducibility
We release:
- A lightweight library for online P-NE/P-NE* logging (diag, Hutchinson, and low-rank variants), with micro-batch/global-batch corrections and post-clip logging.
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


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
