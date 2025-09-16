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
- Strengths: Clear, testable central claim; plausible SDE-based mechanism; optimizer-aware noise metric; strong emphasis on falsification with well-controlled protocols; thoughtful mechanistic and geometric probes; alignment with constraints (AI-focused, no blockchain).
- Gaps that preclude publication as-is:
  - No empirical validation. A leading journal will require at least a subset of completed experiments demonstrating the predicted monotonicity, phase sensitivity, and source-equivalence, or a substantially stronger theoretical result.
  - Theory is heuristic. The “information penalty” and parameter–activation tradeoff are argued qualitatively; no formal statement or bound ties P-NE to ICL capability. The role of preconditioning in the stationary distribution and its link to information in the weights requires a precise formulation.
  - Metric definition and estimation. P-NE lacks normalization/invariance analysis, estimator pseudocode, and runtime/variance characteristics. Gradient clipping, mixed precision, and optimizer hyperparameters substantially distort noise covariance; the draft should specify how P-NE remains meaningful under these.
  - Confounds and controls. Batch-size changes typically couple to LR schedules (linear scaling rules), gradient accumulation, and dataloader RNG; the plan needs explicit LR/batch coupling strategy, clipping regime control, and distributional checks on gradient statistics.
  - Over-strong equivalence claim. “Source equivalence” between batch-size-induced noise and explicit Gaussian noise is not guaranteed under adaptive preconditioning or non-Gaussian gradient noise; this should be reframed and tested with equivalence, not mere non-inferiority.
  - Positioning vs prior work. The novelty relative to gradient-noise scale literature and recent ICL dynamics papers should be sharpened (e.g., precise distinctions: preconditioned metric, phase sensitivity, and weight-vs-activation allocation).
  - Evaluation and power. Define primary ICL metrics, minimal detectable effect sizes, sample sizes, and equivalence testing (e.g., TOST). Add negative controls to demonstrate specificity (e.g., parametric recall/memorization decreasing with higher P-NE).
  - Clarity and consistency. Expand all acronyms on first use (e.g., SGLD), unify terminology (P-NE), specify sampling cadence (done once as N=100—keep consistent), and justify model scales and data mixtures.

Revised Draft
# Gradient Noise Governs In-Context Learning

Abstract
We hypothesize that the stochastic gradient noise (SGN) accumulated during pretraining is a major, controllable driver of in-context learning (ICL) in large language models (LLMs). For models matched on architecture, data, optimizer family, and language modeling (LM) performance, we predict that ICL ability increases with a preconditioned integrated noise exposure experienced during training. Mechanistically, SGN induces an implicit information penalty on parameters, biasing solutions toward context-adaptive computation realized in activations. We formalize this via stochastic differential equation (SDE) views of adaptive optimizers and define a practical, optimizer-aware preconditioned noise exposure (P-NE) metric. We propose pre-registered experiments that vary noise independently of LM loss (via batch size, explicit noise, and scheduling) while controlling tokens, schedules, clipping, and data order. We provide falsification criteria (monotonicity within regime, phase sensitivity, and approximate source equivalence) and mechanistic/geometry diagnostics. If validated, this reframes ICL as a training-dynamics property with actionable levers beyond scale.

1. Introduction
Transformers often exhibit ICL: they infer task structure from prompts and adapt within a forward pass. While scale and data diversity correlate with ICL, the causal training factors remain unclear. We posit a specific driver: integrated, optimizer-preconditioned SGN.

Bounded claim
Conditional on fixed architecture, pretraining distribution with task/format diversity, optimizer family, and matched LM performance, increasing the integrated preconditioned gradient noise (P-NE) tends to increase ICL capability, within an operating regime that does not derail optimization.

Contributions
- Mechanism: An implicit-regularization account that frames SGN as an information penalty on parameters, shifting computation toward prompt-conditioned activations.
- Metric: A practical preconditioned noise exposure, with estimators and implementation details, designed to be comparable across batch sizes and optimizers.
- Falsifiable predictions: Monotonicity (within regime), late-phase sensitivity, and approximate source equivalence (batch vs explicit noise) with pre-registered equivalence margins.
- Protocol: A tightly controlled experimental design that isolates noise from perplexity and other confounds, plus mechanistic and geometric probes.

2. Related Work
- ICL mechanisms: Induction heads and circuit-level accounts; ICL as implicit gradient descent or Bayesian inference in activations.
- SGN as implicit regularization: SGD as approximate Bayesian inference; effects of batch size, flatness, and local entropy; adaptive optimizers and preconditioning.
- Distinction from prior work: Prior studies relate small batches to generalization or propose data/task-structure explanations for ICL. Our novelty is an optimizer-aware, integrated, phase-sensitive noise metric tailored to ICL, with direct causal tests at matched LM performance.

3. Theory: SGN as Information Penalty Favoring Context-Adaptation

3.1 Optimizer noise and stationary bias
Under slowly varying hyperparameters, SGD-like methods admit SDE approximations. For adaptive optimizers with diagonal preconditioner P_t (e.g., Adam’s 1/sqrt(v_t)), the update noise has covariance approximately P_t C_t P_t, where C_t is the mini-batch gradient covariance. The effective temperature scales with η_t^2 Tr(P_t C_t P_t) and biases training toward broad, flat basins (higher local entropy). This imposes a stability cost on fine-grained weight encodings while leaving per-sequence activations unconstrained.

3.2 Parameter–activation allocation
A model can solve tasks by:
- Weight-coded computation: Memorize task-specific mappings in parameters—fragile under high effective temperature.
- Context-adaptive computation (ICL): Infer instance/task structure from the prompt and implement it transiently in activations (attention/MLP fast variables).
Integrated SGN thus shifts the optimum toward reusable, prompt-driven circuits (meta-inference) and away from brittle weight-coded solutions, provided the data contains sufficient task/format diversity to train such circuits.

3.3 A preconditioned noise exposure (P-NE) and its normalization
We define P-NE over training steps t:
P-NE = Σ_t η_t^2 · Tr(P_t C_t P_t)

Practical refinements for comparability:
- Curvature-normalized P-NE: P-NE* = Σ_t η_t^2 · Tr(P_t C_t P_t) / max(ε, Tr(P_t F_t P_t)), where F_t is a Fisher approximation to the Hessian and ε prevents division by near-zero curvature. This reduces sensitivity to learning-rate schedule and loss scale.
- Layerwise aggregation: Compute per-layer contributions and report both total and layerwise P-NE to capture heterogeneity.
- Units and invariance: Report P-NE (and P-NE*) alongside gradient-norm and loss-scale diagnostics to ensure cross-run comparability.

Estimators (online, low overhead):
- Diagonal variance: Estimate diag(C_t) via K micro-batches. Use Tr(P_t C_t P_t) ≈ Σ_i P_t,ii^2 Var[g_t,i] with K≈8–16.
- Hutchinson probe: For v ~ N(0, I), estimate Tr(P_t C_t P_t) ≈ E_v[||P_t Ĝ_t v||^2], where Ĝ_t is a Jacobian-free gradient-sampler using micro-batch resampling.
- Proxies: When covariance estimates are infeasible, use a preconditioned gradient-noise scale GNS_p = Tr(P_t C_t P_t) / ||P_t ḡ_t||^2 and track Σ_t η_t^2 GNS_p.

Implementation notes:
- Measure every N steps (e.g., N=100) with fixed K and fixed micro-batch RNG to stabilize estimates.
- Exclude dropout and data-order randomness from the primary P-NE; track them separately to test specificity.
- Account for gradient clipping by logging unclipped and clipped moments; report clipping rate and its effect on variance.

4. Experimental Design

4.1 Models and data
- Decoder-only Transformers at ≈350M and ≈1.3B parameters—minimal scales with reliable ICL while enabling multi-run studies.
- Fixed tokenizer, curriculum, and diverse data mixture with explicit task/format variation to make ICL learnable.
- Optimizers: AdamW (primary) with fixed β1, β2, weight decay; SGD+momentum as a sensitivity analysis. Mixed precision and gradient scaling held constant across conditions.

4.2 Manipulating noise
- Batch-size sweep: Vary global batch size; follow a pre-specified LR strategy (either fixed LR across batches or linear-scaling rule) and include both regimes to decouple LR and variance effects.
- Explicit gradient-noise injection: Add zero-mean Gaussian noise before preconditioning to match target P-NE trajectories; calibrate per-layer scales to the observed C_t.
- Parameter noise (post-update) as a robustness check; matched in preconditioned coordinates.
- Noise scheduling: Early vs late vs uniform noise schedules with matched total P-NE (and P-NE*).

4.3 Matching and controls
- Perplexity matching: Train runs to a target validation perplexity ±ε and stop; record tokens and updates. Additionally, include fixed-tokens and fixed-updates matched pairs to disentangle exposure from optimization time.
- Hold fixed: Data order (seeded), augmentation, dropout, clipping threshold, LR schedule shape (unless explicitly varied as an ablation), weight decay, gradient accumulation, mixed-precision settings.
- Monitoring: Online P-NE/P-NE*, gradient norms, clipping rates, Fisher trace, and optimizer states to verify manipulation fidelity.

4.4 ICL evaluation
Primary metrics (pre-registered):
- Synthetic probes: In-context linear regression/classification with Gaussian features; symbolic sequence tasks (copy, reverse, addition) and function induction with format variation.
- Natural tasks: Few-shot MMLU and BBH subsets; arithmetic and translation few-shot subsets with controlled prompt formats and shot counts.
Secondary/diagnostic:
- Mechanistic: Induction head prevalence/strength; linear probes for latent task parameters in the residual stream; attention pattern changes with prompt permutations.
- Geometry: Hessian/Fisher trace, SAM sharpness, local-entropy proxies; compare at matched perplexity.
Negative controls:
- Parametric recall: Factual recall (e.g., cloze probes) and synthetic key–value memorization. Prediction: Higher P-NE reduces weight-coded memorization at matched perplexity, separating ICL from rote memory.

4.5 Statistical analysis and falsification criteria
Pre-registration:
- Define primary ICL metrics, evaluation seeds, stopping rules, and exclusion criteria.
- Power plan: Target minimum detectable effect size δ on primary ICL metrics with ≥80% power; specify number of seeds per condition accordingly.
Hypotheses and tests:
- Monotonicity (within regime): At matched perplexity, ICL increases with P-NE (Spearman ρ > 0 with CI excluding 0).
- Phase sensitivity: Late-phase noise yields larger ICL gains than early-phase for matched P-NE; paired comparisons with CIs and Bayes factors.
- Source (approximate) equivalence: Batch-size vs explicit-noise runs matched on P-NE have ICL within a pre-specified equivalence margin (TOST); failure implies source-specific effects.
Multiple comparisons:
- Mixed-effects models with run-level random effects; control FDR across tasks; report per-task and aggregated effects.

5. Results to Report (required for publication)
- Verification that P-NE manipulation succeeded and remained decoupled from LM performance at stopping.
- Primary ICL outcomes with CIs and effect sizes; ablations showing robustness across scales and optimizers.
- Mechanistic and geometric correlations with P-NE at matched perplexity.
- Negative controls demonstrating decreased memorization with increased P-NE.

6. Threats to Validity and Limitations
- Optimizer dependence: Effects may vary across AdamW vs SGD; we include both but do not claim universality.
- Metric estimation error: Multiple estimators (diagonal, Hutchinson, proxy GNS_p) reported with agreement analyses.
- Gradient clipping: Can distort covariance; we log clipping rates and include a no-clipping ablation where optimization permits.
- Data dependence: Without task/format diversity, ICL may not increase; we vary mixtures to delineate scope.
- Non-monotonic extremes: Excessive noise harms optimization; we report the operating regime boundaries.
- Scope: Findings may not transfer to non-Transformer architectures or non-LM objectives (e.g., RL fine-tuning).

7. Implications
If supported, the results provide:
- A tunable lever for ICL independent of scale (e.g., small-batch or late-phase noise schedules).
- A unifying view connecting ICL to optimizer-induced implicit bias and flatness.
- Practical guidance on batch sizing and noise schedules to engineer context-adaptive capability at lower compute.

8. Reproducibility and Artifacts
We will release code to compute P-NE/P-NE* online, estimator implementations, training/eval scripts, pre-registration documents, and logs enabling post hoc verification of noise manipulation.

9. Conclusion
We propose and test an optimizer-aware causal account of ICL: integrated, preconditioned gradient noise imposes an information cost on parameters, biasing solutions toward context-adaptive computation. We define P-NE, provide estimators, and design falsifiable experiments that isolate noise from confounds at matched LM performance. Confirmation would make ICL a controllable property of training dynamics and clarify how optimization statistics shape emergent behaviors.

References
- Akyürek, E. et al. (2023). What learning algorithm is in-context learning? ICLR.
- Brown, T. et al. (2020). Language Models are Few-Shot Learners. NeurIPS.
- Chaudhari, P. et al. (2017). Entropy-SGD: Biasing Gradient Descent into Wide Valleys. JMLR.
- Dai, Z. et al. (2023). Why Can GPT Learn In-Context? Language Models Implicitly Perform Bayesian Inference. arXiv.
- Foret, P. et al. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR.
- Garg, S. et al. (2022). What Can Transformers Learn In-Context? NeurIPS.
- Jastrzębski, S. et al. (2018). Three Factors Influencing Minima in SGD. arXiv.
- Jastrzębski, S. et al. (2020). The Break-Even Point on Optimization Trajectories of Deep Neural Networks. ICLR.
- Keskar, N. S. et al. (2017). On Large-Batch Training for Deep Learning. ICLR.
- Li, Q. et al. (2017). Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms. ICML.
- Mandt, S., Hoffman, M., & Blei, D. (2017). Stochastic Gradient Descent as Approximate Bayesian Inference. JMLR.
- McCandlish, S. et al. (2018). An Empirical Model of Large-Batch Training. arXiv.
- Neelakantan, A. et al. (2015). Adding Gradient Noise Improves Learning. arXiv.
- Olsson, C. et al. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread.
- Smith, S. L., & Le, Q. V. (2018). A Bayesian Perspective on Generalization and SGD. ICLR.
- von Oswald, J. et al. (2023). Transformers Learn In-Context by Gradient Descent. ICML.
- Wager, S., Wang, S., & Liang, P. (2013). Dropout Training as Adaptive Regularization. NIPS.
- Wilson, A. C. et al. (2017). The Marginal Value of Adaptive Gradient Methods in Deep Learning. NeurIPS.
- Zhang, J. et al. (2020). Why Adam Beats SGD for Attention Models. arXiv.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
