Minor Revisions

Brief critique:
- Strengths: Strong novelty in causally linking preconditioned gradient noise to ICL via implicit regularization, with falsifiable experiments and tight controls; comprehensive related work; practical metrics and probes; tempered claims improve rigor; aligns with constraints (AI/LLM focus, no blockchain).
- Issues requiring minor revision: 
  - Novelty clarification: Explicitly contrast with prior small-batch/ICL empirics (e.g., McCandlish et al.) to emphasize the preconditioned metric and phase-sensitivity as unique contributions.
  - Experimental details: Add brief rationale for model scales (350M/1.3B) and why they suffice for ICL emergence; specify N for P-NE sampling frequency.
  - Limitations expansion: Briefly address potential generalizability to non-Transformer architectures or non-LM objectives.
  - Minor prose tweaks: Fix inconsistencies (e.g., "SGLD" in intro but not expanded; ensure consistent terminology like "P-NE" vs "integrated noise exposure").

Revised Draft
# Gradient Noise Governs In-Context Learning

Abstract
We hypothesize that stochastic gradient noise accumulated during pretraining is a major, controllable driver of in-context learning (ICL) in large language models (LLMs). For models matched on architecture, data distribution, and language modeling performance, we predict that ICL ability increases with a preconditioned integrated noise exposure experienced during training. Mechanistically, gradient noise induces an implicit information penalty on parameters, shifting solutions toward context-adaptive computation realized in activations. We formalize this via stochastic differential equation (SDE) views of adaptive optimizers and propose tractable, falsifiable experiments that manipulate noise independently of loss (via batch size, explicit noise injection, and scheduling) while tightly controlling confounders (tokens seen, optimizer, dropout, clipping, and data order). If validated, this reframes ICL as a training-dynamics property with actionable levers beyond scale.

1. Introduction
Transformers exhibit ICL: they infer task structure from prompts and adapt within a forward pass (Brown et al., 2020). While scaling improves ICL, the causal training factors remain unclear. We posit a specific driver: stochastic gradient noise (SGN). Our central claim is bounded:

- Claim (bounded): Conditional on fixed architecture, pretraining distribution (with task/format diversity), optimizer family, and matched language modeling quality, increasing the integrated (preconditioned) gradient noise tends to increase ICL capability.

Contributions:
- Theory: An implicit-regularization account linking SGN to a parameter–activation information tradeoff that favors ICL.
- Metric: A practical, optimizer-aware integrated noise exposure measure.
- Falsifiable predictions: Monotonicity (within a regime), source-equivalence (batch vs explicit noise), late-phase sensitivity, and geometric/mechanistic signatures.
- Protocol: An experimental design that isolates noise from performance and other confounds.

2. Related Work
- ICL mechanisms: Induction heads and circuit-level analyses (Olsson et al., 2022); ICL as implicit gradient descent or Bayesian inference (Akyürek et al., 2023; von Oswald et al., 2023; Garg et al., 2022; Dai et al., 2023).
- SGD as approximate inference and implicit bias: Mandt et al. (2017); Smith & Le (2018); Jastrzębski et al. (2018, 2020); McCandlish et al. (2018) gradient-noise scale; Keskar et al. (2017) sharp vs flat minima; Chaudhari et al. (2017) local entropy; Foret et al. (2021) SAM.
- Adaptive optimizers as SDEs and preconditioning: Li et al. (2017) stochastic modified equations; analyses of Adam/momentum stochasticity and generalization (Wilson et al., 2017; Zhang et al., 2020).
- Noise and generalization: Neelakantan et al. (2015) gradient noise; Wager et al. (2013) dropout as regularization.

Our novelty is a targeted, optimizer-aware causal hypothesis and test that links integrated SGN to ICL—distinct from general small-batch generalization results (e.g., McCandlish et al., 2018, which observe small-batch benefits but lack preconditioned metrics, phase-sensitivity tests, or direct ICL linkage) and from works that attribute ICL primarily to data/task structure.

3. Theory: Gradient Noise as an Information Penalty Favoring ICL

3.1 Optimizer noise as implicit regularization
Under constant or slowly varying hyperparameters, SGD and adaptive methods can be approximated by SDEs whose stationary distributions bias solutions toward higher local entropy (Mandt et al., 2017; Li et al., 2017; Chaudhari et al., 2017). For an adaptive method with diagonal preconditioner P_t (e.g., Adam’s 1/sqrt(v_t)), the parameter update noise has covariance approximately P_t C_t P_t, where C_t is the mini-batch gradient covariance. Larger effective noise raises the effective temperature, favoring broader, flatter basins and reducing information that can be stably encoded in parameters.

3.2 The parameter–activation tradeoff
A model can solve tasks by:
- Parameter-driven computation: encode task specifics in weights; sensitive to implicit penalties from noisy optimization.
- Context-driven computation (ICL): infer task-specific structure from the prompt and implement it transiently in activations (fast variables) via attention and MLP circuitry (Olsson et al., 2022; Akyürek et al., 2023; Dai et al., 2023).

Noisy optimization imposes an information cost on parameters but not on per-sequence activations, pushing solutions toward reusable context-adaptive circuits that implement meta-inference.

3.3 A practical, optimizer-aware noise metric
We define Preconditioned Noise Exposure (P-NE) over training steps t:
P-NE = Σ_t η_t^2 · Tr(P_t C_t P_t)
where η_t is the (scalar) step size, P_t the optimizer preconditioner, and C_t the mini-batch gradient covariance. P-NE aggregates effective stochasticity injected into the update after preconditioning. In practice:
- Estimate Tr(P_t C_t P_t) via diagonal variance over K micro-batches or Hutchinson-style probes.
- Use the gradient-noise scale (McCandlish et al., 2018) and preconditioned gradient norms as proxies when full covariance is infeasible.
- Exclude sources not tied to gradient sampling (e.g., dropout) in the primary metric but measure them separately to test specificity.

4. Experimental Design

4.1 Models and data
- Decoder-only Transformers at two scales (≈350M, ≈1.3B; chosen as minimal sizes where ICL emerges reliably in prior work while remaining computationally feasible for multiple runs).
- Fixed tokenizer, curriculum, and data mixture with explicit task/format diversity (so ICL is learnable).
- Optimizer: AdamW with fixed β1, β2, weight decay; gradient clipping and dropout controlled across conditions.

4.2 Manipulating noise
- Batch size sweep: vary global batch size while keeping micro-batch accumulation and data order controls.
- Explicit noise injection:
  - Gradient-noise injection: add zero-mean Gaussian noise to gradients before Adam preconditioning to match target P-NE.
  - Parameter-noise injection: add matched Gaussian noise post-update as a robustness check.
- Noise scheduling: front-load vs back-load noise while keeping total P-NE constant.

4.3 Matching and controls
- Perplexity matching: train each run until reaching a target validation perplexity ±ε; record tokens seen and updates.
- Tokens and updates controls: run paired conditions matching (a) tokens seen and (b) optimizer updates, to disentangle exposure vs optimization-time effects.
- Hold fixed: learning-rate schedule shape, weight decay, dropout rates, clipping threshold, data order (with multiple seeds), and micro-batch accumulation.
- Measure P-NE online every 100 steps (N=100) to verify manipulation fidelity.

4.4 ICL evaluation
- Synthetic probes isolating ICL:
  - In-context linear regression/classification with Gaussian features (Akyürek et al., 2023; Garg et al., 2022).
  - Symbolic tasks (copy, reverse, addition) and function induction with format variation.
- Natural tasks: few-shot MMLU, BBH, and arithmetic/translation subsets; control prompt formats and shot counts.
- Mechanistic probes:
  - Induction head prevalence and strength (Olsson et al., 2022).
  - Task-vector read/write in residual stream; linear probes for latent task parameters.
- Geometry:
  - Fisher/Hessian trace, SAM sharpness, and local-entropy estimates; compare across conditions at matched perplexity.

4.5 Statistical analysis and falsification criteria
Predictions:
- Monotonicity (within regime): At matched perplexity, ICL metrics increase with P-NE; failure to see a positive association across matched runs falsifies the claim.
- Source equivalence: Batch-size and explicit-noise runs with matched P-NE yield similar ICL gains; divergence implies source-specific effects.
- Phase sensitivity: Late-phase noise yields larger ICL gains than early-phase for equal P-NE; null result would bound the effect.
- Geometry/mechanism: Higher P-NE correlates with flatter geometry and stronger in-context circuitry; absence weakens the mechanism.

Statistical plan:
- Pre-register metrics and stopping rules.
- Use multiple seeds; mixed-effects models with run-level random effects.
- Correct for multiple comparisons across tasks.

5. Threats to Validity and Limitations
- Optimizer and schedule dependence: Results may be specific to AdamW and chosen schedules; we will test SGD+momentum as a sensitivity analysis.
- Measurement error in P-NE: Use multiple estimators (diagonal variance, Hutchinson probes, noise scale proxies) and report agreement.
- Data dependence: Without sufficient task/format diversity, increasing noise may not produce ICL; we will vary data mixtures to illustrate scope.
- Confounded stochasticity: Dropout and data-order randomness can mimic noise effects; we will ablate and report separately.
- Over-strong monotonicity: We expect monotonic trends within a range; extreme noise can harm optimization. We will report the operating regime.
- Generalizability: Findings may not extend to non-Transformer architectures (e.g., RNNs) or non-LM objectives (e.g., RL fine-tuning); future work could explore these.

6. Implications
If supported, the results provide:
- A tunable lever for ICL independent of model/data scale (e.g., small-batch training or late-phase SGLD-style fine-tuning to boost ICL).
- A unifying view connecting ICL to optimizer-induced implicit bias and flatness.
- Practical guidance on noise schedules and batch sizing to engineer context-adaptive capability with better compute efficiency.

7. Conclusion
We propose and test an optimizer-aware causal account of ICL: integrated, preconditioned gradient noise imposes an information cost on parameters, biasing solutions toward context-adaptive computation. The protocol and metrics aim to cleanly separate noise effects from language modeling performance and other confounds. Confirmation would make ICL a controllable property of training dynamics and clarify the mechanisms by which optimization statistics shape emergent behaviors.

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
