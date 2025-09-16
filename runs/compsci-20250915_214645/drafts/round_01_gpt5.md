Title: Gradient Noise Governs In-Context Learning

Abstract
We propose a causal link between stochastic gradient noise during pretraining and in-context learning (ICL) in large language models. Holding architecture, data, and final perplexity fixed, we hypothesize that larger integrated gradient noise—arising from smaller effective batch sizes or explicit Langevin noise—induces stronger ICL. The mechanism is that stochastic training acts as an implicit Bayesian regularizer on parameters, shifting task-specific information storage from weights into fast activations that can be updated by context at inference time. We formalize this connection via SGD-as-sampling and a linear-Gaussian meta-learning model, derive measurable noise statistics, and state falsifiable predictions. We outline an experimental program to manipulate noise independently of loss and to measure monotonic scaling of ICL with integrated noise. If supported, the result reframes ICL as a controllable training-statistics phenomenon rather than an emergent property of scale alone.

Introduction
Transformers trained on next-token prediction can perform in-context learning: they adapt their behavior based on a few examples in the prompt without gradient updates. While ICL strength correlates with scale and data, the training causes remain unclear. We advance the hypothesis that the magnitude of stochastic gradient noise (SGN) during pretraining is a primary driver of ICL, independent of final training loss.

Rationale:
- SGD as approximate Bayesian sampler: With small batches and finite step size, SGD approximates Langevin dynamics with an effective temperature determined by update noise. This induces an implicit prior/regularizer on parameters favoring flatter, lower-information solutions.
- Parameter–activation tradeoff: Next-token prediction on heterogeneous sequences can be solved by (i) encoding task-specific mappings in parameters or (ii) inferring task-specific structure from the observed context within the sequence and applying it via attention. Higher noise penalizes parameter-specific encodings and incentivizes solutions that rely on context-derived fast variables—i.e., ICL.
- Representational consequence: Noisy training promotes circuits that read examples in the prompt, estimate latent task variables online (e.g., class centroids, linear maps), and apply them within the same forward pass.

Core claim
For models matched on architecture, data, and final perplexity, in-context learning ability increases monotonically with the integrated gradient noise experienced during pretraining.

Method
Definitions
- Gradient noise magnitude: For a parameter vector θ at step t, let g_t be the full-batch gradient and ĝ_t the mini-batch gradient. The gradient covariance C_t = Cov(ĝ_t). A practical scalar summary is the gradient noise scale GNS_t = trace(C_t) / ||g_t||^2, or equivalently the critical batch size where noise matches signal.
- Integrated noise exposure: NE = sum over training steps t of η_t^2 · trace(C_t), where η_t is the learning rate. NE captures total stochasticity injected into parameters over training and scales with both variance and step size. Normalized variants (per token or per effective epoch) are also useful.
- Effective temperature: Under standard approximations, SGD with noise has a stationary distribution near minima with covariance proportional to η · C. Larger NE implies a higher effective sampling temperature over training.

Mechanism sketch
- Implicit Bayesian objective: Stochastic training approximately minimizes E_data[L(θ)] + λ · KL(q(θ) || p(θ)), where q is the implicit posterior induced by SGD noise and p a prior determined by initialization and weight decay; λ increases with effective temperature. Thus higher noise enforces a stronger information bottleneck on parameters.
- Two-channel predictor: Decompose the network’s predictive computation into a parameter channel f_param(x; θ) and a context-adaptive channel f_ctx(x; context; θ). The latter implements online estimation of latent task variables from the prompt (e.g., via attention, key–value caching) and applies them to predict. Increasing λ shifts optimal solutions toward higher reliance on f_ctx because f_param incurs KL cost while f_ctx leverages activations that carry no parameter KL penalty.

Analytical toy model (linear-Gaussian, sketch)
- Data: Sequences consist of tasks with latent parameter w ~ N(0, σ_w^2 I). Within a sequence, tokens (x_i, y_i) follow y_i = w^T x_i + ε.
- Model class: A transformer can emulate ŷ = α · ŵ_prompt^T x + (1−α) · v^T x, where ŵ_prompt is the ridge estimator computed from the past few (x, y) pairs in the same sequence via attention, and v are global parameters.
- Training with SGLD yields an objective equivalent to minimizing expected predictive loss plus a penalty β ||v||^2 (from the KL term). The optimal mixing weight α increases monotonically with β, and the optimal ŵ_prompt approaches the Bayesian posterior mean (ridge) as context length grows. Thus higher noise (larger β) increases reliance on in-context estimation and improves few-shot adaptation without changing asymptotic perplexity given enough data.
- Prediction: For fixed final next-token loss, models experiencing higher NE during training exhibit steeper in-context learning curves on synthetic regression/classification and real NLP few-shot tasks.

Measurable predictions
- Monotonicity: ICL metrics (e.g., few-shot accuracy, in-context regression slope) increase with NE, controlling for final validation perplexity.
- Equivalence of manipulations: Decreasing effective batch size, increasing learning rate proportionally, or adding explicit Langevin noise produce similar ICL gains when matched on NE and final loss.
- Phase sensitivity: Noise applied during phases when attention circuits specializing in within-sequence estimation form (typically later training) contributes more to ICL than equally sized early noise, for equal final loss.
- Flatness and information: Higher NE models converge to flatter minima, lower Fisher trace, and lower estimated I(θ; data), alongside stronger ICL.

Experiments (falsification plan)
Design
- Architectures: Identical decoder-only transformers across runs (e.g., 350M, 1.3B parameters), standard tokenizer, same data mixture and curriculum.
- Controlled factors: Same optimizer family, weight decay, LR schedule shape, initialization, training tokens. Only noise-generating factors vary.
- Noise manipulations:
  1) Batch size sweep: B ∈ {256, 1k, 4k, 16k} tokens per step with compensating LR scaling to reach the same target perplexity.
  2) SGLD: Add Gaussian Langevin noise to updates with variance chosen to match target NE at larger batch sizes.
  3) Noise scheduling: Concentrate noise early vs late vs uniform while matching total NE and final loss.
- Matching final loss: Train each run to the same target validation perplexity plateau (±0.05), allowing different step counts. If needed, adjust LR or apply extra fine-tuning to equalize.

Measurement
- Gradient noise estimation: Periodically (e.g., every 5k steps), freeze θ and estimate trace(C_t) using K micro-batch gradients on held-out shards; compute GNS_t and accumulate NE. Record per-layer variants.
- ICL benchmarks:
  - Synthetic: In-context linear regression and classification (rotating tasks per sequence) with evaluation of posterior-optimal slope recovery vs shots; algorithmic tasks (parity, modular addition) with few-shot generalization.
  - NLP: Few-shot accuracy on MMLU, ARC, HellaSwag; cloze tasks (LAMBADA) with k-shot examples; instruction induction tasks.
  - Quantitative ICL score: Area under performance-vs-shots curve; logit lens–based regression on in-context posterior features.
- Controls:
  - Non-ICL generalization: Zero-shot tasks to check that overall capability is matched at equal perplexity.
  - Capacity usage: Parameter flatness (Hessian trace proxies), Fisher information, and Minimum Description Length estimates to test the information bottleneck prediction.
  - Representation probes: Does the model compute prompt-sufficient statistics (e.g., class centroids, linear coefficients) in identifiable attention heads? Compare strength across noise levels.

Statistical analysis
- Primary test: Regress ICL score on NE across runs; test monotonic trend (isotonic regression) with bootstrapped confidence intervals.
- Equivalence: Compare batch-size vs SGLD runs matched on NE; test for indistinguishable ICL outcomes.
- Phase sensitivity: Two-factor ANOVA on ICL with factors {NE total, phase allocation}.

Outcomes
- Support: Significant monotonic increase of ICL with NE while zero-shot holds constant; equivalence across manipulation types; larger flatness/MDL penalties and stronger in-context statistic-computing circuits in high-NE models.
- Falsification: No monotonic relationship when final loss is matched; or ICL differences vanish after adjusting for subtle loss or data differences; or noise phase has no effect.

Discussion
Implications
- Training-statistics lever: ICL strength is tunable via gradient noise, offering a compute-efficient alternative to scale/data-only approaches.
- Mechanistic unification: Connects ICL to implicit Bayesian regularization and flat-minima bias of noisy optimization, explaining why ICL emerges with scale (which often co-varies with noisier long schedules and small effective batches).
- Design guidance:
  - Use increased noise (smaller batches or calibrated Langevin noise) late in training to boost ICL without degrading perplexity.
  - Consider explicit NE targets as part of training recipes; report NE alongside loss to characterize models’ ICL potential.

Relations to prior observations
- Aligns with empirical ties between small batches and generalization, and with reports that SGLD improves calibration and adaptation.
- Extends meta-learning views of transformers: higher noise pushes solutions toward architectures that implement inner-loop inference over prompts.

Limitations
- Matching final loss exactly is challenging; small perplexity differences can confound ICL. Requires careful early stopping and schedule tuning.
- SGD-as-sampling approximations are imperfect in nonconvex, nonstationary regimes; constants linking NE to effective temperature are approximate.
- Noise structure matters: anisotropy and layer-wise effects may influence which circuits emerge, beyond scalar NE.
- Data effects: Document structure and task heterogeneity in pretraining data likely interact with noise; our theory assumes sufficient within-sequence task diversity to benefit from ICL.
- Generality: While argued for transformers on language data, other modalities and architectures (e.g., diffusion U-Nets) require separate validation.

Conclusion
We hypothesize and provide a falsifiable program for a causal link between stochastic gradient noise during pretraining and in-context learning in language models. By viewing SGD as imposing an information bottleneck on parameters, we argue that higher integrated noise shifts computation toward context-adaptive mechanisms, yielding stronger ICL at fixed perplexity. The proposed experiments isolate noise from loss and architecture, offering clear support-or-refute outcomes. If confirmed, this reframes ICL as a controllable property of training statistics and introduces practical levers—noise schedules and SGLD—to reliably elicit in-context abilities.
