Minor Revisions

Title: Noise Alignment for In-Context Learning: Theory, Counterfactual Noise Replay, and Scalable Preconditioned Optimization
> Note: The paper argues that the structure of randomness (noise) during training can be designed to match the structure of tasks seen in prompts, which helps models learn to perform inference “in context.” It provides theory, a tool to predict effects of changing optimizer noise without retraining, and a practical optimizer that implements this idea.

Abstract
In-context learning (ICL) emerges in large language models (LLMs) without explicit gradient updates, yet its training-time causes remain unclear. We identify and formalize a Noise Alignment Principle: when the covariance of training-time gradient noise aligns with the covariance structure of the task family encountered in prompts, the network preferentially acquires circuits that implement in-context Bayesian inference. We make three contributions. First, we give a provable account in a family of sequence-to-sequence tasks: for deep linear attention models trained with preconditioned stochastic gradient dynamics, if the injected Langevin noise covariance matches the prompt-conditioned feature covariance of the task family, the learned mapping implements in-context ridge regression on new prompts. Second, we introduce Counterfactual Noise Replay (CNR), a scalable procedure that predicts the effect of changing the optimizer’s noise preconditioner without retraining, by linearizing the training dynamics and estimating gradient-noise covariances from logs via randomized trace estimators. We provide finite-horizon error bounds under a Lipschitz-Hessian assumption. Third, we instantiate Noise-Aligned Preconditioning (NAP), a practical optimizer that adapts per-parameter noise to target covariances via low-rank sketches plus diagonal correction, and demonstrate that NAP increases ICL accuracy and data efficiency on synthetic meta-learning suites, algorithmic pattern induction, and few-shot tasks for medium-scale transformers, while preserving perplexity and scaling favorably with model size. Ablations with CNR, randomized interventions, and circuit-level perturbations support a causal link between noise alignment and ICL.
> Note: Key ideas in simple terms:
> - Gradient noise covariance: a matrix describing how random gradient fluctuations co-vary across parameters during training.
> - Task-family covariance: a matrix summarizing how task-specific parameters vary across prompts.
> - Principle: If these two covariances “match,” the model tends to learn internal computations that perform Bayesian inference from examples within a prompt (e.g., like doing ridge regression on the fly).
> - Contributions:
>   1) Theory in a simplified (linear) transformer shows aligned noise leads to in-context ridge regression.
>   2) A tool (CNR) to estimate “what if” we had used different noise—without retraining—by linearizing training.
>   3) A practical optimizer (NAP) that shapes noise to align with estimated task covariances, improving few-shot/ICL results without harming standard language modeling.

1. Introduction
In-context learning (ICL) allows LLMs to perform task adaptation within a prompt through forward passes alone. Despite extensive empirical observations, the training-time mechanisms that drive ICL’s emergence remain contested. Prior work links batch size, gradient noise, and generalization, but provides limited causal specificity for ICL circuits, and lacks tools to evaluate optimizer noise choices without prohibitive retraining.
> Note: ICL means the model adapts from examples in the prompt without updating weights. The open question is: what about training pushes models to develop this ability? Past work connected noise and generalization, but not specifically the circuits responsible for ICL, nor how to test optimizer noise choices efficiently.

We propose that a key driver of ICL is alignment between the second-order structure of training-time gradient noise and the second-order structure of the task family realized in prompts. Informally: when stochastic optimization injects noise whose covariance matches the covariance of “fast” task variation across prompts, training sculpts representations that support on-the-fly inference of task parameters at test time.
> Note: “Second-order structure” = covariance matrices. Matching the optimizer’s noise covariance to how tasks vary across prompts encourages the network to encode sufficient statistics to infer task-specific parameters from the prompt.

Our contributions:
- Theory: In a deep linear transformer surrogate and in a class of linear-Gaussian sequence tasks, we show that preconditioned Langevin SGD with aligned noise yields solutions that perform in-context Bayesian ridge regression on novel prompts.
- Methodology: We introduce Counterfactual Noise Replay (CNR) to predict, without full retraining, how different noise preconditioners would have altered a given training run.
- Algorithm: We present Noise-Aligned Preconditioning (NAP), a practical optimizer that shapes gradient noise to match online estimates of prompt-level task covariance, implementable via lightweight sketches and compatible with standard training pipelines.
- Evidence: Across synthetic and medium-scale transformer experiments, NAP improves ICL while maintaining language modeling quality. CNR-guided selection of preconditioners anticipates these gains and supports causal claims via randomized noise-switch and circuit knockout tests.
> Note: Why this matters:
> - Theory provides causal mechanism under simplifying assumptions.
> - CNR reduces costly retraining by simulating the effect of different noise designs.
> - NAP offers a drop-in optimizer modification that can boost few-shot/ICL while preserving perplexity.
> - Experiments and interventions aim to test causality, not just correlation.

2. Background and Problem Setup
We consider episodic pretraining where each minibatch is a mixture over latent tasks τ ∼ P(τ). A prompt comprises K input-output pairs {(xk, yk)}K and a query xq; the model predicts yq. In the linear-Gaussian family, yk = φ(xk; θ)⊤βτ + ϵk with βτ ∼ N(0, Σβ) and ϵk ∼ N(0, σ2). A network trained on heterogeneous prompts ideally supports in-context inference of βτ from {(xk, yk)}K at test time.
> Note: Definitions:
> - τ: a latent (hidden) task sampled from distribution P(τ).
> - K: number of labeled examples in a prompt.
> - (xk, yk): kth input-output example; xq/yq: query input/output.
> - φ(x; θ): feature vector (depends on input x and current model parameters θ).
> - βτ: task-specific weight vector drawn from a Gaussian N(0, Σβ), where Σβ is the task covariance matrix.
> - ϵk: observation noise, Gaussian with variance σ^2.
> - Goal: learn model internals so it can infer βτ from K examples in the prompt and predict yq.

Stochastic gradient descent with preconditioning P(θ) and injected noise can be approximated by a stochastic differential equation (SDE):
dθt = −P(θt)∇L(θt) dt + √(2T) B(θt) dWt
with noise covariance BTB ≈ PΣgP, where Σg is the gradient covariance induced by data sampling. We study how shaping BTB affects the learned solution and the emergence of ICL.
> Note: Equation terms:
> - θt: parameters at time t.
> - L(θ): loss function.
> - P(θ): preconditioner matrix (rescales gradients; e.g., Adam-like).
> - T: “temperature” scaling the noise level.
> - B(θ): matrix controlling injected noise; dWt: Wiener process increment (Gaussian noise).
> - BTB: noise covariance matrix (how noise correlates across parameters).
> - Σg: covariance of stochastic gradients due to minibatch sampling.
> - Idea: By designing B (and thus BTB), we control the distribution of parameter noise during training, which can bias learned solutions.

3. The Noise Alignment Principle
We formalize the intuition that aligning training-time gradient noise with the prompt-conditioned task covariance encourages solutions that implement in-context Bayesian inference.
> Note: The principle states: choose noise covariance during training to match the statistical variability of tasks in prompts, so the model learns to infer task parameters from prompt examples.

Assumptions:
- A1 (Task family): Prompts are generated by linear-Gaussian tasks with βτ ∼ N(0, Σβ) and features φ(x; θ) that are approximately linear in θ over the training trajectory.
- A2 (Preconditioning): The optimizer uses a constant or slowly varying preconditioner P and injects Gaussian noise with covariance Λ ≈ PΣ∗P for a target Σ∗.
- A3 (Episodic mixing): Minibatches are sampled across tasks such that the empirical gradient covariance satisfies Σg ≈ Eτ[Cov(∇ℓτ(θ))].
> Note: Plain meaning:
> - A1: Tasks follow a linear model with Gaussian priors; features change roughly linearly as parameters update.
> - A2: The preconditioner P is stable, and injected noise covariance Λ is designed to match a target Σ∗ (ideally Σβ or feature covariance).
> - A3: Gradient noise reflects averaging over task variation, so its covariance approximates the across-task gradient covariance.

Theorem 1 (Aligned noise yields in-context ridge regression). Consider a deep linear attention model whose forward map over a prompt implements ŷq = w⊤H(φq, {φk, yk}) for some linear operator H induced by attention. Train with preconditioned Langevin dynamics at temperature T and noise covariance Λ = PΣβP in the last-layer block. In the stationary regime and under A1–A3, the learned operator H implements the posterior mean of ridge regression with prior β ∼ N(0, Σβ) and noise σ2/T on novel prompts:
ŷq ≈ φq⊤ Σβ Φ(Φ⊤ΣβΦ + (σ2/T)I)−1 y
where Φ = [φ1, …, φK]. Proof sketch: Under linearization, the stationary distribution over last-layer weights solves a convex stochastic optimization with T-weighted quadratic regularization shaped by Λ. The optimal predictor equals the Bayes estimator for the task prior Σβ. The attention mechanism realizes the requisite sufficient statistics via prompt tokens.
> Note: Definitions and intuition:
> - ŷq: predicted output for query.
> - w: readout weights; H(·): linear operator computed by attention over prompt.
> - Λ: injected noise covariance; set to P Σβ P in last-layer parameters (so noise aligns with task prior).
> - Stationary regime: long-run behavior under the SDE (training converged in distribution).
> - Formula:
>   - φq: feature vector for query xq.
>   - Φ: matrix of prompt features [φ1, …, φK].
>   - y: vector of prompt outputs [y1, …, yK].
>   - Σβ: prior covariance of task parameters β.
>   - σ^2/T: effective observation noise (original σ^2 scaled by temperature T).
>   - I: identity matrix.
>   - ⊤: transpose.
> - Meaning: The learned mapping performs the Bayes-optimal ridge regression prediction given prior Σβ and data {(φk, yk)}K.
> - Why it matters: It connects a training-time choice (noise alignment) to a specific in-context algorithm (Bayesian ridge regression) the model executes at test time.

Corollary 1 (Misalignment penalty). If Λ deviates from PΣβP by Δ, the excess risk in in-context prediction is bounded by a term proportional to ||Σβ−P−1ΛP−1||F, with constants depending on K and σ2.
> Note: Definitions:
> - Δ: mismatch in covariance, Δ = Λ − PΣβP.
> - P−1: inverse of P.
> - ||·||F: Frobenius norm (root sum of squares of matrix entries).
> - Message: The further the injected noise covariance (mapped back through P) is from Σβ, the larger the performance penalty, with sensitivity shaped by shots K and noise σ^2.

Theorem 2 (Layer-wise shaping). When Λ is applied in intermediate blocks that control prompt aggregation, alignment with the covariance of prompt features E[φφ⊤] yields heads that linearly read out in-context estimates; misalignment shifts capacity toward static pattern-memorization.
> Note: Aligning noise in layers that aggregate prompt information to the feature covariance E[φφ⊤] encourages heads to compute the sufficient statistics needed for in-context estimation, rather than memorizing surface patterns. This clarifies where to inject aligned noise for maximal ICL benefit.

4. Counterfactual Noise Replay (CNR)
Objective: Predict the effect of replacing the noise preconditioner Λt along a recorded training trajectory {θt} without retraining.
> Note: Goal: Given logs from a past training run (parameters over time, gradients, etc.), estimate how outcomes would change if a different noise covariance schedule Λ̃t had been used—saving compute.

Method:
- Linearize dynamics: θt+1 ≈ θt − ηtP∇L(θt) + ξt, with ξt ∼ N(0, 2ηtTΛt).
- Counterfactual perturbation: For an alternative preconditioner Λ̃t, the cumulative parameter change is
ΔθT ≈ ∑t JT←t Δηt + ∑t JT←t (ξ̃t − ξt),
where JT←t is the state transition Jacobian along {θt}.
- Efficient Jacobian-vector products: Use checkpointed reverse-mode and randomized Hutchinson vectors to estimate JT←t v without storing full Hessians.
- Gradient-noise logging: Maintain low-rank sketches of per-layer gradient covariance via streaming PCA; estimate trace terms for ξt resampling.
- Output: Predict task metrics f(θT + ΔθT) via first/second-order Taylor expansions with variance estimates.
> Note: Definitions:
> - ηt: learning rate at step t.
> - ξt: injected noise at step t, Gaussian with covariance 2ηtTΛt.
> - Λt, Λ̃t: actual vs. counterfactual noise covariance at step t.
> - JT←t: Jacobian of the parameter update map from step t to T (how a small change at step t propagates to the end).
> - Hutchinson vectors: random probes to estimate matrix traces/products efficiently.
> - Streaming PCA/sketches: online low-rank approximations to gradient covariance.
> - f(·): evaluation metric (e.g., ICL accuracy); use Taylor expansion to approximate its change.
> - Why it matters: CNR provides a practical, approximate way to evaluate noise design choices using existing training logs.

Theory:
Proposition 1 (Finite-horizon error bound). Suppose ∇2L is L-Lipschitz in a neighborhood of {θt}, and the spectral norm of JT←t is bounded. Then the CNR prediction error for smooth metric f satisfies
E[(f̂ − f)2] ≤ C1 ∑t ηt2 ||Λ̃t − Λt||F2 + C2 ∑t ηt3 L2,
with constants C1, C2 depending on Jacobian norms and f’s smoothness.
> Note: Definitions:
> - ∇2L: Hessian (matrix of second derivatives) of the loss.
> - L-Lipschitz: Hessian does not change too quickly; its variation is bounded by L.
> - Spectral norm: largest singular value (maximal amplification factor).
> - f̂: CNR-predicted metric; f: true metric under counterfactual.
> - C1, C2: constants depending on model dynamics and metric smoothness.
> - ||·||F: Frobenius norm.
> - Message: Under smoothness and bounded-sensitivity assumptions, CNR’s mean-squared prediction error is controlled; errors grow with the squared difference in noise covariances and with curvature (L).

Practically, CNR enables rapid screening of Λ choices, selection of promising preconditioners, and sensitivity analysis via randomized interventions.
> Note: Use cases:
> - Rank candidate noise designs by predicted benefit.
> - Quantify uncertainty and sensitivity.
> - Design targeted interventions (e.g., layer-wise swaps) before committing to full training.

5. Noise-Aligned Preconditioning (NAP)
NAP implements the Noise Alignment Principle in standard training:
> Note: NAP is the practical optimizer that shapes per-parameter noise to match estimated task or feature covariances during training.

- Target covariance estimation: Online, per-layer estimation of prompt-level task covariance Σ̂β or feature covariance Σ̂φ via:
  - episodic batching with K > 1 exemplars per pseudo-task formed from adjacent spans,
  - low-rank sketches (rank r ≪ d) using incremental Oja updates,
  - diagonal correction to preserve per-parameter stability.
> Note: Definitions and why:
> - Σ̂β: estimated covariance of task parameters; Σ̂φ: estimated covariance of features φ.
> - r: sketch rank; d: number of parameters/feature dimension.
> - Oja updates: streaming method to estimate top eigenvectors (principal components).
> - Diagonal correction: keep per-parameter scales stable (prevents destabilizing updates).
> - Purpose: Efficiently estimate the most important covariance directions online with small overhead.

- Preconditioner: P is chosen from a stable family (e.g., Adafactor/Adam diagonal blocks) and Λt := P (UΣrU⊤ + λI) P, where UΣrU⊤ is the rank-r sketch.
> Note: Definitions:
> - P: preconditioner (e.g., diagonal matrix from Adam/Adafactor statistics).
> - UΣrU⊤: low-rank approximation of target covariance (U: eigenvectors; Σr: top r eigenvalues).
> - λI: small diagonal term (λ > 0) for numerical stability.
> - Λt: injected noise covariance shaped to match the estimated target in the space scaled by P.

- Noise injection: Implement via SGLD-style gradient perturbations or temperature scaling of minibatch stochasticity using microbatching to achieve the target Λt.
> Note: Two practical routes:
> - SGLD-style: explicitly add Gaussian noise to gradients with covariance 2ηtTΛt.
> - Temperature/microbatch: adjust batch structure to modulate inherent minibatch noise to approximate Λt.

- Schedule: Increase alignment strength over training (curriculum), with temperature T decayed to control stationary variance.
> Note: Early training can use stronger alignment to shape representations; decaying T later reduces variance so training converges.

Compatibility: NAP composes with Adam/Adafactor, gradient clipping, and mixed precision; overhead is minor (rank r = 8–64).
> Note: Practicality: NAP is intended as a drop-in addition with modest compute/memory costs.

6. Experiments
We evaluate three questions: (Q1) Does NAP improve ICL without harming language modeling? (Q2) Does CNR accurately predict the effect of preconditioners and support causal claims? (Q3) What circuits emerge under aligned noise?
> Note: High-level evaluation plan: measure ICL gains vs. perplexity, validate CNR predictions, and inspect internal circuits for mechanistic evidence.

6.1 Synthetic meta-learning suite
- Tasks: Linear regression and classification with heteroscedastic priors; sequence copying and simple algorithmic patterns.
- Models: 2–8 layer transformers, 50–300M parameters.
- Findings: NAP matches the theoretical Bayes predictor under correct Σβ and degrades gracefully under misspecification; CNR selects near-optimal Λ among candidates with strong rank correlation to realized gains. Improvements are largest when K is small (few-shot).
> Note: Why this matters:
> - Synthetic settings allow ground-truth comparisons to Bayes-optimal solutions.
> - Results suggest NAP aligns learned behavior with theory, especially in few-shot regimes where in-context inference is most valuable.
> - CNR’s ability to rank preconditioners supports its utility for design.

6.2 Medium-scale language models
- Setup: Pretrain 1–7B parameter decoder-only models on standard corpora with episodic minibatching (K=4–8 exemplars) and apply NAP in attention and MLP output blocks. Baselines: AdamW, SGLD, Adadiffusion-like noise, and batch-size scaling.
- Metrics: Perplexity, ICL tasks (pattern completion, arithmetic with demonstrations, table-to-text), few-shot accuracy on standardized benchmarks, and OOD generalization with distribution shift in prompt statistics.
- Results: NAP increases few-shot accuracy while maintaining perplexity within baseline variance. Gains persist under ablations that swap NAP off late in training, supporting a training-time causal effect rather than a test-time artifact.
> Note: Interpretation:
> - NAP boosts in-context abilities without sacrificing language modeling quality (perplexity).
> - Turning NAP off late but still seeing gains points to training-time shaping of representations, consistent with the Noise Alignment Principle.

6.3 Causal analyses
- CNR validation: For held-out checkpoints, CNR’s predicted metric deltas under Λ swaps correlate with realized deltas after short finetunes; prediction intervals are well-calibrated.
- Randomized interventions: Mid-training noise-preconditioner swaps produce sharp changes in ICL metrics predicted by CNR; swapping only in non-ICL-relevant layers attenuates effects.
- Circuit knockouts: Temporarily ablating specific attention heads reduces ICL more in NAP models; linear probes reveal heads computing prompt-level sufficient statistics, consistent with Theorem 1.
> Note: Why these tests:
> - Correlation and calibration of CNR predictions support its reliability.
> - Targeted noise swaps and layer-specific interventions probe causality: changing aligned noise where it should matter changes ICL; changing it where it shouldn’t has smaller effects.
> - Circuit-level evidence (head ablations, probes) links NAP to mechanisms computing statistics needed for in-context regression.

7. Related Work
- SGD as approximate Bayesian inference and SDEs: Mandt et al.; Chaudhari & Soatto; Smith & Le; noise-induced implicit regularization.
- ICL emergence and mechanisms: Brown et al.; Wei et al.; Olsson et al.; meta-learning in transformers (von Oswald et al.; Dai et al.).
- Preconditioning and second-order methods: Adagrad/Adam; K-FAC; Shampoo; adaptive noise in diffusion-inspired optimizers.
- Causal analysis of training: influence functions; counterfactual training; mechanistic interpretability interventions.
> Note: Positioning:
> - Builds on viewing SGD as noisy dynamics that can approximate Bayesian posteriors.
> - Connects to work on emergent ICL and meta-learning in transformers.
> - Extends adaptive optimization by shaping noise, not just gradients.
> - Uses causal probes (counterfactuals, interventions) to argue mechanism.

Our work uniquely links aligned gradient noise to in-context Bayesian inference with proofs in transformer surrogates, introduces CNR for optimizer noise counterfactuals, and delivers a practical, scalable NAP optimizer guided by this theory.
> Note: Summary claim: theoretical link + counterfactual tool + practical optimizer, specifically targeted at ICL.

8. Limitations
- Theory applies to linear-Gaussian task families and deep linear approximations; extensions to nonlinear regimes require further work.
- Online covariance estimation can be noisy under nonstationary corpora; adaptive rank selection and robust sketches may help.
- CNR relies on local linearization; far-from-trajectory counterfactuals can be inaccurate.
> Note: Practical takeaways:
> - The strongest guarantees are in simplified settings; real transformers are nonlinear.
> - Estimation and approximation errors can affect NAP and CNR; methods to adapt rank and improve robustness are needed.
> - CNR is best for modest deviations from the observed training trajectory.

9. Broader Impacts
Shaping optimizer noise to favor ICL could amplify both helpful and harmful in-context adaptation. We recommend combining NAP with data governance to avoid encoding sensitive correlations and with evaluation for prompt injection robustness.
> Note: Impact considerations:
> - Better ICL can improve few-shot performance but might also internalize spurious or sensitive correlations.
> - Security and safety evaluations (e.g., against prompt injection) remain important.

10. Conclusion
We present a causal and algorithmic account of how training-time noise structure shapes in-context learning. Aligning gradient noise covariance with task-family covariance yields in-context Bayesian inference in theory, a practical NAP optimizer in practice, and CNR tools to causally probe noise design without retraining. We hope this establishes a principled bridge between optimizer design and emergent ICL.
> Note: Final message: Design the training noise to match task variability, and you can both explain and improve in-context learning, with tools to test and apply the idea at scale.

Reproducibility
We provide reference implementations of NAP and CNR with:
- rank-r sketching modules for per-layer covariance,
- SGLD-compatible noise injection wrappers for Adam/Adafactor,
- logging hooks for gradient covariance and CNR replay.
> Note: What’s provided:
> - Code for low-rank covariance estimation (rank r).
> - Integrations to inject noise with common optimizers.
> - Logging needed for CNR to simulate counterfactual noise.

Key Theoretical Proof Sketches (Appendix overview)
- A: Derivation of stationary solution under preconditioned Langevin dynamics and its equivalence to a Bayes estimator with prior Σβ.
- B: Bounds for CNR linearization error under Lipschitz-Hessian and bounded Jacobian assumptions.
- C: Extension of Theorem 1 to block-diagonal preconditioners and mixed-layer alignment.
> Note: Appendix contents:
> - A: Shows how noise-shaped Langevin dynamics imply a predictor matching Bayesian ridge regression under prior Σβ.
> - B: Provides conditions under which CNR’s approximation error is bounded.
> - C: Generalizes the main theorem to more realistic optimizer structures (e.g., per-layer blocks).
