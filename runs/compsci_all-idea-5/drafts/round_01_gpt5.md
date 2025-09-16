Major Revisions

Title: Target-Entropy Decoding: Closed-Loop Control of Uncertainty for Large Language Model Inference

Abstract
Sampling from large language models (LLMs) is governed by heuristic controls such as temperature and nucleus thresholds, which indirectly and inconsistently regulate uncertainty. We propose Target-Entropy Decoding (TED), a closed-loop decoding method that adjusts temperature at each generation step to match a prescribed per-step Shannon entropy target. For fixed logits, the entropy of the softmax distribution is strictly increasing in temperature; TED leverages this monotonicity to solve, per token, for the unique temperature that attains the target entropy via efficient root-finding (bisection or damped Newton), adding negligible overhead relative to the forward pass. We design simple entropy schedules (constant, position-dependent, and prompt-adaptive) and evaluate on open-ended writing and factual QA. We hypothesize—and will falsify via controlled experiments on small open-source models—that TED reduces repetition and hallucination at equal token budgets versus fixed-temperature and nucleus sampling. We provide theoretical guarantees (existence/uniqueness of the temperature solution and a closed-form derivative of entropy with respect to temperature) and discuss relations to prior perplexity- or surprise-control methods (e.g., Mirostat), highlighting that TED controls the full-distribution entropy deterministically and supports arbitrary schedules.

Introduction
Sampling policies strongly influence the quality of LLM outputs. Fixed temperature and nucleus sampling (top-p) trade off diversity and coherence but provide no direct control over the model’s uncertainty. High temperature can induce hallucinations; low temperature can cause repetition or premature convergence. We study entropy as the primary control signal during decoding. The Shannon entropy of the predictive distribution captures the model’s total uncertainty, not just the tail mass or the surprise of a single sampled token.

We introduce Target-Entropy Decoding (TED), which:
- Sets a per-step target entropy H*(t) and tunes temperature T online so that H(softmax(z/T)) ≈ H*(t).
- Uses closed-loop, deterministic control with monotonicity guarantees to compute T efficiently at each step.
- Supports flexible schedules to shape uncertainty across a response (e.g., high early to diversify, low later to commit).
- Is model-agnostic, easy to implement, and adds minimal overhead.

Contributions
- A principled decoding objective: match a target Shannon entropy at each step via temperature control.
- Theory: continuity, strict monotonicity, existence/uniqueness of T for a given target, and a closed-form derivative enabling fast Newton updates.
- Algorithms: robust per-step solvers (bisection and damped Newton) with practical bracketing and numerical safeguards.
- Evaluation plan: falsifiable experiments on open-source models showing reduced repetition and hallucination at fixed token budgets versus fixed temperature, top-p, and Mirostat-like surprise control; ablations over entropy schedules.

Method
Problem setup
At decoding step t, with logits s ∈ R^V (vocabulary size V) from the LLM, define the temperature-scaled distribution:
p_i(T) = exp(s_i / T) / Z(T), Z(T) = Σ_j exp(s_j / T), T > 0.

Define the Shannon entropy H(T) = -Σ_i p_i(T) log p_i(T). Let H*(t) be the desired target entropy at step t. TED sets T_t to solve H(T_t) = H*(t) and samples y_t ∼ p(T_t).

Theoretical properties
Let β = 1/T and let μ(β) = E_p[s] where p depends on β. Then:
- Entropy identity: H(β) = -β μ(β) + log Z(β).
- Derivative with respect to β: dH/dβ = -β Var_p(s).
- Chain rule gives dH/dT = Var_p(s) / T^3.

Immediate consequences:
- Continuity and strict monotonicity: Var_p(s) ≥ 0 with equality only in the degenerate case (all logits equal after scaling). Thus H(T) is continuous, strictly increasing in T, with limits H(0+) = 0 and H(∞) = log V. Therefore, for any H* ∈ (0, log V), there exists a unique T solving H(T) = H*.
- Newton step: T_new = T - (H(T) - H*) / (dH/dT) = T - (H(T) - H*) T^3 / Var_p(s), with damping and clipping to maintain T > 0.

Algorithms
We implement two solvers:

1) Bracketed bisection (robust)
- Choose T_min = ε (e.g., 1e-3) and T_max such that H(T_max) ≥ H* (e.g., T_max = 50–200; in practice H(T_max) ≈ log V).
- Repeat for B iterations (e.g., B = 6–10): evaluate H at mid, shrink bracket by sign(H(mid) − H*).
- Stop when |H − H*| ≤ δ_H (e.g., 1e-3) or bracket small; use T_mid.

2) Damped Newton (fast)
- Initialize T with previous step’s T (warm start), or from a heuristic (e.g., T_init = 1).
- Iterate: compute H, Var_p(s), update T via damped Newton; project into [T_min, T_max]; stop when |H − H*| ≤ δ_H.
- If Newton fails (e.g., tiny Var), fall back to bisection.

Complexity and implementation notes
- Cost: Each solver iteration requires a pass over the logits to compute the temperature-scaled softmax, entropy, and variance. This adds O(BV) ops per token, but the dominant cost remains the model forward. With B ≤ 6 and V ≈ 30–50k, wall-clock overhead is typically <5% for GPU inference. Top-K approximation (e.g., K=5k–10k) can further reduce cost with negligible effect on H if K covers ≥99.9% mass.
- Numerical stability: subtract max logit before scaling; use float32 for normalization; cap T ∈ [1e-3, 200].
- Compatibility: TED can be combined with standard constraints (e.g., repetition penalties, bad-token masks). When combined with top-p or top-k, compute H and Var on the truncated, renormalized distribution—this changes the target’s interpretation but preserves monotonicity.

Entropy schedules H*(t)
We consider three families:
- Constant (TED-Const): H*(t) = h0.
- Linear ramp (TED-Ramp): H*(t) = h0 + α t up to a cap h_max or ramp-down later, enabling “explore then commit.”
- Prompt-adaptive (TED-Adapt): Select among preset schedules based on a quick classifier (e.g., factual vs creative prompt) or on early-step features (e.g., logit gap Δ = s_top1 − s_top2). For factual prompts, set a lower target; for creative prompts, higher early target tapering later.

Relation to prior work
- Temperature and nucleus sampling: These indirectly shape entropy but cannot target a specific entropy value. Nucleus (top-p) enforces tail mass, not global uncertainty.
- Mirostat and surprise control: Mirostat targets a desired expected surprise (negative log probability) of the sampled token via stochastic approximation and dynamic top-k. TED differs in that it controls the full-distribution Shannon entropy deterministically, admits exact per-step tracking with monotonicity guarantees, and supports arbitrary schedules. Empirically, surprise control and entropy control need not coincide; TED can be combined with or compared against Mirostat.

Experiments (falsification plan)
Goals
- Test whether controlling full-distribution entropy reduces repetition and hallucination while maintaining utility at fixed token budgets.
- Verify robustness across models and prompts.
- Compare against strong baselines.

Models
- Small open-source LMs: Llama-2-7B, Mistral-7B, Gemma-7B, Pythia-2.8B, Phi-2 (2.7B).
- Implementation via HuggingFace Transformers with custom sampler.

Tasks and datasets
- Open-ended generation: WritingPrompts, XSum (abstractive summaries), creative prompts from public eval sets.
- Factual QA: TruthfulQA, HaluEval, PubMedQA-lite or BioASQ snippets for domain stress.
- Long-form QA: NaturalQuestions-Open short answers to stress hallucination.

Baselines
- Fixed temperature T ∈ {0.7, 0.9, 1.0, 1.2}.
- Nucleus top-p ∈ {0.8, 0.9, 0.95} with T=1.
- Top-k ∈ {40, 100} with T=1.
- Mirostat v2 with standard parameters (target surprise τ ∈ {5, 7, 9}).
- Contrastive Decoding (if applicable to chosen models) for robustness check.

TED configurations
- TED-Const with h0 ∈ {2.5, 3.0, 3.5, 4.0} nats.
- TED-Ramp: h0=3.5 ramp to 2.5 over first 40 tokens (for factual); inverse for creative.
- TED-Adapt: binary prompt classifier (few-shot or zero-shot rule) to pick TED-Const(h0=3.0) for factual and TED-Ramp(high→low) for creative.

Metrics
- Repetition/diversity: Distinct-n (n=2,3), repetition rate (exact n-gram repeats), self-BLEU, mean run-length of repeated tokens.
- Factuality: TruthfulQA accuracy, HaluEval metrics, FactScore (or entity-level precision/recall), model-based hallucination detectors as auxiliary.
- Utility/quality: Human pairwise preference (crowd workers) on 300–500 prompts; automatic proxies (COMETKiwi or G-Eval with strong evaluator model held fixed).
- Calibration: Empirical entropy vs target tracking error; average sampled-token surprise; perplexity on held-out text to observe regularization effects.
- Efficiency: Tokens/s and incremental latency overhead.

Protocol
- Match token budgets across methods; sample 3–5 seeds per prompt.
- Hyperparameter tuning via a small validation set; report test performance at tuned settings.
- For TED, report tracking error |H − H*| and solver iteration statistics.
- Statistical tests: bootstrap CIs; paired tests for human preferences.

Ablations and analyses
- Solver: bisection vs Newton; effect of iteration cap B.
- Truncation: full softmax vs top-K entropy calculation.
- Schedule shape: const vs ramp vs adapt.
- Combination with top-p: compute entropy on truncated support; observe effects.
- Robustness across prompt types (factual vs creative).

Falsification criteria
- If TED does not reduce repetition metrics vs matched baselines at equal quality, or fails to reduce hallucination rates on TruthfulQA/HaluEval without harming utility, the core claim is falsified.
- If entropy tracking error remains large (>0.2 nats median) or induces substantial latency (>10%) on 7B models, practicality is undermined.

Discussion
Why entropy?
- Entropy measures the whole distribution’s uncertainty, not just tail mass or the realized token’s surprise. Matching a target entropy constrains the sampler to remain neither overconfident (risking repetition and brittle errors) nor overly diffuse (risking incoherence).
Control-theoretic perspective
- Monotone, differentiable mapping T ↦ H enables stable SISO control. Warm-started Newton provides fast convergence with negligible compute compared to the transformer forward pass.
Interaction with hallucination
- Lower target entropy on factual prompts biases toward confident peaks, reducing spurious alternatives. Conversely, medium entropy can prevent brittle commitment to an incorrect token early in a span by allowing exploration, then tapering to commit.
Schedule design
- Simple schedules already offer strong control. More advanced schedules could be learned (e.g., via offline RL from human preferences) or conditioned on uncertainty signals (logit gap, predictive entropy trend, or token-level risk classifiers).

Limitations
- Entropy is agnostic to semantics. Controlling uncertainty does not ensure truthfulness; it only modulates confidence. Miscalibrated models can remain confidently wrong at low entropy.
- Very high target entropy can degrade coherence; very low targets can resurrect exposure-bias loops. Schedules must be chosen carefully.
- Computing entropy on the full vocabulary adds small overhead; extreme V or CPU-bound inference may feel this more.
- Interactions with additional decoding heuristics (repetition penalties, constraints) can complicate the interpretation of the target entropy.

Conclusion
Target-Entropy Decoding reframes LLM sampling as closed-loop uncertainty control: set a desired entropy and solve for the temperature that achieves it at each step. With theoretical guarantees, negligible overhead, and flexible schedules, TED provides a principled alternative to heuristic decoding. Our experimental plan will test whether entropy control reduces repetition and hallucination versus strong baselines while preserving utility on small open-source models. If validated, TED offers a simple, general tool for shaping generation behavior across tasks and models.
