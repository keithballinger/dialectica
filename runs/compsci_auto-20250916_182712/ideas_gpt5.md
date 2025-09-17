1) Predictive KV Eviction via Future-Attention Surrogates
Summary: Future attention to cached tokens is predictable from current local signals, enabling proactive KV eviction with negligible accuracy loss.
For a smart layperson: During generation, the model “looks back” at past words; we cache those looks. We claim you can predict which old words will matter in the near future and discard the rest ahead of time. That saves memory and speed without changing the answer.
Falsification: Train a lightweight predictor on small models (e.g., Pythia/Mistral) to forecast whether each cached token will receive significant attention in the next Δ steps using current logits, margins, and recent attention. Replace standard cache with predictor-driven eviction and compare perplexity, exact-match, and speed to recency- and attention-mass baselines. If accuracy drops more than baseline at the same memory budget, the theory is false.
Novelty: Unlike prior reactive pruning based on past attention, this posits and tests learned forward-looking attention prediction for cache management.

2) Entropy-Optimal Adaptive Decoding
Summary: Local logit entropy and margin suffice to decide when additional search (beam/nucleus) improves output quality per unit compute.
For a smart layperson: When the model is “uncertain,” extra effort exploring alternatives helps; when it’s confident, it’s wasted. We claim a simple uncertainty score from the model’s next-word probabilities tells you when to spend extra compute. This yields better answers for the same time.
Falsification: Implement a controller that switches among greedy, top-p, and small-beam decoding by thresholds on entropy/margin, and compare compute-normalized accuracy to fixed decoding across QA/summarization tasks. Verify the predicted monotonic relation between entropy and marginal gain from extra search. Failure to outperform tuned static baselines or to show the monotonic relation falsifies the claim.
Novelty: It proposes a general, model-agnostic law linking token-level uncertainty to compute allocation across decoding strategies.

3) Low-Rank Recurrent State Replacement
Summary: Beyond a short window, a low-dimensional recurrent state can replace distant KV entries without changing next-token predictions.
For a smart layperson: The model remembers long histories, but we propose most of that memory can be compressed into a tiny “summary state.” After a few recent words, keeping this state is enough to keep outputs the same. This would cut memory without hurting results.
Falsification: Learn a small state update module that absorbs KVs older than window W into a k-dimensional vector per layer and regenerates their effect on attention; compare token-level agreement/perplexity with full-cache and sliding-window baselines on open models. If agreement drops significantly at moderate k while sliding-window performs similarly, the theory fails.
Novelty: It shifts from lossy KV truncation to functional replacement by an explicit recurrent state during inference.

4) Semantic Beam Convergence
Summary: Beam search paths rapidly collapse to semantically equivalent continuations, making greedy decoding near-optimal once a semantic frame is set.
For a smart layperson: Different plausible continuations often say the same thing in different words. We claim that, after a few steps, all the “best” beams become variations of one idea, so simple decoding works almost as well. This explains why big beams rarely help.
Falsification: Run varying beam widths; cluster partial beams by sentence-embedding similarity over time; measure when clusters collapse and whether greedy outputs match the dominant cluster’s meaning and task scores. If beams remain semantically diverse and outperform greedy after frame establishment, the theory is false.
Novelty: It formalizes beam utility in semantic space rather than token space and predicts an observable convergence pattern.

5) Error-Detecting Verifier from Speculative Rejections
Summary: The rate and pattern of speculative decoding rejections predicts imminent hallucinations and factual errors.
For a smart layperson: In fast decoding, a small “draft” guess is checked by the full model; rejected guesses are a kind of warning signal. We claim bursts of rejections mean the model is about to go off-track. Watching these lets us intervene before errors appear.
Falsification: Instrument draft-verify decoding to record rolling rejection rates; correlate with hallucination labels on QA/closed-book tasks; trigger conservative decoding when rejection spikes and measure error reduction versus cost. Lack of predictive correlation or ineffective interventions falsifies the claim.
Novelty: It reframes an efficiency signal (verification rejections) as a reliability predictor for real-time error control.

6) Attention-Head Minimality
Summary: For most steps, a small, stable subset of attention heads per layer suffices to reproduce the next-token distribution.
For a smart layperson: Transformers have many “heads” looking at context; we propose only a few are usually needed to get the next word right. Identifying and using just those can keep answers unchanged while saving compute.
Falsification: Using a held-out set, greedily select minimal head subsets that preserve logits; fix these subsets and evaluate agreement/perplexity across tasks and prompts on small LLMs. If required subsets are large or highly input-specific, or accuracy collapses when fixed, the theory is false.
Novelty: It claims stability of minimal head sets across inputs for exact prediction, beyond per-example head importance.

7) Eigen-Aligned KV Quantization
Summary: Quantization error aligned with principal eigenvectors of attention maps has sublinear impact on loss, enabling attention-aware asymmetric KV quantization.
For a smart layperson: Compressing memory introduces noise; we argue that noise in certain “important directions” hurts less than expected. By shaping compression to avoid harmful directions, quality stays high at low bitrates.
Falsification: Estimate per-layer attention covariance; build quantizers that minimize error in the orthogonal complement vs standard uniform/greedy schemes; compare perplexity and token agreement under equal bit budgets. If attention-aware quantizers do not outperform baselines, the claim fails.
Novelty: It links the geometry of attention to where quantization noise is least harmful and tests a concrete design.

8) Residence-Time Law for Context Use
Summary: The probability a past token is ever re-attended after lag t follows a power law with a stable exponent across prompts and models.
For a smart layperson: We track how long-ago words get looked at again; we predict their chances drop in a specific heavy-tailed way. This law would let us prune memory more safely and predict when old context still matters.
Falsification: Log time-to-next-attention for tokens during inference on diverse datasets; fit tail distributions and test for power-law with consistent exponents across small open models; compare against exponential/lognormal fits. Failure to observe a robust power-law falsifies the theory.
Novelty: It proposes a universal, quantitative law of attention reuse to ground principled cache eviction.

9) Entropy-Annealed Temperature Control
Summary: A deterministic schedule that lowers temperature as local entropy falls improves exactness and stability at fixed compute.
For a smart layperson: When the model is unsure, we let it be more random; when it becomes sure, we cool it down. This simple rule aims to keep creativity when needed but avoid drift once the answer is clear.
Falsification: Implement T_t = f(H_t) with simple monotone forms; compare exact-match, ROUGE/BLEU, and diversity to tuned fixed temperatures across tasks; ablate f to test robustness. If no consistent gains appear, the theory is false.
Novelty: It posits a general control law mapping token-level uncertainty to temperature with broad, testable benefits.

10) Minority-Vote Decoding Equivalence
Summary: A small ensemble of low-precision replicas with independent noise, voting per token, better matches high-precision outputs than any single replica at equalized cost.
For a smart layperson: Two or three cheap, slightly noisy copies can “vote” each step to cancel out their individual mistakes, mimicking a more precise model. This can upgrade quality without slowing things down.
Falsification: Create independently quantized replicas of small LLMs; run per-step majority vote at low temperatures; measure agreement with a high-precision reference and task metrics under matched latency/compute. If ensembles don’t beat single replicas, the claim fails.
Novelty: It brings classic noise-canceling ensembles to autoregressive token voting under quantization noise in LLM inference.
