1) Prompt-Local Predictability of MLP Sparsity
Summary: The set of active MLP neurons per token is accurately predictable from a tiny linear probe on the residual stream, enabling safe skipping of inactive neurons at inference.
For a smart layperson: Inside an LLM, most tiny “switches” (neurons) don’t matter for a given word; only a few turn on. We claim you can cheaply predict which ones will matter next, so you only compute those. That means faster answers without changing what the model says.
Falsification: Train a small linear/logistic probe on residual states (frozen model) to predict top-k active MLP units per layer; run TinyLlama/Pythia with only predicted units computed and compare token-level agreement/KL/perplexity versus full compute. If prediction-driven skipping causes >X drop in agreement at the same sparsity as oracle top-k selection, the theory is falsified.
Novelty: Unlike static pruning or magnitude gating, this posits a causal, per-token predictability of neuron activity from the residual stream that enables zero-cost dynamic sparsity.

2) A Measurable Influence Horizon Governs Attention
Summary: For most tokens, a bounded, data-dependent horizon of prior tokens determines the next-token distribution, which can be estimated online to safely truncate the KV cache.
For a smart layperson: Not all past words matter equally; beyond a point, adding more context barely changes the next word. We claim you can estimate that cutoff on the fly and throw away older memory safely. This reduces memory and speeds generation without changing outputs.
Falsification: Define an online bound from attention energy tails or influence estimates; evict KV entries beyond the predicted horizon and measure token-level KL divergence and exact-match rate against full context on TinyLlama/Mistral-instruct. If measured errors frequently exceed bound predictions, the theory fails.
Novelty: Provides a concrete, per-token estimator with an error bound for context truncation, moving beyond heuristic sliding windows.

3) Logit Margin Is a Model-Agnostic Sufficient Statistic for Speculative Acceptance
Summary: The difference between top-1 and top-2 logits from a draft model predicts teacher acceptance probability with near model-independent calibration, enabling optimal acceptance thresholds without teacher queries.
For a smart layperson: If the draft model is very confident about the next word (big gap between first and second choice), the big model usually agrees. We claim this “gap” works the same way across different model pairs, so you can accept more drafts without asking the big one as often.
Falsification: For multiple student/teacher pairs (e.g., Pythia-160M→Pythia-1.4B, TinyLlama→Llama-2-7B), compute acceptance vs. margin reliability diagrams; if a single monotone calibration curve doesn’t generalize across pairs (e.g., high ECE/KL), the theory is falsified.
Novelty: Asserts cross-architecture invariance of a single scalar statistic to predict speculative acceptance, not just within one pair.

4) Predictive-Entropy Governs Safe Dynamic Precision
Summary: The model’s own predictive entropy determines the minimal arithmetic precision needed per token to preserve outputs, enabling entropy-driven per-token quantization.
For a smart layperson: When the model is sure of the next word, you don’t need many digits of math precision; when it’s unsure, you need more. Use the model’s uncertainty to dial precision up or down on the fly, saving time and energy.
Falsification: Implement per-token dynamic bitwidth (weights/activations/accumulators) as a function of entropy and compare outputs and speed on Pythia-410M/TinyLlama; failure to maintain agreement/KL at predicted bitwidths across entropy bins falsifies the claim.
Novelty: Links a principled, observable statistic (entropy) to per-token numeric precision requirements, beyond static or per-layer quantization.

5) Linear Prompt Sketches Are Sufficient Statistics for Next-Token Prediction
Summary: A small set of linear projections of the prompt’s residual stream (a “sketch”) suffices to reconstruct logits within a fixed error, enabling inference from compressed states.
For a smart layperson: Instead of remembering every detail of the prompt, the model only needs a few combined measurements to predict the next word. Compute and store those few measurements and generate nearly the same output, faster.
Falsification: Learn or fix random projection matrices that compress residual streams to r dimensions; train a lightweight linear reconstructor to logits and evaluate KL/perplexity vs. full model on TinyLlama; if no r ≪ d achieves low error consistently across prompts, the theory is falsified.
Novelty: Posits a low-dimensional linear sufficient statistic for inference-time state, not just weight compression or attention sparsity.

6) Q–K Norm Alignment Predicts Head Utility
Summary: The utility of an attention head for a token is predictable from pre-attention query–key norm/angle statistics, enabling dynamic head skipping without accuracy loss.
For a smart layperson: Each “attention head” is like a helper with a specialty; you can tell if a helper will help by looking at a quick alignment score before doing the work. Skip helpers that won’t help and still get the same answer faster.
Falsification: Train a tiny predictor on per-head q/k norms and cosine to classify “useful vs. skippable” heads; skip predicted heads on Pythia/TinyLlama and measure agreement/KL; if predicted skipping causes disproportionate errors vs. oracle masking, the theory fails.
Novelty: Introduces a physics-like, pre-attention alignment criterion to gate heads, rather than post-hoc importance or fixed pruning.

7) Cross-Prompt KV Reuse via Semantic Anchors
Summary: KV caches of common phrases can be reused across prompts when aligned by a semantic anchor embedding, yielding near-identical outputs for those spans.
For a smart layperson: If two different prompts include the same phrase in a similar meaning, the internal “memory” for that phrase can be recycled. This avoids recomputing repeated text and speeds up generation across tasks.
Falsification: Build a library of KV segments for frequent n-grams with anchor embeddings; on new prompts, align and splice KVs for matched phrases in TinyLlama; compare outputs vs. recomputing; if reuse often changes logits beyond a small KL/accuracy tolerance, the theory is falsified.
Novelty: Proposes cross-sequence KV caching conditioned on semantic alignment, not only intra-sequence reuse or exact-position caching.

8) Residual Energy Conservation Predicts Safe Early Exits
Summary: The L2 “energy” of the residual stream follows a conserved budget per token such that diminishing updates signal layers can be skipped with minimal output change.
For a smart layperson: The model’s internal signal changes a lot early and then settles; when changes become tiny, extra layers don’t affect the next word much. Use that to stop early and still say the same thing.
Falsification: Measure per-layer residual norm deltas; define an exit rule when cumulative delta < ε; compare outputs vs. full depth on Pythia/TinyLlama; if low deltas don’t correlate with small logit differences across datasets, the theory is falsified.
Novelty: Frames early-exit gating on a conservation-law-like residual norm schedule, not heuristic confidence or learned halting.

9) Cross-Layer Low-Rank Structure in Keys and Values
Summary: Keys and values across layers lie in a shared, low-dimensional subspace that is stable across prompts, enabling a single projection for KV compression without retraining.
For a smart layperson: The model reuses the same few “directions” of information across many layers. If you compress along those shared directions, you keep what matters and speed up inference.
Falsification: Compute PCA/autoencoder on concatenated K/V from multiple layers using a small corpus; apply shared projection at inference on TinyLlama/Mistral and measure accuracy/KL; if no low-rank r reproduces outputs across layers, the theory is falsified.
Novelty: Claims a cross-layer, prompt-stable KV subspace enabling unified compression, distinct from per-layer or per-head compression.

10) Stochastic Causal Masking Identifies Redundant Context Tokens
Summary: Monte Carlo causal tracing via random context dropouts provides unbiased influence estimates, allowing safe removal of low-influence tokens at inference.
For a smart layperson: Try hiding random past words and see how much the next-word prediction changes; if hiding a word barely matters, you can ignore it. This shrinks context without changing the result.
Falsification: Run multiple masked-forwards per token to estimate influence scores; prune lowest-scoring tokens and compare outputs vs. full context on Pythia/TinyLlama; if estimated low-influence tokens frequently cause large logit shifts when removed, the theory is falsified.
Novelty: Uses online stochastic causal experiments to guide context pruning, rather than relying on attention weights or fixed windows.
