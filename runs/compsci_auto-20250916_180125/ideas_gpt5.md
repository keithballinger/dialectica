1) Margin-Stability Early Exit for LLM Decoding
Summary: If the top-1 logit margin remains stable across successive upper layers during a token step, the final next-token will match, enabling safe early exits without retraining.
For a smart layperson: Often the model already “knows” the next word before finishing all its layers. If the favorite choice stays clearly ahead for a few computation steps, we can stop early and output it.
Falsification: Train a simple linear probe (logit lens) for each layer on a small corpus, measure per-token top-1 margin stability, and early-exit when the margin exceeds a threshold and changes less than epsilon over w layers; compare exact-match rate and speedup against full-depth decoding on standard test sets.
Novelty: Introduces a margin-stability criterion for depth skipping that is training-free and layer-agnostic, rather than using ad hoc entropy thresholds or model retraining.

2) Attention-Weighted Online Low-Rank KV Cache
Summary: Keys/values concentrate in a low-dimensional subspace weighted by realized attention, allowing streaming low-rank KV compression with negligible loss.
For a smart layperson: The model mostly reuses a small set of patterns when “looking back” at the conversation. We can keep a compact summary of what it really pays attention to instead of storing everything in full.
Falsification: Implement per-head attention-weighted randomized SVD updated online with rank r and reconstruct attention using the compressed basis; measure perplexity/accuracy vs memory and speed on open benchmarks for small LMs.
Novelty: Uses streaming, attention-weighted factorization during inference (not offline pruning) to compress KV cache adaptively to the actual query distribution.

3) Feature-Divergence-Gated Speculative Decoding
Summary: The L2 distance between draft and target model hidden features for candidate tokens predicts acceptance better than logit agreement, improving speculative decoding efficiency.
For a smart layperson: When a small “draft” model suggests words for a big model, comparing their internal thoughts can tell us which guesses will stick. This avoids wasting time on guesses the big model will reject.
Falsification: Pair a small open model as draft with a larger open model as target, compute per-candidate hidden-state distances from a shared intermediate layer, accept only candidates below a learned threshold, and compare acceptance rate, speedup, and quality to logit-only gating on standard tasks.
Novelty: Proposes training-free feature-space divergence as the gating signal for acceptance, rather than only using probability ratios or exact-match checks.

4) Hash-Indexed KV Retrieval with False-Positive Control
Summary: Replacing dense KV lookup with per-head locality-sensitive hashing to buckets bounds attention errors by a controllable false-positive rate while accelerating inference.
For a smart layperson: Instead of checking every past word, we file memories into bins by similarity and only look inside the few relevant bins. We can tune how often the system looks in the wrong bin, and see the quality trade-off.
Falsification: Implement per-head SimHash (or LSH) of keys into B buckets, restrict attention to tokens in the query’s bucket(s), and measure perplexity/accuracy and speed as a function of bucket count and multi-probe settings on small models.
Novelty: Adds an explicit, tunable FP-rate abstraction to KV retrieval for causal attention, connecting approximate indexing to controlled quality loss at inference time.

5) Entropy-Guided Temperature Scheduling
Summary: Dynamically setting temperature as a deterministic function of per-step output entropy improves sample quality at fixed expected log-likelihood.
For a smart layperson: The model knows when it’s unsure (high entropy) or confident (low entropy). We can cool or heat the randomness knob on the fly to get better writing without hurting correctness.
Falsification: Implement T_t = f(H_t) with a monotone schedule (e.g., T_t = a/(b+H_t)), evaluate human and automatic metrics (e.g., MAUVE, distinct-n, perplexity) versus fixed-temperature baselines on open datasets with small models.
Novelty: Formalizes a deterministic entropy-to-temperature mapping for decoding, rather than static temperatures or heuristic repetition penalties.

6) State-Convergent Branch-and-Merge Decoding
Summary: Maintaining multiple branches and merging them when hidden states become sufficiently similar yields better quality-compute trade-offs than beam search under equal FLOPs.
For a smart layperson: Try a few continuations in parallel, but if two paths lead the model into almost the same internal state, fuse them and avoid duplicate work. This keeps diversity without wasting compute.
Falsification: Implement beam-like decoding with cosine-based state merging and compute-aware throttling; compare quality (task metrics) vs total FLOPs against beam search and diverse beam on small models.
Novelty: Introduces hidden-state equivalence merging during inference, not just token-prefix deduplication, to exploit representational convergence.

7) Top-k Logit Variance as a Hallucination Early-Warning Signal
Summary: Spikes in variance among the top-k logits predict factual errors better than entropy alone, enabling on-the-fly mitigation triggers.
For a smart layperson: When the model’s top few choices disagree strongly, it’s a sign of uncertainty that correlates with making things up. We can detect these moments and ask for help (e.g., retrieval) before committing.
Falsification: Measure correlation of top-k variance (and its change over positions) with factuality errors on open QA datasets; gate a retrieval/abstain action when variance crosses a tuned threshold and quantify factuality gains at fixed token budgets.
Novelty: Uses the structure of top-k dispersion (variance) as a distinct, stronger predictor than aggregate entropy for hallucination risk during inference.

8) Influence-Estimated Layer Dropping at Inference
Summary: One-step influence estimates of each layer’s contribution to next-token loss enable per-token layer skipping without any fine-tuning.
For a smart layperson: We can estimate how much each layer matters for the next word by a quick sensitivity check. If some layers barely matter right now, we skip them to save time.
Falsification: Compute per-layer saliency via a single Jacobian-vector product with respect to the next-token logit, skip layers below a threshold, and evaluate speed-perplexity trade-offs vs uniform layer dropping on small models.
Novelty: Applies influence-function-style sensitivity at inference for dynamic depth control, avoiding training-time modifications or learned controllers.

9) Saliency-Guided Per-Token Head Sparsification
Summary: Selecting only the most salient attention heads per token using a cheap gradient-norm proxy preserves accuracy while cutting attention FLOPs.
For a smart layperson: Not all attention heads are useful for every word; we can quickly score which ones matter and use only those, saving computation.
Falsification: Estimate per-head saliency as the squared gradient norm of the log-probability w.r.t. head outputs via a lightweight backward pass, mask low-saliency heads, and measure throughput and perplexity on small open models.
Novelty: Introduces on-the-fly, per-token head selection based on saliency rather than static pruning or heuristic head importance.

10) Alternating Rounding Phases Reduce Quantization Bias
Summary: Alternating rounding modes or zero-point phases across layers cancels systematic low-bit quantization bias and better preserves next-token distributions.
For a smart layperson: Rounding errors in compressed models can add up in one direction; by flipping the rounding strategy layer by layer, errors cancel out instead of stacking.
Falsification: Quantize small models to 3–4 bits with per-layer alternating stochastic vs nearest rounding (or phase-shifted zero-points), then compare KL divergence of next-token distributions and task metrics to standard uniform quantization.
Novelty: Proposes cross-layer rounding-phase design to exploit error cancellation, a dimension missing from existing per-layer quantization schemes.
