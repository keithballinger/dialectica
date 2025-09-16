1) Entropy-Matched Adaptive Depth Decoding
Summary: Per-token decoding can stop at a variable layer when the model’s predicted entropy matches the teacher-forcing entropy profile, preserving accuracy while reducing compute.
For a smart layperson: Not every word needs the full “thinking” depth of the model. If the model is already confident enough—matching how confident it was during training—you can stop early for that word. This can make responses faster without hurting quality.
Falsification: Measure per-token entropy at intermediate layers and stop when it reaches the training-time entropy curve (estimated offline on a corpus). Compare accuracy and latency against full-depth decoding on standard benchmarks with small open-source LLMs; if accuracy drops significantly or latency doesn’t improve, the theory is false.
Novelty: Unlike prior early-exit methods, this uses a training-entropy matching criterion to govern per-token depth without extra training.

2) Low-Rank KV Cache Subspace Hypothesis
Summary: The keys/values accumulated during decoding lie predominantly in a low-rank subspace per head, enabling strong KV compression with negligible loss.
For a smart layperson: The model stores memories (KV cache) as it writes a sentence; this memory may be mostly redundant. If most of the useful information lives in a small number of directions, we can compress it without changing the output much.
Falsification: Apply per-head SVD or randomized low-rank projection to the KV cache during generation at varying ranks and measure perplexity/accuracy vs. baseline on small models; if low-rank projections cause large performance drops at modest ranks, the hypothesis is false.
Novelty: Prior KV compression focuses on quantization or eviction; this posits and tests an intrinsic low-rank structure per head at inference time.

3) Cross-Prompt KV Projection Warm-Start
Summary: For semantically similar prompts, a linear projection from one prompt’s KV cache can warm-start another’s cache to reduce initial latency without harming accuracy.
For a smart layperson: If you ask two very similar questions, the model’s “mental state” should also be similar. A simple mapping can reuse the earlier state to start the next answer faster.
Falsification: Build a small library of prefixes and a linear projector from prefix embeddings to KV caches (fit on held-out pairs). Warm-start decoding for similar prompts and compare first-token latency and output quality vs. cold start on small models; failure to improve latency or quality indicates falsification.
Novelty: This proposes reusing and projecting entire KV caches across prompts via a learned linear map, rather than caching logits or embeddings alone.

4) Stochastic KV Noise Improves OOD Robustness
Summary: Injecting small, layer-dependent Gaussian noise into the KV cache at decoding time improves robustness and reduces repetition on out-of-distribution inputs.
For a smart layperson: Adding a tiny bit of randomness to the model’s internal memory can prevent it from getting stuck in loops or making brittle mistakes when the input is unusual. It acts like a safety margin.
Falsification: Add calibrated noise to KVs during generation on OOD benchmarks and adversarial prompts, measuring calibration (ECE), repetition rate, and task accuracy vs. no-noise baselines; if robustness does not improve or accuracy worsens broadly, the theory is false.
Novelty: Prior stochastic decoding targets logits or sampling; this targets the KV cache internals with structured noise for robustness.

5) Entropy-Triggered Sparse Beam Search
Summary: Activating beam search only on high-entropy steps preserves accuracy of full-beam decoding at much lower compute cost.
For a smart layperson: Use multiple candidate continuations only when the model is unsure about the next word. When it’s confident, stick to the best guess to save time.
Falsification: Run beam search with a simple entropy threshold to toggle beam width between 1 and B on small models; compare exact match/F1 and token throughput to always-on beam search; if quality degrades beyond statistical noise or compute isn’t reduced, the theory is false.
Novelty: This formalizes beam search as a conditional resource that is activated by uncertainty, rather than a fixed global setting.

6) Logit-Margin PID Control for Calibration
Summary: A feedback controller that adjusts temperature to maintain a target logit-margin distribution yields better calibration and fewer hallucinations than fixed temperature.
For a smart layperson: When the model is too confident or not confident enough, a controller nudges its “boldness” to stay in a healthy range. This can keep it from confidently saying wrong things.
Falsification: Implement a PID controller on-the-fly using the top-1 vs. top-2 logit margin to tune temperature per token; evaluate ECE, factuality (e.g., TruthfulQA), and task metrics vs. fixed temperature on small models; lack of calibration/factuality gains falsifies the claim.
Novelty: Introduces closed-loop control based on logit margins for decoding calibration, rather than static or heuristic temperature schedules.

7) Linearized Leapfrog Decoding
Summary: Using a first-order Taylor approximation of logits with respect to the last-layer state enables skipping every other forward pass with minimal quality loss.
For a smart layperson: Predict two words while computing the heavy math only once by estimating how the next step will change. It’s like taking a careful shortcut every other step.
Falsification: Compute Jacobian-vector products for the last layer to approximate step t+2 after committing token t, alternating exact and linearized steps; compare perplexity and task scores vs. full decoding on small models; significant degradation falsifies the theory.
Novelty: Applies local linearization across decoding steps to halve forward passes without pretraining changes.

8) Entropy-Gated Attention Head Skipping
Summary: Dynamically skipping a subset of attention heads when next-token entropy is low preserves accuracy while reducing compute.
For a smart layperson: Not all of the model’s “attention eyes” are needed for easy words. Turn off some when the model is sure, and turn them back on when it’s unsure.
Falsification: Rank heads by learned or heuristic importance and gate them off for low-entropy steps; measure latency and accuracy vs. full-head decoding on small models; if accuracy drops meaningfully without speed gains, the theory is false.
Novelty: Prior head pruning is static; this proposes per-token, entropy-driven head scheduling at inference time.

9) Temperature-Mismatched Speculative Decoding Improves Acceptance
Summary: Using a hotter draft model (higher temperature) than the target model increases speculative decoding acceptance rates up to a model-dependent optimum.
For a smart layperson: When a small helper proposes options with a bit more variety than the main model, the main model more often finds something it likes and accepts it, speeding things up.
Falsification: Fix a target model and vary the draft model temperature while measuring acceptance rate and quality on small models; observe a non-monotonic improvement vs. matched temperatures; absence of improvement falsifies the theory.
Novelty: Challenges the common assumption of matching draft and target sampling, predicting a specific acceptance behavior under temperature mismatch.

10) Sparse Residual Stream Hypothesis
Summary: Most predictive signal for the next token resides in a small, stable subset of residual stream dimensions that can be zeroed elsewhere with minimal loss.
For a smart layperson: The model’s internal vector is large, but only some components matter for choosing the next word. Ignoring the rest can save computation without hurting results.
Falsification: Identify important residual dimensions via magnitude or saliency and zero the rest during decoding; track perplexity/accuracy vs. sparsity on small models; if performance collapses at moderate sparsity, the hypothesis is false.
Novelty: Proposes dimension-level sparsity in the residual stream at inference time, distinct from weight sparsity or pruning.
