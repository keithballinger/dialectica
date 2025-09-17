1) Entropy-Budgeted Depth for LLM Inference
Summary: The minimal number of transformer layers needed per token scales as a simple function of the token’s predictive entropy, enabling safe early exit without auxiliary classifiers.
For a smart layperson: When the model is very sure about the next word, you don’t need to run all its layers to get the same answer. This proposes a formula that turns the model’s confidence into how much compute to spend. It promises the same quality with less work by stopping early only when it’s safe.
Falsification: On small open models (e.g., Pythia-1.4B, OPT-1.3B), attach a shared provisional output head to all layers, compute token entropy at an intermediate layer, and exit if below a threshold derived from the formula; measure perplexity and task metrics vs full-depth. Test whether error increases remain within predicted bounds across datasets (WikiText, OpenWebText). If bounds are violated or savings are minimal, the theory fails.
Novelty: Prior early-exit methods use learned classifiers or heuristics; this posits a universal, model-agnostic entropy-to-depth law with explicit error guarantees.

2) Token Influence Sparsity Law in Attention
Summary: For most tokens, next-token prediction depends on only O(√L) past positions (L=context length), identifiable on-the-fly by cumulative attention mass without retraining.
For a smart layperson: Even though the model “looks” at the whole history, only a small, predictable slice really matters for the next word. This suggests you can keep just the few most influential past words and throw away the rest. That would make long-context inference cheaper without hurting accuracy.
Falsification: For small models (TinyLlama, OPT-1.3B), prune per head the KV cache to the smallest set of past positions whose attention mass reaches a target (e.g., 95%), with a cap scaling as c·√L; measure perplexity and QA accuracy vs full cache as L grows. If performance degrades or required set scales faster than √L, the theory is falsified.
Novelty: Introduces a concrete √L scaling law for online KV sparsification discoverable via native attention, not offline training or fixed windows.

3) Curvature-Temperature Coupling for Decoding
Summary: The local curvature of the logit landscape (Hessian diagonal proxy) predicts an optimal sampling temperature that reduces hallucinations without reducing diversity.
For a smart layperson: If the model’s probability surface is steep, tiny changes matter and sampling should be cautious; if it’s flat, sampling can be bolder. This links a measurable property of the model’s state to how “random” you should be when choosing words. It aims to keep creativity while avoiding confident nonsense.
Falsification: Approximate curvature using Hutchinson’s trick or Jacobian-vector products on small models (GPT-2 medium, Pythia-410M), set temperature = f(curvature) from the hypothesis, and compare hallucination rates (TruthfulQA, factual probes) and diversity vs fixed temperatures. If no consistent gains or correlation appear, reject the theory.
Novelty: Proposes a mechanistic, test-time curvature-derived temperature schedule instead of heuristic or tuned temperatures.

4) Spectral Attention Instability Predicts Hallucination
Summary: High spectral norm or condition number of attention matrices at decode time predicts and causally contributes to hallucinations; spectral clipping mitigates them.
For a smart layperson: When the model’s attention is dominated by a few unstable patterns, it’s more likely to “make things up.” By measuring and gently taming this instability, we can make answers more reliable. This uses linear algebra of the attention maps themselves.
Falsification: Instrument small LLMs (Mistral-7B, OPT-1.3B) to log per-token attention spectra; correlate top singular value/condition number with hallucination on fact-check datasets. Then apply spectral norm clipping (re-scaling attention logits) and test for reduced hallucinations at similar perplexity; lack of correlation or effect falsifies the theory.
Novelty: Connects a concrete spectral property of attention to hallucination and proposes a simple, testable causal intervention at inference.

5) Low-Rank Sliding Memory Preserves Logits
Summary: Replacing old KV cache entries with a fixed-size, low-rank sliding memory yields near-identical next-token logits up to a provable tolerance.
For a smart layperson: Instead of remembering every past word verbatim, the model can keep a compact summary that captures what matters. This theory claims a small, smart memory can stand in for a long history without changing the model’s next-word guesses much. It would make long-context usage cheaper.
Falsification: Implement online low-rank compression (e.g., incremental PCA/SVD of keys/values per head) on small models with long inputs; compare KL divergence between original and compressed logits and task metrics (long QA, book summarization) as memory size varies. If divergence doesn’t stay within predicted tolerance or performance drops sharply, reject.
Novelty: Provides a concrete logit-preservation claim for post-hoc low-rank KV compression on pretrained LLMs without retraining.

6) Quantization-Compensated Logit Dithering
Summary: Small, calibrated Gaussian noise added to logits at sampling compensates for systematic bias from weight quantization, restoring baseline perplexity and task scores.
For a smart layperson: Rounding the model’s numbers for speed can skew its choices; a touch of randomness in the right amount can cancel that skew. This is like adding dither to a compressed audio signal to regain fidelity. It’s simple and testable at decode time.
Falsification: Quantize small models to 4–8 bits (GPT-2 medium, Pythia-1B) and measure perplexity and downstream accuracy; then add zero-mean logit noise with variance estimated from quantization error of matmuls. Check if performance recovers to near FP16/FP32; failure to recover falsifies the claim.
Novelty: Frames logit-space dither as a principled, calibration-based compensator for quantization bias at inference rather than a generic noise trick.

7) Consistency-Gated Beam Search
Summary: Conditioning beam expansion on agreement among multiple lightweight linear probes of the hidden state yields better quality–compute tradeoffs than likelihood-only beams.
For a smart layperson: Instead of trusting just one score, you consult a few simple “opinions” about what the model is thinking and keep only beams they agree on. This extra check can prune bad paths early without heavy cost. It could make beam search smarter per unit compute.
Falsification: Train frozen linear probes on hidden states to predict next-token classes/coarse features; during beam search on small models, weight beam scores by probe agreement and compare BLEU/ROUGE/QA accuracy and speed vs standard beams. No improvement or regressions falsify the theory.
Novelty: Introduces probe-based internal-consistency gating for decoding without fine-tuning the base model.

8) Lipschitz-Bounded Speculative Decoding Without a Drafter
Summary: Estimating a token-level Lipschitz bound on logit change allows safe pre-emission of k tokens without a separate drafter model, yielding speedups with bounded rollback.
For a smart layperson: If we can bound how much the model’s next-word guesses can change as we add words, we can jump ahead by several words and only roll back when the bound is violated. This speeds up typing without needing a second model. It’s a safety net for guessing in chunks.
Falsification: Estimate per-layer/operator Lipschitz constants (power iteration on weights) to derive a conservative bound; implement speculative decoding with rollback on small models and measure speedup and correction rates vs standard speculation (with drafter) and greedy. If errors exceed bounds or speedups are negligible, reject.
Novelty: Provides a theory-driven, single-model speculative decoding mechanism using Lipschitz analysis instead of a learned drafter.

9) Surprise-Triggered Retrieval is Sufficient
Summary: Retrieval augmentation triggered solely by spikes in token-level surprise (negative log-probability) improves factuality while adding negligible average latency.
For a smart layperson: The model asks for outside help only when it’s surprised by what it’s writing, which is when mistakes are most likely. This keeps the system fast most of the time and careful when needed. It’s a simple on/off rule based on the model’s own uncertainty.
Falsification: Implement a retrieval hook (e.g., BM25 over Wikipedia) that activates when surprise exceeds a threshold; test on open-domain QA with small LLMs and measure accuracy, latency, and retrieval frequency vs always-on/off retrieval. If gains don’t materialize or latency increases too much, falsify.
Novelty: Proposes a minimal, purely intrinsic trigger for RAG at inference with no retraining or external uncertainty models.

10) Variance-Suppressed Neuron Masking Stabilizes Decoding
Summary: Masking a small set of high-variance neurons at inference reduces hallucinations and variance across samples with minimal impact on perplexity.
For a smart layperson: Some internal “wires” in the model are erratic and cause flakiness; temporarily quieting them can make answers steadier. You identify the risky wires by how wildly they swing during generation. This aims for calmer, more reliable outputs.
Falsification: Estimate per-neuron activation variance via Monte Carlo dropout or input perturbations on small models; mask or shrink top-p% neurons during decoding and evaluate hallucination metrics and perplexity vs baseline. If stability doesn’t improve or perplexity worsens significantly, reject.
Novelty: Introduces a targeted, runtime-only neuron selection rule grounded in activation variance to control output stability and hallucination.
