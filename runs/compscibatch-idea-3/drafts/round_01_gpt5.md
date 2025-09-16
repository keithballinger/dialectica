Major Revisions

Title
Cross-Prompt KV Projection Warm-Start: Reducing First-Token Latency via Linear Cache Transfer

Abstract
Prefill dominates end-to-end latency for large language models (LLMs), and cache reuse only helps when the prefix is exactly identical. We propose Cross-Prompt KV Projection (CP-KVP): a learned linear projector that maps the cached keys/values (KV) of a previously answered, semantically similar prompt to a compact set of virtual KV “summary slots” that warm-start decoding for a new prompt. CP-KVP avoids full prefill for the first token(s), producing immediate logits with minimal accuracy loss and enabling background cache refinement. The approach is architecturally non-invasive, trained with simple ridge regressions on small open-source models, and validated on paraphrase-style prompt families. We detail a falsification protocol: if CP-KVP fails to reduce first-token latency without degrading next-token distribution (KL) or task quality, the hypothesis is rejected.

Introduction
- Problem: Autoregressive decoders incur O(T) prefill cost to build per-layer KV caches before emitting the first generated token. Existing cache reuse works only for exact-prefix hits; near-duplicates still pay the full prefill.
- Hypothesis: For semantically similar prompts, the functional effect of a prefix on immediate next-token logits lies in a low-dimensional subspace that is linearly related across prompts. Therefore, projecting a cached KV state from a similar source prompt can initialize a surrogate cache for a new target prompt, reducing first-token latency without harming accuracy.
- Contribution:
  1) A linear, cross-prompt projector over compact KV “summary slots” that can be dropped into standard HF models.
  2) A retrieval+projection inference recipe that yields first-token logits without prefill, with a background refinement pathway.
  3) A falsification-oriented evaluation that can be replicated on small models.

Method
Setting
- Decoder-only transformer with L layers, H heads, head dimension d. For a prefix of length T, the per-layer cache is Kℓ ∈ R^{H×T×d}, Vℓ ∈ R^{H×T×d}.
- We introduce m virtual summary slots per layer, Sℓ ∈ R^{H×m×2d}, that stand in for the full prefix KV at first token. m is small (e.g., 8–32).

Offline training
We learn two linear maps:

1) Cache-to-summary compression Cℓ
- Goal: Map a full cache for a prefix to m summary slots per layer.
- Data: A corpus of prompts D. For each prompt p, run a standard forward pass to record true Kℓ(p), Vℓ(p) for all layers. Also record the next-token query vectors Qℓ_next(p) (the queries at the first generation position) and the resulting attention outputs Oℓ_next(p) under the true cache.
- Parameterization: For each layer ℓ and head h, learn a linear map Cℓh: R^{T×2d} → R^{m×2d}. We use ridge regression to learn Cℓh that minimizes the discrepancy between the attention output induced by the summary slots and the true Oℓ_next(p), across prompts p ∈ D:
  - Loss: sum over p of || Attn(Qℓh_next(p), Cℓh[Kℓh(p)‖Vℓh(p)]) − Oℓh_next(p) ||² + λ||Cℓh||².
  - Attn denotes a standard causal attention read with the m summary slots as the only keys/values. This keeps training targeted at first-token fidelity and avoids reconstructing the entire cache.

2) Cross-prompt projector Mℓ
- Goal: Map source summaries Sℓ(p_s) to target summaries Sℓ(p_t) for semantically similar prompts.
- Data: A set of similar prompt pairs P = {(p_s, p_t)} collected from paraphrase datasets (e.g., QuoraQP, PAWS) and/or templated prompt families. Compute Sℓ(p) = Cℓ[Kℓ(p)‖Vℓ(p)] once, offline.
- Parameterization: For each layer ℓ, learn Mℓ: R^{H×m×2d} → R^{H×m×2d} as independent ridge regressions per head (vectorizing the m×2d tensor). Minimize || Mℓ Sℓ(p_s) − Sℓ(p_t) ||² + γ||Mℓ||² over (p_s, p_t) ∈ P.

Inference
- Library: Maintain an index of previously processed prompts with their stored summaries {Sℓ(p_i)} and an embedding (e.g., all-MiniLM-L6-v2) for retrieval.
- Given a new prompt p_t:
  1) Retrieve nearest neighbor p_s by cosine similarity; require sim ≥ τ (fallback to cold prefill otherwise).
  2) Load Sℓ(p_s) from disk. Compute projected summaries Ŝℓ(p_t) = Mℓ Sℓ(p_s).
  3) Initialize past_key_values with m virtual tokens per layer/head using Ŝℓ(p_t). For absolute positional models (e.g., GPT-2), assign positions 0..m−1 to the virtual tokens and start decoding at position m. For RoPE models, use standard rotary phases for these positions.
  4) Emit first token(s) directly. In a low-priority thread, run an exact prefill for p_t to build the true cache; when ready, swap caches seamlessly (optionally with a one-step MSE alignment).

Complexity and integration
- Training: Two rounds of ridge regressions; parallelizable per layer/head. No model finetuning.
- Storage: O(L×H×m×d) per cached prompt (tiny vs full T×d).
- Runtime: First-token latency dominated by a retrieval and a few matrix multiplications; no forward pass over T tokens before the first emission.

Experiments (falsification plan)
Models
- GPT-2 small (124M) and medium (355M), Pythia-160M/410M. All HuggingFace.
- Mixed positional encodings: learned absolute (GPT-2), RoPE (small Llama-like variant if feasible).

Datasets and prompt families
- Paraphrase pairs: Quora Question Pairs, PAWS.
- Templated instructions: Few-shot QA and summarization with fixed system templates and variable questions (e.g., SQuAD questions rephrased).
- Held-out prompts for testing, disjoint from training pairs.

Baselines
- Cold start: standard prefill.
- Exact prefix cache hit: upper bound latency.
- Embedding-only warm-start: initialize with m learned, prompt-agnostic summary slots (no cross-prompt projection).
- Hidden-state warm-start: initialize only the last-layer hidden state for the next-token position (no KV cache), to show the need for KV summaries.

Metrics
- Latency: wall-clock to first token; throughput of first 8 tokens.
- Next-token divergence: KL(p_warm || p_cold) at the first decoding position; report mean and 95% CI.
- Task quality: 
  - Perplexity delta on held-out continuations (first 32 tokens).
  - For instruction-style prompts: exact match/F1 on short-form QA; ROUGE-L for summarization.
- Robustness: performance vs. semantic similarity bin; m ∈ {4, 8, 16, 32}; τ threshold ablations.

Decision thresholds for falsification
- Success if:
  - ≥35–60% reduction in first-token latency on average (model- and T-dependent).
  - Next-token KL ≤ 0.05 (or ≤ 5% relative perplexity increase) on ≥80% of test prompts within high-similarity bin.
  - Task metrics within 1% absolute of cold start for high-similarity prompts.
- Failure if latency gains are <10% or quality degradations exceed thresholds above.

Implementation notes (reproducible with small models)
- Hook into HF past_key_values to inject Ŝℓ(p_t) as m-length cache per layer/head.
- For GPT-2: absolute positions are straightforward. For RoPE, assign virtual tokens to the earliest positions; since only relative phases matter, this approximates an early-prefix effect.
- Regularization: λ, γ via 3-fold CV on training pairs; standardize features; use double precision for stability in small m.
- Similarity index: sentence-transformer embeddings with FAISS; τ chosen from ROC on train.

Why a linear map can work
- Locally linear behavior: For small edits or paraphrases, the residual stream and attention readouts vary approximately linearly in the neighborhood of a prompt’s representation.
- Low intrinsic dimensionality: The effect of a prefix on the immediate next-token logits often concentrates in a small number of directions (e.g., style, topic, task intent). Summary slots capture these.
- Linearity is enough for warm-start: We need only a good initial readout for the first step(s); exact cache equality is unnecessary if we background-refine.

Discussion
- CP-KVP generalizes cache reuse from exact-prefix hits to semantic neighborhoods, shifting first-token cost from O(T×L) to O(m×L).
- The summary-slot view ties to prefix-tuning: we construct per-instance, per-layer soft prefixes via linear maps rather than learned via backprop through the model.
- Retrieval gating is crucial: far neighbors must fall back to cold start. Confidence can be estimated from sim score and projection residuals on a small validation set.

Limitations
- Domain generalization: Mℓ trained on paraphrases and templates may not transfer to domain-shifted prompts; maintain multiple domain-specific Mℓ or adapt online.
- Architecture sensitivity: RoPE and multi-query attention variants need careful positional handling; we restrict to standard decoder-only models in initial validation.
- m vs. fidelity: Very small m can underfit; very large m erodes latency gains.
- Only first-step accuracy guaranteed: We rely on background prefill for long generations; purely projected caches for many steps may drift.
- Retrieval coverage: Requires a sufficiently dense library of prior prompts to find good neighbors.

Conclusion
We introduce CP-KVP, a simple, code-friendly method to warm-start LLM decoding by projecting cached KV summaries across semantically similar prompts. By replacing the prefill with a retrieval and a set of linear projections, CP-KVP can substantially reduce first-token latency with minimal impact on accuracy, and is readily falsifiable on small open-source models. If validated, it offers an orthogonal axis of inference acceleration complementary to prompt caching and speculative decoding.
