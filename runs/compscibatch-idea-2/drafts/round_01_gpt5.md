Title: Low-Rank KV Cache Subspaces in Decoding: A Testable Hypothesis and an Online Compression Algorithm

Abstract:
During autoregressive decoding, transformers accumulate a key–value (KV) cache per attention head that grows linearly with sequence length and dominates memory and bandwidth at long context. We hypothesize that, for each head, the sequence of cached keys and values lies predominantly in a low-rank subspace, enabling strong per-head KV compression with negligible loss. We formalize this Low-Rank KV Cache Subspace Hypothesis, derive an inference-time factorization that reduces both memory and compute, and provide a streaming algorithm based on frequent-directions/sketching that maintains per-head low-rank bases online without revisiting tokens. The proposal is falsifiable with small open-source models: simply project K and V onto rank-r head-specific subspaces during generation and measure perplexity or accuracy versus baseline across ranks. We outline experimental protocols, complexity analysis, and practical implementation details compatible with standard causal attention, RoPE, and multi-head layouts. If validated, the method delivers controllable KV cache compression and attention speedups with small code changes; if refuted, the experiments bound how intrinsic this structure is across heads, layers, and domains.

Introduction:
The KV cache is the dominant state for large-language-model (LLM) decoding: for each layer and head, it stores one key vector and one value vector per past token. Existing approaches reduce KV cost via quantization or eviction (sliding windows), trading fidelity or long-range recall for memory. A complementary, untested claim is that the set of cached vectors per head lies close to a low-dimensional linear subspace. This would reflect redundancy induced by training, token statistics, and architectural constraints (e.g., rotary position embeddings and shared projections). If true, we can compress per-head KV caches by projecting onto a learned or online-estimated subspace of rank r much smaller than the head dimension, with minimal effect on outputs.

Related lines of work suggest plausibility but do not test this at inference: Linformer and Nyström approximations impose low rank on attention maps via learned projections during training; LoRA constrains weight updates to be low-rank; KV quantization compresses coordinates but not intrinsic dimensionality; eviction and streaming attention prune time steps but not per-head subspace. We instead target the row-space of K and V during decoding, per head, without retraining.

We make three contributions:
- Hypothesis: For each attention head, the matrix of cached keys (and values) across time lies predominantly in a low-rank subspace, enabling projection with small output error.
- Method: A drop-in inference-time factorization that stores per-token r-dimensional coefficients instead of full d-dimensional K/V, plus a small per-head basis. It reduces KV memory O(t d) to O(t r) and replaces O(t d) attention scoring with O(t r) given precomputed bases.
- Validation plan: A falsification-oriented experimental suite on small open-source models that projects K and V at varying per-head ranks and reports perplexity/accuracy and compute-memory trade-offs, including ablations for global vs. online-adaptive bases and interactions with RoPE.

Method:
Notation and attention with low-rank KV
- For a given head h with key/value dimensions d_k and d_v, at time t we have:
  - Query q_t ∈ R^{d_k}
  - Past keys K ∈ R^{T×d_k} and values V ∈ R^{T×d_v}, where T = t − 1
  - Causal attention logits are ℓ = q_t K^T / sqrt(d_k); weights α = softmax(ℓ); output o_t = α V

Low-rank factorization across time (row-space compression)
- Suppose K ≈ C B, where B ∈ R^{r×d_k} is an orthonormal basis (rows span subspace; r ≤ d_k) and C ∈ R^{T×r} are per-token coefficients. Similarly, V ≈ D E with E ∈ R^{r_v×d_v}, D ∈ R^{T×r_v}.
- Then logits and outputs compute as:
  - ℓ ≈ (q_t B^T) C^T / sqrt(d_k) = q̃ C^T / sqrt(d_k), where q̃ ∈ R^{r} is the projected query
  - o_t ≈ α D E, where α = softmax(ℓ), so we first compute α D ∈ R^{r_v}, then multiply by E ∈ R^{r_v×d_v}
- Storage and compute:
  - Replace storing K (T×d_k) with C (T×r) plus B (r×d_k); replace V (T×d_v) with D (T×r_v) plus E (r_v×d_v)
  - Attention scoring cost per step drops from O(T d_k) to O(T r + r d_k) for keys; output aggregation drops from O(T d_v) to O(T r_v + r_v d_v) for values
  - If r and r_v are much smaller than d_k and d_v, we achieve both memory and compute reductions for long contexts

Two practical variants
1) Global per-head PCA bases (SubSpace-G)
  - Offline: Run a single forward pass over a calibration corpus to collect K and V for each head and layer; compute top-r principal components (e.g., via randomized SVD) for K and top-r_v for V, producing bases B and E fixed across sequences.
  - Inference: For each new token, compute c_t = k_t B^T and d_t = v_t E^T; store c_t and d_t, discard k_t and v_t. Compute attention with q̃ = q_t B^T and output via α, D, and E.
  - Pros: Deterministic and simple; zero per-token basis update cost; reproducible and easy to validate.
  - Cons: May underfit idiosyncratic sequences or domains; rank must cover worst-case across domains.

2) Online streaming bases via sketching (SubSpace-A)
  - Maintain a per-head frequent-directions sketch for keys and values. FD keeps a 2r×d matrix whose SVD yields an approximate top-r subspace with guarantees on Frobenius error relative to the best rank-r.
  - At each step, update the sketch with new k_t and v_t; when its SVD is refreshed (e.g., amortized every m steps), rotate the basis and apply the small r×r rotation to all stored coefficients C and D in-place. This keeps past coefficients consistent without revisiting original K/V.
  - Pros: Adapts to sequence; tighter rank for the same loss; does not store full K/V.
  - Cons: Slight extra compute for sketch updates and occasional r×r coefficient rotations.

Compatibility
- Causal masking is unchanged.
- Rotary embeddings: Apply projection after RoPE (i.e., compress the rotated keys). Global bases should be computed on RoPE-applied keys/values to match inference. Empirically, RoPE distributes energy across specific paired dimensions; modest increases in r accommodate this.
- Multi-query/grouped-query attention: Keys are per-head; values may be shared; we apply the corresponding per-head or shared bases.

Approximation error and stability
- Logit error: The attention logit vector error is bounded by ||q_t||·||K − C B|| in operator norm divided by sqrt(d_k). If the spectral tail of K decays rapidly, choosing r at the knee yields small logit perturbations.
- Softmax stability: Softmax is Lipschitz on bounded domains; under small logit perturbations relative to temperature and max-margin between top logits, output drift is small. Value compression error compounds linearly through α D E; choosing r_v by energy capture (e.g., 95–99%) bounds output error in Frobenius norm.
- Mixed-precision: Projections and coefficients can be stored in 16-bit or 8-bit with minor accuracy impact; combine with quantization for multiplicative compression.

Implementation details
- Minimal integration into a standard PyTorch/HF decoding loop:
  1) Precompute per-head B_K and E_V (SubSpace-G) or initialize FD sketches (SubSpace-A).
  2) At each token:
     - Compute q_t, k_t, v_t as usual.
     - Compute q̃ = q_t B_K^T; c_t = k_t B_K^T; d_t = v_t E_V^T.
     - Append c_t to C and d_t to D; optionally update FD sketches and rotate C, D if basis changes.
     - Compute logits ℓ = q̃ C^T / sqrt(d_k), then α = softmax(ℓ + mask).
     - Compute õ = α D (size r_v); output o_t = õ E_V.
  3) Proceed with the rest of the transformer block.
- Complexity: Additional per-step cost is r·d_k to project q_t and k_t, and r_v·d_v to project v_t and reconstruct o_t, typically negligible once T is large. Memory reduces asymptotically by a factor ≈ d_k/r for keys and d_v/r_v for values.
- Code: A reference implementation can be built atop small HF models (e.g., GPT2-124M, Pythia-410M, TinyLlama-1.1B) with tensorized per-head projections and batched coefficient matmuls.

Experiments (falsification plan):
Hypothesis test
- Null hypothesis H0: The per-head low-rank assumption is false; modest ranks (e.g., r ≤ d_k/4) cause significant degradation in perplexity/accuracy.
- Alternative H1: There exist ranks r ≪ d_k and r_v ≪ d_v (head- and layer-dependent) such that perplexity increase is ≤ 1% and task accuracy change is within statistical noise.

Models and data
- Models: GPT-2 Small (124M), GPT-Neo 125M, Pythia-410M, TinyLlama-1.1B, and optionally Mistral-7B-instruct for scale trends.
- Datasets:
  - Perplexity: WikiText-103, C4 validation shards, The Pile validation shards.
  - Zero-shot tasks: LAMBADA (acc), PIQA, ARC-easy/ARC-challenge, HellaSwag (n-way acc).
  - Long-context stress: PG19, BookCorpus2, long-code repos.

Protocols
- SubSpace-G: Compute B_K and E_V per head via randomized SVD on a 1–5M token calibration split (identical tokenizer), for target energy capture thresholds {90%, 95%, 98%, 99%}. Translate to ranks r_K(h,ℓ), r_V(h,ℓ).
- SubSpace-A: Run FD with sketch size 2r per head online; refresh basis every m steps (e.g., m = 64) and rotate coefficients. Choose r matching SubSpace-G’s energy thresholds for comparability.
- Per-head vs. shared ranks: Evaluate uniform r across heads vs. adaptive per-head r from energy spectra.
- Values-only or keys-only compression ablations to isolate contributions.
- RoPE ablations: Compare bases computed pre- vs. post-RoPE; report best.
- Quantization combination: Store C and D in 8-bit per-channel scales to measure multiplicative gains.

Metrics and analysis
- Perplexity delta vs. baseline across r; targeted threshold for “negligible loss” is ≤ 1% relative increase.
- Task accuracy deltas with bootstrap CIs.
- Compute and memory:
  - Peak KV memory reduction at given sequence lengths (e.g., 8k, 32k).
  - Wall-clock tokens/sec and attention FLOPs reduction vs. baseline for long sequences; break-even lengths.
- Error localization: Per-layer/head sensitivity curves; visualize spectral decay of K and V per head.
- Failure patterns: Tasks or heads requiring high ranks; correlation with syntactic vs. semantic heads (e.g., positional/induction heads vs. lexical heads).

Clear falsifiers
- If for small models and standard corpora no setting with r ≤ d_k/4 and r_v ≤ d_v/4 achieves ≤ 1% PPL increase, the hypothesis is refuted.
- If adaptive online methods do not significantly outperform global bases at the same rank, the purported per-sequence adaptivity is weak or unnecessary.
- If RoPE consistently destroys low-rank structure (requiring r near d_k), the hypothesis fails in practical setups.

Discussion:
Why might KV caches be low-rank?
- Token statistics are heavy-tailed and clustered; head projections are low-dimensional filters; RoPE imposes structured rotations; and attention tends to reuse motifs (copying, induction). Empirically, activation manifolds in transformers often have rapidly decaying spectra, suggesting redundancy exploitable at runtime.

Relationship to prior work
- Differs from Linformer/Nyström: no retraining; projection bases can be global or adaptive at inference; exact causal attention shape preserved (full-length logits produced).
- Complements quantization and eviction: Orthogonal axis of compression; combines multiplicatively with 8-bit/4-bit storage and windowing for ultra-long contexts.
- Bridges training-time low-rank ideas (LoRA) with inference-time state compression by targeting the row-space of realized KV tensors.

Practical implications
- Memory: For d_k = d_v = 64 and r = r_v = 8, per-token per-head KV drops 8×. With 32 layers, 32 heads, and 32k context, savings are gigabytes in FP16.
- Compute: For long contexts, attention scoring scales with r instead of d_k, leading to speedups proportional to d_k/r beyond a modest break-even length. This is particularly valuable for CPU-bound or bandwidth-bound deployments.

Limitations:
- Worst-case heads: Some heads (e.g., those encoding fine-grained positional detail) may require higher ranks, reducing gains. Per-head adaptivity mitigates but does not eliminate this.
- Short sequences: Overhead to project queries/values can outweigh gains at small T; use a length threshold before enabling compression.
- Basis drift: Global bases may underperform out-of-domain; online bases introduce small overhead and algorithmic complexity (sketch updates, coefficient rotations).
- Numerical precision: Aggressive low-bit storage for coefficients can interact with softmax sensitivity; per-channel scaling and calibration help.
- RoPE interaction: Rotational structure can increase effective rank; basis computed post-RoPE and modestly larger r address this, but very long contexts may stress the approximation.

Conclusion:
We propose and operationalize the Low-Rank KV Cache Subspace Hypothesis: per-head KV caches during decoding largely inhabit low-dimensional subspaces. A simple factorization that stores per-token coefficients and small per-head bases yields substantial memory and compute savings while preserving outputs. The hypothesis is directly falsifiable with small open-source models by sweeping rank and measuring perplexity and accuracy. If validated, the method provides a principled and practical new axis for KV compression that composes with existing quantization and eviction techniques, enabling faster and longer-context LLM inference with minimal code changes.
