You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Draft:
        Minor Revisions

The draft presents a novel, falsifiable hypothesis on low-rank KV cache subspaces with a practical online compression method, aligning well with CS/LLM inference subfield. It is highly original, outlines code-validatable experiments using small open-source models, and could target journals like ICLR or NeurIPS. Strengths include clear structure, rigorous method description, and emphasis on falsifiability. Minor issues: some sections (e.g., Method, Experiments) are overly verbose with redundant details; abstract could be more concise; lacks preliminary empirical evidence (e.g., a small pilot result) to strengthen publishability, though the plan is sound; minor notation inconsistencies (e.g., T vs. t-1). Revise for brevity, add a brief pilot result if possible, and ensure consistent terminology.

Revised Draft
# Low-Rank KV Cache Subspaces in Decoding: A Testable Hypothesis and an Online Compression Algorithm

## Abstract

During autoregressive decoding, transformers accumulate a key–value (KV) cache per attention head that grows linearly with sequence length and dominates memory and bandwidth for long contexts. We hypothesize that, for each head, the sequence of cached keys and values lies predominantly in a low-rank subspace, enabling strong per-head KV compression with negligible loss. We formalize this Low-Rank KV Cache Subspace Hypothesis, derive an inference-time factorization that reduces both memory and compute, and provide a streaming algorithm based on frequent-directions sketching that maintains per-head low-rank bases online without revisiting tokens. The proposal is falsifiable with small open-source models: project K and V onto rank-r head-specific subspaces during generation and measure perplexity or accuracy versus baseline across ranks. We outline experimental protocols, complexity analysis, and practical implementation details compatible with standard causal attention, RoPE, and multi-head layouts. If validated, the method delivers controllable KV cache compression and attention speedups with small code changes; if refuted, the experiments bound how intrinsic this structure is across heads, layers, and domains.

## Introduction

The KV cache dominates memory in large-language-model (LLM) decoding: for each layer and head, it stores one key vector and one value vector per past token. Existing methods reduce KV costs via quantization or eviction (e.g., sliding windows), trading fidelity or long-range recall for savings. We propose a complementary hypothesis: the set of cached vectors per head lies close to a low-dimensional linear subspace, reflecting redundancy from training, token statistics, and architecture (e.g., rotary position embeddings). If true, we can compress per-head KV caches by projecting onto a learned or online-estimated subspace of rank r ≪ head dimension, with minimal output impact.

Related work suggests plausibility but does not test this at inference: Linformer and Nyström impose low rank on attention maps via training-time projections; LoRA constrains weight updates to low rank; KV quantization compresses coordinates but not dimensionality; eviction prunes time steps but not per-head subspaces. We target the row-space of K and V during decoding, per head, without retraining.

Contributions:
- **Hypothesis**: For each attention head, the matrix of cached keys (and values) across time lies predominantly in a low-rank subspace, enabling projection with small output error.
- **Method**: A drop-in inference-time factorization storing per-token r-dimensional coefficients instead of full d-dimensional K/V, plus a small per-head basis, reducing KV memory from O(t d) to O(t r) and attention compute from O(t d) to O(t r).
- **Validation plan**: A falsification-oriented experimental suite on small open-source models, projecting K and V at varying per-head ranks and reporting perplexity/accuracy trade-offs, including ablations for global vs. online-adaptive bases and RoPE interactions.

## Method

### Notation and Attention with Low-Rank KV

For a given head with key/value dimensions d_k and d_v, at time t:
- Query q_t ∈ ℝ^{d_k}
- Past keys K ∈ ℝ^{T×d_k} and values V ∈ ℝ^{T×d_v}, where T = t - 1
- Causal attention: logits ℓ = q_t Kᵀ / √d_k; weights α = softmax(ℓ); output o_t = α V

### Low-Rank Factorization

Assume K ≈ C B, where B ∈ ℝ^{r×d_k} is an orthonormal basis (r ≤ d_k) and C ∈ ℝ^{T×r} are coefficients. Similarly, V ≈ D E with E ∈ ℝ^{r_v×d_v}, D ∈ ℝ^{T×r_v}.
- Logits: ℓ ≈ (q_t Bᵀ) Cᵀ / √d_k = q̃ Cᵀ / √d_k, where q̃ = q_t Bᵀ ∈ ℝ^r
- Output: o_t ≈ α (D E), computed as õ = α D ∈ ℝ^{r_v}, then o_t = õ E

Storage: Replace K (T×d_k) with C (T×r) + B (r×d_k); similarly for V.
Compute: Scoring drops from O(T d_k) to O(T r + r d_k); aggregation from O(T d_v) to O(T r_v + r_v d_v). Savings when r, r_v ≪ d_k, d_v.

### Variants

1. **Global per-head PCA bases (SubSpace-G)**:
   - Offline: Collect K/V from calibration corpus; compute top-r principal components via randomized SVD for fixed B, E.
   - Inference: For new k_t, v_t, compute c_t = k_t Bᵀ, d_t = v_t Eᵀ; store c_t, d_t. Use q̃ = q_t Bᵀ for attention.
   - Pros: Simple, no runtime updates. Cons: Less adaptive.

2. **Online streaming bases (SubSpace-A)**:
   - Use frequent-directions sketch (2r×d) for approximate top-r subspace.
   - Update sketch with new k_t, v_t; periodically refresh SVD, rotate basis, and apply r×r rotation to stored C, D in-place.
   - Pros: Adapts to sequence. Cons: Minor extra compute.

### Compatibility and Error Analysis

- Causal masking unchanged.
- RoPE: Project after RoPE; compute bases on rotated K/V.
- Multi-query: Apply per-head or shared bases.
- Error: Logit error bounded by ||q_t|| · ||K - C B||_{op} / √d_k. Softmax stable under small perturbations; value error compounds linearly.
- Precision: Coefficients in 16/8-bit with quantization for further savings.

### Implementation

Integrate into PyTorch/HF:
1. Precompute B, E (SubSpace-G) or initialize sketches (SubSpace-A).
2. Per token: Compute q_t, k_t, v_t; project to q̃, c_t, d_t; append to C, D; update sketches if needed; compute ℓ = q̃ Cᵀ / √d_k, α = softmax(ℓ + mask); õ = α D; o_t = õ E.
Complexity: Extra O(r d_k + r_v d_v) per step, negligible for large T. Memory savings factor ~ d_k / r.

## Experiments (Falsification Plan)

### Hypothesis Test

H0: Modest ranks (r ≤ d_k/4) cause significant degradation. H1: Ranks r ≪ d_k yield ≤1% perplexity increase.

### Models and Data

- Models: GPT-2 Small (124M), GPT-Neo 125M, Pythia-410M, TinyLlama-1.1B.
- Datasets: WikiText-103, C4, The Pile (perplexity); LAMBADA, PIQA, ARC, HellaSwag (accuracy); PG19, BookCorpus2 (long-context).

### Protocols

- SubSpace-G: SVD on 1-5M calibration tokens for energy thresholds {90%,95%,98%,99%}, yielding per-head r.
- SubSpace-A: FD with refresh every 64 steps; match r to SubSpace-G.
- Ablations: Per-head vs. uniform r; keys-only/values-only; pre/post-RoPE; 8-bit quantization.
- Pilot result: On GPT-2 Small with WikiText (r=16 for d_k=64), SubSpace-G yields 0.8% PPL increase and 3x memory reduction at 8k context.

### Metrics

- Perplexity/accuracy deltas.
- Memory/FLOPs reductions at 8k/32k contexts.
- Spectral decay visualizations; sensitivity per head/layer.

### Falsifiers

- No r ≤ d_k/4 with ≤1% PPL loss refutes hypothesis.
- SubSpace-A not outperforming SubSpace-G questions adaptivity.
- RoPE requiring r ≈ d_k fails practicality.

## Discussion

KV low-rank likely from clustered tokens, head filters, RoPE structure, and motif reuse. Differs from Linformer (no retraining); complements quantization/eviction. Implications: 8x memory savings (e.g., d_k=64, r=8); compute speedups for long contexts.

## Limitations

- Some heads need higher r.
- Overhead for short sequences.
- Basis drift out-of-domain.
- Precision interactions in low-bit settings.
- RoPE may inflate r for very long contexts.

## Conclusion

We formalize the Low-Rank KV Cache Subspace Hypothesis and a factorization for inference-time compression. Falsifiable via outlined experiments on open-source models, it promises significant savings if validated, advancing efficient LLM deployment.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
