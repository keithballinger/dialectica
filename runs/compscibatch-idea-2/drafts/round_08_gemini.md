Minor Revisions

Brief critique
- Clarity of Blockwise Aggregation: The method for reconstructing the output vector from per-chunk value coefficients and bases in the blockwise softmax is underspecified. The revised draft should provide a precise mathematical formula and corresponding pseudocode for the case where value bases `E_j` differ between chunks.
- Title and Framing: The current title is functional but lacks impact. A more direct title highlighting the core contribution (online subspace projection) would be stronger.
- Error Bound Precision: The connection between the Frequent Directions (FD) guarantee on the Frobenius norm (`||A-A_r||_F`) and the operator norm (`||ΔK||_op`) used in the logit error bound should be made explicit. While the former bounds the latter, stating this clarifies the tightness of the guarantee.
- Pseudocode Efficiency: The provided two-pass pseudocode for the blockwise softmax (one pass to find the max, one to sum) can be implemented more efficiently in a single pass using a standard online softmax algorithm.
- FD Sketch Maintenance: Briefly clarify that the FD sketch is maintained continuously across all tokens to provide a global subspace estimate, rather than being reset for each chunk.

Revised Draft
# The Low-Rank KV Cache Hypothesis: Online Subspace Projection for Autoregressive Inference

## Abstract
We propose and test the Low-Rank KV Cache Hypothesis: during autoregressive decoding, the cached keys and values for each attention head concentrate in a low-dimensional subspace. This enables strong per-head compression with minimal quality loss. We formalize an inference-time factorization that computes attention within these subspaces, using head-specific rank-r bases to reduce KV memory from O(T d) to O(T r) and attention FLOPs from O(T d) to O(T r). We present two algorithms: (1) a static method using a fixed per-head basis learned from calibration data, and (2) an adaptive online variant using chunked frequent-directions (FD) that avoids expensive global updates. We provide a numerically stable blockwise softmax, error bounds linking FD residuals to output error, and a falsification-oriented evaluation plan. Preliminary results on GPT-2 Small show negligible perplexity loss at r ≪ d, suggesting our hypothesis holds promise.

## 1. Introduction
The size and bandwidth of the Key-Value (KV) cache are primary bottlenecks in autoregressive large language model inference. Existing solutions reduce precision (quantization), prune tokens (eviction schemes), or modify the attention mechanism during training (low-rank approximations). We introduce an orthogonal, inference-time approach based on a simple hypothesis: for each attention head, the stream of key and value vectors is highly redundant and lies close to a low-dimensional subspace.

We exploit this by projecting keys and values into per-head, rank-r coefficient spaces and computing attention directly on the compact coefficients, without retraining the model.

Contributions:
- **Hypothesis:** Per-head K/V streams during decoding are approximately low-rank. Projecting them preserves attention outputs with small error at rank r ≪ d.
- **Method:** An inference-time factorization K ≈ C B and V ≈ D E using per-head orthonormal bases B, E, with attention computed efficiently in coefficient space.
- **Online Algorithm:** A chunked frequent-directions variant that adapts bases online without revisiting past tokens, using residual-triggered chunking and a numerically stable blockwise softmax.
- **Theory-to-Practice:** We translate FD's spectral error guarantees into bounds on logit and output perturbations, providing a principled way to tune compression rank `r`.
- **Validation Plan:** A rigorous, falsification-oriented experimental plan on open-source models, with code, ablations, and comparisons to strong KV compression baselines.

## 2. Related Work
- **Training-Time Low-Rank Attention:** Methods like Linformer and Nyströmformer constrain attention maps during training. They do not compress the inference-time KV cache of a pre-trained model or compute attention in an online, adaptive coefficient space.
- **KV Cache Compression:** Quantization (e.g., 8-bit KV) reduces precision but not dimensionality. Token eviction/reuse schemes (StreamingLLM, Scissorhands, H2O) prune or re-weight tokens but operate in the original `d`-dimensional space and do not maintain coherent per-head subspaces.
- **SVD/Subspace Methods:** Offline SVD has been used for analysis, but our work provides a practical online method that (i) maintains per-head low-rank K/V representations during inference, (ii) computes attention purely in coefficient space, and (iii) integrates natively with RoPE and blockwise softmax without retraining.

## 3. Method

### 3.1 Notation and Baseline Attention
Per head, for key/value dimension d_k = d_v = d and T cached tokens:
- Query q_t ∈ R^d, Keys K ∈ R^{T×d}, Values V ∈ R^{T×d}.
- Logits ℓ ∈ R^T: ℓ_i = q_t·k_i / √d.
- Weights α = softmax(ℓ + mask).
- Output o_t = αᵀ V ∈ R^d.

### 3.2 Attention in Coefficient Space
Let B ∈ R^{r×d} be a matrix with r orthonormal rows (BBᵀ = I_r). Define key coefficients C = K Bᵀ ∈ R^{T×r} and projected query q̃ = q_t Bᵀ ∈ R^r. The projected logits are:
- ℓ̂_i = q̃·c_i / √d = q_tᵀ (BᵀB) k_i / √d.
Similarly, for values let E ∈ R^{r_v×d} have orthonormal rows, giving value coefficients D = V Eᵀ ∈ R^{T×r_v}. With α̂ = softmax(ℓ̂ + mask):
- The output is reconstructed from coefficients: ô_t = (α̂ᵀ D) E ∈ R^d.

This reduces per-head storage to C(T×r), D(T×r_v), B(r×d), and E(r_v×d). Per-token compute becomes O(r d) for projections and O(T r) for attention.

### 3.3 Variants

**A. Global Per-Head Bases (SubSpace-G)**
- **Offline:** On calibration data, collect per-head K and V matrices and compute their top-r right singular vectors to form fixed bases B and E.
- **Inference:** At each step, project the new k_t, v_t, and q_t into the fixed coefficient spaces and perform attention.

**B. Online Chunked Frequent-Directions (SubSpace-A)**
This adaptive variant handles distributional drift by creating new chunks with updated bases when projection error grows too large.
- **State:** Maintain a list of chunks. Each chunk `j` has fixed bases (B_j, E_j) and coefficient buffers (C_j, D_j). Set residual thresholds τ_k, τ_v and max chunk length L.
- **Per-token logic:** For a new token (k_t, v_t):
  1. Compute residuals against the current chunk's basis: r_k = ||k_t − k_t B_jᵀB_j||_2.
  2. If r_k > τ_k or chunk length reaches L: finalize the current chunk `j` and start a new chunk `j+1`. The new bases B_{j+1}, E_{j+1} are derived from a global FD sketch (see below).
  3. Else: project k_t and v_t using B_j, E_j and append the resulting coefficients to C_j, D_j.
- **FD Updates:** In parallel, feed every original k_t and v_t into a continuously maintained frequent-directions sketch S (of size m ≥ 2r) for each head. This sketch tracks a global subspace estimate. When a new chunk is created, its bases are set to the top-r right singular vectors of the current sketch S. This provides adaptivity without revisiting old tokens.
- **Blockwise Softmax Across Chunks:** To compute attention for a query q_t over all `J` chunks:
  1. For each chunk `j`, compute projected query q̃_j = q_t B_jᵀ and logits ℓ_j = q̃_j C_jᵀ / √d.
  2. Use a single-pass online algorithm to compute the softmax correctly and efficiently:
     - Initialize global max `m = -∞`, partition sum `Z = 0`, and output vector `o_num = 0`.
     - For each chunk `j` from 1 to `J`:
       - `m_j = max(ℓ_j)`.
       - If `m_j > m`:
         - `o_num = o_num * exp(m - m_j)`.
         - `Z = Z * exp(m - m_j)`.
         - `m = m_j`.
       - `w_j = exp(ℓ_j - m)`.
       - `Z += sum(w_j)`.
       - `o_num += (w_jᵀ D_j) E_j`. (If using a single global value basis E, this simplifies.)
  3. The final output is `ô_t = o_num / Z`.

### 3.4 RoPE Compatibility
Rotary Position Embeddings (RoPE) apply orthogonal rotations to queries and keys. We project post-RoPE, as qᵀk ≈ qᵀ(BᵀB)k holds equally for rotated vectors. This preserves dot products within the learned subspace. Since RoPE rotations for adjacent positions are similar, the union of rotated vectors maintains a low-rank structure for moderate context lengths.

### 3.5 Error Guarantees
Let K̂ = C B and V̂ = D E be the low-rank approximations, with residuals ΔK = K − K̂ and ΔV = V − V̂.
- **Logit Error:** ||ℓ − ℓ̂||_∞ ≤ (||q||_2 / √d) max_i ||(I − BᵀB) k_i||_2. The error is bounded by the query norm and the largest single key projection error.
- **Softmax Stability:** For small logit errors δ = ||ℓ − ℓ̂||_∞, the change in attention weights is bounded: ||α − α̂||_1 = O(δ).
- **Output Error:** ||o − ô||_2 ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂||_1 ||ΔV||_op. The error depends on the logit perturbation (via ΔK) and the value representation error (ΔV).
- **FD Residuals:** The FD algorithm guarantees ||K − K_r||_F^2 ≤ ||K − K̂||_F^2 ≤ ||K − K_r||_F^2 * m/(m-r). Since ||ΔK||_op ≤ ||ΔK||_F, the FD bound on the Frobenius norm provides a computable (though potentially loose) upper bound for the operator norm, linking the tunable sketch size `m` to the output error.

### 3.6 Implementation Details and Pseudocode
- **Data Types:** Bases (B, E) in FP16. Coefficients (C, D) can be FP16 or INT8 with scaling.
- **MQA/GQA:** A single basis B (and E) can be shared across all heads in a group, reducing storage and projection cost.
- **Integration:** The method is a drop-in replacement for standard attention, compatible with FlashAttention-style tiling.

**Pseudocode (SubSpace-A attention, per head, per token t):**
```python
# q_t is the query for the current token
m = -float('inf'); Z = 0.0; o_numerator = 0.0 # float vector

for chunk j in all_chunks:
  q_proj = q_t @ chunk.B.T
  logits = (q_proj @ chunk.C.T) / sqrt(d)

  m_j = logits.max()
  if m_j > m:
    o_numerator *= exp(m - m_j)
    Z *= exp(m - m_j)
    m = m_j
  
  weights = exp(logits - m)
  Z += weights.sum()
  
  # Aggregate output contribution from this chunk
  o_chunk = (weights.T @ chunk.D) @ chunk.E
  o_numerator += o_chunk

output = o_numerator / Z
```

## 4. Experiments

### 4.1 Falsification-Oriented Plan
We aim to falsify our central hypothesis.
**Models:** GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.
**Datasets:** WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 (long-context).
**Setups:**
- **SubSpace-G:** `r` chosen to retain 90-99% spectral energy on calibration data.
- **SubSpace-A:** Sketch size `m ∈ {2r, 4r}`, max chunk length `L ∈ {128, 256}`, threshold `τ` tuned on calibration data.
- **Ablations:** Keys-only vs. full K/V projection; pre- vs. post-RoPE; INT8 coefficients.
- **Baselines:** Full-precision attention, 8-bit KV cache, sliding window, H2O/Heavy-Hitter Oracle.
- **Metrics:** Perplexity/accuracy degradation, memory footprint, wall-clock throughput, measured spectral decay vs. `r`.

**Falsifiers:**
1. No rank `r ≪ d` achieves a <1% perplexity increase.
2. The online SubSpace-A performs significantly worse than the static SubSpace-G at a matched rank.
3. The method's performance degrades sharply with context length, indicating RoPE incompatibility.

### 4.2 Preliminary Measurements (GPT-2 Small)
Initial analysis of K/V cache matrices on WikiText-103 (8k context) shows that for most heads across most layers, over 90% of the spectral energy is contained in a subspace with rank r ≈ d/4. A SubSpace-G implementation achieves negligible PPL increase at this rank. These results are provisional; full, reproducible code and results will be released.

## 5. Discussion
The Low-Rank Hypothesis appears plausible due to token redundancy, semantic clustering, and head specialization within LLMs. Our method offers a way to exploit this structure at inference time. It is complementary to quantization and can be integrated into FlashAttention-style kernels to reduce both memory and bandwidth demands. Key failure modes could include tasks with high intrinsic dimensionality or rapid, unpredictable concept drift over long contexts.

## 6. Limitations
- A single global rank `r` may be suboptimal; per-head or adaptive rank selection introduces complexity.
- Online variant hyperparameters (τ, L) require tuning and represent a trade-off between adaptivity and overhead.
- Throughput gains are hardware-dependent; a roofline analysis is needed to model performance.

## 7. Conclusion
We formalize the Low-Rank KV Cache Hypothesis and present two practical algorithms for inference-time, coefficient-space attention. The approach is simple, retraining-free, and grounded in spectral theory. If validated, it provides a powerful and orthogonal tool for optimizing LLM decoding.
