Minor Revisions

The paper presents a novel and well-supported method for KV cache compression. The core hypothesis is compelling, the proposed algorithms are clearly described, and the theoretical analysis is rigorous. The falsification-oriented experimental design is a model of good scientific practice. The following minor revisions would further strengthen the manuscript.

- **Hypothesis Definition:** The central "Low-Rank KV Cache Hypothesis" should be defined more formally in the introduction, perhaps by referencing the rapid decay of singular values or the high proportion of spectral energy captured by a low-rank subspace. This would make the core claim more precise and directly testable.
- **FLOPs Analysis:** The complexity analysis in Section 3.2 is correct but would be more impactful if it directly compared the total per-token FLOPs to the baseline `O(Td)` attention. Explicitly showing the savings from `O(Td)` to `O(T(r+r_v) + (r+r_v)d)` would make the computational benefit clearer.
- **RoPE Interaction:** The explanation in Section 3.5 is brief. It could be enhanced by clarifying *why* RoPE challenges the low-rank assumption (i.e., per-token rotations complicate the global spectrum) and how adaptive chunking specifically addresses this by finding locally consistent subspaces.
- **Error Bound Interpretation:** The multi-chunk error bound at the end of Section 4 is dense. A concluding sentence summarizing it in plain language—e.g., that the total error depends on the worst-case key projection error for weights and the worst-case value projection error for the output—would improve readability.
- **Algorithm Naming:** The names "SubSpace-G" and "SubSpace-A" are functional but could be more descriptive. Consider "Static Subspace" and "Adaptive Subspace" for immediate clarity.
- **Experimental Validation:** The experimental plan should explicitly list "Direct validation of the Low-Rank Hypothesis via spectral decay analysis" as a primary goal, reinforcing the paper's central theme.

### Revised Draft
# Online Subspace Projection for KV Caches: Low-Rank Attention at Inference Time

## Abstract
We propose and test the Low-Rank KV Cache Hypothesis: during autoregressive decoding, an attention head’s keys and values concentrate in a low-dimensional subspace. This structure enables a training-free compression scheme that computes attention in a coefficient space defined by head-specific, rank‑r bases. This factorization reduces per-head KV memory from O(T d) to O(T r) and attention FLOPs from O(T d) to O(T r). We introduce two algorithms: (1) a static method with fixed bases learned from calibration data, and (2) an online adaptive variant using Frequent Directions (FD) sketches with residual-triggered chunking. We provide a numerically stable blockwise softmax to aggregate attention across chunks with distinct bases, derive operator-norm error bounds, and outline a falsification-oriented evaluation. Preliminary results on GPT-2 Small show minimal perplexity degradation at r ≪ d.

## 1. Introduction
The Key-Value (KV) cache is a primary memory and bandwidth bottleneck in autoregressive LLM inference. Existing approaches compress this cache via quantization, eviction, or retraining with specialized low-rank architectures. We propose a training-free alternative: compute attention in per-head low-dimensional subspaces, learned offline or adapted online from the K/V streams.

**Contributions:**
- **Hypothesis and Factorization:** We formalize the Low-Rank KV Cache Hypothesis: the singular value spectrum of K and V matrices decays rapidly, allowing a rank-r approximation to capture most of the variance. We compute attention in a coefficient space defined by rank‑r (keys) and rank‑r_v (values) bases.
- **Adaptive Algorithm:** A continuous Frequent Directions (FD) sketch per head supplies subspaces; residual-triggered chunking yields local bases without revisiting historical tokens.
- **Stable Aggregation:** A single-pass, numerically stable blockwise softmax aggregates logits and value contributions across chunks that use different bases.
- **Guarantees:** We provide operator-norm attention error bounds linked to FD sketch guarantees, incorporating a tight softmax sensitivity bound.
- **Validation:** A falsification-oriented experimental plan on small open-source models with reproducible code.

## 2. Related Work
- **Training-time low-rank attention** (e.g., Linformer, Nyströmformer) constrains attention during training. We operate post-hoc, at inference, adapting subspaces to the live data stream.
- **KV compression/eviction** (8-bit KV, StreamingLLM, H2O) operate in the original d-dimensional space; we compute attention within a compressed coefficient space, reducing FLOPs as well as memory.
- **Spectral analyses** have reported low-rank structure in offline datasets; we provide an online, numerically stable algorithm compatible with standard kernels and RoPE.

## 3. Method

### 3.1 Notation and Baseline
Per head, with key/value dimension d and T tokens:
- Query q_t ∈ R^d, keys K ∈ R^{T×d}, values V ∈ R^{T×d}.
- Logits ℓ_i = q_t·k_i / √d; causal mask M ∈ {0, −∞}^T.
- Weights α = softmax(ℓ + M); output o_t = αᵀ V.
- Baseline complexity per token is dominated by attention: O(T d).

### 3.2 Coefficient-Space Attention (Single Basis)
Let B ∈ R^{r×d}, E ∈ R^{r_v×d} have orthonormal rows (B Bᵀ = I_r, E Eᵀ = I_{r_v}). Define projectors P_B = Bᵀ B, P_E = Eᵀ E (both d×d).

- Key coefficients C = K Bᵀ ∈ R^{T×r}, value coefficients D = V Eᵀ ∈ R^{T×r_v}.
- Projected query q̃ = q_t Bᵀ ∈ R^r.

Projected attention:
- Logits: ℓ̂_i = q̃·c_i / √d = q_tᵀ P_B k_i / √d.
- Weights: α̂ = softmax(ℓ̂ + M).
- Output: ô_t = (α̂ᵀ D) E ∈ R^d.

**Complexity per head:**
- Storage: C(T×r), D(T×r_v), plus bases B(r×d), E(r_v×d).
- Per-token FLOPs: O(rd) for query projection, O(Tr) for logits, and O(Tr_v + r_v d) for output. For T ≫ d, the total is O(T(r+r_v)), a significant reduction from the baseline O(Td) when r ≪ d.

*Remark on scaling:* We retain √d to approximate qᵀk/√d with qᵀP_B k/√d; an optional scalar calibration can be ablated.

### 3.3 Adaptive Online Variant with Chunking
We adapt to distributional drift using chunk-specific bases guided by continuous FD sketches.

**State per head:**
- FD sketches: S_K ∈ R^{ℓ_k×d} for keys, S_V ∈ R^{ℓ_v×d} for values (ℓ_k ≳ r, ℓ_v ≳ r_v).
- Active chunk j with bases (B_j, E_j), coefficient buffers (C_j, D_j), length L_j.
- Hyperparameters: residual thresholds τ_k, τ_v; max chunk length L; sketch sizes ℓ_k, ℓ_v.

**Per token (post-RoPE k_t, v_t, q_t):**
1. **Residuals:** res_k = ||k_t(I − P_{B_j})||_2, res_v = ||v_t(I − P_{E_j})||_2.
2. **If** res_k > τ_k or res_v > τ_v or L_j = L:
   - Finalize chunk j.
   - Snapshot SVDs of S_K and S_V to form new bases B_{j+1} (top‑r right singular vectors) and E_{j+1} (top‑r_v).
3. Append coefficients: c_t = k_t B_jᵀ to C_j; d_t = v_t E_jᵀ to D_j; increment L_j.
4. Update sketches with k_t and v_t (FD per-update cost O(dℓ_k) and O(dℓ_v)).

The sketches persist across tokens; chunk bases are snapshots taken only when triggers fire.

### 3.4 Blockwise Softmax Across Chunks
Given J historical chunks with (C_j ∈ R^{L_j×r}, D_j ∈ R^{L_j×r_v}, B_j, E_j):
- Per-chunk query: q̃_j = q_t B_jᵀ ∈ R^r.
- Per-chunk logits vector: ℓ_j = (q̃_j C_jᵀ)/√d + M_j ∈ R^{L_j}.

A numerically stable one-pass aggregation computes the final output:
- Initialize m = −∞, Z = 0, N = 0_d.
- For j = 1..J:
  1. m_j = max(ℓ_j).
  2. If m_j > m: scale Z ← Z·exp(m − m_j); N ← N·exp(m − m_j); m ← m_j.
  3. w_j = exp(ℓ_j − m) ∈ R^{L_j}.
  4. Z ← Z + sum(w_j).
  5. N ← N + (w_jᵀ D_j) E_j ∈ R^d.
- Return ô_t = N / Z.

This integrates causal masking (via M_j), never revisits past tokens, and remains numerically stable.

### 3.5 RoPE Compatibility
Projection is applied post-RoPE. While RoPE applies a unique rotation to each token's embedding, which complicates the global K/V spectrum, we empirically observe that the subspace remains locally stable. The adaptive residual-triggered chunking method is designed to exploit this, creating new bases precisely when these rotations shift the active subspace significantly.

## 4. Error Guarantees
Let ΔK = K(I − P_B) and ΔV = V(I − P_E) be the residual matrices.

- **Logit error** (single basis): For any query q, the error in the logits is bounded by:
  ||ℓ − ℓ̂||_∞ = (1/√d) ||K(I-P_B)q||_∞ ≤ (1/√d) ||K(I − P_B)||_op ||q||_2.
- **Softmax sensitivity** (tight form):
  ||α − α̂||_1 ≤ 2 tanh(||ℓ − ℓ̂||_∞ / 2) ≤ ||ℓ − ℓ̂||_∞.
- **Output error**:
  ||o − ô||_2 = ||αᵀV − α̂ᵀ(V P_E)||_2
  ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂ᵀ ΔV||_2
  ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||ΔV||_op.

**Frequent Directions linkage:** For a matrix A ∈ R^{T×d}, an FD sketch S ∈ R^{ℓ×d} with ℓ > r satisfies ||AᵀA − SᵀS||_op ≤ ||A − A_r||_F^2 / (ℓ − r) = ε. Let P be the projector onto the top‑r right singular subspace of S. Then ||A(I − P)||_op ≤ √ε. Applying this with A=K and A=V links the sketch accuracy to the operator norms of the residual matrices:
||ΔK||_op ≤ √( ||K − K_r||_F^2 / (ℓ_k − r) ), and ||ΔV||_op ≤ √( ||V − V_{r_v}||_F^2 / (ℓ_v − r_v) ).

**Multi-chunk extension:** With chunk-specific projectors P_{B_j}, P_{E_j}, the bounds extend naturally.
- The logit error depends on the maximum error over any chunk:
  ||ℓ − ℓ̂||_∞ ≤ (||q||_2/√d) max_j ||K_j(I − P_{B_j})||_op.
- The total output error is bounded by:
  ||o − ô||_2 ≤ 2 tanh( (||q||_2/2√d) max_j ||K_j(I − P_{B_j})||_op ) max_i ||v_i||_2 + max_j ||V_j(I − P_{E_j})||_op.

In short, the total error is governed by the worst-case projection error across all key chunks (affecting weights) plus the worst-case projection error across all value chunks (affecting the output). Sketch sizes (ℓ_k, ℓ_v), ranks (r, r_v), and chunking control this error.

## 5. Implementation Notes
- **Data types:** Store B, E in FP16; C, D in FP16 or INT8 with per-tile scales.
- **MQA/GQA:** Share bases within a query group to amortize projections.
- **Costs:** FD updates are O(dℓ) per token; snapshot SVDs on ℓ×d sketches are O(dℓr). With d ≈ 128–256 and ℓ ≈ 2r, overhead is modest.
- **Memory per head:** O(T(r + r_v) + J(r + r_v)d + (ℓ_k + ℓ_v)d). The approach is memory-efficient when basis overhead is amortized: Jd ≪ T (i.e., long chunks, few chunks).
- **Integration:** Implement as a drop-in attention module. The blockwise softmax aligns well with FlashAttention-style streaming computation.

## 6. Algorithms
- **Static Subspace:** From calibration data K,V, compute fixed bases B,E (top right singular vectors). Use single-basis attention (Section 3.2) at inference.
- **Adaptive Subspace:** Maintain sketches S_K, S_V per head. On residual triggers or every L tokens, snapshot subspaces to define new chunk bases. Aggregate via blockwise softmax (Section 3.4).

## 7. Experiments
**Models:** GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.
**Datasets:** WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 (long-context).

**Setups:**
- **Static Subspace:** r selected to retain 90–99% spectral energy on calibration data.
- **Adaptive Subspace:** ℓ_k ∈ {2r, 4r}, ℓ_v ∈ {2r_v, 4r_v}, L ∈ {128, 256}; thresholds τ_k, τ_v tuned; r_v ∈ {r, r/2}.
- **Ablations:** Keys-only vs K+V; pre- vs post-RoPE; INT8 coefficients; shared vs per-head bases; optional output-scale calibration.
- **Baselines:** FP16 attention, 8-bit KV cache, sliding window, H2O.
- **Metrics:** ΔPPL/accuracy, throughput, KV memory, FD error ε vs observed error. A primary metric will be a direct validation of the Low-Rank Hypothesis via spectral decay analysis on K/V matrices.

**Falsifiers:**
1. No rank r ≪ d achieves ≤1% PPL increase.
2. Adaptive Subspace underperforms Static Subspace at matched r (adaptivity not beneficial).
3. Error grows sharply with context length despite chunking (incompatibility with RoPE).
4. The FD error bound ε poorly predicts logit/output error in practice.

*Preliminary (GPT-2 Small):* On WikiText-103 (8k context), most heads retain ≥90% spectral energy at r ≈ d/4. The Static Subspace method shows negligible PPL increase at this rank. Code will be released.

## 8. Discussion
Coefficient-space attention reduces KV memory and bandwidth without retraining, complementing quantization and eviction. The main failure mode is intrinsically high-dimensional K/V structure where low-rank approximations are ineffective. Our adaptive algorithm mitigates this risk but cannot eliminate it.

## 9. Limitations
- A single global r may be suboptimal; per-head or adaptive rank selection adds complexity.
- The adaptive method introduces hyperparameters (τ, L, ℓ) that require tuning.
- Throughput gains depend on high-quality kernel implementations; a roofline analysis is warranted.

## 10. Conclusion
We operationalize the Low-Rank KV Cache Hypothesis through an online subspace projection framework. Our static and adaptive algorithms compute attention in a compressed coefficient space with principled error control. If validated, this approach can materially reduce the memory and bandwidth demands of LLM inference, making large models more accessible on commodity hardware.
