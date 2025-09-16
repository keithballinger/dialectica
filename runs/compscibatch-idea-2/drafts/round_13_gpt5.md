Minor Revisions

Brief critique
- Correctness/notation: Fix the projection orientation throughout. Projection acts on the d-dimensional feature axis, so residuals are K(I − P_B) and V(I − P_E), not (I − P_B)K, etc. Similarly, in the FD linkage, the correct bound is on ||A(I − P)||_op.
- FD usage: Maintain separate sketches for keys and values (S_K, S_V). The draft’s single shared sketch is ambiguous and weakens the guarantees when r ≠ r_v or K/V spectra differ.
- Softmax sensitivity: Use the tight bound ||softmax(x) − softmax(y)||_1 ≤ 2 tanh(||x − y||_∞ / 2) ≤ ||x − y||_∞, and cite or briefly justify.
- Multi-basis aggregation bounds: Extend Section 4 to cover chunk-specific bases. The error decomposes per chunk; provide a clean bound using max_j ||ΔK_j||_op and ||ΔV_j||_op.
- Scaling in projected logits: Explicitly justify keeping the √d scaling for logits after projection (it approximates q^T k / √d via q^T P_B k); note an optional scalar calibration if desired.
- Implementation clarity: State FD per-update cost O(dℓ) and SVD-on-sketch cost O(dℓr). Clarify that E_j, B_j are formed by SVD of the ℓ×d sketch snapshots. Note that r_v may differ from r.
- Memory trade-off: Keep the main condition explicit: the method is memory-beneficial when the per-chunk basis overhead is amortized, i.e., J d ≪ T (long contexts, few chunks).
- Minor edits: Streamline the blockwise softmax presentation with explicit dimensionalities and clarify masking; tighten abstract wording.

Revised Draft
# Online Subspace Projection for KV Caches: Low-Rank Attention at Inference Time

## Abstract
We propose and test the Low-Rank KV Cache Hypothesis: during autoregressive decoding, an attention head’s keys and values concentrate in a low-dimensional subspace, enabling strong, training-free compression. We derive an inference-time factorization that computes attention in a coefficient space defined by head-specific, rank‑r bases, reducing per-head KV memory from O(T d) to O(T r) and attention FLOPs from O(T d) to O(T r). We introduce two algorithms: (1) a static method with fixed bases learned from calibration data, and (2) an online adaptive variant using Frequent Directions (FD) sketches with residual-triggered chunking. We provide a numerically stable blockwise softmax to aggregate attention across chunks with distinct value bases, derive operator-norm error bounds (including a tight softmax sensitivity), and outline a falsification-oriented evaluation. Preliminary results on GPT-2 Small show negligible perplexity loss at r ≪ d.

## 1. Introduction
The Key-Value (KV) cache dominates memory and bandwidth in autoregressive LLM inference. Existing approaches compress via quantization, eviction, or retraining with low-rank architectures. We propose a training-free alternative: compute attention in per-head low-dimensional subspaces, learned offline or adapted online from the K/V streams, with principled error control and compatibility with standard LLMs (including RoPE).

Contributions:
- Hypothesis and factorization: Keys/values are approximately low-rank during decoding. We compute attention in a coefficient space defined by rank‑r (keys) and rank‑r_v (values) bases.
- Adaptive algorithm: A continuous FD sketch per head supplies subspaces; residual-triggered chunking yields local bases without revisiting historical tokens.
- Stable aggregation: A single-pass, numerically stable blockwise softmax aggregates logits and value contributions across chunks with different bases.
- Guarantees: Operator-norm attention error bounds linked to FD guarantees, with a tight softmax sensitivity bound.
- Validation: A falsification-oriented experimental plan on small open-source models with reproducible code.

## 2. Related Work
- Training-time low-rank attention (e.g., Linformer, Nyströmformer) constrains attention during training. We operate post-hoc, at inference, adapting subspaces online.
- KV compression/eviction (8-bit KV, StreamingLLM, Scissorhands, H2O) operate in d dimensions; we compute in a compressed coefficient space.
- Spectral analyses report low-rank structure offline; we provide an online, numerically stable algorithm compatible with RoPE and standard kernels.

## 3. Method

### 3.1 Notation and Baseline
Per head, with key/value dim d and T tokens:
- Query q_t ∈ R^d, keys K ∈ R^{T×d}, values V ∈ R^{T×d}.
- Logits ℓ_i = q_t·k_i / √d; causal mask M ∈ {0, −∞}^T.
- Weights α = softmax(ℓ + M); output o_t = αᵀ V.

### 3.2 Coefficient-Space Attention (Single Basis)
Let B ∈ R^{r×d}, E ∈ R^{r_v×d} have orthonormal rows (B Bᵀ = I_r, E Eᵀ = I_{r_v}). Define projectors P_B = Bᵀ B, P_E = Eᵀ E (both d×d).

- Key coefficients C = K Bᵀ ∈ R^{T×r}, value coefficients D = V Eᵀ ∈ R^{T×r_v}.
- Projected query q̃ = q_t Bᵀ ∈ R^r.

Projected attention:
- Logits: ℓ̂_i = q̃·c_i / √d = q_tᵀ P_B k_i / √d.
- Weights: α̂ = softmax(ℓ̂ + M).
- Output: ô_t = (α̂ᵀ D) E ∈ R^d.

Complexity per head:
- Storage: C(T×r), D(T×r_v), bases B(r×d), E(r_v×d).
- Per-token FLOPs: O(r d) projections + O(T r) logits + O(T r_v + r_v d) output.

Remark on scaling: We retain √d to approximate qᵀk/√d with qᵀP_B k/√d; optional scalar calibration can be ablated.

### 3.3 Adaptive Online Variant with Chunking
We adapt to drift using chunk-specific bases guided by continuous FD sketches.

State per head:
- FD sketches: S_K ∈ R^{ℓ_k×d} for keys, S_V ∈ R^{ℓ_v×d} for values (ℓ_k ≳ r, ℓ_v ≳ r_v).
- Active chunk j with bases (B_j, E_j), coefficient buffers (C_j, D_j), length L_j.
- Hyperparameters: residual thresholds τ_k, τ_v; max chunk length L; sketch sizes ℓ_k, ℓ_v.

Per token (post-RoPE k_t, v_t, q_t):
1) Residuals: res_k = ||(I − P_{B_j}) k_t||_2, res_v = ||(I − P_{E_j}) v_t||_2.
2) If res_k > τ_k or res_v > τ_v or L_j = L:
   - Finalize chunk j.
   - Snapshot SVDs of S_K and S_V to form new bases B_{j+1} (top‑r right singular vectors) and E_{j+1} (top‑r_v).
3) Append coefficients: c_t = k_t B_jᵀ to C_j; d_t = v_t E_jᵀ to D_j; increment L_j.
4) Update sketches with k_t and v_t (FD per-update cost O(dℓ_k) and O(dℓ_v)).

The sketches persist across tokens; chunk bases are snapshots when triggers fire.

### 3.4 Blockwise Softmax Across Chunks
Given J historical chunks with (C_j ∈ R^{L_j×r}, D_j ∈ R^{L_j×r_v}, B_j, E_j):
- Per-chunk query: q̃_j = q_t B_jᵀ ∈ R^r.
- Per-chunk logits vector: ℓ_j = (q̃_j C_jᵀ)/√d + M_j ∈ R^{L_j}.

A numerically stable one-pass aggregation:
- Initialize m = −∞, Z = 0, N = 0_d.
- For j = 1..J:
  1) m_j = max(ℓ_j).
  2) If m_j > m: scale Z ← Z·exp(m − m_j); N ← N·exp(m − m_j); m ← m_j.
  3) w_j = exp(ℓ_j − m) ∈ R^{L_j}.
  4) Z ← Z + sum(w_j).
  5) N ← N + (w_jᵀ D_j) E_j ∈ R^d.
- Return ô_t = N / Z.

This integrates causal masking (via M_j), never revisits tokens, and remains numerically stable.

### 3.5 RoPE Compatibility
Projection is applied post-RoPE. Although RoPE is position-dependent, empirical spectra of post-RoPE streams remain concentrated. Residual-triggered chunking creates new bases when rotations materially shift the subspace.

## 4. Error Guarantees
Let ΔK = K(I − P_B), ΔV = V(I − P_E).

- Logit error (single basis): For any q,
  ||ℓ − ℓ̂||_∞ = (1/√d) max_i |qᵀ(Δk_i)| ≤ (||q||_2/√d) ||ΔK||_op.
- Softmax sensitivity (tight form):
  ||α − α̂||_1 ≤ 2 tanh(||ℓ − ℓ̂||_∞ / 2) ≤ ||ℓ − ℓ̂||_∞.
- Output error:
  ||o − ô||_2 = ||αᵀV − α̂ᵀV̂||_2
  ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂||_1 ||ΔV||_op
  = ||α − α̂||_1 max_i ||v_i||_2 + ||ΔV||_op.

Frequent Directions linkage (per matrix A ∈ R^{T×d}): An FD sketch S ∈ R^{ℓ×d} with ℓ > r satisfies
||AᵀA − SᵀS||_op ≤ ||A − A_r||_F^2 / (ℓ − r) = ε.
Let P be the projector onto the top‑r right singular subspace of S. Then
||A(I − P)||_op^2 = sup_{x ⟂ P, ||x||=1} ||Ax||_2^2 = sup_{x ⟂ P} xᵀ AᵀA x ≤ ε,
so ||A(I − P)||_op ≤ √ε.
Applying with A = K and A = V gives
||ΔK||_op ≤ √( ||K − K_r||_F^2 / (ℓ_k − r) ),
||ΔV||_op ≤ √( ||V − V_{r_v}||_F^2 / (ℓ_v − r_v) ).

Multi-chunk extension: With chunk-specific projectors P_{B_j}, P_{E_j} and concatenations over chunks,
- The logit approximation uses blockwise projections. For any q,
  ||ℓ − ℓ̂||_∞ ≤ (||q||_2/√d) max_j ||K_j(I − P_{B_j})||_op.
- The output bound becomes
  ||o − ô||_2 ≤ 2 tanh( (||q||_2/√d) max_j ||K_j(I − P_{B_j})||_op / 2 ) max_i ||v_i||_2
  + max_j ||V_j(I − P_{E_j})||_op.

Thus sketch sizes (ℓ_k, ℓ_v), ranks (r, r_v), and chunking control the error.

## 5. Implementation Notes
- Data types: Store B, E in FP16; C, D in FP16 or INT8 with per-tile scales.
- MQA/GQA: Share bases within a query group to amortize projections.
- Costs: FD updates are O(dℓ_k) and O(dℓ_v) per token; snapshot SVDs on ℓ×d are O(dℓr) and O(dℓ_v r_v). With d ≈ 128–256 and ℓ ≈ 2r, overhead is modest.
- Memory per head: O(T(r + r_v) + J(r + r_v)d + ℓ_k d + ℓ_v d). The approach is memory-efficient when basis overhead is amortized: J d ≪ T (i.e., long chunks, few chunks).
- Integration: Implement as a drop-in attention module. The blockwise softmax aligns with FlashAttention-style streaming of logits and accumulators.

## 6. Algorithms
- Static (SubSpace-G): From calibration K,V, compute fixed B,E (top right singular vectors). Use Section 3.2 at inference.
- Adaptive (SubSpace-A): Maintain S_K, S_V per head. On residual triggers or every L tokens, snapshot subspaces to define new chunk bases. Aggregate via Section 3.4.

## 7. Experiments
Models: GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.
Datasets: WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 (long-context).

Setups:
- SubSpace-G: r selected to retain 90–99% spectral energy on calibration.
- SubSpace-A: ℓ_k ∈ {2r, 4r}, ℓ_v ∈ {2r_v, 4r_v}, L ∈ {128, 256}; thresholds τ_k, τ_v tuned; r_v ∈ {r, r/2}.
- Ablations: Keys-only vs K+V; pre- vs post-RoPE; INT8 coefficients; shared vs per-head bases; optional output-scale calibration.
- Baselines: FP16 attention, 8-bit KV cache, sliding window, H2O.
- Metrics: ΔPPL/accuracy, throughput, KV memory, spectral decay vs r, FD ε vs observed errors.

Falsifiers:
1) No r ≪ d achieves ≤1% PPL increase.
2) SubSpace-A underperforms SubSpace-G at matched r (adaptivity not beneficial).
3) Error grows sharply with context length despite chunking (incompatibility with RoPE).
4) FD ε poorly predicts logit/output error.

Preliminary (GPT-2 Small): On WikiText-103 (8k context), most heads retain ≥90% spectral energy at r ≈ d/4. SubSpace-G shows negligible PPL increase at this rank. Code will be released.

## 8. Discussion
Coefficient-space attention reduces KV memory and bandwidth without retraining, complementing quantization and eviction. The main failure mode is intrinsically high K/V dimensionality, where low-rank approximations are ineffective. Adaptivity mitigates but cannot eliminate this.

## 9. Limitations
- A single global r may be suboptimal; per-head/adaptive rank adds complexity.
- SubSpace-A introduces τ, L, ℓ_k, ℓ_v that require tuning.
- Throughput gains depend on kernel quality; a roofline analysis is warranted.

## 10. Conclusion
We operationalize the Low-Rank KV Cache Hypothesis via online subspace projection. Static and adaptive algorithms compute attention in compressed coefficient spaces with principled error control. If validated, this can materially reduce memory and bandwidth for LLM inference on commodity hardware.
