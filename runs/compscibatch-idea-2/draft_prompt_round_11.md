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

Brief critique
- Blockwise aggregation: The cross-chunk softmax with distinct value bases needs an explicit formula and a numerically stable single-pass algorithm that incorporates causal masking. The current pseudocode is nearly correct but omits mask handling and the exact reconstruction formula.
- FD-to-operator-norm link: The draft references FD via a Frobenius bound but does not state a clean spectral/operator bound for the projection residual used in the logit error. A standard FD inequality can yield ||(I − P)K||_op ≤ sqrt(||K − K_r||_F^2/(ℓ − r)); this should be stated and cited.
- Memory model for per-chunk bases: The online chunked variant implies storing E_j (and possibly B_j) per chunk; the overhead and how to keep it bounded should be made explicit.
- Masking and RoPE details: Clarify that the blockwise softmax integrates causal masks and that projection is applied post-RoPE. Note that RoPE is position-varying; orthogonality preserves norms but does not guarantee identical subspaces—state the practical implication and fallback (adaptive rank/chunking).
- Title and framing: A crisper title foregrounding online subspace projection improves impact.

Revised Draft
# Online Subspace Projection for KV Caches: Low-Rank Attention at Inference Time

## Abstract
We posit the Low-Rank KV Cache Hypothesis: during autoregressive decoding, per-head keys and values concentrate in a low-dimensional subspace, enabling strong compression without retraining. We derive an inference-time factorization that computes attention in coefficient space using head-specific rank-r bases, reducing per-head KV memory from O(T d) to O(T r) and attention FLOPs from O(T d) to O(T r). We introduce two practical algorithms: (1) a static method with fixed per-head bases learned from calibration, and (2) an adaptive online variant using a continuous Frequent-Directions (FD) sketch and residual-triggered chunking. We give numerically stable blockwise softmax across chunks with distinct value bases, operator-norm error bounds from FD, and a falsification-oriented evaluation. Preliminary GPT-2 Small results show negligible perplexity loss at r ≪ d.

## 1. Introduction
KV cache bandwidth dominates autoregressive LLM inference. Existing methods reduce precision (quantization), prune tokens (eviction), or retrain architectures (low-rank attention). We propose an orthogonal, training-free approach: compute attention in low-dimensional, per-head subspaces learned from the K/V streams themselves.

Contributions:
- Hypothesis: Per-head K/V streams are approximately low-rank during decoding; projecting them preserves attention outputs at r ≪ d.
- Method: Inference-time factorization K ≈ C B and V ≈ D E with orthonormal-row bases (B, E), computing logits and outputs in coefficient space.
- Adaptive algorithm: Continuous FD sketching to maintain a global subspace with residual-triggered chunking, avoiding global recomputation and revisiting past tokens.
- Stable aggregation: A single-pass, numerically stable blockwise softmax that aggregates logits and value contributions across chunks with different bases.
- Guarantees: Operator-norm error bounds derived from FD’s covariance guarantee, linking sketch size and rank to logit/output error.
- Validation: An ablation-driven, falsification-oriented plan on small open-source models with reproducible code.

## 2. Related Work
- Training-time low-rank attention (e.g., Linformer, Nyströmformer) constrains attention during training, not post-hoc KV caches or online inference subspaces.
- KV compression and eviction (8-bit KV, StreamingLLM, Scissorhands, H2O) reduce precision or tokens but continue to operate in d dimensions and do not compute attention in an adaptive, compressed coefficient space.
- Subspace/SVD analyses provide offline evidence; we present an online, numerically stable, inference-time realization compatible with standard LLMs and RoPE.

## 3. Method

### 3.1 Notation and baseline
Per head, with key/value dim d and T tokens:
- Query q_t ∈ R^d, keys K ∈ R^{T×d}, values V ∈ R^{T×d}.
- Logits ℓ_i = q_t·k_i / √d; causal mask M ∈ {0, −∞}^T.
- Weights α = softmax(ℓ + M); output o_t = αᵀ V.

### 3.2 Coefficient-space attention (single basis)
Let B ∈ R^{r×d}, E ∈ R^{r_v×d} have orthonormal rows (BBᵀ = I_r, EEᵀ = I_{r_v}). Define
- Key coeffs C = K Bᵀ ∈ R^{T×r}, value coeffs D = V Eᵀ ∈ R^{T×r_v}.
- Projected query q̃ = q_t Bᵀ ∈ R^r.

Then projected logits and output:
- ℓ̂_i = q̃·c_i / √d = q_tᵀ(BᵀB)k_i/√d.
- α̂ = softmax(ℓ̂ + M).
- ô_t = (α̂ᵀ D) E.

Compute/storage per head:
- Storage: C(T×r), D(T×r_v), B(r×d), E(r_v×d).
- Per-token: O(r d) for projections, O(T r) for logits, O(T r_v + r_v d) for output.

### 3.3 Adaptive online variant with chunking
We allow chunk-specific bases to adapt to drift while maintaining a continuous global FD sketch.

State per head:
- FD sketch S ∈ R^{ℓ×d} (ℓ ≥ 2r) updated on every new k_t and v_t.
- Active chunk j with bases (B_j ∈ R^{r×d}, E_j ∈ R^{r_v×d}), coefficient buffers C_j, D_j, length L_j.
- Hyperparameters: residual thresholds τ_k, τ_v, max chunk length L, sketch size ℓ.

Per-token procedure (post-RoPE k_t, v_t, q_t):
1) Compute residuals r_k = ||(I − B_jᵀB_j) k_t||_2, r_v = ||(I − E_jᵀE_j) v_t||_2.
2) If r_k > τ_k or r_v > τ_v or L_j = L:
   - Finalize chunk j.
   - Form new chunk j+1 with B_{j+1}, E_{j+1} from the top-r (and r_v) right singular vectors of the current FD sketch S.
3) Append coefficients: c_t = k_t B_jᵀ to C_j; d_t = v_t E_jᵀ to D_j; increment L_j.
4) Update FD sketch S with k_t and v_t (streaming).

Notes:
- The FD sketch S is continuous across all tokens; chunk bases are snapshots of S when thresholds fire.
- To bound overhead, keep ℓ small (e.g., ℓ ∈ {2r, 4r}) and r_v ≤ r; only E_j and D_j are needed to reconstruct values from chunk j.

### 3.4 Blockwise softmax across chunks with distinct value bases
Let chunks j = 1..J, with C_j ∈ R^{T_j×r}, D_j ∈ R^{T_j×r_v}, bases (B_j, E_j). For a new query q_t:
- Per chunk logits ℓ_j ∈ R^{T_j}: ℓ_{j} = (q_t B_jᵀ) C_jᵀ / √d + M_j,
  where M_j is the slice of the causal mask for tokens in chunk j.

Define m = max_{j,i} ℓ_{j,i}, Z = ∑_{j,i} exp(ℓ_{j,i} − m), and
N = ∑_{j} [ (∑_{i=1}^{T_j} exp(ℓ_{j,i} − m) d_{j,i}) E_j ],
where d_{j,i} is the i-th row of D_j.

Then the numerically stable, exact blockwise aggregation is:
- ô_t = N / Z.

Single-pass streaming algorithm:
- Initialize m = −∞, Z = 0, N = 0_d.
- For j in 1..J:
  - Compute q̃_j = q_t B_jᵀ; ℓ_j = q̃_j C_jᵀ / √d + M_j.
  - m_j = max(ℓ_j).
  - If m_j > m: scale Z ← Z·exp(m − m_j), N ← N·exp(m − m_j), m ← m_j.
  - w_j = exp(ℓ_j − m).
  - Z ← Z + sum(w_j).
  - N ← N + (w_jᵀ D_j) E_j.
- Return ô_t = N / Z.

This handles masks and avoids revisiting chunks while preserving numerical stability.

### 3.5 RoPE compatibility
Apply projection post-RoPE. RoPE applies orthogonal, position-dependent rotations; orthogonality preserves norms and inner products within any fixed subspace. Although rotations vary with position, empirical spectra of post-RoPE K/V remain concentrated; when drift increases, residual-triggered chunking refreshes the basis.

## 4. Error guarantees
Let P_B = BᵀB, P_E = EᵀE be orthogonal projectors; define ΔK = (I − P_B)K and ΔV = (I − P_E)V.

- Logit error: For any query q,
  ||ℓ − ℓ̂||_∞ = (1/√d) max_i |qᵀΔk_i| ≤ (||q||_2/√d) max_i ||Δk_i||_2 ≤ (||q||_2/√d) ||ΔK||_op.
- Softmax sensitivity: For vectors x,y, ||softmax(x) − softmax(y)||_1 ≤ 2 ||x − y||_∞.
  Hence ||α − α̂||_1 ≤ 2 ||ℓ − ℓ̂||_∞.
- Output error:
  ||o − ô||_2 = ||αᵀV − α̂ᵀV̂||_2
  ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||ΔV||_op
  ≤ 2(||q||_2/√d) ||ΔK||_op max_i ||v_i||_2 + ||ΔV||_op.

Frequent-Directions linkage (sketch size ℓ ≥ r+1):
- For any A ∈ R^{T×d} and FD sketch S ∈ R^{ℓ×d}, FD ensures 0 ≼ AᵀA − SᵀS ≼ ε I with ε ≤ ||A − A_r||_F^2/(ℓ − r).
- Let P be the projector onto the top-r right singular vectors of S. Then
  ||(I − P)A||_op^2 = ||AᵀA − AᵀPA||_op ≤ ||AᵀA − SᵀS||_op ≤ ε.
- Therefore, with A = K or V,
  ||ΔK||_op ≤ sqrt(||K − K_r||_F^2/(ℓ − r)), and similarly for V.
This makes the dependence on ℓ and r explicit and tunable.

## 5. Implementation notes
- Data types: Store bases (B, E) in FP16; coefficients (C, D) in FP16 or INT8 with per-tile scales.
- MQA/GQA: Share bases within a group to amortize projection cost; optionally use distinct r per group.
- Memory: Per head, KV coefficients O(T(r + r_v)); bases O(J(r + r_v)d) for J chunks; FD sketch O(ℓ d). With L-length chunks, J ≈ ⌈T/L⌉; pick r, r_v, ℓ, L to keep J(r + r_v)d small relative to T(r + r_v).
- Integration: Implement as a drop-in attention module; tile chunks to fit SRAM; compatible with FlashAttention-style kernels (stream logits and coefficient-weighted value accumulators).

## 6. Algorithms

Static bases (SubSpace-G):
- Calibration: Collect K,V per head on a held-out set, compute top-r (and r_v) right singular vectors for B,E.
- Inference: Project q,k,v once; run coefficient-space attention as in 3.2.

Adaptive bases (SubSpace-A):
- Maintain a continuous FD sketch S per head; on residual exceedance or every L tokens, snapshot top-r (r_v) singular vectors to form a new chunk basis.
- Use the blockwise softmax aggregator to compute outputs across all chunks.

Pseudocode (per head, per token t):
- See 3.4 for the single-pass blockwise softmax with masks.

## 7. Experiments

Models: GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.

Datasets: WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 (long-context).

Setups:
- SubSpace-G: r chosen to retain 90–99% spectral energy on calibration.
- SubSpace-A: ℓ ∈ {2r, 4r}, L ∈ {128, 256}, τ tuned on calibration; r_v ∈ {r, r/2}.
- Ablations: keys-only vs K+V; pre- vs post-RoPE; INT8 coefficients; shared vs per-head bases.
- Baselines: FP16 attention, 8-bit KV cache, sliding window, H2O.
- Metrics: PPL/accuracy delta, throughput, KV memory, spectral decay vs r, FD ε vs observed errors.

Falsifiers:
1) No r ≪ d achieves ≤1% PPL increase.
2) SubSpace-A significantly underperforms SubSpace-G at matched r.
3) Error grows sharply with context length despite chunking, indicating RoPE misalignment.
4) FD ε poorly predicts logit/output error.

Preliminary results (GPT-2 Small): On WikiText-103 (8k context), most heads retain ≥90% spectral energy at r ≈ d/4. SubSpace-G achieves negligible PPL increase at this r. Full, reproducible code will follow.

## 8. Discussion
Coefficient-space attention offers a simple, retraining-free path to reduce KV memory and bandwidth. It is complementary to quantization and token pruning. Likely failure modes include tasks with high intrinsic dimensionality or rapid concept drift; our adaptive variant and tunable ℓ, r, L mitigate but do not eliminate these risks.

## 9. Limitations
- A single global r may be suboptimal; per-head adaptive ranks add complexity.
- SubSpace-A introduces hyperparameters (τ, L, ℓ) and basis-overhead trade-offs.
- Realized throughput gains are hardware- and kernel-dependent; a roofline analysis is needed.

## 10. Conclusion
We formalize and operationalize the Low-Rank KV Cache Hypothesis via online subspace projection. Static and adaptive algorithms compute attention in low-dimensional coefficient spaces with principled error control. If validated, this can materially reduce KV memory and attention bandwidth in LLM decoding with small open-source models and beyond.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
