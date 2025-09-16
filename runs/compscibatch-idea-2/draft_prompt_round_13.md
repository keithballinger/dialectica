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

The draft presents a novel, well-structured, and theoretically grounded method for inference-time KV cache compression. The core ideas—adaptive subspace projection via Frequent Directions and a numerically stable blockwise softmax—are significant contributions. The error analysis directly links algorithmic parameters to performance bounds, and the experimental plan is rigorous and falsification-oriented. The paper is strong and close to publication-ready.

Revisions are minor and aim to improve clarity and precision:
1.  **Softmax Error Bound:** The output error bound in Section 4 uses a softmax sensitivity of `||softmax(x) − softmax(y)||_1 ≤ 2 ||x − y||_∞`. The tighter, standard bound is `||softmax(x) − softmax(y)||_1 ≤ ||x − y||_∞`. Using the tighter bound strengthens the analysis.
2.  **Blockwise Softmax Formula:** The formula for the numerator `N` in Section 3.4 is correct but notationally dense. It can be simplified using vector-matrix products (`w_jᵀ D_j`) to align more directly with the subsequent single-pass algorithm description, enhancing readability.
3.  **Memory Analysis:** The memory trade-off in Section 5 can be stated more directly. The key condition for the method's memory efficiency is that the cost of storing bases, `O(J r d)`, is amortized over a long context, i.e., `J d ≪ T`. Explicitly stating this condition would clarify the method's domain of applicability.
4.  **Abstract and Conclusion:** Minor wording adjustments can improve flow and impact. The abstract is dense, and a slight rephrasing can make the core contributions more immediately accessible.

These revisions are minor polishes on an already excellent paper.

---
### Revised Draft
# Online Subspace Projection for KV Caches: Low-Rank Attention at Inference Time

## Abstract
We propose and test the Low-Rank KV Cache Hypothesis: during autoregressive decoding, an attention head's keys and values concentrate in a low-dimensional subspace, enabling strong, training-free compression. We derive an inference-time factorization that computes attention in a coefficient space defined by head-specific, rank-`r` bases. This reduces per-head KV memory from `O(T d)` to `O(T r)` and attention FLOPs from `O(T d)` to `O(T r)`. We introduce two algorithms: (1) a static method with fixed bases learned from calibration data, and (2) an online adaptive variant using a Continuous Frequent-Directions (FD) sketch with residual-triggered chunking. We provide a numerically stable blockwise softmax for aggregating attention across chunks with distinct value bases, derive operator-norm error bounds from FD guarantees, and propose a falsification-oriented evaluation. Preliminary results on GPT-2 Small show negligible perplexity loss at `r ≪ d`.

## 1. Introduction
The Key-Value (KV) cache is a primary memory and bandwidth bottleneck in autoregressive LLM inference. Existing methods reduce this cost by lowering numerical precision (quantization), pruning tokens (eviction), or retraining the architecture (low-rank attention). We propose an orthogonal, training-free approach: compute attention in low-dimensional, per-head subspaces learned online from the K/V streams.

**Contributions:**
-   **Hypothesis:** Per-head K/V streams are approximately low-rank during decoding, allowing projection to preserve attention outputs at rank `r ≪ d`.
-   **Method:** An inference-time factorization `K ≈ C B` and `V ≈ D E` with orthonormal-row bases (`B`, `E`), enabling computation of logits and outputs in a compressed coefficient space.
-   **Adaptive Algorithm:** A continuous FD sketch maintains a global subspace, while residual-triggered chunking creates local bases, avoiding global recomputation and revisiting past tokens.
-   **Stable Aggregation:** A single-pass, numerically stable blockwise softmax aggregates logits and value contributions across chunks that use different bases.
-   **Guarantees:** Operator-norm error bounds derived from FD’s covariance guarantee, linking sketch size and rank to logit and output error.
-   **Validation:** An ablation-driven, falsification-oriented experimental plan on open-source models with reproducible code.

## 2. Related Work
-   **Training-time low-rank attention** (e.g., Linformer, Nyströmformer) constrains attention during training. Our work applies post-hoc to standard models and adapts the subspace during inference.
-   **KV compression and eviction** (8-bit KV, StreamingLLM, Scissorhands, H2O) reduce precision or token count but continue to operate in `d` dimensions. We compute attention in a compressed coefficient space.
-   **Subspace/SVD analyses** provide offline evidence of low-rank structure. We present an online, numerically stable, inference-time algorithm compatible with standard LLMs and Rotary Position Embeddings (RoPE).

## 3. Method

### 3.1 Notation and Baseline
Per head, with key/value dimension `d` and `T` tokens:
-   Query `q_t ∈ R^d`, keys `K ∈ R^{T×d}`, values `V ∈ R^{T×d}`.
-   Logits `ℓ_i = q_t·k_i / √d`; causal mask `M ∈ {0, −∞}^T`.
-   Weights `α = softmax(ℓ + M)`; output `o_t = αᵀ V`.

### 3.2 Coefficient-Space Attention (Single Basis)
Let `B ∈ R^{r×d}` and `E ∈ R^{r_v×d}` have orthonormal rows (`BBᵀ = I_r`, `EEᵀ = I_{r_v}`). Define:
-   Key coefficients `C = K Bᵀ ∈ R^{T×r}`, value coefficients `D = V Eᵀ ∈ R^{T×r_v}`.
-   Projected query `q̃ = q_t Bᵀ ∈ R^r`.

The projected attention is then computed as:
-   Projected logits: `ℓ̂_i = q̃·c_i / √d = q_tᵀ(BᵀB)k_i/√d`.
-   Projected weights: `α̂ = softmax(ℓ̂ + M)`.
-   Projected output: `ô_t = (α̂ᵀ D) E`.

**Compute/Storage per head:**
-   Storage: `C(T×r)`, `D(T×r_v)`, `B(r×d)`, `E(r_v×d)`.
-   Per-token FLOPs: `O(rd)` for projections, `O(Tr)` for logits, `O(Tr_v + r_v d)` for output.

### 3.3 Adaptive Online Variant with Chunking
We allow chunk-specific bases to adapt to distributional drift, using a single continuous FD sketch to inform basis creation.

**State per head:**
-   FD sketch `S ∈ R^{ℓ×d}` (e.g., `ℓ ≈ 2r`) updated with every new `k_t` and `v_t`.
-   Active chunk `j` with bases (`B_j ∈ R^{r×d}`, `E_j ∈ R^{r_v×d}`), coefficient buffers `C_j`, `D_j`, and current length `L_j`.
-   Hyperparameters: residual thresholds `τ_k`, `τ_v`; max chunk length `L`; sketch size `ℓ`.

**Per-token procedure (post-RoPE `k_t`, `v_t`, `q_t`):**
1.  Compute projection residuals: `res_k = ||(I − B_jᵀB_j) k_t||_2`, `res_v = ||(I − E_jᵀE_j) v_t||_2`.
2.  If `res_k > τ_k` or `res_v > τ_v` or `L_j = L`:
    -   Finalize chunk `j`.
    -   Form new chunk `j+1` with bases `B_{j+1}`, `E_{j+1}` from the top-`r` (and `r_v`) right singular vectors of the current FD sketch `S`.
3.  Append coefficients: `c_t = k_t B_jᵀ` to `C_j`; `d_t = v_t E_jᵀ` to `D_j`; increment `L_j`.
4.  Update the continuous FD sketch `S` with `k_t` and `v_t`.

The FD sketch `S` is continuous across all tokens; chunk bases are snapshots of its top subspace when a trigger condition is met.

### 3.4 Blockwise Softmax Across Chunks
For a new query `q_t` and `J` historical chunks, each with coefficients (`C_j`, `D_j`) and bases (`B_j`, `E_j`):
-   Compute per-chunk logits: `ℓ_j = (q_t B_jᵀ) C_jᵀ / √d + M_j`, where `M_j` is the relevant slice of the causal mask.

To aggregate attention outputs, we define intermediate values for a numerically stable one-pass computation:
-   Global maximum logit: `m = max_{j,i} ℓ_{j,i}`
-   Unnormalized weights: `w_j = exp(ℓ_j − m)`
-   Numerator (unnormalized output): `N = ∑_{j=1}^{J} (w_jᵀ D_j) E_j`
-   Denominator (normalization term): `Z = ∑_{j=1}^{J} sum(w_j)`

The exact aggregated output is `ô_t = N / Z`.

**Single-pass streaming algorithm:**
-   Initialize `m = −∞`, `Z = 0`, `N = 0_d`.
-   For `j` in `1..J`:
    1.  Compute `q̃_j = q_t B_jᵀ` and `ℓ_j = q̃_j C_jᵀ / √d + M_j`.
    2.  `m_j = max(ℓ_j)`.
    3.  If `m_j > m`:
        -   `Z ← Z · exp(m − m_j)`
        -   `N ← N · exp(m − m_j)`
        -   `m ← m_j`
    4.  `w_j = exp(ℓ_j − m)`.
    5.  `Z ← Z + sum(w_j)`.
    6.  `N ← N + (w_jᵀ D_j) E_j`.
-   Return `ô_t = N / Z`.

This algorithm correctly integrates causal masking and avoids revisiting past tokens while maintaining numerical stability.

### 3.5 RoPE Compatibility
Projection is applied post-RoPE. RoPE applies orthogonal, position-dependent rotations. While rotations vary with position, the empirical spectra of post-RoPE K/V streams remain concentrated. When rotational drift causes the subspace to shift, our residual-triggered chunking mechanism naturally creates a new, better-aligned basis.

## 4. Error Guarantees
Let `P_B = BᵀB` and `P_E = EᵀE` be orthogonal projectors. The projection errors are `ΔK = (I − P_B)K` and `ΔV = (I − P_E)V`.

-   **Logit Error:** For any query `q`, the infinity norm of the logit error is bounded:
    `||ℓ − ℓ̂||_∞ = (1/√d) max_i |qᵀΔk_i| ≤ (||q||_2/√d) ||ΔK||_op`.
-   **Softmax Sensitivity:** The L1-norm error of the softmax output is bounded by the logit error:
    `||α − α̂||_1 ≤ ||ℓ − ℓ̂||_∞`.
-   **Output Error:** The final output error is bounded by a combination of key and value projection errors:
    `||o − ô||_2 = ||αᵀV − α̂ᵀV̂||_2 ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂||_1 ||ΔV||_op`
    `≤ (||q||_2/√d) ||ΔK||_op max_i ||v_i||_2 + ||ΔV||_op`.

**Frequent-Directions Linkage:** For any matrix `A ∈ R^{T×d}`, an FD sketch `S ∈ R^{ℓ×d}` with `ℓ > r` guarantees `||AᵀA − SᵀS||_op ≤ ||A − A_r||_F^2/(ℓ − r) = ε`, where `A_r` is the best rank-`r` approximation of `A`. Let `P` be the projector onto the top-`r` right singular subspace of `S`. Then:
`||(I − P)A||_op^2 ≤ ||AᵀA − SᵀS||_op ≤ ε`.
Thus, by setting `A=K` or `A=V`, we can directly bound the operator norms:
`||ΔK||_op ≤ sqrt(||K − K_r||_F^2/(ℓ − r))` and `||ΔV||_op ≤ sqrt(||V − V_{r_v}||_F^2/(ℓ − r_v))`.
This makes the trade-off between sketch size `ℓ` and projection error explicit and tunable.

## 5. Implementation Notes
-   **Data Types:** Bases (`B`, `E`) can be stored in FP16. Coefficients (`C`, `D`) can be stored in FP16 or INT8 with per-tile scales.
-   **MQA/GQA:** Bases can be shared within a query group to amortize projection cost.
-   **Memory:** Per head, total memory is `O(T(r + r_v) + J(r + r_v)d + ℓd)`, comprising coefficients, bases for `J` chunks, and the FD sketch. The method is memory-efficient when the cost of bases is amortized, i.e., when `Jd ≪ T`. This is achieved with long chunks (`L`) and a small number of chunks `J`.
-   **Integration:** The method can be implemented as a drop-in attention module compatible with FlashAttention-style kernels, which stream logits and value accumulators.

## 6. Algorithms
-   **Static (SubSpace-G):** Use calibration data to compute a single, fixed set of bases (`B`, `E`) from the top right singular vectors of the collected K/V matrices. At inference, use the single-basis method from 3.2.
-   **Adaptive (SubSpace-A):** Maintain a continuous FD sketch per head. On a residual threshold trigger or every `L` tokens, snapshot the sketch's top subspace to form a new chunk basis. Use the blockwise softmax (3.4) for aggregation.

## 7. Experiments
**Models:** GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.
**Datasets:** WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 (long-context).

**Setups:**
-   **SubSpace-G:** `r` chosen to retain 90–99% of spectral energy on a calibration set.
-   **SubSpace-A:** `ℓ ∈ {2r, 4r}`, `L ∈ {128, 256}`, `τ` tuned on calibration; `r_v ∈ {r, r/2}`.
-   **Ablations:** Keys-only vs. K+V projection; pre- vs. post-RoPE projection; INT8 coefficients; shared vs. per-head bases.
-   **Baselines:** FP16 attention, 8-bit KV cache, sliding window, H2O.
-   **Metrics:** PPL/accuracy delta, throughput, KV memory, spectral decay vs. `r`, FD error `ε` vs. observed output error.

**Falsifiers:**
1.  No rank `r ≪ d` achieves ≤1% PPL increase, falsifying the core hypothesis.
2.  SubSpace-A significantly underperforms SubSpace-G at matched `r`, indicating that adaptivity provides no benefit.
3.  Error grows sharply with context length despite chunking, indicating a fundamental incompatibility with RoPE.
4.  The FD error bound `ε` proves to be a poor predictor of logit/output error in practice.

**Preliminary Results (GPT-2 Small):** On WikiText-103 (8k context), most heads retain ≥90% spectral energy at `r ≈ d/4`. SubSpace-G achieves a negligible PPL increase at this rank. Full, reproducible code will be released.

## 8. Discussion
Coefficient-space attention offers a principled, retraining-free method to reduce KV memory and bandwidth, complementary to quantization and token pruning. The primary failure mode is high intrinsic dimensionality in K/V streams, where no low-rank approximation is effective. Our adaptive variant is designed to mitigate this risk but does not eliminate it.

## 9. Limitations
-   A single global `r` may be suboptimal; per-head or adaptive rank selection adds complexity.
-   SubSpace-A introduces hyperparameters (`τ`, `L`, `ℓ`) that require tuning.
-   Realized throughput gains depend on hardware-specific kernel implementations; a detailed roofline analysis is needed.

## 10. Conclusion
We formalize and operationalize the Low-Rank KV Cache Hypothesis through online subspace projection. Our static and adaptive algorithms compute attention in a compressed coefficient space with principled error control. If confirmed by our experimental plan, this approach can materially reduce the memory and bandwidth requirements for LLM inference.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
