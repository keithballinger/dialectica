Major Revisions

-   **Computational practicality:** The proposed online basis update is computationally infeasible. An `O(T r^2)` cost per token to rotate all stored coefficients will dominate inference at long contexts. A practical, amortized update scheme (e.g., chunking) with a formal complexity analysis is required.
-   **Novelty:** The contribution is not clearly delineated from prior work in KV cache compression (quantization, eviction, SVD-based methods). The core novelty—maintaining per-head online subspaces and projecting queries to compute attention directly in coefficient space without retraining—must be explicitly claimed and defended.
-   **RoPE Integration:** The interaction with RoPE is hand-waved. The paper must provide a concrete, implementable algorithm for handling RoPE, including an analysis of whether projection should occur before or after the rotation, and acknowledge the impact of RoPE's structure on the subspace's effective rank.
-   **Error Analysis:** The error analysis is purely qualitative. To be credible, it must be grounded in formal guarantees. The manuscript should leverage a framework like frequent-directions to provide spectral error bounds and translate these into bounds on logit and output errors.
-   **Validation:** An unverified "pilot result" is not acceptable. It must be replaced with a rigorous, falsifiable experimental plan that can be executed on small, accessible open-source models to validate the core hypothesis.
-   **Exposition:** The writing is imprecise. Notation must be consistent (e.g., `T` vs. `t-1`, `d_k` vs. `d_v`). The compute and memory trade-offs, including storage for the bases, must be stated explicitly.

Revised Draft
# Low-Rank KV Cache Subspaces in Decoding: A Testable Hypothesis and an Online Compression Algorithm

## Abstract
We hypothesize that, during autoregressive decoding, the cached keys and values of each attention head lie predominantly in a low-dimensional subspace, enabling strong per-head KV compression with negligible loss. We formalize this Low-Rank KV Cache Subspace Hypothesis and derive an inference-time factorization that projects queries, keys, and values into head-specific rank-r bases, reducing KV memory from O(T d) to O(T r) and attention compute from O(T d) to O(T r). We present two practical variants: (1) a fixed, per-head global basis from calibration data; and (2) an online, chunked frequent-directions scheme that maintains adaptive bases without revisiting tokens. We detail RoPE-compatible implementations, complexity/error bounds, and falsification-oriented experiments using small open-source models. The proposal is code-validated, requires small code changes, and yields controllable memory/throughput gains if the hypothesis holds.

## 1. Introduction
KV cache growth dominates LLM decoding memory and bandwidth. Existing approaches compress by quantization or token eviction (sliding windows), or alter attention structure with training-time low-rank approximations. We propose a complementary, inference-time hypothesis: per-head sequences of keys and values concentrate in a low-rank subspace, permitting projection to rank r ≪ d with minimal accuracy loss. Unlike prior KV quantization/eviction, we reduce intrinsic dimensionality and compute by operating attention directly in coefficient space at inference, without retraining.

Contributions:
- Hypothesis: per-head cached K/V matrices concentrate in a low-rank subspace during decoding.
- Method: per-head factorization K ≈ C B, V ≈ D E with online updates; attention operates on coefficients, reducing memory and compute.
- Online algorithm: a chunked frequent-directions (FD) approach avoiding O(T r^2) global rotations.
- Error/complexity analysis: FD-based spectral guarantees translated to logit/output errors; practical settings for r and chunking.
- Falsification plan: small-model experiments probing ranks, RoPE interaction, and global vs. online bases.

## 2. Related Work
- Training-time low-rank attention (e.g., Linformer, Nyströmformer) constrains attention maps during training, not inference caches.
- KV cache compression via quantization and vector quantization reduces precision, not dimensionality; eviction methods (e.g., sliding windows, streaming) prune tokens but do not exploit subspace structure.
- SVD/blockwise approximations to attention exist, but we are not aware of an inference-time method that (i) maintains per-head low-rank K/V subspaces online, (ii) projects queries to compute logits in coefficient space, and (iii) integrates with RoPE without retraining.

## 3. Method

### 3.1 Notation and baseline attention
Per head with key/value dims d_k, d_v at time t:
- Query q_t ∈ R^{d_k}, past keys K ∈ R^{T×d_k}, values V ∈ R^{T×d_v}, T = t−1.
- Logits ℓ = q_t Kᵀ / √d_k; α = softmax(ℓ + mask); output o_t = α V.

### 3.2 Low-rank factorization and attention in coefficient space
Assume per-head K ≈ C B, with B ∈ R^{r×d_k} orthonormal rows and C ∈ R^{T×r}. Similarly V ≈ D E with E ∈ R^{r_v×d_v}, D ∈ R^{T×r_v}.
- Project query: q̃ = q_t Bᵀ ∈ R^{r}.
- Logits: ℓ ≈ q̃ Cᵀ / √d_k.
- Values: õ = α D ∈ R^{r_v}, then o_t ≈ õ E.

Storage per head/layer: C (T×r) + B (r×d_k) and D (T×r_v) + E (r_v×d_v). Compute per token: O(r d_k + r_v d_v) for projections plus O(T r) and O(T r_v) for score/aggregation.

Scaling: We retain division by √d_k to match baseline logits since q_t·k ≈ q̃·c when k lies in span(B).

### 3.3 Variants

A. Global per-head bases (SubSpace-G)
- Offline: Collect K/V on a calibration set; compute top-r PCs per head via randomized SVD.
- Inference: For each token, compute c_t = k_t Bᵀ, d_t = v_t Eᵀ, and q̃ = q_t Bᵀ. Use coefficient attention as above.
- Pros: Simple, fast; Cons: May be suboptimal out-of-domain.

B. Online chunked frequent-directions (SubSpace-A)
Challenge: Recomputing a global basis and rotating all coefficients costs O(T r^2), impractical for long T.

We propose chunked FD:
- Maintain a current chunk j with its FD sketch S_j ∈ R^{m×d} (m = 2r) for keys and values separately; derive bases B_j, E_j via thin SVD when starting the chunk.
- For each new token: compute c_t = k_t B_jᵀ, d_t = v_t E_jᵀ; append to C_j, D_j; update S_j with k_t, v_t (O(m d)).
- When the FD residual grows (or after L tokens), close the chunk and start a new chunk j+1 with fresh bases B_{j+1}, E_{j+1}. Do not revisit earlier chunks.
- Attention at time t: concatenate coefficients across chunks: ℓ = [q_t B_1ᵀ C_1ᵀ, …, q_t B_Jᵀ C_Jᵀ] / √d_k; α = softmax(ℓ + mask); õ = α [D_1; …; D_J]; o_t = õ [E_1; …; E_J]-aware, implemented by summing per-chunk contributions õ_j E_j.

Complexity:
- Per token: O(J r d_k + T r/J) in practice remains O(T r) as J is small (e.g., ≤ 4–8).
- Memory: sum_j |chunk_j|·r plus small per-chunk bases. Choose L to keep J bounded for target context length.

Notes:
- Optional periodic “merge”: reproject the smallest old chunk into the newest basis to cap J; amortized cost O(|chunk| r d_k) executed infrequently.

### 3.4 RoPE compatibility
- Default: Project after applying RoPE to K/V/Q (i.e., bases capture rotated features). This is plug-in for standard causal attention.
- Alternative structured option: treat RoPE as block 2×2 rotations and maintain bases over paired dimensions; empirically, post-RoPE projection suffices for moderate contexts. We include an ablation “pre- vs post-RoPE.”

### 3.5 Error guarantees (sketch)
Frequent-directions guarantees for any matrix A ∈ R^{T×d} with sketch Â of rank r:
- Covariance approximation: for all unit x, |xᵀ(AᵀA − ÂᵀÂ)x| ≤ ε_FD, where ε_FD ≤ ||A − A_r||_F^2/(m−r).
Translate to logits:
- Let k_t rows form K and K̂ = C B. For any query q, the logit error satisfies
  |qᵀk_i − q̃ᵀc_i| ≤ ||q||·||K − K̂||_{op}.
Since softmax is 1-Lipschitz in ℓ_∞ on additive shifts and smooth elsewhere, small operator-norm error yields small changes in attention weights; value error is bounded by ||α||_1·||V − V̂|| plus logit-induced weight perturbations. We report empirical spectral decay and FD residuals to connect r to expected PPL deltas.

### 3.6 Implementation details
- Integration: Minimal changes to attention: replace K/V appends with coefficient appends; project q each step; compute logits and aggregation in coefficient space; reconstruct with E (values) only at the end.
- Precision: Store C, D in fp16 or int8 (per-channel scaled). Bases B, E in fp16/bf16. Coexists with standard KV quantization for additional savings.
- Architectures: Works with MHA, MQA/GQA (apply per key/value head). Causal masking unchanged.
- Overheads: For short contexts (T small), overhead of projections may outweigh savings; enable after a configurable warmup length.

## 4. Experiments (Falsification Plan)

Models: GPT-2 Small (124M), GPT-Neo 125M, Pythia-410M, TinyLlama-1.1B.

Datasets:
- Perplexity: WikiText-103, C4 (validation splits), The Pile (subset).
- Accuracy: LAMBADA, PIQA, ARC-e/c, HellaSwag.
- Long-context stress: PG19, BookCorpus2 subsets at 8k–32k tokens.

Protocols:
- SubSpace-G: Per-head r selected by energy thresholds (90/95/98/99%) on 1–5M calibration tokens; evaluate perplexity/accuracy vs. baseline across ranks and thresholds.
- SubSpace-A: FD with m = 2r; chunk length L ∈ {64, 128, 256}; max chunks J_max ∈ {4, 8}. Compare to SubSpace-G at matched r.
- Ablations: keys-only vs values-only compression; pre- vs post-RoPE; uniform r vs per-head r; coefficient int8 quantization; MQA/GQA settings.
- Baselines: 8-bit KV quantization, sliding-window eviction at matched memory; any available SVD/blockwise KV compression.

Metrics:
- Quality: PPL deltas; zero-shot accuracy deltas.
- Efficiency: KV memory vs. context length; attention FLOPs; wall-clock throughput at 8k/32k tokens.
- Diagnostics: Singular value spectra of K/V per head; FD residuals; per-head sensitivity (heatmaps vs. layer/head).

Falsifiers:
- No r ≤ d_k/4 achieving ≤1% PPL increase across tasks refutes the hypothesis for practical ranks.
- SubSpace-A failing to match or exceed SubSpace-G indicates limited adaptivity.
- Post-RoPE projection requiring r ≈ d_k to retain accuracy undermines practicality under RoPE.

Reproducibility:
- Implement as a drop-in attention module in PyTorch/HF; seed configs and scripts for GPT-2 Small; public code and logs recommended.

## 5. Discussion
Why low rank may hold: redundancy across tokens, head specialization, and RoPE-induced structure concentrate energy in a few directions. Our approach differs from quantization/eviction by reducing intrinsic dimensionality and compute. It is complementary to precision reduction and windowing, and compatible with paged attention and FlashAttention kernels once coefficient space operations are integrated.

## 6. Limitations
- Some heads/layers may require larger r; per-head r selection adds complexity.
- Chunking introduces a small J factor in compute; pathological drift could increase J.
- Very long contexts with RoPE may inflate effective rank; structured RoPE-aware bases are future work.
- For short sequences, projection overhead can outweigh gains.
- Error guarantees are spectral; task-level robustness still empirical.

## 7. Conclusion
We propose a falsifiable hypothesis that per-head KV caches occupy low-rank subspaces and a practical inference-time factorization—global and online-chunked—that compresses KV memory and reduces attention compute. The method is readily testable on small open-source models; if validated, it offers significant efficiency gains with minimal code changes.
