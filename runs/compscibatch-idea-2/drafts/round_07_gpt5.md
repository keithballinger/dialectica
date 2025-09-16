Major Revisions

Brief critique
- Algorithmic clarity: The chunked frequent-directions (FD) variant conflates per-chunk concatenation with summation and does not specify a numerically stable blockwise softmax across chunks. Without a correct streaming softmax, the method is underspecified and potentially wrong.
- Error analysis: The mapping from FD spectral guarantees to logit, softmax, and output errors is hand-wavy. Provide precise bounds for end-to-end attention error (logit perturbation, normalization error, and output bias) and conditions under which these translate to small perplexity deltas.
- Novelty and positioning: The related work needs sharper contrasts to H2O/Heavy-Hitter Oracle, Scissorhands, StreamingLLM, KV quantization, and training-time low-rank methods (Linformer/Nyströmformer). Explicitly argue why computing attention in coefficient space with online per-head subspaces is distinct and when it is superior/orthogonal.
- Empirical rigor: Preliminary results on a single model are insufficient. You need broader small-model validation, ablations for r and chunk parameters, comparisons to strong baselines (8-bit KV, sliding window, H2O/Scissorhands), and a bandwidth/performance model (roofline) showing when r yields actual throughput gains.
- RoPE handling: Provide a principled rationale for post-RoPE projection and demonstrate failure modes (e.g., very long contexts/high frequencies) rather than only suggesting an ablation.
- Implementation specifics: Precisely define basis orthogonality, coefficient updates, residual thresholds for chunk rollover, basis reuse/merge policy, MQA/GQA compatibility, and memory layout. Include blockwise-softmax pseudocode consistent with FlashAttention-style streaming.
- Claims discipline: Avoid strong performance numbers without detailed setup and released code. Frame early findings as preliminary and reproducible.

Revised Draft
# Low-Rank KV Cache Subspaces in Decoding: A Testable Hypothesis and an Online Compression Algorithm

## Abstract
We hypothesize that, during autoregressive decoding, the cached keys and values within each attention head concentrate in low-dimensional subspaces, enabling strong per-head compression with minimal loss. We formalize this Low-Rank KV Cache Subspace Hypothesis and derive an inference-time factorization that computes attention in coefficient space via head-specific rank-r bases, reducing KV memory from O(T d) to O(T r) and attention FLOPs from O(T d) to O(T r). We present (1) a fixed per-head basis learned from calibration data, and (2) an adaptive online variant using chunked frequent-directions (FD) that avoids global rotations and revisiting tokens. We detail a numerically stable, RoPE-compatible blockwise-softmax implementation; provide complexity and error bounds that connect FD residuals to logit and output errors; and outline a falsification-oriented evaluation on small open-source models. Preliminary measurements on GPT-2 Small suggest favorable spectra and negligible loss at r ≪ d; full code and extended results will be released.

## 1. Introduction
KV cache size and bandwidth dominate autoregressive decoding. Prior work reduces precision (KV quantization), prunes tokens (eviction and heavy-hitter schemes), or changes attention structure during training (Linformer/Nyströmformer). We propose an inference-time dimensionality reduction: per-head streams of keys/values lie predominantly in low-dimensional subspaces, enabling projection into rank-r coefficient spaces with attention computed directly on coefficients, without retraining.

Contributions:
- Hypothesis: Per-head K/V streams during decoding are approximately low-rank; the induced projector preserves logits and outputs with small error at r ≪ d.
- Method: An inference-time factorization K ≈ C B and V ≈ D E with per-head orthonormal-row bases B, E, computing logits and outputs in coefficient space.
- Online algorithm: A chunked FD variant with residual-triggered rollover, stable blockwise softmax across chunks, and no global coefficient rotations.
- Theory-to-practice: Translate FD residual bounds into logit/softmax/output perturbation bounds; provide tunable r/m thresholds linked to empirical operator norms.
- Validation plan: Code-validated experiments on small open-source models (GPT-2, Pythia, TinyLlama), with ablations and comparisons to strong KV compression baselines.

## 2. Related Work
- Training-time low-rank attention: Linformer, Performer/Nyström approximations constrain attention maps during training; they do not compress inference-time KV caches nor compute attention in learned coefficient spaces without retraining.
- KV cache compression: Quantization (e.g., 8-bit KV, vector quantization) reduces precision but not intrinsic dimensionality. Token eviction/reuse (StreamingLLM, Scissorhands, H2O/Heavy-Hitter Oracle) prunes or reweights tokens based on attention, but still operates in the original key/value dimension and does not maintain per-head subspaces or coefficient-space attention.
- SVD/subspace methods: Offline SVD of K/V has been explored in analysis and blockwise approximations, but we are not aware of an inference-time method that (i) maintains per-head low-rank K/V representations online, (ii) projects queries to compute logits and outputs purely in coefficient space, and (iii) integrates with RoPE and blockwise softmax without retraining.

## 3. Method

### 3.1 Notation and baseline attention
Per head, key/value dim d_k = d_v = d, time t with T = t − 1 cached tokens:
- q_t ∈ R^d, K ∈ R^{T×d}, V ∈ R^{T×d}.
- Logits ℓ ∈ R^T: ℓ_i = q_t·k_i / √d.
- Weights α = softmax(ℓ + mask).
- Output o_t = αᵀ V ∈ R^d.

### 3.2 Attention in coefficient space via low-rank projection
Let B ∈ R^{r×d} with orthonormal rows (BBᵀ = I_r). Define C = K Bᵀ ∈ R^{T×r} (key coefficients) and q̃ = q_t Bᵀ ∈ R^r. Then:
- Projected logits: ℓ̂_i = q̃·c_i / √d = q_tᵀ P_B k_i / √d with P_B = BᵀB (rank-r projector).
Similarly, for values let E ∈ R^{r_v×d} with orthonormal rows and D = V Eᵀ ∈ R^{T×r_v}. With α̂ = softmax(ℓ̂ + mask):
- Coefficient output: õ = α̂ᵀ D ∈ R^{r_v}.
- Reconstruct: ô_t = õ E ∈ R^d.

Storage per head: C(T×r), D(T×r_v), bases B(r×d), E(r_v×d). Per-token compute: O(r d + r_v d) for projections and O(T r + T r_v) for logits/aggregation. We keep scaling 1/√d to match baseline logits.

### 3.3 Variants

A. Global per-head bases (SubSpace-G)
- Offline: Collect per-head K and V on calibration tokens; compute top-r right singular vectors (rows orthonormal) to form B and E (e.g., randomized SVD).
- Inference: For each token, compute c_t = k_t Bᵀ, d_t = v_t Eᵀ, and q̃ = q_t Bᵀ; run attention in coefficient space.

B. Online chunked frequent-directions (SubSpace-A)
- Maintain an active chunk j with fixed bases (B_j, E_j) and coefficient buffers (C_j, D_j). Define residual thresholds τ_k, τ_v and max length L.
- For each new token:
  - Compute residual norms r_k = ||k_t − P_{B_j} k_t||_2 and r_v = ||v_t − P_{E_j} v_t||_2.
  - If r_k > τ_k or r_v > τ_v or chunk length = L: close chunk j; start new chunk j+1 by seeding B_{j+1}, E_{j+1} from an FD sketch (see below). Do not revisit old tokens.
  - Else: append c_t = k_t B_jᵀ to C_j and d_t = v_t E_jᵀ to D_j.
- FD updates: In parallel, feed original k_t and v_t into an FD sketch S (size m ≥ 2r) per head to maintain a global subspace estimate. When starting a new chunk, compute (B_{j+1}, E_{j+1}) from the current FD right singular vectors. This avoids per-token global rotations while keeping adaptivity across chunks.
- Blockwise softmax across chunks: For a query q_t,
  - For each existing chunk j, compute q̃_j = q_t B_jᵀ, logits ℓ_j = q̃_j C_jᵀ / √d, and per-chunk max m_j = max(ℓ_j).
  - Maintain global max m = max_j m_j.
  - Compute partition Z = Σ_j Σ_i exp(ℓ_{j,i} − m) using the standard log-sum-exp accumulation across blocks.
  - Compute weighted coefficient sums S = Σ_j exp(m_j − m) (exp(ℓ_j − m_j)ᵀ D_j) ∈ R^{r_v}, where exp(·) is elementwise on ℓ_j.
  - Output ô_t = (S / Z) Ē with Ē = blockwise application of E_j if values use per-chunk E_j; if values use a single E, then Ē = E.
  - This reproduces the exact softmax over concatenated logits given the approximated logits per chunk, without materializing all ℓ.
- Complexity: With J chunks and typical r ≪ d, per token costs O(J r d + Σ_j |C_j| r) for logits plus O(J r_v d + Σ_j |D_j| r_v) for aggregation. For bounded J (e.g., via L and τ), the asymptotic attention term is O(T r + T r_v).

Remarks:
- Using a single value basis E across all chunks simplifies aggregation and avoids per-chunk E_j; empirically, r_v ≤ r often suffices.
- To cap J, periodically merge old chunks by re-projecting their coefficients into a refreshed basis derived from the FD sketch (infrequent, amortized).

### 3.4 RoPE compatibility
RoPE applies position-dependent block-diagonal rotations to q and k. Projecting post-RoPE preserves the exact dot products within the projected subspace: qᵀk ≈ qᵀP_B k with both q,k already rotated. Since rotations are orthogonal, subspace quality is governed by the union of rotated token directions; empirically, low frequencies dominate, keeping effective rank low for moderate contexts. We default to post-RoPE projection and ablate pre- vs. post-RoPE; very long contexts and high-frequency bands may require larger r.

### 3.5 Error guarantees
Let K̂ = C B with projector P_B = BᵀB and V̂ = D E with projector P_E. Define ΔK = K − K̂ and ΔV = V − V̂.

- Logit error: For any token i, |qᵀk_i − qᵀP_B k_i| ≤ ||q||_2 ||(I − P_B) k_i||_2. For the vector of logits ℓ, ||ℓ − ℓ̂||_∞ ≤ ||q||_2 max_i ||(I − P_B) k_i||_2, and ||ℓ − ℓ̂||_2 ≤ ||q||_2 ||ΔK||_op.
- Softmax stability: With δ = ||ℓ − ℓ̂||_∞, the softmax is 1-Lipschitz in ℓ under L1 with bound ||α − α̂||_1 ≤ 2 sinh(δ/2) (tight for two-class); for small δ, ||α − α̂||_1 = O(δ).
- Output error: ||o − ô||_2 ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂||_1 ||ΔV||_op. Since ||α̂||_1 = 1, the dominant terms are controlled by (i) logit perturbation via ΔK and (ii) value projection residual via ΔV.
- FD residuals: For a stream matrix A (rows are tokens), an FD sketch Â of size m satisfies for all unit x: 0 ≤ ||A x||_2^2 − ||Â x||_2^2 ≤ ||A − A_r||_F^2/(m − r). Using A = K and A = V yields computable upper bounds on ||ΔK||_op and ||ΔV||_op via Gram errors, which we estimate empirically during calibration to choose r and m.

These bounds provide a principled, measurable knob: increase r (or m) until operator-norm residuals fall below a target threshold correlated with acceptable PPL deltas.

### 3.6 Implementation details and pseudocode
- Bases: Orthonormal rows via thin SVD; store in FP16. Coefficients C,D in FP16 or INT8 with per-row scaling; maintain per-head per-chunk length.
- RoPE: Apply RoPE, then project to coefficients.
- MQA/GQA: Share B (and optionally E) across groups or all heads in MQA; per-query projection cost remains O(r d).
- Integration: Replace attention kernels with coefficient-space kernels and blockwise softmax accumulation; compatible with FlashAttention-style tiling.
- Warmup: Optionally use full-d attention for first W tokens to stabilize bases.

Pseudocode (per head, per token t):
- For each chunk j:
  - q̃_j = q_t B_jᵀ
  - ℓ_j = q̃_j C_jᵀ / √d
  - m = max(m, max(ℓ_j))
- Initialize Z = 0, S = 0
- For each chunk j:
  - w_j = exp(ℓ_j − m)
  - Z += sum(w_j)
  - S += w_jᵀ D_j
- ô_t = (S / Z) E  // if a single E is used; else apply per-chunk E_j during aggregation

## 4. Experiments

### 4.1 Falsification-oriented plan
Models: GPT-2 Small (124M), GPT-Neo 125M, Pythia-410M, TinyLlama-1.1B.

Datasets/metrics:
- Perplexity: WikiText-103, C4 subsets, The Pile subsets.
- Accuracy: LAMBADA, PIQA, ARC, HellaSwag.
- Long-context: PG19, BookCorpus2 (extended contexts).

Setups:
- SubSpace-G: r chosen by retained energy (90–99%) on 1–5M calibration tokens; report per-head and global r.
- SubSpace-A: m ∈ {2r, 3r}, L ∈ {64,128,256}, τ tuned from calibration residual percentiles; cap J via L and optional merges.
- Ablations: keys-only vs values-only projection; shared vs per-head bases; pre-/post-RoPE; INT8 coefficients; MQA/GQA.
- Baselines: 8-bit KV cache, vector quantization, sliding window, H2O/Heavy-Hitter Oracle, Scissorhands; measure both quality and wall-clock throughput.
- Metrics: ΔPPL/Δaccuracy, memory footprint, attention FLOPs and wall-clock throughput, operator norms and residuals vs r/m, distribution of per-head effective ranks across layers.

Falsifiers:
- No r ≪ d achieves ≤1% PPL increase on small models.
- Online SubSpace-A significantly underperforms SubSpace-G at matched r.
- Post-RoPE projection consistently degrades vs pre-RoPE beyond small deltas, or rank scales poorly with context length.

Reproducibility: PyTorch/HF implementation with FlashAttention-style kernels; scripts to reproduce all tables; release code and configs.

### 4.2 Preliminary measurements (GPT-2 Small)
On WikiText-103 with 8k contexts, singular spectra of per-head K/V suggest 90% energy at r ≈ d/4 for many heads. A fixed-basis variant achieves small PPL deltas at r ≪ d with measurable memory and throughput gains; the online variant matches quality with modest chunk counts. Detailed tables, seeds, and code will be released; these early results should be treated as provisional until full evaluation.

## 5. Discussion
Why low rank? Token redundancy, head specialization, and RoPE’s structured rotations bias K/V streams toward low-dimensional manifolds. The approach complements quantization and eviction and can be combined with FlashAttention to reduce both dimension and bandwidth. Failure modes include distributional drift across long contexts and heads with inherently high intrinsic dimension.

## 6. Limitations and threats to validity
- Rank variability across heads/layers complicates a single r; adaptive per-head r adds bookkeeping.
- Chunk rollover hyperparameters (τ, L) trade accuracy for memory/latency; extreme drift can increase chunk counts.
- Guarantees are spectral; task-level robustness requires empirical validation across datasets and prompts.
- Throughput gains depend on hardware balance; provide roofline analyses and end-to-end benchmarks.

## 7. Conclusion
We propose and test the Low-Rank KV Cache Subspace Hypothesis and introduce global and online coefficient-space attention for inference. The method is simple to integrate, RoPE-compatible, and amenable to rigorous, code-validated evaluation on small open-source models. If the hypothesis holds broadly, it offers a practical, orthogonal lever for KV memory and bandwidth reduction during decoding.
