Minor Revisions
Strong, well-motivated idea with clear algorithms and theory; needs minor clarifications on calibration/γB stability, FD update overhead in compute-bound regimes, and explicit handling of worst-case high-rank heads.

# Online Subspace Projection for KV Caches: Training-Free Low-Rank Attention at Inference Time
> Note: This paper proposes doing attention computations in a smaller “coefficient space” learned on the fly, to cut both memory (KV cache) and compute during inference, without retraining the model.

## Abstract
We propose a training-free method to reduce both compute and memory in autoregressive LLM inference by computing attention in per-head low-dimensional subspaces. We formalize the Low-Rank KV Cache Hypothesis—keys and values concentrate in low-dimensional subspaces during decoding—and operationalize it with two inference-time algorithms: (1) a static subspace learned from calibration data and (2) an adaptive subspace maintained online via Frequent Directions (FD) sketches with residual-triggered chunking. We introduce a numerically stable blockwise softmax that aggregates attention across chunks with distinct bases, derive operator-norm error bounds that link attention error to sketch accuracy, and provide a roofline-style FLOPs/bytes analysis. Experiments on GPT-2 Small, Pythia-410M, and TinyLlama show that substantial KV memory and FLOPs reductions are achievable at minimal accuracy loss; we also supply falsification tests that directly probe the low-rank hypothesis. Code and configs are released for full reproducibility.
> Note: Key idea: during decoding, each attention head’s keys (K) and values (V) often lie near a low-dimensional subspace. Projecting K/V to that subspace (per head) lets you compute attention using fewer numbers (coefficients), cutting bandwidth and FLOPs. Two modes: a fixed subspace learned from a short calibration run, or an adaptive subspace updated online using Frequent Directions (a streaming PCA-like sketch). A special “blockwise softmax” ensures numeric stability when combining chunks with different subspaces. They give theory (error bounds tied to sketch quality), systems analysis (FLOPs/bytes), and experiments.

## 1. Introduction
The KV cache dominates memory traffic and latency in autoregressive decoding. Existing compression methods quantize or evict tokens in the original d-dimensional space or require retraining to impose low-rank structure. We instead compute attention in an online-learned coefficient space: keys and values are projected to rank‑r and rank‑r_v subspaces, and attention is performed on coefficients, reducing per-token FLOPs and KV memory.
> Note: KV cache = stored past keys (K) and values (V) for attention; reading it dominates bandwidth. Here, “d” is the per-head hidden size. Instead of quantizing in the original space or retraining, they project K/V into a low-rank subspace of dimension r (for keys) and r_v (for values), then do attention on these lower-dim coefficients.

Low-Rank KV Cache Hypothesis (formal): For a given model layer and head h and for any prefix length T, there exist subspaces U_K, U_V ⊆ R^d with dim(U_K)=r, dim(U_V)=r_v and orthogonal projectors P_K, P_V such that the singular values of K_h(T) and V_h(T) exhibit rapid decay and
- ||K_h(T)(I−P_K)||_op ≤ ε_K(T, r), ||V_h(T)(I−P_V)||_op ≤ ε_V(T, r_v),
with ε_K, ε_V small for r, r_v ≪ d. Under RoPE, this holds locally within chunks of tokens.
> Note: Definitions: d = head dimension; T = number of past tokens; U_K, U_V = low-dimensional subspaces for keys/values; P_K, P_V = orthogonal projectors onto those subspaces; I = identity. ||·||_op = operator norm (largest singular value). ε_K, ε_V quantify how much of K/V lies outside the subspace (smaller is better). “Rapid singular value decay” means most energy is in a few directions. RoPE (rotary positional embedding) can disturb global low-rankness, so the claim is it holds within shorter “chunks.”

Contributions:
- Online coefficient-space attention: per-head subspace projection reduces per-token attention from O(Td) to O(T(r+r_v)) FLOPs and per-token KV memory from O(Td) to O(T(r+r_v)).
> Note: Complexity: baseline attention per token scales with T×d. After projection, it scales with T×(r + r_v), which is cheaper if r, r_v ≪ d. KV memory stored becomes coefficients instead of full K/V.

- Adaptive subspaces via FD: a streaming sketch yields chunk-specific bases without revisiting history; residual triggers detect subspace drift with O(1) overhead per token using cached coefficients.
> Note: Frequent Directions (FD) is a deterministic streaming algorithm approximating top singular vectors. “Residual triggers” use how much energy is lost by the current basis to decide when to start a new chunk (basis).

- Blockwise softmax across chunks: a single-pass, numerically stable aggregator combining logits and value contributions computed in distinct bases, with clear equivalence conditions.
> Note: When different chunks use different bases, they compute per-chunk attention logits and value contributions, then combine them in a numerically stable way (like log-sum-exp across blocks).

- Theory: operator-norm error bounds for logits and outputs, tying overall error to FD sketch accuracy; stability guarantees for the aggregator.
> Note: They bound the error in attention scores (logits) and outputs in terms of projection errors (operator norms) and FD sketch quality.

- Systems analysis: per-token FLOPs/bytes roofline model, quantifying FD overhead and bandwidth savings.
> Note: “Roofline model” = a simple performance model balancing compute vs memory bandwidth. They quantify how many FLOPs and bytes are saved and what extra overhead FD adds.

- Validation: a falsification-oriented experimental design with direct spectral diagnostics on small open-source models and reproducible code.
> Note: They test the low-rank hypothesis directly (spectra of K/V), and also try to disprove it (falsifiers), making the evaluation more rigorous.

## 2. Related Work
- KV cache compression: quantization (e.g., 8-bit KV, KVQuant), eviction/streaming (StreamingLLM, H2O), and token selection (ShadowKV, SnapKV, Scissorhands, Quest). These operate in the original space or prune tokens; we compute attention in a compressed coefficient space and reconstruct outputs.
> Note: Prior methods compress or prune in the original d-dimensional space; here, the novelty is computing attention itself in a lower-dimensional coefficient space and reconstructing the final output.

- Low-rank/approximate attention: training-time constraints (Linformer, Nyströmformer) or structural sparsity. We remain post-hoc and streaming at inference.
> Note: Linformer/Nyström enforce low-rank during training. This approach is training-free and works online during decoding.

- Subspace/spectral analyses for LMs: prior reports of low-rank structure motivate but do not provide online, RoPE-compatible algorithms.
> Note: Prior analyses suggest low-rankness but don’t give practical online methods compatible with RoPE.

- Sketching: Frequent Directions provides deterministic guarantees for streaming PCA; we leverage FD to maintain subspaces with error control.
> Note: FD gives provable, streaming approximations to principal components (top singular vectors), which they use to adapt the subspace.

We evaluate against the strongest KV-compression/pruning baselines (ShadowKV, SnapKV, Scissorhands, Quest), quantization, and eviction methods.
> Note: Baselines include pruning, quantization, and eviction methods to compare quality and speed.

## 3. Preliminaries and Baseline
Per head with hidden size d and prefix length T:
> Note: We analyze one attention head. d = head dimension; T = number of cached tokens so far (context length).

- q_t ∈ R^d, K ∈ R^{T×d}, V ∈ R^{T×d}; logits ℓ_i = q_t·k_i / √d; α = softmax(ℓ + mask); output o_t = αᵀ V.
> Note: q_t = current query vector at time t; k_i, v_i are the i-th key and value (rows of K, V). “·” is dot product. √d is the standard attention scaling. ℓ ∈ R^T are logits; mask applies causal or attention masks. softmax(x)_i = exp(x_i)/sum_j exp(x_j). α ∈ R^T are attention weights. o_t ∈ R^d is output: weighted sum of values, αᵀV = sum_i α_i v_i.

- Baseline per-token cost: O(Td) FLOPs and O(Td) bytes (for reading K/V) dominate; optimized kernels (e.g., FlashAttention2) approach bandwidth limits.
> Note: Computing q·K^T and reading K/V costs scale with T×d; in practice, memory bandwidth is the bottleneck even with fast kernels.

## 4. Coefficient-Space Attention (Single Basis)
Let B ∈ R^{r×d}, E ∈ R^{r_v×d} with orthonormal rows (B Bᵀ = I_r, E Eᵀ = I_{r_v}). Define projectors P_B = Bᵀ B, P_E = Eᵀ E.
> Note: B and E are row-orthonormal matrices defining r- and r_v-dimensional subspaces (per head). I_r is the r×r identity. P_B and P_E are d×d orthogonal projectors onto the subspaces spanned by B and E.

- Key coefficients C = K Bᵀ ∈ R^{T×r}, value coefficients D = V Eᵀ ∈ R^{T×r_v}, projected query q̃ = q_t Bᵀ ∈ R^r.
> Note: Project K and V into lower dimensions to get coefficient matrices C (keys) and D (values). q̃ is the low-dim version of the query.

- Projected logits: ℓ̂_i = γ_B q̃·c_i / √d = γ_B q_tᵀ P_B k_i / √d.
> Note: ℓ̂_i are approximate logits using projected keys. γ_B is a scalar calibration factor. c_i is the i-th row of C. q_tᵀP_Bk_i = dot product after projecting k_i onto the B-subspace.

- Calibration for γ_B:
  - Default: γ_B^0 = √(trace(P_B)/d) = √(r/d).
  - Calibrated: choose γ_B to minimize E[(ℓ − ℓ̂(γ_B))^2] over a held-out calibration set (per head/layer), where ℓ are baseline logits and ℓ̂(γ_B) are projected logits. Empirically γ_B ≈ (1.05–1.20)·γ_B^0.
> Note: γ_B rescales projected logits to better match full-space logits. trace(P_B)=r since P_B is rank r. The calibrated γ_B is fit via mean-squared error on logits from a small dataset.

- Weights: α̂ = softmax(ℓ̂ + mask).
> Note: α̂ are attention weights computed from the projected logits.

- Output: ô_t = (α̂ᵀ D) E ≈ α̂ᵀ (V P_E).
> Note: Compute weighted sum in coefficient space, then map back to d-dim via E. Equivalently, use projected values VP_E.

Complexity per head per token:
- FLOPs: O(rd) (query projection) + O(Tr) (logits) + O(Tr_v + r_v d) (output) vs O(Td) baseline.
> Note: Costs: projecting q_t costs r×d; computing logits uses T×r; computing output uses T×r_v plus reconstructing to d via r_v×d.

- Bytes moved: read C and D instead of K and V → O(T(r+r_v)) vs O(Td).
> Note: Bandwidth scales with number of coefficients read: T×(r+r_v) instead of T×d.

When T ≫ d and r, r_v ≪ d, the dominant term drops from O(Td) to O(T(r+r_v)).
> Note: In long contexts (large T), the linear-in-T term dominates, so reductions in r+r_v directly reduce runtime.

## 5. Adaptive Online Variant with Residual-Triggered Chunking
We adapt bases to distributional drift and RoPE phase via chunking.
> Note: Because token distributions and RoPE phases change, a fixed basis can degrade. They divide the sequence into chunks, each with its own basis, updated online.

State per head:
- FD sketches S_K ∈ R^{ℓ_k×d}, S_V ∈ R^{ℓ_v×d} (ℓ_k ≥ r+Δ, ℓ_v ≥ r_v+Δ).
> Note: S_K, S_V are Frequent Directions sketches with sketch sizes ℓ_k, ℓ_v; choose ℓ slightly larger than the target ranks (Δ is slack).

- Active chunk j with bases (B_j, E_j), buffers (C_j ∈ R^{L_j×r}, D_j ∈ R^{L_j×r_v}).
> Note: For current chunk j: B_j/E_j are bases; C_j/D_j store coefficients for L_j tokens in this chunk.

- Hyperparameters: thresholds τ_k, τ_v, max chunk length L, sketch sizes ℓ_k, ℓ_v.
> Note: τ_k, τ_v are residual thresholds to trigger new chunks; L caps chunk size; ℓ_k, ℓ_v control sketch accuracy/overhead.

Per token (post-RoPE q_t, k_t, v_t):
> Note: All operations occur after applying RoPE to queries/keys/values, which aligns the subspace with positional rotations.

1. Compute coefficients: c_t = B_j k_t, d_t = E_j v_t; append to C_j, D_j. This is required regardless.
> Note: Always compute and store coefficients for current bases; this both serves attention and allows cheap residual computation.

2. Residuals (free using cached coefficients):
   - res_k^2 = ||k_t||_2^2 − ||c_t||_2^2, res_v^2 = ||v_t||_2^2 − ||d_t||_2^2.
   - Relative residuals: ρ_k = res_k / ||k_t||_2, ρ_v = res_v / ||v_t||_2.
> Note: ||·||_2 is Euclidean norm. res_k is energy of k_t outside the current key subspace (by Pythagoras for orthogonal projection). ρ_k, ρ_v are fraction of energy lost; they indicate basis mismatch.

3. Trigger: if ρ_k > τ_k or ρ_v > τ_v or L_j = L, finalize chunk j. Snapshot new bases (B_{j+1}, E_{j+1}) from SVD(S_K), SVD(S_V) (top‑r, top‑r_v). Reset (C_{j+1}, D_{j+1}, L_{j+1}←0).
> Note: Start a new chunk if residuals exceed thresholds or chunk is too long. New bases are the top singular vectors from the current FD sketches. SVD = singular value decomposition.

4. FD update: update S_K, S_V with k_t, v_t (per-update cost O(dℓ_k)+O(dℓ_v)).
> Note: Each token updates the sketches; cost scales with head dimension d and sketch sizes ℓ.

Memory per head: O(T(r+r_v)) for coefficients + O(J(r+r_v)d) for bases + O((ℓ_k+ℓ_v)d) for sketches; choose τ, L to keep J small.
> Note: T = total tokens; J = number of chunks. Bases are relatively large (d×r), so too many chunks increases memory; thresholds control J.

## 6. Blockwise Softmax Across Chunks
Let chunks j = 1..J with (B_j, E_j, C_j, D_j). For query q_t:
> Note: We need to combine attention across all previous tokens even though they are stored by chunk with different bases.

- q̃_j = B_j q_t ∈ R^r; per-chunk logits vector ℓ_j = (γ_B/√d) q̃_j C_jᵀ + mask_j ∈ R^{L_j}.
> Note: For each chunk j, project the query to that chunk’s basis (q̃_j) and compute logits for tokens in chunk j. mask_j applies causal masks for those positions.

Numerically stable one-pass aggregation:
- Initialize m = −∞, Z = 0, N = 0_d.
> Note: m tracks the running max logit (for stability), Z is the softmax normalizer (denominator), N ∈ R^d accumulates the numerator (weighted value sum).

- For j = 1..J:
  - m_j = max(ℓ_j); if m_j > m: scale Z ← Z·exp(m − m_j); N ← N·exp(m − m_j); m ← m_j.
  - w_j = exp(ℓ_j − m) ∈ R^{L_j}; Z ← Z + sum(w_j); N ← N + (w_jᵀ D_j) E_j.
> Note: Classic log-sum-exp trick per block: rescale partial sums when a larger max appears to avoid overflow. w_j are per-token exponentiated, shifted logits. (w_jᵀ D_j) gives the weighted sum of value coefficients for chunk j; multiply by E_j to map back to d-dim and add to N.

- Return ô_t = N / Z.
> Note: Final output is the normalized weighted sum across all chunks; exactly equivalent to a global softmax if logits/values were exact.

Equivalence and approximation:
- If, for all j, ℓ_j equal the exact logits for tokens in chunk j and D_j E_j = V_j (i.e., values are not projected), then ô_t equals the standard softmax output softmax(ℓ)ᵀ V.
> Note: This establishes correctness of the aggregator: with exact pieces, the blockwise process reproduces standard attention.

- In our method, ℓ_j and D_j E_j are approximations from projected keys/values; the aggregator computes the exact softmax over approximated logits and applies it to approximated values. The global-max rescaling ensures numerical stability.
> Note: Practically, it’s exact w.r.t. the approximations used; numeric stability is preserved by the running max.

## 7. RoPE Compatibility
RoPE applies per-position complex rotations in 2D frequency planes, which globally scramble spectra but are locally coherent. We:
> Note: RoPE rotates pairs of dimensions per position; across long spans, the basis orientation changes, but over short spans it’s coherent.

- Apply projection post-RoPE; chunking limits phase drift within each chunk.
> Note: Project after RoPE so the basis sees the actual rotated vectors; chunks keep rotations from drifting too far.

- Basis anchoring without lookahead (streaming default): When starting chunk j at position t0, freeze (B_j, E_j) using the current FD state, which has integrated a trailing window of recent tokens. Optionally, warm-start FD with a trailing window of size W (e.g., W=256) by maintaining ring buffers; this aligns bases with the recent RoPE phase.
> Note: The FD sketch effectively captures recent directions; using a ring buffer to warm-start can better match the current RoPE phase without peeking ahead.

- Optional small-lookahead variant (offline or added-latency mode): allow Lh tokens of lookahead (e.g., Lh = W/2) to center the anchoring window around the chunk. We report both modes; streaming results use no lookahead.
> Note: With lookahead, the basis can reflect both past and near-future tokens, potentially improving alignment (at the cost of latency).

- Diagnostics: track ρ_k, ρ_v vs token position; triggers correlate with accumulated RoPE phase.
> Note: Rising residuals often indicate RoPE-induced misalignment; this validates the chunking triggers.

- Optional rotation-aware basis: learn bases in the complex view per frequency pair and transport them within the chunk via known phase shifts; we ablate this variant.
> Note: More sophisticated approach: model RoPE analytically in the basis; they test but don’t rely on it.

## 8. Error Guarantees
Let ΔK = K(I−P_B), ΔV = V(I−P_E). For a single basis:
> Note: ΔK and ΔV are the parts of K and V that lie outside the chosen subspaces (projection residuals). P_B, P_E are projectors; I is identity.

- Logit error: for any q, ||ℓ − ℓ̂||_∞ ≤ (γ_B/√d) ||ΔK||_op ||q||_2.
> Note: ℓ are true logits; ℓ̂ are projected logits. ||·||_∞ is max absolute entry; ||·||_op is operator norm (largest singular value); ||q||_2 is Euclidean norm. Interpretation: larger key residuals (ΔK) or query norms worsen logit error; γ_B/√d scales it.

- Softmax sensitivity: ||α − α̂||_1 ≤ 2 tanh(||ℓ − ℓ̂||_∞/2) (tight in ℓ∞; also ≤ ||ℓ − ℓ̂||_∞).
> Note: α, α̂ are true vs projected attention weights. ||·||_1 is sum of absolute values. tanh(·) is hyperbolic tangent; this bounds how logit perturbations affect probabilities.

- Output error:
  ||o − ô||_2 ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂ᵀ ΔV||_2 ≤ 2 tanh(||ℓ − ℓ̂||_∞/2) max_i ||v_i||_2 + ||ΔV||_op.
> Note: o, ô are true vs projected outputs. First term: weight error times largest value norm. Second term: value projection residual under α̂; bounded by ||ΔV||_op. This cleanly separates key vs value projection effects.

FD linkage (deterministic): For A ∈ R^{T×d}, an FD sketch S ∈ R^{ℓ×d} with ℓ > r ensures
||AᵀA − SᵀS||_op ≤ ||A − A_r||_F^2/(ℓ − r) = ε.
Let P be the projector onto the top‑r right singular space of S. Then
||A(I−P)||_op ≤ √ε + η,
where η is a subspace perturbation term controlled by the Davis–Kahan theorem: η ≲ 2ε/γ, with γ the spectral gap around the r-th eigenvalue of AᵀA. Intuitively, η measures how much the sketch-derived subspace deviates from the true one due to sketch noise and small gaps. Applying to K and V yields operator-norm residual bounds that drive attention error.
> Note: A_r is the best rank‑r approximation of A (via SVD). ||·||_F is Frobenius norm. ε measures sketch error. P is the projector onto the sketch’s top‑r subspace. Davis–Kahan bounds subspace misalignment; γ is the eigenvalue gap (larger is better). This links sketch size ℓ and rank r to projection residuals used in the earlier bounds.

Multi-chunk extension: With chunk-specific projectors P_{B_j}, P_{E_j} and concatenated K = [K_1;…;K_J],
- ||ℓ − ℓ̂||_∞ ≤ (γ_B/√d) ||q||_2 max_j ||K_j(I−P_{B_j})||_op.
- ||o − ô||_2 ≤ 2 tanh( (γ_B||q||_2/2√d) max_j ||K_j(I−P_{B_j})||_op ) max_i ||v_i||_2 + max_j ||V_j(I−P_{E_j})||_op.
> Note: In multi-chunk settings, the worst (largest) per-chunk residual dominates the bound. K_j, V_j are keys/values within chunk j.

Interpretation: worst-case key projection error controls the weights; worst-case value projection error controls the reconstructed output. Sketch sizes (ℓ_k, ℓ_v), ranks (r, r_v), and chunking govern these errors.
> Note: Practical tuning: increase ℓ or r to reduce residuals, or shorten chunks when RoPE drift is high.

## 9. Systems and Roofline Analysis
- FLOPs per token per head: baseline ≈ c1·Td; ours ≈ c2·T(r+r_v) + c3·(r+r_v)d + c4·d(ℓ_k+ℓ_v) (FD updates). For T ≫ d and modest ℓ ≈ 2r, the T(r+r_v) term dominates.
> Note: c1–c4 are constants reflecting kernel efficiency. The extra terms include query/output projections and FD updates. In long contexts, the T-dependent term dominates runtime.

- Bytes per token per head: baseline reads ≈ b1·Td; ours reads ≈ b2·T(r+r_v) + b3·J(r+r_v)d (amortized basis reads).
> Note: b1–b3 are constants. J(r+r_v)d reflects reading per-chunk bases; amortized over many tokens.

- Regimes:
  - Bandwidth-bound (typical on HBM GPUs): reductions in bytes/token from O(d) to O(r+r_v) directly improve tokens/sec; FD overhead is hidden if compute is underutilized.
  - Compute-bound (small T or very small d): ensure r+r_v is small enough that c3·(r+r_v)d + c4·d(ℓ_k+ℓ_v) does not dominate; subsampling FD updates helps.
> Note: On memory-bound hardware (usual), bandwidth savings translate to speedups. On compute-bound cases (short contexts or tiny heads), keep r small and reduce FD update frequency.

- Kernels: implement coefficient attention and blockwise softmax with fused reads/writes; reuse FlashAttention2 tiling. FD updates can be batched or subsampled to reduce overhead.
> Note: Efficient kernels minimize memory traffic by fusing operations; FD can be updated less often or in batches to limit compute overhead.

## 10. Algorithms
- Static Subspace (training-free): From calibration runs, compute per-head bases B, E (top right singular vectors of K, V). Use Section 4 at inference. Optionally per-layer/head ranks via energy thresholds.
> Note: “Calibration” = run the model on a small dataset, stack K/V per head, compute SVD, take top‑r vectors. Energy threshold = choose r so cumulative singular values reach e.g., 95–99%.

- Adaptive Subspace: Maintain S_K, S_V via FD; when residual thresholds fire or every L tokens, snapshot B_j, E_j; compute coefficients and aggregate via Section 6.
> Note: Online version: update sketches each token, switch bases when residuals indicate drift or chunk cap is reached, and combine chunks with blockwise softmax.

Hyperparameters and calibration:
- r, r_v: per-head ranks chosen to retain 90–99% energy on calibration traces; adaptive per-head selection ablated.
> Note: “Energy” = sum of squared singular values. Different heads can have different optimal ranks.

- ℓ_k, ℓ_v: sketch sizes in {2r, 4r}.
> Note: Larger ℓ gives better sketches (smaller ε) but costs more compute/memory.

- τ_k, τ_v: set on a held-out set to target relative residual percentiles, e.g., choose τ_k, τ_v such that P(ρ_k ≤ τ_k) ≥ 95% and P(ρ_v ≤ τ_v) ≥ 95% at chosen ranks (typical targets τ ≈ 0.1–0.2).
> Note: Thresholds are tuned so residuals rarely exceed them; ρ is the relative residual defined earlier.

- L ∈ {128, 256, 512} (trade off subspace freshness vs basis overhead).
> Note: Shorter L = fresher bases but more basis overhead; longer L reduces overhead but risks misalignment (especially with RoPE).

- γ_B: default √(r/d) or calibrated by MSE minimization on logits (Section 4).
> Note: Use default scaling or fit γ_B on a small calibration set to better match logits.

- Quantization: coefficients in INT8 with per-tile scales; bases in FP16 by default; optional FP8/INT8 bases ablated.
> Note: Quantizing coefficients further reduces bandwidth; bases often kept higher precision to preserve orthogonality.

## 11. Experiments
Models: GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.
> Note: Tested across small to ~1B-parameter open models to allow full instrumentation.

Datasets: WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 and LongBench (long-context).
> Note: Perplexity (PPL) measures language modeling quality; classification tasks test accuracy; PG19/LongBench stress long contexts.

Baselines: FP16 attention (FlashAttention2), 8-bit KV, sliding window, H2O, StreamingLLM, ShadowKV, SnapKV, Scissorhands, Quest.
> Note: A broad set of strong KV compression/pruning/eviction baselines for fair comparison.

Setups:
- Rank selection: per-head r, r_v chosen to retain 90–99% energy; adaptive per-head selection ablated.
> Note: They test fixed vs adaptive rank choices to see sensitivity.

- RoPE handling: post-RoPE projection; streaming anchoring without lookahead by default; small-lookahead and rotation-aware variants ablated.
> Note: Evaluate the impact of RoPE strategies on quality/speed.

- Quantization: FP16 vs INT8 coefficients; FP16 vs quantized bases ablated.
> Note: Tests how far precision can be reduced without harming accuracy.

- Sharing: per-head vs per-group (MQA/GQA) bases.
> Note: Multi-query/grouped attention can share bases to reduce overhead; this is assessed.

Metrics:
- Quality: ΔPPL, Δaccuracy vs baseline; long-context degradation vs length.
> Note: Report changes relative to full-precision attention, including how performance drops with longer contexts.

- Efficiency: tokens/sec, GPU utilization, HBM bytes/token, KV memory footprint.
> Note: Measures throughput and memory bandwidth savings, plus actual KV cache size.

- Error tracking: measured ||ℓ − ℓ̂||∞, ||o − ô||_2 vs FD ε and operator residuals; correlation analyses.
> Note: Empirical validation of the theoretical error predictors (residuals, sketch ε).

- Hypothesis validation: spectral decay (σ_i) of K/V over time, per head/layer, with RoPE on/off, and across chunks; fraction of energy vs r.
> Note: σ_i are singular values; they examine how quickly they decay to support low-rankness, including RoPE effects.

Falsifiers:
1) No r ≪ d achieves ≤1% PPL increase across tasks.
> Note: If you need large r close to d to keep quality, low-rank hypothesis fails.

2) Adaptive Subspace underperforms Static Subspace at matched r.
> Note: If online adaptation doesn’t help or hurts, the adaptive method is not justified.

3) Error or quality degrades sharply with context length despite chunking (RoPE incompatibility).
> Note: Would indicate that chunking doesn’t mitigate RoPE drift.

4) FD ε and operator-norm residuals fail to predict observed logit/output errors.
> Note: Would undermine the theoretical error control claims.

## 12. Results (summary)
- On GPT-2 Small and Pythia-410M, many heads show ≥90% energy at r ≈ d/4; Static Subspace with r=d/4 yields ≤0.3 PPL increase on WT-103 and ≤0.5% accuracy change on LAMBADA, with ≈3–4× reduction in KV bytes and ≈2–3× attention FLOPs.
> Note: Empirically, a quarter of the dimensions suffice for many heads with minimal quality loss, achieving substantial memory and compute reductions.

- Adaptive Subspace maintains quality at longer contexts, reducing chunk counts and stabilizing errors under RoPE; tokens/sec improves by 1.4–2.2× end-to-end with Triton kernels.
> Note: Online adaptation helps in long contexts and yields real throughput gains on GPU.

- FD ε correlates with measured operator residuals and observed errors; bounds are conservative but predictive.
> Note: The theoretical predictors track actual errors, though bounds are not tight (as expected).

(Full tables, ablations, and profiler traces in Appendix; code and configs released.)
> Note: Supplementary materials include detailed numbers, ablations, and performance traces to support claims.

## 13. Discussion
Coefficient-space attention complements quantization and eviction by reducing both KV memory and attention FLOPs without retraining. Adaptive chunking manages RoPE-induced drift. The main failure mode is intrinsically high-dimensional K/V structure; adaptive ranks and chunking mitigate but cannot eliminate this.
> Note: This approach can stack with other methods (e.g., quantization). Limitation: if some heads truly need high rank (little spectral decay), projecting will hurt; adaptation helps but can’t fix fundamentally high-dimensional behavior.

## 14. Limitations and Future Work
- Per-chunk basis overhead J(r+r_v)d can grow if chunks are short; dynamic thresholds, basis reuse across adjacent chunks, and basis quantization can help.
> Note: Too many chunks increase basis read cost; strategies include adjusting triggers, reusing or compressing bases.

- FD updates add overhead; sparse or subsampled updates and lighter sketches are promising.
> Note: Reducing FD update frequency or sketch size can cut compute load, trading some accuracy.

- Better RoPE-aware bases (complex-plane transport) and per-head rank selection could further improve quality/efficiency.
> Note: Modeling RoPE analytically or smarter rank selection per head may yield gains.

- Extending to cross-attention and multimodal encoders remains future work.
> Note: Current work targets self-attention in decoders; extending to other modules is open.

## 15. Reproducibility
We release code (CUDA/Triton kernels), training-free calibration scripts, and experiment configs. All results use public models and data; we report hardware details, kernel versions, and numerics. We include unit tests that validate blockwise softmax equivalence in the exact case and end-to-end error tracking hooks.
> Note: Strong reproducibility: public code/configs, environment details, and tests for key components (e.g., aggregator equivalence, error tracking).

## 16. Conclusion
We introduce an online subspace projection framework for KV caches that performs attention in a compressed coefficient space, with provable error control and practical kernels. If validated broadly, this approach can reduce the memory and bandwidth demands of LLM inference, enabling longer contexts and faster decoding on commodity hardware.
> Note: Takeaway: compute attention in low-dimensional, per-head subspaces learned online to reduce bandwidth/compute with bounded error and no retraining, potentially enabling longer and faster inference.
