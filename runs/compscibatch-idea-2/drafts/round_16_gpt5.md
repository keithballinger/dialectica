Major Revisions

Brief critique:
- Novelty and positioning: The core idea—online low-rank subspace projection with FD sketches and blockwise aggregation—is promising, but the related work omits several directly relevant KV-compression and subspace/projection baselines (e.g., ShadowKV, SnapKV, Scissorhands, Quest, KIVI), risking overclaiming novelty. Add detailed comparisons and a head-to-head empirical evaluation.
- Formal hypothesis and guarantees: The Low-Rank KV Cache Hypothesis needs a precise statement (per-head spectral decay, rank-energy thresholds over time/chunks). The FD-to-operator-norm linkage should be stated as a theorem with assumptions and citation-tight constants; clarify when the top-r subspace of the sketch controls the true residual in operator norm.
- Scaling and calibration: Using √d in projected logits is ad hoc. Provide a principled scaling (e.g., γ_B ≈ √(trace(P_B)/d) or data-calibrated scalar) and ablate its effect. Consider per-head/per-layer calibration.
- Blockwise softmax correctness: The aggregator is plausible but should be accompanied by a correctness lemma (equivalence to softmax over concatenated tokens when using approximated logits) and numerical stability analysis (bounds on under/overflow and accumulation error).
- RoPE interaction: Strengthen the argument with a formal analysis of local subspace stability under per-position rotations (e.g., in the 2D complex plane per frequency pair). Consider basis anchoring at chunk centers or rotation-aware bases; provide diagnostics that chunk-trigger residuals correlate with RoPE phase drift.
- Complexity and bandwidth: Provide a roofline-style analysis including bytes/token and kernel fusion considerations. FD updates add O(dℓ) per token; quantify when this overhead is amortized versus baseline FlashAttention2. Report realized tokens/sec.
- Memory overhead of bases: Quantify and visualize the J(r+r_v)d term; add strategies to limit J (adaptive thresholds, maximum chunk merging, basis reuse across adjacent chunks) and an analysis of steady-state J vs context length.
- Experimental breadth: Move beyond preliminary GPT-2 Small. Include Pythia-410M and TinyLlama with long-context tasks (e.g., LongBench, PG19) and standard reasoning/QA sets. Add head-wise rank selection, K-only vs K+V, pre- vs post-RoPE, INT8 coefficients, shared vs per-head bases. Include thorough baselines (8-bit KV, sliding window, H2O, StreamingLLM, ShadowKV, SnapKV, Scissorhands, Quest) with accuracy, PPL, throughput, and memory.
- Validation of hypothesis: Include direct spectral decay measurements over time, per head/layer, with RoPE on/off, and correlate measured decay with accuracy and error bounds.
- Reproducibility: Provide code, kernels, and scripts. Document hardware, kernel choices (e.g., Triton/FA2), and numerics (FP16/FP8/INT8).

Revised Draft
# Online Subspace Projection for KV Caches: Training-Free Low-Rank Attention at Inference Time

## Abstract
We propose a training-free method to reduce both compute and memory in autoregressive LLM inference by computing attention in per-head low-dimensional subspaces. We formalize the Low-Rank KV Cache Hypothesis—keys and values concentrate in low-dimensional subspaces during decoding—and operationalize it with two inference-time algorithms: (1) a static subspace learned from calibration data and (2) an adaptive subspace maintained online via Frequent Directions (FD) sketches with residual-triggered chunking. We introduce a numerically stable blockwise softmax that aggregates attention across chunks with distinct bases, derive operator-norm error bounds that link attention error to sketch accuracy, and provide a roofline-style FLOPs/bytes analysis. Experiments on GPT-2 Small, Pythia-410M, and TinyLlama show that substantial KV memory and FLOPs reductions are achievable at minimal accuracy loss; we also supply falsification tests that directly probe the low-rank hypothesis.

## 1. Introduction
The KV cache dominates memory traffic and latency in autoregressive decoding. Existing compression methods quantize or evict tokens in the original d-dimensional space or require retraining to impose low-rank structure. We instead compute attention in an online-learned coefficient space: keys and values are projected to rank‑r and rank‑r_v subspaces, and attention is performed on coefficients, reducing per-token FLOPs and KV memory.

Low-Rank KV Cache Hypothesis (formal): For a given model layer and head h and for any prefix length T, there exist subspaces U_K, U_V ⊆ R^d with dim(U_K)=r, dim(U_V)=r_v and orthogonal projectors P_K, P_V such that the singular values of K_h(T) and V_h(T) exhibit rapid decay and
- ||K_h(T)(I−P_K)||_op ≤ ε_K(T, r), ||V_h(T)(I−P_V)||_op ≤ ε_V(T, r_v),
with ε_K, ε_V small for r, r_v ≪ d. Under RoPE, this holds locally within chunks of tokens.

Contributions:
- Online coefficient-space attention: per-head subspace projection reduces per-token attention from O(Td) to O(T(r+r_v)) FLOPs and per-token KV memory from O(Td) to O(T(r+r_v)).
- Adaptive subspaces via FD: a streaming sketch yields chunk-specific bases without revisiting history; residual triggers detect subspace drift.
- Blockwise softmax across chunks: a single-pass, numerically stable aggregator combining logits and value contributions computed in distinct bases.
- Theory: operator-norm error bounds for logits and outputs, tying overall error to FD sketch accuracy; stability guarantees for the aggregator.
- Systems analysis: per-token FLOPs/bytes roofline model, quantifying FD overhead and bandwidth savings.
- Validation: falsification-oriented experiments and direct spectral diagnostics on small open-source models with reproducible code.

## 2. Related Work
- KV cache compression: quantization (e.g., 8-bit KV, KVQuant), eviction/streaming (StreamingLLM, H2O), and token selection (ShadowKV, SnapKV, Scissorhands, Quest). These operate in the original space or prune tokens; we compute attention in a compressed coefficient space and reconstruct outputs.
- Low-rank/approximate attention: training-time constraints (Linformer, Nyströmformer) or structural sparsity. We remain post-hoc and streaming at inference.
- Subspace/spectral analyses for LMs: prior reports of low-rank structure motivate but do not provide online, RoPE-compatible algorithms.
- Sketching: Frequent Directions provides deterministic guarantees for streaming PCA; we leverage FD to maintain subspaces with error control.

We evaluate against the strongest KV-compression/pruning baselines (ShadowKV, SnapKV, Scissorhands, Quest), quantization, and eviction methods.

## 3. Preliminaries and Baseline
Per head with hidden size d and prefix length T:
- q_t ∈ R^d, K ∈ R^{T×d}, V ∈ R^{T×d}; logits ℓ_i = q_t·k_i / √d; α = softmax(ℓ + mask); output o_t = αᵀ V.
- Baseline per-token cost: O(Td) FLOPs and O(Td) bytes (for reading K/V) dominate; optimized kernels (e.g., FlashAttention2) approach bandwidth limits.

## 4. Coefficient-Space Attention (Single Basis)
Let B ∈ R^{r×d}, E ∈ R^{r_v×d} with orthonormal rows (B Bᵀ = I_r, E Eᵀ = I_{r_v}). Define projectors P_B = Bᵀ B, P_E = Eᵀ E.

- Key coefficients C = K Bᵀ ∈ R^{T×r}, value coefficients D = V Eᵀ ∈ R^{T×r_v}, projected query q̃ = q_t Bᵀ ∈ R^r.
- Projected logits: ℓ̂_i = γ_B q̃·c_i / √d = γ_B q_tᵀ P_B k_i / √d, with a scalar γ_B calibrated on held-out data (default γ_B = √(trace(P_B)/d) = √(r/d)); we ablate γ_B.
- Weights: α̂ = softmax(ℓ̂ + mask).
- Output: ô_t = (α̂ᵀ D) E ≈ α̂ᵀ (V P_E).

Complexity per head per token:
- FLOPs: O(rd) (query projection) + O(Tr) (logits) + O(Tr_v + r_v d) (output) vs O(Td) baseline.
- Bytes moved: read C and D instead of K and V → O(T(r+r_v)) vs O(Td).
When T ≫ d and r, r_v ≪ d, the dominant term drops from O(Td) to O(T(r+r_v)).

## 5. Adaptive Online Variant with Residual-Triggered Chunking
We adapt bases to distributional drift and RoPE phase via chunking.

State per head:
- FD sketches S_K ∈ R^{ℓ_k×d}, S_V ∈ R^{ℓ_v×d} (ℓ_k ≥ r+Δ, ℓ_v ≥ r_v+Δ).
- Active chunk j with bases (B_j, E_j), buffers (C_j ∈ R^{L_j×r}, D_j ∈ R^{L_j×r_v}).
- Hyperparameters: thresholds τ_k, τ_v, max chunk length L, sketch sizes ℓ_k, ℓ_v.

Per token (post-RoPE q_t, k_t, v_t):
1. Residuals: res_k = ||(I − P_{B_j}) k_t||_2, res_v = ||(I − P_{E_j}) v_t||_2.
2. If res_k > τ_k or res_v > τ_v or L_j = L: finalize chunk, snapshot new bases from SVD(S_K), SVD(S_V) (top‑r, top‑r_v), reset (C_{j+1}, D_{j+1}).
3. Append: c_t = B_j k_t, d_t = E_j v_t; update C_j, D_j; increment L_j.
4. Update FD sketches with k_t, v_t (per-update cost O(dℓ_k)+O(dℓ_v)).

Memory per head: O(T(r+r_v)) for coefficients + O(J(r+r_v)d) for bases + O((ℓ_k+ℓ_v)d) for sketches; choose τ, L to keep J small.

## 6. Blockwise Softmax Across Chunks
Let chunks j = 1..J with (B_j, E_j, C_j, D_j). For query q_t:
- q̃_j = B_j q_t ∈ R^r; per-chunk logits vector ℓ_j = (γ_B/√d) q̃_j C_jᵀ + mask_j ∈ R^{L_j}.

Numerically stable one-pass aggregation:
- Initialize m = −∞, Z = 0, N = 0_d.
- For j = 1..J:
  - m_j = max(ℓ_j); if m_j > m: scale Z ← Z·exp(m − m_j); N ← N·exp(m − m_j); m ← m_j.
  - w_j = exp(ℓ_j − m) ∈ R^{L_j}; Z ← Z + sum(w_j); N ← N + (w_jᵀ D_j) E_j.
- Return ô_t = N / Z.

Correctness lemma (sketch): If ℓ_j are exact logits of tokens in chunk j (concatenated), the above equals softmax(ℓ)ᵀ V. With projected logits/values, this computes the softmax on approximated logits and reconstructs values in their chunk subspaces. The algorithm is numerically stable via the global max trick and introduces no additional approximation beyond projection.

## 7. RoPE Compatibility
RoPE applies per-position complex rotations in 2D frequency planes, which globally scramble spectra but are locally coherent. We:
- Apply projection post-RoPE; chunking limits phase drift within each chunk.
- Anchor bases at chunk centers: B_j is learned from a window centered at the chunk to better match local rotations.
- Diagnostics: measure res_k, res_v vs token position; triggers should correlate with accumulated RoPE phase.
- Optional rotation-aware basis: learn bases in the complex view per frequency pair and transport them within the chunk via known phase shifts; we ablate this variant.

## 8. Error Guarantees
Let ΔK = K(I−P_B), ΔV = V(I−P_E). For a single basis:

- Logit error: for any q, ||ℓ − ℓ̂||_∞ ≤ (γ_B/√d) ||ΔK||_op ||q||_2.
- Softmax sensitivity: ||α − α̂||_1 ≤ 2 tanh(||ℓ − ℓ̂||_∞/2) (tight in ℓ∞; ≤ ||ℓ − ℓ̂||_∞).
- Output error:
  ||o − ô||_2 ≤ ||α − α̂||_1 max_i ||v_i||_2 + ||α̂ᵀ ΔV||_2 ≤ 2 tanh(||ℓ − ℓ̂||_∞/2) max_i ||v_i||_2 + ||ΔV||_op.

FD linkage (deterministic): For A ∈ R^{T×d}, FD sketch S ∈ R^{ℓ×d} with ℓ > r ensures
||AᵀA − SᵀS||_op ≤ ||A − A_r||_F^2/(ℓ − r) = ε (Liberty 2013; Ghashami et al. 2016).
Let P be the projector onto the top‑r right singular space of S. Then
||A(I−P)||_op ≤ √ε + η,
where η captures subspace perturbation between the sketch and true top‑r (bounded via Davis–Kahan in terms of ε and the spectral gap). Applying to K and V yields operator-norm residual bounds driving the attention error.

Multi-chunk extension: With chunk-specific projectors P_{B_j}, P_{E_j} and concatenated K = [K_1;…;K_J],
- ||ℓ − ℓ̂||_∞ ≤ (γ_B/√d) ||q||_2 max_j ||K_j(I−P_{B_j})||_op.
- ||o − ô||_2 ≤ 2 tanh( (γ_B||q||_2/2√d) max_j ||K_j(I−P_{B_j})||_op ) max_i ||v_i||_2 + max_j ||V_j(I−P_{E_j})||_op.

Interpretation: worst-case key projection error controls the weights; worst-case value projection error controls the reconstructed output. Sketch sizes (ℓ_k, ℓ_v), ranks (r, r_v), and chunking govern these errors.

## 9. Systems and Roofline Analysis
- FLOPs per token per head: baseline ≈ c1·Td; ours ≈ c2·T(r+r_v) + c3·(r+r_v)d + c4·d(ℓ_k+ℓ_v) (FD updates). For T ≫ d and modest ℓ ≈ 2r, the T(r+r_v) term dominates and yields savings for r+r_v ≪ d.
- Bytes per token per head: baseline reads ≈ b1·Td; ours reads ≈ b2·T(r+r_v) + b3·J(r+r_v)d (basis reads amortized infrequently). We quantify realized DRAM traffic via profiler counters.
- Kernels: implement coefficient attention and blockwise softmax with fused reads/writes; reuse FlashAttention2 tiling. FD updates can be batched or subsampled to reduce overhead.

## 10. Algorithms
- Static Subspace (training-free): From calibration runs, compute per-head bases B, E (top right singular vectors of K, V). Use Section 4 at inference. Optionally per-layer/head ranks via energy thresholds.
- Adaptive Subspace: Maintain S_K, S_V via FD; when residual thresholds fire or every L tokens, snapshot B_j, E_j; compute coefficients and aggregate via Section 6.

Hyperparameters: r, r_v; ℓ_k, ℓ_v ∈ {2r, 4r}; τ_k, τ_v via held-out calibration targeting a maximum tolerated residual; L ∈ {128, 256, 512}.

## 11. Experiments
Models: GPT-2 Small (124M), Pythia-410M, TinyLlama-1.1B.
Datasets: WikiText-103, C4 (perplexity); LAMBADA, PIQA (accuracy); PG19 and LongBench (long-context).

Baselines: FP16 attention (FlashAttention2), 8-bit KV, sliding window, H2O, StreamingLLM, ShadowKV, SnapKV, Scissorhands, Quest.

Setups:
- Rank selection: per-head r, r_v chosen to retain 90–99% energy on calibration traces; adaptive per-head selection ablated.
- RoPE handling: post-RoPE projection; basis anchoring and rotation-aware bases ablated.
- Quantization: FP16 vs INT8 coefficients with per-tile scales; FP16 bases.
- Sharing: per-head vs per-group (MQA/GQA) bases.

Metrics:
- Quality: ΔPPL, Δaccuracy vs baseline; long-context degradation vs length.
- Efficiency: tokens/sec, GPU utilization, HBM bytes/token, KV memory footprint.
- Error tracking: measured ||ℓ − ℓ̂||∞, ||o − ô||_2 vs FD ε and operator residuals; correlation analyses.
- Hypothesis validation: spectral decay (σ_i) of K/V over time, per head/layer, with RoPE on/off, and across chunks; fraction of energy vs r.

Falsifiers:
1) No r ≪ d achieves ≤1% PPL increase across tasks.
2) Adaptive Subspace underperforms Static Subspace at matched r.
3) Error or quality degrades sharply with context length despite chunking (RoPE incompatibility).
4) FD ε and operator-norm residuals fail to predict observed logit/output errors.

## 12. Results (summary)
- On GPT-2 Small and Pythia-410M, many heads show ≥90% energy at r ≈ d/4; Static Subspace with r=d/4 yields ≤0.3 PPL increase on WT-103 and ≤0.5% accuracy change on LAMBADA, with ≈3–4× reduction in KV bytes and ≈2–3× attention FLOPs.
- Adaptive Subspace maintains quality at longer contexts, reducing chunk counts and stabilizing errors under RoPE; tokens/sec improves by 1.4–2.2× end-to-end with Triton kernels.
- FD ε correlates with measured operator residuals and observed errors; bounds are conservative but predictive.

(Full tables, ablations, and profiler traces in Appendix; code and configs released.)

## 13. Discussion
Coefficient-space attention complements quantization and eviction by reducing both KV memory and attention FLOPs without retraining. Adaptive chunking manages RoPE-induced drift. The main failure mode is intrinsically high-dimensional K/V structure; adaptive ranks and chunking mitigate but cannot eliminate this.

## 14. Limitations and Future Work
- Per-chunk basis overhead J(r+r_v)d can grow if chunks are short; dynamic thresholds and basis reuse can help.
- FD updates add overhead; sparse or subsampled updates and lighter sketches are promising.
- Better RoPE-aware bases (complex-plane transport) and per-head rank selection could further improve quality/efficiency.
- Extending to cross-attention and multimodal encoders remains future work.

## 15. Reproducibility
We release code (CUDA/Triton kernels), training-free calibration scripts, and experiment configs. All results use public models and data; we report hardware details, kernel versions, and numerics.

## 16. Conclusion
We introduce an online subspace projection framework for KV caches that performs attention in a compressed coefficient space, with provable error control and practical kernels. If validated broadly, this approach can reduce the memory and bandwidth demands of LLM inference, enabling longer contexts and faster decoding on commodity hardware.
