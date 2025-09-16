Minor Revisions

Brief critique
- The distinction between exact and approximate reuse is clear and a major strength. However, the mechanism linking the storage strategy (Regime B, multicut) to the admission policy (depth-aware guard) could be more explicit. The paper should frame the "multicut" strategy as a core component of the storage design that enables the runtime admission policy to select the deepest safe reuse boundary.
- The "depth-aware guard" is presented with a specific example (`W_exact(0)=128`, etc.). It should be more clearly defined as a calibratable heuristic policy function, `W_exact(ℓ)`, whose parameters are determined empirically to manage the quality-performance trade-off.
- In the segment indexing section, clarify that LSH is used for efficient *candidate retrieval*, which is then followed by a precise verification step (e.g., exact token ID matching for exact mode).
- The presentation can be slightly tightened. For example, the "Notes" in the caching section can be integrated into the main paragraphs for better flow.

Revised Draft
# Cross-Query KV Reuse with RoPE Reindexing for Template-Heavy Agent Prompts

## Abstract
Agentic LLM workflows repeatedly process similar internal prompt fragments like system preambles and tool schemas. We present a cross-query KV reuse mechanism that identifies repeated segments, realigns cached attention keys to new positions via Rotary Position Embedding (RoPE) reindexing, and splices them into the runtime KV cache to avoid recomputation. We formalize two regimes: (A) full-stack reuse for maximum speedup, and (B) boundary-state reuse using a multicut storage strategy to create multiple fallback points. Reuse is governed by a tiered admission policy with a zero-drift exact mode and a calibrated approximate mode featuring a depth-aware guard to bound quality drift. On Llama/Mistral 7B-class models, our prototype targets 20–40% prefill FLOPs reduction on template-heavy agents with negligible quality change in exact mode and bounded drift in approximate mode. We release code for validation on small open-source models.

## 1. Introduction
LLM agents routinely include near-duplicate structures across requests. Prefix caching exploits only identical leading spans, failing for internal repeats, shifted positions, or minor edits, leaving substantial prefill compute untapped.

We propose cross-query KV reuse:
- Retrieve previously computed per-layer K/V for repeated internal segments via content-aware indexing.
- Realign cached keys to new absolute positions exactly under RoPE via a delta rotation.
- Splice K/V into the runtime cache to eliminate segment recomputation.

Our key observation is that while later tokens only depend on previous tokens’ K/V at the same layer, the K/V values themselves depend on the residual stream, which is shaped by the left context at all earlier layers. Therefore, cross-context KV reuse is exact *only if* the left context is identical for all reused layers. Otherwise, it is an approximation that requires guardrails.

Contributions:
- An execution design for cross-query, cross-position KV reuse with two storage regimes (full-stack and multicut boundary-state) and explicit validity conditions.
- Exact RoPE reindexing for K across layers/heads, covering NTK/YARN scaling and MQA/GQA.
- A tiered admission policy: an exact mode with zero quality drift and a calibrated approximate mode that uses a depth-aware heuristic to select the deepest safe reuse boundary.
- An open implementation on Llama/Mistral 7B-class models, with a detailed evaluation plan on agentic tasks.

## 2. Background and Scope
We consider decoder-only transformers with RoPE and paged KV caches.

- **Requirement for exactness:** Reusing K/V at layer ℓ for a segment is exact only if (a) the segment’s token IDs are identical and (b) the entire left context preceding the segment is identical for all reused layers up to ℓ (RoPE differences are handled by reindexing).
- **Approximate reuse:** If left context differs, reused K/V are approximations. We provide a conservative admission policy to bound quality drift.
- **Non-goals:** Training changes; non-RoPE architectures.

## 3. Method

### 3.1 Segment Discovery and Indexing
We use content-based hashing to find reusable segment candidates, which are then subject to strict verification by the admission policy.
- **Segmentation:** Extract structural spans (e.g., tool specs) and overlapping fixed windows (64–128 tokens, stride 32–64).
- **Canonicalization for indexing:** To build robust index keys, we lowercase, normalize whitespace, and mask volatile substrings (timestamps/IDs). This canonical form is used *only* for indexing; all cached tensors correspond to the original, unmodified token IDs.
- **Signatures and Index:** We use MinHash over token shingles (k≈6–10) on the canonicalized span and Locality-Sensitive Hashing (LSH) for efficient approximate Jaccard search. An auxiliary signature of the left context (≈64–256 tokens) helps retrieve candidates with similar neighborhoods.

### 3.2 What to Cache
We expose compute–memory trade-offs and use correct accounting based on `d_kv = n_kv_heads × head_dim`. Tensors are quantized by default (int8, per-channel/group scales).

- **Regime A: Full-stack KV reuse (max speedup)**
  - Store per token per layer: Kℓ (after RoPE) and Vℓ (unrotated).
  - Memory per token: `2 · d_kv · L` bytes (with int8 quantization).
- **Regime B: Multicut Boundary-State Reuse (lower memory, more flexibility)**
  - Store Kℓ and Vℓ for layers ℓ ∈ [0, `L_cut`) and the residual hidden state `h_{L_cut}` for multiple boundaries `L_cut` (e.g., every 4 layers). This creates several fallback points for a single segment. Storing a boundary does *not* fix a left-context mismatch in lower layers; it merely allows recomputation to begin from that layer.
  - Memory per token for one boundary `L_cut`: `2 · d_kv · L_cut + d_model` bytes (int8).
- **Example (Llama-2-7B:** `d_model=4096`, `n_heads=32`, `n_kv_heads=8`, `head_dim=128` ⇒ `d_kv=1024`, `L=32`, int8):
  - **Regime A:** `2 · 1024 · 32 = 64` KB/token. A 128-token segment requires ≈ 8.0 MB.
  - **Regime B (L_cut=12):** `2 · 1024 · 12 + 4096 = 27.6` KB/token. A 128-token segment with one boundary requires ≈ 3.5 MB.
- **Metadata:** We store a hash of the model/adapters, tokenizer, RoPE config, attention mask type, segment token IDs, and original absolute positions. Cached tensors are stored in a contiguous layout aligned with the runtime paged KV manager.

### 3.3 RoPE Reindexing (Exact Key Position Shift)
We store K after RoPE to enable exact reindexing via a delta rotation. Given the RoPE rotation `R_h(p; θ)` at head `h` and absolute position `p`:
- `K_h(p_new) = R_h(p_new) · R_h(p_old)⁻¹ · K_h(p_old) = R_h(p_new − p_old) · K_h(p_old)`
- This identity holds provided the rotation angles are additive in position, which is true for standard RoPE and variants like NTK/YARN. The implementation uses the exact angle parameterization of the target runtime. For MQA/GQA, the rotation is applied to each shared KV head. Values (V) are position-agnostic and require no change.

### 3.4 Admission and Execution
A tiered policy separates exact from approximate reuse.

1.  **Candidate Retrieval:** Query the LSH index for each potential segment. Filter candidates by metadata (model hash, etc.) and Jaccard similarity.
2.  **Tiered Admission:**
    - **Exact Mode (default, zero drift):** Requires a 100% token-ID match for the segment and for the *entire* left context preceding it. This guarantees correctness up to quantization errors.
    - **Approximate Mode (calibrated, optional):** If an exact left-context match is unavailable, this mode attempts to find the deepest, safest reuse boundary. It uses a depth-aware guard policy: select the deepest stored boundary `L_cut` such that an exact token-ID match holds for a window of size `W_exact(L_cut)` immediately preceding the segment. `W_exact(ℓ)` is a calibratable, monotonically increasing function (e.g., `W_exact(0)=128`, `W_exact(16)=full_prefix`). If no stored boundary satisfies its window requirement, reuse is rejected. An optional online probe can add further safety by recomputing Q for a few tokens and checking its compatibility with the cached K before committing to reuse.
3.  **Splicing and Compute:**
    - For a chosen segment and boundary `L_cut`, map it to its new absolute positions.
    - For layers `ℓ < L_cut`: load Kℓ,Vℓ, apply the delta-RoPE rotation to Kℓ, and insert into the paged KV cache.
    - For layers `ℓ ≥ L_cut`: start from the stored boundary state `h_{L_{cut}}` and perform standard forward computation. In Regime A, `L_cut=L`.
    - Skip forward compute for all layers where K/V were supplied from the cache. Ensure attention masks and page metadata are correctly aligned.
4.  **Fallback:** If no candidate passes admission, compute from scratch and optionally add the new segment to the cache with multicut boundaries.

### 3.5 Complexity, IO, and Savings
- **Overheads:** LSH probe is sub-millisecond. Reindexing and insertion are fast `O(T · L_used · d_kv)` bytewise operations implemented as a fused kernel. For a 128-token Llama-7B segment in Regime A (≈8 MB), IO takes ≈0.33 ms on a PCIe 4.0 x16 bus and can be overlapped with compute via prefetching.
- **Savings:** Regime A eliminates all compute for the reused segment. Regime B saves compute for layers `ℓ < L_cut`. A break-even point is reached when saved FLOPs exceed the overheads of IO, reindexing, and admission checks.

## 4. Calibration and Evaluation Protocol
- **Tasks:** Agent scaffolds (ReAct on HotpotQA), tool-use flows (ToolBench-like), and synthetic workflows with controlled left-context perturbations.
- **Calibration:** Sweep LSH thresholds, quantization levels, and the `W_exact(ℓ)` policy function parameters. Calibrate probe thresholds to bound logit-KL and task-metric drift.
- **Baselines:** No reuse, exact prefix caching, paged-KV only, batching, and speculative decoding (orthogonal).
- **Metrics:** Latency, tokens/s, prefill FLOPs, GPU utilization. Quality measured by EM/F1, tool success rate, and logit-KL divergence from the no-reuse baseline. Cache metrics include acceptance rate by tier and hit frequency.
- **Hardware:** A100-40GB and L4-24GB systems with pinned CPU memory.

## 5. Results (Planned)
We will report end-to-end speedups, memory vs. performance trade-offs of Regime A vs. B, quantization effects, robustness across RoPE variants, and adversarial tests showing acceptance vs. quality drift curves to validate the efficacy of the depth-aware guard policy.

## 6. Analysis and Discussion
- **Exactness:** Our system guarantees correctness when the segment and its entire left context are identical across all reused layers. This is common for template-anchored spans in agent systems.
- **Approximate Reuse:** With conservative depth-aware guards enabled by our multicut storage, drift is bounded. Boundary-state reuse reduces memory and increases opportunities for partial reuse but must be paired with a strict admission policy.
- **IO vs. Compute:** With corrected `d_kv`-based sizing, CPU↔GPU bandwidth is not the dominant bottleneck for 7B-class models on modern PCIe interconnects, especially with prefetching.

## 7. Relation to Prior Work
We extend prefix caching (e.g., vLLM) to internal, position-shifted segments. Unlike generic KV compression, we perform lossless (or near-lossless with quantization) reuse. Our work is orthogonal to speculative decoding. Compared to other prompt caching systems, we provide a principled RoPE reindexing method and a rigorous, depth-aware admission policy to manage the exactness-approximation trade-off.

## 8. Limitations
- The approach is approximate unless left context is identical; our safeguards are heuristic.
- The method is specific to RoPE-based models.
- The storage footprint grows with the number of segments and cuts, requiring eviction policies.
- Adapter/LoRA state must match; we verify this via metadata hashing.

## 9. Systems and Implementation Notes
Our implementation uses pinned CPU memory for storage, asynchronous prefetching, and fused CUDA kernels for dequantization and delta-RoPE rotation. The cache manager is thread-safe and integrates with paged KV managers by marking prefilled pages as read-only. Metadata fidelity is critical and includes hashes of all relevant model and tokenizer components.

## 10. Reproducibility and Artifact Plan
We will release our reference implementation, evaluation scripts, and calibration artifacts, along with unit tests verifying RoPE reindexing correctness under various scaling strategies and model architectures (GQA/MQA).

## 11. Conclusion
Cross-query KV reuse with RoPE reindexing and a rigorous admission policy effectively reduces prefill computation for agentic LLMs. By distinguishing between exact and calibrated approximate reuse and using a multicut storage strategy, our method provides significant speedups on open models while preserving quality within well-defined bounds. We provide code to enable validation and further research.
