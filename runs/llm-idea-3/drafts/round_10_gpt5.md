Major Revisions

Brief critique
- Correctness is overstated. Reusing per-layer K/V across different left contexts is not exact. Even with identical segment tokens and perfect RoPE reindexing, K/V depend on the residual stream shaped by the left context at earlier layers. The draft needs a precise statement of when reuse is exact (rare) versus approximate, and guardrails for the latter.
- Admission policy still allows approximate reuse without an online check. To make the approach principled, add a tiered policy: (a) exact mode requiring identical segment and identical left context for all reused layers, and (b) calibrated approximate mode with either a strict left-context exact-match window per reused depth or a cheap online probe. Clarify when each mode is used.
- Memory accounting for KV storage is incorrect. The draft uses d_model in per-token costs; KV cache size scales with d_kv = n_kv_heads × head_dim, not d_model. This error propagates to examples and systems claims.
- Boundary-state reuse does not “protect correctness” if the lower-layer K/V were computed under a different left context. Explain the validity conditions and present a multicut strategy (multiple stored boundaries) to choose the deepest safe boundary at runtime.
- RoPE reindexing is correct in spirit, but details for NTK/YARN scaling need to state the exact angle parameterization and when the additive delta holds. Explicitly constrain usage to configurations where θ(d) is layer/head-constant and additive in position.
- Systems/IO: With corrected memory figures, the CPU↔GPU bandwidth story changes. Quantify per-segment bytes, prefetch timelines, and break-even points with the corrected d_kv-based sizes.
- Evaluation must include adversarial left-context shifts to characterize drift, per-layer cosine/KL diagnostics, and acceptance-rate vs. quality trade-offs. Also include adapter/LoRA metadata compatibility tests.

Revised Draft
# Cross-Query KV Reuse with RoPE Reindexing for Template-Heavy Agent Prompts

## Abstract
Agentic LLM workflows repeatedly process similar internal prompt fragments (system preambles, tool schemas, chain-of-thought scaffolds). We present a cross-query KV reuse mechanism that identifies repeated segments across requests, realigns attention keys under RoPE to new positions, and splices cached per-layer K/V into the runtime KV cache to avoid recomputation. We formalize two regimes: (A) full-stack KV reuse (store per-layer K/V; skip all compute for the segment) and (B) boundary-state reuse (store K/V for lower layers and a layer-boundary hidden state to recompute upper layers). We provide a practical design for RoPE reindexing under NTK/YARN scaling and MQA/GQA, a tiered admission policy with exact and calibrated-approximate modes, and integration with paged KV managers. On Llama/Mistral 7B-class models, our prototype targets 20–40% prefill FLOPs reduction on template-heavy agents with negligible quality change in exact mode and bounded drift in approximate mode. We release code for validation on small open-source models.

## 1. Introduction
LLM agents routinely include near-duplicate internal structures across steps and users. Prefix caching exploits only identical leading spans. It fails for internal repeats, shifted positions, or minor edits, leaving substantial prefill compute untapped.

We propose cross-query KV reuse:
- Retrieve previously computed per-layer K/V for repeated internal segments via content-aware indexing.
- Realign cached keys to new absolute positions exactly under RoPE via a delta rotation.
- Splice K/V into the runtime cache to eliminate segment recomputation.

Key observation: During prefill/decoding, later tokens only need previous tokens’ K/V at the same layer; they do not need those tokens’ hidden states. However, the K/V of a token depend on its residual stream, which is influenced by the left context in earlier layers. Therefore, cross-context KV reuse is exact only when the left context is identical for all reused layers; otherwise it is an approximation that requires guardrails.

Contributions:
- An execution design for cross-query, cross-position KV reuse with two regimes (full-stack and boundary-state) and explicit validity conditions.
- Exact RoPE reindexing for K across layers/heads, covering NTK/YARN and MQA/GQA.
- A tiered admission policy: an exact mode with zero quality drift and a calibrated approximate mode with depth-aware guards and an optional online probe.
- An open implementation on Llama/Mistral 7B-class models, with an evaluation plan on agentic tasks.

## 2. Background and Scope
We consider decoder-only transformers with RoPE, paged KV caches, and token-by-token decoding.

- Requirement for exactness: Reusing K/V at layer ℓ for a segment is exact only if (a) the segment’s token IDs are identical and (b) the entire left context up to the segment is identical for all reused layers (RoPE differences handled by reindexing).
- Approximate reuse: If left context differs, reused K/V are approximations. We provide conservative acceptance and calibration to bound drift.
- Non-goals: Training changes; non-RoPE architectures.

## 3. Method

### 3.1 Segment Discovery and Indexing
- Segmentation:
  - Extract structural spans (e.g., tool specs, system preambles).
  - Add overlapping fixed windows (64–128 tokens, stride 32–64).
- Canonicalization for indexing only:
  - Lowercase, normalize whitespace, mask volatile substrings (timestamps/IDs). Canonicalization is used solely to build index keys; all cached tensors correspond to the original, unmodified token IDs.
- Signatures and Index:
  - MinHash over token shingles (k≈6–10) on the canonicalized span; LSH for approximate Jaccard search.
  - Auxiliary left-context signature for a window W_ctx (≈64–256 tokens) to favor matches with similar neighborhoods.

### 3.2 What to Cache
We expose compute–memory trade-offs and correct accounting in terms of d_kv = n_kv_heads × head_dim.

- Regime A: Full-stack KV reuse (max speedup)
  - Store per token per layer: Kℓ after RoPE and Vℓ (unrotated), quantized (int8 default).
  - Memory per token: 2 · d_kv · L bytes (int8).
- Regime B: Boundary-state reuse (lower memory)
  - Store for layers ℓ ∈ [0, L_reuse): Kℓ (after RoPE) and Vℓ.
  - Store residual hidden state h_{L_reuse} (post-layer L_reuse) per token.
  - Memory per token: 2 · d_kv · L_reuse + d_model bytes (int8).
- Practical example (Llama-2-7B: d_model=4096, n_heads=32, n_kv_heads=8, head_dim=128 ⇒ d_kv=1024, L=32, L_reuse=12, int8):
  - Regime A: 2 · 1024 · 32 = 65,536 B/token ≈ 64 KB; a 128-token segment ≈ 8.0 MB.
  - Regime B: 2 · 1024 · 12 + 4096 = 28,288 B/token ≈ 27.6 KB; a 128-token segment ≈ 3.5 MB.
- Metadata: model/adapters hash, tokenizer hash, n_heads, n_kv_heads, head_dim, RoPE base/scaling, attention mask type, segment token IDs, original absolute positions.

Notes:
- Quantization: int8 (per-channel/group scales) by default; optional int4 halves KV bytes.
- Layout: Store per-layer contiguous KV blocks aligned with the runtime paged KV layout.

### 3.3 RoPE Reindexing (Exact Key Position Shift)
We store K after RoPE to enable exact reindexing via a delta rotation.

Given the RoPE rotation R_h(p; θ) applied at head h and absolute position p:
- For cached K at old position p_old and new position p_new:
  - K_h(p_new) = R_h(p_new) · R_h(p_old)^{-1} · K_h(p_old) = R_h(p_new − p_old) · K_h(p_old),
  provided θ(d) is head-constant and additive in position (standard RoPE; NTK/YARN modify θ but preserve additivity).
Implementation:
- Llama/Mistral interleaved 2D rotations with per-dim frequencies θ(d) computed identically to the target runtime (including NTK/YARN scaling).
- MQA/GQA: Apply per-KV-head rotation to shared KV heads; Q is computed fresh.
- Values V are position-agnostic and unchanged.

### 3.4 Admission and Execution
We adopt a tiered policy that separates exact from approximate reuse.

1) Candidate retrieval
- For each potential segment, query the LSH using the canonical signature.
- Filter by model/adapters/tokenizer/RoPE metadata and similarity thresholds on segment Jaccard (τ_seg) and left-context signature (τ_ctx).

2) Tiered admission
- Exact mode (default when available; zero drift):
  - Require 100% token-ID match for the segment and 100% token-ID match for the entire left context that precedes the segment in the new prompt relative to the cached exemplar used to create the K/V for all layers to be reused.
  - This guarantees correctness up to numeric quantization; RoPE reindexing restores positional alignment.
- Approximate mode (calibrated; optional):
  - Depth-aware guard: select the deepest reuse boundary ℓ* such that an exact left-context token-ID match holds for a window of size W_exact(ℓ*) immediately preceding the segment, where W_exact grows with depth (e.g., W_exact(0)=128, W_exact(8)=512, W_exact(16)=full prefix). Fall back to shallower reuse (smaller ℓ*) if the condition fails.
  - Optional online probe (≤1–2% overhead): for one or two probe tokens in the segment, recompute Q (and optionally the first k layers) under the current context and measure cosine(Q·K_cached, Q·K_fresh_left) for a small left-context slice; reject if below a calibrated threshold. This bounds drift without recomputing the whole segment.
  - If neither condition is satisfied, do not reuse.

3) Splicing and compute
- Map the segment to absolute positions in the new prompt.
- For each layer ℓ:
  - Regime A: Load Kℓ,Vℓ, apply per-head delta RoPE R_h(Δp), and insert into the paged KV cache at target positions.
  - Regime B: For ℓ < L_reuse, same as above. For ℓ ≥ L_reuse, start from stored h_{L_reuse} and compute layers ℓ ≥ L_reuse to produce the K/V needed by subsequent tokens at upper layers.
- Skip forward compute for the segment at all layers where K/V are supplied by the cache.
- Ensure attention masks and block/page metadata align; mark prefilled pages read-only.

4) Fallback and learning
- If no candidate passes admission, compute from scratch and optionally add the segment to the store with multicut boundaries (e.g., every 4 layers) to maximize future opportunities.

### 3.5 Complexity, IO, and Savings
- Overheads:
  - LSH probe: sub-millisecond on CPU.
  - Reindex + insert: O(T · L_used · d_kv) bytewise ops; implemented as fused int8 dequant + delta-RoPE kernels.
  - IO: For Llama-7B, 128-token segment in Regime A ≈ 8 MB (int8). PCIe 4.0 x16 sustained ~24 GB/s ⇒ ≈0.33 ms per segment; overlappable with compute via prefetching.
- Savings:
  - Regime A removes all segment compute (KV projections, attention, MLP) across reused layers.
  - Regime B saves lower-layer FLOPs; upper layers recomputed.
  - Break-even reuse frequency f* when saved FLOPs – (IO + reindex + probe) ≥ 0; we report f* empirically per hardware.
- Integration:
  - Compatible with batching, FlashAttention/FlashDecoding, and paged KV managers.

## 4. Calibration and Evaluation Protocol
Datasets and tasks:
- Agent scaffolds on HotpotQA (distractor) with ReAct; ToolBench-like tool-use flows; synthetic templated workflows with controlled left-context perturbations.

Calibration:
- Sweep τ_seg, τ_ctx, quantization (int8/int4), and depth guards W_exact(ℓ).
- For approximate mode, calibrate optional probe thresholds to bound logit-KL and EM drift.

Baselines:
- No reuse; exact prefix caching; paged-KV only; batching + FlashAttention; speculative decoding (orthogonal).

Metrics:
- Latency, tokens/s, prefill FLOPs, GPU utilization, CPU↔GPU bytes.
- Quality: EM/F1 (HotpotQA), tool success/answer accuracy; logit-KL vs. no-reuse.
- Stability: output consistency across seeds.
- Cache: acceptance rate by tier, segment hit frequency, memory footprint, eviction stats.

Hardware:
- A100-40GB and L4-24GB; pinned CPU memory; prefetch overlap.

## 5. Results
We will report:
- End-to-end speedups and FLOPs reductions on agentic benchmarks.
- Regime A vs. Regime B and multicut boundaries: memory vs. speed vs. acceptance.
- Quantization effects (int8 vs. int4) on quality and savings.
- Robustness across RoPE variants (NTK/YARN) and GQA/MQA.
- Adversarial left-context shifts: acceptance vs. drift curves; probe efficacy.

## 6. Analysis and Discussion
- Exactness: Guaranteed only when both the segment and the entire left context preceding the segment are identical across reused layers (RoPE reindexed). This is common for template-anchored spans in agent systems.
- Approximate reuse: With conservative depth-aware guards and (optional) probes, drift is bounded for templated prompts. Boundary-state reuse reduces memory but does not “fix” lower-layer context mismatch; it must be paired with the admission policy above.
- IO vs. compute: With corrected d_kv-based sizing, CPU↔GPU bandwidth is not the dominant bottleneck for 7B-class models on PCIe, and overlaps well with prefill.

## 7. Relation to Prior Work
- Prefix caching and paged KV (e.g., vLLM): We extend reuse to internal, shifted segments via content retrieval and RoPE reindexing.
- KV compression/pruning: Orthogonal; complementary to reduce memory.
- Speculative decoding: Orthogonal; targets decode, not prefill redundancy.
- Retrieval/caching beyond prefix (e.g., segment-level reuse, prompt caching systems): We provide a principled RoPE reindexing plus depth-aware admission and multicut storage for agent templates.

## 8. Limitations
- Cross-context reuse is approximate unless left context is identical; safeguards are required.
- RoPE-only; non-RoPE models need alternative reindexing.
- Storage grows with the number of cuts and segments; requires eviction and prefetch policies.
- Adapter/LoRA/state must match; we include adapter hashes in metadata.

## 9. Systems and Implementation Notes
- Storage: CPU memory with pinned pages; GPU staging buffers; asynchronous prefetch to paged KV.
- Kernels: Fused int8 dequant + delta-RoPE rotation; vectorized per KV head; supports MQA/GQA.
- Runtime: Thread-safe LSH; batched lookups; deterministic page mapping and read-only segments; metrics hooks.
- Metadata fidelity: include model weights hash, adapter/LoRA hashes, tokenizer hash, RoPE config, attention mask type.

## 10. Reproducibility and Artifact Plan
We will release:
- Reference implementation (cache manager, RoPE kernels, LSH index) for Llama/Mistral.
- Evaluation scripts, configs, and seeds.
- Calibration artifacts and instructions for A100/L4.
- Unit tests covering RoPE delta correctness under NTK/YARN and GQA/MQA.

## 11. Conclusion
Cross-query KV reuse with RoPE reindexing exploits repeated internal prompt segments to cut prefill compute in agentic LLMs. With a rigorous admission policy (exact and calibrated approximate modes), depth-aware reuse, and practical systems integration, it yields meaningful speedups on open models while preserving quality within calibrated bounds. We provide code to enable validation on small open-source models.
