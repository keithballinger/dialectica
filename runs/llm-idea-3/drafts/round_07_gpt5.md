Major Revisions

Brief critique
- Core methodological flaw: Layer-limited reuse as written is internally inconsistent. Later tokens require K/V of all previous tokens at every layer. If you only cache K/V for lower layers, you must either (a) recompute the lower layers’ hidden states for all segment tokens to proceed to higher layers (negating the savings), or (b) store boundary hidden states to jump to upper layers. The current draft stores only K/V for lower layers and no boundary states, so the execution path cannot be correct as described.
- Acceptance test is not costed correctly. Comparing “reference” versus “reuse” logits for halo tokens requires K/V for the entire segment under the “no reuse” path, which costs O(segment) and defeats the purpose. A practical acceptance test must avoid recomputing the segment.
- RoPE details are underspecified. You must handle per-head frequencies, NTK/YARN scaling, rope base mismatches, and MQA/GQA head sharing. Specify whether K is stored pre- or post-RoPE per layer and how delta rotation is applied in each variant.
- Context dependence is underplayed. K/V beyond the first layer depend on left context via attention and layernorm. Exact reuse across contexts is not guaranteed; you need either (1) full-stack K/V reuse or (2) boundary-state checkpoints enabling recomputation of upper layers, plus conservative matching thresholds and offline calibration. Make these assumptions explicit.
- Evaluation missing. To be publishable, include end-to-end latency/throughput, FLOPs, accuracy, and stability across at least Llama-2-7B/Mistral-7B on concrete agent workflows, with competitive baselines (prefix caching across requests, paged KV, batching, FlashDecoding). Include ablations on memory-accuracy trade-offs (quantization levels, cache size/eviction, segment length), and rope variants.
- Systems details needed. Clarify cache placement (CPU/GPU), prefetch/swap policy, page mapping into paged KV managers (e.g., vLLM-like), concurrency and isolation, and memory budgets. Quantify per-segment memory with int8/int4 and show break-even reuse frequency.

Revised Draft
# Cross-Query KV Reuse with RoPE Reindexing for Template-Heavy Agent Prompts

## Abstract
Agentic LLM workflows repeatedly process similar internal prompt fragments (system preambles, tool schemas, chain-of-thought scaffolds). We present a cross-query KV reuse mechanism that identifies repeated segments across requests, realigns their attention keys under RoPE to new positions, and splices the cached per-layer K/V directly into the runtime KV cache to avoid recomputing the segment. We detail two execution regimes: (A) full-stack KV reuse (store per-layer K/V; zero recompute for the segment) and (B) boundary-state reuse (store per-layer K/V for lower layers plus layer-boundary hidden states to enable recomputation of upper layers). We provide a practical design for RoPE reindexing under common variants (NTK/YARN scaling, MQA/GQA), a conservative, runtime-free acceptance policy based on content and left-context similarity, and a system that integrates with paged KV managers. On Llama/Mistral 7B-class models, our implementation targets 20–40% prefill FLOPs reduction on template-heavy agents with negligible quality change. We release code for validation on small open-source models.

## 1. Introduction
LLM agents routinely include near-duplicate internal structures across steps and users. Existing prefix caching exploits only identical leading spans. It fails for internal repeats, shifted positions, or minor edits, leaving substantial prefill compute on the table.

We propose cross-query KV reuse:
- Retrieve previously computed per-layer K/V for repeated internal segments via content-aware indexing.
- Realign cached keys to new absolute positions exactly under RoPE via a delta rotation.
- Splice K/V into the runtime cache to eliminate recomputation of those tokens.

Key idea: Later tokens only need previous tokens’ K/V at the same layer; they do not need the previous tokens’ hidden states. Thus, if per-layer K/V for a segment are available, we can skip compute for that segment and let subsequent tokens attend to it as usual.

Contributions:
- A correct execution design for cross-query, cross-position KV reuse, including two regimes (full-stack and boundary-state) with explicit compute–memory trade-offs.
- Exact RoPE reindexing for K across layers and heads, covering NTK/YARN scaling and MQA/GQA.
- A conservative, low-overhead acceptance policy that avoids runtime recomputation.
- An open implementation on Llama/Mistral 7B-class models, with an evaluation plan on agentic tasks.

## 2. Background and Scope
We consider decoder-only transformers with rotary positional embeddings (RoPE), paged KV caches, and token-by-token decoding.

Observations:
- Attention at layer ℓ for token t requires K/V for all tokens < t at the same layer ℓ. It does not require their hidden states.
- K/V for a token at layer ℓ depend on its local residual stream, which is influenced by left context via attention/LN in earlier layers. Hence, KV reuse across contexts is approximate unless left context is similar.

Non-goals:
- Training changes. Our method is inference-only.
- Architectures without RoPE (would need alternate reindexing).

## 3. Method

### 3.1 Segment Discovery and Indexing
- Segmentation:
  - Extract structural spans (e.g., tool schemas, “Thought/Action/Observation” blocks).
  - Add overlapping fixed windows (64–128 tokens, stride 32–64) to capture unanchored repeats.
- Canonicalization (for indexing only):
  - Lowercase, normalize whitespace, mask volatile substrings (timestamps/IDs) with placeholder tokens. Execution uses true token IDs.
- Signatures and Index:
  - Compute MinHash over token shingles (k≈6–10). Store in LSH for approximate Jaccard search.
  - Store auxiliary signature for W-token left context (W≈64–128) to favor matches with similar neighborhoods.

### 3.2 What to Cache
We provide two options with different compute–memory trade-offs.

- Regime A: Full-stack KV reuse (max speedup)
  - For each segment and each layer ℓ:
    - Store Kℓ after RoPE and Vℓ (unrotated).
  - At reuse time, later tokens will attend to these per-layer K/V; the segment is not recomputed at any layer.

- Regime B: Boundary-state reuse (lower memory)
  - Store:
    - Kℓ after RoPE and Vℓ for layers ℓ ∈ [0, L_reuse) only (lower layers).
    - The residual hidden state h_{L_reuse} for each token at the boundary (after layer L_reuse).
  - At reuse time:
    - Skip compute for layers [0, L_reuse) for the segment (use cached K/V).
    - Resume from stored h_{L_reuse} and compute layers ℓ ∈ [L_reuse, L_total) for the segment to produce K/V needed by later tokens at upper layers.

Notes:
- Quantization: Store int8 (or int4) with per-channel/group scales. Int8 is default.
- Metadata: model ID, d_model, n_heads, n_kv_heads, RoPE base/scaling, tokenizer hash, segment token IDs, original absolute positions.

Memory per token (L layers, d_model dims; int8):
- Regime A: ≈ 2 · L · d_model bytes.
- Regime B: ≈ (2 · L_reuse + 1) · d_model bytes.
Example (Llama-7B: d_model=4096, L=32, L_reuse=12):
- Regime A: ≈ 262 KB/token; 128-token segment ≈ 33.5 MB.
- Regime B: ≈ 106 KB/token; 128-token segment ≈ 13.6 MB.

### 3.3 RoPE Reindexing (Exact Key Position Shift)
We store K after RoPE to enable exact reindexing by a delta rotation per head and frequency.

Given RoPE rotation Rℓ,h(p) applied at layer ℓ, head h, position p (includes base θ, per-dim frequencies, and any NTK/YARN scaling):
- For cached K at old position p_old and new position p_new:
  - Kℓ,h(p_new) = Rℓ,h(p_new) · Rℓ,h(p_old)^{-1} · Kℓ,h(p_old) = Rℓ,h(p_new − p_old) · Kℓ,h(p_old).
Implementation:
- Support Llama/Mistral-style interleaved 2D rotations.
- Respect NTK scaling/YARN: compute the same θ(d) used by the target model for the new absolute position.
- MQA/GQA: Apply per-KV-head rotation per shared KV head; Q is computed fresh.

Values V are position-agnostic and require no rotation.

### 3.4 Reuse and Execution
For a new prompt:
1. Candidate retrieval:
   - For each potential segment, query the LSH by segment signature.
   - Filter by model metadata, token length, Jaccard thresholds τ_seg and τ_ctx on segment and left-context signatures, and token-ID exact match rate ≥ ρ (e.g., 0.9).
2. Admission (no runtime recompute):
   - We use conservative static thresholds (τ_seg, τ_ctx, ρ) derived from offline calibration (Section 4) so that admitted segments have bounded drift without per-request recomputation.
3. Splicing:
   - Map the segment to absolute positions in the new prompt.
   - For each layer ℓ:
     - Regime A: Load Kℓ,Vℓ for segment tokens, apply per-head delta RoPE Rℓ,h(Δp), and insert into the runtime paged KV cache at the target positions.
     - Regime B: For ℓ < L_reuse, do as above. For ℓ ≥ L_reuse, reconstruct Kℓ,Vℓ by running layers ℓ ≥ L_reuse only, starting from stored h_{L_reuse}.
   - Skip all forward compute for the segment at layers where K/V are provided by cache.
   - Compute the remainder of the prompt normally; later tokens attend to cached K/V.
4. Fallback:
   - If no candidate passes thresholds, compute from scratch and optionally add the segment to the store.

### 3.5 Complexity and Savings
- Overheads:
  - LSH probe: sub-millisecond on CPU.
  - Reindex + insert: O(T · L_used · d_model) bytewise ops; fused per-layer/head kernels amortize well.
  - Regime B adds compute for upper layers across the segment; still saves lower-layer FLOPs.
- Savings:
  - Regime A eliminates all per-layer compute for the segment (KV projections, attention, MLP).
  - Break-even reuse frequency f* occurs when saved FLOPs – (I/O + reindex cost) ≥ 0; we report f* empirically.
- Integration:
  - Works with batching, FlashAttention/FlashDecoding, and paged KV managers (segment pages are pre-filled and marked read-only).

## 4. Calibration and Evaluation Protocol
Offline calibration for admission thresholds:
- Datasets: Agent scaffolds on HotpotQA (distractor) with ReAct; ToolBench-like flows; synthetic templated workflows.
- Procedure:
  - For a held-out set, measure logit-KL and answer accuracy drift when reusing segments admitted under varying τ_seg, τ_ctx, ρ and quantization levels.
  - Choose thresholds ensuring ≤0.5% EM drop with high acceptance.
- Models: Llama-2-7B, Mistral-7B, instruct variants.
- Baselines:
  - No reuse; cross-request exact prefix caching; paged KV only; batching+FlashAttention; speculative decoding (orthogonal).
- Metrics:
  - Wall-clock latency; tokens/s; prefill FLOPs; GPU util.
  - Task quality: EM/F1 (HotpotQA); tool success/answer accuracy.
  - Stability: output consistency across seeds.
  - Cache: acceptance rate, hit frequency, memory footprint, swap/prefetch overheads.
- Hardware: A100-40GB and L4-24GB; CPU pinned-memory for cache.

## 5. Results
We will report:
- End-to-end speedups and FLOPs reductions versus baselines on agentic benchmarks.
- Regime A vs Regime B trade-offs: memory vs speed.
- Quantization effects (int8 vs int4) on quality and savings.
- Ablations on τ_seg, τ_ctx, ρ, segment length, and cache size/eviction.
- RoPE variant robustness (NTK/YARN).
[We will include code, configs, and full tables/figures in the camera-ready.]

## 6. Analysis and Discussion
- Correctness: Under identical tokens and similar left context, lower-layer K/V vary little; full-stack reuse is empirically robust for templated spans. Boundary-state reuse protects quality while reducing memory.
- Failure modes:
  - Context drift: mitigated by conservative admission thresholds and offline calibration.
  - Memory pressure: addressed via quantization, LRU eviction, CPU store + GPU prefetch.
  - Tokenization drift: reduce via canonicalization for indexing; exact token-ID match filter.
- Security/Privacy:
  - Per-tenant stores; encryption at rest; no cross-tenant sharing; zeroization on eviction.

## 7. Relation to Prior Work
- Prefix caching and paged KV (e.g., vLLM): Our method generalizes to internal, shifted segments across queries by content-aware retrieval and RoPE reindexing.
- KV compression/pruning: Orthogonal; we avoid compute by reusing precomputed K/V. Can be combined to reduce memory.
- Speculative/draft decoding: Orthogonal; targets decode, not prefill redundancy.
- Architectural variants (MQA/GQA, FlashDecoding): Compatible; we rotate per KV head, reuse V unchanged, and integrate with flash kernels.

## 8. Limitations
- Approximate across contexts: exactness is not guaranteed if left context differs; we rely on conservative admission and calibration.
- RoPE-only: Non-RoPE models need alternate reindexing.
- Memory: Full-stack reuse can be tens of MB per 128-token segment even with int8; requires careful eviction/prefetch.

## 9. Systems and Implementation Notes
- Storage: CPU memory with pinned pages; GPU-side staging buffers; asynchronous prefetch into paged KV.
- Kernels: Fused per-layer/head delta RoPE rotation; vectorized int8 dequant.
- Runtime: Thread-safe LSH; batched lookups; deterministic page mapping for KV splicing; metrics hooks.

## 10. Reproducibility and Artifact Plan
We will release:
- Reference implementation (cache manager, RoPE kernels, LSH index) for Llama/Mistral.
- Evaluation scripts, configs, and seeds.
- Prebuilt calibration artifacts and instructions for A100/L4.

## 11. Conclusion
Cross-query KV reuse with RoPE reindexing exploits repeated internal prompt segments to cut prefill compute in agentic LLMs. With a correct execution model (full-stack or boundary-state reuse), conservative admission, and practical systems integration, it offers meaningful speedups on open models while preserving quality.
