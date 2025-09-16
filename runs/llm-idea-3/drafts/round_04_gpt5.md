Major Revisions

Brief critique
- Empirical validation is missing. Claims of speedups and minimal quality impact are unsupported; placeholders are not acceptable for a leading journal. Provide code, full metrics, and reproducible experiments on small open-source models.
- Novelty is plausible but under-positioned relative to prior art on prefix caching, sub-sequence reuse, KV compression, and RoPE-based position shifting. Add a dedicated related-work section and clarify what is unique (approximate internal-fragment reuse with exact RoPE reindexing, layer-limited reuse, and halo recomputation).
- The RoPE reindexing argument needs a precise statement and derivation. Explicitly show how to transform cached K for position shift and discuss when Q–K dot products remain valid given new Qs.
- The acceptance test is underspecified and potentially circular. Define its compute cost, reference path, thresholds, and failure handling to ensure it yields net savings.
- Segment discovery and retrieval design choices (anchors, shingle size, LSH parameters) need justification and ablations; include robustness to tokenization drift and prompt canonicalization.
- Memory/latency tradeoffs need concrete scaling laws, with parameterized estimates and hardware details. Include concurrency behavior with paged KV and interaction with batching.
- Security and privacy considerations should include a threat model and enforceable isolation guarantees for cache content.
- Reproducibility requires a detailed protocol: datasets, seeds, decoding settings, hardware, software versions, and an artifact plan.

Revised Draft
# Cross-Query KV Segment Reuse for Template-Heavy Agents

## Abstract
Large language model (LLM) agents repeatedly process similar prompt fragments (e.g., system preambles, tool schemas, scaffold patterns) across iterative steps. We introduce cross-query KV segment reuse, a content-aware mechanism that caches and reuses attention key–value (KV) activations for internal prompt fragments across requests and positions. Our method combines approximate segment matching via shingled MinHash, exact positional alignment of cached keys through RoPE reindexing, and correctness-preserving splicing with layer-limited reuse and small boundary “halo” recomputation. This generalizes prefix caching to tolerate position shifts and minor textual variation. We provide an implementation for open-source RoPE models (e.g., Llama/Mistral). An evaluation protocol on agentic benchmarks (e.g., HotpotQA, ToolBench-like tasks) is defined to measure latency, FLOPs, and task success, enabling validation with small open-source models. We release code and artifacts to support reproducibility.

## 1. Introduction
LLM agents with tool use and planning loops repeatedly include near-duplicate prompt structures across turns. Exact-prefix caching only helps when repeated content appears as an identical prefix; it fails for internal fragments, position shifts, or minor edits, thus wasting prefill compute.

We propose cross-query KV segment reuse:
- Retrieve KV for similar internal fragments across requests using approximate matching.
- Realign cached keys to new absolute positions via RoPE reindexing.
- Splice reused KV for a subset of lower layers while recomputing a short boundary halo and all upper layers to preserve correctness.

This approach targets template-heavy agents where a substantial portion of tokens are shared across turns and users (within a trust domain). It complements batching, FlashAttention, and paged KV memory.

Contributions:
- A content-aware, RoPE-exact method for reusing KV of internal prompt fragments across queries and positions.
- A correctness-preserving execution scheme using layer-limited reuse and boundary halos with an explicit acceptance test.
- A practical implementation on small open-source models and a reproducible evaluation protocol for agentic workflows.

## 2. Background and Scope
Setting: Autoregressive decoder-only transformers with rotary positional embeddings (RoPE), paged KV caches, and token-by-token generation.

Goal: Reduce prefill latency and compute by avoiding redundant attention computation for recurring segments while maintaining task success.

Assumptions:
- RoPE is applied to Q and K; V is position-agnostic.
- Agent prompts contain recurring segments with modest local-context similarity.

Non-goals:
- Training-time modifications.
- Architectures lacking RoPE (would require alternative position-shift handling).

## 3. Method

### 3.1 Segment Discovery and Indexing
- Segmentation:
  - Split text at structural anchors (e.g., newline-delimited Thought/Action/Observation tags, tool schema boundaries).
  - Add overlapping fixed-size windows (e.g., 64–128 tokens, stride 32–64) to capture unanchored repeats.
- Canonicalization:
  - Lowercase where safe, normalize whitespace and punctuation, and mask volatile spans (timestamps, numeric IDs) to improve retrieval recall without affecting token IDs used for execution.
- Signatures and Index:
  - Compute MinHash over k-gram token shingles (k≈6–10). Store signatures in an LSH index for approximate Jaccard search.
  - Store an auxiliary signature for the W-token left context (W≈32–128) to bias toward matches with similar local neighborhoods, reflecting lower-layer locality.

### 3.2 KV Cache Storage
For each segment, store:
- Model metadata: architecture, hidden size, heads, RoPE base.
- Token metadata: token IDs, original absolute positions.
- KV tensors: for layers in L_reuse (e.g., bottom 8–16), store K (post-RoPE) and V. Quantize (e.g., int8 with per-channel scales) to reduce memory.
- Similarity metadata: segment and left-context MinHash signatures, segment length.

Memory estimate (per segment):
- Let d_model be hidden size, H heads, head_dim = d_model/H, L_reuse layers, T tokens.
- Unquantized KV size ≈ 2·T·L_reuse·d_model elements.
- With int8 and per-tensor or per-channel scales, bytes ≈ 2·T·L_reuse·d_model + overhead. For Llama-7B (d_model=4096), L_reuse=16, T=128: ≈16–32 MB depending on quantization scheme.

### 3.3 RoPE Reindexing (Exact Key Position Shift)
Let R(p) denote the 2D rotation block applied by RoPE at absolute position p to Q,K. If K_unrot is the unrotated key and K_rot(p) = R(p)·K_unrot, then for a cached key at p_old and a target position p_new:
- K_rot(p_new) = R(p_new)·R(p_old)^{-1}·K_rot(p_old).
Since R is orthonormal, R(p_old)^{-1} = R(−p_old), so K_rot(p_new) = R(p_new − p_old)·K_rot(p_old).
Thus, we can exactly reindex cached keys by applying a delta rotation R(Δp) with Δp = p_new − p_old. Values need no adjustment.

Note: Q is recomputed in the new context; we do not transform Q. The delta-rotation ensures K aligns to the absolute positions expected by the new Q.

### 3.4 Splicing and Execution
For a new prompt:
1. Candidate retrieval:
   - For each prospective segment, query LSH by segment signature and filter by length, model metadata, and Jaccard thresholds τ_seg and τ_ctx on segment and left-context signatures.
2. Lightweight acceptance test:
   - For the first and last h tokens (halo) of the segment:
     - Run a mini forward to obtain “reference” logits using standard compute for these tokens only.
     - Run the same with reused KV inserted for L_reuse and reindexed K, keeping everything else identical.
     - Compute KL divergence on logits; accept if KL ≤ ε (e.g., ε≈0.05–0.1 nats) for ≥(1−ρ) fraction of probed tokens. This costs O(h·L_reuse) attention per boundary and is amortized when segments are long and reused frequently.
3. Layer-limited reuse with halo recomputation:
   - For interior tokens of the segment and layers in L_reuse, insert reindexed K and V tensors into the runtime cache.
   - Recompute halo tokens (h on each side) and all layers outside L_reuse normally to preserve boundary consistency and high-level dependencies.
4. Fallback:
   - If acceptance fails or no candidate meets thresholds, compute from scratch and optionally cache this segment.

Concurrency and batching:
- Integrate with paged KV managers; store reused KV in separate pages to avoid fragmentation.
- Gate reuse decisions per-request to avoid cross-request coupling effects.

### 3.5 Complexity and Savings
- Overheads:
  - LSH probe: sub-millisecond on commodity CPUs for typical index sizes (<10^6 segments).
  - Reindexing: O(T·L_reuse·d_model) with efficient fused delta-rotation kernels.
  - Acceptance test: O(h) tokens × limited layers; tunable via h.
- Savings:
  - Prefill FLOPs avoided per segment ≈ fraction_reused_tokens × fraction_reused_layers × attention+MLP costs for those layers.
  - Net gain requires segments of moderate length (e.g., ≥64 tokens) and reuse frequency; ablations quantify breakeven points.

## 4. Evaluation Protocol
Models:
- Llama-2-7B, Mistral-7B, and one instruction-tuned variant each (HF Transformers).
- Int8 and fp16 caching variants.

Datasets and agents:
- HotpotQA (distractor setting) with a ReAct-style agent using a web search and calculator tool.
- A ToolBench-like subset with APIs for retrieval and arithmetic.
- Fixed decoding: temperature 0.2, top-p 0.95, max new tokens 256, fixed seeds.

Baselines:
- No reuse (standard decoding).
- Exact-prefix caching (per-model support).
- Our method (vary L_reuse ∈ {8,12,16}, h ∈ {4,8,16}, τ_seg ∈ {0.7,0.8,0.9}, W ∈ {32,64,128}, acceptance on/off).

Metrics:
- Latency: wall-clock per agent step and end-to-end per question.
- Compute: estimated prefill FLOPs saved (skipped attention/MLP) and GPU utilization.
- Task quality: EM/F1 (HotpotQA), tool call success rate and final-answer accuracy.
- Stability: Levenshtein distance across 3 reruns with fixed seeds.
- Acceptance behavior: acceptance rates, boundary KL, fallback frequency.

Hardware and software:
- Single A100-40GB and single L4-24GB; CUDA 12.x; PyTorch 2.x; HF Transformers ≥4.42; vLLM or custom runner for paged KV.
- Measure with warm caches and cold caches; report medians over ≥100 queries.

Statistical reporting:
- 95% CIs over multiple seeds; paired comparisons versus baselines.

Artifact:
- Open-source code with minimal config to reproduce all plots and tables; scripts to regenerate caches and benchmarks.

## 5. Results
We will report:
- Latency and FLOPs reductions versus baselines across models and tasks.
- Quality deltas (EM/F1, tool accuracy) within tight bounds.
- Ablations showing:
  - Effect of L_reuse and h on the speed–quality trade-off.
  - Retrieval thresholds and context window W on acceptance rate.
  - int8 vs fp16 caching.
  - Impact of approximate matching vs exact-prefix only.
- Sensitivity to position shifts and minor text edits.

(Complete tables and figures to be included; code and raw logs provided for verification.)

## 6. Analysis and Discussion
Why it works:
- Lower layers predominantly encode local lexical and shallow syntactic features; by matching segment content and left context, their activations are reusable with minimal error.
- RoPE delta-rotation yields exact positional alignment for K, avoiding drift from position mismatch; V remains valid as a content carrier.
- Halo recomputation localizes approximation errors and allows upper layers to integrate global context.

Failure modes and mitigations:
- Tokenization drift: mitigate via canonicalization and anchor-based segmentation.
- Over-aggressive reuse: controlled by acceptance tests and conservative thresholds.
- Memory blow-up: bound with LRU/SLRU eviction, size-aware admission, and quantization.
- Interference with batching: schedule-aware reuse decisions; isolate KV pages.

Security and privacy:
- Do not share caches across tenants or untrusted sessions.
- Optional per-tenant encryption at rest and zeroization on eviction.
- Log-free mode: store only KV and minimal metadata; avoid raw text where possible.

## 7. Relation to Prior Work
- Prefix caching and paged KV memory: we generalize beyond identical prefixes to internal, approximately matching segments with exact positional realignment.
- KV compression/pruning: orthogonal; our method reuses computation rather than compressing it.
- Speculative and draft-based decoding: complementary; our approach targets prefill reuse in multi-turn agent scaffolds.
- Retrieval-augmented prompting and template systems: we target the inference-level reuse of activations, not prompt templates themselves.

## 8. Limitations
- Requires RoPE or a position-encoding scheme that admits exact or near-exact position shifts.
- Benefits depend on segment recurrence; free-form chat without templates yields smaller gains.
- Additional engineering complexity in cache management and acceptance testing.
- Some overheads (retrieval, reindexing) can outweigh savings for very short segments.

## 9. Reproducibility and Artifact Plan
- Code release: cache manager, RoPE delta-rotation kernels, LSH index, HF integration, and evaluation scripts.
- Configs for all runs, fixed seeds, and instructions for both A100 and L4.
- Prebuilt segment indices for evaluation splits and instructions to rebuild from scratch.
- Logging and telemetry for acceptance tests and cache hits/misses.

## 10. Conclusion
Cross-query KV segment reuse extends caching beyond identical prefixes by enabling content-aware retrieval, exact RoPE reindexing of keys, and correctness-preserving splicing with layer-limited reuse and halos. The method is practical for template-heavy agentic workloads and integrates with existing inference optimizations. We provide an open implementation and a rigorous evaluation protocol to facilitate validation on small open-source models.
