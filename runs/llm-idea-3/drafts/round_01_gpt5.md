Major Revisions

Title
Cross-Query KV Segment Reuse for Template-Heavy Agents

Abstract
Large language model (LLM) agents repeatedly process similar prompt fragments across iterative loops (e.g., ReAct-style “Thought → Action → Observation” turns). We propose cross-query KV segment reuse: content-similarity-driven retrieval and splicing of cached key–value (KV) activations for repeated prompt fragments, even when they occur at different positions and within slightly different contexts. The method uses shingled MinHash to match segments, RoPE reindexing to align positions, and partial-layer reuse with a small “halo” recomputation at segment boundaries to maintain correctness. Unlike exact-prefix caching, our approach matches and grafts KV blocks for internal prompt fragments across turns and tasks. We outline a falsification plan using small open-source models in a ReAct agent with tools on HotpotQA and ToolBench-like tasks, measuring latency, compute, and task success against no-reuse and naive prompt caching. The method is implementable with modest engineering effort and makes specific, testable predictions about performance, robustness, and stability of agent behavior.

Introduction
LLM agents in tool-use and planning loops repeatedly encounter near-duplicate prompt structure across steps: system preambles, tool schemas, scratchpad delimiters, and recurrent “Thought/Action/Observation” scaffolds. Today’s inference pipelines do little to exploit this repetition beyond exact-prefix caching, which only triggers when a new query starts with an identical token prefix. As a result, agents pay full prefill cost for repeated fragments that appear later in the context, or that differ slightly in content or position.

We introduce cross-query KV segment reuse: a mechanism to reuse and splice cached key–value activations for similar prompt fragments across different prompts and positions. The key ideas are:
- Content similarity, not exact identity: approximate match of token segments by shingled MinHash.
- Position alignment via RoPE reindexing: invert and reapply rotary position encodings so cached K can be shifted to new absolute positions exactly; V is unaffected by RoPE.
- Partial-layer reuse and boundary halos: reuse selected layers’ KV for the matched segment and recompute a small boundary band of tokens to absorb cross-segment dependencies.
- Guardrails: fast acceptance tests and fallbacks prevent quality regressions.

This targets a practical gap: template-heavy agents with modest variations across steps. The approach is complementary to batching, FlashAttention, paged KV memory, and exact-prefix caching.

Method
Setting and notation
- Model: autoregressive decoder with rotary positional embeddings (RoPE), e.g., Llama- or Mistral-class models.
- Agent loop: ReAct-style chains with tool calls on tasks such as HotpotQA and ToolBench-like tool suites.
- Goal: Reduce prefill compute and latency while preserving task success, by reusing KV for repeated segments.

Segment discovery and indexing
- Segmentation: Split token sequences at agent scaffolding anchors (e.g., “\nThought:”, “\nAction:”, bracketed tool arguments) and punctuation. Also form overlapping fixed-size windows (e.g., length 64–128) to capture repeated phrases not aligned to anchors.
- Shingled MinHash: For each candidate segment, compute k-gram shingles (e.g., k=8), then a MinHash signature (e.g., 128 permutations). Store a locality-sensitive hash (LSH) index for approximate Jaccard retrieval.
- Context signature: Also compute MinHash for the W-token left context (e.g., W=64) to ensure local context similarity, which empirically dominates attention in lower layers.

KV cache entries
For each stored segment occurrence:
- Model metadata: architecture ID, RoPE base, hidden size, number of layers/heads.
- Token metadata: token IDs, original absolute positions (start index), length.
- Layer subset: the set of layers L_reuse to reuse (e.g., bottom 16 of 32), chosen via ablation.
- KV tensors: per layer in L_reuse, K and V for all heads and segment token positions. Store K after RoPE application; V as produced. Optionally quantize (e.g., 8-bit per channel) to bound memory.
- Checksums: MinHash signatures for segment tokens and left-context window.

RoPE reindexing (exact for K)
- In standard RoPE, K and Q are rotated by a position-dependent complex (or 2D) rotation before dot-product. If we store K after rotation at original position p_old, we can recover the unrotated K by applying the inverse rotation for p_old, then apply the rotation for the new position p_new. This yields an exact position shift for K. V is not RoPE-rotated and needs no adjustment.
- Implementation detail: Precompute sin/cos tables; for each head, apply inverse rotation blockwise to cached K, then apply new rotation for p_new. This is linear-time in the number of cached tokens.

Splicing and execution
Given a new prompt with tokens t[0..N):
1) Candidate retrieval
- For each discovered segment S at indices [i, j), query LSH for top-M nearest cached segments by Jaccard over segment shingles. Filter by context-similarity threshold over W-token left context signatures. Require length compatibility within ±r tokens to permit small edits.
2) Fast acceptance tests
- Optional boundary probe: compute from-scratch forward pass for the first h and last h tokens of S at a small number of top layers (or just logits at j−1). With reused KV filled, compare logits against full compute for those boundary tokens on a tiny micro-batch. If mean KL divergence exceeds a threshold, reject candidate.
3) Layerwise reuse with halo recomputation
- Choose L_reuse (e.g., bottom 8–16 layers) and halo size h (e.g., 8 tokens). For tokens in the interior [i+h, j−h), skip full-layer computation and directly insert reindexed K,V for layers in L_reuse into the runtime caches. For boundary bands and for layers outside L_reuse, compute normally.
- This maintains cross-attention between the segment and its neighbors while avoiding most compute within the interior.
4) Fallbacks
- If no candidate passes, compute normally. If any numerical checks fail (e.g., NaNs after reindexing), clear and recompute.

Memory and complexity
- Memory: KV per token is O(layers × heads × head_dim). For 7B-class models, storing 16 layers for 128 tokens at 8-bit quantization is tens of MB per segment. Use an LRU eviction policy and segment-level deduplication by MinHash to cap memory.
- Runtime: Retrieval (LSH) is sub-millisecond; RoPE reindexing is linear in cached tokens; the main savings scale with fraction of reused tokens × fraction of reused layers. Overheads are amortized in agent loops with many repeated fragments.

Implementation details
- Framework: PyTorch with Hugging Face Transformers; hook attention modules to read/write per-layer caches.
- Tokenization: model tokenizer; ensure consistent special tokens for anchors.
- LSH/MinHash: libraries such as datasketch; store indices in-memory.
- Quantization: per-channel symmetric int8 with offline scale/zero-point per KV tensor; dequantize on splice.
- Models: Llama-2-7B, Mistral-7B-Instruct, Qwen-7B; all RoPE-based.

Experiments (falsification plan)
Tasks and agents
- ReAct agent with calculator and search tools.
- Datasets: HotpotQA (distractor setting) and a ToolBench-like suite (retrieved subsets sufficient to exercise tools).
- Decoding: temperature 0.2, top-p 0.95, max new tokens per step capped; fixed seeds to measure determinism.

Conditions
- No Reuse: standard decoding, no cross-request cache.
- Exact-Prefix Cache: cache only if exact prefix identity across requests; typical for batch reruns, weak for agent loops.
- Naive Prompt Cache: text-substring matching without KV graft; recompute KV; serves as upper bound on match recall but no speed gain.
- KV Segment Reuse (ours):
  - Ours-base: reuse bottom 12 layers, halo h=8, W=64 context window, Jaccard threshold τ=0.8, segment length 64–256.
  - Ablations: vary L_reuse ∈ {8,12,16,24}, h ∈ {0,4,8,16}, τ ∈ {0.6,0.7,0.8,0.9}, W ∈ {32,64,128}; try int8 vs fp16 caches; disable acceptance tests; ALiBi models (if any) to test portability.

Metrics
- Wall-clock latency per agent step and per episode.
- FLOP proxy: count of attention matmuls skipped (tokens × layers × heads × head_dim).
- Task success: EM/F1 for HotpotQA; tool success rate and final-answer correctness for ToolBench-like tasks.
- Output stability: Levenshtein distance and token-level entropy across 5 reruns (fixed seeds, minor sampling noise).
- Quality guard: average KL divergence at boundary probes when reuse is accepted; distribution of probe failures and fallbacks.

Hypotheses that can be falsified
- H1: In template-heavy agents, KV segment reuse reduces wall-clock latency by at least 20% relative to No Reuse, without degrading task success beyond 1–2% absolute.
- H2: Partial-layer reuse with small halos (h ≤ 8) maintains boundary KL below 0.1 (nats) on accepted segments.
- H3: MinHash+context signatures achieve >70% precision for reusable segments at τ=0.8; precision–recall tradeoff can be tuned.
- H4: Reindexing is numerically stable for RoPE models; acceptance-test fallbacks cap worst-case quality regressions.

Analysis and ablations
- Plot speedup vs fraction of segment tokens reused; study diminishing returns with larger L_reuse.
- Identify failure modes: long-range dependencies (W too small), noisy segments (low Jaccard), and mismatch across tool descriptions.
- Compare stability: reused KV should reduce within-run variance in internal scratchpad tokens up to the boundary halos.

Discussion
Why this works
- Agent prompts have strong local regularities: repeated scaffolds and short-distance dependencies dominate lower-layer attention. By matching segments and ensuring left-context similarity, lower-layer KV carries over well. RoPE reindexing makes position alignment exact for K. Halo recomputation handles cross-segment interactions.
Relation to prior work
- Extends exact-prefix caching beyond the prefix and across requests by content similarity.
- Orthogonal to FlashAttention and paged KV memory: we reduce compute, not just memory bandwidth.
- Related in spirit to partial-layer recomputation and activation reuse; differs by content-driven selection and precise RoPE position shifting.
Engineering impact
- Drop-in speedups for tool-using agents and RAG chains with repeated instructions.
- Stabilizes agent behavior by reusing internal states for boilerplate fragments, helpful for evaluation and regression testing.

Limitations
- Architectural scope: relies on RoPE to enable exact K reindexing; ALiBi and other positional schemes require different adjustments and may be less accurate.
- Context dependence: KV for a segment still depends on its full prefix; enforcing similarity over a finite window W is an approximation. Halo recomputation mitigates but does not eliminate mismatch.
- Memory footprint: Storing multi-layer KV for many segments is expensive; requires quantization, LRU eviction, and careful scoping of what to cache.
- Security and safety: Reusing internal states across requests must respect isolation boundaries; do not share caches across users or trust domains.
- Quality guardrails: Acceptance tests add overhead; overly strict thresholds reduce hit rate, overly lax thresholds risk quality.

Conclusion
We propose cross-query KV segment reuse: approximate matching and layerwise grafting of cached KV blocks for similar prompt fragments in template-heavy agent loops. Using MinHash-based retrieval, RoPE reindexing, partial-layer reuse, and boundary halos, the method is implementable with small open-source models and standard toolchains. A clear falsification plan on HotpotQA and ToolBench-like tasks evaluates speed, compute savings, task success, and stability against strong baselines. If validated, this technique offers a practical, novel route to reduce inference cost and stabilize agentic workflows beyond what exact-prefix caching can deliver.
