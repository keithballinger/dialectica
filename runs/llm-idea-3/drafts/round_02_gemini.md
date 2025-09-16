Minor Revisions. The paper presents a novel and impactful method for accelerating LLM agent inference with a well-defined experimental plan. The core idea is sound. Revisions should focus on improving conciseness and precision in the abstract, introduction, and method sections, and adopting standard scientific paper section titles (e.g., "Experimental Design" instead of "falsification plan").

### Revised Draft
# Cross-Query KV Segment Reuse for Template-Heavy Agents

## Abstract
Large language model (LLM) agents repeatedly process similar prompt fragments, such as instructions or tool schemas, across iterative reasoning steps. We propose cross-query KV segment reuse, a method to reduce redundant computation by caching and reusing Key-Value (KV) activations for these fragments. Our approach uses shingled MinHash for content-aware retrieval of cached segments, allowing for approximate matches. It then splices these segments into new contexts, even at different positions, using RoPE reindexing for exact positional alignment of the Key cache and partial-layer reuse with a small "halo" recomputation at boundaries to ensure correctness. This technique generalizes exact-prefix caching to handle internal prompt fragments that vary slightly across queries. We outline an experimental design to validate the method on agentic benchmarks (HotpotQA, ToolBench-like tasks), predicting significant reductions in latency and compute with minimal impact on task success.

## Introduction
LLM agents that use tools or planning loops frequently encounter near-duplicate prompt structures across turns, including system preambles, tool schemas, and recurrent scaffolds like "Thought/Action/Observation". Current inference pipelines exploit repetition only through exact-prefix caching, which fails if a repeated fragment occurs later in the context or differs even slightly. Consequently, agents pay the full prefill cost for these recurring fragments in each step.

We introduce cross-query KV segment reuse, a mechanism to reuse and splice cached KV activations for similar prompt fragments across different prompts and positions. The technique combines three key components:
1.  **Content-Similarity Matching:** We use shingled MinHash to identify and retrieve cached KV segments that are approximately similar to segments in the current prompt, tolerating minor variations.
2.  **Positional Alignment:** We use RoPE reindexing to exactly shift the rotary position encodings of cached Key tensors to their new absolute positions. Value tensors are position-agnostic.
3.  **Correctness-Preserving Splicing:** We reuse KV activations for a subset of the model's lower layers and recompute a small "halo" of tokens at the segment boundaries. This approach preserves local dependencies within the segment while correctly integrating it into the new context.

Our method targets the practical bottleneck of template-heavy agents with modest variations between steps and is complementary to existing optimizations like batching, FlashAttention, and paged KV memory.

## Method
**Setting:** We target autoregressive decoder models with rotary positional embeddings (RoPE), such as Llama or Mistral, used in ReAct-style agent loops. The goal is to reduce prefill compute and latency by reusing KV activations for repeated prompt segments while preserving task success.

### Segment Discovery and Indexing
-   **Segmentation:** Token sequences are split at structural anchors (e.g., `\nThought:`, `\nAction:`) and punctuation. We also generate overlapping fixed-size windows (e.g., 64–128 tokens) to capture repeated phrases not aligned to anchors.
-   **Indexing:** For each candidate segment, we compute a MinHash signature from its k-gram shingles (e.g., k=8). We store these signatures in a locality-sensitive hash (LSH) index for fast approximate similarity search. We also compute and store a MinHash signature for the W-token left context (e.g., W=64) to ensure local contextual similarity, which strongly influences lower-layer attention.

### KV Cache Storage
Each cache entry for a segment includes:
-   **Model Metadata:** Architecture ID, RoPE base, hidden size, layer/head count.
-   **Token Metadata:** Token IDs and original absolute positions.
-   **KV Tensors:** For a subset of layers `L_reuse` (e.g., bottom 16 of 32), we store the K and V tensors. K is stored post-RoPE application. Tensors can be quantized (e.g., int8) to reduce memory.
-   **Signatures:** MinHash signatures for the segment and its left-context window.

### RoPE Reindexing
In RoPE, a cached K tensor, rotated at an original position `p_old`, can be exactly shifted to a new position `p_new` by applying the inverse rotation for `p_old` and then the forward rotation for `p_new`. This operation is linear in the number of cached tokens and requires precomputed sin/cos tables. V tensors are not rotated and require no adjustment.

### Splicing and Execution
For a new prompt:
1.  **Candidate Retrieval:** For each segment in the prompt, query the LSH index to find the top-M nearest cached segments based on Jaccard similarity. Filter candidates by a context-similarity threshold and length compatibility.
2.  **Fast Acceptance Test (Optional):** To guard against quality degradation, perform a small forward pass on the first and last `h` tokens of the segment. Compare the resulting logits against those produced with the reused KV. If the KL divergence exceeds a threshold, reject the candidate.
3.  **Layerwise Reuse with Halo Recomputation:** Select a layer subset `L_reuse` (e.g., bottom 8–16 layers) and a halo size `h` (e.g., 8 tokens). For the segment's interior, directly insert the reindexed K and V into the runtime caches for layers in `L_reuse`. For boundary "halo" tokens and all layers outside `L_reuse`, compute KV activations normally.
4.  **Fallback:** If no suitable candidate is found or an acceptance test fails, compute the segment from scratch.

### Complexity
-   **Memory:** A 128-token segment from 16 layers of a 7B model requires tens of MB with 8-bit quantization. An LRU eviction policy caps total memory usage.
-   **Runtime:** LSH retrieval is sub-millisecond. RoPE reindexing is linear in cached tokens. Savings scale with the fraction of reused tokens multiplied by the fraction of reused layers, amortizing overheads in agent loops.

## Experimental Design
We propose experiments to falsify claims about the method's performance and correctness.

**Tasks and Agents:** A ReAct agent with calculator and search tools will be evaluated on HotpotQA and a subset of ToolBench tasks. Decoding uses fixed seeds and low temperature (0.2) to measure stability.

**Conditions:**
-   **No Reuse:** Standard decoding baseline.
-   **Exact-Prefix Cache:** Caching only on identical prompt prefixes.
-   **KV Segment Reuse (Ours):** Our proposed method with `L_reuse=12`, halo `h=8`, context window `W=64`, and Jaccard threshold `τ=0.8`.
-   **Ablations:** We will vary `L_reuse`, `h`, `τ`, and `W`; compare int8 vs fp16 caching; and disable acceptance tests to measure their impact.

**Metrics:**
-   **Performance:** Wall-clock latency per agent step; FLOPs saved (proxied by skipped attention matmuls).
-   **Task Success:** EM/F1 for HotpotQA; tool success rate and final answer correctness for ToolBench.
-   **Stability:** Levenshtein distance of model outputs across reruns with fixed seeds.
-   **Quality Guard:** Average KL divergence at boundary probes; rate of fallbacks.

**Hypotheses:**
-   **H1 (Performance):** KV segment reuse reduces wall-clock latency by >20% over the No Reuse baseline with <2% absolute degradation in task success.
-   **H2 (Correctness):** Partial-layer reuse with small halos (h ≤ 8) maintains boundary KL divergence below a small threshold (e.g., 0.1 nats) on accepted segments.
-   **H3 (Retrieval):** The MinHash retrieval mechanism achieves >70% precision for identifying reusable segments at a Jaccard threshold `τ=0.8`.
-   **H4 (Robustness):** RoPE reindexing is numerically stable, and acceptance-test fallbacks effectively cap worst-case quality regressions.

## Discussion
**Mechanism:** The method works because lower-layer attention in transformers is dominated by local dependencies. By matching segments with similar local contexts, we can reuse these lower-layer activations effectively. RoPE reindexing provides exact positional alignment for Key tensors, while halo recomputation mitigates errors from cross-segment attention dependencies in higher layers.

**Relation to Prior Work:** This work extends exact-prefix caching to non-prefix, approximately matching segments. It is orthogonal to and compatible with hardware- and memory-focused optimizations like FlashAttention and paged KV management.

**Limitations:**
-   **Architecture:** The method relies on RoPE for exact reindexing; other position-encoding schemes may require approximations.
-   **Context Dependence:** Similarity over a finite context window `W` is an approximation of full-prefix dependence. Halos mitigate but do not eliminate this.
-   **Memory Footprint:** Caching requires significant memory, necessitating quantization and an eviction policy.
-   **Security:** Caches must not be shared across different users or trust domains to prevent data leakage.

## Conclusion
We propose cross-query KV segment reuse, a technique for accelerating template-heavy agent workflows by reusing KV activations for similar, non-prefix prompt fragments. The method combines MinHash-based retrieval, exact RoPE reindexing, and partial-layer reuse with boundary halos. Our proposed experimental plan provides a clear path to validate its effectiveness on standard agentic tasks using small open-source models. If validated, this technique offers a practical path to reduce the computational cost and improve the stability of LLM agents.
