You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.
        Field: general

        Constraints of Paper:
        From: constraints/llm.md

- Research focused on Large Language Model inference
- Very impactful on quality, performance, or agentic workflows
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models


        Draft:
        Minor Revisions

Brief critique
- The primary revision required is the execution of the proposed evaluation and the inclusion of empirical results. The current draft is a proposal, not a complete paper.
- The acceptance test in Section 3.4.2 needs a more precise operational definition. Clarify how the "reference" and "reused" logits are generated and compared without incurring significant overhead that negates the savings.
- The canonicalization step (3.1.2) should briefly specify the mechanism for masking volatile spans to establish its practicality.
- The "Relation to Prior Work" section is adequate but should be expanded to more sharply distinguish the method from speculative decoding variants and to clarify its orthogonality to architectural changes like MQA/GQA.

Revised Draft
# Cross-Query KV Segment Reuse for Template-Heavy Agents

## Abstract
Large language model (LLM) agents repeatedly process similar prompt fragments (e.g., system preambles, tool schemas, scaffold patterns) across iterative steps. We introduce cross-query KV segment reuse, a content-aware mechanism that caches and reuses attention key–value (KV) activations for internal prompt fragments across requests and positions. Our method combines approximate segment matching via shingled MinHash, exact positional alignment of cached keys through RoPE reindexing, and a correctness-preserving splicing technique using layer-limited reuse and boundary halo recomputation. This generalizes prefix caching to tolerate position shifts and minor textual variation. We provide an implementation for open-source RoPE models (e.g., Llama/Mistral). An evaluation protocol on agentic benchmarks (e.g., HotpotQA, ToolBench-like tasks) is defined to measure latency, FLOPs, and task success, enabling validation with small open-source models. We release code and artifacts to support reproducibility.

## 1. Introduction
LLM agents with tool use and planning loops repeatedly include near-duplicate prompt structures across turns. Exact-prefix caching only helps when repeated content appears as an identical prefix; it fails for internal fragments, position shifts, or minor edits, thus wasting prefill compute.

We propose cross-query KV segment reuse:
- Retrieve KV for similar internal fragments across requests using approximate matching.
- Realign cached keys to new absolute positions via RoPE reindexing.
- Splice reused KV for a subset of lower layers while recomputing a short boundary halo and all upper layers to preserve correctness.

This approach targets template-heavy agents where a substantial portion of tokens are shared across turns and users within a trusted domain. It complements batching, FlashAttention, and paged KV memory.

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
- **Segmentation**:
  - Split text at structural anchors (e.g., newline-delimited `Thought`/`Action`/`Observation` tags, tool schema boundaries).
  - Add overlapping fixed-size windows (e.g., 64–128 tokens, stride 32–64) to capture unanchored repeats.
- **Canonicalization**:
  - To improve retrieval recall, normalize text before signature generation by lowercasing, standardizing whitespace, and masking volatile spans (e.g., timestamps, numeric IDs) with placeholder tokens. This does not affect the token IDs used for execution.
- **Signatures and Index**:
  - Compute MinHash over k-gram token shingles (k≈6–10). Store signatures in an LSH index for approximate Jaccard search.
  - Store an auxiliary signature for the W-token left context (W≈32–128) to bias toward matches with similar local neighborhoods, reflecting lower-layer locality.

### 3.2 KV Cache Storage
For each segment, store:
- Model metadata: architecture, hidden size, heads, RoPE base.
- Token metadata: token IDs, original absolute positions.
- KV tensors: for a subset of layers $L_{reuse}$ (e.g., bottom 8–16), store K (post-RoPE) and V. Quantize (e.g., int8 with per-channel scales) to reduce memory.
- Similarity metadata: segment and left-context MinHash signatures, segment length.

Memory estimate (per segment):
- Let $d_{model}$ be hidden size, $H$ heads, $L_{reuse}$ layers, $T$ tokens.
- Unquantized KV size ≈ $2 \cdot T \cdot L_{reuse} \cdot d_{model}$ elements.
- With int8, bytes ≈ $T \cdot L_{reuse} \cdot d_{model}$. For Llama-7B ($d_{model}=4096$), $L_{reuse}=16$, $T=128$: ≈8 MB.

### 3.3 RoPE Reindexing (Exact Key Position Shift)
Let $R(p)$ denote the rotation matrix applied by RoPE at absolute position $p$ to Q and K. If $K_{unrot}$ is the unrotated key and $K_{rot}(p) = R(p) \cdot K_{unrot}$, then for a cached key at $p_{old}$ and a target position $p_{new}$:
$K_{rot}(p_{new}) = R(p_{new}) \cdot R(p_{old})^{-1} \cdot K_{rot}(p_{old})$.
Since $R$ is orthonormal, $R(p_{old})^{-1} = R(-p_{old})$, so $K_{rot}(p_{new}) = R(p_{new} - p_{old}) \cdot K_{rot}(p_{old})$.
Thus, we can exactly reindex cached keys by applying a delta rotation $R(\Delta p)$ where $\Delta p = p_{new} - p_{old}$. Values require no adjustment. Q is computed fresh in the new context.

### 3.4 Splicing and Execution
For a new prompt:
1.  **Candidate Retrieval**: For each prospective segment, query LSH by its signature and filter by length, model metadata, and Jaccard thresholds $\tau_{seg}$ and $\tau_{ctx}$ on segment and left-context signatures.
2.  **Lightweight Acceptance Test**: To validate a candidate, we perform a cheap consistency check at its boundaries. For the first and last $h$ tokens (the "halo") of the segment, we compute two sets of output logits for layer $L_{max\_reuse}$:
    - **Reference Logits**: Computed by performing a standard forward pass for only the halo tokens, conditioned on the true preceding context.
    - **Reuse Logits**: Computed by performing a forward pass for the halo tokens, conditioned on the preceding context *and* the reindexed, reused KV cache for the segment's interior.
    We accept the reuse candidate if the KL divergence between reference and reuse logits is below a threshold $\epsilon$ (e.g., 0.05 nats) for a majority of halo tokens. This test isolates the impact of splicing and has a cost proportional to $h$, amortized over the length of the reused segment.
3.  **Layer-Limited Reuse with Halo Recomputation**: If accepted, insert reindexed K and V tensors into the runtime KV cache for layers in $L_{reuse}$ and for the interior tokens of the segment. Then, recompute the halo tokens (h on each side) and all layers outside $L_{reuse}$ normally to preserve boundary consistency and high-level dependencies.
4.  **Fallback**: If the acceptance test fails or no candidate meets thresholds, compute from scratch and optionally cache the new segment.

### 3.5 Complexity and Savings
- **Overheads**:
  - LSH probe: Sub-millisecond on commodity CPUs.
  - Reindexing: $O(T \cdot L_{reuse} \cdot d_{model})$ via efficient fused kernels.
  - Acceptance test: $O(h)$ tokens $\times$ limited layers; cost is tunable via $h$.
- **Savings**:
  - Prefill FLOPs avoided per segment are proportional to the number of reused tokens and layers, covering both attention and MLP computation. Net gain requires segments of moderate length (e.g., ≥64 tokens) and reuse frequency.

## 4. Evaluation Protocol
- **Models**: Llama-2-7B, Mistral-7B, and one instruction-tuned variant each.
- **Datasets**: HotpotQA (distractor setting) with a ReAct-style agent; a ToolBench-like subset with retrieval and arithmetic APIs.
- **Baselines**: No reuse (standard decoding); exact-prefix caching.
- **Ablations**: Vary $L_{reuse} \in \{8,12,16\}$, $h \in \{4,8\}$, $\tau_{seg} \in \{0.7,0.8,0.9\}$, and acceptance test on/off.
- **Metrics**:
  - Performance: End-to-end latency per query; prefill FLOPs saved.
  - Quality: EM/F1 (HotpotQA); tool call success rate and final answer accuracy.
  - Stability: Levenshtein distance across reruns with fixed seeds.
  - Cache Behavior: Acceptance rates, boundary KL divergence, fallback frequency.
- **Hardware**: Single A100-40GB and single L4-24GB.

## 5. Results
(Tables and figures demonstrating latency/FLOPs reductions and quality preservation to be included. The provided code and artifacts will enable full reproduction.)

## 6. Analysis and Discussion
- **Mechanism**: Lower layers encode local features, making their activations reusable when segment content and left context match. RoPE reindexing provides exact positional alignment for keys. Halo recomputation localizes approximation errors, allowing upper layers to correctly integrate global context.
- **Failure Modes**: Tokenization drift (mitigated by canonicalization); over-aggressive reuse (controlled by acceptance tests); memory overhead (managed by LRU eviction and quantization).
- **Security and Privacy**: Caches must not be shared across untrusted tenants. Per-tenant encryption at rest and zeroization on eviction are recommended.

## 7. Relation to Prior Work
- **Prefix Caching / Paged KV**: We generalize beyond identical prefixes to internal, approximately matching segments with exact positional realignment. Our method integrates with, and benefits from, paged KV managers.
- **KV Compression/Pruning**: Orthogonal. Our method avoids recomputation entirely for certain blocks, while compression reduces the cost or size of all KV states. They can be combined.
- **Speculative/Draft-based Decoding**: Complementary. Speculative methods reuse computation for *autoregressively generated* sequences, while our method reuses computation for *prompt (prefill)* sequences.
- **Attention variants (MQA/GQA)**: Orthogonal. Architectural changes like MQA/GQA reduce the memory and bandwidth cost of the KV cache, while our method reduces the compute cost of populating it.

## 8. Limitations
- Requires a position-encoding scheme like RoPE that allows exact or near-exact position shifting.
- Benefits are contingent on segment recurrence, yielding smaller gains for free-form chat.
- Introduces engineering complexity in cache management and execution logic.

## 9. Reproducibility and Artifact Plan
We release open-source code for the cache manager, RoPE delta-rotation kernels, LSH index, and evaluation scripts. Configuration files, seeds, and instructions for reproducing all results on both A100 and L4 GPUs will be provided.

## 10. Conclusion
Cross-query KV segment reuse extends caching beyond identical prefixes by combining content-aware retrieval, exact RoPE reindexing, and correctness-preserving splicing. The method is practical for template-heavy agentic workloads, integrates with existing inference optimizations, and can be validated on small open-source models.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
