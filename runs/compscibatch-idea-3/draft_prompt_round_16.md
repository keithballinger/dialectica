You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Draft:
        Minor Revisions

Brief critique:
The paper presents a novel and high-impact method for warm-starting LLM inference, CP-KVP. The core contribution—position-consistent summary projection—is well-motivated, technically sound, and the proposed validation is rigorous. The distinction between alignment for RoPE and absolute positional embeddings is a key strength. The draft is strong and nearly ready for publication. The recommended revisions are minor clarifications to improve reproducibility and formal precision.

1.  **Training Objective Ambiguity:** The compressor training objective `L_C` in Section 2.2 is underspecified. The `Attn` function requires positions for the `m` key/value slots. It should be explicitly stated that these positions are derived from the stored relative offsets (`t'_j = T - ô_j`) for this calculation.
2.  **RoPE Justification:** The formal reason for the global phase shift's sufficiency (`t̃ - t̂ = T_tgt - T_src`) in Section 2.4 is correct but could be stated more explicitly as a derivation to solidify this critical insight.
3.  **Notation:** Mathematical notation for tensor dimensions (e.g., `R[T,d]`) should be standardized to the more common `R^{T \times d}`. Complexity analysis in Section 2.6 could more clearly define model dimension `D = H \cdot d`.

These minor changes will enhance the paper's clarity and formal rigor, leaving no ambiguity for implementation.

Revised Draft
# Warm-Starting Transformer Inference via Position-Consistent Linear Projection of KV Summaries

## Abstract
First-token latency in autoregressive transformers is dominated by the O(T²) prefill over T prompt tokens. While exact Key-Value (KV) caching eliminates this cost for identical prefixes, its utility is limited. We propose CP-KVP (Cross-Prompt Key-Value Projection), a method that enables cache reuse across semantically similar prompts. CP-KVP learns (i) a linear compression mapping per-layer, per-head KV tensors into compact `m`-slot summaries, and (ii) a linear projector that transfers these summaries between prompts. Our key technical contribution is a position-consistent alignment framework that preserves each summary slot’s relative offset to the final prompt token, ensuring correct attention mechanics for both RoPE and absolute positional embeddings. For RoPE models, this alignment simplifies to a single global rotary phase shift, `Δ = T_tgt − T_src`, per forward pass. At inference, we retrieve a similar prompt from a cache, project its KV summaries, align positions, and compute the first new token in a single forward pass, entirely skipping the prefill stage. We provide an open-source implementation for small models, including integer-safe position overrides for absolute embeddings and phase correction for RoPE, and pre-register falsification criteria balancing latency gains against distributional drift.

## 1. Introduction
First-token latency in autoregressive transformers is governed by the computationally expensive prefill phase over the input prompt. Existing KV cache reuse is restricted to scenarios with exact prefix matches, limiting its applicability. We hypothesize that for semantically similar prompts, the KV state's influence on next-token prediction lies on a low-dimensional, locally linear manifold. This implies that a compact summary of the KV cache can be linearly transferred from a source prompt to a target prompt, provided the alignment respects the model’s positional encoding scheme. Our key insight is that next-token attention is primarily dependent on the relative offsets between the final query and the keys. For RoPE, translation equivariance further implies a global index shift suffices.

Our contributions are: (1) a per-head linear KV compression into compact summaries that explicitly store relative positional offsets; (2) a per-layer linear projector that transfers these summaries across prompts, trained on paraphrase pairs with position normalization; (3) an inference pipeline that replaces prefill with retrieval, projection, and a single forward pass; and (4) an open-source implementation with unit tests and pre-registered validation criteria to enable rigorous evaluation on small language models.

## 2. Method

### 2.1 Notation and Setting
- Decoder-only transformer with L layers, H heads, and head dimension d. Model dimension `D_model = H \cdot d`.
- For a prompt `p` of length T, at layer `ℓ` and head `h`:
  - Keys, values: `K_{ℓh} \in R^{T \times d}`, `V_{ℓh} \in R^{T \times d}`. The final query is `Q^{(T)}_{ℓh}(p)`.
- We compress the KV cache into `m \ll T` summary slots per head:
  - Per-head summary: `K'_{ℓh}, V'_{ℓh} \in R^{m \times d}`.
  - Per-layer concatenated summary `S_ℓ(p) \in R^{m \times 2Hd}` by stacking heads along the channel dimension: `[K'_{ℓ1}|V'_{ℓ1}|…|K'_{ℓH}|V'_{ℓH}]`.

### 2.2 Linear Compression C with Relative-Offset Bookkeeping
We approximate the attention output at the final prompt position T using `m` summary slots per head.

- **Length-Agnostic Pooling:** Define normalized positions `u_t = t/T` and a triangular basis `W_T \in R^{m \times T}`. Row `j` is centered at `c_j = (j − 0.5)/m` with width `Δ = 1/m`:
  - `W_T[j,t] = max(0, 1 − |u_t − c_j|/Δ)`. Rows are L1-normalized.
- **RoPE-Aware Pooling for K:** RoPE is applied to per-token keys `K_{ℓh}` before pooling. Values `V_{ℓh}` are not rotated.
- **Per-Head Adapters:** Learn linear maps `A^K_{ℓh}, A^V_{ℓh} \in R^{d \times d}`:
  - `K'_{ℓh} = (W_T \cdot K_{ℓh}^{\text{rope}}) \cdot A^K_{ℓh}`
  - `V'_{ℓh} = (W_T \cdot V_{ℓh}) \cdot A^V_{ℓh}`
- **Relative-Offset Storage:** For each slot `j`, we store the weighted mean position `t̂_{ℓh,j} = Σ_t W_T[j,t] \cdot t` and its relative offset from the end of the prompt: `ô_{ℓh,j} = T − t̂_{ℓh,j}`.
- **Training Objective (Adapters Only):** Let `O_{ℓh}(p)` be the teacher head output at position T with the full KV cache. We optimize the adapters `A` by minimizing the reconstruction loss on the attention output:
  - `L_C = Σ_{p,ℓ,h} || Attn(Q^{(T)}_{ℓh}, K'_{ℓh}, V'_{ℓh}) − O_{ℓh}(p) ||₂² + λ (||A^K_{ℓh}||_F² + ||A^V_{ℓh}||_F²)`
  with the base model frozen. The attention mechanism `Attn` uses key positions `t'_{j} = T - ô_{ℓh,j}` derived from the stored relative offsets.

**Approximation Note:** Pooling after applying RoPE to per-token keys mixes different rotation angles within a single summary slot. The induced error scales with the slot’s positional support width and the maximum rotary frequency `ω_max`. We mitigate this with narrow supports (`Δ = 1/m`) and show in ablations that increasing `m` monotonically reduces logit MSE. Optionally, we can band-limit RoPE to its lowest K frequencies during compression to further reduce this error.

### 2.3 Cross-Prompt Slot-Space Projector M
We learn a linear map from a source prompt's summary to a target prompt's summary.

- **Position-Normalized Summaries for Training:** To disentangle content from positional information, we "canonize" summaries for training the projector. For RoPE models, we remove the global phase corresponding to the last token's position, effectively rotating `K'` such that the last token is at index 0 (`Δ = −T`). For absolute PE models, we use relative offsets `ô` directly as the canonical position representation.
- **Slot-Space Projector (Per Layer):** Learn `M_ℓ \in R^{m \times m}`, shared across all channels. Treating `S_ℓ` as a tensor of shape `(m, 2Hd)`, the projection is `Ŝ_ℓ = M_ℓ S_ℓ`.
- **Training Data:** Paraphrase pairs `{(p_s, p_t)}` (e.g., from QQP, PAWS), filtered for minimal lexical overlap and bounded length ratio.
- **Training (Ridge Regression):** With position-normalized summaries `S̄_ℓ`, we solve:
  - `argmin_{M_ℓ} Σ_{(p_s,p_t)} || M_ℓ S̄_ℓ(p_s) − S̄_ℓ(p_t) ||_F² + γ||M_ℓ||_F²`

### 2.4 Position Mechanics and Alignment
The core principle is that next-token attention depends on the relative offsets between the final query and the keys.

- **Relative-Offset Translation:** We store offsets `ô_{ℓh,j}` computed at source length `T_src`. For a target prompt of length `T_tgt`, we reconstruct the effective key positions as `t̃_{ℓh,j} = T_tgt − ô_{ℓh,j}`.
- **RoPE Models (Global Phase Shift):** RoPE is equivariant to index translation. Therefore, the required rotary correction is a single global phase shift `Δ = T_tgt − T_src`. This single shift is sufficient because the relative position change between any reconstructed target slot `t̃` and its source slot `t̂` is constant: `t̃ − t̂ = (T_tgt − ô) − (T_src − ô) = T_tgt − T_src`. This shift `Δ` is applied to all cached keys.
- **Absolute Positional Embeddings (Integer-Safe):**
  1.  **Integer Offsets:** Quantize offsets: `ô_int = round(ô)`.
  2.  **Target Indices:** Reconstruct integer target indices: `t̃_int = T_tgt − ô_int`.
  3.  **Monotonic Repair:** To preserve causality, enforce strictly increasing positions. Sort slots by `t̃_int` and apply a cumulative maximum: `t̃_int[j] = max(t̃_int[j], t̃_int[j−1]+1)`. Clamp values to `[0, T_tgt]`.
  4.  These repaired integer `position_ids` are used for the `m` cached slots; the new token uses position `T_tgt`.

**Ablations:** We compare relative-offset translation to alternative heuristics: (i) inheriting absolute source positions and (ii) normalized rescaling (`t̂/T_src \cdot T_tgt`). Relative-offset translation yields superior performance in terms of logit fidelity. For absolute PEs, monotonic repair is essential to prevent causal mask violations and has a negligible impact on fidelity for `m ≤ 32`.

### 2.5 Inference Pipeline
- **Cache Library:** For each prompt `p_i`, store its embedding `e(p_i)`, per-layer summaries `{S_ℓ(p_i)}`, offsets `{ô_{ℓh,j}}`, and length `T_src`.
- **Given a new prompt `p_t` (length `T_tgt`):**
  1. Retrieve nearest neighbor `p_s` via cosine similarity on embeddings `e(·)`. If similarity `< τ` or the length ratio is outside `[0.5, 2]`, fall back to a cold prefill.
  2. For each layer `ℓ`, compute the projected summary: `Ŝ_ℓ = M_ℓ S_ℓ(p_s)`.
  3. For RoPE models, compute the global phase shift: `Δ = T_tgt − T_src`.
  4. For absolute PE models, reconstruct integer positions via the monotonic-repair procedure.
  5. Inject `Ŝ` as `past_key_values` with shape `[L, 2, B, H, m, d]`.
     - **RoPE:** Pass `rope_phase_delta = Δ` to the attention kernel, which applies the rotation.
     - **Absolute PEs:** Pass `position_ids` of shape `[B, m+1]` containing the repaired indices `t̃_int` and the final token's position `T_tgt`.
  6. Execute a single forward pass on the last token of `p_t` to generate first-token logits.
  7. (Optional) Initiate an exact prefill in the background; swap in the exact KV cache for subsequent tokens when it becomes available.

### 2.6 Complexity and Storage
- **Storage per prompt:** `O(L \cdot m \cdot 2Hd)` floats for summaries plus `O(L \cdot H \cdot m)` for offsets.
- **Inference Cost:**
  - Retrieval: Negligible cost of a single vector similarity search.
  - Projection: Per-layer matrix-vector products `M_ℓ S_ℓ` cost `O(L \cdot m² \cdot 2Hd)`.
  - Forward Pass: A single-token pass with `m` cached slots.
  - The `O(T² \cdot D_{model} \cdot L)` prefill is replaced by retrieval and projection, which is significantly faster for `T \gg m`.

## 3. Experiments

### 3.1 Models and Datasets
- **Models:** GPT-2 Small/Medium (124M/355M; absolute PEs), Pythia 160M/410M (RoPE).
- **Datasets:** Quora Question Pairs and PAWS for paraphrase data; templated QA and summarization tasks to evaluate performance on varied semantics and lengths.
- **Retrieval Encoders:** MiniLM-L6 (default) and model-derived CLS embeddings (ablation).

### 3.2 Baselines
- Cold Start (full prefill), Exact Prefix Cache, Last-`m` (using the final `m` tokens' exact KV state), Static Learned Prefix (prefix-tuning), No Projection (direct transfer of `S(p_s)`), and adapted KV compression baselines (e.g., H2O, SnapKV) for the first-token latency task.

### 3.3 Metrics
- **Latency:** Wall-clock time to first token, including retrieval and projection.
- **Fidelity:** KL-divergence `KL(p_warm || p_cold)` on first-token logits; Logit MSE.
- **Quality:** Perplexity; Task-specific metrics (e.g., QA F1, ROUGE-L).
- **Robustness:** Sensitivity analysis on retrieval similarity `τ`, prompt length `T`, summary size `m`, and library size.

### 3.4 Success Criteria (Pre-registered)
For high-similarity pairs (cosine > 0.9), we pre-register the following success criteria:
- ≥ 40% mean reduction in first-token latency vs. cold start.
- Mean next-token KL divergence ≤ 0.05.
- ≤ 1% absolute degradation on downstream task metrics.

### 3.5 Implementation Details
- **Framework:** HuggingFace Transformers with hooks to support arbitrary-length `past_key_values`, integer `position_ids`, and `rope_phase_delta`.
- **Optimization:** AdamW for adapters (lr=1e-3), ridge regression for `M` (γ=1e-3).
- **Hyperparameters:** `m \in \{8,16,32\}`; `τ` tuned on a development set.
- **Unit Tests:** We include tests to verify (1) RoPE rephasing equivalence under global `Δ`, (2) monotonic position repair prevents causality violations, and (3) exactness is recovered for identical-prefix reuse.

## 4. Related Work
- **KV Compression/Selection (H2O, SnapKV, StreamingLLM):** These methods reduce memory and compute for long contexts but still require a full prefill. CP-KVP eliminates the prefill for the first token via cross-prompt transfer.
- **Retrieval-Augmented Generation:** RAG retrieves external textual documents. In contrast, CP-KVP retrieves and projects internal KV summaries without modifying the base model's weights or input context.
- **Prefix-Tuning/Soft Prompts:** These methods learn static prefixes that are prepended to all inputs. CP-KVP synthesizes dynamic, prompt-specific pseudo-prefixes through retrieval and projection.

## 5. Discussion
- **Why Linear Projection Works:** We posit that for semantically close prompts, the mapping between their low-dimensional KV summaries is locally linear. Position normalization effectively isolates content variation, making this linear approximation more accurate.
- **The Centrality of Position Alignment:** Our results demonstrate that correct positional alignment is critical. Relative offsets, implemented as a global RoPE phase shift or repaired integer IDs, are both theoretically sound and practically superior to naive alignment heuristics.
- **Practicality:** The overhead of retrieval and projection is negligible compared to long prefills. The option for background refinement allows for safe deployment where subsequent token generation must be exact.

## 6. Limitations
- **Domain Shift:** Projectors trained on natural language paraphrases may not generalize effectively to structured domains like code or mathematics without fine-tuning on domain-specific data.
- **Retrieval Dependency:** Performance is contingent on the quality of the retrieval encoder and the density of the cache library.
- **Engineering Overhead:** The method requires backend control over attention kernels (`rope_phase_delta`) and `position_ids`, which may necessitate custom implementations in some inference frameworks.
- **Approximation Error:** The method is inherently an approximation. For safety-critical applications, a high retrieval similarity threshold (`τ`) and the use of background refinement are recommended.

## 7. Conclusion
CP-KVP significantly reduces first-token latency by replacing the expensive prefill operation with retrieval and position-consistent linear projection of compact KV summaries. Our central contribution, a method for preserving relative positional offsets—implemented as a global RoPE phase shift or integer-safe `position_ids`—resolves the key alignment challenge across different transformer architectures. Our open-source implementation and pre-registered validation framework enable robust testing and adoption by the community.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
