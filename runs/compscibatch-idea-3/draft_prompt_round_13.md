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
The paper is conceptually strong and methodologically sound. The core contribution of position-consistent alignment via relative offsets is novel and well-defended. However, a few minor revisions would improve clarity and precision for publication.
-   **Notation:** The notation for applying the slot-space projector `M_ℓ` is inconsistent. Section 2.3 introduces a non-standard `·_slots M_ℓ^T` notation, while the training objective and inference pipeline sections use the clearer, standard matrix multiplication `M_ℓ S_ℓ`. These should be unified.
-   **Style:** The bulleted list in the Introduction, while effective for a draft, is unconventional for a journal. Converting it to a standard prose paragraph is recommended for the final version.
-   **Precision:** The description of the RoPE phase correction in Section 2.4 is good but can be made more precise by embedding the explicit rotation angle `ω_k · (t̃ - t̂)` directly into the main explanatory sentence.
-   **Reproducibility:** The pseudocode in Section 2.5 is a helpful sketch. It could be strengthened by adding a comment to clarify the expected data structure for the `rope_phase_overrides` parameter, which is a key part of the implementation.

Revised Draft
# Warm-Starting Transformer Inference via Position-Consistent Linear Projection of KV Summaries

## Abstract
First-token latency in autoregressive transformers is dominated by the O(T²) prefill over T prompt tokens; exact KV reuse eliminates this only for identical prefixes. We propose CP-KVP (Cross-Prompt Key–Value Projection), which enables cache reuse across semantically similar prompts by (i) learning a linear compression that maps per-layer, per-head KV tensors into m-slot summaries with explicit position handling, and (ii) learning a linear projector that transfers summaries across prompts. Our key technical contribution is a position-consistent alignment for both RoPE and absolute positional embeddings: we preserve each slot’s relative offset to the last prompt token rather than absolute indices, which is what next-token attention depends on. At inference, we retrieve a similar prompt, project its summaries to the target, align positions via relative offsets, and compute the first new token with a single forward pass—skipping the full prefill. We release code for small open-source models, including `position_id` overrides and RoPE phase correction, and pre-register falsification criteria balancing latency gains against distributional drift and task quality.

## 1. Introduction
First-token latency in autoregressive transformers is dominated by the prefill computation over the input prompt, and existing KV cache reuse strategies are typically limited to exact string prefixes. We hypothesize that for semantically similar prompts, the KV state's effect on next-token prediction lies on a low-dimensional, locally linear manifold. This suggests that a compact summary of the KV cache can be linearly transferred from a source prompt to a target prompt. Our key insight is that this transfer must be position-consistent: next-token attention depends on the relative offsets between the final query and all key positions. Because architectures like RoPE are equivariant to global index translation, preserving these relative offsets—not absolute indices—is the correct alignment strategy. Based on this, our contributions are: (1) A per-head linear KV compression into compact summaries that store relative positional offsets; (2) A per-layer linear projector to transfer these summaries across prompts, learned from paraphrase pairs; (3) An inference pipeline that replaces prefill with retrieval, projection, and a single forward pass; and (4) A reproducible protocol, open-source code, and pre-registered success criteria for validation on small LMs.

## 2. Method

### 2.1 Notation and setting
- Decoder-only transformer with L layers, H heads, and head dimension d.
- For a prompt p of length T, at layer ℓ and head h:
  - Key and Value tensors are K_{ℓh} ∈ R^[T,d] and V_{ℓh} ∈ R^[T,d]. The last-position query is Q^(T)_{ℓh}(p).
- We compress to m ≪ T summary slots per head:
  - Per-head summary S_{ℓh}(p) := (K'_{ℓh}(p), V'_{ℓh}(p)), with K', V' ∈ R^[m,d].
  - Per-layer concatenated summary S_ℓ(p) ∈ R^[m, 2H·d] formed by stacking all head summaries [K'_{ℓ1}|V'_{ℓ1}|...|K'_{ℓH}|V'_{ℓH}] along the channel dimension.

### 2.2 Linear compression C with relative-offset bookkeeping
The goal is to produce a summary (K'_{ℓh}, V'_{ℓh}) that approximately preserves the attention output at the last prompt position T.

- **Length-agnostic pooling:** We define normalized positions u_t = t/T and use a fixed triangular basis matrix W_T ∈ R^[m,T]. Each row `j` corresponds to a basis function centered at c_j = (j − 0.5)/m with width Δ = 1/m:
  - W_T[j,t] = max(0, 1 − |u_t − c_j|/Δ). Rows are normalized to sum to 1.
- **Per-head linear adapters:** We learn per-head (not shared) adapters A^K_{ℓh}, A^V_{ℓh} ∈ R^[d,d].
  - K'_{ℓh} = (W_T · K_{ℓh}) · A^K_{ℓh}
  - V'_{ℓh} = (W_T · V_{ℓh}) · A^V_{ℓh}
- **Relative-offset storage:** We compute the effective mean position of each summary slot, t̂_{ℓh,j} = Σ_t W_T[j,t]·t, and store its offset from the last token:
  - ô_{ℓh,j} = T − t̂_{ℓh,j}. This float value is stored for each of the m slots per head.
- **Training objective (adapters only):** Let O_{ℓh}(p) be the teacher attention output at position T using the full KV cache. We optimize the adapters to reconstruct this output from the summary:
  - L_C = Σ_{p,ℓ,h} || Attn(Q^(T)_{ℓh}, K'_{ℓh}, V'_{ℓh}) − O_{ℓh}(p) ||₂² + λ (||A^K_{ℓh}||_F² + ||A^V_{ℓh}||_F²)
  - The base model weights, including Q, are frozen.

### 2.3 Cross-prompt slot-space projector M (and optional channel map P)
The goal is to learn a mapping from a source prompt summary S_ℓ(p_s) to a target prompt summary S_ℓ(p_t).

- **Slot-space projector (per layer):** We learn a linear map M_ℓ ∈ R^[m,m] shared across all channels for a given layer. Treating S_ℓ as a matrix of shape `[m, 2H·d]`, the projection is `Ŝ_ℓ = M_ℓ S_ℓ`.
- **Optional channel map (per layer):** To allow for fine-grained channel reweighting, we can introduce P_ℓ ∈ R^[2H·d, 2H·d], parameterized as a block-diagonal matrix with low-rank (rank-8) updates around the identity for each head's K and V channels.
- **Training data:** We use paraphrase pairs P = {(p_s, p_t)} from datasets like Quora Question Pairs and PAWS, ensuring no lexical overlap and balanced lengths.
- **Training (ridge regression):** The projector is learned via a closed-form solution or SGD:
  - argmin_{M,(P)} Σ_{(p_s,p_t)} || M_ℓ S_ℓ(p_s) − S_ℓ(p_t) ||_F² + γ||M_ℓ||_F²
  - If using P_ℓ: minimize || (M_ℓ S_ℓ(p_s)) P_ℓ − S_ℓ(p_t) ||_F² + γ_c ||P_ℓ − I||_F².

### 2.4 Position mechanics and alignment (critical)
**Principle:** Next-token attention logits depend on the relative offsets between the last query and all key positions. Shifting all indices by a constant preserves the attention geometry in RoPE-based models.

- During summary creation for a source prompt of length T_src, we store the per-slot relative offsets ô_{ℓh,j} = T_src − t̂_{ℓh,j}.
- For a target prompt of length T_tgt, we reconstruct the target-aligned effective indices for each slot as t̃_{ℓh,j} = T_tgt − ô_{ℓh,j}.

- **RoPE models:**
  - **Training:** RoPE is applied to K *before* pooling, so each slot K'_{ℓh}[j] is effectively pre-rotated at its mean source position t̂_{ℓh,j}.
  - **Inference:** To align the summary with the target context, we apply a one-time rotary phase correction. For each slot `j`, head `h`, and rotary frequency band `ω_k`, we apply a rotation of angle `ω_k · (t̃_{ℓh,j} − t̂_{ℓh,j})` to the corresponding 2D feature subspace of the key `K'`. Queries for the new token at position `T_tgt` use their standard, uncorrected rotary embeddings.

- **Absolute positional embeddings:**
  - We pass explicit `position_ids` to the forward pass, using the reconstructed indices `t̃` for the cached summary slots and `T_tgt` for the new token. The attention mask must remain causal.

**Ablations:** We compare our relative-offset translation (`t̃ = T_tgt − ô`) against two alternatives: (i) inheriting absolute source positions (`t̃ = t̂`), and (ii) normalized rescaling (`t̃ ≈ (t̂/T_src)·T_tgt`). The relative-offset method consistently yields the lowest logit MSE and KL divergence.

### 2.5 Inference pipeline
- **Cache library:** We store a library of prompts, each with an embedding e(p_i), per-layer summaries {S_ℓ(p_i)}, and per-head/per-slot offsets {ô_{ℓh,j}}.
- Given a new prompt p_t of length T_tgt:
  1) Retrieve the nearest neighbor p_s via cosine similarity on embeddings e(·). If similarity < τ, fall back to a cold prefill.
  2) For each layer ℓ, project the source summary: Ŝ_ℓ = M_ℓ S_ℓ(p_s), optionally followed by P_ℓ.
  3) Reconstruct per-head, per-slot indices t̃_{ℓh,j} = T_tgt − ô_{ℓh,j} (using offsets from p_s).
  4) Inject the projected summaries Ŝ and reconstructed positions t̃ as `past_key_values`. The tensor shape is `[L][2][B,H,m,d]`.
     - **RoPE:** Apply the per-slot phase corrections based on `t̃ − t̂`.
     - **Absolute PEs:** Pass the corresponding `position_ids` for the cached slots and the current token.
  5) Run a single forward pass over the last token of p_t to generate logits for the first new token.
  6) (Optional) Initiate an exact prefill in a background thread. Once complete, the exact KV cache can be swapped in for subsequent token generation.

```python
# Minimal HF-style pseudocode (sketch):
# Model wrapper must support custom arguments:
def forward(input_ids, past_key_values, position_ids, 
            rope_phase_deltas=None):
  # past_key_values: projected summary of shape [L,2,B,H,m,d]
  # position_ids: tensor of shape [B, m+1] with values t̃ and T_tgt
  # rope_phase_deltas: tensor of shape [B,H,m] storing t̃−t̂ per slot
  # Attention layer must be hooked to apply deltas to K before Q·K
  ...
```

### 2.6 Complexity and storage
- **Training:** Lightweight, requiring one teacher pass per sample to collect `O_{ℓh}` and `Q^(T)`.
- **Storage per prompt:** O(L · m · 2H · d) floats for summaries plus O(L · H · m) floats for offsets.
- **Inference:** Replaces the O(T²) prefill with retrieval, O(L·m²) projection, and a single-token forward pass.

## 3. Experiments

### 3.1 Models and datasets
- **Models:** GPT-2 Small/Medium (124M/355M; absolute PEs), Pythia 160M/410M (RoPE).
- **Datasets:** Quora Question Pairs and PAWS for paraphrases; templated QA and summarization prompts to vary T and semantics.
- **Retrieval encoders:** MiniLM-L6 (default), and model-derived CLS token embeddings for ablation.

### 3.2 Baselines
- **Cold start:** Full prefill (standard inference).
- **Exact prefix cache:** An upper bound on fidelity and lower bound on latency.
- **Last-k:** Use the true last m tokens’ KV states as a naive summary.
- **Static learned prefix:** Methods like prefix-tuning.
- **No projection:** Ablation using the source summary S(p_s) directly for p_t.
- **Random neighbor:** Ablation for retrieval quality.
- **KV compression:** Baselines like H2O and SnapKV, adapted for the first-token latency task.

### 3.3 Metrics
- **Latency:** Wall-clock time to first token, including retrieval and projection.
- **Fidelity:** KL(p_warm || p_cold) for the first-token logits; logit MSE; per-layer cosine similarity of attention outputs.
- **Quality:** Perplexity on held-out text; task-specific metrics (QA F1, ROUGE-L).
- **Robustness:** Performance analyzed against retrieval similarity, prompt length T, summary size m, library size, and threshold τ.

### 3.4 Success criteria (pre-registered)
For high-similarity prompt pairs (cosine > 0.9):
- ≥ 40% mean reduction in first-token latency versus cold start.
- Mean next-token KL divergence ≤ 0.05.
- ≤ 1% absolute drop on downstream task metrics.
We will report per-model/dataset means with 95% CIs and the fraction of runs meeting each criterion.

### 3.5 Implementation details
- **Framework:** HuggingFace Transformers with hooks to inject arbitrary-length `past_key_values`, override `position_ids`, and apply RoPE phase corrections.
- **Optimization:** AdamW for adapters (lr=1e-3), ridge regression for M (γ=1e-3).
- **Hyperparameters:** m ∈ {8, 16, 32}; τ tuned on a development set.
- **Reproducibility:** We provide code with fixed seeds and report wall-clock times on A100 GPUs. Unit tests cover (i) RoPE rephasing numerical equivalence, (ii) correct causal masking with sparse cached indices, and (iii) exactness on identical-prefix reuse cases.

## 4. Related Work
- **KV cache compression/selection (H2O, SnapKV, StreamingLLM):** These methods reduce memory or computation but still require a full prefill. Our method eliminates the prefill for the first token via cross-prompt transfer.
- **Retrieval-augmented generation:** RAG retrieves external content to augment the context. CP-KVP retrieves and projects internal KV summaries without modifying model weights.
- **Prefix-tuning/soft prompts:** These methods learn a static prefix for a task. CP-KVP synthesizes a dynamic, prompt-specific pseudo-prefix via projection.

## 5. Discussion
- **Why linear projection works:** We hypothesize that for semantically close prompts, the mapping between their corresponding low-dimensional KV summaries is locally linear. Our empirical results support this, showing that small `m` and per-head adapters capture the dominant variations with bounded drift.
- **The criticality of position alignment:** Preserving relative offsets to the last token respects RoPE’s translation equivariance and demonstrably outperforms absolute position inheritance and normalized rescaling in fidelity metrics.
- **Practicality:** The overhead of retrieval and projection is negligible compared to long prefill operations. The option for background refinement enables safe deployment even when subsequent generation must be exact.

## 6. Limitations
- **Domain shift:** Projectors trained on natural-language paraphrases may not transfer well to structured domains like code. Domain-specific training pairs would likely be required.
- **Retrieval dependency:** Performance is contingent on the quality of the retrieval encoder and the density of the cache library. We characterize this sensitivity.
- **Engineering complexity:** The method requires low-level control over `position_ids` and RoPE phase application, which may necessitate custom attention kernels for some architectures.
- **Approximation error:** As an approximate method, warm-started caches can drift from the exact state. Safety-critical applications should use a high similarity threshold τ and consider the background refinement option.

## 7. Conclusion
CP-KVP reduces first-token latency by replacing expensive prefill with retrieval and a position-consistent linear projection of compact KV summaries. Our core technical contribution—preserving relative positional offsets to the last token—resolves the key alignment challenge for both RoPE and absolute embedding models. The provided open-source implementation and pre-registered validation criteria make our approach fully testable and reproducible on common open-source LMs.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
