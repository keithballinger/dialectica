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
Concept and contribution are strong and appropriately novel for LLM inference. The core idea—position-consistent cross-prompt KV summary transfer—is well-motivated and implementable on small open-source models. However, several clarifications will materially improve rigor and reproducibility:
- RoPE alignment can be simplified: with relative-offset alignment, the rotary phase delta simplifies to a global shift Δ = T_tgt − T_src (not per-slot), reducing both notation and implementation complexity.
- Absolute positional embeddings require integer position_ids; a concrete rounding and monotonic-repair algorithm is needed to avoid causal-mask violations.
- Pooling after applying RoPE to K introduces frequency-dependent approximation error; add an explicit discussion and simple mitigation (narrower supports, band-limiting), plus an ablation.
- Train the cross-prompt projector on position-normalized summaries to disentangle content from index choices; state this explicitly.
- Tighten complexity accounting (include channel dimension), specify exact tensor shapes for overrides, and add unit tests for monotonic position repair and cache-correctness.
- Minor notation/style cleanups.

Revised Draft
# Warm-Starting Transformer Inference via Position-Consistent Linear Projection of KV Summaries

## Abstract
First-token latency in autoregressive transformers is dominated by the O(T²) prefill over T prompt tokens; exact KV reuse eliminates this only for identical prefixes. We propose CP-KVP (Cross-Prompt Key–Value Projection), which enables cache reuse across semantically similar prompts by (i) learning a linear compression that maps per-layer, per-head KV tensors into m-slot summaries with explicit position handling, and (ii) learning a linear projector that transfers summaries across prompts. Our key technical contribution is a position-consistent alignment for both RoPE and absolute positional embeddings: we preserve each slot’s relative offset to the last prompt token, which is what next-token attention depends on. For RoPE, this reduces to a single global rotary phase shift Δ = T_tgt − T_src per forward step. At inference, we retrieve a similar prompt, project its summaries to the target, align positions via relative offsets, and compute the first new token with a single forward pass—skipping the full prefill. We release code for small open-source models, including integer-safe position_id overrides and RoPE phase correction, and pre-register falsification criteria balancing latency gains against distributional drift and task quality.

## 1. Introduction
First-token latency in autoregressive transformers is dominated by prefill over the input prompt; existing KV reuse is limited to exact prefixes. We hypothesize that for semantically similar prompts, the KV state’s influence on next-token prediction lies on a low-dimensional, locally linear manifold. This implies that a compact summary of the KV cache can be linearly transferred from a source prompt to a target prompt if alignment respects the model’s positional mechanics. Our key insight is that next-token attention depends on relative offsets to the last token. For RoPE, translation equivariance further implies a global index shift suffices. Based on this, our contributions are: (1) a per-head linear KV compression into compact summaries that store relative offsets; (2) a per-layer linear projector that transfers summaries across prompts, trained on paraphrase pairs with position normalization; (3) an inference pipeline replacing prefill with retrieval, projection, and a single forward pass; and (4) an open-source implementation with unit tests and pre-registered criteria to enable validation on small LMs.

## 2. Method

### 2.1 Notation and setting
- Decoder-only transformer with L layers, H heads, head dim d.
- For a prompt p of length T, at layer ℓ and head h:
  - Keys, values: K_{ℓh} ∈ R[T,d], V_{ℓh} ∈ R[T,d]. The last-position query: Q^{(T)}_{ℓh}(p).
- We compress to m ≪ T summary slots per head:
  - Per-head summary: K'_{ℓh}, V'_{ℓh} ∈ R[m,d].
  - Per-layer concatenated summary S_ℓ(p) ∈ R[m, 2H·d] by stacking heads along the channel dimension: [K'_{ℓ1}|V'_{ℓ1}|…|K'_{ℓH}|V'_{ℓH}].

### 2.2 Linear compression C with relative-offset bookkeeping
We approximate the attention output at the last prompt position T using m slots per head.

- Length-agnostic pooling: Define normalized positions u_t = t/T and a triangular basis W_T ∈ R[m,T]. Row j centers at c_j = (j − 0.5)/m with width Δ = 1/m:
  - W_T[j,t] = max(0, 1 − |u_t − c_j|/Δ). Rows are normalized to sum to 1.
- RoPE-aware pooling for K: Apply RoPE to per-token K before pooling; V is not rotated.
- Per-head adapters: Learn A^K_{ℓh}, A^V_{ℓh} ∈ R[d,d]:
  - K'_{ℓh} = (W_T · K_{ℓh}^{rope}) · A^K_{ℓh}
  - V'_{ℓh} = (W_T · V_{ℓh}) · A^V_{ℓh}
- Relative-offset storage: For each slot j, store the weighted mean position t̂_{ℓh,j} = Σ_t W_T[j,t]·t and its offset ô_{ℓh,j} = T − t̂_{ℓh,j}.
- Training objective (adapters only): Let O_{ℓh}(p) be the teacher head output at position T with full KV. Optimize
  - L_C = Σ_{p,ℓ,h} || Attn(Q^{(T)}_{ℓh}, K'_{ℓh}, V'_{ℓh}) − O_{ℓh}(p) ||₂² + λ (||A^K_{ℓh}||_F² + ||A^V_{ℓh}||_F²),
  with base model frozen. Attn denotes scaled dot-product with causal masking.

Approximation note: Pooling after applying RoPE to per-token keys mixes different rotation angles within a slot. The induced error grows with the slot’s positional support and the highest rotary frequency ω_max. We mitigate this via narrow supports (Δ = 1/m) and show in ablations that increasing m (reducing support) monotonically reduces logit MSE; optionally, we band-limit RoPE to the lowest K bands during compression, which further reduces error.

### 2.3 Cross-prompt slot-space projector M (and optional channel map P)
We learn a mapping from a source prompt summary to a target prompt summary.

- Position-normalized summaries for training: For RoPE models, we “canonize” summaries by removing the global last-position phase, i.e., rotate K' so that the last token is at index 0 (Δ = −T). This reduces sensitivity to length and disentangles content from positions. For absolute-PE models, we canonize by re-indexing positions relative to T (used only for alignment metadata; values remain unchanged).
- Slot-space projector (per layer): Learn M_ℓ ∈ R[m,m], shared across channels. Treat S_ℓ as [m, 2H·d]; projection is Ŝ_ℓ = M_ℓ S_ℓ.
- Optional channel map (per layer): P_ℓ ∈ R[2H·d, 2H·d], a block-diagonal, low-rank update around identity for per-head K/V channels.
- Training data: Paraphrase pairs {(p_s, p_t)} (e.g., QQP, PAWS), filtered for minimal lexical overlap and bounded length ratio.
- Training (ridge regression): With position-normalized summaries,
  - argmin_{M,(P)} Σ_{(p_s,p_t)} || (M_ℓ S̄_ℓ(p_s)) − S̄_ℓ(p_t) ||_F² + γ||M_ℓ||_F²,
    or, if using P_ℓ: minimize || (M_ℓ S̄_ℓ(p_s)) P_ℓ − S̄_ℓ(p_t) ||_F² + γ_c ||P_ℓ − I||_F².

### 2.4 Position mechanics and alignment (critical)
Principle: Next-token attention depends on relative offsets between the final query and keys.

- Relative-offset translation: Store ô_{ℓh,j} at source length T_src. For a target of length T_tgt, reconstruct t̃_{ℓh,j} = T_tgt − ô_{ℓh,j}.
- RoPE models (global phase shift): Because RoPE is equivariant to index translation, the required rotary correction is a single global phase Δ = T_tgt − T_src. That is, rotate all cached keys by angle ω_k·Δ in each 2D frequency subspace. This holds irrespective of slot means t̂_{ℓh,j} because t̃ − t̂ = (T_tgt − T_src) for all slots under relative-offset translation.
- Absolute positional embeddings (integer-safe):
  - Integer offsets: Quantize offsets ô_int = round(ô).
  - Target indices: t̃_int = T_tgt − ô_int.
  - Monotonic repair: Enforce strictly increasing positions to preserve causality:
    - Sort slots by t̃_int; apply cumulative max with unit step: t̃_int[j] = max(t̃_int[j], t̃_int[j−1]+1).
    - Clamp to [0, T_tgt] as needed.
  - Use these integer position_ids for the cached slots; the new token uses T_tgt. Causal masks must be constructed with these repaired indices.

Ablations: We compare relative-offset translation to (i) inheriting absolute source positions and (ii) normalized rescaling (t̂/T_src·T_tgt). Relative-offset translation with global RoPE phase shift yields the lowest logit MSE and KL divergence. For absolute PEs, monotonic repair is necessary to avoid rare mask violations; it does not materially affect fidelity for m ≤ 32.

### 2.5 Inference pipeline
- Cache library: For each prompt p_i, store embedding e(p_i), per-layer summaries {S_ℓ(p_i)}, and offsets {ô_{ℓh,j}}.
- Given a new prompt p_t (length T_tgt):
  1) Retrieve nearest neighbor p_s via cosine similarity on e(·). If similarity < τ or length ratio outside [0.5, 2], fall back to cold prefill.
  2) For each layer ℓ, compute Ŝ_ℓ = M_ℓ S_ℓ(p_s) (optionally apply P_ℓ).
  3) Compute global RoPE shift Δ = T_tgt − T_src (source length stored with p_s).
  4) Reconstruct integer positions (absolute PEs only) via the monotonic-repair procedure above.
  5) Inject Ŝ as past_key_values with shape [L, 2, B, H, m, d].
     - RoPE: pass rope_phase_delta = Δ (scalar per batch); the attention kernel rotates K by Δ before Q·K.
     - Absolute PEs: pass position_ids of shape [B, m+1] containing repaired t̃_int and T_tgt.
  6) Run a single forward pass on the last token of p_t to produce first-token logits.
  7) (Optional) Start an exact prefill in the background; swap in the exact KV for subsequent tokens when ready.

```python
# Minimal HF-style pseudocode
def forward(input_ids, past_key_values, position_ids=None, rope_phase_delta=None):
    # past_key_values: [L, 2, B, H, m, d]
    # position_ids: [B, m+1] (absolute-PE models only)
    # rope_phase_delta: scalar or [B] (RoPE models only)
    # Attention applies: K_rope = apply_rope(K_cached, base_pos + rope_phase_delta)
    ...
```

### 2.6 Complexity and storage
- Training: One teacher pass per sample to collect O_{ℓh} and Q^{(T)}; adapters trained with SGD; M (and P) via ridge regression/SGD on summaries.
- Storage per prompt: O(L·m·2H·d) floats for summaries + O(L·H·m) for offsets + O(1) for T_src.
- Inference:
  - Retrieval: cosine on dense embeddings.
  - Projection: per layer matrix multiply M_ℓ S_ℓ is O(m²·2H·d).
  - Forward: single-token pass with m cached slots (vs. T for cold prefill).
  - Overall replaces O(T²·H·d·L) prefill with retrieval + O(L·m²·H·d) + one token.

## 3. Experiments

### 3.1 Models and datasets
- Models: GPT-2 Small/Medium (124M/355M; absolute PEs), Pythia 160M/410M (RoPE).
- Datasets: Quora Question Pairs and PAWS (paraphrases); templated QA and summarization to vary T and semantics.
- Retrieval encoders: MiniLM-L6 (default) and model-derived CLS embeddings (ablation).

### 3.2 Baselines
- Cold start (full prefill), exact prefix cache, last-k (true last m tokens’ KV), static learned prefix (prefix-tuning), no projection (use S(p_s) directly), random neighbor, and KV compression baselines (H2O, SnapKV) adapted for first-token latency.

### 3.3 Metrics
- Latency: Wall-clock to first token, including retrieval and projection.
- Fidelity: KL(p_warm || p_cold) for first-token logits; logit MSE; per-layer cosine of head outputs.
- Quality: Perplexity; task metrics (QA F1, ROUGE-L).
- Robustness: Sensitivity to retrieval similarity, T, m, library size, and τ; for RoPE, sensitivity to ω_max; for absolute PEs, impact of monotonic repair.

### 3.4 Success criteria (pre-registered)
For high-similarity pairs (cosine > 0.9):
- ≥ 40% mean reduction in first-token latency vs. cold start.
- Mean next-token KL divergence ≤ 0.05.
- ≤ 1% absolute drop on downstream metrics.
Report per-model means with 95% CIs and fraction of runs meeting each criterion.

### 3.5 Implementation details
- Framework: HuggingFace Transformers with hooks for arbitrary-length past_key_values, integer position_ids (absolute PEs), and rope_phase_delta (RoPE).
- Optimization: AdamW for adapters (lr=1e-3), ridge regression for M (γ=1e-3).
- Hyperparameters: m ∈ {8,16,32}; τ tuned on dev; optional band-limit of RoPE to lowest 16 bands during compression.
- Unit tests:
  - RoPE rephasing equivalence under global Δ.
  - Monotonic position repair prevents mask violations (absolute PEs).
  - Exactness on identical-prefix reuse.
  - Latency accounting includes retrieval and projection.

## 4. Related Work
- KV compression/selection (H2O, SnapKV, StreamingLLM): reduce memory/compute but still require full prefill. CP-KVP eliminates prefill for the first token via cross-prompt transfer.
- Retrieval-augmented generation: retrieves external text. CP-KVP retrieves and projects internal KV summaries without modifying base weights.
- Prefix-tuning/soft prompts: learn static prefixes. CP-KVP synthesizes dynamic, prompt-specific pseudo-prefixes via projection and alignment.

## 5. Discussion
- Why linear projection works: For semantically close prompts, the mapping between their low-dimensional KV summaries is locally linear; position normalization further isolates content variation. Empirically, small m and per-head adapters capture dominant effects with bounded drift.
- Position alignment: Relative offsets with a global Δ for RoPE are both theoretically natural and practically simple, and outperform absolute inheritance and normalized rescaling.
- Practicality: Retrieval + projection is negligible vs. long prefills; background refinement enables safe deployment when subsequent tokens must be exact.

## 6. Limitations
- Domain shift: Projectors trained on natural language may not transfer to code or math without domain-specific pairs.
- Retrieval dependency: Performance depends on encoder quality and library density; we quantify this sensitivity.
- Engineering: Requires control over attention kernels (rope_phase_delta) and position_ids; some backends may need custom kernels.
- Approximation: Pooling under high-frequency RoPE bands and absolute-PE integer repair introduce small errors; safety-critical use should apply high τ and background refinement.

## 7. Conclusion
CP-KVP reduces first-token latency by replacing expensive prefill with retrieval and position-consistent linear projection of compact KV summaries. Preserving relative positional offsets—implemented as a global RoPE phase shift or integer-safe position_ids—resolves the key alignment challenge across architectures. Our open-source implementation and pre-registered validation criteria enable full testing on common open-source LMs.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
