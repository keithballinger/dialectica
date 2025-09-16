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
- Notation inconsistency: switch cleanly between per-head S_{ℓh} and per-layer S_ℓ; define both once and use consistently. Make explicit that adapters A^K/A^V are per-head (not shared) and that the projector M_ℓ acts on slots, with optional channel map P_ℓ that is block-diagonal per head.
- RoPE mechanics need a precise, implementation-ready description of the phase correction (how Δ is computed and applied across rotary frequency bands) and an explicit statement that both K and Q must be consistently rephased or that Q remains at T_tgt while only K are corrected to t̃.
- Specify the low-rank update rank once (e.g., rank-8) and keep it consistent across text and experiments.
- Success criteria: report concrete numbers with CIs and clearly state whether each pre-registered criterion was met for each model/dataset. Avoid “see Table 1” without including the table or summary values.
- Training/data clarity: define paraphrase pair construction, library sizes, and retrieval thresholds τ; include robustness to τ and the effect of library size on latency.
- Reproducibility: include a one-screen reproducible snippet (HF wrapper signature, shapes) and explicitly note required Transformers hooks; add unit-test descriptions for position_id overrides.

Revised Draft

# Warm-Starting Transformer Inference via Position-Consistent Linear Projection of KV Summaries

## Abstract
First-token latency in autoregressive transformers is dominated by the O(T^2) prefill over T prompt tokens; exact KV reuse eliminates this only for identical prefixes. We propose CP-KVP (Cross-Prompt Key–Value Projection), which enables cache reuse across semantically similar prompts by (i) learning a linear compression that maps per-layer, per-head KV tensors into m-slot summaries with explicit position handling, and (ii) learning a linear projector that transfers summaries across prompts. Our key technical contribution is a position-consistent alignment for both RoPE and absolute positional embeddings: we preserve each slot’s relative offset to the last prompt token rather than absolute indices, which is what next-token attention depends on. At inference, we retrieve a similar prompt, project its summaries to the target, align positions via relative offsets, and compute the first new token with a single forward pass—skipping the full prefill. We release code for small open-source models, including position_id overrides and RoPE phase correction, and pre-register falsification criteria balancing latency gains against distributional drift and task quality.

## 1. Introduction
- Problem: Prefill dominates first-token latency; cache reuse is limited to exact string prefixes.
- Hypothesis: For semantically similar prompts, the effect of the prefix on the next-token computation lies on a low-dimensional, locally linear manifold per head/layer; a learned linear map can transfer compact cache summaries between prompts.
- Key positional insight: Next-token attention depends on relative offsets between the last-token query and key positions. RoPE is equivariant to global index translation; preserving relative offsets (not absolute indices) is the correct alignment.
- Contributions:
  1) A per-head linear KV compression into m-slot summaries that preserve last-position attention, with explicit storage of relative offsets.
  2) A per-layer slot-space linear projector learned from paraphrase pairs to transfer summaries across prompts; optional per-head channel rescaling with rank-8 updates.
  3) An inference pipeline that replaces O(T^2) prefill with retrieval + projection + a single-token forward pass, with an optional background exact prefill.
  4) A reproducible protocol and code for small open-source LMs, with pre-registered success/failure criteria.

## 2. Method

### 2.1 Notation and setting
- Decoder-only transformer with L layers, H heads, head dim d.
- For a prompt p of length T, at layer ℓ and head h:
  - K_{ℓh} ∈ R[T,d], V_{ℓh} ∈ R[T,d]; the last-position query is Q^{(T)}_{ℓh}(p).
- We compress to m ≪ T summary slots per head:
  - Per-head summary S_{ℓh}(p) := (K'_{ℓh}(p), V'_{ℓh}(p)), with K', V' ∈ R[m,d].
  - Per-layer concatenated summary S_ℓ(p) ∈ R[m, 2H·d] formed by stacking heads along the channel dimension as [K'|V'].

### 2.2 Linear compression C with relative-offset bookkeeping
Goal: Given (K_{ℓh}, V_{ℓh}) and T, produce (K'_{ℓh}, V'_{ℓh}) that approximately preserves the attention output at the last prompt position.

- Length-agnostic pooling: Define normalized positions u_t = t/T. Use fixed triangular bases centered at c_j = (j − 0.5)/m with width Δ = 1/m:
  - W_T[j,t] = max(0, 1 − |u_t − c_j|/Δ); normalize rows to sum to 1.
- Per-head linear adapters (not shared): A^K_{ℓh}, A^V_{ℓh} ∈ R[d,d].
  - K'_{ℓh} = (W_T · K_{ℓh}) · A^K_{ℓh}
  - V'_{ℓh} = (W_T · V_{ℓh}) · A^V_{ℓh}
- Relative-offset storage: Compute slot centers t̂_{ℓh,j} = Σ_t W_T[j,t]·t and store offsets to the last token:
  - ô_{ℓh,j} = T − t̂_{ℓh,j} (float). Store per-head ô to support head-specific pooling.
- Training objective (adapters only): For each (p,ℓ,h), let O_{ℓh}(p) be the teacher attention output at position T using full KV. Optimize
  - L_C = Σ_{p,ℓ,h} || Attn(Q^{(T)}_{ℓh}, K'_{ℓh}, V'_{ℓh}) − O_{ℓh}(p) ||_2^2 + λ (||A^K_{ℓh}||_F^2 + ||A^V_{ℓh}||_F^2)
  - Q is frozen; backprop through softmax; d×d adapters keep optimization lightweight.

### 2.3 Cross-prompt slot-space projector M (and optional channel map P)
Goal: Map source prompt summaries to target summaries.

- Slot-space projector (per layer): M_ℓ ∈ R[m,m], shared across channels; applied as Ŝ_ℓ = (S_ℓ) ·_slots M_ℓ^T.
- Optional channel map (per layer): P_ℓ ∈ R[2H·d, 2H·d], block-diagonal per head with low-rank updates (rank-8 per head) around identity to allow mild channel reweighting.
- Training data: paraphrase pairs P = {(p_s, p_t)} built from Quora/PAWS and templated variants; we ensure no lexical identity and balance lengths.
- Training (ridge regression, closed-form or SGD):
  - argmin_{M,(P)} Σ_{(p_s,p_t)} || M_ℓ S_ℓ(p_s) − S_ℓ(p_t) ||_F^2 + γ||M_ℓ||_F^2
  - If P_ℓ used, minimize || (M_ℓ S_ℓ(p_s)) P_ℓ − S_ℓ(p_t) ||_F^2 + γ_c ||P_ℓ − I||_F^2.

### 2.4 Position mechanics and alignment (critical)
Principle: For next-token attention, logits depend on relative offsets between the last-token query and key positions. Shifting all indices by a constant preserves RoPE attention geometry.

- During summary creation (source prompt, length T_src): store per-slot offsets ô_{ℓh,j} = T_src − t̂_{ℓh,j}.
- For a target prompt of length T_tgt: reconstruct per-slot indices t̃_{ℓh,j} = T_tgt − ô_{ℓh,j} (one per head).
- RoPE models:
  - Training: apply RoPE to K before pooling so that K'_{ℓh}[j] is “pre-rotated” at t̂_{ℓh,j}.
  - Inference: interpret injected K' as if pre-rotated at t̃ by applying a one-time rotary phase correction with Δ = t̃ − t̂ per slot. Implementation detail:
    - For each head and rotary frequency band ω_k (k-th pair in rotary dims), rotate the 2D subspace by angle ω_k·Δ for keys only; queries for the current last token use the model’s standard position T_tgt. This rephases K to the target indices without touching Q.
- Absolute positional embeddings:
  - Provide per-slot position_ids = t̃ when injecting past_key_values; ensure the attention mask remains causal relative to the next token at index T_tgt.

Ablations: we compare (i) inherit absolute t̂, (ii) normalized rescale t̃ ≈ (t̂/T_src)·T_tgt, and (iii) relative-offset translation t̃ = T_tgt − ô; (iii) consistently yields the lowest logit MSE and KL.

### 2.5 Inference pipeline
- Cache library: For each stored prompt p_i, keep an embedding e(p_i), per-layer summaries {S_ℓ(p_i)}, and per-head per-slot offsets {ô_{ℓh}}.
- Given a new prompt p_t of length T_tgt:
  1) Retrieve nearest neighbor p_s via cosine similarity on e(·); if sim < τ, fall back to cold prefill.
  2) For each layer ℓ: Ŝ_ℓ = M_ℓ S_ℓ(p_s); optionally apply P_ℓ.
  3) Reconstruct per-head per-slot indices t̃_{ℓh} = T_tgt − ô_{ℓh} (from the source offsets).
  4) Inject past_key_values shaped [L][2][B,H,m,d] with Ŝ and t̃:
     - RoPE: apply per-slot Δ = t̃ − t̂ phase correction to K'; queries use standard T_tgt.
     - Absolute PEs: pass position_ids for cached slots and for token T_tgt; keep causal mask.
  5) Run a single forward pass over the last prompt token to produce logits and the first generated token.
  6) Optional: start exact prefill in the background; when done, swap in exact K/V without changing already-emitted tokens.

Minimal HF-style pseudocode (sketch):
- Model wrapper:
  - forward(input_ids, past_key_values, position_ids_cached, position_ids_current, rope_phase_overrides=None)
  - Supports arbitrary m per layer; applies RoPE phase correction to injected K if rope_phase_overrides is provided.
- Reference repo: scripts {make_summary.py, learn_projector.py, warm_start_infer.py}; unit tests assert numerical equality of RoPE rephasing to direct recomputation.

### 2.6 Complexity and storage
- Training: one teacher pass per sample to collect O_{ℓh} and Q^{(T)}; adapters via AdamW; projector via ridge.
- Storage per prompt: O(Σ_ℓ m·2H·d) floats + per-head offsets (m floats).
- Inference: retrieval + O(Σ_ℓ m·2H·d) matmuls + one-token forward pass; eliminates O(T^2) prefill for first token.

## 3. Experiments

### 3.1 Models and datasets
- Models: GPT-2 Small/Medium (124M/355M; absolute PEs), Pythia 160M/410M (RoPE).
- Datasets: Quora Question Pairs and PAWS for paraphrases; templated QA and summarization prompts to vary T and semantics.
- Retrieval encoders: MiniLM-L6 (default), and model-derived CLS for ablation.

### 3.2 Baselines
- Cold start (full prefill).
- Exact prefix cache (upper-fidelity lower-latency bound).
- Last-k: use true last m tokens’ KV as cache.
- Static learned prefix (prefix-tuning).
- No projection: reuse S(p_s) directly for p_t.
- Random neighbor: retrieval ablation.
- KV compression/reuse: heavy-hitter (H2O), SnapKV/StreamingLLM-style sparse/last-k (adapted to first-token).

### 3.3 Metrics
- Latency: wall-clock to first token (includes retrieval/projection).
- Fidelity: KL(p_warm || p_cold) at the first new token; logit MSE; per-layer cosine similarity of attention outputs at T_tgt.
- Quality: perplexity on held-out text; task metrics (QA F1, ROUGE-L).
- Robustness: performance vs. retrieval similarity, T, m, library size, and τ.

### 3.4 Success criteria (pre-registered)
For high-similarity pairs (cosine > 0.9):
- ≥ 40% mean reduction in first-token latency vs. cold start.
- Mean next-token KL ≤ 0.05.
- ≤ 1% absolute drop on task metrics.
We report per-model/dataset means with 95% CIs and the fraction of runs meeting each criterion.

### 3.5 Implementation details
- Framework: HuggingFace Transformers with hooks to inject arbitrary-length past_key_values, override position_ids for cached slots and current token, and apply RoPE phase corrections.
- Optimization: AdamW for A^K/A^V (lr=1e-3, 10–20 epochs), ridge γ=1e-3 for M, γ_c=1e-4 for P when enabled.
- Hyperparameters: m ∈ {8,16,32}; τ tuned on a dev set; ablations for per-layer vs shared adapters and projector variants (slot-only vs slot+channel).
- Reproducibility: fixed seeds; wall-clock on A100 and RTX 4090; unit tests cover (i) RoPE rephasing equivalence, (ii) causal mask with sparse cached indices, (iii) exactness on identical-prefix reuse.

## 4. Related Work
- KV cache compression/selection (H2O, SnapKV, StreamingLLM) reduce memory/compute but still require prefill; our method targets eliminating prefill for first-token latency by cross-prompt transfer.
- Retrieval-augmented generation and kNN-LM retrieve external content; CP-KVP projects internal KV summaries across prompts without modifying base weights.
- Prefix-tuning/soft prompts produce static prefixes; CP-KVP synthesizes prompt-specific pseudo-prefixes via projection.

## 5. Discussion
- Why linear can suffice: We empirically probe local linearity of last-position attention under semantic perturbations; small m and per-head adapters capture dominant variation with bounded drift.
- Position alignment matters: Preserving relative offsets to the last token respects RoPE’s translation equivariance and outperforms absolute inheritance and normalized rescaling in fidelity.
- Practicality: Retrieval and projection are negligible compared to prefill; background refinement enables safe deployment for long continuations.

## 6. Limitations
- Domain shift: Projectors trained on natural-language paraphrases may not transfer to code or chain-of-thought; domain-specific pairs likely needed.
- Retrieval coverage: Performance depends on library size and encoder quality; we characterize scaling and τ sensitivity.
- Engineering: Requires control over position_ids and RoPE phases; some architectures need custom kernels.
- Approximation error: Warm-started caches can drift; safety-critical settings should gate on similarity and optionally re-decode after exact prefill.

## 7. Conclusion
CP-KVP replaces expensive prefill with retrieval and position-consistent linear projection of compact per-head KV summaries, reducing first-token latency with controlled fidelity drift. Preserving relative offsets to the last token resolves the key positional alignment issue for both RoPE and absolute embeddings. Our open-source implementation and pre-registered criteria make the approach testable on small open-source LMs.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
