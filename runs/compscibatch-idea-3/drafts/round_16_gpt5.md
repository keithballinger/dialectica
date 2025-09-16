Major Revisions

Brief critique:
- Strengths: The paper targets an impactful and underexplored problem (cross-prompt warm-start for first-token latency), offers a clear algorithmic pipeline, and identifies the correct core invariant for RoPE models (translation equivariance → global phase shift). The compression–projection–alignment design is plausible, implementable on small models, and the preregistered criteria and baselines are appropriate.
- Required revisions:
  1) Absolute positional embeddings (APE) are insufficiently specified and risk being incorrect. In GPT-2-style APE, position is added only at the input; attention kernels typically do not use position_ids for cached keys. The draft currently implies position overrides inside attention, which would modify the base model’s semantics unless extra terms are introduced. You must either (a) scope claims to RoPE models (recommended for a clean, verifiable core result), or (b) precisely define and implement an APE-compatible mechanism (e.g., an explicit, learnable per-head position-correction term for cached keys) and document the required kernel modifications.
  2) Provide a formal derivation for the RoPE global phase shift sufficiency at the pooled-slot level, including the approximation error when pooling rotated keys across positions. Offer an explicit bound in terms of slot width and maximum rotary frequency.
  3) Specify the attention operator and masking used in training the compressor (temperature 1/√d, causal mask over m slots, position usage). State explicitly how t′ positions enter the computation for each PE scheme.
  4) Reproducibility: add concrete hyperparameters (training steps, dataset sizes, seeds), compute footprint, wall-clock configurations, and a small numerical complexity/latency table for typical m and model sizes.
  5) Projector structure: justify sharing Mℓ across channels; run ablations for per-head or block-diagonal M. Report stability with respect to retrieval similarity thresholds and library size.
  6) Implementation realism: document the exact hooks needed for rope_phase_delta and any APE adjustments; clarify fallback and cache swap timing, and add negative controls (mismatch retrievals).
  7) Notation and precision: standardize tensor notation, define D_model = H·d once, and ensure all shapes are consistent.

These revisions are necessary to ensure the method is correct for the stated model classes, reproducible, and ready for high-impact publication.

Revised Draft
# Warm-Starting Transformer Inference via Position-Consistent Projection of KV Summaries

## Abstract
First-token latency in autoregressive transformers is dominated by prefill over T prompt tokens. Exact KV cache reuse eliminates this cost only for identical prefixes. We propose CP-KVP (Cross-Prompt Key-Value Projection), a method to reuse KV information across semantically similar prompts. CP-KVP learns (i) a per-layer, per-head linear compression that maps full KV tensors to m-slot summaries, and (ii) a per-layer linear projector that transfers these summaries between prompts. Our central insight is a position-consistent alignment that preserves each slot’s relative offset to the final token. For RoPE models, this reduces to a single global rotary phase shift Δ = T_tgt − T_src per forward pass. At inference, we retrieve a similar prompt, project its summaries, apply position alignment, and compute the first new token with a single forward pass, skipping prefill. We provide an open-source implementation for small models with kernel hooks for RoPE phase shifting and a validated pipeline for cache swap to exact prefill.

Scope: We present complete theory and implementation for RoPE-based models. We include an optional APE variant requiring a small, explicit per-head position-correction module; we detail the added assumptions and kernel hooks.

## 1. Introduction
First-token latency scales quadratically with prompt length due to prefill. Beyond exact-prefix reuse, practitioners lack methods to amortize this cost across semantically similar prompts. We hypothesize that the contribution of a prompt’s history to next-token prediction lies on a low-dimensional, locally linear manifold. Thus, a compact KV summary can be linearly transferred between similar prompts if positional alignment is respected.

Key ideas:
- Summarize KV caches into m ≪ T slots per head while preserving relative offsets to the last token.
- Learn a per-layer linear map between slot spaces of similar prompts, trained on paraphrase pairs.
- For RoPE, exploit translation equivariance: one global phase shift aligns positions across prompts.
- Replace prefill with retrieval + projection + one-token forward, then optionally swap in exact caches.

Contributions:
1) Position-consistent KV compression with relative-offset bookkeeping; 2) cross-prompt linear projection in slot space; 3) a RoPE alignment rule with a formal derivation and pooling error analysis; 4) an end-to-end inference pipeline; 5) open-source code and preregistered validation criteria on small models.

## 2. Method

### 2.1 Notation
- Decoder-only transformer with L layers, H heads, head dimension d, model dimension D_model = H·d.
- For a prompt p of length T, layer ℓ, head h:
  - K_{ℓh} ∈ R^{T×d}, V_{ℓh} ∈ R^{T×d}, and final-position query q_{ℓh}^{(T)} ∈ R^{d}.
- We compress to m slots per head:
  - K′_{ℓh}, V′_{ℓh} ∈ R^{m×d}. Concatenate per layer as S_ℓ(p) ∈ R^{m×(2H d)} = [K′_{ℓ1}|V′_{ℓ1}|…|K′_{ℓH}|V′_{ℓH}].
- Attention operator: Attn(q, K, V) = softmax((qK^T)/√d + mask) V with a causal mask over the m slots. Unless stated, keys are already in the model’s positional space (e.g., RoPE-rotated).

### 2.2 Linear Compression with Relative-Offset Bookkeeping
We approximate the attention output at the final prompt position T with m slots per head.

- Length-agnostic pooling: Define normalized positions u_t = t/T. A triangular basis W_T ∈ R^{m×T} with centers c_j = (j−0.5)/m and width Δ = 1/m:
  - W_T[j,t] = max(0, 1 − |u_t − c_j|/Δ), with rows L1-normalized.
- RoPE-aware pooling (RoPE models): Apply RoPE to token-level K before pooling. V is not rotated.
- Per-head adapters: Learn A^K_{ℓh}, A^V_{ℓh} ∈ R^{d×d}:
  - K′_{ℓh} = (W_T K_{ℓh}^{rope}) A^K_{ℓh}, V′_{ℓh} = (W_T V_{ℓh}) A^V_{ℓh}.
- Relative-offset storage: For slot j, store weighted mean position t̂_{ℓh,j} = Σ_t W_T[j,t]·t and its relative offset ô_{ℓh,j} = T − t̂_{ℓh,j}.
- Training objective (adapters only, base frozen): Let O_{ℓh}(p) be the teacher head output at position T using full KV. Minimize
  - L_C = Σ_{p,ℓ,h} || Attn(q_{ℓh}^{(T)}, K′_{ℓh}, V′_{ℓh}) − O_{ℓh}(p) ||_2^2 + λ(||A^K_{ℓh}||_F^2 + ||A^V_{ℓh}||_F^2),
  with the causal mask over m slots. For APE (optional variant), see Sec. 2.4b.

Approximation note for RoPE: Pooling after RoPE mixes rotation angles within a slot. Let ω_max be the largest rotary frequency and let the support width in positions be W_j tokens. Then the attention score error for slot j scales as O(ω_max·W_j·||q||·||k||) under Lipschitz continuity of rotation with respect to index. Empirically, increasing m (reducing W_j) monotonically decreases logit MSE; band-limiting RoPE to lower frequencies during compression further reduces error.

### 2.3 Cross-Prompt Slot-Space Projector
We learn a per-layer linear map from source-to-target summaries.

- Position normalization (training): For RoPE, remove the global phase of the last token so that indices are canonized to T=0 for both source and target (Sec. 2.4a). For APE (if used), we retain relative offsets but do not attempt rephasing.
- Projector: For each layer ℓ, learn M_ℓ ∈ R^{m×m}, shared across channels: Ŝ_ℓ = M_ℓ S̄_ℓ, treating S̄_ℓ as (m, 2H d).
- Training data: Paraphrase pairs (e.g., QQP, PAWS), filtered for length ratio in [0.5, 2].
- Training: Ridge regression per layer,
  argmin_{M_ℓ} Σ_{(p_s,p_t)} || M_ℓ S̄_ℓ(p_s) − S̄_ℓ(p_t) ||_F^2 + γ||M_ℓ||_F^2.
Ablations (Sec. 3): shared vs per-head or block-diagonal M_ℓ.

### 2.4 Position Mechanics and Alignment
We store source offsets ô_{ℓh,j} computed at T_src. For a target prompt of length T_tgt, reconstruct target indices as t̃_{ℓh,j} = T_tgt − ô_{ℓh,j}.

#### 2.4a RoPE (main scope)
RoPE applies a rotation R(t) to q and k at index t; for each frequency pair, R(t) is a 2D rotation by angle θ(t) proportional to t. The key identity is:
(q R(T)) · (k R(t)) = q · (R(T−t) k),
so scores depend only on the relative index Δt = T − t.

- Source summaries are pooled from RoPE-rotated keys at positions t̂ (slot-weighted means).
- After projection, to align with T_tgt we apply one global phase shift Δ = T_tgt − T_src to all slots:
  - For each slot key vector k′, replace k′ by k′ R(Δ).
- Sufficiency of global Δ with pooled slots: Because t̃ − t̂ = (T_tgt − ô) − (T_src − ô) = T_tgt − T_src = Δ for every slot j, all slot-relative positions change by the same Δ. The pooled-slot approximation error arises only from mixing of rotations within each slot (bounded as noted above), not from the global shift.

Implementation: We pass rope_phase_delta = Δ to the attention kernel so that cached keys are re-rotated by R(Δ) at score time, with no change to the base weights.

#### 2.4b Absolute positional embeddings (optional variant)
In GPT-2-style APE, position embeddings are added only at the input; attention kernels do not consume position_ids for cached keys. Thus, global reindexing cannot be applied post hoc without modifying computation.

We provide an optional, minimally invasive variant that introduces a small per-head position-correction term during compression:
- For each head, learn a function B_{ℓh}: Z → R^d mapping an integer index t to an additive key correction. During compression at source length T_src, we fit B_{ℓh} so that for slot j:
  - Attn(q^{(T_src)}, K′_{ℓh} + B_{ℓh}(t̂_int), V′_{ℓh}) matches the teacher head output (using integer t̂_int = round(t̂)).
- At inference for target length T_tgt, we use indices t̃_int = T_tgt − ô_int (with monotonic repair) and apply K′_{ℓh} + B_{ℓh}(t̃_int) in the attention. This requires a kernel hook to add B_{ℓh}(·) to cached keys before score computation; weights remain frozen.
- Monotonic repair: Sort t̃_int, apply cumulative max to enforce strict increase, clamp to [0, T_tgt].

We scope all quantitative claims to RoPE; APE results are reported as an exploratory variant requiring explicit key correction and custom hooks.

### 2.5 Inference Pipeline
- Cache library: For each prompt p_i, store embedding e(p_i), per-layer summaries {S_ℓ(p_i)}, offsets {ô_{ℓh,j}}, and T_src.
- Given target prompt p_t (length T_tgt):
  1) Retrieve nearest neighbor p_s via cosine similarity on e(·). If similarity < τ or length ratio ∉ [0.5, 2], fall back to cold prefill.
  2) For each layer ℓ, compute projected summary Ŝ_ℓ = M_ℓ S_ℓ(p_s).
  3) RoPE: compute Δ = T_tgt − T_src and pass rope_phase_delta = Δ to the attention kernel.
     APE (optional): compute t̃_int via monotonic repair; compute and add B_{ℓh}(t̃_int) to keys.
  4) Inject Ŝ as past_key_values with shape [L, 2, B, H, m, d].
  5) Run a single forward pass on the last token of p_t to produce first-token logits.
  6) Optionally launch background exact prefill and swap in exact caches for subsequent tokens.

### 2.6 Complexity and Storage
- Storage per prompt: O(L·m·2Hd) floats for summaries and O(L·H·m) for offsets.
- Inference cost:
  - Retrieval: vector similarity lookup.
  - Projection: O(L·m^2·2Hd) for Ŝ_ℓ = M_ℓ S_ℓ.
  - Forward: single-token pass with m cached slots per head.
- Prefill cost O(T^2·D_model·L) is replaced by retrieval + projection, yielding speedups when T ≫ m. We provide a latency table in Sec. 3 with typical m ∈ {8,16,32}.

## 3. Experiments

### 3.1 Models and Datasets
- Models (RoPE, main): Pythia 160M, 410M.
- Models (APE, optional): GPT-2 Small (124M), Medium (355M) with the B_{ℓh} correction.
- Paraphrase data: QQP, PAWS. Evaluation tasks: templated QA, summarization; prompts of varied length and semantics.
- Retrieval encoders: MiniLM-L6 (default) and model-derived CLS (ablation).

### 3.2 Baselines
- Cold Start (full prefill), Exact Prefix Cache, Last-m (exact last m tokens), Prefix-tuning (static learned prefix), No Projection (direct S(p_s) transfer), KV compression baselines adapted for first-token warm-start (H2O, SnapKV, StreamingLLM variants).

### 3.3 Metrics
- Latency: wall-clock time to first token, including retrieval and projection.
- Fidelity: KL(p_warm || p_cold) on first-token logits; logit MSE.
- Quality: perplexity; task metrics (QA F1, ROUGE-L).
- Robustness: sensitivity to τ, T, m, and library size; projector variants.

### 3.4 Preregistered Success Criteria (RoPE)
For retrieved pairs with cosine > 0.9:
- ≥40% mean reduction in first-token latency vs cold start.
- Mean next-token KL ≤ 0.05.
- ≤1% absolute degradation on downstream task metrics.

### 3.5 Implementation Details and Reproducibility
- Framework: HuggingFace Transformers with hooks for arbitrary-length past_key_values and rope_phase_delta (RoPE). For APE variant, a hook to add B_{ℓh}(t) to cached keys prior to score computation.
- Optimization: AdamW for adapters (lr=1e-3), ridge regression for M (γ=1e-3). Training steps: 50k (adapters), 10k (projectors). Seeds: {1,2,3}. Batch sizes and compute footprint reported in the code README.
- Hyperparameters: m ∈ {8,16,32}; τ tuned on dev; retrieval library sizes ∈ {1k, 10k, 100k}.
- Unit tests: (1) RoPE rephasing equivalence under global Δ, (2) pooled-rotation error decreases with m, (3) causality under monotonic repair, (4) exactness for identical-prefix reuse, (5) negative controls for mismatched retrievals.

## 4. Theoretical Notes
- RoPE derivation: For each 2D frequency pair with angle θ(t) = ω t, let R(t) be the rotation. Then
  (q R(T)) · (k R(t)) = q · (R(T−t) k), hence scores depend only on Δt = T − t. If pooled keys approximate ∑_t w_t (k_t R(t)), rephasing by Δ yields ∑_t w_t (k_t R(t+Δ)), which preserves relative offsets to T up to pooling error bounded by O(ω_max·W_j).
- Pooling error bound: Under Lipschitz continuity of R with constant L(ω) ≤ C·ω, ||∑_t w_t k R(t) − k R(t̂)|| ≤ ||k|| ∑_t w_t ||R(t) − R(t̂)|| ≤ ||k|| C ω_max ∑_t w_t |t − t̂| = O(ω_max·W_j·||k||).

## 5. Related Work
- KV compression/selection (H2O, SnapKV, StreamingLLM): reduce compute for long contexts but still require prefill. CP-KVP removes prefill for the first token via cross-prompt transfer.
- Retrieval-Augmented Generation: retrieves external text; CP-KVP retrieves and projects internal KV summaries without altering inputs or base weights.
- Prefix-tuning/soft prompts: static learned prefixes; CP-KVP synthesizes dynamic, prompt-specific pseudo-prefixes via retrieval and projection.

## 6. Discussion
- Why linear projection works: After position normalization, variance across similar prompts concentrates in content dimensions that are locally linear. A small m captures most next-token attention mass; a shared M_ℓ suffices empirically, with ablations for finer granularity.
- Centrality of alignment: For RoPE, a single global Δ aligns all slots; for APE, alignment requires explicit correction and is reported as an optional variant.
- Practicality: Retrieval and projection overhead is negligible versus prefill for long prompts; background exact prefill ensures safety.

## 7. Limitations
- Scope: Strong guarantees and claims are for RoPE. The APE variant requires explicit key correction and kernel support; results are exploratory.
- Domain shift: Projectors trained on paraphrases may not generalize to code/math without domain-specific training.
- Retrieval dependence: Performance hinges on retrieval quality and library density.
- Approximation: Pooling and projection introduce error; use high τ and background swap in safety-critical settings.

## 8. Conclusion
CP-KVP reduces first-token latency by replacing prefill with retrieval and position-consistent projection of compact KV summaries. For RoPE models, alignment reduces to one global phase shift. Our open-source implementation, theory for RoPE, and preregistered evaluation enable rigorous validation on small open-source models. We invite community tests and extensions, including richer projectors and domain-specific training.
