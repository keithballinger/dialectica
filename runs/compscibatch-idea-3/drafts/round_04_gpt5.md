Major Revisions

Brief critique
- The core idea—projecting KV caches across semantically similar prompts—is interesting and timely, but key methodological details are incorrect or underspecified.
- The “ridge regression” claim for learning the compression map C is not valid as written: the objective backpropagates through a softmax attention operator and is non-linear. A correct training procedure must use gradient-based optimization or a different convex surrogate.
- Variable-length handling and position encodings are not rigorously addressed. For both absolute embeddings and RoPE, correct placement of pseudo-positions and query position-ids is essential; “positions 0..m−1” is generally wrong and will cause large errors.
- The inference path omits that one still must process the final prompt token with the projected cache to produce the first new token. This is a critical point for latency accounting.
- Background cache “swap” needs care: swapping prompt K/V after some tokens have been generated is acceptable for future steps, but it cannot retroactively fix already-emitted tokens; clarify guarantees and when swap occurs.
- Compression shape and linear maps are ambiguous for variable T; define a length-agnostic scheme.
- Some baselines are ill-posed (“initialize only the final hidden state”). Replace with more realistic baselines (e.g., last-k tokens, static learned prefix, zero-shot speculative warmup).
- Stronger notation and explicit training/optimization details are needed to make this reproducible on small open-source models.

Revised Draft
Title
Warm-Starting Transformer Inference via Position-Aware Linear Projection of KV Caches

Abstract
First-token latency in autoregressive transformers is dominated by the O(T^2) prefill over T prompt tokens. Exact KV cache reuse removes this cost only for identical prefixes. We propose CP-KVP, a method that extends cache reuse to semantically similar prompts by (i) learning a position-aware linear compression that maps a full per-layer KV tensor to m “summary slots,” and (ii) learning a linear projector that maps summaries across prompts. At inference, we retrieve a similar source prompt, project its summary into a target summary, and compute the first new token by processing only the final prompt token against this projected cache—skipping the full prefill. We provide a reproducible protocol on small open-source models and explicit falsification criteria: substantial first-token latency reduction with tight bounds on next-token distributional drift and task quality.

1. Introduction
- Problem: Prefill is the dominant first-token latency. Cache reuse works only for exact string prefixes.
- Hypothesis: For semantically similar prompts, the functional effect of the prefix on the next-token computation lies in a low-dimensional, locally linear manifold per head/layer; a learned linear map can transfer compact cache summaries between prompts.
- Contributions:
  1) A position-aware linear compression that produces m-slot summaries preserving the next-token attention outputs.
  2) A cross-prompt linear projector learned from paraphrase pairs to transfer summaries between prompts.
  3) An inference pipeline that replaces the O(T^2) prefill with retrieval + projection + a single-token forward pass, plus an optional exact-prefill background refinement.
  4) A reproducible protocol with pre-registered success/failure criteria on small open-source models.

2. Method
2.1 Notation and setting
- Decoder-only transformer with L layers, H heads, head dimension d.
- For prompt p of length T, at layer ℓ and head h:
  - Keys K_{ℓh} ∈ R^{T×d}, Values V_{ℓh} ∈ R^{T×d}.
  - The next-token (last prompt position) query at layer ℓ is computed online during inference; no precomputation is assumed.
- We choose m ≪ T summary slots per head. Let S_{ℓh}(p) := (K'_{ℓh}(p), V'_{ℓh}(p)) with K', V' ∈ R^{m×d}.

2.2 Position-aware linear compression (C)
Goal: Given (K_{ℓh}, V_{ℓh}) and prompt length T, produce (K'_{ℓh}, V'_{ℓh}) that approximately preserves the attention output for the next-token position across layers/heads.

Design (length-agnostic pooling + linear adapters):
- Define a deterministic normalized pooling W_T ∈ R^{m×T} that maps token positions 1..T to m bins along normalized position u = t/T. We use fixed triangular basis functions centered at c_j = (j−0.5)/m:
  - For t ∈ {1..T}, u_t = t/T.
  - W_T[j, t] = max(0, 1 − |u_t − c_j|/Δ) with Δ = 1/m, followed by row-normalization.
- Apply shared per-layer/head linear adapters A^K_{ℓh}, A^V_{ℓh} ∈ R^{d×d}:
  - K'_{ℓh} = (W_T K_{ℓh}) A^K_{ℓh}
  - V'_{ℓh} = (W_T V_{ℓh}) A^V_{ℓh}
This is linear in (K,V), length-agnostic, and cheap.

Position handling:
- For RoPE models, we assign absolute indices to summary slots as the W_T-weighted average of original indices:
  - t̂_{ℓh,j} = Σ_t W_T[j,t] · t
  - Store K' after applying RoPE rotation at phase index t̂_{ℓh,j}; at inference, pass the query with position-id T (overriding default m).
- For absolute position embeddings, we likewise override position_ids at inference to T for the current token; we store per-slot absolute ids t̂_{ℓh,j} alongside K',V' and ensure the implementation uses these ids when computing attention scores.

Training objective for C (gradient-based, not closed-form regression):
- For a dataset D of prompts, we compute teacher signals once via a standard forward pass:
  - For each p ∈ D, and each (ℓ,h), collect O_{ℓh}(p) = Attn(Q_{ℓh}^{(T)}(p), K_{ℓh}(p), V_{ℓh}(p)), the attention output at the last prompt position T using the true cache.
- Optimize A^K_{ℓh}, A^V_{ℓh} by minimizing:
  ```
  L_C = Σ_{p∈D} Σ_{ℓ,h} || Attn(Q_{ℓh}^{(T)}(p),
                                  K'_{ℓh}(p; A^K),
                                  V'_{ℓh}(p; A^V)) − O_{ℓh}(p) ||_2^2
        + λ (||A^K||_F^2 + ||A^V||_F^2)
  ```
- We backprop through the softmax attention with Q frozen to teacher values; A^K,A^V are small (d×d), making this optimization fast.

2.3 Cross-prompt summary projector (M)
Goal: Map the source prompt summary to a target prompt summary.

Representation:
- Concatenate per-head summaries within a layer into S_ℓ(p) ∈ R^{m×(2Hd)} by stacking [K'|V'] over heads.
- Learn a per-layer linear map M_ℓ ∈ R^{m×m} acting along the slot dimension (shared across feature channels), or a block-diagonal map over heads if preferred.

Data and training (convex, closed-form):
- Build a set P = {(p_s, p_t)} of semantically similar pairs (e.g., paraphrases).
- Targets are S_ℓ(p_t); inputs are S_ℓ(p_s).
- Ridge regression per layer:
  ```
  M_ℓ = argmin_M Σ_{(p_s,p_t)∈P} || M S_ℓ(p_s) − S_ℓ(p_t) ||_F^2 + γ ||M||_F^2
  ```
This is standard and has an efficient closed form or can be solved with LSQR.

2.4 Inference pipeline
- Cache library: For each past prompt p_i, store embedding e(p_i) (e.g., MiniLM or model-derived), and layered summaries {S_ℓ(p_i)}.
- Given new prompt p_t:
  1) Retrieve nearest neighbor p_s by cosine similarity over e(·). If similarity < τ, fall back to cold prefill.
  2) For each layer ℓ, compute projected summary Ŝ_ℓ(p_t) = M_ℓ S_ℓ(p_s). Retain per-slot position ids t̂_{ℓ,j} from the compression rule (we reuse the same W_T rule for the length T of p_t to set the query position-id to T and to annotate Ŝ_ℓ with t̂ appropriate for T).
  3) Initialize past_key_values with Ŝ_ℓ(p_t). Ensure:
     - For RoPE: keys in Ŝ are pre-rotated at t̂_{ℓ,j}; pass query with position-id T.
     - For absolute PEs: override position_ids for the current token to T; store per-slot absolute ids t̂ and ensure the kernel uses them in attention score computation.
  4) Process only the final prompt token x_T once to compute logits at position T and generate the first new token. No O(T^2) prefill is performed.
  5) Optional background refinement: start exact prefill on p_t; upon completion, replace only the prompt segment of past_key_values. This does not retroactively change already-emitted tokens but improves fidelity for subsequent steps.

2.5 Complexity and storage
- Training:
  - Teacher pass to collect O_{ℓh} and Q_{ℓ}^{(T)} per prompt (once).
  - Optimize A^K,A^V via lightweight gradient descent; learn M_ℓ via ridge regression.
- Storage per prompt: O(Σ_ℓ m · 2Hd) floats + per-slot position ids; typically orders of magnitude smaller than full caches.
- Inference time: retrieval + matrix multiplications O(Σ_ℓ m · 2Hd) + one token forward pass. Eliminates O(T^2) prefill.

3. Experiments
3.1 Models and datasets
- Models: GPT-2 small/medium (124M, 355M; absolute PEs), Pythia (160M, 410M; RoPE).
- Datasets:
  - Paraphrases: Quora Question Pairs, PAWS.
  - Instruction-style: templated QA, summarization prompts.
- Retrieval encoder: MiniLM-L6 or model-derived CLS pooled representation; sensitivity analysis across encoders.

3.2 Baselines
- Cold start (full prefill).
- Exact prefix cache (ideal lower-latency/upper-fidelity bound).
- Last-k tokens: use the true last m tokens’ KV as cache (structure-preserving baseline).
- Static learned prefix: task-level learned prefix of length m (prefix-tuning style).
- No projection: use S(p_s) directly for p_t.
- Random neighbor: use projector from a mismatched prompt to test retrieval dependency.

3.3 Metrics
- Latency: wall-clock to first generated token, including retrieval and projection.
- Distributional fidelity: KL(p_warm || p_cold) at the first new token; per-layer cosine similarity of attention outputs at position T.
- Quality: perplexity on held-out text; task metrics (F1 for QA, ROUGE-L for summarization).
- Robustness: performance vs. retrieval similarity, prompt length T, m.

3.4 Success criteria (pre-registered)
For high-similarity pairs (cosine > 0.9):
- ≥40% mean reduction in first-token latency vs. cold start.
- Mean next-token KL ≤ 0.05.
- ≤1% absolute drop on task metrics.
Failure to meet any criterion falsifies the hypothesis.

3.5 Implementation details
- HuggingFace Transformers with custom position_id overrides for both absolute embeddings and RoPE.
- RoPE handling: store pre-rotated K' with per-slot absolute indices t̂; set query index to T at inference.
- Optimization: AdamW for A^K,A^V (1e−3, 10–20 epochs), ridge γ=1e−3 for M.
- m in {8, 16, 32}; ablation across per-layer vs. shared A^K,A^V.

4. Discussion
- Relation to prefix-tuning and retrieval-augmented inference: Unlike static prefixes, CP-KVP produces prompt-specific pseudo-prefixes via linear projection, without finetuning the base model.
- Why linear can suffice: empirical evidence suggests local linearity of attention outputs under small semantic perturbations; we test this via falsification criteria.
- Background refinement: swapping prompt K/V after emission of the first token cannot change past outputs but improves future steps; in safety-critical contexts, either disable warm start or re-decode once exact prefill completes.

5. Limitations
- Domain shift: projectors trained on paraphrases may not transfer to reasoning-heavy or coding prompts; incorporate domain-specific P sets.
- Retrieval coverage: method relies on a dense prompt library; quantify performance vs. library size.
- Positional mechanics: requires careful control of position_ids; some model implementations may not expose this cleanly.
- Approximation error: warm-started cache introduces bounded drift; for strict-fidelity applications, fallback or re-decode is needed.

6. Conclusion
CP-KVP replaces expensive prefill with retrieval and position-aware linear projection of compact KV summaries, enabling substantial first-token latency reductions with controllable fidelity trade-offs. The method is simple, model-agnostic, and testable on small open-source LMs. We provide a clear protocol and falsification criteria to assess when linear cache projection works—and when it does not.
