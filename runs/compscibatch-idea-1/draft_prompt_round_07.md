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
- The revised draft is technically sound, novel (combining entropy-based exits with a monotonic KV-consistency constraint in a training-free manner), and aligns with constraints: focused on LLM inference, validatable via code on OSS models like Pythia/Llama, with a falsification plan suitable for a leading journal (e.g., NeurIPS). The proof is formalized as suggested, and core ideas are well-motivated.
- However, the pseudocode could be restructured for clarity by explicitly showing state management (e.g., D_t updates) and handling batches.
- The compute budget to quantile mapping is still slightly vague; add a sentence clarifying how q is selected (e.g., binary search on validation FLOPs).
- Motivate m more explicitly as guarding against proxy estimation errors.
- Add a concrete formula for proxy overhead (e.g., FLOPs = d·r + r·|V| vs. block FLOPs ≈ 2d·d_ffn + 2d^2 for attention/MLP).
- Tighten abstract and intro for conciseness (e.g., merge redundant sentences); minor typos (e.g., "Wᵀ" should be consistent).

Revised Draft
# Entropy-Matched Adaptive-Depth Decoding

## Abstract
We introduce Entropy-Matched Adaptive-Depth (EMAD) decoding, a training-free method to reduce autoregressive inference costs in Large Language Models (LLMs). EMAD dynamically truncates forward passes by exiting at intermediate layers when predictive uncertainty meets layer-specific targets derived from the model's teacher-forcing entropy profile. This ensures model- and scale-aware decisions. We enforce a monotone non-increasing depth schedule across tokens, guaranteeing KV-cache consistency without backfilling or retraining. Low-overhead entropy proxies minimize decision costs. A GPU-aware, depth-stratified batching strategy converts savings to wall-clock speedups. Validation on open models targets ≥10% wall-clock speedup at matched accuracy or ≥15% FLOP reduction with <1% accuracy loss.

## 1. Introduction
Inference in decoder-only Transformers scales with sequence length and model depth. Many tokens are "easy," not requiring full depth, but per-token dynamic depths complicate KV-cache consistency: attention at layer ℓ needs prior tokens' KVs up to ℓ.

Existing methods often maintain fixed depths, reducing sequential steps (e.g., speculative decoding) or intensity (e.g., quantization). Early-exit techniques for classifiers require retraining and auxiliary heads, ignoring autoregressive KV issues.

EMAD addresses this via:
1. **Calibrated Exit Rule:** Exit when uncertainty (via low-cost proxy) falls below layer-wise targets from a pre-computed entropy profile, matching typical certainty at that depth.
2. **KV-Consistent Schedule:** Enforce monotone non-increasing depths (D_t ≤ D_{t-1}), ensuring cache validity without overhead.

Contributions: (1) Training-free entropy-matching exits. (2) Calibrated low-overhead proxies. (3) Provably correct monotonic scheduling with GPU batching. (4) Falsification-oriented experiments on OSS models.

## 2. Method

### 2.1 Preliminaries
Consider a decoder-only Transformer with L blocks. Let h_{ℓ,t} be the hidden state for token t after block ℓ. The LM head (Norm(·) and unembedding W) yields logits z_{ℓ,t} = W^T Norm(h_{ℓ,t}). Predictive entropy is H_{ℓ,t} = H(softmax(z_{ℓ,t}/T)), with temperature T. Goal: Choose D_t ≤ L per token to minimize computation while preserving accuracy.

### 2.2 KV-Consistent Monotonic Scheduling
Enforce: D_t ≤ D_{t-1} for t > 1, D_1 = L.

**Claim:** This ensures KV-cache validity for any t and k ≤ D_t (KVs for tokens 1..t-1 exist at k).

**Proof by induction:**
- Base (t=2): D_1=L; for k ≤ D_2 ≤ L, token 1's KVs at k exist.
- Step: Assume for t. For t+1, D_{t+1} ≤ D_t. For k ≤ D_{t+1} ≤ D_t, induction gives KVs for 1..t-1 at k; token t's KVs at k exist since k ≤ D_t.

**Extensions:** Add floors (D_t ≥ F_t, slow-decreasing) or windowed recovery, evaluated as ablations, to prevent collapse.

### 2.3 Teacher-Forcing Entropy Profile
Calibrate offline on dataset C:
1. **Collection:** For probe layers P ⊆ {1..L}, compute H_{ℓ,t} under teacher forcing, using final Norm for consistency.
2. **Targets:** τ_ℓ = q-quantile of {H_{ℓ,t}} for ℓ ∈ P. Select q via binary search on validation set to meet user FLOP budget (e.g., target average depth).
3. **Monotonicity:** Apply isotonic regression to ensure τ_ℓ non-increasing.

### 2.4 Low-Overhead Entropy Proxies
Proxies approximate H_{ℓ,t} cheaply:
- **Low-Rank Readout:** Approximate W ≈ U V^T (U ∈ ℝ^{|V|×r}, V ∈ ℝ^{d×r}, r ≪ min(d, |V|)). Logits ˆz_{ℓ,t} = U^T (V^T Norm(h_{ℓ,t})). FLOPs: d·r + r·|V| (e.g., <5% of block FLOPs ≈ 2d·d_ff + 2d^2 for r=128, d=4096, |V|=50272).
- **Calibration:** Fit Ĥ_{ℓ,t} = f(p_{ℓ,t}) via isotonic regression on C, where p_{ℓ,t} are statistics of ˆz_{ℓ,t}.

### 2.5 Dynamic Exit Rule
For token t (max depth D_{t-1}):
- Forward layer-by-layer up to D_{t-1}.
- At each probe ℓ ≤ D_{t-1}: Compute Ĥ_{ℓ,t}; if Ĥ_{ℓ,t} ≤ τ_ℓ - m, set D_t = ℓ and exit (m is safety margin against proxy errors, tuned on validation for robustness).
- Default: D_t = D_{t-1}.
- Sample y_t from logits at D_t.

### 2.6 System Implementation
- **Probes:** Sparse, early-biased (e.g., every 2-4 layers).
- **Fused Kernels:** Combine Norm, projection, features.
- **Batching:** Group sequences by current max depth; execute in depth segments, peeling off shallow groups.

## 3. Falsification Plan

### 3.1 Models and Data
- Models: Pythia-{410M,1.4B}, OPT-1.3B, Llama-1-7B (bfloat16/fp16).
- Calibration: 20M tokens from The Pile.
- Tasks: WikiText-103 (PPL), ARC/GSM8K/HumanEval (accuracy).
- Decoding: Greedy, nucleus (top-p=0.9, T=0.7/1.0).

### 3.2 Baselines
- Full-depth.
- Fixed-depth truncation.
- Global entropy threshold.
- Speculative decoding (for complementarity).

### 3.3 Metrics
- Accuracy: Relative PPL change, absolute task deltas.
- Efficiency: FLOP savings, tokens/sec, GPU FLOPs.
- Overhead: Proxy/decision fraction.

### 3.4 Ablations
- Proxy: r impact, error effects.
- Scheduling: Monotonic vs. floored.
- Calibration: Size/domain sensitivity.
- Targeting: q trade-offs.

### 3.5 Falsification Criteria
Ineffective if: >1% accuracy drop at ≥15% FLOP reduction, or <10% speedup at matched accuracy.

## 4. Relation to Prior Work
EMAD is training-free adaptive-depth for autoregressive LLMs, unlike retraining-based early exits or fixed-depth accelerators. Monotonic scheduling simplifies KV handling vs. complex alternatives. Complements speculative decoding by reducing depth, not steps.

## 5. Conclusion
EMAD reduces LLM inference costs via entropy-matched exits and monotonic KV consistency. Its falsification plan enables code-based validation on OSS models.

## 6. Pseudocode

```python
# Per-sequence state: D_prev, kv_cache

def emad_token_step(h_in, D_prev, kv_cache, model):
    """Stateful per-token EMAD forward pass."""
    h = h_in
    D_t = D_prev  # Initialize

    for l in range(1, D_prev + 1):
        h = model.blocks[l](h, kv_cache)  # Uses cache up to l (<= D_prev)

        if l in PROBE_LAYERS:
            h_norm = model.Norm(h)
            entropy_est = model.calibrated_proxy(h_norm, l)
            if entropy_est <= TARGET_ENTROPY[l] - SAFETY_MARGIN:
                D_t = l
                break

    # Final logits from D_t's h
    final_h = model.Norm(h)
    logits = final_h @ model.W_unembed.T

    # Update cache only up to D_t
    update_kv_cache(kv_cache, h_in, h, depth=D_t)  # Commit KVs for layers 1 to D_t

    return logits, D_t  # D_t for next token's D_prev
```


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
