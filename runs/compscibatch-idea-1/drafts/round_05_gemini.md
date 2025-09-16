Minor Revisions

Brief critique:
- The draft is strong, with a clear method and rigorous validation plan. The core ideas—entropy matching for an exit criterion and a monotone depth schedule for KV-cache consistency—are well-motivated and sound.
- The proof of KV-consistency, while correct, can be stated more formally as a brief proof by induction for improved rigor.
- The relationship between the user-specified compute budget, the resulting quantile targets `τℓ`, and the safety margin `m` is slightly underspecified. The paper would benefit from a clearer explanation of how a budget maps to quantiles and a more explicit motivation for the safety margin `m` as a separate hyperparameter.
- The pseudocode is functional but could be restructured for clarity, treating the per-token decoding step as a stateful function that explicitly manages `Dt`.
- The computational overhead of the entropy proxy should be briefly analyzed with a formula to make the `<5%` claim more concrete.
- The presentation is clear but could be tightened in several places for conciseness, particularly in the abstract and introduction.

Revised Draft
# Entropy-Matched Adaptive-Depth Decoding

## Abstract
We introduce Entropy-Matched Adaptive-Depth (EMAD) decoding, a training-free method to reduce the computational cost of autoregressive inference in Large Language Models (LLMs). EMAD dynamically truncates the forward pass, exiting at an intermediate layer if the model's predictive uncertainty meets a pre-calibrated, layer-specific target. These targets are derived from the model's own teacher-forcing entropy profile, making exit decisions model- and scale-aware. Crucially, we enforce a monotone non-increasing depth schedule across tokens, a simple constraint that rigorously guarantees KV-cache consistency without backfilling or retraining. EMAD uses low-overhead entropy proxies to keep the decision cost minimal. We propose a GPU-aware, depth-stratified batching strategy to translate computational savings into wall-clock speedup. We validate EMAD via a falsification-oriented plan on open models, defining success as achieving ≥10% wall-clock speedup at matched accuracy or ≥15% FLOP reduction with <1% accuracy loss.

## 1. Introduction
The inference cost of decoder-only Transformers scales with both sequence length and model depth. While many tokens in a sequence are "easy" and do not require the full network depth to be predicted accurately, dynamically varying the computation depth on a per-token basis is non-trivial. The primary obstacle is the Key-Value (KV) cache: self-attention at layer `ℓ` requires access to cached keys and values from all prior tokens computed up to layer `ℓ`. Naively exiting early for one token breaks this dependency for subsequent tokens.

Existing acceleration methods often preserve a fixed-depth computation path, focusing instead on reducing sequential steps (e.g., speculative decoding) or operational intensity (e.g., quantization). Early-exit techniques developed for classifiers typically require auxiliary heads and retraining, and they do not address the KV-cache consistency problem in autoregressive generation.

EMAD provides a practical solution by jointly addressing the exit criterion and the KV-cache constraint:
1.  **A Calibrated Exit Rule:** We exit at the first layer where the model's predictive uncertainty, estimated via a low-cost proxy, falls below a target. This target is not a global constant but is derived from a layer-wise entropy profile pre-computed on a calibration set, effectively asking the model to stop when it is "as certain as it usually is" at that depth.
2.  **A KV-Consistent Schedule:** We enforce a monotone non-increasing depth schedule (`Dt ≤ Dt-1`). This simple rule guarantees that the KV cache is always valid for any chosen exit depth `Dt`, eliminating the need for complex management or costly backfilling.

Our contributions are: (1) A principled, training-free exit criterion based on matching a model's own teacher-forcing entropy profile. (2) Low-overhead, calibrated entropy proxies that avoid full softmax computations at intermediate layers. (3) A provably correct monotonic scheduling rule for KV-cache consistency, paired with a GPU-aware batching strategy. (4) A rigorous, falsification-oriented experimental plan on open-source models.

## 2. Method

### 2.1 Preliminaries
We consider a decoder-only Transformer with `L` blocks. Let `hℓ,t` be the hidden state for token `t` after block `ℓ`. The model's language model (LM) head, consisting of a final normalization `Norm(·)` and an unembedding matrix `W`, can be applied to any `hℓ,t` to produce logits `zℓ,t = WᵀNorm(hℓ,t)`. The predictive entropy at layer `ℓ` is `Hℓ,t = H(softmax(zℓ,t/T))`, where `T` is the decoding temperature. Our goal is to choose a depth `Dt ≤ L` for each token `t` to minimize computation while preserving accuracy.

### 2.2 KV-Consistent Monotonic Scheduling
To ensure KV-cache validity, we enforce the following constraint on the sequence of per-token depths `{D_t}`:
**Monotonicity Constraint:** `Dt ≤ Dt−1` for all `t > 1`, with `D1 = L`.

**Claim:** This constraint guarantees that for any token `t` and any layer `k ≤ Dt`, the KV-cache contains valid keys and values for all previous tokens `1, ..., t-1` at layer `k`.

**Proof (by induction on `t`):**
- **Base Case (t=2):** `D1=L`. For any `k ≤ D2`, we have `k ≤ D1`, so the KVs for token 1 at layer `k` exist.
- **Inductive Step:** Assume the claim holds for token `t`. For token `t+1`, we choose `Dt+1 ≤ Dt`. For any layer `k ≤ Dt+1`, the monotonicity implies `k ≤ Dt ≤ Dt-1 ≤ ... ≤ D1`. By the inductive hypothesis, the KVs for tokens `1..t-1` exist at layer `k`. Since `k ≤ Dt`, we computed and cached KVs for token `t` at layer `k`. Thus, the cache is valid for token `t+1` at layer `k`.

**Optional Extensions:** To mitigate the risk of premature depth collapse, this strict rule can be augmented with a floor (`Dt ≥ F_t`, where `F_t` decreases slowly) or bounded-window recovery mechanisms, which we evaluate as ablations.

### 2.3 Teacher-Forcing Entropy Profile
We calibrate the exit criterion using a single, offline pass over a calibration dataset `C`.
1.  **Data Collection:** For a sparse set of probe layers `P ⊆ {1,…,L}` and each token in `C`, we compute the predictive entropy `Hℓ,t` under teacher forcing. The logits `zℓ,t` are always computed using the model's final `Norm` layer for consistency. This step is performed for each target decoding configuration (e.g., temperature).
2.  **Target Profile:** For each probe layer `ℓ ∈ P`, we define the entropy target `τℓ` as a quantile `q` of the collected entropies `{Hℓ,t}`. The quantile `q` is a hyperparameter that directly trades off performance and computational savings; it can be selected by performing a simple sweep on a held-out set to meet a desired FLOP budget.
3.  **Monotonicity:** We enforce that `τℓ` is non-increasing with `ℓ` via isotonic regression. This reflects the natural trend of uncertainty reduction with depth.

### 2.4 Low-Overhead Entropy Proxies
Evaluating the full entropy `Hℓ,t` at each probe layer is prohibitively expensive. We instead use cheap, low-overhead proxies that are calibrated to predict the true entropy.
- **Low-Rank Readout:** We approximate the unembedding matrix `W ≈ UVᵀ`, where `U ∈ R^(|V|×r)`, `V ∈ R^(d×r)`, and `r ≪ d, |V|`. The logits are approximated as `ẑℓ,t = Norm(hℓ,t)V Uᵀ`. The cost of this proxy projection, `O(d·r + r·|V|)`, is much lower than a full Transformer block's `O(d·d_ffn)` and can be kept below 5% of a block's FLOPs for small `r` (e.g., 64-256).
- **Calibration:** We learn a simple function `Ĥℓ,t = f(pℓ,t)` that maps proxy features `pℓ,t` (e.g., statistics from `ẑℓ,t`) to an entropy estimate. We use isotonic regression for `f`, trained on the calibration set, to ensure a monotonic relationship.

### 2.5 Dynamic Exit Rule
At inference time, for each token `t` with previous depth `Dt-1`, we perform the forward pass layer by layer. For each probe layer `ℓ ∈ P` such that `ℓ ≤ Dt-1`:
1.  Compute the entropy proxy and its calibrated estimate `Ĥℓ,t`.
2.  If `Ĥℓ,t ≤ τℓ`, we set `Dt = ℓ` and terminate the forward pass.
A small safety margin `m` can be subtracted from `τℓ` for robustness, tuned on a validation set. If no probe layer satisfies the condition, we set `Dt = Dt-1`. The next token `y_t` is then sampled from the logits computed at layer `Dt`.

### 2.6 System Implementation
- **Probe Placement:** Probes are placed sparsely (e.g., every 2-4 layers), biased towards earlier layers where exits are more probable.
- **Fused Kernels:** The `Norm`, low-rank projection, and proxy feature extraction are implemented as a single fused kernel to minimize memory bandwidth overhead.
- **Depth-Stratified Batching:** To maintain high GPU utilization with varying per-sequence depths, we use depth-stratified execution. Within a batch, sequences are grouped by their maximum allowed depth (`Dt-1`). The forward pass is executed in segments, with sequences in shallower groups "peeling off" and skipping computation for deeper blocks.

## 3. Falsification Plan

### 3.1 Models and Data
- **Models:** Pythia-{410M, 1.4B}, OPT-1.3B, Llama-1-7B (using standard bfloat16/fp16 checkpoints).
- **Calibration Data:** 20M tokens from The Pile.
- **Evaluation Tasks:** WikiText-103 (perplexity), ARC (accuracy), HumanEval (pass@1), GSM8K (accuracy).
- **Decoding:** Greedy and nucleus sampling (top-p=0.9, T={0.7, 1.0}), with settings matched between calibration and evaluation.

### 3.2 Baselines
- **Full-depth:** Standard `L`-layer inference.
- **Fixed-depth:** The best-performing fixed truncation to `m < L` layers.
- **Global Threshold:** An early-exit baseline using a single, globally tuned entropy threshold.
- **Speculative Decoding:** To test for orthogonality and composition of speedups.

### 3.3 Metrics
- **Accuracy:** Perplexity (relative change) and task-specific scores (absolute change).
- **Efficiency:** Theoretical FLOPs saved, wall-clock latency (tokens/sec), and profiled GPU FLOPs.
- **Overhead:** Time spent in proxy computation and decision logic as a fraction of total inference time.

### 3.4 Ablations
- **Proxy Design:** Low-rank `r` vs. other proxies; impact of proxy error on accuracy.
- **Scheduling:** Monotonic-only vs. schedules with recovery floors.
- **Calibration:** Sensitivity to calibration data size and domain shift.
- **Targeting:** Quantile `q` trade-off curve (accuracy vs. FLOPs).

### 3.5 Falsification Criteria
We consider EMAD to be ineffective if, for any model:
1.  **Accuracy Cost:** A ≥15% FLOP reduction results in an accuracy drop of >1% (relative for PPL, absolute for tasks).
2.  **Wall-Clock Irrelevance:** At matched accuracy, the wall-clock speedup is <10% using depth-stratified batching.

## 4. Relation to Prior Work
EMAD is a training-free, adaptive-depth method for autoregressive models. Unlike early-exit for classification, it requires no auxiliary heads or architectural changes. Unlike speculative decoding, it reduces per-token computational depth rather than sequential steps, making the two approaches complementary. By enforcing KV-cache consistency via a simple monotonic schedule, it avoids the complexities of methods that require modifying the attention mechanism or training process to handle dynamic depth.

## 5. Conclusion
EMAD offers a practical, training-free approach to reduce LLM inference cost by dynamically adapting computation depth. Its core principles—matching a calibrated entropy profile and enforcing a monotonic depth schedule—provide a robust and provably correct way to manage dynamic execution while preserving the integrity of the KV cache. Our falsification-oriented experimental plan provides a clear framework for validating its practical utility.

## 6. Pseudocode

```python
# State for each sequence in batch: D_prev (previous depth)
# Model: blocks, Norm, W_unembed, proxies, targets

def emad_token_step(h_in, D_prev, state):
    """Computes one token forward pass with EMAD."""
    h = h_in
    D_t = D_prev  # Default to previous depth

    for l in range(1, D_prev + 1):
        h = model.blocks[l](h, kv_cache_for_layer=l)
        
        if l in PROBE_LAYERS:
            # Low-overhead uncertainty check
            h_norm = model.Norm(h)
            entropy_estimate = calibrated_proxy(h_norm, layer=l)
            
            if entropy_estimate <= TARGET_ENTROPY[l]:
                D_t = l
                break  # Exit early
    
    # Final projection from exit layer's hidden state
    final_h = model.Norm(h) # h is from layer D_t
    logits = final_h @ W_unembed.T
    
    # Update KV caches only up to depth D_t
    commit_kv_cache(state, h_in, ..., depth=D_t)
    
    return logits, D_t # D_t becomes D_prev for next step
```
