Major Revisions

Brief critique:
- Technical gaps: The draft does not specify how the LM head is applied at intermediate layers (e.g., final RMSNorm use), making entropy estimates ill-defined. Entropy estimation overhead is underexplained; top-K via MIPS on GPU and low-rank readout need concrete, low-overhead designs and calibration details.
- Scheduling/system concerns: The monotone non-increasing rule ensures KV consistency but risks premature depth collapse without recovery; practical, bounded-recovery or floor mechanisms are not provided. There is no GPU-aware batching/scheduling plan to avoid throughput loss from intra-batch depth divergence, which can erase wall-clock gains.
- Decision rule/calibration: Using mean teacher-forcing entropy as a target is brittle; a quantile-based target and proxy-to-entropy calibration are needed. Interactions with decoding temperature/top-p and domain shift should be addressed.
- Validation plan: Baselines omit recent LLM early-exit/skip-depth methods and lack beam search considerations. Falsification thresholds are stated but no pseudocode, complexity analysis, or overhead accounting is provided.
- Presentation: Add a short formal proof of KV consistency, pseudocode, and a clearer relation to prior lines of work.

Revised Draft

# Entropy-Matched Adaptive-Depth Decoding

## Abstract
We propose Entropy-Matched Adaptive-Depth (EMAD) decoding, a training-free method to reduce per-token inference cost in large language models (LLMs). EMAD dynamically truncates the forward pass by exiting at an intermediate layer when a calibrated uncertainty criterion is met. The criterion matches a pre-computed teacher-forcing entropy profile, ensuring token-wise exit decisions are model- and scale-aware. To maintain KV-cache consistency in autoregressive generation, EMAD enforces a monotone non-increasing depth schedule across tokens, optionally augmented with bounded recovery mechanisms. EMAD requires no retraining or auxiliary heads and uses low-overhead entropy proxies calibrated offline. We provide a GPU-aware implementation plan and a falsification-oriented evaluation on small open models. EMAD is considered ineffective if it fails to yield ≥10% wall-clock speedup at matched accuracy or if ≥15% FLOP reduction incurs >1% accuracy loss.

## 1. Introduction
Inference cost for decoder-only Transformers scales with both context length and depth. Many tokens are easy and do not require full depth, motivating dynamic-depth inference. However, per-token depth variation naively breaks KV caching because self-attention at layer ℓ requires cached keys/values for all previous tokens at layer ℓ.

Existing accelerations either reduce steps (e.g., draft-and-verify decoding) or operation cost (quantization), but typically keep full depth per token. Early-exit methods for classification rely on auxiliary heads or retraining and do not directly handle KV consistency in generation.

EMAD addresses both challenges:
- A training-free exit rule: match inference-time uncertainty to a pre-calibrated, teacher-forcing entropy profile over depth.
- A KV-consistent schedule: constrain per-token depth to be monotone non-increasing, optionally with bounded recovery to mitigate premature collapse.

We contribute:
- A principled uncertainty target derived from a single offline teacher-forcing pass.
- Low-overhead entropy proxies with calibration to avoid full softmax at probe layers.
- A KV-consistent decoding schedule with theoretical correctness and GPU-aware batching.
- A rigorous falsification plan on small open models.

## 2. Method

### 2.1 Setting and notation
- Decoder-only Transformer with L blocks. Let hℓ,t be the hidden state after block ℓ for the current token t.
- The LM head uses the model’s final output normalization Norm(·) and unembedding matrix W and bias b, producing logits zℓ,t = Wᵀ Norm(hℓ,t) + b.
- Predictive entropy at layer ℓ is Hℓ,t = H(softmax(zℓ,t/Temp)), with decoding temperature Temp (and nucleus/top-k if applicable).

We compute per-token depth Dt ≤ L, determining the last executed block for token t; blocks > Dt are skipped.

### 2.2 KV-consistent scheduling
Constraint: Dt ≤ Dt−1 for all t, with D0 = L. This guarantees that, for any layer k ≤ Dt, cached keys/values exist for all prior tokens (non-increasing depths ensure D1 ≥ D2 ≥ … ≥ Dt). Hence, attention at layer k at step t is well-defined without backfilling.

Optional bounded recovery:
- Periodic floor: maintain a slowly decreasing floor Ft with Ft+1 ≥ Ft − δ, and set Dt ≥ Ft. Choose δ and initial F0 = L to avoid rapid collapse.
- Windowed backfill (optional): allow Dt > Dt−1 only if we backfill KV for layers in (Dt−1, Dt] over a limited sliding window of W previous tokens; this bounds extra cost by O(W · (Dt−Dt−1)) and is disabled by default.

We use the monotone rule by default and evaluate bounded recovery as an ablation.

### 2.3 Teacher-forcing entropy profile
Calibration pass:
- Data: a disjoint calibration set C (e.g., 5–100M tokens).
- For each token position and a sparse set of probe layers P ⊆ {1,…,L}, compute zℓ,t and Hℓ,t under teacher forcing (ground-truth history). Always apply the same final normalization layer used before the LM head.
- Target profile: for each ℓ ∈ P, define τℓ as a quantile qℓ of {Hℓ,t : (x,t) ∈ C}. We choose qℓ by mapping a user budget (desired FLOP reduction) to quantiles via a held-out calibration sweep. We enforce τℓ to be non-increasing in ℓ via isotonic regression.
- Optional conditioning: stratify τℓ by simple contexts (e.g., position bucket, previous-token confidence gap, newline indicator) to tighten match without model changes.

Rationale: Quantile targets expose a budget-accuracy knob; mean targets are brittle.

### 2.4 Low-overhead entropy proxies
Computing full-vocabulary logits at multiple probe layers is expensive. We use proxies ẑ that correlate with entropy and calibrate them to H via monotone regression on the calibration set:

- Logit-gap proxy: gℓ,t = z(1) − z(2) at a small shortlist S. Approximate z(1), z(2) via:
  - Shortlist reuse: union of previous token’s top-k and a small frequent-token list; or
  - Low-rank readout: factorize W ≈ U Vᵀ with rank r ≪ d. Compute ẑ = (Norm(hℓ,t)ᵀ U) Vᵀ. This reduces MatMul cost from O(d·|Vocab|) to O(d·r + r·|Vocab|). Choose r (e.g., 64–256) to keep overhead <5% of a block.
- Calibrated entropy: Ĥℓ,t = f(gℓ,t, sℓ,t), where sℓ,t optionally includes top-1 prob over S or low-rank tail-mass estimate. Fit f as an isotonic or spline regressor to predict H from proxies on calibration data.
- Validation: report proxy error (MAE of Ĥ vs H) and its impact on exit decisions.

We avoid GPU-host transfers and implement U,V in GPU memory with fused kernels.

### 2.5 Exit rule
For token t and probe layers ℓ in increasing order:
1) Compute proxies and Ĥℓ,t.
2) If (a) Ĥℓ,t ≤ τℓ − m and (b) ℓ ≤ Dt−1, set Dt = ℓ and stop. Here m ≥ 0 is a safety margin tuned on a validation set.
If no probe satisfies (a), set Dt = min(Dt−1, L).

Token is generated from logits at layer Dt. We apply the same decoding settings (temperature, top-p) as used to calibrate τℓ.

### 2.6 Systems considerations
- Probe placement: Place probes sparsely (e.g., every 2–4 blocks) to limit checks. Prefer earlier blocks where exits are likely.
- Fused kernels: Compute Norm(·), low-rank projection, and proxy stats in a fused op to reduce memory bandwidth.
- Batch packing by depth: Within a batch at step t, group sequences by their permitted max depth Dt−1 and execute blocks in a segmented fashion (shallower groups peel off earlier). This mitigates warp divergence and preserves utilization.
- Composition: EMAD composes with draft-and-verify decoding by applying depth adaptation independently to the target model. With beam search, apply EMAD per beam (the monotone constraint holds per sequence).

### 2.7 Correctness (KV consistency)
Claim: Under Dt ≤ Dt−1, self-attention at any layer k ≤ Dt has valid KVs for all prior tokens.
Sketch: By monotonicity, for all i < t, Di ≥ Dt. Hence for any k ≤ Dt, we have k ≤ Di, so KVs at layer k for token i were computed and cached. Thus the current attention at layer k can attend to all prior tokens at that layer without backfilling.

## 3. Experiments (Falsification Plan)

### 3.1 Models and data
- Models: Pythia-410M, Pythia-1.4B, OPT-1.3B, and LLaMA-1 7B (FP16/bfloat16, no retraining).
- Calibration: 5–20M tokens from The Pile for initial sweeps; scale to 100M for sensitivity analysis.
- Evaluation:
  - Language modeling: WikiText-103 perplexity.
  - Commonsense: ARC (acc).
  - Code: HumanEval pass@1.
  - Math: GSM8K (acc).
  - Decoding configs: greedy and nucleus (top-p 0.9) at temperatures {0.7, 1.0}, matched between calibration and evaluation.

### 3.2 Baselines
- Full-depth (L layers).
- Fixed-depth truncation (best m < L).
- Global entropy threshold: exit at first probe with Ĥℓ,t ≤ h0 (tuned), without profile matching.
- Token-skipping (confidence-based) where applicable (e.g., skip full step if early confidence is high).
- Acceleration orthogonals (speculative decoding) to demonstrate complementarity.

### 3.3 Metrics and accounting
- Accuracy: perplexity and task-specific metrics.
- Compute: theoretical FLOPs and profiled GPU FLOPs (Nsight).
- Latency: tokens/sec on A100, with and without batch packing.
- Overhead breakdown: fraction of time spent in probes (low-rank ops, proxies, control flow).

### 3.4 Ablations
- Probe frequency and placement.
- Entropy proxy variants and ranks; proxy calibration function f.
- Quantile targets vs mean; safety margin m; optional conditioning features.
- Monotone-only vs bounded recovery (window size W, periodic floor).
- Sensitivity to calibration size and domain shift (evaluate on out-of-domain sets).

### 3.5 Falsification criteria
For any model/configuration:
- Accuracy degradation: At ≥15% FLOP reduction, if perplexity worsens by >1% (relative) or any other benchmark drops by >1% (absolute).
- Practical irrelevance: At matched accuracy, if wall-clock speedup <10%.
Failure on either criterion falsifies EMAD’s usefulness.

## 4. Implementation details
- Intermediate readout uses the model’s final normalization parameters (e.g., RMSNorm) before the LM head to keep logits comparable across layers.
- Low-rank matrices (U,V) are fit via randomized SVD on W; ranks chosen to keep probe overhead <5% of a block’s FLOPs.
- Proxy-to-entropy calibration uses isotonic regression per probe layer (or shared across layers with layer index as a feature).
- Batching: We implement segmented execution with depth buckets and test micro-batch sizes to maintain GPU occupancy.
- Code: PyTorch implementation with Triton/CUDA fused kernels; scripts to reproduce calibration and evaluation on listed models.

## 5. Relation to prior work
- Early-exit in classification often requires auxiliary heads and retraining; EMAD is training-free and tailored to autoregressive KV constraints.
- Confidence-adaptive language modeling skips token steps under confidence; EMAD reduces per-token depth while preserving KV cache validity.
- Draft-and-verify decoding reduces sequential steps; EMAD is orthogonal and composes to reduce per-step depth.
- Layer-skip approaches that modify training are out of scope for training-free settings; we include fixed-depth truncation as a strong baseline.

## 6. Discussion and limitations
- Monotone schedules can over-collapse; we quantify this and evaluate bounded recovery and floor strategies. We avoid unbounded backfilling to keep latency predictable.
- Teacher-forcing profiles may misalign under distribution shift or different decoding temperatures; we condition profiles on decoding configs and evaluate robustness.
- Entropy proxies are approximate; we report proxy error and its effect on exit rates and accuracy, and cap aggressiveness via m.
- Throughput depends on batching strategy; we report both naive and depth-bucketed results.

## 7. Conclusion
EMAD offers a simple, training-free mechanism for adaptive-depth decoding in LLMs by matching inference-time uncertainty to a calibrated teacher-forcing profile under a KV-consistent schedule. With low-overhead proxies and GPU-aware batching, EMAD targets real wall-clock gains without retraining. Our falsification-oriented plan on small open models provides a clear path to validation and to identifying regimes where EMAD is ineffective.

## 8. Pseudocode (inference-time)

- Inputs: model blocks {B1..BL}, Norm, W,b; probe set P; targets {τℓ}; safety margin m; previous depth Dt−1.
- For each token t:
  1) h0,t ← embedding + pos enc
  2) Dt ← L
  3) For ℓ in 1..L:
     - hℓ,t ← Bℓ(hℓ−1,t; KV caches up to layer ℓ)
     - If ℓ ∈ P and ℓ ≤ Dt−1:
       - u ← Norm(hℓ,t)
       - Proxy features s ← low-rank(U,V,u) and shortlist stats
       - Ĥℓ,t ← f(s)
       - If Ĥℓ,t ≤ τℓ − m: Dt ← ℓ; break
  4) z ← Wᵀ Norm(hDt,t) + b; sample ŷt
  5) Commit KVs for layers 1..Dt; skip higher layers
  6) Set Dt as current depth for next step; proceed to t+1

We release code, calibration artifacts, and scripts for all listed models to enable independent validation.
