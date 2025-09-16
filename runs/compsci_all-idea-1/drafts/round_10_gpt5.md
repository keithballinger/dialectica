Minor Revisions

Brief critique:
- Soundness: The local-level (random-walk) model and steady-state scalar Kalman gain are correct; the closed-form K expressions are consistent with the Riccati solution. The method-of-moments identification via first-difference autocovariances is appropriate.
- Novelty/practicality: Positioning this as a post-hoc, weight-free online denoiser for quantized logits is novel within LLM inference and immediately testable with open models.
- Needed fixes/clarifications:
  1) Greedy argmax bug: When mixing smoothed logits on the active set with raw logits elsewhere, the argmax must compare max over ŝ on S versus max over y on V\S; comparing to max over y on all V is incorrect.
  2) Candidate set for sampling: If T_t is built solely from y_t, tokens whose ŝ would have entered top-k/top-p can be missed. Include a union with top-ranked ŝ over S (or a small safety spillover).
  3) Complexity: Top-k over y_t is O(V) (as in standard decoding); KLS adds O(M) beyond this. The current text overstates O(M) end-to-end.
  4) Calibration boundaries: Explicitly reset y_prev and clear per-i d_prev at sequence boundaries to avoid cross-sequence lag contamination.
  5) Numerical details: Recommend float64 for A = sqrt(Q^2 + 4QR) and stable algebra (e.g., use K = Q/X with X solving the quadratic) when Q ≪ R; specify epsilon behavior and clamping more precisely.
  6) Heteroskedastic R: Note R can depend on logit magnitude/quantizer; motivate banding by this and mention optional magnitude-conditioned bands.
  7) Baselines: Add a strong EMA baseline (tuned α) and report comparisons to ensure KLS gains aren’t just from generic smoothing.
  8) Documentation: Clarify warm-up rationale (filter transient), and that “KLS” uses a filter (not a fixed-interval smoother).
  9) Pseudocode: Add sequence reset in calibration; fix greedy and candidate-set logic; minor streamlining.

Revised Draft
# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

## Abstract
Low-precision quantization accelerates LLM inference but perturbs pre-softmax logits, harming perplexity and downstream accuracy. We model logit evolution as a local-level state-space process and introduce Kalman-Logit Smoothing (KLS), a lightweight online filter that denoises quantized logits at each decoding step. KLS requires no retraining, can be calibrated using only quantized outputs from a short token stream, and adds negligible overhead beyond standard top-k/top-p selection with a top-M active-set implementation. We derive a diagonal steady-state Kalman filter and provide a robust, streaming method-of-moments estimator for process and observation noise variances. Efficient, numerically stable algorithms are given for both full-vocabulary and top-M variants. A comprehensive falsification plan with open models (Pythia, Mistral), datasets (WikiText-103, C4), and 4–8 bit quantization schemes is provided, including strong EMA baselines and diagnostics for model mismatch.

## 1. Introduction
Low-bit quantization (e.g., 4–8 bits) reduces LLM inference cost but injects noise into pre-softmax logits. Most mitigation strategies alter quantization or involve fine-tuning. We propose an orthogonal, post-hoc approach: online denoising of the final logit vector before decoding.

We treat logits as a multivariate time series. For each token dimension, the true logit evolves smoothly across steps, while the observed quantized logit is a noisy measurement. This maps directly to a state-space model. Kalman-Logit Smoothing (KLS) applies an independent scalar Kalman filter per logit (or band of logits) to produce a denoised estimate on-the-fly.

Contributions:
1) A state-space formalization of logit evolution under quantization noise and a corresponding online, diagonal steady-state Kalman filter that integrates with decoding at O(M) overhead beyond standard top-k/top-p for the active-set variant.
2) A quantization-only calibration procedure using method-of-moments on logit first-differences, implemented as a robust, unbiased streaming estimator.
3) Practical algorithms with complexity analysis, numerically stable gain computation, and integration with greedy/top-k/nucleus/beam decoding (including a correct argmax and candidate-set construction when mixing smoothed and raw logits).
4) A falsification plan with rigorous baselines (including tuned EMA), ablations, and diagnostics that identify when KLS helps or hurts.

Note: While named “smoothing” for intuition, the online component is a steady-state Kalman filter (no future look-ahead).

## 2. Method

### 2.1 Local-Level State-Space Model
For vocabulary size V and time step t:
- State: s_{t,i} = s_{t−1,i} + w_{t,i}, with w_{t,i} ~ N(0, Q_i).
- Observation: y_{t,i} = s_{t,i} + v_{t,i}, with v_{t,i} ~ N(0, R_i).

We assume diagonal Q and R, enabling V independent scalar filters. To reduce parameters, we group dimensions into B ≪ V bands sharing (Q_b, R_b). Banding can reflect heteroskedastic observation noise (e.g., logit magnitude dependence under quantization).

### 2.2 Steady-State Scalar Kalman Filter
Let X_i be the steady-state a priori variance solving X_i^2 − Q_i X_i − Q_i R_i = 0 with positive root X_i = (Q_i + sqrt(Q_i^2 + 4 Q_i R_i))/2. The steady-state gain is:
- K_i = X_i / (X_i + R_i) = (Q_i + A_i) / (Q_i + A_i + 2 R_i) = (A_i − Q_i) / (2 R_i),
  where A_i = sqrt(Q_i^2 + 4 Q_i R_i).

Online update per dimension:
ŝ_{t,i} = ŝ_{t−1,i} + K_i (y_{t,i} − ŝ_{t−1,i}), with initialization ŝ_{0,i} = y_{0,i}.

Numerical stability:
- Compute A_i and X_i in float64; clamp Q_i, R_i ≥ ε (e.g., 1e-12).
- Stable alternative: compute K_i = Q_i / X_i, avoiding cancellation when Q_i ≪ R_i.
- Clamp K_i ∈ [K_min, 1 − K_min] (e.g., K_min = 1e-4).

Interpretation: This is a principled EMA with α = K_i, where α is learned from (Q_i, R_i).

### 2.3 Calibration via Method of Moments
Define differences d_{t,i} = y_{t,i} − y_{t−1,i}. Under the model:
- γ_{0,i} = Var(d_{t,i}) = Q_i + 2 R_i.
- γ_{1,i} = Cov(d_{t,i}, d_{t−1,i}) = −R_i.
- γ_{h,i} ≈ 0 for |h| > 1.

Given estimates γ̂_0 and γ̂_1 (pooled per band b):
- R̂_b = max(ε, −γ̂_1), Q̂_b = max(ε, γ̂_0 + 2 γ̂_1).

Estimation details:
- Streaming, unbiased estimators for variance and lag-1 covariance over a calibration set of T tokens.
- Pool across dimensions within a band; maintain per-i d_prev to form (d_{t,i}, d_{t−1,i}) pairs correctly.
- De-mean differences if empirical mean drift is non-zero; robustify with clipping/Huber to mitigate heavy tails.
- Optional shrinkage: Q̃_b = (1−λ) Q̂_b + λ mean(Q̂), R̃_b analogous, with λ ∈ [0.1, 0.5].
- Sequence boundaries: reset y_prev and clear all per-i d_prev maps between independent sequences to avoid cross-sequence lag contamination.

Band construction:
- B ∈ {8, 16}; form bands by quantiles of a per-token statistic (e.g., mean |logit| or token frequency) to capture heteroskedastic R.
- Ensure N_min ≈ 10k differences per band for stable moments; merge small bands.

Diagnostics:
- Report the fraction of bands with γ̂_1 ≥ 0 as a mismatch indicator; expect worse performance when this increases.

### 2.4 Implementation

Variants:
- Full vocabulary: update all V dimensions. Memory traffic ~3V floats per step. Store ŝ and K in fp16, compute updates in fp32/64.
- Top-M active set (recommended): update only a dynamic set S_t of M indices (e.g., M=1024).

Active-set algorithm:
1) Candidate selection:
   - I_y = top-k(y_t, k = M/2) over V (standard decoding already performs a top-k/top-p scan).
   - I_s = state.topk(k = M/2) over stored entries (≤ M).
2) Set construction: S_t = I_y ∪ I_s; optionally include a small reservoir of frequent tokens.
3) Eviction: If |S_t| > M, drop lowest priority p_i = max(y_{t,i}, ŝ_{t−1,i}).
4) Update: For i ∈ S_t, s_prev = state.get(i, default = y_{t,i}); s_new = s_prev + K_{band(i)} (y_{t,i} − s_prev); state.set(i, s_new).
5) Output mixture:
   - For greedy: compute m_in = max_i∈S ŝ_{t,i}, idx_in = argmax over S; compute m_out = max_i∈(V\S) y_{t,i}, idx_out = argmax over V\S; choose argmax between (m_in, m_out). This avoids comparing ŝ on S to y on S.
   - For top-k/top-p: build candidate set T_t = T_y ∪ T_s where T_y is from y_t (e.g., standard top-k/top-p), and T_s is top-L over ŝ on S (small L, e.g., 64–256) to avoid missing tokens promoted by smoothing. Scores for i ∈ T_t are ŝ_{t,i} if i ∈ S_t else y_{t,i}.

Complexity:
- As in standard decoding, scanning y_t for top-k/top-p is O(V). KLS adds O(M) work/memory traffic for active-set maintenance and updates. Choice of M controls overhead.

Decoding modes and state handling:
- Warm-up: skip updates for first W tokens (W≈2–4) to bypass filter transient from cold-start (P_0 mismatch). Alternatively, linearly ramp K to steady-state over W steps.
- Sequence resets: reset ŝ at sequence boundaries; in batching, maintain independent state per sequence and per beam during beam search.

### 2.5 Pseudocode

Calibration (Streaming, per band, with sequence resets):
```python
# Inputs:
#   streams: iterable of sequences; each sequence yields V-dim logits y_t (from quantized model)
#   band_map: array len V mapping token i -> band b in [0, B)
# Output:
#   K_b per band

import math
eps = 1e-12

stats = [
    dict(n=0, sum_d=0.0, sum_d2=0.0,
         sum_prod_lag1=0.0, sum_X_lag=0.0, sum_Y_lag=0.0, n_lag=0,
         d_prev={})  # per-i previous difference within this band
    for _ in range(B)
]

for seq in streams:
    it = iter(seq)
    try:
        y_prev = next(it)
    except StopIteration:
        continue
    # Clear per-i d_prev at sequence start
    for s in stats:
        s['d_prev'].clear()

    for y_curr in it:
        d = y_curr - y_prev  # vectorized
        for i, di in enumerate(d):
            b = band_map[i]
            s = stats[b]
            # Optionally robustify: di = huber(di, delta)
            s['n'] += 1
            s['sum_d'] += di
            s['sum_d2'] += di * di

            if i in s['d_prev']:
                dpi = s['d_prev'][i]
                s['sum_prod_lag1'] += di * dpi
                s['sum_X_lag'] += di
                s['sum_Y_lag'] += dpi
                s['n_lag'] += 1

            s['d_prev'][i] = di
        y_prev = y_curr

K = [0.0] * B
for b, s in enumerate(stats):
    n = s['n']
    if n < 2 or s['n_lag'] < 1:
        K[b] = 0.0
        continue
    mean_d = s['sum_d'] / n
    gamma0 = (s['sum_d2'] - n * mean_d * mean_d) / max(1, (n - 1))

    n_lag = s['n_lag']
    mean_X = s['sum_X_lag'] / n_lag
    mean_Y = s['sum_Y_lag'] / n_lag
    gamma1 = (s['sum_prod_lag1'] - n_lag * mean_X * mean_Y) / max(1, (n_lag - 1))

    R = max(eps, -gamma1)
    Q = max(eps, gamma0 + 2.0 * gamma1)

    # Stable gain computation
    # Solve X^2 - Q X - Q R = 0 for positive root
    A = math.sqrt(Q * Q + 4.0 * Q * R)  # use float64
    X = 0.5 * (Q + A)
    K[b] = max(1e-4, min(1.0 - 1e-4, Q / X))
```

Inference (Top-M, correct argmax and candidate union):
```python
# Inputs: y_t (V-dim logits), state (sparse map i -> ŝ_i), K_map (band -> K), M
# Outputs: sampling scores over a candidate set; updated state

# 1) Active set selection
I_y = topk_indices(y_t, k=M//2)          # O(V) as in standard decoding
I_s = state.topk_indices(k=M//2)          # ≤ M, O(M log M) or selection
S_t = union(I_y, I_s)
if len(S_t) > M:
    S_t = evict_by_priority(S_t, y_t, state, M)  # O(M)

# 2) State update on S_t
for i in S_t:
    s_prev = state.get(i, y_t[i])  # cold start -> y_t[i]
    k = K_map[band(i)]
    s_new = s_prev + k * (y_t[i] - s_prev)
    state[i] = s_new

# 3) Build candidate set for decoding
T_y = topk_or_top_p_indices(y_t)          # standard candidate construction
T_s = topk_indices_on_state(state, L=128) # from ŝ over S_t; small L
T_t = union(T_y, T_s)

# 4) Scores for sampling
scores = {i: (state[i] if i in state else y_t[i]) for i in T_t}

# 5) Greedy argmax (if needed):
#    max over ŝ on S_t vs max over y on V \ S_t
m_in, idx_in = max_over_indices(state, S_t)           # ŝ on S_t
m_out, idx_out = max_over_indices(y_t, complement(S_t))
greedy_idx = idx_in if m_in >= m_out else idx_out
```

## 3. Experiments (Falsification Plan)

Hypotheses:
- H1: At matched or higher throughput vs. quantized baseline, KLS reduces perplexity, with larger gains at lower bit widths.
- H2: KLS narrows the gap to FP16 baselines on perplexity and downstream tasks.
- H3: Gains are sensitive to correct calibration; distorted/shuffled gains degrade performance.
- H4: Improvement correlates with the mismatch diagnostic (fraction of bands with γ̂_1 < 0).

Models and quantization:
- Pythia-1.4B/2.8B, Mistral-7B (base/instruct).
- 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).

Data and tasks:
- Perplexity: WikiText-103, C4 validation.
- Tasks: ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).

Conditions:
1) FP16 baseline.
2) Quantized baseline.
3) Quantized + KLS (Top-M, M=1024).

Baselines and ablations:
- EMA baseline: Exponential moving average of logits with α grid-searched on validation (α ∈ {0.1,…,0.9}) per band/global; report quality vs. overhead relative to KLS.
- Top-M size: M ∈ {128, 512, 1024, 2048}.
- Calibration distortion: Multiply/divide R̂, Q̂ by factors {0.25, 0.5, 2, 4}; random shuffle K across bands.
- Banding: B ∈ {1 (global), 8, 16, V}.
- Decoding modes: greedy, top-k, nucleus; beam search with per-beam state.
- Diagnostics: fraction of bands with γ̂_1 ≥ 0 and its correlation with Δperplexity.

Calibration details:
- 50k–200k tokens from C4 (train); reset at sequence boundaries.
- Robustification: Huber with δ tuned on a held-out set; N_min ≈ 10k per band.
- Shrinkage λ ∈ {0.1, 0.3}.

Latency/throughput:
- Report tokens/sec and p95 latency. Emphasize that the O(V) top-k/top-p scan is standard; KLS adds O(M) work and memory traffic.

## 4. Discussion
KLS exploits temporal smoothness in logits partially obscured by quantization noise. The local-level model yields a principled EMA-like filter with gains tied to identifiable noise parameters, distinguishing it from ad-hoc smoothing. Expected failures occur under sharp distribution shifts (random-walk mismatch) or non-additive/temporally correlated quantization noise. Banding mitigates heteroskedastic R but cannot capture cross-token correlations—an avenue for heavier multivariate filters. Diagnostics (γ̂_1) warn about mismatch and predict reduced gains.

## 5. Limitations
- Diagonal filtering ignores cross-token correlations; multivariate extensions are costlier.
- Method-of-moments can be sensitive to heavy tails; robustification reduces but may not eliminate bias.
- Benefits diminish at higher bit-widths (small R).
- Autoregressive temporal assumptions may weaken under non-AR or aggressive batching.
- Fixed steady-state K ignores potential context-dependent R; online adaptation is future work.

## 6. Conclusion
KLS is a practical, theoretically grounded, and calibration-only method for improving quantized LLM decoding quality. It integrates seamlessly with existing decoding, incurs minimal overhead, and includes diagnostics for when it should be applied. The falsification plan and strong EMA baselines ensure rigorous evaluation with open models and code.

## Reproducibility Checklist
- Code (CUDA/PyTorch):
  - Streaming calibration with robustification, sequence-boundary resets, and banding.
  - Top-M active-set maintenance with correct argmax/candidate logic.
  - Integration with HuggingFace generate (greedy, top-k, top-p, beam).
  - Benchmarks of tokens/sec and p95 latency.
- Data:
  - Release learned (Q, R, K) per model/quantization pair; logs of per-band γ̂_0, γ̂_1, diagnostics.
  - Scripts to reproduce all results, including EMA baselines and ablations.
