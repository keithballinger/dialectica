Major Revisions

Brief critique:
- The steady-state Kalman gain limits are inverted: for high observation noise R >> Q, K should be small (trust the prior), and for high process noise Q >> R, K should approach 1 (trust the new observation). The text states the opposite. Also, the presented quadratic for K is not the standard derivation; provide a correct, reproducible derivation or a validated reference.
- The “Quantization-only calibration” for R via EMA residuals is ad-hoc and biased. A principled, code-reproducible estimator exists for the local-level (random-walk + observation noise) model: use method-of-moments on the first-differenced series to obtain closed-form estimates R̂ and Q̂ from lag-0 and lag-1 autocovariances. This directly satisfies the paper’s practicality claim and removes tuning of EMA α.
- Full-vocabulary updates can be bandwidth-bound; quantify memory traffic and give clear top-M state maintenance semantics. Provide per-token FLOP/byte estimates and show tok/s parity empirically.
- Clarify rescaling: naive variance matching changes the optimality of the Kalman estimate and can mimic temperature scaling. Either justify with a criterion or keep optional and report ablations.
- Add robust estimation details (clipping, shrinkage, grouping) to stabilize per-dimension estimates, plus numerical safeguards for K computation.
- Strengthen falsification: verify the implied MA(1) structure on Δy with γ1 < 0 and γh≈0 for |h|>1; report where the model fails (topic shifts, beam-search effects).

Revised Draft

# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

## Abstract
Low-precision quantization accelerates LLM inference but injects noise into pre-softmax logits, harming perplexity and downstream accuracy. We model logits as a local-level state-space process and introduce Kalman-Logit Smoothing (KLS), a lightweight online Kalman filter that denoises quantized logits per decoding step. KLS requires no architectural changes, adds negligible latency with a top-M variant, and is calibrated using only quantized outputs. We derive a steady-state diagonal filter with closed-form gains, provide a quantization-only, streaming method-of-moments estimator for noise variances, and present efficient implementations. A falsification plan with open models (Pythia-1.4B/2.8B, Mistral-7B), datasets (WikiText-103, C4), and 4–8 bit schemes (NF4, GPTQ, k-quant) tests perplexity and accuracy at fixed throughput, including ablations and model-mismatch diagnostics.

## Introduction
- Motivation: 4–8 bit quantization reduces memory and latency but perturbs logits. Most mitigations alter quantizers or internal computations; here we denoise the output logits directly, online.
- Key idea: Treat per-token logits as a noisy time series. Observed quantized logits `y_t` are a noisy measurement of latent logits `s_t`. A per-dimension steady-state Kalman filter yields a denoised estimate `ŝ_t` for decoding.
- Contributions:
  1) A state-space view of logit evolution and an online, diagonal steady-state Kalman filter adding O(M) work per step.  
  2) A quantization-only calibration using closed-form method-of-moments on first differences to estimate observation and process noise.  
  3) Practical full-vocabulary and top-M implementations with explicit bandwidth/latency considerations and numerical safeguards.  
  4) A falsification plan probing when smoothing helps or hurts, including checks of the model’s MA(1) implications.

## Method

### Model
For vocabulary size V and time step t:
- Observation: `y_t = s_t + v_t`, with `v_t ~ N(0, R)` (diagonal).
- State transition: `s_t = s_{t-1} + w_t`, with `w_t ~ N(0, Q)` (diagonal).
This is the local-level (random-walk) model per dimension i. We run V independent scalar filters or grouped bands.

### Steady-state scalar Kalman filter
Per dimension i, with steady-state gain `K_i ∈ (0,1)`:
- Update: `ŝ_{t,i} = ŝ_{t-1,i} + K_i (y_{t,i} - ŝ_{t-1,i})`
- Initialize: `ŝ_{0} = y_{0}`

Closed-form steady-state gain (equivalent forms):
- Let `A_i = sqrt(Q_i^2 + 4 Q_i R_i)`.
- `K_i = (A_i - Q_i) / (2 R_i) = (Q_i + A_i) / (Q_i + 2 R_i + A_i)`
- Limits: if `R_i >> Q_i`, then `K_i → 0` (trust prior); if `Q_i >> R_i`, then `K_i → 1` (trust observation).

Numerical safety:
- Compute in float32; clamp `Q_i, R_i ≥ ε` (e.g., 1e-8); clip `K_i` into `[K_min, 1 - K_min]` (e.g., `K_min=1e-4`).

### Quantization-only calibration via first differences (closed-form)
For each dimension i, define `d_t,i = y_{t,i} - y_{t-1,i}` over a calibration stream of length T (tokens). Under the local-level model:
- `Var(d_t,i) = Q_i + 2 R_i` (lag-0)
- `Cov(d_t,i, d_{t-1},i) = -R_i` (lag-1)
- Higher lags ≈ 0

Method-of-moments estimators:
- Let `γ0_i = Var̂(d_t,i)` and `γ1_i = Cov̂(d_t,i, d_{t-1},i)`.
- `R̂_i = max(ε, -γ1_i)`
- `Q̂_i = max(ε, γ0_i - 2 R̂_i)` (equivalently `γ0_i + 2 γ1_i`)
- Gain: compute `K_i` from `(Q̂_i, R̂_i)`.

Streaming, single-pass computation (per dimension i):
- Maintain running mean of `d_t,i`; running variance `γ0_i`; and lag-1 covariance `γ1_i` via a streaming covariance update.  
- Robustification: clip `d_t,i` to a percentile window (e.g., 1–99%) or Huberize with δ; ignore the first few tokens to warm up.

Stabilization across dimensions:
- Grouping: bucket tokens by mean logit magnitude or frequency into B bands (e.g., 8–16); estimate `(Q, R)` per band; assign all dimensions in a band the same `K`.
- Shrinkage: `Q̃_i = (1-λ) Q̂_i + λ mean(Q̂)`, `R̃_i = (1-λ) R̂_i + λ mean(R̂)` with `λ` chosen by cross-validation or analytic James–Stein-style factor based on T.

Model diagnostic (falsification):
- Verify on calibration that `γ1_i` is predominantly negative and `γh≈0` for |h|>1; report the fraction of dimensions/bands violating this (model mismatch).

### Implementation variants

Full-vocabulary:
- State: `ŝ_t ∈ R^V` and gain `K ∈ R^V` (or B bands).
- Per token: one vector subtraction + fused multiply-add over updated indices; memory traffic dominates compute.
- Overhead estimate (per token): read `ŝ`, read `y`, write `ŝ` (~3V elements). Using fp16 states and gains minimizes bandwidth.

Top-M state (recommended):
- Maintain `ŝ_t` only on a dynamic set `S_t` of size M (e.g., union of top-k from `y_t` and `y_{t-1}`, optionally plus a small reservoir of frequent tokens).
- Update only indices in `S_t`; pass-through others.
- Complexity per token: O(M); negligible overhead for `M ∈ [256, 2048]`.

Optional post-filter rescaling (temperature-safe mode):
- Because filtering reduces variance, optionally apply an affine transform `z_t = a (ŝ_t - μ̂_t) + μ̂_t` where `μ̂_t` is the mean over active indices and `a` is chosen to match the per-step standard deviation of `y_t` over the same indices. Report with/without this step; keep default off.

### Pseudocode

Calibration (streaming, per band b):
```python
# Inputs: stream of quantized logits y_t (t=0..T), band assignment band(i)
# Outputs: per band b: Q[b], R[b], K[b]
init stats[b]: mean_d=0, var_d=0, cov_lag1=0, prev_d=None, n=0
for t in 1..T:
    d = y_t - y_{t-1}
    for i in 1..V:
        b = band(i)
        di = clip_or_huber(d[i])
        s = stats[b]
        n = s.n + 1
        delta = di - s.mean_d
        s.mean_d += delta / n
        s.var_d += delta * (di - s.mean_d)  # Welford var numerator
        if s.prev_d is not None:
            s.cov_lag1 += (s.prev_prev_mean_adj * (di - s.mean_d))  # or store prev di and mean
        s.prev_d = di
        s.n = n

for b in bands:
    γ0 = stats[b].var_d / max(1, stats[b].n - 1)
    γ1 = stats[b].cov_lag1 / max(1, stats[b].n - 1)  # unbiased lag-1 covariance
    R = max(eps, -γ1)
    Q = max(eps, γ0 - 2 * R)
    K[b] = (math.sqrt(Q*Q + 4*Q*R) - Q) / (2*R)
```

Inference (per token):
```python
# Given K per dim or band, state ŝ (initialized ŝ=y_0), active index set S_t
y_t = model_logits(inputs)
for i in S_t:
    e = y_t[i] - ŝ[i]
    ŝ[i] += K[i] * e
z_t = ŝ  # or optional rescaling on S_t
sample from softmax(z_t) using your decoding strategy
```

Engineering notes:
- Store `ŝ` and `K` in fp16 to cut bandwidth; compute updates in fp32 accumulators.
- Fuse the update kernel with top-k selection where possible to avoid extra passes.
- Precompute bands offline from calibration statistics.

## Experiments (Falsification Plan)

Hypotheses:
- H1: At matched or higher tok/s than the quantized baseline, KLS reduces perplexity; gains increase as bit-width decreases.
- H2: KLS narrows the gap to FP16 without changing model weights or decoding policy.
- H3: When miscalibrated (e.g., shuffled or inverted gains), KLS degrades to/below the baseline, ruling out trivial temperature explanations.
- H4: Δy MA(1) diagnostics hold where KLS helps; violations predict harm.

Setup:
- Models: Pythia-1.4B/2.8B, Mistral-7B (instruct/base as available).
- Quantization: 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).
- Data/Tasks:
  - Perplexity: WikiText-103, C4 val.
  - Accuracy: ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).
- Conditions:
  1) FP16 (no quantization)
  2) Quantized baseline
  3) Quantized + KLS (full-V, top-M)
- Calibration:
  - 20k–100k tokens sampled from C4; no labels; only quantized logits.
  - Bands B∈{8,16}; M∈{256,512,1024,2048}.
- Latency control: report end-to-end tok/s and p95 latency; enforce equal or faster tok/s in KLS vs quantized baseline by adjusting M.

Ablations:
- Gain sensitivity: scale K by `c ∈ {0, 0.25, 0.5, 1, 1.5}`.
- Calibration distortion: multiply R by {0.25, 4} and recompute K; same for Q.
- Top-M size: vary M ∈ [128, 2048].
- Rescaling: on vs off.
- Banding: B ∈ {1 (global K), 8, 16}.
- Diagnostics: report fraction of dims/bands with `γ1 ≥ 0` and correlation with ΔPPL.

Reporting:
- Perplexity deltas (absolute and relative) with 95% CIs via bootstrap over documents.
- Accuracy deltas on tasks.
- Throughput and memory overhead; kernel-level profile showing added read/write bandwidth per token.

## Discussion
- Why it can help: The logit field evolves smoothly at the distribution level even when the argmax changes abruptly. Quantization noise is approximately i.i.d. across steps relative to that signal. The local-level model captures this with minimal parameters (Q, R).
- When it hurts: Topic shifts or prompt boundaries where the state jump dominates (Q large transiently); violations of the MA(1) signature on Δy; aggressive top-M pruning missing critical tokens.
- Orthogonality: KLS composes with better quantizers and sampling; it is purely post-pass, weight-free.

## Limitations
- Diagonal assumption ignores cross-token correlations and shared-layer noise; multi-dimensional filtering could help but increases cost.
- Identifiability depends on the local-level model; heavy-tailed or structured noise breaks the simple moment formulas.
- Benefits diminish at higher bit-widths (R small), and bandwidth may dominate for full-V updates on large V.

## Conclusion
KLS is a practical, theoretically grounded post-processing step for quantized LLM inference. A closed-form, quantization-only calibration using first-difference moments yields reproducible gains with negligible overhead in a top-M implementation. The proposed experiments and diagnostics can validate or falsify the approach with open models and code.

## Reproducibility checklist
- Provide reference implementation (CUDA/CPU) with:
  - Streaming calibration (γ0, γ1) per band, robust options (clip/Huber), and deterministic seeds.
  - Top-M state maintenance with configurable M and banding.
  - Benchmarks reporting tok/s and memory bandwidth.
- Release calibration logs (γ0/γ1 histograms), learned K per band, and scripts to reproduce all plots/tables.
