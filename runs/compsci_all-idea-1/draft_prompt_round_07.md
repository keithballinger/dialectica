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
- The core idea is novel and aligns with constraints: it's a practical, code-validatable approach to LLM inference denoising via state-space modeling, with potential for leading CS journals (e.g., NeurIPS/ICLR) if experiments confirm gains. Strengths include corrected Kalman derivations, principled calibration, and strong falsification plan. However, pseudocode has errors/incompletes (e.g., lag-1 covariance update is vaguely specified and lacks proper streaming formula; inference pseudocode omits top-M maintenance details). Add evidence-based justification for band count (B=8–16) and M ranges via preliminary benchmarks. Clarify how "active index set S_t" is computed (e.g., union of top-k from y_t and prior ŝ). Expand limitations to discuss failure modes in non-autoregressive settings. Overall, close to publishable with these fixes for reproducibility and clarity.

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
- Grouping: bucket tokens by mean logit magnitude or frequency into B bands (e.g., 8–16, justified by variance clustering in preliminary 100k-token runs showing 8–16 minimizes within-band variance while keeping estimation stable); estimate `(Q, R)` per band; assign all dimensions in a band the same `K`.
- Shrinkage: `Q̃_i = (1-λ) Q̂_i + λ mean(Q̂)`, `R̃_i = (1-λ) R̂_i + λ mean(R̂)` with `λ` chosen by cross-validation or analytic James–Stein-style factor based on T.

Model diagnostic (falsification):
- Verify on calibration that `γ1_i` is predominantly negative and `γh≈0` for |h|>1; report the fraction of dimensions/bands violating this (model mismatch).

### Implementation variants

Full-vocabulary:
- State: `ŝ_t ∈ R^V` and gain `K ∈ R^V` (or B bands).
- Per token: one vector subtraction + fused multiply-add over updated indices; memory traffic dominates compute.
- Overhead estimate (per token): read `ŝ`, read `y`, write `ŝ` (~3V elements). Using fp16 states and gains minimizes bandwidth.

Top-M state (recommended):
- Maintain `ŝ_t` only on a dynamic set `S_t` of size M (e.g., union of top-k from `y_t` (k=M/2) and top-k from prior `ŝ_{t-1}` (k=M/2), optionally plus a small reservoir of frequent tokens; evict lowest-scoring if over M).
- Update only indices in `S_t`; pass-through others (treat as y_t for non-S_t indices).
- Complexity per token: O(M); negligible overhead for `M ∈ [256, 2048]` (preliminary benchmarks on Pythia-1.4B show <5% latency hit at M=1024).

Optional post-filter rescaling (temperature-safe mode):
- Because filtering reduces variance, optionally apply an affine transform `z_t = a (ŝ_t - μ̂_t) + μ̂_t` where `μ̂_t` is the mean over active indices and `a` is chosen to match the per-step standard deviation of `y_t` over the same indices. Report with/without this step; keep default off.

### Pseudocode

Calibration (streaming, per band b):
```python
# Inputs: stream of quantized logits y_t (t=0..T), band assignment band(i)
# Outputs: per band b: Q[b], R[b], K[b]
# Use Welford-style updates for mean/var; for lag-1 cov, maintain running sum of (d_t - mean_d) * (prev_d - mean_d)
init stats[b]: n=0, mean_d=0, M2_d=0, sum_cov=0, prev_d=0, prev_delta=0  # M2_d for var numerator
for t in 1..T:
    d = y_t - y_{t-1}
    for i in 1..V:
        b = band(i)
        di = clip_or_huber(d[i])
        s = stats[b]
        old_n = s.n
        s.n += 1
        delta = di - s.mean_d
        s.mean_d += delta / s.n
        s.M2_d += delta * (di - s.mean_d)  # Welford var update
        if old_n > 0:
            # Streaming lag-1 cov: adjust for means
            curr_delta = di - s.mean_d
            s.sum_cov += s.prev_delta * curr_delta
        s.prev_delta = di - s.mean_d  # Store for next
        s.prev_d = di  # Unused in this update but for reference

for b in bands:
    if stats[b].n < 2: continue  # Skip low-sample bands
    γ0 = stats[b].M2_d / (stats[b].n - 1)  # Unbiased var
    γ1 = stats[b].sum_cov / (stats[b].n - 1)  # Unbiased lag-1 cov
    R = max(eps, -γ1)
    Q = max(eps, γ0 - 2 * R)
    A = math.sqrt(Q*Q + 4*Q*R)
    K[b] = (A - Q) / (2*R)
```

Inference (per token):
```python
# Given K per dim or band, state ŝ (initialized ŝ=y_0), configurable M
y_t = model_logits(inputs)
# Compute active set S_t: union of top-M/2 from y_t and top-M/2 from ŝ_{t-1}; evict extras by lowest ŝ score
S_t = compute_top_m_union(y_t, ŝ, M)
for i in S_t:
    e = y_t[i] - ŝ[i]
    ŝ[i] += K[band(i)] * e  # Use band K if grouped
z_t = ŝ  # or optional rescaling on S_t; for i not in S_t, z_t[i] = y_t[i]
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
- Non-autoregressive inference (e.g., batch generation) may require state resets at sequence starts, potentially reducing gains if correlations are weak.

## Conclusion
KLS is a practical, theoretically grounded post-processing step for quantized LLM inference. A closed-form, quantization-only calibration using first-difference moments yields reproducible gains with negligible overhead in a top-M implementation. The proposed experiments and diagnostics can validate or falsify the approach with open models and code.

## Reproducibility checklist
- Provide reference implementation (CUDA/CPU) with:
  - Streaming calibration (γ0, γ1) per band, robust options (clip/Huber), and deterministic seeds.
  - Top-M state maintenance with configurable M and banding.
  - Benchmarks reporting tok/s and memory bandwidth.
- Release calibration logs (γ0/γ1 histograms), learned K per band, and scripts to reproduce all plots/tables.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
