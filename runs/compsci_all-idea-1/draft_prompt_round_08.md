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
- Correctness: The local-level model and method-of-moments identities for Δy are sound. However, the streaming lag-1 covariance pseudocode is biased; switch to a sum-of-lagged-products estimator with proper pairwise means. Provide a single canonical steady-state K formula and show an equivalent alternative to avoid confusion.
- Reproducibility: The top-M active-set maintenance is underspecified (union policy, eviction, re-entry initialization, optional reservoir). Add exact algorithms and complexity. Clarify how to compute z_t without materializing a dense vector when using top-k/nucleus sampling.
- Calibration: Specify banding construction and selection (B), minimum tokens per band, and robustification. Add concrete token budgets and shrinkage defaults. Provide an explicit warm-up and sequence boundary handling policy.
- Diagnostics and ablations: Good falsification plan; add beam search and batch decoding notes, and explicitly report the fraction of bands with γ1 ≥ 0 alongside effect sizes.
- Engineering: Add numerically stable implementations (safe sqrt, fp16 storage with fp32 math), memory/bandwidth footprint, and initialization on re-entry to S_t.

Revised Draft
# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

## Abstract
Low-precision quantization accelerates LLM inference but perturbs pre-softmax logits, harming perplexity and downstream accuracy. We model logits as a local-level state-space process and introduce Kalman-Logit Smoothing (KLS), a lightweight online filter that denoises quantized logits per decoding step. KLS requires no architectural changes, can be calibrated using only quantized outputs, and adds negligible latency with a top-M variant. We derive a steady-state diagonal filter, give a quantization-only, streaming method-of-moments estimator for process/observation noise, and present efficient implementations. A falsification plan with open models (Pythia-1.4B/2.8B, Mistral-7B), datasets (WikiText-103, C4), and 4–8 bit schemes (NF4, GPTQ, k-quant) tests perplexity and accuracy at fixed throughput, including ablations and model-mismatch diagnostics.

## 1. Introduction
- Motivation: 4–8 bit quantization reduces memory and latency but injects noise into logits. Most mitigations modify quantizers or internal layers; we instead denoise the output logits online, post hoc.
- Key idea: Treat per-token logits as a noisy time series. Observed quantized logits y_t are a noisy measurement of latent logits s_t. A per-dimension steady-state Kalman filter yields a denoised estimate ŝ_t for decoding.
- Contributions:
  - A state-space view of logit evolution and an online, diagonal steady-state Kalman filter adding O(M) work per step.
  - A quantization-only calibration using first-difference moments to estimate observation and process noise with streaming, robust updates.
  - Practical full-vocabulary and top-M implementations with explicit bandwidth/latency considerations and numerical safeguards.
  - A falsification plan probing when smoothing helps or hurts, with Δy MA(1) diagnostics.

## 2. Method

### 2.1 Local-level model
For vocabulary size V and step t:
- Observation: y_t = s_t + v_t, v_t ~ N(0, R) (diagonal).
- State: s_t = s_{t-1} + w_t, w_t ~ N(0, Q) (diagonal).

We run V independent scalar filters or grouped bands B ≪ V.

### 2.2 Steady-state scalar Kalman filter
For dimension i:
- Update: ŝ_{t,i} = ŝ_{t-1,i} + K_i (y_{t,i} − ŝ_{t-1,i}), with ŝ_0 = y_0.
- Steady-state gain from Q_i,R_i:
  - Let A_i = sqrt(Q_i^2 + 4 Q_i R_i).
  - Canonical form: K_i = (A_i + Q_i) / (A_i + Q_i + 2 R_i).
  - Equivalent form: K_i = (A_i − Q_i) / (2 R_i).
  - Limits: R_i ≫ Q_i ⇒ K_i ≈ √(Q_i/R_i) → 0; Q_i ≫ R_i ⇒ K_i → 1.

Numerics: compute A_i in float32 with Q_i,R_i ≥ ε (e.g., 1e-8); clip K_i ∈ [K_min, 1 − K_min] (K_min ~ 1e-4).

### 2.3 Quantization-only calibration via Δy moments
Define per-dimension first differences d_{t,i} = y_{t,i} − y_{t-1,i}. Under the model:
- Var(d_{t,i}) = Q_i + 2 R_i ≡ γ0_i.
- Cov(d_{t,i}, d_{t-1,i}) = −R_i ≡ γ1_i.
- Higher lags ≈ 0.

Method-of-moments (per i or per band b):
- R̂ = max(ε, −γ̂1), Q̂ = max(ε, γ̂0 − 2 R̂) = max(ε, γ̂0 + 2 γ̂1).
- Gain K from (Q̂,R̂) as above.

Robust streaming estimation (per band b):
- Maintain, over a calibration stream of length T tokens:
  - n = number of differences (T − 1)
  - sum_d = Σ_t d_t, sum_d2 = Σ_t d_t^2
  - sum_dlag1 = Σ_{t≥2} d_t d_{t-1}
  - d_first = d_1, d_prev = last d_t
- Optional robustification: clip or Huberize d_t per band (e.g., Huber δ = 3×IQR).
- Unbiased estimates at end (per band):
  - mean_d = sum_d / n
  - var(d) (γ̂0) = (sum_d2 − n·mean_d^2) / (n − 1)
  - Pairwise means for lag-1:
    - mean_X = (sum_d − d_first)/(n − 1)  # d_2..d_n
    - mean_Y = (sum_d − d_prev)/(n − 1)   # d_1..d_{n−1}
    - Ê[d_t d_{t−1}] = sum_dlag1 / (n − 1)
    - γ̂1 = Ê[d_t d_{t−1}] − mean_X·mean_Y
- Stability: require at least N_min differences per band (e.g., N_min = 10k); otherwise merge with nearest band.

Band assignment:
- Compute per-dimension calibration stats (mean|logit| or frequency) over T tokens.
- Partition into B bands by quantiles of mean|logit| or token frequency. B ∈ {8, 16} works well in preliminary runs (100k tokens) by minimizing within-band variance while keeping estimates stable.
- Shrinkage toward global means: Q̃_b = (1−λ) Q̂_b + λ mean_b(Q̂_b), R̃_b analogously; λ ∈ [0.1, 0.5] chosen via held-out PPL.

Model diagnostics (falsification):
- Report per-band fractions with γ̂1 ≥ 0 and with |γ̂h| (|h|>1) exceeding a small threshold; correlate with ΔPPL to detect mismatch.

### 2.4 Implementation variants

Full vocabulary:
- State: ŝ ∈ R^V; gain K ∈ R^V or per-band.
- Per step: fused multiply-add over V with memory traffic ~3V elements; store ŝ,K in fp16, compute in fp32.

Top-M active state (recommended):
- Maintain a dynamic set S_t of size M:
  - S_t = top-k_y(y_t, k=M/2) ∪ top-k_s(ŝ_{t−1}, k=M/2) ∪ R (optional small reservoir of frequent tokens).
  - If |S_t| > M: evict by ascending priority p_i = max(y_t[i], ŝ_{t−1}[i]).
- Update only i ∈ S_t; for i ∉ S_t, pass-through: z_t[i] = y_t[i].
- Re-entry: if i enters S_t for the first time or after eviction, initialize ŝ[i] = y_t[i] (cold start).
- Complexity: O(M) compute and memory access per step. Preliminary microbenchmarks on Pythia-1.4B show <5% latency overhead at M=1024.

Producing z_t without dense writes:
- If downstream decoding uses top-k or nucleus sampling, compute the sampling set T_t on y_t, map to z_t by applying updates only for i ∈ S_t ∩ T_t; for i ∈ T_t \ S_t, use y_t[i]. No need to materialize dense z_t.
- For greedy decoding, only argmax is needed; adjust only indices considered in top-k_y.

Optional variance rescaling (temperature-safe mode):
- Filtering reduces variance. Optionally scale around the mean over updated indices: z_t = μ̂ + a (ŝ_t − μ̂), with a = std(y_t; i ∈ S_t)/std(ŝ_t; i ∈ S_t). Default off; report both.

Sequence boundaries and warm-up:
- At sequence start, set ŝ_0 = y_0 and skip updates for the first W tokens (e.g., W=2–4) to accumulate a stable prior.
- Reset state between independent sequences; for beam search, maintain one ŝ per beam.

### 2.5 Pseudocode

Calibration (streaming, per band):
```python
# Inputs: quantized logits y_t (t=0..T), band(i): {0..B-1}
# Outputs: per band b: K[b], with intermediate Q[b], R[b]
# Robustify via clip_or_huber()
stats = {b: dict(n=0, sum_d=0.0, sum_d2=0.0, sum_dlag1=0.0,
                 d_first=None, d_prev=None) for b in range(B)}
for t in range(1, T+1):
    d = y[t] - y[t-1]  # vector length V
    for i in range(V):
        b = band(i)
        di = clip_or_huber(d[i])
        s = stats[b]
        if s['n'] == 0:
            s['d_first'] = di
        else:
            s['sum_dlag1'] += di * s['d_prev']
        s['n'] += 1
        s['sum_d'] += di
        s['sum_d2'] += di * di
        s['d_prev'] = di

eps = 1e-8
for b in range(B):
    s = stats[b]
    if s['n'] < 2:
        continue
    n = s['n']
    mean_d = s['sum_d'] / n
    gamma0 = (s['sum_d2'] - n * mean_d * mean_d) / (n - 1)
    mean_X = (s['sum_d'] - s['d_first']) / (n - 1)
    mean_Y = (s['sum_d'] - s['d_prev']) / (n - 1)
    Edlag1 = s['sum_dlag1'] / (n - 1)
    gamma1 = Edlag1 - mean_X * mean_Y
    R = max(eps, -gamma1)
    Q = max(eps, gamma0 - 2.0 * R)
    A = math.sqrt(Q*Q + 4.0*Q*R)
    K[b] = (A + Q) / (A + Q + 2.0*R)
```

Inference (per token):
```python
# Inputs: y_t (logits), ŝ (state, sparse: indices->values), K per band b, M
# Outputs: z_t (virtual view for decoding), ŝ updated in-place

y_t = model_logits(inputs)  # quantized model
I_y = topk_indices(y_t, k=M//2)
I_s = topk_indices_sparse(ŝ, k=M//2)
S_t = union_with_reservoir(I_y, I_s, reservoir_R)
if len(S_t) > M:
    S_t = evict_by_priority(S_t, y_t, ŝ, M)  # keep top by max(y_t[i], ŝ.get(i, -inf))

for i in S_t:
    s_i = ŝ.get(i, y_t[i])   # cold-start from observation
    e = y_t[i] - s_i
    k = K[band(i)]
    s_new = s_i + k * e
    ŝ[i] = s_new

# Decoding without dense materialization:
# - For greedy or top-k/nucleus sampling, fetch candidate set T_t and use:
#   score(i) = ŝ[i] if i in S_t else y_t[i]
token = sample_with_virtual_logits(y_t, ŝ, S_t, decoding_cfg)
```

Engineering notes:
- Store ŝ and K in fp16; compute updates in fp32 accumulators.
- Use stable sqrt and clamp Q,R,K to avoid NaNs under extreme clipping.
- Fuse selection and update kernels to minimize memory passes; precompute and cache band(i).

## 3. Experiments (Falsification Plan)

Hypotheses:
- H1: At matched or higher tok/s vs quantized baseline, KLS reduces perplexity; gains increase as bit-width decreases.
- H2: KLS narrows the gap to FP16 without changing weights or decoding policy.
- H3: Miscalibrated gains (e.g., shuffled or scaled K) degrade performance toward/below baseline, excluding trivial temperature effects.
- H4: Δy MA(1) diagnostics (γ1 < 0, higher lags ≈ 0) predict help; violations predict harm.

Setup:
- Models: Pythia-1.4B/2.8B, Mistral-7B (base/instruct).
- Quantization: 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).
- Data/Tasks:
  - Perplexity: WikiText-103, C4 val.
  - Accuracy: ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).
- Conditions:
  1) FP16 (no quantization)
  2) Quantized baseline
  3) Quantized + KLS (full-V, top-M variants)
- Calibration:
  - 50k–200k tokens from C4; no labels; only quantized logits.
  - Bands B ∈ {8,16}; minimum N_min = 10k diffs per band; shrinkage λ ∈ {0.1,0.3}.
  - Robustification on.
- Latency control: report end-to-end tok/s and p95 latency; enforce equal-or-faster tok/s for KLS than quantized baseline by adjusting M.

Ablations:
- Gain scaling: scale K by c ∈ {0, 0.25, 0.5, 1, 1.5}.
- Calibration distortion: multiply R by {0.25, 4} and recompute K; same for Q.
- Top-M size: M ∈ {128, 256, 512, 1024, 2048}.
- Rescaling: on vs off.
- Banding: B ∈ {1 (global K), 8, 16}.
- Decoding modes: greedy, top-k, nucleus; beam search (independent ŝ per beam).
- Diagnostics: fraction of bands with γ̂1 ≥ 0; correlation with ΔPPL.

Reporting:
- Perplexity deltas (absolute and relative) with 95% CIs via bootstrap over documents.
- Accuracy deltas on tasks (same seeds/decoding).
- Throughput and memory overhead; GPU kernel profile showing added read/write bandwidth per token.

## 4. Discussion
- Why it helps: Logit fields evolve smoothly in distribution even when argmax flips. Quantization noise is approximately additive and weakly time-correlated relative to that signal; the local-level model captures this with just Q and R.
- When it hurts: Topic/prompt shifts (transient large Q), aggressive pruning (small M), or model mismatch (γ̂1 ≈ 0/+). Diagnostics flag these cases.
- Orthogonality: KLS composes with quantizers and decoding strategies; it is weight-free and post-hoc.

## 5. Limitations
- Diagonal assumption ignores cross-logit correlations; multivariate filters could help but raise cost/complexity.
- Heavy-tailed or structured noise can break Δy moment identities; robustification mitigates but not fully.
- Benefits diminish at higher bit-widths (small R); full-V updates can be bandwidth-bound for large V.
- Non-autoregressive/batched generation reduces temporal correlation; state resets may limit gains.

## 6. Conclusion
KLS is a practical, theoretically grounded post-processing step for quantized LLM inference. A closed-form, quantization-only calibration using Δy moments yields reproducible gains with negligible overhead in a top-M implementation. The proposed experiments and diagnostics can validate or falsify the approach with open models and code.

## Reproducibility checklist
- Reference implementation (CUDA/CPU) with:
  - Streaming calibration (γ0, γ1) per band, robust options (clip/Huber), deterministic seeds.
  - Top-M state maintenance with explicit union/eviction/re-entry, and per-beam states.
  - “Virtual logits” decoding to avoid dense writes; kernels fused with selection.
  - Benchmarks reporting tok/s, p95 latency, and memory bandwidth.
- Release calibration logs (γ0/γ1 histograms), learned K per band, and scripts to reproduce plots/tables.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
