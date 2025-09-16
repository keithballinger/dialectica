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
- Strengths: The core idea—post-hoc, weight-free, online denoising of quantized logits via a steady-state scalar Kalman filter—is novel for LLM inference, theoretically sound under a local-level model, and practical to validate on small open models. The MoM identification via first-difference autocovariances is correct; complexity and active-set design are largely appropriate.
- Gaps to address:
  - Decoding pipeline placement: Specify exact order relative to temperature, repetition penalty, and other LogitsProcessors; provide a consistent nucleus variant.
  - Quantization bias: Acknowledge and offer an optional per-band affine de-bias or magnitude-conditioned banding to address non-zero-mean and magnitude-dependent quantization errors.
  - Online adaptation: Add an EW/rolling update option to handle domain shifts without re-calibration.
  - Experimental plan: Include ablations for pipeline order, top-p consistency, and penalty interactions; report entropy/diversity changes to guard against oversmoothing.
  - Implementation details: Clarify GPU kernel fusion options, memory traffic, and where O(V) scanning is reused; note batch/beam-safe state handling and per-sequence RNG determinism.
  - Theory and diagnostics: Briefly connect logit MSE reduction to expected cross-entropy improvements; add an explicit derivation of the MoM identities and mismatch checks (e.g., γ̂1 sign) with thresholds.

Revised Draft
# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

## Abstract
Low-precision quantization accelerates LLM inference but perturbs pre-softmax logits, raising perplexity and harming downstream accuracy. We model logit evolution as a local-level state-space process and introduce Kalman-Logit Smoothing (KLS), a lightweight online filter that denoises quantized logits per decoding step. KLS is post-hoc (no retraining), calibration-only (using a short stream of quantized logits), and adds O(M) overhead beyond the standard O(V) top-k/top-p scan via a top-M active set. We derive a diagonal steady-state Kalman filter, provide robust streaming method-of-moments estimators for process and observation noise, and give numerically stable algorithms for full-vocabulary and top-M variants. We specify correct integration with greedy/top-k/nucleus/beam decoding (including candidate-set logic), pipeline placement relative to temperature and penalties, and a “consistent nucleus” option. A falsification plan with open models (Pythia, Mistral), datasets (WikiText-103, C4), and 4–8 bit quantization (NF4, GPTQ, k-bit) includes strong EMA baselines, ablations, latency, and diagnostics for model mismatch.

## 1. Introduction
Low-bit quantization (4–8 bits) reduces LLM inference cost but injects noise into pre-softmax logits. Mitigations typically alter quantization or fine-tune models. We propose an orthogonal, post-hoc approach: online denoising of the final logit vector before decoding, requiring no weight changes.

We treat logits as a multivariate time series. Each token dimension’s latent “true” logit evolves smoothly across steps; the observed quantized logit is a noisy measurement. Kalman-Logit Smoothing (KLS) applies an independent scalar steady-state Kalman filter per logit (or per band of logits) to produce a denoised estimate on-the-fly.

Contributions:
- A state-space formalization of quantized logit evolution and a corresponding online, diagonal steady-state Kalman filter integrating with decoding at O(M) overhead beyond standard top-k/top-p.
- A quantization-only calibration via method-of-moments on first-difference autocovariances, implemented as a robust streaming estimator with sequence-safe handling.
- Practical algorithms: numerically stable gain computation, active-set maintenance, correct greedy/top-k/nucleus/beam integration, and an optional consistent nucleus variant.
- Diagnostics for model mismatch and a falsification plan with strong EMA baselines, ablations (banding, active-set size, pipeline order), and latency profiling.

Note: While named “smoothing,” the online component is a steady-state filter (no look-ahead).

## 2. Method

### 2.1 Local-Level State-Space Model
For vocabulary size V and time step t:
- State: s_{t,i} = s_{t−1,i} + w_{t,i}, w_{t,i} ~ N(0, Q_i).
- Observation: y_{t,i} = s_{t,i} + v_{t,i}, v_{t,i} ~ N(0, R_i).

Assume diagonal Q and R (independent scalar filters). To reduce parameters and capture heteroskedasticity from quantization, group dimensions into B ≪ V bands sharing (Q_b, R_b), e.g., by logit magnitude or token frequency.

Optional bias handling:
- Quantization can induce non-zero-mean and magnitude-dependent errors. Two lightweight remedies:
  - Magnitude-conditioned bands (default): band by |y| quantiles.
  - Optional per-band affine preconditioning: y′ = a_b y + c_b with a_b > 0, c_b small, estimated during calibration by regressing y differences on magnitude buckets (details in §2.3). Differences remove constant bias; affine preconditioning addresses magnitude-dependent bias.

### 2.2 Steady-State Scalar Kalman Filter
Let X_i solve X_i^2 − Q_i X_i − Q_i R_i = 0; choose the positive root X_i = 0.5(Q_i + sqrt(Q_i^2 + 4 Q_i R_i)). The steady-state gain is
- K_i = X_i / (X_i + R_i) = Q_i / X_i = (sqrt(Q_i^2 + 4 Q_i R_i) − Q_i) / (2 R_i).

Online update per dimension i:
ŝ_{t,i} = ŝ_{t−1,i} + K_i (y_{t,i} − ŝ_{t−1,i}), initialized with ŝ_{0,i} = y_{0,i}.

Numerical stability:
- Compute in float64; clamp Q_i, R_i ≥ ε (1e-12).
- Prefer K_i = Q_i / X_i to avoid cancellation when Q_i ≪ R_i.
- Clamp K_i ∈ [K_min, 1 − K_min] (e.g., K_min = 1e-4).

Interpretation: A principled EMA with α = K_i learned from (Q_i, R_i). As R→0, K→1; as Q→0, K→0.

### 2.3 Calibration via Method of Moments
Define differences d_{t,i} = y_{t,i} − y_{t−1,i}. Under the model:
- γ_0 = Var(d_t) = Q + 2R.
- γ_1 = Cov(d_t, d_{t−1}) = −R.
- γ_h ≈ 0 for |h| > 1.

Per band b with pooled estimates (γ̂_0, γ̂_1):
- R̂_b = max(ε, −γ̂_1), Q̂_b = max(ε, γ̂_0 + 2 γ̂_1).
- Gain K_b from §2.2.

Estimation details:
- Streaming, unbiased estimators for variance and lag-1 covariance over T tokens, pooled over dimensions in the band.
- Maintain per-i d_prev to form (d_t, d_{t−1}) pairs correctly.
- Robustify with clipping/Huber to mitigate heavy tails; de-mean d_t if empirical drift exists.
- Optional shrinkage across bands with λ ∈ [0.1, 0.5].
- Optional affine preconditioning: stratify by |y| buckets within a band and fit y′ = a y + c to reduce magnitude-dependent bias, then compute moments on y′.
- Sequence boundaries: reset y_prev and clear all per-i d_prev between independent sequences.

Online adaptation (optional):
- Maintain exponential-weighted moments with decay β (e.g., 0.99–0.999) during inference to adapt K_b slowly across domains, bounded within [0.5×, 2×] of calibrated values.

Diagnostics:
- Report fraction of bands with γ̂_1 ≥ 0 (should be near 0 under the model) as a mismatch indicator; rising values predict reduced gains.
- Track stability of K_b and the effective α vs. a tuned EMA baseline.

### 2.4 Integration with Decoding

Pipeline placement and order:
- Apply KLS after all deterministic logit transforms that express user policy (e.g., repetition penalty, min-length, bad-words) and before softmax/sampling.
- Temperature: apply temperature scaling before KLS to preserve the user’s entropy target, or treat temperature as a final rescale after KLS; both work but choose one consistently. We default to: penalties → temperature → KLS → sampling.
- LogitsProcessors (HF): register KLS as the last LogitsProcessor.

Decoding modes:
- Greedy: compare max over ŝ on active set S_t vs. max over raw y on V \ S_t (correct mixed-domain argmax).
- Top-k/top-p (nucleus):
  - Standard: build candidates T_y from y (as usual) and augment with top-L from ŝ on S_t to avoid missing tokens promoted by smoothing; score T_t = T_y ∪ T_s using ŝ on S_t and y elsewhere.
  - Consistent nucleus (optional): recompute nucleus over T_t using these mixed scores. Exact nucleus under full ŝ would require O(V) regardless; this option provides a consistent approximation while keeping O(M) overhead.

Beam search:
- Maintain independent states per beam; copy/update on beam expansion; reset on sequence start/end.

Warm-up and resets:
- Warm-up W≈2–4 steps (skip updates or ramp K linearly) to bypass transient from cold start.
- Reset all state at sequence boundaries; in batching, keep per-sequence state isolation and deterministic RNG behavior.

### 2.5 Active-Set Implementation

Variants:
- Full-V: update all V dimensions (high memory traffic).
- Top-M active set (recommended): update only a dynamic set S_t of size M (e.g., 512–2048).

Active-set algorithm:
1) Candidate selection:
   - I_y = top-k(y_t, k = M/2) over V (reuses standard O(V) scan).
   - I_s = top-k over current ŝ entries (≤ M).
2) Set S_t = I_y ∪ I_s; if |S_t| > M, evict by priority p_i = max(y_{t,i}, ŝ_{t−1,i}).
3) Update i ∈ S_t: s_prev = state.get(i, y_{t,i}); s_new = s_prev + K_{band(i)} (y_{t,i} − s_prev); state[i] = s_new.
4) Greedy or sampling:
   - Greedy: compare max ŝ on S_t vs. max y on V \ S_t.
   - Top-k/top-p: T_y from standard policy; T_s as top-L over ŝ on S_t; T_t = T_y ∪ T_s; score with ŝ on S_t else y. Optional consistent nucleus on T_t.

Complexity and kernels:
- The O(V) top-k/top-p scan is standard. KLS adds O(M) compute/memory traffic for active-set maintenance and updates.
- Implement as a fused CUDA kernel that: computes top-k over y, merges with top-ŝ, updates S_t in-place, and emits candidate scores. Avoid CPU round-trips. Store ŝ and K in fp16/bfloat16; compute updates in fp32/64.

### 2.6 Pseudocode (Top-M, streaming calibration)

Calibration (per band, with resets and robustification):
```python
eps = 1e-12
for b in bands: init stats_b with unbiased var/cov accumulators and d_prev map

for seq in streams:
    for b in bands: stats_b.d_prev.clear()
    y_prev = next(seq, None)
    if y_prev is None: continue
    for y in seq:                         # y is V-dim quantized logits
        d = y - y_prev
        for i, di in enumerate(d):
            b = band_map[i]
            di = huber(di, delta)         # optional robustification
            s = stats[b]
            s.add_variance_sample(di)
            if i in s.d_prev:
                s.add_lag1_pair(di, s.d_prev[i])
            s.d_prev[i] = di
        y_prev = y

for b in bands:
    gamma0 = stats[b].unbiased_var()
    gamma1 = stats[b].unbiased_lag1_cov()
    R = max(eps, -gamma1); Q = max(eps, gamma0 + 2*gamma1)
    # optional shrinkage and affine preconditioning
    A = math.sqrt(Q*Q + 4*Q*R)
    X = 0.5*(Q + A)
    K[b] = clip(Q / X, 1e-4, 1 - 1e-4)
```

Inference (Top-M with correct argmax and candidate union):
```python
def kls_step(y_t, state, K_map, M, build_Ty):
    Iy = topk_indices(y_t, k=M//2)            # O(V), reused from baseline
    Is = state.topk_indices(k=M//2)           # ≤ M
    S = union(Iy, Is)
    if len(S) > M: S = evict_by_priority(S, y_t, state, M)

    for i in S:
        s_prev = state.get(i, y_t[i])
        k = K_map[band(i)]
        state[i] = s_prev + k * (y_t[i] - s_prev)

    # Greedy argmax
    m_in, idx_in = max_over(state, S)         # ŝ on S
    m_out, idx_out = max_over(y_t, complement(S))
    greedy_idx = idx_in if m_in >= m_out else idx_out

    # Sampling
    Ty = build_Ty(y_t)                        # standard top-k/top-p
    Ts = topk_on_state(state, L=128)          # from ŝ on S
    T = union(Ty, Ts)
    scores = {i: (state[i] if i in state else y_t[i]) for i in T}

    # Optional: consistent nucleus over T using 'scores'
    return greedy_idx, T, scores, state
```

## 3. Theory and Expected Impact on Log Loss
Under small additive observation noise v with Var(v)=R and a locally smooth state s, the filter reduces logit MSE E[(ŝ−s)^2] relative to y by a factor depending on K (standard Kalman algebra). For softmax cross-entropy with true class c and small perturbations, a second-order expansion yields an expected loss reduction approximately proportional to the MSE reduction on y_c and to the curvature of log-sum-exp at s. Thus, when R is non-negligible (low-bit quantization), KLS should reduce perplexity; diminishing gains are expected as bit-width increases (R→0).

## 4. Experiments (Falsification Plan)

Hypotheses:
- H1: At matched or higher throughput vs. quantized baseline, KLS reduces perplexity, with larger gains at lower bit widths.
- H2: KLS narrows the gap to FP16 baselines on perplexity and downstream tasks.
- H3: Gains disappear or reverse when K is distorted/shuffled or γ̂_1 mismatch rises.
- H4: A tuned EMA baseline cannot match KLS across models/bits at similar overhead.

Models/quantization:
- Pythia-1.4B/2.8B, Mistral-7B (base/instruct).
- 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).

Data/tasks:
- Perplexity: WikiText-103, C4 validation.
- Tasks: ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).

Conditions:
1) FP16 baseline.
2) Quantized baseline.
3) Quantized + KLS (Top-M, M∈{512,1024}).

Baselines/ablations:
- EMA: Exponential moving average of logits with α grid-searched (global and per-band).
- Banding: B ∈ {1, 8, 16, V}; magnitude-conditioned vs. frequency-based.
- Calibration distortion: scale Q̂, R̂ by {0.25, 0.5, 2, 4}; shuffle K across bands.
- Pipeline order: penalties → KLS → temperature vs. penalties → temperature → KLS.
- Nucleus consistency: standard T_y-only vs. candidate union vs. consistent nucleus on T.
- Online adaptation: static K vs. EW-updated moments (β ∈ {0.99, 0.995, 0.999}).
- Active set: M ∈ {128, 512, 1024, 2048}; L ∈ {32, 128, 256}.
- Diversity: entropy of sampled distributions, distinct-n, repetition rates.

Latency/throughput:
- Tokens/sec and p95 latency. Report overhead breakdown: O(V) scan (shared) vs. O(M) KLS update; GPU kernel fusion impact; memory traffic (reads/writes per step).

Diagnostics:
- Fraction of bands with γ̂_1 ≥ 0; correlation with Δperplexity.
- Stability of K_b and sensitivity to banding.

## 5. Implementation Notes
- HF integration: Provide a LogitsProcessor that applies KLS after penalties and temperature, before sampling; supports greedy/top-k/top-p/beam. Deterministic under fixed seeds.
- GPU kernels: Fuse top-k scan, active-set update, and candidate scoring; keep state (ŝ, K_b) in device memory; prefer fp16 storage with fp32 compute. Avoid host-device sync.
- Batching/beam: Maintain per-sequence (and per-beam) states; reset at boundaries; handle variable-length batches.
- Numerical: Use float64 for K computation; clamp Q,R,K; robustify moments with Huber loss; set W warm-up steps.

## 6. Related Work
- Quantized LLM inference: GPTQ, AWQ, NF4 and efficient kernels focus on weight/activation quantization. KLS is orthogonal as a post-hoc logit denoiser.
- Temporal smoothing: EMA and heuristic smoothing have been used for stabilization; KLS provides a principled, identifiable α via state-space modeling.
- Output calibration: Temperature scaling and repetition penalties alter decoding policies; KLS targets noise reduction prior to sampling, not policy bias.

## 7. Limitations
- Diagonal filtering ignores cross-token correlations; multivariate filters are costlier.
- Method-of-moments assumes additive, near-white observation noise; deterministic quantizers can induce bias and temporal correlation; banding and robustification mitigate but do not eliminate mismatch.
- Gains diminish at higher bit-widths (small R).
- Approximate nucleus with candidate union may deviate slightly from full-ŝ nucleus; a consistent variant is provided but remains approximate unless O(V) is used.

## 8. Conclusion
KLS is a practical, theoretically grounded, calibration-only method that improves decoding quality under quantization with minimal overhead. It integrates cleanly with standard decoding, includes diagnostics for applicability, and is readily validated with open models and code.

## Appendix A: Derivation of MoM Identities
With s_t = s_{t−1} + w_t, y_t = s_t + v_t, define d_t = y_t − y_{t−1} = w_t + v_t − v_{t−1}. Then
- Var(d_t) = Var(w_t) + Var(v_t) + Var(v_{t−1}) = Q + 2R.
- Cov(d_t, d_{t−1}) = Cov(w_t + v_t − v_{t−1}, w_{t−1} + v_{t−1} − v_{t−2}) = −Var(v_{t−1}) = −R,
assuming independence across time and between w and v. Higher-lag covariances vanish.

## Reproducibility Checklist
- Code (CUDA/PyTorch):
  - Streaming calibration with robustification, sequence-boundary resets, banding, optional affine preconditioning, and online adaptation.
  - Fused kernel for top-k scan, active-set update, and candidate scoring; correct greedy/candidate logic.
  - HuggingFace LogitsProcessor integration (greedy, top-k, top-p, beam), deterministic under seeds.
  - Benchmarks: tokens/sec, p95 latency, memory traffic.
- Data:
  - Release learned (Q, R, K) per model/quantization; per-band γ̂_0, γ̂_1, mismatch diagnostics.
  - Scripts for all baselines/ablations, including EMA and pipeline-order studies.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
