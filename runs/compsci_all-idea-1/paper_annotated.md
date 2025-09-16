Minor Revisions

# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference
> Note: This paper proposes a lightweight statistical filter to clean up noisy logits (the raw scores before softmax) produced by low-bit quantized large language models (LLMs), aiming to recover accuracy with minimal runtime cost.

## Abstract
Low-precision quantization accelerates LLM inference but perturbs pre-softmax logits, raising perplexity and harming downstream accuracy. We model logit evolution as a local-level state-space process and introduce Kalman-Logit Smoothing (KLS), a lightweight online filter that denoises quantized logits per decoding step. KLS is post-hoc (no retraining), calibration-only (using a short stream of quantized logits), and adds O(M) overhead beyond the standard O(V) top-k/top-p scan via a top-M active set. We derive a diagonal steady-state Kalman filter, provide robust streaming method-of-moments estimators for process and observation noise, and give numerically stable algorithms for full-vocabulary and top-M variants. We specify correct integration with greedy/top-k/nucleus/beam decoding (including candidate-set logic), pipeline placement relative to temperature and penalties, and a “consistent nucleus” option. A falsification plan with open models (Pythia, Mistral), datasets (WikiText-103, C4), and 4–8 bit quantization (NF4, GPTQ, k-bit) includes strong EMA baselines, ablations, latency, and diagnostics for model mismatch.
> Note: Key ideas:
> - Pre-softmax logits get noisier under low-bit quantization, increasing perplexity (a measure of predictive uncertainty).
> - Treat each token’s logit over time as a hidden smooth signal plus noise; denoise it with a simple Kalman filter per token dimension.
> - O(V) is the usual cost of scanning all V vocabulary logits; KLS adds O(M) work by only filtering a small active set of size M (e.g., 512–2048).
> - “Diagonal” means each logit is filtered independently; “steady-state” means the filter gain is fixed after calibration.
> - EMA = exponential moving average baseline; KLS aims to learn the EMA weight from data via method-of-moments.
> - The plan tests on open models/datasets and multiple quantizers and measures both quality and latency.

## 1. Introduction
Low-bit quantization (4–8 bits) reduces LLM inference cost but injects noise into pre-softmax logits. Mitigations typically alter quantization or fine-tune models. We propose an orthogonal, post-hoc approach: online denoising of the final logit vector before decoding, requiring no weight changes.
> Note: Quantization compresses weights/activations to fewer bits (e.g., 4–8), which speeds up inference but adds error to logits. Instead of changing the model, KLS cleans logits right before sampling the next token.

We treat logits as a multivariate time series. Each token dimension’s latent “true” logit evolves smoothly across steps; the observed quantized logit is a noisy measurement. Kalman-Logit Smoothing (KLS) applies an independent scalar steady-state Kalman filter per logit (or per band of logits) to produce a denoised estimate on-the-fly.
> Note: Think of each logit as a temperature reading over time: the true temperature changes gradually, but your thermometer (quantized logits) is noisy. A Kalman filter fuses past and current readings to get a better estimate.

Contributions:
- A state-space formalization of quantized logit evolution and a corresponding online, diagonal steady-state Kalman filter integrating with decoding at O(M) overhead beyond standard top-k/top-p.
> Note: Formal model and efficient runtime integration: adds small overhead (proportional to the active set size M) to the usual full-vocabulary scan.

- A quantization-only calibration via method-of-moments on first-difference autocovariances, implemented as a robust streaming estimator with sequence-safe handling.
> Note: Calibrates filter parameters from short streams of quantized logits using simple variance/covariance identities, robust to outliers and respecting sequence boundaries.

- Practical algorithms: numerically stable gain computation, active-set maintenance, correct greedy/top-k/nucleus/beam integration, and an optional consistent nucleus variant.
> Note: Covers details needed to avoid numerical issues, keep only the most relevant tokens in memory, and work correctly with common decoding strategies (greedy, top-k, top-p/nucleus, beam).

- Diagnostics for model mismatch and a falsification plan with strong EMA baselines, ablations (banding, active-set size, pipeline order), and latency profiling.
> Note: Provides tests to show when/why KLS helps or fails (e.g., if noise assumptions are violated) and compares against tuned EMA to ensure gains are not just from simple smoothing.

Note: While named “smoothing,” the online component is a steady-state filter (no look-ahead).
> Note: “Smoothing” often implies using future information; here the online version uses only past and current steps (a filter), with fixed gain after calibration.

## 2. Method
> Note: This section defines the statistical model, how to compute the filter gain from data, and how to integrate it efficiently into decoding.

### 2.1 Local-Level State-Space Model
For vocabulary size V and time step t:
- State: s_{t,i} = s_{t−1,i} + w_{t,i}, w_{t,i} ~ N(0, Q_i).
> Note: Definitions:
> - V: number of tokens in the vocabulary.
> - t: decoding step (time index).
> - i: token/logit index (1..V).
> - s_{t,i}: hidden “true” logit at time t for token i.
> - w_{t,i}: process noise (how much the true logit changes step to step), assumed Gaussian with variance Q_i.
> - Q_i: process noise variance for token i (larger Q_i = faster-changing true logit).

- Observation: y_{t,i} = s_{t,i} + v_{t,i}, v_{t,i} ~ N(0, R_i).
> Note: Definitions:
> - y_{t,i}: observed quantized logit at time t for token i.
> - v_{t,i}: observation noise from quantization etc., Gaussian with variance R_i.
> - R_i: observation noise variance (larger R_i = noisier measurement).

Assume diagonal Q and R (independent scalar filters). To reduce parameters and capture heteroskedasticity from quantization, group dimensions into B ≪ V bands sharing (Q_b, R_b), e.g., by logit magnitude or token frequency.
> Note: “Diagonal” means no coupling across tokens: each token has its own 1D filter. To limit parameters, group tokens into B bands (e.g., by |logit|) and share Q and R within each band to reflect that some logits are systematically noisier.

Optional bias handling:
> Note: Quantizers can introduce bias (systematic shift) that depends on logit size; the following options mitigate that.

- Quantization can induce non-zero-mean and magnitude-dependent errors. Two lightweight remedies:
  - Magnitude-conditioned bands (default): band by |y| quantiles.
> Note: Group tokens by the size of their observed logits |y|; this captures that large-magnitude logits may have different noise/bias than small ones.

  - Optional per-band affine preconditioning: y′ = a_b y + c_b with a_b > 0, c_b small, estimated during calibration by regressing y differences on magnitude buckets (details in §2.3). Differences remove constant bias; affine preconditioning addresses magnitude-dependent bias.
> Note: Definitions:
> - y′: adjusted logit.
> - a_b, c_b: per-band scale and offset (a_b positive). This small linear correction reduces magnitude-dependent bias. Using differences removes constant bias; the affine step addresses residual slope effects.

### 2.2 Steady-State Scalar Kalman Filter
Let X_i solve X_i^2 − Q_i X_i − Q_i R_i = 0; choose the positive root X_i = 0.5(Q_i + sqrt(Q_i^2 + 4 Q_i R_i)). The steady-state gain is
- K_i = X_i / (X_i + R_i) = Q_i / X_i = (sqrt(Q_i^2 + 4 Q_i R_i) − Q_i) / (2 R_i).
> Note: Definitions:
> - X_i: steady-state error covariance term solving a quadratic from Kalman algebra.
> - K_i: steady-state Kalman gain (how much to trust the new observation vs. the previous estimate). Algebraically equivalent forms are given to improve numerical stability.

Online update per dimension i:
ŝ_{t,i} = ŝ_{t−1,i} + K_i (y_{t,i} − ŝ_{t−1,i}), initialized with ŝ_{0,i} = y_{0,i}.
> Note: Definitions:
> - ŝ_{t,i}: filtered (denoised) estimate of the true logit at time t.
> - Update rule: move the previous estimate toward the new observation by fraction K_i of the residual. Initialize with the first observation.

Numerical stability:
- Compute in float64; clamp Q_i, R_i ≥ ε (1e-12).
> Note: Avoid underflow/negative variances by using higher precision and minimum floors.

- Prefer K_i = Q_i / X_i to avoid cancellation when Q_i ≪ R_i.
> Note: This formula avoids subtracting nearly equal numbers, which can cause precision loss when observation noise dominates.

- Clamp K_i ∈ [K_min, 1 − K_min] (e.g., K_min = 1e-4).
> Note: Prevents extreme gains (0 or 1) that can destabilize updates due to finite precision or model mismatch.

Interpretation: A principled EMA with α = K_i learned from (Q_i, R_i). As R→0, K→1; as Q→0, K→0.
> Note: EMA analogy:
> - α (EMA weight) corresponds to K_i.
> - If measurements are nearly noise-free (R small), trust y more (K→1).
> - If the true signal barely changes (Q small), trust the past more (K→0).

### 2.3 Calibration via Method of Moments
Define differences d_{t,i} = y_{t,i} − y_{t−1,i}. Under the model:
- γ_0 = Var(d_t) = Q + 2R.
> Note: Definitions:
> - d_{t,i}: step-to-step change in observed logits for token i.
> - γ_0: variance of differences; equals process variance plus twice the observation variance (per band or per token).

- γ_1 = Cov(d_t, d_{t−1}) = −R.
> Note: γ_1: lag-1 covariance of differences; negative and equal to −R because shared v_{t−1} appears with opposite signs in d_t and d_{t−1}.

- γ_h ≈ 0 for |h| > 1.
> Note: Higher-lag covariances vanish in this simple model due to independence over time.

Per band b with pooled estimates (γ̂_0, γ̂_1):
- R̂_b = max(ε, −γ̂_1), Q̂_b = max(ε, γ̂_0 + 2 γ̂_1).
> Note: Estimators:
> - γ̂_0, γ̂_1: empirical estimates pooled over tokens in band b.
> - R̂_b from −γ̂_1; Q̂_b from γ̂_0 + 2γ̂_1; clamp to small ε to avoid nonpositive values.

- Gain K_b from §2.2.
> Note: Once Q̂_b and R̂_b are estimated, compute K_b using the steady-state formula.

Estimation details:
- Streaming, unbiased estimators for variance and lag-1 covariance over T tokens, pooled over dimensions in the band.
> Note: Compute γ̂_0 and γ̂_1 on-the-fly without storing all data; pool across tokens for robustness.

- Maintain per-i d_prev to form (d_t, d_{t−1}) pairs correctly.
> Note: For each token, remember its previous difference to compute the lag-1 covariance correctly.

- Robustify with clipping/Huber to mitigate heavy tails; de-mean d_t if empirical drift exists.
> Note: Huber/clipping reduces the influence of outliers; subtracting the mean difference corrects small drifts violating the zero-mean assumption.

- Optional shrinkage across bands with λ ∈ [0.1, 0.5].
> Note: λ: shrinkage factor to pull noisy band estimates toward a global average, improving stability.

- Optional affine preconditioning: stratify by |y| buckets within a band and fit y′ = a y + c to reduce magnitude-dependent bias, then compute moments on y′.
> Note: Apply linear correction before estimating moments if bias varies with |y|.

- Sequence boundaries: reset y_prev and clear all per-i d_prev between independent sequences.
> Note: Prevents mixing statistics across unrelated prompts or documents.

Online adaptation (optional):
- Maintain exponential-weighted moments with decay β (e.g., 0.99–0.999) during inference to adapt K_b slowly across domains, bounded within [0.5×, 2×] of calibrated values.
> Note: β: forgetting factor for updating γ̂ online; allows gentle adaptation while constraining drift to avoid instability.

Diagnostics:
- Report fraction of bands with γ̂_1 ≥ 0 (should be near 0 under the model) as a mismatch indicator; rising values predict reduced gains.
> Note: If γ̂_1 is not negative, the noise model is likely wrong (e.g., correlated or biased noise).

- Track stability of K_b and the effective α vs. a tuned EMA baseline.
> Note: Compare learned K_b to the best EMA α to verify KLS isn’t just replicating a trivial setting.

### 2.4 Integration with Decoding

Pipeline placement and order:
- Apply KLS after all deterministic logit transforms that express user policy (e.g., repetition penalty, min-length, bad-words) and before softmax/sampling.
> Note: Denoise after enforcing constraints so you don’t undo policy effects, but before sampling where noise affects probabilities.

- Temperature: apply temperature scaling before KLS to preserve the user’s entropy target, or treat temperature as a final rescale after KLS; both work but choose one consistently. We default to: penalties → temperature → KLS → sampling.
> Note: Temperature rescales logits; fixing the order avoids double-counting effects. Default: do it before KLS.

- LogitsProcessors (HF): register KLS as the last LogitsProcessor.
> Note: In Hugging Face, ensure KLS runs last among processors that modify logits.

Decoding modes:
- Greedy: compare max over ŝ on active set S_t vs. max over raw y on V \ S_t (correct mixed-domain argmax).
> Note: Because only some tokens are filtered, compute the maximum carefully: use filtered scores where available and raw scores elsewhere.

- Top-k/top-p (nucleus):
  - Standard: build candidates T_y from y (as usual) and augment with top-L from ŝ on S_t to avoid missing tokens promoted by smoothing; score T_t = T_y ∪ T_s using ŝ on S_t and y elsewhere.
> Note: Definitions:
> - Top-k: sample from the k highest logits.
> - Top-p (nucleus): sample from the smallest set whose probabilities sum to p.
> - T_y: candidate set from raw logits; T_s: extra candidates that look better after filtering; T_t is their union. Score each candidate with ŝ if available, else y.

  - Consistent nucleus (optional): recompute nucleus over T_t using these mixed scores. Exact nucleus under full ŝ would require O(V) regardless; this option provides a consistent approximation while keeping O(M) overhead.
> Note: Ensures nucleus is defined with respect to the scores you actually use, without scanning all V tokens.

Beam search:
- Maintain independent states per beam; copy/update on beam expansion; reset on sequence start/end.
> Note: Each beam has its own filter state because its token history differs.

Warm-up and resets:
- Warm-up W≈2–4 steps (skip updates or ramp K linearly) to bypass transient from cold start.
> Note: Prevents early-step instability when the filter lacks history.

- Reset all state at sequence boundaries; in batching, keep per-sequence state isolation and deterministic RNG behavior.
> Note: Avoid state leakage between different prompts; keep reproducibility.

### 2.5 Active-Set Implementation

Variants:
- Full-V: update all V dimensions (high memory traffic).
> Note: Accurate but slow due to touching every logit.

- Top-M active set (recommended): update only a dynamic set S_t of size M (e.g., 512–2048).
> Note: Definitions:
> - M: number of tokens to filter/update each step (active set size).
> - S_t: indices of active tokens at time t. This limits cost to O(M).

Active-set algorithm:
1) Candidate selection:
   - I_y = top-k(y_t, k = M/2) over V (reuses standard O(V) scan).
> Note: I_y: top tokens by raw logits; you already compute this in normal decoding.

   - I_s = top-k over current ŝ entries (≤ M).
> Note: I_s: tokens that look strong after smoothing; limited to current active state.

2) Set S_t = I_y ∪ I_s; if |S_t| > M, evict by priority p_i = max(y_{t,i}, ŝ_{t−1,i}).
> Note: Merge and cap to M by keeping tokens with highest priority p_i (best of raw vs. previous smoothed).

3) Update i ∈ S_t: s_prev = state.get(i, y_{t,i}); s_new = s_prev + K_{band(i)} (y_{t,i} − s_prev); state[i] = s_new.
> Note: If a token is new to the state, initialize with its current y; update using the band-specific gain K_{band(i)}.

4) Greedy or sampling:
   - Greedy: compare max ŝ on S_t vs. max y on V \ S_t.
> Note: Ensures the argmax is correct even though only some tokens are smoothed.

   - Top-k/top-p: T_y from standard policy; T_s as top-L over ŝ on S_t; T_t = T_y ∪ T_s; score with ŝ on S_t else y. Optional consistent nucleus on T_t.
> Note: L: number of extra smoothed candidates to add; reduces the risk of missing tokens improved by smoothing.

Complexity and kernels:
- The O(V) top-k/top-p scan is standard. KLS adds O(M) compute/memory traffic for active-set maintenance and updates.
> Note: Most cost is already paid by baseline decoding; KLS’s extra work scales with M, not V.

- Implement as a fused CUDA kernel that: computes top-k over y, merges with top-ŝ, updates S_t in-place, and emits candidate scores. Avoid CPU round-trips. Store ŝ and K in fp16/bfloat16; compute updates in fp32/64.
> Note: Kernel fusion and on-GPU state minimize latency; mixed precision stores memory while preserving compute accuracy.

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
> Note: This computes γ̂_0 and γ̂_1 from differences d, then estimates R and Q per band and converts them to the steady-state gain K[b]. Variables:
> - bands: grouping of tokens.
> - band_map[i]: band index for token i.
> - huber(di, delta): robust clipping.
> - gamma0/gamma1: pooled moments.
> - A, X: intermediates for stable K computation.

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
> Note: Variables:
> - state: map from token index to smoothed score ŝ.
> - K_map: per-band gains.
> - build_Ty: function producing candidates under current top-k/top-p policy.
> - scores: mixed-domain scores (ŝ where available, else y), used for sampling.

## 3. Theory and Expected Impact on Log Loss
Under small additive observation noise v with Var(v)=R and a locally smooth state s, the filter reduces logit MSE E[(ŝ−s)^2] relative to y by a factor depending on K (standard Kalman algebra). For softmax cross-entropy with true class c and small perturbations, a second-order expansion yields an expected loss reduction approximately proportional to the MSE reduction on y_c and to the curvature of log-sum-exp at s. Thus, when R is non-negligible (low-bit quantization), KLS should reduce perplexity; diminishing gains are expected as bit-width increases (R→0).
> Note: Definitions:
> - MSE: mean squared error between estimate and true logit.
> - Softmax cross-entropy: loss for predicting class c from logits.
> - Reducing logit noise (especially for the true class c) lowers expected loss; as quantization noise R shrinks (more bits), gains fade.

## 4. Experiments (Falsification Plan)

Hypotheses:
- H1: At matched or higher throughput vs. quantized baseline, KLS reduces perplexity, with larger gains at lower bit widths.
> Note: Tests that quality improves without slowing down, especially when quantization is aggressive.

- H2: KLS narrows the gap to FP16 baselines on perplexity and downstream tasks.
> Note: Checks whether denoising recovers accuracy lost to quantization.

- H3: Gains disappear or reverse when K is distorted/shuffled or γ̂_1 mismatch rises.
> Note: Sanity checks: breaking the learned gains or violating model assumptions should remove benefits.

- H4: A tuned EMA baseline cannot match KLS across models/bits at similar overhead.
> Note: Ensures improvements are not explainable by a well-tuned simple EMA.

Models/quantization:
- Pythia-1.4B/2.8B, Mistral-7B (base/instruct).
> Note: Diverse model sizes and families to test generality.

- 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).
> Note: Different quantization schemes and bit-widths to probe noise regimes.

Data/tasks:
- Perplexity: WikiText-103, C4 validation.
> Note: Standard language modeling benchmarks for perplexity.

- Tasks: ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).
> Note: Downstream reasoning and commonsense benchmarks to see if improvements carry over.

Conditions:
1) FP16 baseline.
> Note: Upper-bound accuracy without quantization noise.

2) Quantized baseline.
> Note: Performance after low-bit quantization, without KLS.

3) Quantized + KLS (Top-M, M∈{512,1024}).
> Note: KLS applied with practical active set sizes to measure quality/speed trade-offs.

Baselines/ablations:
- EMA: Exponential moving average of logits with α grid-searched (global and per-band).
> Note: Strong baseline matching the form of KLS but without model-based gains.

- Banding: B ∈ {1, 8, 16, V}; magnitude-conditioned vs. frequency-based.
> Note: Tests how grouping tokens affects stability and performance.

- Calibration distortion: scale Q̂, R̂ by {0.25, 0.5, 2, 4}; shuffle K across bands.
> Note: Probes sensitivity to calibration accuracy; shuffling should harm if KLS is meaningful.

- Pipeline order: penalties → KLS → temperature vs. penalties → temperature → KLS.
> Note: Confirms recommended placement and whether ordering matters.

- Nucleus consistency: standard T_y-only vs. candidate union vs. consistent nucleus on T.
> Note: Tests candidate-set logic to avoid missing smoothed promotions.

- Online adaptation: static K vs. EW-updated moments (β ∈ {0.99, 0.995, 0.999}).
> Note: Evaluates benefits/risks of adapting gains during inference.

- Active set: M ∈ {128, 512, 1024, 2048}; L ∈ {32, 128, 256}.
> Note: Measures quality/latency tradeoffs versus the number of tracked tokens and extra candidates.

- Diversity: entropy of sampled distributions, distinct-n, repetition rates.
> Note: Checks that denoising doesn’t reduce generation diversity or increase repetition.

Latency/throughput:
- Tokens/sec and p95 latency. Report overhead breakdown: O(V) scan (shared) vs. O(M) KLS update; GPU kernel fusion impact; memory traffic (reads/writes per step).
> Note: Ensures speed targets are met and quantifies where time is spent.

Diagnostics:
- Fraction of bands with γ̂_1 ≥ 0; correlation with Δperplexity.
> Note: Validates the model-mismatch indicator and its relation to gains.

- Stability of K_b and sensitivity to banding.
> Note: Checks robustness of learned gains across grouping choices.

## 5. Implementation Notes
- HF integration: Provide a LogitsProcessor that applies KLS after penalties and temperature, before sampling; supports greedy/top-k/top-p/beam. Deterministic under fixed seeds.
> Note: Plug-and-play integration point in Hugging Face; preserves reproducibility.

- GPU kernels: Fuse top-k scan, active-set update, and candidate scoring; keep state (ŝ, K_b) in device memory; prefer fp16 storage with fp32 compute. Avoid host-device sync.
> Note: Practical steps to minimize latency and memory traffic.

- Batching/beam: Maintain per-sequence (and per-beam) states; reset at boundaries; handle variable-length batches.
> Note: Correct state management across multiple simultaneous sequences/beams.

- Numerical: Use float64 for K computation; clamp Q,R,K; robustify moments with Huber loss; set W warm-up steps.
> Note: Prevents numerical issues and early-step instability.

## 6. Related Work
- Quantized LLM inference: GPTQ, AWQ, NF4 and efficient kernels focus on weight/activation quantization. KLS is orthogonal as a post-hoc logit denoiser.
> Note: KLS complements, not replaces, quantization methods by addressing output noise.

- Temporal smoothing: EMA and heuristic smoothing have been used for stabilization; KLS provides a principled, identifiable α via state-space modeling.
> Note: Unlike ad-hoc EMA, KLS derives its smoothing weight from estimated noise/process variances.

- Output calibration: Temperature scaling and repetition penalties alter decoding policies; KLS targets noise reduction prior to sampling, not policy bias.
> Note: KLS aims to recover the model’s intended scores, leaving policy choices intact.

## 7. Limitations
- Diagonal filtering ignores cross-token correlations; multivariate filters are costlier.
> Note: Some tokens co-vary (e.g., synonyms). Modeling that could help but would increase runtime.

- Method-of-moments assumes additive, near-white observation noise; deterministic quantizers can induce bias and temporal correlation; banding and robustification mitigate but do not eliminate mismatch.
> Note: If quantization errors are structured (not like random noise), KLS may help less.

- Gains diminish at higher bit-widths (small R).
> Note: As quantization noise decreases, there’s less to denoise.

- Approximate nucleus with candidate union may deviate slightly from full-ŝ nucleus; a consistent variant is provided but remains approximate unless O(V) is used.
> Note: Exact consistency would require rescoring all tokens, which is more expensive.

## 8. Conclusion
KLS is a practical, theoretically grounded, calibration-only method that improves decoding quality under quantization with minimal overhead. It integrates cleanly with standard decoding, includes diagnostics for applicability, and is readily validated with open models and code.
> Note: Summary: clean logits post-quantization with a small, learned EMA-like filter; improves perplexity with little speed cost and is easy to adopt and test.

## Appendix A: Derivation of MoM Identities
With s_t = s_{t−1} + w_t, y_t = s_t + v_t, define d_t = y_t − y_{t−1} = w_t + v_t − v_{t−1}. Then
- Var(d_t) = Var(w_t) + Var(v_t) + Var(v_{t−1}) = Q + 2R.
> Note: Uses independence and zero-mean of w and v; variances add.

- Cov(d_t, d_{t−1}) = Cov(w_t + v_t − v_{t−1}, w_{t−1} + v_{t−1} − v_{t−2}) = −Var(v_{t−1}) = −R,
assuming independence across time and between w and v. Higher-lag covariances vanish.
> Note: The only shared term between d_t and d_{t−1} is v_{t−1} with opposite signs, giving −R; other cross-terms vanish under independence.

## Reproducibility Checklist
- Code (CUDA/PyTorch):
  - Streaming calibration with robustification, sequence-boundary resets, banding, optional affine preconditioning, and online adaptation.
> Note: Ensures others can recover KLS parameters and behavior from scratch.

  - Fused kernel for top-k scan, active-set update, and candidate scoring; correct greedy/candidate logic.
> Note: Critical for achieving the claimed O(M) overhead.

  - HuggingFace LogitsProcessor integration (greedy, top-k, top-p, beam), deterministic under seeds.
> Note: Plug-in module for easy reproduction in common tooling.

  - Benchmarks: tokens/sec, p95 latency, memory traffic.
> Note: Reports both speed and resource use transparently.

- Data:
  - Release learned (Q, R, K) per model/quantization; per-band γ̂_0, γ̂_1, mismatch diagnostics.
> Note: Sharing calibration artifacts allows verification and reuse.

  - Scripts for all baselines/ablations, including EMA and pipeline-order studies.
> Note: Full experiment scripts enable exact replication and robustness checks.
