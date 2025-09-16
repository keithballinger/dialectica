Minor Revisions

The manuscript presents a novel and promising method for denoising quantized LLM logits. The theoretical framing is sound, and the proposed experimental plan is rigorous and well-designed. The primary weakness lies in the "Method" section, specifically the "Quantization-only calibration" procedure. The description for estimating the observation noise variance `R` is too vague. The revised draft must provide a concrete, reproducible algorithm for this estimation, as it is central to the paper's claim of practicality. Minor notational inconsistencies should also be resolved, and the text can be sharpened for greater conciseness.

---
### Revised Draft

# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

## Abstract

Low-precision quantization accelerates large language model (LLM) inference but injects noise into token logits, degrading perplexity and downstream accuracy. We model successive pre-softmax logits as a noisy time series and introduce Kalman-Logit Smoothing (KLS): a lightweight, online Kalman filter that denoises quantized logits at each decoding step. KLS requires no architectural changes, adds negligible latency, and can be calibrated on a small set of token outputs. We derive a steady-state diagonal filter with closed-form gains, propose a calibration procedure that uses only quantized model outputs, and present efficient full-vocabulary and top-k variants. A falsification plan with open models (Pythia-1.4B/2.8B, Mistral-7B), standard datasets (WikiText-103, C4), and common 4–8 bit schemes (NF4, GPTQ, k-quant) will test for improvements in perplexity and task accuracy at fixed latency. Ablations of filter gains and model mis-specification will probe when smoothing helps or hurts. This work posits that time-series denoising is an underexplored, orthogonal axis to quantization and sampling for robust, low-precision LLM inference.

## Introduction

**Motivation:** Low-precision quantization (e.g., 4–8 bits) is essential for reducing the memory footprint and latency of LLM inference. However, it perturbs activations and pre-softmax logits, increasing perplexity and degrading task accuracy. Existing mitigation strategies focus on improving quantization schemes or compensating for errors within the model's forward pass. We ask: can we denoise the output logits directly, online, with near-zero overhead?

**Key Idea:** We treat the sequence of logit vectors from a quantized model as a noisy time series. At each step `t`, the observed quantized logits `y_t` are a noisy measurement of latent, "true" logits `s_t`. We model the evolution of `s_t` with a simple state-space model and apply a per-dimension Kalman filter to compute a denoised estimate, `ŝ_t`, which is then used for decoding. If quantization noise is approximately zero-mean and independent across vocabulary dimensions, the Kalman filter is the Minimum Mean Squared Error (MMSE) optimal linear estimator for `s_t`.

**Contributions:**

1.  A state-space model for logit evolution in quantized LLMs and an online, diagonal Kalman filter that adds only two vector operations per token.
2.  A closed-form steady-state filter gain that eliminates per-step covariance updates, and a practical calibration procedure that estimates noise parameters from quantized model outputs alone.
3.  Efficient full-vocabulary and top-k implementations compatible with standard sampling methods.
4.  A detailed falsification plan to test the method's efficacy on open models and datasets under strict latency controls, including ablations to probe its sensitivity and failure modes.

## Method

### Problem Framing

Let `s_t ∈ R^V` be the latent, unquantized pre-softmax logits at time step `t` for a vocabulary of size `V`. Quantized inference produces observed logits `y_t ∈ R^V`, which we model as:

*   **Observation Model:** `y_t = s_t + v_t`, where `v_t ~ N(0, R)` is i.i.d. quantization noise.
*   **State-Transition Model:** `s_t = a s_{t-1} + w_t`, where `w_t ~ N(0, Q)` is i.i.d. process noise.

We assume the noise covariance matrices `R` and `Q` are diagonal, allowing for `V` independent scalar Kalman filters. For simplicity and robustness, we set the autoregressive parameter `a=1`, modeling the latent logits as a random walk. This reduces the filter to a form of adaptive exponential smoothing.

### Steady-State Kalman Filter

The standard Kalman filter recursions for each vocabulary dimension `i` simplify under a steady-state gain `K_i`, avoiding online variance updates. The update rule becomes a single exponential smoothing step:

`ŝ_{t,i} = ŝ_{t-1,i} + K_i (y_{t,i} - ŝ_{t-1,i})`

The steady-state gain `K` is a diagonal matrix with entries `K_i` determined by the ratio of process noise `Q_i` to observation noise `R_i`. For `a=1`, `K_i` is the positive root of `R_i K_i^2 - Q_i (1-K_i) = 0`, given by:

`K_i = (sqrt(Q_i^2 + 4 Q_i R_i) - Q_i) / (2 R_i)`

*   If `R_i ≫ Q_i` (high observation noise), `K_i → 1`, heavily weighting the new observation `y_{t,i}`.
*   If `Q_i ≫ R_i` (high process noise), `K_i → 0`, heavily weighting the prior estimate `ŝ_{t-1,i}`.

The filter state is initialized with the first observation: `ŝ_0 = y_0`.

### Calibration from Quantized Observations

To compute the steady-state gain `K`, we must estimate the diagonal variances `Q` and `R` using only a small calibration set of quantized outputs `y_t`.

1.  **Estimate Observation Noise `R`**: We assume `s_t` evolves smoothly, implying that high-frequency variations in `y_t` are dominated by quantization noise `v_t`. We can thus estimate `R_i` as the variance of the residual after light exponential smoothing:
    `R_i ≈ Var(y_{t,i} - EMA_α(y_{t,i}))`
    where `EMA_α` is an exponential moving average with a small smoothing factor (e.g., `α=0.1`), computed over a calibration stream of a few thousand tokens.

2.  **Estimate Process Noise `Q`**: With `R` estimated and `a=1`, we use the variance of the first-difference of the observations. Since `y_t - y_{t-1} = (s_t - s_{t-1}) + (v_t - v_{t-1}) = w_t + v_t - v_{t-1}`, and noise terms are independent, their variances add:
    `Var(y_t - y_{t-1}) = Var(w_t) + Var(v_t) + Var(v_{t-1}) = Q + 2R`
    We can then solve for `Q_i`:
    `Q_i = max(0, Var(y_{t,i} - y_{t-1,i}) - 2 R_i)`

For practicality, gains `K_i` can be grouped into 8-16 bands based on mean logit magnitude to reduce estimation noise and memory.

### Implementation Variants

*   **Full-Vocabulary:** Maintains a state vector `ŝ_t ∈ R^V`. Per step, this requires two vector operations (one subtraction, one fused multiply-add), adding sub-millisecond latency on modern hardware.
*   **Top-k Variant:** To guarantee negligible overhead, the filter state `ŝ_t` is maintained only for a small, dynamic set of `M` token indices (e.g., `M=1024`), corresponding to the union of top-k indices in `y_t` and `y_{t-1}`. Logits for other tokens pass through unchanged. This reduces per-step complexity to `O(M)`.
*   **Post-filtering Rescaling:** Optionally, the variance of the smoothed logits `ŝ_t` can be rescaled to match the variance of the input `y_t`. This prevents the filter from implicitly altering the softmax temperature.

### Pseudocode (Vectorized, Steady-State)

```python
# Pre-computation
# 1. Collect y_cal = [y_0, y_1, ...] on a calibration set.
# 2. Estimate R, Q from y_cal.
# 3. Compute steady-state gain K.
s_hat = y_0 # Initialize state

# Per-step inference loop
for t in 1...T:
  y_t = model_logits(inputs)
  residual = y_t - s_hat
  s_hat += K * residual # Element-wise update
  # Use s_hat for sampling, argmax, etc.
```

## Experiments (Falsification Plan)

### Hypotheses

*   **H1:** KLS reduces perplexity compared to standard quantized inference at matched wall-clock latency, with larger gains for lower-precision (4–5 bit) models.
*   **H2:** KLS narrows the perplexity and task-accuracy gap between quantized and full-precision (FP16) inference without modifying the base model.
*   **H3:** Performance degrades towards or below the quantized baseline if gains `K` are mis-specified (e.g., inverted), falsifying trivial explanations like simple temperature scaling.

### Setup

*   **Models:** Pythia-1.4B/2.8B, Mistral-7B.
*   **Quantization Schemes:** 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).
*   **Datasets & Tasks:**
    *   **Perplexity:** WikiText-103, C4 (validation sets).
    *   **Downstream Accuracy:** ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).
*   **Conditions:**
    1.  **FP16 Baseline:** Full-precision, no quantization.
    2.  **Quantized Baseline:** Standard quantized inference.
    3.  **Quantized + KLS:** Our method, using quantization-only calibration.
*   **Latency Control:** End-to-end token generation throughput (tok/s) will be measured. The KLS condition must demonstrate improvement at throughput equal to or greater than the quantized baseline.

### Ablations

*   **Gain Sensitivity:** Evaluate perplexity while sweeping a shared, scalar gain `K` from 0 to 1.
*   **Calibration Mis-specification:** Artificially inflate/deflate estimated `R` by {0.25x, 4x} and measure performance impact.
*   **Top-k Size:** Vary `M` in the top-k variant from 128 to 2048 to find the performance/cost trade-off point.
*   **Rescaling:** Compare performance with and without post-filter variance matching.

## Discussion

KLS leverages the temporal smoothness of logit distributions during autoregressive decoding. While the top-1 token can change abruptly, the broader distribution often shifts gradually, creating a high signal-to-noise ratio that filtering can exploit. Quantization introduces noise `v_t` that is largely uncorrelated step-to-step, while the underlying signal `s_t` is highly correlated.

KLS is orthogonal to and compatible with improved quantization methods. It is a post-processing step that does not alter model weights or the core decoding algorithm. Its primary failure mode occurs when logits change unpredictably (low signal-to-noise), such as at sharp topic boundaries, where the random walk model is violated.

## Limitations

*   The diagonal covariance assumption ignores cross-token correlations in logit evolution and quantization noise.
*   Calibration is performed on a general-purpose text corpus; performance may degrade on out-of-distribution prompts.
*   Benefits are expected to diminish for higher-bit quantization (e.g., 8-bit), where the observation noise `R` is small relative to the natural process noise `Q`.

## Conclusion

Kalman-Logit Smoothing is a lightweight, theoretically-grounded method for mitigating quantization noise in LLM inference. By treating logits as a time series, it applies a near-zero-cost denoising filter at each decoding step. The proposed experiments are designed to rigorously test whether this approach provides meaningful improvements in perplexity and task accuracy under real-world latency constraints, establishing state-space denoising as a viable tool for robust low-precision inference.
