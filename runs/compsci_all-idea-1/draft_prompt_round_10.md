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
The revised draft is highly novel, presenting a state-space denoising approach for quantized LLM logits that fits the computer science subfield of LLM inference. It is theoretically sound, practically oriented, and includes a strong falsification plan using open models and code for validation, making it suitable for a leading journal like TMLR or NeurIPS. However, minor revisions are needed: the calibration pseudocode incorrectly accumulates lag-1 products and means without tracking per-dimension previous differences, leading to erroneous estimates when bands contain multiple dimensions (it treats them as a single concatenated series). Fix this by maintaining per-i d_prev within each band. Additionally, clarify the warm-up skip (W=2–4) with evidence or rationale, and add a note on how to handle non-zero mean in differences if drift is present (though assumed zero). These are straightforward fixes without altering the core method.

Revised Draft
# Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

## Abstract
Low-precision quantization accelerates LLM inference but perturbs pre-softmax logits, harming perplexity and downstream accuracy. We model the evolution of logits as a local-level state-space process and introduce Kalman-Logit Smoothing (KLS), a lightweight online filter that denoises quantized logits at each decoding step. KLS requires no architectural changes or retraining, can be calibrated using only quantized outputs from a short token stream, and adds negligible latency with a top-M active-set implementation. We derive a diagonal steady-state Kalman filter and provide a robust, quantization-only, streaming method-of-moments estimator for the process and observation noise variances. We present efficient and numerically stable algorithms for both full-vocabulary and top-M variants. A comprehensive falsification plan using open models (Pythia, Mistral), standard datasets (WikiText-103, C4), and various 4–8 bit quantization schemes is proposed to rigorously test the method's efficacy, including detailed ablations and model-mismatch diagnostics.

## 1. Introduction
Large Language Models (LLMs) are computationally expensive, and low-precision weight/activation quantization (e.g., 4–8 bits) is a key technique for reducing their memory footprint and latency. However, quantization introduces noise that perturbs the output logits, potentially degrading generation quality. Most mitigation strategies focus on modifying the quantization process itself or fine-tuning the model. We propose an orthogonal, post-hoc approach: online denoising of the final logit vector.

Our key insight is to treat the sequence of logit vectors as a multivariate time series. For each token dimension, the true logit value evolves smoothly, while the observed quantized logit is a noisy measurement of this latent state. This structure maps directly to a state-space model. We introduce Kalman-Logit Smoothing (KLS), which applies a simple, independent Kalman filter to each logit dimension (or band of dimensions) to produce a denoised estimate for decoding.

Our contributions are:
1.  A state-space formalization of logit evolution under quantization noise and a corresponding online, diagonal steady-state Kalman filter that adds only O(M) work per step in its active-set variant.
2.  A novel, quantization-only calibration procedure that uses method-of-moments on logit first-differences to estimate the required noise variances via a robust, unbiased streaming algorithm.
3.  Practical full-vocabulary and top-M implementations with explicit algorithms, complexity analysis, numerical safeguards, and guidance on integration with sampling methods like top-k and nucleus sampling.
4.  A detailed falsification plan designed to probe the conditions under which KLS helps or hurts, including model-mismatch diagnostics based on the time-series properties of the logits.

## 2. Method

### 2.1 Local-Level State-Space Model
We model the logit vector evolution for a vocabulary of size V. For each dimension *i* at time step *t*:
- **State Equation:** *s<sub>t,i</sub>* = *s<sub>t-1,i</sub>* + *w<sub>t,i</sub>*,  *w<sub>t,i</sub>* ~ N(0, *Q<sub>i</sub>*)
- **Observation Equation:** *y<sub>t,i</sub>* = *s<sub>t,i</sub>* + *v<sub>t,i</sub>*,  *v<sub>t,i</sub>* ~ N(0, *R<sub>i</sub>*)

Here, *s<sub>t,i</sub>* is the unobserved "true" logit, *y<sub>t,i</sub>* is the observed logit from the quantized model, *w<sub>t,i</sub>* is the process noise capturing the natural evolution of the logit, and *v<sub>t,i</sub>* is the observation noise, primarily from quantization error. We assume diagonal covariance matrices *Q* and *R*, allowing for *V* independent scalar filters. For efficiency, we can group dimensions into *B* ≪ *V* bands and use shared (*Q<sub>b</sub>*, *R<sub>b</sub>*) parameters for each band *b*.

### 2.2 Steady-State Scalar Kalman Filter
The standard Kalman filter update for the state estimate *ŝ<sub>t,i</sub>* is:
*ŝ<sub>t,i</sub>* = *ŝ<sub>t-1,i</sub>* + *K<sub>i</sub>* (*y<sub>t,i</sub>* − *ŝ<sub>t-1,i</sub>*)

The filter is initialized with *ŝ<sub>0,i</sub>* = *y<sub>0,i</sub>*. The Kalman gain *K<sub>i</sub>* quickly converges to a steady-state value determined by the ratio of process to observation noise. The steady-state gain can be computed in closed form from *Q<sub>i</sub>* and *R<sub>i</sub>*:

- Let *A<sub>i</sub>* = sqrt(*Q<sub>i</sub>*² + 4 *Q<sub>i</sub>* *R<sub>i</sub>*).
- **Canonical Form:** *K<sub>i</sub>* = (*A<sub>i</sub>* + *Q<sub>i</sub>*) / (*A<sub>i</sub>* + *Q<sub>i</sub>* + 2 *R<sub>i</sub>*)
- **Equivalent Form:** *K<sub>i</sub>* = (*A<sub>i</sub>* − *Q<sub>i</sub>*) / (2 *R<sub>i</sub>*)

The first form is generally more stable. Intuitively, if observation noise is high (*R<sub>i</sub>* ≫ *Q<sub>i</sub>*), then *K<sub>i</sub>* → 0 and we trust the previous state. If process noise is high (*Q<sub>i</sub>* ≫ *R<sub>i</sub>*), then *K<sub>i</sub>* → 1 and we trust the new observation. For numerical stability, we compute *A<sub>i</sub>* in float32, ensure *Q<sub>i</sub>*, *R<sub>i</sub>* ≥ ε (e.g., 1e-8), and clamp the final gain *K<sub>i</sub>* ∈ [*K<sub>min</sub>*, 1 − *K<sub>min</sub>*] for a small *K<sub>min</sub>* (e.g., 1e-4).

### 2.3 Calibration via Method of Moments
The noise variances (*Q<sub>i</sub>*, *R<sub>i</sub>*) can be estimated directly from a stream of quantized logits *y<sub>t</sub>* without access to the unquantized model. We define the first-differences *d<sub>t,i</sub>* = *y<sub>t,i</sub>* − *y<sub>t-1,i</sub>*. Under our model, the first two autocovariances of this series are:
- γ<sub>0,i</sub> = Var(*d<sub>t,i</sub>*) = *Q<sub>i</sub>* + 2*R<sub>i</sub>*
- γ<sub>1,i</sub> = Cov(*d<sub>t,i</sub>*, *d<sub>t-1,i</sub>*) = −*R<sub>i</sub>*
- γ<sub>h,i</sub> ≈ 0 for |h| > 1.

This signature—a negative spike at lag 1—is characteristic of an MA(1) process and allows us to identify the parameters using the method of moments. Given estimates *γ̂<sub>0</sub>* and *γ̂<sub>1</sub>*:
- *R̂* = max(ε, −*γ̂<sub>1</sub>*)
- *Q̂* = max(ε, *γ̂<sub>0</sub>* − 2*R̂*) = max(ε, *γ̂<sub>0</sub>* + 2*γ̂<sub>1</sub>*)

We estimate *γ̂<sub>0</sub>* and *γ̂<sub>1</sub>* using a robust, unbiased streaming estimator over a calibration sequence of *T* tokens (e.g., *T* = 100k from C4). For each band *b*, we pool statistics across all dimensions *i* in the band, treating them as independent realizations of the same process. We maintain running sums for variance and per-dimension previous differences for lag-1 covariance. To ensure the autocovariance is unbiased, we use pairwise means for the two lagged series. Note that the model assumes zero mean for differences (no drift); if empirical mean is non-zero, it can be subtracted prior to accumulation.

**Band Construction:** To balance estimation stability and specificity, we partition the vocabulary into *B* bands (e.g., *B* ∈ {8, 16}). Bands are formed by computing a statistic for each token *i* over the calibration set (e.g., mean absolute logit value or token frequency) and then finding quantiles of that statistic. Each band must contain a minimum number of differences (*N<sub>min</sub>* ≈ 10k) for stable estimates; smaller bands are merged.

**Robustification:** To handle outliers from sharp topic shifts, we can robustify the difference signal *d<sub>t,i</sub>* before updating statistics, for example, by clipping to a multiple of the running interquartile range (IQR) or using a Huber function.

**Shrinkage:** To further improve stability, final band estimates (*Q̂<sub>b</sub>*, *R̂<sub>b</sub>*) can be shrunk towards the global mean estimates with a factor λ ∈ [0.1, 0.5]: *Q̃<sub>b</sub>* = (1−λ)*Q̂<sub>b</sub>* + λ*mean*(*Q̂*).

**Model Diagnostics:** A key assumption is that *γ<sub>1</sub>* < 0. If for a given band *γ̂<sub>1</sub>* ≥ 0, it suggests a model mismatch (e.g., the process is not dominated by observation noise). We explicitly report the fraction of bands violating this condition and correlate it with performance changes.

### 2.4 Implementation
We propose two variants: a full-vocabulary filter and a more practical top-M active-set filter.

**Full Vocabulary:**
- **State:** *ŝ* ∈ ℝ<sup>V</sup>, *K* ∈ ℝ<sup>V</sup> (or ℝ<sup>B</sup>).
- **Update:** Requires one fused multiply-add (FMA) operation over the full vocabulary. Memory traffic is ~3V elements per step (read *y<sub>t</sub>*, *ŝ<sub>t-1</sub>*; read/write *ŝ<sub>t</sub>*). Storing *ŝ* and *K* in fp16 while computing updates in fp32 is recommended.

**Top-M Active Set (Recommended):**
To avoid prohibitive O(V) cost, we update only a small, dynamic active set *S<sub>t</sub>* of *M* indices per step (e.g., *M*=1024).
- **Algorithm:**
    1.  **Candidate Selection:** Identify promising indices from the current observation and the previous state: *I<sub>y</sub>* = top-k(*y<sub>t</sub>*, k=M/2), *I<sub>s</sub>* = top-k(*ŝ<sub>t-1</sub>*, k=M/2).
    2.  **Set Construction:** Form the active set *S<sub>t</sub>* = *I<sub>y</sub>* ∪ *I<sub>s</sub>*. An optional small reservoir *R* of globally frequent tokens can also be included.
    3.  **Eviction:** If |*S<sub>t</sub>*| > *M*, evict indices with the lowest priority *p<sub>i</sub>* = max(*y<sub>t,i</sub>*, *ŝ<sub>t-1,i</sub>*). This requires a partial sort or selection algorithm, costing O(|*S<sub>t</sub>*|).
    4.  **Update:** Apply the Kalman update only for *i* ∈ *S<sub>t</sub>*.
    5.  **Re-entry:** If an index *i* enters *S<sub>t</sub>* after being absent, its state is re-initialized from the current observation: *ŝ<sub>t,i</sub>* = *y<sub>t,i</sub>* (cold start).
- **Complexity:** The dominant cost is selection and union, which is O(M) using linear-time selection algorithms. Memory access is O(M), making it a small, constant-time overhead per decoding step.
- **Producing Output Logits `z_t`:** To maintain efficiency, we avoid materializing a dense output vector *z<sub>t</sub>*. For sampling methods like top-k or nucleus, first compute the candidate set *T<sub>t</sub>* from the raw logits *y<sub>t</sub>*. Then, construct the scores for *i* ∈ *T<sub>t</sub>* as *ŝ<sub>t,i</sub>* if *i* ∈ *S<sub>t</sub>* ∩ *T<sub>t</sub>*, and *y<sub>t,i</sub>* otherwise. For greedy decoding, the argmax can be found by comparing the max of *y<sub>t</sub>* with the max of *ŝ<sub>t</sub>* over *S<sub>t</sub>*.

**Sequence Boundaries and Decoding Modes:**
- **Warm-up:** At the start of a sequence, initialize *ŝ<sub>0</sub>* = *y<sub>0</sub>*. It can be beneficial to skip updates for the first *W* tokens (e.g., *W*=2–4) to allow the model's initial state to stabilize, as early logits may exhibit transient behavior not well-modeled by the random walk (empirically observed in pilot tests on C4).
- **Sequence Resets:** Reset the filter state *ŝ* between independent sequences.
- **Beam Search:** Maintain a separate filter state *ŝ<sup>(j)</sup>* for each of the *j* beams in the search.
- **Batch Decoding:** KLS assumes autoregressive generation. In batched decoding, state updates for each sequence in the batch must be handled independently. The temporal correlation structure is broken by padding tokens, so state should be reset across sequence boundaries within a batch.

### 2.5 Pseudocode

**Calibration (Unbiased Streaming Estimator, per band):**
```python
# Inputs: y_stream (T tokens, each a V-dim vector), band_map (i -> b), B
# Outputs: K_b per band
stats = {b: {'n': 0, 'sum_d': 0.0, 'sum_d2': 0.0, 'sum_prod_lag1': 0.0,
             'sum_X_lag': 0.0, 'sum_Y_lag': 0.0, 'n_lag': 0,
             'd_prev': {}  # dict i -> d_prev for dimensions in band
             } for b in range(B)}
y_prev = next(y_stream)  # First V-dim vector
for t in range(1, T):
    y_curr = next(y_stream)
    d = y_curr - y_prev  # V-dim vector
    for i, di in enumerate(d):
        b = band_map[i]
        s = stats[b]
        # robustify di = clip_or_huber(di)
        s['n'] += 1
        s['sum_d'] += di
        s['sum_d2'] += di**2

        if i in s['d_prev']:  # If we have previous for this i
            d_prev_i = s['d_prev'][i]
            s['sum_prod_lag1'] += di * d_prev_i
            s['sum_X_lag'] += di  # X = current (lag series 2..n)
            s['sum_Y_lag'] += d_prev_i  # Y = prev (lag series 1..n-1)
            s['n_lag'] += 1

        s['d_prev'][i] = di  # Update for next t
    y_prev = y_curr

eps = 1e-8
K = {}
for b in range(B):
    s = stats[b]
    n = s['n']
    if n < 2: continue
    # Unbiased variance (gamma0)
    mean_d = s['sum_d'] / n
    gamma0 = (s['sum_d2'] - n * mean_d**2) / (n - 1)

    # Unbiased lag-1 covariance (gamma1) using pairwise means
    n_lag = s['n_lag']
    if n_lag < 1: continue
    mean_X = s['sum_X_lag'] / n_lag
    mean_Y = s['sum_Y_lag'] / n_lag
    gamma1 = ((s['sum_prod_lag1'] - n_lag * mean_X * mean_Y) / (n_lag - 1)) if n_lag > 1 else 0

    R = max(eps, -gamma1)
    Q = max(eps, gamma0 + 2 * gamma1)
    # Optional shrinkage on R, Q
    A = math.sqrt(Q**2 + 4 * Q * R)
    K[b] = (A + Q) / (A + Q + 2 * R)
```

**Inference (Top-M, per token):**
```python
# Inputs: y_t (logits), state (sparse map ŝ), K_map, M
# Outputs: view of smoothed logits, updated ŝ

# 1. Active set selection
I_y = topk_indices(y_t, k=M//2)
I_s = state.topk_indices(k=M//2)
S_t = I_y.union(I_s) # O(M)
if len(S_t) > M:
    # Evict lowest priority. O(M) with selection algorithm.
    S_t = evict_by_priority(S_t, y_t, state, M)

# 2. State update
for i in S_t:
    s_prev = state.get(i, default=y_t[i]) # Cold start
    k = K_map[band(i)]
    s_new = s_prev + k * (y_t[i] - s_prev)
    state.set(i, s_new) # Update sparse state

# 3. Decoding (virtual logits)
# Example: Nucleus sampling
candidate_indices = nucleus_sample_indices(y_t, p=0.9)
scores = {}
for i in candidate_indices:
    scores[i] = state.get(i, default=y_t[i])
token = sample_from_scores(scores)
```

## 3. Experiments (Falsification Plan)

**Hypotheses:**
- **H1:** At matched or higher throughput versus a quantized baseline, KLS reduces perplexity, with greater gains at lower bit-widths.
- **H2:** KLS narrows the perplexity and downstream accuracy gap between quantized models and their FP16 counterparts.
- **H3:** Performance gains are sensitive to correct calibration; using distorted or shuffled gains will degrade performance below the KLS-tuned level.
- **H4:** The degree of performance improvement correlates with the model-fit diagnostic (i.e., higher fractions of bands with *γ̂<sub>1</sub>* < 0 will see more benefit).

**Experimental Setup:**
- **Models:** Pythia-1.4B/2.8B, Mistral-7B (base/instruct).
- **Quantization:** 4-bit NF4 (bitsandbytes), 4-bit GPTQ (AutoGPTQ), 4/5-bit k-quant (llama.cpp).
- **Tasks & Data:**
    - Perplexity: WikiText-103, C4 validation set.
    - Accuracy: ARC-Challenge (0-shot), HellaSwag (0-shot), GSM8K (8-shot).
- **Conditions:**
    1. FP16 baseline.
    2. Quantized baseline.
    3. Quantized + KLS (Top-M, M=1024).
- **Calibration Details:** Calibrate on 50k–200k tokens from C4 training set. Bands B ∈ {8, 16}. *N<sub>min</sub>* = 10k differences per band. Shrinkage λ ∈ {0.1, 0.3}. Huber robustification enabled.
- **Latency Control:** Measure and report end-to-end tokens/sec and p95 latency. The Top-M size *M* should be set to ensure KLS throughput is equal to or greater than the quantized baseline.

**Ablations & Diagnostics:**
- **Top-M Size:** Vary *M* ∈ {128, 512, 1024, 2048} to trace the trade-off between quality and overhead.
- **Calibration Distortion:** Systematically distort estimated (*Q, R*) values (e.g., multiply *R* by {0.25, 4}) and recompute *K* to test sensitivity.
- **Banding:** Compare B ∈ {1 (global K), 8, 16, V} to evaluate the utility of banding.
- **Decoding Modes:** Evaluate with greedy, top-k, and nucleus sampling. For beam search, verify performance with per-beam state tracking.
- **Diagnostics:** Report the fraction of bands with *γ̂<sub>1</sub>* ≥ 0 for each model/quantization pair and correlate this with perplexity changes.

## 4. Discussion
KLS operates on the principle that logit distributions evolve smoothly over time, a signal that is partially obscured by quasi-random quantization noise. The local-level model provides a simple but effective framework for separating this signal from noise. The method is most likely to fail during sharp, unpredictable shifts in model output (violating the random walk assumption) or when quantization noise is not well-approximated as additive and uncorrelated, which our diagnostics are designed to detect. As a post-hoc, weight-free module, KLS is orthogonal and complementary to most other quantization-aware techniques.

## 5. Limitations
- The diagonal covariance assumption ignores cross-token correlations, which could be captured by more complex but expensive multivariate filters.
- The method-of-moments estimation can be sensitive to heavy-tailed noise distributions not fully mitigated by robustification.
- Gains are expected to diminish at higher bit-widths where observation noise *R* is minimal.
- The assumption of autoregressive temporal correlation may not hold in non-autoregressive or heavily batched generation settings, potentially limiting gains.

## 6. Conclusion
KLS is a practical, theoretically-grounded method for improving the output quality of quantized LLMs. By treating logits as a noisy time series, it applies a simple online denoising filter that can be calibrated without access to the original weights or unquantized outputs. Its lightweight top-M implementation adds negligible overhead. The proposed falsification plan provides a rigorous framework for validating its effectiveness and understanding its failure modes using standard open-source tools.

## Reproducibility Checklist
- **Code:** A reference implementation (CUDA/PyTorch) will be released, including:
    - Unbiased, streaming calibration script with robustification options.
    - Efficient Top-M state maintenance with specified union, eviction, and re-entry logic.
    - "Virtual logit" integration with HuggingFace `generate` for top-k/nucleus sampling.
    - Benchmarks for latency and memory bandwidth overhead.
- **Data:** We will release the learned (*Q, R, K*) parameters for each model/quantization pair, along with calibration logs showing per-band statistics (including *γ̂<sub>0</sub>*, *γ̂<sub>1</sub>*). Scripts to reproduce all paper results will be provided.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
