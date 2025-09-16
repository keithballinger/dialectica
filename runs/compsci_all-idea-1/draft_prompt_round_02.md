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
        Major Revisions

Title
Kalman-Logit Smoothing: Online State-Space Denoising for Quantized LLM Inference

Abstract
Low-precision quantization accelerates large language model (LLM) inference but injects noise into token logits, degrading perplexity and downstream accuracy. We model successive pre-softmax logits as a noisy time series and introduce Kalman-Logit Smoothing (KLS): a lightweight, online Kalman filter that denoises quantized logits at each decoding step. KLS requires no architectural changes, adds negligible latency, and can be calibrated on a small validation set. We derive a steady-state diagonal filter with closed-form gains, propose quantization-only calibration rules that avoid full-precision runs, and present efficient full-vocabulary and top-k variants. A falsification plan with open models (Pythia-1.4B/2.8B, Mistral-7B), standard datasets (WikiText-103, C4), and common 4–8 bit schemes (NF4, GPTQ, k-quant) tests improvements in perplexity and task accuracy at fixed latency. Ablations of filter gains and mis-specification probe when smoothing helps or hurts. This work posits that time-series denoising is an underexplored, orthogonal axis to quantization and sampling for robust, low-precision LLM inference.

Introduction
- Motivation: Quantization (4–8 bit) reduces memory and latency for LLM inference but perturbs activations and pre-softmax logits, typically increasing perplexity and reducing task accuracy. Existing mitigation focuses on better quantizers, rounding strategies, or activation compensation. We ask: can we denoise the output logits directly, online, with near-zero overhead?
- Key idea: Treat the quantized logits y_t^q at step t as noisy observations of latent “true” logits s_t evolving across steps. Apply a per-dimension Kalman filter to produce denoised logits ŝ_t used for decoding. If quantization noise is approximately zero-mean and independent across vocabulary dimensions, the Kalman filter provides MMSE-optimal denoising under a simple state-space model.
- Contributions:
  1) A state-space view of logit evolution for quantized LLMs and an online, diagonal Kalman filter that adds only a few vector operations per token.
  2) Closed-form steady-state gains that avoid per-step covariance updates, plus quantization-only calibration rules that do not require full-precision references.
  3) Practical implementations: full-vocabulary and top-k variants; minimal memory footprint; compatibility with sampling and beam search.
  4) A falsification plan on open models/datasets quantized to 4–8 bits, with latency-controlled comparisons and ablations of filter gains and mis-specification.

Method
Problem framing
- Let s_t ∈ R^V be the unquantized pre-softmax logits at time t for vocabulary size V.
- Quantized inference yields y_t ∈ R^V, where y_t = s_t + v_t and v_t is quantization-induced noise.
- Assume a simple per-dimension AR(1) latent dynamics: s_t = a s_{t-1} + w_t, with process noise w_t and observation noise v_t modeled as zero-mean, independent, diagonal Gaussians: w_t ~ N(0, Q), v_t ~ N(0, R). We often use a = 1 (random walk), which makes filtering equivalent to adaptive exponential smoothing.

Online Kalman filter (diagonal, vectorized)
- For each dimension i (token id), the scalar Kalman recursions are:
  - Prior mean: ŝ_t|t-1 = a ŝ_{t-1}
  - Prior variance: P_t|t-1 = a^2 P_{t-1} + Q
  - Kalman gain: K_t = P_t|t-1 / (P_t|t-1 + R)
  - Posterior mean: ŝ_t = ŝ_t|t-1 + K_t (y_t - ŝ_t|t-1)
  - Posterior variance: P_t = (1 - K_t) P_t|t-1
- We use a steady-state gain to avoid updating P online.

Closed-form steady-state gain (a = 1)
- Define S = P + Q. At steady state, S solves S^2 - Q S - Q R = 0, so:
  - S = (Q + sqrt(Q^2 + 4 Q R)) / 2
  - K = S / (S + R)
- With this K, the filter reduces to ŝ_t = ŝ_{t-1} + K (y_t - ŝ_{t-1}).
- Special cases:
  - K → 1 when R ≫ Q (trust observation; weak smoothing).
  - K → 0 when Q ≫ R (trust prior; strong smoothing).
- We apply K element-wise (per token id). For simplicity, K can be shared across the vocabulary or grouped by token frequency bands.

Calibration: estimating Q and R
- If full-precision references are available:
  - R_i = Var(y_{t,i} - s_{t,i}) across validation steps.
  - Q_i = Var(s_{t,i} - s_{t-1,i}) across validation steps.
- Quantization-only calibration (no full-precision run):
  - Assume v_t independent over time and independent of s_t.
  - From the observation model: Var(Δy_i) = Var(Δs_i) + 2 R_i, where Δy_i = y_{t,i} - y_{t-1,i}.
  - Estimate R_i via short-run within-step roundoff statistics or via block-wise noise proxies exposed by quantization libraries (e.g., per-channel quantization error, activation scales). If unavailable, approximate R_i by the residual variance around a local EMA of y_t.
  - Then set Q_i = max(Var(Δy_i) - 2 R_i, ε), with ε small for numerical stability.
- Practical defaults:
  - Use a single shared K or 8–16 groupwise K’s over the vocab sorted by average logit magnitude; this reduces calibration noise and memory.
  - Clip K to [K_min, K_max], e.g., [0.05, 0.95].
  - Optionally rescale ŝ_t to match the pre-filter logit variance to avoid unintended temperature changes.

Top-k variant
- To reduce memory and guarantee negligible overhead, maintain filter state only for the union of top-M logits from {y_{t-1}, y_t} (e.g., M = 256–1024). Non-tracked logits pass through unchanged. This focuses denoising where it matters for decoding.

Complexity and memory
- Full-vocab diagonal K: per step, two vector add/mul over R^V and one state buffer ŝ_t ∈ R^V. For V ≈ 50k, this is sub-millisecond on modern CPUs/GPUs and typically hidden under sampling softmax cost.
- Top-k: O(M) operations and memory per step.
- No matrix multiplications, no extra attention/state reads; compatible with batch decoding.

Numerical stability and invariances
- Filtering operates pre-softmax, preserving normalization by the downstream softmax. A post-filter affine rescaling can recover intended temperature if needed.
- Initialize ŝ_0 = y_0 and warm up with 1–3 steps with larger K to avoid cold-start bias.

Pseudocode (steady-state, vectorized)
- Precompute per-dimension (or grouped) K from calibrated Q,R.
- For each decoding step t:
  1) y_t = model_logits(x_≤t; quantized_model)
  2) ŝ_t = ŝ_{t-1} + K ⊙ (y_t - ŝ_{t-1})  [element-wise]
  3) Use ŝ_t for sampling/argmax/beam scores.

Experiments (falsification plan)
Hypotheses
- H1: KLS reduces perplexity versus plain quantized inference at fixed latency, with larger gains for 4–5 bit settings.
- H2: KLS narrows the gap to full precision without modifying the model or decoding algorithm.
- H3: Mis-specified gains degrade performance toward or below the quantized baseline, falsifying trivial explanations.

Models and quantization
- Models: Pythia-1.4B, Pythia-2.8B (EleutherAI), Mistral-7B (Apache 2.0).
- Quantization:
  - 4-bit NF4 and 8-bit via bitsandbytes.
  - GPTQ 4–8 bit (AutoGPTQ).
  - llama.cpp k-quant (k-s, k-m) for Mistral-7B.
- Ensure same tokenizer and decoding settings across conditions.

Datasets and tasks
- Language modeling: WikiText-103, C4 validation shards. Metric: perplexity.
- Zero-/few-shot tasks (to test distribution-level effects):
  - ARC-Challenge zero-shot accuracy.
  - HellaSwag zero-shot accuracy.
  - GSM8K 8-shot exact match (cot-free, fixed prompts).
- Decoding: greedy and nucleus sampling (p=0.9); fixed temperature = 1.0 unless noted.

Conditions
- FP16 (non-quantized) baseline.
- Quantized baseline (no KLS).
- Quantized + KLS (full-vocab and top-k variants).
- Equal-latency control:
  - Measure end-to-end tok/s for each condition; ensure KLS adds ≤1–2% overhead or adjust the baseline’s sampling temperature/top-k to match wall-clock latency if needed.
- Calibration regimes:
  - Full-reference calibration (uses FP logs on a small held-out set).
  - Quantization-only calibration.

Ablations
- Gain sensitivity: sweep K ∈ {0.1, 0.2, …, 0.9} shared across vocab; report perplexity curves.
- Mis-specification:
  - Inflate/deflate R by ×{0.25, 0.5, 2, 4}.
  - Set a ∈ {0.8, 0.9, 1.0} with recomputed steady-state K; compare to a=1.
- Top-k size M ∈ {128, 256, 512, 1024}.
- Rescaling: with/without post-filter variance matching.
- Robustness: batch size, context length, and sampling temperature sweeps.

Reporting
- Perplexity deltas versus quantized baseline (absolute and relative).
- Task accuracy deltas with 95% CIs via bootstrap over examples.
- Latency, memory overhead, and throughput.
- Failure cases: instances where KLS harms performance.

Implementation details
- Code: PyTorch wrapper around model.generate or llama.cpp decoding loop.
- Calibration: collect y_t streams for 1–2k tokens on validation text; compute per-dim/group Q,R; cache K.
- Numerical: use float32 state for ŝ_t even if model logits are fp16/bf16 to avoid accumulation error.

Discussion
- Why smoothing helps: Empirically, quantization induces approximately zero-mean, near-diagonal noise on logits due to layerwise rounding and quantizer mismatch. Over short horizons, logits exhibit temporal correlation because the context evolves gradually; even when the top-1 token changes, the broader distribution shifts smoothly. The Kalman filter leverages this structure, adaptively weighting current observations against a denoised prior.
- Relation to prior work: Prior defenses against quantization error operate inside the network (e.g., per-channel scales, rounding-to-nearest-even, activation clipping) or adjust decoding (e.g., temperature, top-k). KLS is orthogonal: a post-hoc, online denoiser on logits with theory-backed gains and negligible overhead.
- When it might not help: If the token distribution changes abruptly between steps (e.g., after a delimiter), or if quantization noise is highly structured and non-diagonal, simple diagonal smoothing may underperform. Ablations quantify this boundary.
- Extensions:
  - Low-rank KLS: learn a small subspace (e.g., PCA of logits) and filter only subspace coefficients.
  - Context-aware gains: modulate K based on entropy or surprise (|y_t - ŝ_{t-1}|).
  - Training-time alignment: small KL regularizer encouraging temporal smoothness of logits to increase KLS efficacy post-quantization.

Limitations
- The diagonal assumption ignores cross-token correlations; full covariance is infeasible at vocab scale.
- Calibration drift across domains or prompts can reduce gains; per-session online re-estimation of R is possible but adds complexity.
- Benefits likely diminish for 8-bit quantization and strong instruction-tuned models with sharper logits, where measurement noise is smaller relative to process noise.
- Memory overhead for full-vocab state can matter for large batches; the top-k variant mitigates this at a small accuracy trade-off.

Conclusion
Kalman-Logit Smoothing is a simple, theoretically grounded, and practical method to counteract quantization-induced logit noise during LLM inference. It requires only a few vector operations per token, integrates seamlessly with existing decoders, and can be calibrated with or without full-precision references. The proposed experiments will determine whether KLS reliably improves perplexity and task accuracy at fixed latency, and delineate when state-space denoising is most beneficial for low-precision LLMs.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
