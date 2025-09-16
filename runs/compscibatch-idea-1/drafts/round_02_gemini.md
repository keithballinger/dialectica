Minor Revisions.

This is an excellent, well-structured paper presenting a novel, practical, and falsifiable method. The core idea of matching a pre-calibrated teacher-forcing entropy profile is elegant and well-motivated. The paper correctly identifies the critical KV-cache problem for adaptive-depth generation and proposes a simple, effective scheduling solution. The experimental plan is exemplary in its rigor and focus on falsification. The proposed revisions are minor, aiming to further tighten the prose and consolidate a few redundant explanations for improved conciseness.

***

### Revised Draft

**Title: Entropy-Matched Adaptive Depth Decoding**

**Abstract**
We propose Entropy-Matched Adaptive Depth (EMAD) decoding, a training-free method to reduce per-token inference cost in large language models (LLMs). EMAD dynamically truncates the forward pass by exiting at an intermediate layer once its predictive entropy falls below a target profile. This profile, representing the model's typical uncertainty reduction, is estimated from a single, offline teacher-forcing pass over a calibration corpus. To maintain KV-cache consistency in autoregressive generation, EMAD enforces a monotone non-increasing depth schedule across tokens. The method requires no model retraining or auxiliary heads. Our validation plan for small open-source models specifies precise falsification criteria: EMAD is considered ineffective if it fails to improve latency by at least 10% or if it degrades accuracy on standard benchmarks by more than 1% for a given compute budget.

**Introduction**
Inference cost for decoder-only Transformers scales with both sequence length and model depth. Intuitively, many tokens are "easy" to predict and do not require the model's full computational depth. While prior work on dynamic depth for classification models has demonstrated the potential of early exiting, applying this to autoregressive generation is complicated by the KV-cache: since layer `ℓ` at timestep `t` depends on the cached keys and values from layer `ℓ` at all previous timesteps, naively varying the depth per token would require expensive backfilling of skipped layers.

Current acceleration methods largely bypass this issue. Speculative decoding reduces the number of sequential generation *steps* but not the computational depth per step. Other techniques like quantization reduce the cost of operations but do not alter the computation graph.

We introduce Entropy-Matched Adaptive Depth (EMAD), a training-free criterion for early exit that is compatible with standard KV-caching. The core insight is that a model's predictive entropy under teacher forcing (i.e., when conditioned on ground-truth tokens) follows a characteristic, decreasing profile as layer depth increases. EMAD pre-computes this profile and, during inference, terminates the forward pass for a given token at the first layer where the predictive entropy meets this pre-calibrated target. To prevent KV-cache inconsistency, EMAD constrains the per-token depth `D_t` to be monotone non-increasing (`D_t ≤ D_{t-1}`). This simple schedule is highly effective, as uncertainty often decreases as context accumulates, and it guarantees that all required KV-cache entries are available without backfilling.

Our contributions are:
1.  A novel, training-free early-exit criterion based on matching a pre-calibrated, teacher-forcing entropy profile.
2.  A KV-consistent, monotone non-increasing depth schedule that enables practical speedups.
3.  A comprehensive and falsifiable experimental plan to validate the method on open-source models.

**Method**
**Problem Setting**
Given a decoder-only Transformer with `L` layers, standard autoregressive decoding computes all `L` layers for each token `t`. We seek to determine a per-token depth `D_t ≤ L` to compute, minimizing cost while preserving accuracy.

**Teacher-Forcing Entropy Profile Calibration**
Let `H(p(·|x_{<t}, y_{t-1}))` be the predictive entropy of the model for the next token `y_t` given a prefix `x_{<t}`. When running the model with teacher forcing (i.e., `y_{t-1}` is the ground-truth token), we can measure this entropy after any layer `ℓ` by passing the layer's hidden state to the language model head. Let `H_ℓ` be this value.

On a calibration corpus, we perform a single teacher-forcing pass and record `H_ℓ` at a sparse set of probe layers `P`. The target entropy profile `τ_ℓ` for each probe layer `ℓ ∈ P` is then computed as the expected entropy `E[H_ℓ]` over the calibration set. We smooth `τ_ℓ` to be non-increasing with `ℓ`. Optionally, `τ_ℓ` can be conditioned on simple features like token position or shallow-layer confidence by stratifying the expectation calculation.

**EMAD Decoding**
At inference, for each token `t`, we compute the forward pass layer-by-layer. At each probe layer `ℓ`, we estimate the current predictive entropy `Ĥ_ℓ`. The chosen depth for the current token, `D_t`, is the shallowest probe layer `ℓ` that satisfies two conditions:
1.  **Entropy Match:** `Ĥ_ℓ ≤ τ_ℓ - m`, where `m ≥ 0` is a configurable safety margin.
2.  **KV-Cache Consistency:** `ℓ ≤ D_{t-1}`, where `D_{t-1}` is the depth used for the previous token (with `D_0 = L`).

If no probe layer satisfies the entropy match criterion, we set `D_t = min(L, D_{t-1})`. After determining `D_t`, we generate the token using the hidden state from layer `D_t`. All subsequent layers are skipped for the current token. The monotone schedule `D_t ≤ D_{t-1}` guarantees that for any layer `k ≤ D_t`, the required KV-cache values from all prior tokens `1...t-1` have been computed and stored.

**Entropy Estimation**
To minimize overhead, full-vocabulary softmax calculations at every probe layer can be avoided. We support two efficient estimators for `Ĥ_ℓ`:
1.  **Top-K Softmax:** Use approximate nearest neighbor search to find the top-K logits (e.g., `K=128`). Compute entropy over this partial distribution, using a calibrated correction factor for the tail mass.
2.  **Low-Rank Readout:** Pre-compute a low-rank factorization of the unembedding matrix. Project hidden states at probe layers into this low-rank space to estimate entropy with minimal cost.

**Relation to Prior Work**
Unlike early-exit methods for classification, EMAD requires no auxiliary heads or retraining. Unlike confidence-based methods that skip entire tokens (e.g., CALM), EMAD adapts the computational depth *within* a token's forward pass. It is complementary to and can be composed with step-skipping methods like speculative decoding.

**Experiments (Falsification Plan)**
**Goal:** Validate that EMAD reduces latency and FLOPs while preserving accuracy across diverse tasks.

**Models and Data:**
-   Models: Pythia-{410M, 1.4B}, OPT-1.3B, and LLaMA-1-7B. All models are used without modification.
-   Calibration: Profiles will be generated on a 100M-token subset of The Pile.
-   Evaluation: Standard benchmarks for language modeling (WikiText-103 perplexity), commonsense reasoning (ARC accuracy), code generation (HumanEval pass@1), and math reasoning (GSM8K accuracy).

**Baselines:**
1.  **Full-depth:** Standard `L`-layer decoding.
2.  **Fixed-depth:** Truncation at a fixed layer `m < L`, with `m` tuned for optimal performance.
3.  **Confidence-based:** Early exit when `Ĥ_ℓ ≤ h_0`, where `h_0` is a globally tuned entropy threshold.

**Metrics:**
-   Task-specific accuracy (Perplexity, Exact Match, pass@1).
-   Computational cost: FLOPs saved vs. full-depth.
-   Latency: Wall-clock tokens/second on A100 GPUs.

**Ablations:**
-   Impact of probe layer frequency, entropy estimators, and conditioning features.
-   Sensitivity analysis of the safety margin `m`.
-   Demonstration of the necessity of the monotone schedule by measuring the cost of a hypothetical backfilling implementation.

**Falsification Criteria:**
The EMAD hypothesis will be falsified if, for any model, either of the following is true:
1.  **Accuracy Degradation:** At any operating point with ≥15% FLOP reduction, perplexity increases by >1% (relative) or accuracy on any other benchmark drops by >1% (absolute).
2.  **Practical Irrelevance:** At matched accuracy, the wall-clock latency improvement is less than 10%.

**Discussion**
EMAD's use of a teacher-forcing profile provides a principled, model-specific target for early exit, implicitly adapting to architecture and scale. The monotone depth schedule is a pragmatic constraint that makes the method viable for real-world autoregressive systems. While this schedule prevents "recovering" to a greater depth for a difficult token that follows a sequence of easy ones, our hypothesis is that this trade-off is favorable in practice. The calibration process introduces a dependency on the calibration data, but we will test robustness to domain shift.

**Conclusion**
EMAD is a simple, training-free method for adaptive-depth decoding in LLMs. By matching inference-time entropy to a pre-calibrated teacher-forcing profile under a KV-consistent schedule, it has the potential to significantly reduce computational cost without model modification or accuracy loss. Our rigorous, falsification-oriented experimental plan provides a clear framework for validating this claim on standard open-source models and benchmarks.
