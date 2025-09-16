Title: Entropy-Matched Adaptive Depth Decoding

Abstract
We propose Entropy-Matched Adaptive Depth (EMAD) decoding, a training-free method that reduces per-token inference depth in large language models (LLMs) by stopping computation when an intermediate layer’s predictive entropy matches a teacher-forcing entropy profile measured offline. EMAD uses a small calibration pass to estimate the expected entropy-vs-depth curve under teacher forcing and applies it at inference to gate early exits with a monotone non-increasing depth schedule that preserves KV-cache consistency. The method requires no retraining, no auxiliary teachers, and can be implemented with small open-source models. We outline a falsification plan comparing accuracy and latency against full-depth decoding, fixed-depth truncation, and confidence-based baselines, and describe practical entropy estimators that add negligible overhead. If accuracy degrades or latency does not improve on standard benchmarks, the approach is falsified.

Introduction
Inference in decoder-only Transformers scales linearly with depth and sequence length. Many tokens in generation appear “easy,” exhibiting high confidence early in the forward pass. Prior dynamic-depth methods for classification attach early-exit heads and train them; for generation, early-exit faces a key challenge: KV-cache dependencies couple layers across time, so per-token variable depth can require expensive backfilling. Existing acceleration techniques (e.g., speculative decoding) reduce steps, not per-step depth, and typically rely on an auxiliary draft model.

We introduce EMAD, a simple, training-free criterion for early exit that leverages a property of teacher-forcing: as depth increases, the per-token predictive entropy under teacher forcing follows a characteristic decreasing profile. EMAD estimates this profile offline and stops decoding for a token at the first layer where the current entropy matches the teacher-forcing profile (optionally with a safety margin). To avoid KV backfill, EMAD enforces a monotone non-increasing per-step depth schedule. The result is a practical, calibration-based, adaptive-depth decoder that preserves accuracy while reducing compute.

Contributions:
- A training-free, entropy-matching criterion that uses teacher-forcing entropy profiles to govern per-token depth.
- A KV-consistent decoding schedule (monotone non-increasing depth) enabling real speedups in standard Transformer decoders.
- Lightweight entropy estimation techniques usable with small open-source models.
- A falsification-oriented experimental plan with clear metrics, ablations, and open-source reproducibility.

Method
Problem setting
Consider a decoder-only Transformer with L blocks and a tied LM head (unembedding). For a sequence of length T, standard decoding computes all L layers per token. We seek a per-step depth D_t ≤ L such that computing only the first D_t layers for token t causes minimal degradation.

Teacher-forcing entropy profile
Let H_ℓ denote the entropy of the model’s predictive distribution when applying the LM head to the hidden state after layer ℓ, under teacher forcing. For a calibration corpus, we run a single pass with teacher forcing and record H_ℓ at a small set of probe layers ℓ ∈ P (e.g., every 4 layers). We then compute a smoothed, non-increasing baseline curve E*[H_ℓ] over tokens (optionally stratified by simple context features; see below). This baseline represents how uncertainty typically decreases with depth when conditioning on ground-truth history.

Entropy-matched early exit
At inference, for each token t we compute intermediate entropies Ĥ_ℓ at probe layers as we build the forward pass. We choose the smallest layer ℓ in increasing order such that:
- Ĥ_ℓ ≤ τ_ℓ,
where τ_ℓ is the calibrated teacher-forcing target at that layer. To ensure KV consistency, we also enforce D_t ≤ D_{t-1} (monotone non-increasing depth across time). If no probe layer satisfies the criterion, we use full depth.

Calibration details
- Global profile: τ_ℓ = mean teacher-forcing entropy at layer ℓ over the calibration set, smoothed to be non-increasing in ℓ.
- Safety margin: τ_ℓ can be tightened with a margin m ≥ 0: τ_ℓ := τ_ℓ − m to bias toward deeper computation when uncertain.
- Conditioning: τ_ℓ can be conditioned by one or two coarse features that are cheap to compute and stable across domains, such as:
  - token position bucket (early vs late in sequence),
  - unigram frequency bucket of the target token (optional if labels available in calibration only; at inference use the top-1 predicted token’s frequency),
  - shallow-layer confidence (e.g., top-1 probability at the first probe layer).
  In practice, a 1D or 2D histogram (feature bucket × layer) suffices.

Entropy estimation at inference
Computing full softmax over a large vocabulary at every probe layer can be expensive. EMAD supports three estimators:
1) Full softmax at probe layers: simplest; practical for small models and sparse probes.
2) Top-K softmax entropy: obtain top-K logits via approximate MIPS on the unembedding matrix (K ≈ 32–128); compute partial softmax and upper-bound the tail mass by log-sum-exp of a shared floor. Calibrate Ĥ_ℓ to match full entropies on a held-out set via linear regression.
3) Low-rank readout: precompute an r-rank factorization of the unembedding matrix (r ≪ d). At inference, project hidden states to r-dim and compute top-K as above. This amortizes cost across layers and maintains accuracy of entropy ranking.

KV-consistent scheduling
Variable depth per token is only efficient if it avoids backfilling KVs for skipped layers. EMAD adopts a monotone non-increasing schedule D_t ≤ D_{t-1}. Intuitively, early tokens often require more reasoning depth; as the context grows and uncertainty drops, EMAD reduces depth, yielding actual compute savings. This schedule guarantees that when computing token t at layer k ≤ D_t, all required KVs from previous tokens exist.

Algorithm (per token)
- For ℓ in ascending order over probe layers up to D_{t-1}:
  - Run layer ℓ forward with KV caching.
  - Compute Ĥ_ℓ using the chosen estimator.
  - If Ĥ_ℓ ≤ τ_ℓ, set D_t := ℓ and break.
- If no early exit, set D_t := min(L, D_{t-1}).
- Emit token using the LM head at depth D_t; do not execute layers > D_t.

Complexity and expected speedup
Let f be the fraction of tokens exiting at or before layer m on average, and let probes be every s layers. The compute cost per token is approximately:
- Transformer blocks: E[D_t] / L of full compute.
- Entropy overhead: O(|P|) readouts per token until exit. With s ≈ 4–8 and early exits common after a few probes, overhead is typically <5% of block FLOPs in small models. Low-rank/top-K estimators further reduce overhead.

Relation to prior work
- Early exit for classification typically trains auxiliary heads; EMAD is training-free and designed for generation with KV consistency.
- Confidence-based decoding (e.g., CALM) skips future steps when confident; EMAD adapts per-step depth instead and can complement those methods.
- Speculative decoding reduces steps via a draft model; EMAD reduces per-step depth and can be composed with speculative or parallel decoding.

Experiments (falsification plan)
Goals:
- Validate whether entropy matching preserves accuracy while reducing latency and FLOPs.
- Test robustness across tasks, models, and calibration data.
- Identify failure modes via ablations.

Models:
- Pythia-410M and 1.4B
- OPT-1.3B
- LLaMA-1 7B (optional, if resources permit)
All models used as-is, with tied unembedding; no retraining.

Calibration data and profiles:
- Calibration corpus: WikiText-103 train + a 10M-token subset of The Pile.
- Probe layers: every 4 layers for small models; every 6–8 layers for 7B.
- Compute τ_ℓ as mean entropy per layer; smooth with isotonic regression; optionally stratify by:
  - position bucket: [1–32], [33–128], [129+]
  - top-1 prob at first probe layer: [0–0.4], [0.4–0.7], [0.7–1.0]
- Select margin m on a small validation set to target an accuracy drop ≤0.1 perplexity points or ≤0.5% absolute EM drop.

Baselines:
- Full-depth decoding (reference).
- Fixed truncation: use first m layers for all tokens, m tuned for best accuracy.
- Confidence thresholding without teacher-forcing profile: exit when Ĥ_ℓ ≤ h0, with h0 tuned.
- CALM-like chunking (if implemented) to compare step-skipping vs depth-skipping.
- Speculative decoding (orthogonal; report composition with EMAD if feasible).

Tasks and metrics:
- Language modeling: WikiText-103 and Pile val perplexity.
- QA/commonsense: ARC-e, ARC-c EM/accuracy (greedy decoding).
- Code completion: HumanEval pass@1 (small models may be weak; still informative).
- Reasoning: GSM8K subset EM (expect limited success on small models; watch for degradation).
- Latency: wall-clock tokens/sec on A100 and consumer GPU; report 50th/90th percentile depth per token; FLOP estimates.
- Calibration stability: profiles trained on Corpus A tested on Corpus B.

Ablations:
- Probe stride s ∈ {2,4,6,8}.
- Entropy estimator: full, top-K, low-rank.
- Conditioning features: none vs 1D vs 2D.
- Margin m sensitivity curves (accuracy vs speed).
- Schedule constraint: with vs without monotone depth (the latter must backfill; report no speedup to illustrate necessity).
- Exit to non-probe layers: optional interpolation between probes.

Falsification criteria:
- If, at any operating point with ≥15% FLOP reduction, perplexity increases by >1% relative or EM/accuracy drops by >1% absolute on two or more tasks, consider EMAD falsified for that model class.
- If wall-clock speed improves by <10% at equal batch size and prompt length, EMAD fails practical relevance.

Implementation details:
- Minimal modifications to HuggingFace Transformers: expose intermediate hidden states at probe layers; share final RMSNorm and tied unembedding for entropy estimation; maintain monotone depth state D_{t-1}.
- Top-K entropy via FAISS or Torch MIPS over unembedding; calibrate Ĥ_ℓ with a linear map on held-out tokens.
- Public code with one-click scripts for models and tasks above.

Discussion
Why teacher-forcing entropy?
Under teacher forcing, intermediate representations reflect how the model refined uncertainty during training. Matching this entropy trajectory during free decoding provides a calibration target that is model- and layer-specific, avoiding ad hoc global thresholds. It implicitly adapts to architecture, scale, and tokenizer effects.

Monotone depth as an inductive bias
Although per-token ideal depth may fluctuate, the KV constraint makes monotone non-increasing schedules not only practical but also reasonable: uncertainty typically decreases as context grows. This bias trades minimal flexibility for substantial systems simplicity and real speedups.

Composability
EMAD is orthogonal to token-level accelerations (e.g., quantization, CUDA kernels), step-level accelerations (e.g., speculative decoding), and cache optimizations (e.g., paged KV). Composing EMAD with speculative decoding is straightforward: apply EMAD to both draft and verifier decoders, or only to the verifier to cut worst-case cost.

Security and safety implications
Depth reduction changes the internal computation path; while we expect preserved outputs within thresholds, rare edge cases might alter generations. For safety-critical deployments, use conservative margins and task-specific validation.

Limitations
- KV dependency forces monotone non-increasing depth; EMAD cannot increase depth later without backfilling, potentially missing late hard tokens.
- Calibration shift: τ_ℓ depends on the calibration corpus. Severe domain shift can miscalibrate thresholds; conditioning and margins mitigate this but do not eliminate it.
- Entropy estimation cost: for very large vocabularies, per-probe readout may be nontrivial; low-rank/top-K estimators help but add engineering complexity.
- Small models may show weaker benefits if entropy decreases slowly with depth (flatter profiles).
- EMAD assumes tied unembedding and compatible layernorm for intermediate readouts; deviations require minor adaptations.

Conclusion
EMAD is a training-free, entropy-matching approach to adaptive-depth decoding that uses teacher-forcing entropy profiles to govern early exit at intermediate layers under a KV-consistent schedule. It is simple to implement on small open-source models, admits clear falsification criteria, and composes with other inference accelerators. If validated, EMAD offers a principled, low-overhead path to reduce LLM inference cost without sacrificing accuracy. If not, the falsification plan will clarify where entropy matching fails to predict when “enough compute” has been spent on a token, guiding future dynamic-depth designs.
