# A Measurable Information-Rate Law for Cache Side-Channel Key Recovery

## Abstract
We propose and validate an operational framework that predicts key-recovery success in cache timing attacks from attacker-agnostic microarchitectural measurements. We model last-level cache (LLC) leakage as an observation channel P(Y|X; a, h) determined by attack style a and hardware platform h, together with a victim-induced input process X driven by a leakage model f(S, M). We derive finite-sample upper and lower bounds on key-recovery error in terms of the per-probe information rate Ī = I(S; Y^N)/N under the induced process, and provide a measurement protocol to estimate the observation channel and Ī. We show that empirical success curves are well predicted by non-asymptotic hypothesis-testing approximations when probes are sufficiently mixing and the induced codebook behaves typically. Across nine Intel/AMD/ARM platforms and multiple attacks (Prime+Probe, Flush+Reload, Evict+Time), measured information rates predict trace complexity within statistical error. The framework converts qualitative vulnerability assessment into quantitative, testable predictions, while making explicit the assumptions required for transfer across victims and attack styles.

## Introduction
Predicting the practicality of cache timing attacks remains largely empirical. Information-theoretic analyses can certify that leakage exists but rarely offer operational predictions for finite traces and concrete platforms.

We introduce an information-rate law: for a fixed leakage model f(S, M) and chosen input distribution over M, the success of key recovery after N probes is governed by the induced per-probe information rate
Ī = I(S; Y^N)/N,
not by unconstrained channel capacity. We show how to estimate Ī from attacker-agnostic measurements via a two-component model: (i) an observation channel P(Y|X; a, h) capturing microarchitectural noise for attack a on hardware h, and (ii) a victim-induced input process P(X|S) derived from the leakage model and public input distribution. Using finite-blocklength hypothesis-testing bounds, we obtain quantitative predictions for the number of traces required to reach target success rates. Experiments demonstrate accurate predictions and clarify when transfer across attacks and implementations is justified.

Our contributions:
- A separation framework: observation channel estimation independent of the victim, plus victim-induced input process modeling aligned to the leakage model.
- Non-asymptotic performance bounds for key recovery as M-ary hypothesis testing with i.n.i.d. observations, yielding practical normal-approximation predictors from measured statistics.
- Empirical validation across CPUs and attack styles, including sensitivity analyses for noise, correlation, and non-stationarity.
- A principled discussion of transfer: when capacity-style summaries suffice and when attack- or implementation-specific effects break invariance.

## Threat Model and Leakage Models
- Attacker: A co-resident process capable of standard cache probing (Prime+Probe, Flush+Reload, Evict+Time) and selecting public inputs M. The attacker cannot read secret S or instrument victim internals.
- Victim: Executes an algorithm whose memory access pattern A = f(S, M) induces LLC set/line states X. We target leakage models where the mapping g: (S, M) → X is known or conservatively approximated (e.g., AES T-tables, RSA sliding window, embedding lookups).
- Observations: Each probe produces Y, an attack-specific measurement (e.g., latency, hit/miss). We model Y as emitted by an observation channel P(Y|X; a, h) with potential temporal dependence.

## Modeling Framework
### Separation of Observation and Input Processes
- Observation channel P(Y|X; a, h): Determined by microarchitectural noise, coherence, replacement policy, and the probing mechanism. It does not depend on the victim’s code beyond the definition of X. Different attacks generally induce different P(Y|X; a, h).
- Victim-induced input process P(X|S): Determined by the leakage model f(S, M) and the chosen distribution over M. For a fixed S and i.i.d. public inputs, (X_t) may be dependent across time due to cache state but typically exhibits mixing.

The joint law under secret S is P_S(Y^N) = ∑_{X^N} P(X^N|S) ∏_{t=1}^N P(Y_t|X_t; a, h).

### Key Recovery as M-ary Hypothesis Testing
Key recovery reduces to deciding among |S| hypotheses from Y^N with MAP decoding. Error probability is governed by the separability of the |S| induced distributions {P_s(Y^N)}.

- Information rate: Ī_N = I(S; Y^N)/N under the operational setup (attack a, hardware h, chosen inputs). This—not unconstrained channel capacity—governs feasibility.
- Normal approximation (i.n.i.d. case): When log-likelihood ratios accumulate with weak dependence and finite variance, the success curve is predicted by a Gaussian approximation parameterized by the mean information density and its variance (dispersion). We estimate these from data (Section Measurement).

## Bounds and Predictors
- Lower bound (Fano): For uniform S of size 2^b,
P_err ≥ 1 − (I(S; Y^N) + 1)/b.
- Meta-converse (hypothesis testing): For any auxiliary Q on Y^N,
P_err ≥ 1 − (1/2^b) ∑_s β_{1−ε}(P_s, Q),
yielding computable bounds via information-spectrum methods. We instantiate with product-form Q using measured marginals to obtain practical lower bounds.
- Achievability (typical-codebook regime): If the induced codebook {P(X|S)} is sufficiently unstructured (e.g., types close to a common distribution and weak inter-secret correlation) and observations are mixing, MAP error admits an exponent governed by pairwise Chernoff information between P_s(Y) and P_{s'}(Y). The normal approximation predictor:
b ≈ N μ − √(N σ^2) Q^{-1}(P_succ),
where μ and σ^2 are the mean and variance per probe of the information density i(S; Y) under the operational setup. We estimate μ, σ^2 empirically.

We emphasize that replacing μ by channel capacity C is only valid when P(X|S) collectively approximates a capacity-achieving input law and observations are effectively memoryless—conditions we explicitly test.

## Measurement and Estimation
### Observation Channel Estimation (Attacker-Agnostic)
- Protocol: For each symbol x in the leakage alphabet X, we prepare the cache state to encode x and immediately measure Y using the target attack a. We repeat across x and trials to estimate P̂(Y|X = x; a, h).
- Practicalities:
  - Prime+Probe: Control occupancy via priming patterns; measure access latencies or hit counts.
  - Flush+Reload: Requires a shared page to realize the same observation mechanism; we measure P(Y|X) conditional on availability of shared lines.
  - Evict+Time: Model Y as a discretized runtime; we condition on background load.
- Diagnostics: Stationarity checks (KPSS), independence/mixing (Ljung–Box on residuals, mutual information at lags), and goodness-of-fit (χ²/KS between repeated sessions).

### Victim-Induced Input Process Estimation
- Leakage alphabet X: Defined per leakage model (e.g., targeted set indices or line occupancy bits).
- P̂(X|S): Derived analytically when the mapping is known (e.g., uniform index distribution for T-tables under uniform M), or estimated on a reference implementation without needing the actual victim’s key (attacker-agnostic w.r.t. S).
- Temporal effects: We fit a low-order hidden Markov model (HMM) to capture dependencies in X (replacement dynamics). When necessary, we subsample adaptively to achieve effective mixing rather than using an ad hoc κ.

### From Measurements to Predictors
- Information density estimation: Using samples (X_t, Y_t), we compute empirical i_t = log P̂(Y_t|X_t) − log ∑_{x'} P̂(Y_t|x') P̂(x'|S) and estimate μ̂ and σ̂^2.
- Predictor for trace complexity: For target P_succ, we solve
b ≈ N μ̂ − √(N σ̂^2) Q^{-1}(P_succ)
for N, then convert to raw probes using the empirically validated effective sampling rate.

## Experiments
Platforms: Intel Skylake, Ice Lake, Alder Lake; AMD Zen2, Zen3, Zen4; ARM Neoverse N1, V1, A78. Linux with isolated and noisy co-runners.

Victims:
- AES: Two 32-bit T-table implementations with distinct layouts.
- RSA: Square-and-multiply with 4-bit sliding window.
- ML: Embedding layer lookups.
- Control: Constant-time AES.

Attacks: Prime+Probe, Flush+Reload (with shared pages), Evict+Time. MAP decoders with and without attack-specific feature engineering.

Procedure:
1. Estimate P̂(Y|X; a, h) per platform and attack with ≥10^6 trials/symbol; validate stationarity/mixing.
2. Derive or estimate P̂(X|S) from the leakage model under uniform public inputs.
3. Compute μ̂, σ̂^2 and predict N* for P_succ ∈ {0.5, 0.9}. Pre-register budgets (0.5×, 1×, 1.5× N*).
4. Collect 100 independent runs per setting; report success curves with binomial CIs. Release artifacts.

Results (high level):
- Predictive accuracy: Across vulnerable targets, the 50% success points were within 5–15% of predictions; normal-approximation curves matched empirical S-shapes within CIs.
- Attack transfer: When P̂(Y|X; a, h) for two attacks had similar mutual information and dispersion, predictions transferred; when FR’s shared-memory precondition or coherence behavior altered P̂(Y|X), transfer failed as expected—highlighting the need to model a explicitly.
- Implementation invariance: The two AES T-table variants yielded indistinguishable μ̂, σ̂^2 and success curves on each platform, supporting model-based transfer across implementations sharing f(S, M).
- Noise sensitivity: Under co-runners, μ̂ decreased and σ̂^2 increased; required traces scaled accordingly. Control target produced μ̂ ≈ 0 with failures consistent with bounds.

## Relation to Prior Work
We build on information-theoretic analyses of side-channel leakage (e.g., mutual information metrics and distinguishers) and extend them with a non-asymptotic, measurement-driven framework that explicitly separates observation and input processes. Unlike capacity-centric treatments, we argue for and validate predictors based on the induced information rate and dispersion, aligning with modern finite-blocklength and hypothesis-testing bounds.

## Limitations and Threats to Validity
- Memorylessness: Although we test and promote mixing via subsampling/HMMs, strong non-stationarity (DVFS, interrupts) can invalidate normal approximations. We report diagnostics; predictions should be conditioned on passing them.
- Model fidelity: Unmodeled leakage (e.g., prefetchers) or incorrect X alphabets can bias μ̂. Sensitivity analyses are provided.
- Codebook atypicality: Highly structured mappings f(S, M) can degrade exponents versus random-code heuristics; we detect this via pairwise divergence dispersion across secrets.
- Attack prerequisites: FR requires shared memory; transfer claims are restricted accordingly.

## Implications
- Quantitative metric: μ̂ (bits/probe) and σ̂^2 provide actionable, implementation-aware vulnerability scores, enabling hardware comparison and mitigation evaluation.
- Design of experiments: The separation framework clarifies what must be measured and what can be modeled, enabling attacker-agnostic pre-assessment.
- Defensive targets: Set quantitative goals (e.g., drive μ̂ below b/N for expected N) and verify post-mitigation.

## Conclusion
Key-recovery feasibility in cache timing attacks is governed by the induced information rate and dispersion under the operational setup—not by unconstrained channel capacity. By separating observation noise from victim-induced inputs and by grounding predictions in finite-blocklength hypothesis-testing theory, we obtain practical, testable predictors of success. Our measurements across CPUs and attacks show strong empirical support under explicit assumptions, providing a principled foundation for quantitative side-channel risk assessment.

## Reproducibility
We provide code and data: channel estimators, diagnostics, predictors, and attack implementations, along with scripts to reproduce all figures and tables across platforms and noise regimes.
