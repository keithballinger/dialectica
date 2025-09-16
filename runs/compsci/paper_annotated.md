Minor Revisions

Rationale: The paper is clear and technically strong but benefits from added plain-language definitions of key terms (e.g., information rate, dispersion, mixing) and brief motivation for the measurement and estimation steps. The following annotated version provides those clarifications without altering the original content.

# A Measurable Information-Rate Law for Cache Side-Channel Key Recovery
> Note: The title signals a practical rule (“law”) that links how much information leaks per measurement (“information rate”) to the feasibility of recovering cryptographic keys via cache side channels (timing-based leakage from CPU caches). “Measurable” means the quantities can be estimated from experiments on real hardware.

## Abstract
We propose and validate an operational framework that predicts key-recovery success in cache timing attacks from attacker-agnostic microarchitectural measurements. We model last-level cache (LLC) leakage as an observation channel P(Y|X; a, h) determined by attack style a and hardware platform h, together with a victim-induced input process X driven by a leakage model f(S, M). We derive finite-sample upper and lower bounds on key-recovery error in terms of the per-probe information rate Ī = I(S; Y^N)/N under the induced process, and provide a measurement protocol to estimate the observation channel and Ī. We show that empirical success curves are well predicted by non-asymptotic hypothesis-testing approximations when probes are sufficiently mixing and the induced codebook behaves typically. Across nine Intel/AMD/ARM platforms and multiple attacks (Prime+Probe, Flush+Reload, Evict+Time), measured information rates predict trace complexity within statistical error. The framework converts qualitative vulnerability assessment into quantitative, testable predictions, while making explicit the assumptions required for transfer across victims and attack styles.
> Note: Plainly: They present a way to forecast how many cache-timing measurements (“probes”) it takes to recover a secret key, using measurements that don’t depend on a specific victim program. Key terms: LLC = last-level cache. X = the cache state the victim causes (depends on secret S and chosen public inputs M via f(S, M)). Y = what the attacker measures (e.g., latencies). P(Y|X; a, h) = probability model of observations given cache state X for attack a on hardware h. I(S; Y^N) = mutual information between secret S and N observations Y^N; Ī = information per probe. “Finite-sample bounds” predict performance with limited traces. “Non-asymptotic hypothesis testing” here uses statistical tools to predict success curves realistically (not just infinite-data limits). “Mixing” means successive measurements are not too correlated. “Induced codebook” = the set of possible cache-state sequences X^N that each secret S induces; “typical” means not pathological. Findings: Across many CPUs and attack types (Prime+Probe, Flush+Reload, Evict+Time), measured Ī predicted the number of traces needed within statistical error.

## Introduction
Predicting the practicality of cache timing attacks remains largely empirical. Information-theoretic analyses can certify that leakage exists but rarely offer operational predictions for finite traces and concrete platforms.
> Note: Most prior work either demonstrates leakage qualitatively or uses asymptotic theory; it doesn’t tell you “how many traces do I need on this CPU for this attack?”

We introduce an information-rate law: for a fixed leakage model f(S, M) and chosen input distribution over M, the success of key recovery after N probes is governed by the induced per-probe information rate
Ī = I(S; Y^N)/N,
not by unconstrained channel capacity. We show how to estimate Ī from attacker-agnostic measurements via a two-component model: (i) an observation channel P(Y|X; a, h) capturing microarchitectural noise for attack a on hardware h, and (ii) a victim-induced input process P(X|S) derived from the leakage model and public input distribution. Using finite-blocklength hypothesis-testing bounds, we obtain quantitative predictions for the number of traces required to reach target success rates. Experiments demonstrate accurate predictions and clarify when transfer across attacks and implementations is justified.
> Note: Definitions: N = number of probes; Ī = information per probe about secret S from the whole sequence Y^N. “Channel capacity” is the maximum possible information rate with optimal inputs; they argue it’s not the right predictor because the victim’s inputs and dynamics are constrained by f(S, M). The two-component model separates (a) how hardware/attack translate cache states X into observations Y and (b) how the victim’s secret and chosen inputs generate X. “Finite-blocklength” means predictions for finite N. “Transfer” means whether measurements/predictions carry over between attacks or implementations.

Our contributions:
- A separation framework: observation channel estimation independent of the victim, plus victim-induced input process modeling aligned to the leakage model.
- Non-asymptotic performance bounds for key recovery as M-ary hypothesis testing with i.n.i.d. observations, yielding practical normal-approximation predictors from measured statistics.
- Empirical validation across CPUs and attack styles, including sensitivity analyses for noise, correlation, and non-stationarity.
- A principled discussion of transfer: when capacity-style summaries suffice and when attack- or implementation-specific effects break invariance.
> Note: “i.n.i.d.” = independent but not identically distributed (observations may vary over time). “M-ary hypothesis testing” = deciding among many possible secrets. “Normal approximation” uses Gaussian statistics to approximate success curves using the mean and variance of information per probe. They validate on multiple CPUs and attacks and explain when simple capacity-like summaries fail.

## Threat Model and Leakage Models
- Attacker: A co-resident process capable of standard cache probing (Prime+Probe, Flush+Reload, Evict+Time) and selecting public inputs M. The attacker cannot read secret S or instrument victim internals.
- Victim: Executes an algorithm whose memory access pattern A = f(S, M) induces LLC set/line states X. We target leakage models where the mapping g: (S, M) → X is known or conservatively approximated (e.g., AES T-tables, RSA sliding window, embedding lookups).
- Observations: Each probe produces Y, an attack-specific measurement (e.g., latency, hit/miss). We model Y as emitted by an observation channel P(Y|X; a, h) with potential temporal dependence.
> Note: Attacker can send chosen inputs M to the victim (e.g., plaintexts) but cannot see S or victim internals. The victim’s memory accesses A (driven by S and M through f) change the cache (X). The attacker measures Y (like timing or hit/miss). P(Y|X; a, h) captures hardware/attack effects; it can have time dependence. Examples of f: AES table lookups, RSA sliding window, ML embedding accesses.

## Modeling Framework
### Separation of Observation and Input Processes
- Observation channel P(Y|X; a, h): Determined by microarchitectural noise, coherence, replacement policy, and the probing mechanism. It does not depend on the victim’s code beyond the definition of X. Different attacks generally induce different P(Y|X; a, h).
- Victim-induced input process P(X|S): Determined by the leakage model f(S, M) and the chosen distribution over M. For a fixed S and i.i.d. public inputs, (X_t) may be dependent across time due to cache state but typically exhibits mixing.
> Note: Key idea: decouple “how the cache state is sensed” (Y given X) from “how the victim creates cache states” (X given S). “Mixing” means correlations between X_t and X_{t+k} fade with k (important for using normal approximations).

The joint law under secret S is P_S(Y^N) = ∑_{X^N} P(X^N|S) ∏_{t=1}^N P(Y_t|X_t; a, h).
> Note: This equation says: the probability of observing the sequence Y^N given secret S is the sum over all possible cache-state sequences X^N of (probability that the victim produces X^N given S) times (the product over time of the chance that each Y_t arises from X_t under attack a and hardware h). Symbols: P_S(Y^N) = distribution of Y^N conditioned on S; X^N = (X_1,...,X_N); Y^N = (Y_1,...,Y_N); P(X^N|S) = victim-induced process; P(Y_t|X_t; a, h) = observation channel at time t.

### Key Recovery as M-ary Hypothesis Testing
Key recovery reduces to deciding among |S| hypotheses from Y^N with MAP decoding. Error probability is governed by the separability of the |S| induced distributions {P_s(Y^N)}.
> Note: “MAP decoding” = choose the secret S that maximizes its posterior probability given Y^N. Success depends on how distinct the probability distributions over Y^N are for different secrets.

- Information rate: Ī_N = I(S; Y^N)/N under the operational setup (attack a, hardware h, chosen inputs). This—not unconstrained channel capacity—governs feasibility.
- Normal approximation (i.n.i.d. case): When log-likelihood ratios accumulate with weak dependence and finite variance, the success curve is predicted by a Gaussian approximation parameterized by the mean information density and its variance (dispersion). We estimate these from data (Section Measurement).
> Note: Ī_N = mutual information per probe; it quantifies, on average, how many bits about S each probe reveals under real conditions. “Information density” i(S;Y) = log P(Y|S) − log P(Y); its mean is mutual information and its variance is the “dispersion.” If dependencies are weak, sums of i(S;Y_t) behave approximately Gaussian, enabling closed-form predictors.

## Bounds and Predictors
- Lower bound (Fano): For uniform S of size 2^b,
P_err ≥ 1 − (I(S; Y^N) + 1)/b.
- Meta-converse (hypothesis testing): For any auxiliary Q on Y^N,
P_err ≥ 1 − (1/2^b) ∑_s β_{1−ε}(P_s, Q),
yielding computable bounds via information-spectrum methods. We instantiate with product-form Q using measured marginals to obtain practical lower bounds.
- Achievability (typical-codebook regime): If the induced codebook {P(X|S)} is sufficiently unstructured (e.g., types close to a common distribution and weak inter-secret correlation) and observations are mixing, MAP error admits an exponent governed by pairwise Chernoff information between P_s(Y) and P_{s'}(Y). The normal approximation predictor:
b ≈ N μ − √(N σ^2) Q^{-1}(P_succ),
where μ and σ^2 are the mean and variance per probe of the information density i(S; Y) under the operational setup. We estimate μ, σ^2 empirically.
> Note: Symbols: P_err = probability of error; S uniform over 2^b secrets (b = key bits or bit-equivalent hypothesis count). Fano’s inequality gives a necessary condition (lower bound on error). The meta-converse uses binary hypothesis testing: β_{1−ε}(P_s, Q) = minimum Type II error (false acceptance) achievable with power ≥ 1−ε against an auxiliary distribution Q; choosing Q well gives computable bounds. “Product-form Q” means assuming independence across probes using measured single-probe distributions. “Codebook” = the set of sequence distributions induced by different secrets. “Chernoff information” measures distinguishability between two distributions and dictates error exponents. Predictor formula: μ = E[i(S;Y)] (bits per probe); σ^2 = Var[i(S;Y)] (dispersion); N = number of probes; P_succ = desired success probability; Q^{-1} is the inverse Gaussian tail function (e.g., Q^{-1}(0.9) ≈ −1.281). This predicts the number of probes N needed to resolve b bits at target P_succ. Capacity C is not used unless the induced inputs mimic a capacity-achieving distribution and memory effects are negligible.

We emphasize that replacing μ by channel capacity C is only valid when P(X|S) collectively approximates a capacity-achieving input law and observations are effectively memoryless—conditions we explicitly test.
> Note: Practically: don’t just measure channel capacity; use the observed μ and σ^2 under the real victim+attack process unless you’ve verified memoryless behavior and capacity-achieving inputs.

## Measurement and Estimation
### Observation Channel Estimation (Attacker-Agnostic)
- Protocol: For each symbol x in the leakage alphabet X, we prepare the cache state to encode x and immediately measure Y using the target attack a. We repeat across x and trials to estimate P̂(Y|X = x; a, h).
- Practicalities:
  - Prime+Probe: Control occupancy via priming patterns; measure access latencies or hit counts.
  - Flush+Reload: Requires a shared page to realize the same observation mechanism; we measure P(Y|X) conditional on availability of shared lines.
  - Evict+Time: Model Y as a discretized runtime; we condition on background load.
- Diagnostics: Stationarity checks (KPSS), independence/mixing (Ljung–Box on residuals, mutual information at lags), and goodness-of-fit (χ²/KS between repeated sessions).
> Note: Goal: empirically characterize P(Y|X; a,h) without touching the victim. “Leakage alphabet X” = the discrete set of cache states the model cares about (e.g., which cache set is touched). They repeatedly force a known X and record Y to estimate P̂(Y|X). Diagnostics: KPSS tests whether the time series is stationary; Ljung–Box checks for autocorrelation; χ²/KS compare distributions across sessions. These ensure the channel model is stable and mixing enough for the theory to apply.

### Victim-Induced Input Process Estimation
- Leakage alphabet X: Defined per leakage model (e.g., targeted set indices or line occupancy bits).
- P̂(X|S): Derived analytically when the mapping is known (e.g., uniform index distribution for T-tables under uniform M), or estimated on a reference implementation without needing the actual victim’s key (attacker-agnostic w.r.t. S).
- Temporal effects: We fit a low-order hidden Markov model (HMM) to capture dependencies in X (replacement dynamics). When necessary, we subsample adaptively to achieve effective mixing rather than using an ad hoc κ.
> Note: They either compute P(X|S) from algorithmic structure (e.g., AES T-tables map inputs to table indices approximately uniformly) or estimate it empirically. HMMs capture time-dependence (e.g., cache replacement). Subsampling (using every k-th probe) can reduce correlation (“achieve mixing”), improving normal-approximation accuracy.

### From Measurements to Predictors
- Information density estimation: Using samples (X_t, Y_t), we compute empirical i_t = log P̂(Y_t|X_t) − log ∑_{x'} P̂(Y_t|x') P̂(x'|S) and estimate μ̂ and σ̂^2.
- Predictor for trace complexity: For target P_succ, we solve
b ≈ N μ̂ − √(N σ̂^2) Q^{-1}(P_succ)
for N, then convert to raw probes using the empirically validated effective sampling rate.
> Note: They compute per-probe information density i_t from measured channel P̂(Y|X) and modeled P̂(X|S): i_t = log-likelihood ratio log[P̂(Y_t|S)/P̂(Y_t)], where P̂(Y_t|S) = ∑_x P̂(Y_t|x) P̂(x|S) and P̂(Y_t) = average over S. Then μ̂ = average of i_t; σ̂^2 = variance of i_t. Using the predictor equation with μ̂ and σ̂^2 yields the required number of probes N for a desired success probability P_succ. “Effective sampling rate” accounts for any subsampling/mixing adjustments so predicted N maps to actual probe counts.

## Experiments
Platforms: Intel Skylake, Ice Lake, Alder Lake; AMD Zen2, Zen3, Zen4; ARM Neoverse N1, V1, A78. Linux with isolated and noisy co-runners.
> Note: A broad set of CPUs (Intel/AMD/ARM) and operating conditions (quiet vs. background noise) to test generality.

Victims:
- AES: Two 32-bit T-table implementations with distinct layouts.
- RSA: Square-and-multiply with 4-bit sliding window.
- ML: Embedding layer lookups.
- Control: Constant-time AES.
> Note: Targets include known leaky designs (AES T-tables, RSA sliding window), a data-dependent ML workload (embedding lookups), and a hardened baseline (constant-time AES) to verify the method reports near-zero information rate when leakage is mitigated.

Attacks: Prime+Probe, Flush+Reload (with shared pages), Evict+Time. MAP decoders with and without attack-specific feature engineering.
> Note: They evaluate several common cache attacks. MAP decoders are used both straightforwardly and with tailored features to check robustness of predictions across decoding strategies.

Procedure:
1. Estimate P̂(Y|X; a, h) per platform and attack with ≥10^6 trials/symbol; validate stationarity/mixing.
2. Derive or estimate P̂(X|S) from the leakage model under uniform public inputs.
3. Compute μ̂, σ̂^2 and predict N* for P_succ ∈ {0.5, 0.9}. Pre-register budgets (0.5×, 1×, 1.5× N*).
4. Collect 100 independent runs per setting; report success curves with binomial CIs. Release artifacts.
> Note: Why these steps matter: (1) High-sample estimation ensures accurate channel models and diagnostics. (2) Ground-truth the input process consistent with the chosen leakage model and inputs. (3) Make upfront predictions for how many probes are needed; preregistration guards against tuning to outcomes. (4) Multiple independent runs and confidence intervals test prediction accuracy and variability; releasing code/data supports reproducibility.

Results (high level):
- Predictive accuracy: Across vulnerable targets, the 50% success points were within 5–15% of predictions; normal-approximation curves matched empirical S-shapes within CIs.
- Attack transfer: When P̂(Y|X; a, h) for two attacks had similar mutual information and dispersion, predictions transferred; when FR’s shared-memory precondition or coherence behavior altered P̂(Y|X), transfer failed as expected—highlighting the need to model a explicitly.
- Implementation invariance: The two AES T-table variants yielded indistinguishable μ̂, σ̂^2 and success curves on each platform, supporting model-based transfer across implementations sharing f(S, M).
- Noise sensitivity: Under co-runners, μ̂ decreased and σ̂^2 increased; required traces scaled accordingly. Control target produced μ̂ ≈ 0 with failures consistent with bounds.
> Note: Key takeaways: The predictor works within error bars. Transfer across attacks/implementations is justified only when the measured channel statistics (mutual information μ and dispersion σ^2) match. Added noise reduces information per probe (μ) and increases uncertainty (σ^2), so more probes are needed. The constant-time control shows near-zero leakage as expected.

## Relation to Prior Work
We build on information-theoretic analyses of side-channel leakage (e.g., mutual information metrics and distinguishers) and extend them with a non-asymptotic, measurement-driven framework that explicitly separates observation and input processes. Unlike capacity-centric treatments, we argue for and validate predictors based on the induced information rate and dispersion, aligning with modern finite-blocklength and hypothesis-testing bounds.
> Note: Prior work often measures leakage via mutual information or focuses on channel capacity. This work emphasizes operational conditions (finite N, real inputs, memory) and uses modern coding-theory tools (finite-blocklength bounds) to produce actionable predictions.

## Limitations and Threats to Validity
- Memorylessness: Although we test and promote mixing via subsampling/HMMs, strong non-stationarity (DVFS, interrupts) can invalidate normal approximations. We report diagnostics; predictions should be conditioned on passing them.
- Model fidelity: Unmodeled leakage (e.g., prefetchers) or incorrect X alphabets can bias μ̂. Sensitivity analyses are provided.
- Codebook atypicality: Highly structured mappings f(S, M) can degrade exponents versus random-code heuristics; we detect this via pairwise divergence dispersion across secrets.
- Attack prerequisites: FR requires shared memory; transfer claims are restricted accordingly.
> Note: Risks: Time-varying systems (DVFS = dynamic voltage/frequency scaling) or interrupts can break assumptions. Missing leakage sources or wrong abstractions (X) distort estimates. Some algorithms may induce structured patterns that weaken typical-case exponents. Flush+Reload needs shared pages, limiting generalization.

## Implications
- Quantitative metric: μ̂ (bits/probe) and σ̂^2 provide actionable, implementation-aware vulnerability scores, enabling hardware comparison and mitigation evaluation.
- Design of experiments: The separation framework clarifies what must be measured and what can be modeled, enabling attacker-agnostic pre-assessment.
- Defensive targets: Set quantitative goals (e.g., drive μ̂ below b/N for expected N) and verify post-mitigation.
> Note: Practically: You can rank platforms/defenses by measured μ̂ and σ̂^2, plan measurements efficiently, and set concrete defense goals (e.g., reduce μ̂ so that feasible N for an attacker exceeds realistic budgets).

## Conclusion
Key-recovery feasibility in cache timing attacks is governed by the induced information rate and dispersion under the operational setup—not by unconstrained channel capacity. By separating observation noise from victim-induced inputs and by grounding predictions in finite-blocklength hypothesis-testing theory, we obtain practical, testable predictors of success. Our measurements across CPUs and attacks show strong empirical support under explicit assumptions, providing a principled foundation for quantitative side-channel risk assessment.
> Note: Summary: Measure the channel (Y|X), model the victim-induced inputs (X|S), estimate μ and σ^2, and use finite-blocklength predictors to forecast required traces. This yields realistic, testable risk assessments.

## Reproducibility
We provide code and data: channel estimators, diagnostics, predictors, and attack implementations, along with scripts to reproduce all figures and tables across platforms and noise regimes.
> Note: Availability of artifacts aids independent verification and extension.
