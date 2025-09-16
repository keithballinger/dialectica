Title: A Capacity Law for Cache Side-Channel Key Recovery

Abstract
We propose a capacity law for last-level cache (LLC) timing side channels: for any fixed leakage model with secret-dependent memory accesses, the success probability of key recovery is asymptotically determined by the measured information-carrying capacity of the microarchitectural channel, which depends only on noise and eviction dynamics. We formalize cache leakage as a discrete memoryless channel per probe, prove finite-sample upper and lower bounds on attack success in terms of measured mutual information, and show that the transition from failure to success occurs at a rate threshold equal to channel capacity. We provide an attacker-agnostic measurement protocol to estimate the channel law and compute capacity without accessing victim internals. A falsification plan across multiple architectures and attack styles tests whether measured capacity alone predicts success rates for diverse victim implementations sharing the same leakage model. This framework converts qualitative vulnerability assessments into quantitative, testable predictions.

Introduction
Cache timing side-channel attacks remain a practical threat, yet predicting exploitability from system measurements is ad hoc. Practitioners rely on qualitative heuristics (e.g., “noisy but exploitable”) or attack-specific empirical tuning. Information-theoretic analyses exist for side channels, but they rarely yield operational guidance that maps lab measurements to key-recovery success probabilities across implementations.

We claim a simple law: given a fixed leakage model in which the victim’s memory accesses depend on a secret, the success of last-level cache attacks is determined by the information capacity of the cache-timing channel, measurable via controlled stimulus–response experiments performed by an external observer. Specifically:
- The per-probe leakage can be modeled as a discrete memoryless channel Y|X parameterized by hardware noise and eviction dynamics.
- The viability of key recovery over N probes depends on the code rate R = b/N, where b is the number of secret bits to be recovered, relative to the channel capacity C (bits per probe).
- If R < C, success probability approaches 1 with an error exponent depending only on the measured channel; if R > C, success probability approaches 0; at finite N, success is bounded above and below by functions of the measured mutual information and error exponents.

This capacity law abstracts away idiosyncrasies of victim code as long as the leakage model (mapping secret to addresses) is fixed and exhibits typicality (roughly, that different secrets induce address sequences whose statistical separation is comparable to random codes). As a result, a single measured C predicts attack performance across multiple implementations of the same leakage model (e.g., table-based AES, RSA square-and-multiply, or lookup-heavy ML inference) and across attack styles that read out the same underlying eviction signal (Prime+Probe, Flush+Reload, Evict+Time).

Method
Threat and leakage models
- Attacker: co-resident process capable of performing standard cache probing (e.g., Prime+Probe or Flush+Reload) and recording timing outcomes Y across repeated victim operations; no access to victim internals beyond public inputs and a trigger to cause encryptions/signatures.
- Victim leakage model: the victim maps secret S and public input M to a sequence of virtual addresses A = f(S, M) that resolve to LLC sets/lines X. We consider settings where the attacker’s probing signal responds to X but not to other internal state.
- Channel: each probe yields an output Y (binary hit/miss, multi-level latency, or eviction count) drawn from P(Y|X) governed by microarchitectural noise (co-tenant activity, prefetchers, replacement policy stochasticity, TLB/page coloring, frequency scaling) and measurement noise. Across probes separated by at least a coherence gap τ (chosen empirically), we approximate independence (memoryless channel).

Channel estimation and capacity
- Estimation: the attacker estimates P̂(Y|X) via stimulus–response experiments in which they:
  1) Prepare cache state to encode input X (e.g., prime selected sets).
  2) Induce a known victim or synthetic access that realizes X deterministically.
  3) Measure Y via probing (latency, hits/misses).
  4) Repeat to obtain empirical conditional distributions.
- Capacity computation: compute mutual information I(P_X, P̂(Y|X)) for empirical input distributions P_X matching the leakage model’s support. Capacity C is the supremum over P_X on this support. Practically, use Blahut–Arimoto on P̂(Y|X) to obtain C (bits per probe).

Operational rate and effective uses
- Secret size: b = log2|S| (e.g., b = 128 for an AES round key).
- Number of independent channel uses: N = K/κ, where K is the number of raw probe measurements and κ ≥ 1 accounts for temporal correlation due to replacement dynamics and scheduler interference (estimated via autocorrelation of Y). We treat N as the effective sample size.
- Operational rate: R = b/N bits per probe.

Capacity law: bounds and asymptotics
We state two finite-length bounds and an asymptotic threshold.

- Converse (universal lower bound on error via Fano): For any decoder,
  P_error ≥ 1 − (N I1 + 1)/b,
  where I1 = I(X;Y) under the actual input distribution induced by the leakage model. Since C ≥ I1 for any input distribution, a looser but universal bound is P_error ≥ 1 − (N C + 1)/b.

- Achievability (random-coding error exponent): For code rate R < C, there exist decoders (e.g., maximum-likelihood over key hypotheses) such that
  P_error ≤ exp(−N E_r(R)),
  where E_r(R) is Gallager’s random-coding error exponent computed from P̂(Y|X). For typical cryptographic leakage in which the mapping from secrets to address sequences is close to a random code on the channel support, the average-case performance over secrets closely follows this exponent.

- Threshold law (asymptotic): As N → ∞, success probability tends to 1 if R < C and to 0 if R > C. The “50% success” transition width is governed by 1/√N and the channel dispersion V, yielding finite-blocklength normal approximation:
  b ≈ N C − √(N V) Q^{-1}(P_succ) + o(√N),
  where V is computed from P̂(Y|X) and Q^{-1} is the Gaussian quantile. This provides a practical trace-complexity predictor.

Interpretation
- The only quantities entering the law are C, V, and κ, all measurable from P̂(Y|X) and Y’s autocorrelation; the victim code affects success only via the leakage model’s support (which sets the channel input alphabet) and typicality. Thus, different AES T-table implementations with distinct layouts but identical leakage support share the same predicted success curve once capacity is measured.

Mapping attacks to the same channel
- Prime+Probe, Flush+Reload, and Evict+Time differ in how the attacker excites and samples the channel but ultimately observe Y drawn from the same P(Y|X) up to monotone transformations (binary hit/miss or continuous latency). After discretization, the induced channel matrices are equivalent for capacity computation. Therefore, capacity measured by one probing style predicts success for others that read the same eviction signal.

Experiments (falsification plan)
Platforms
- CPUs: at least three microarchitectures each from Intel (Skylake, Ice Lake, Alder Lake), AMD (Zen2, Zen3, Zen4), and ARM (Neoverse N1, V1).
- OS: Linux with performance isolation toggles (isolcpus, governor), and noisy multi-tenant settings (co-runners generating memory traffic).

Leakage models and victims
- AES T-table implementations (32-bit and 64-bit tables), constant-time AES as a negative control.
- RSA square-and-multiply with sliding windows.
- Lookup-heavy ML inference kernels (embedding lookups).
For each, ensure secret-dependent addresses are the only intended source of leakage; keep public inputs randomized.

Channel measurement protocol
- Construct the input alphabet X as observed cache-set/line groups relevant to the leakage model (e.g., 64 sets corresponding to T-table index MSBs).
- For each X:
  1) Prime corresponding sets/lines.
  2) Trigger a controlled access to realize X; for table lookups, drive inputs that deterministically select X.
  3) Probe and record Y (latency or hit/miss).
  4) Repeat 10^5–10^6 times to estimate P̂(Y|X).
- Compute:
  - Capacity C via Blahut–Arimoto on P̂(Y|X).
  - Dispersion V and κ via temporal analysis of Y.
- Produce predicted trace complexity N* for 50% success for each b using the normal approximation; map to raw probes K* = κ N*.

Key-recovery attacks
- Implement standard, published attacks without capacity-aware tuning:
  - Prime+Probe on AES T-tables (L3 monitoring). Decoder: maximum-likelihood over key-byte hypotheses using per-set statistics.
  - Flush+Reload on shared libraries for AES and RSA. Decoder: same ML principle on line-level observations.
  - Evict+Time for RSA (timing of entire operation).
- For each platform and leakage model:
  - Collect K traces at multiple budgets around K* (e.g., 0.5×, 0.75×, 1×, 1.25×, 1.5×).
  - Record success probability over 100 independent runs per budget.

Evaluation criteria
- Primary falsification test: does the empirical success curve align with the predicted S-shaped curve centered at K* with width set by V and κ? Systematic deviations (success significantly better or worse than predicted across platforms or victims) falsify the law.
- Secondary tests:
  - Attack style invariance: capacity measured with one probing method predicts success with another method that senses the same sets/lines.
  - Victim implementation invariance: distinct T-table layouts but identical leakage support yield the same predicted curve after capacity measurement.
  - Noise sensitivity: co-run varying memory pressure; predicted K* should scale inversely with measured C as noise increases.

Discussion
Implications
- Standardized vulnerability metric: C (bits per probe) and κ provide an implementation-agnostic score for cache leakage exposure. Compared to ad hoc “signal-to-noise” ratios, capacity directly predicts traces required for key recovery.
- Defense evaluation: microarchitectural mitigations (way partitioning, random replacement, selective flush, timer fuzzing) can be compared by how much they reduce C and increase κ, yielding quantitative hardening targets.
- Attack design: capacity measurement identifies the best probing modality and sampling interval (minimizing κ) to maximize effective information rate.

Relation to prior work
- Information-theoretic approaches to side channels have analyzed leakage via mutual information, but typically stop at estimating I or provide asymptotic statements without operational predictions for key-recovery success. Our contribution is to (1) tie measured LLC dynamics to an explicit channel model; (2) use capacity, dispersion, and error exponents to predict concrete success curves; and (3) validate invariance across victims and attack styles sharing the leakage model.

Practical guidance
- To quickly assess risk: measure P̂(Y|X) for the relevant leakage support, compute C and κ, and derive K* ≈ κ b/C as the ballpark number of probes for 50% success. If K* is below realistic trace budgets for an attacker, the system is practically vulnerable.

Limitations
- Memorylessness and stationarity: Replacement policies, SMT interference, and OS scheduling can induce temporal dependence. We partially compensate by estimating κ and using block sampling, but strong nonstationarity can break the model.
- Leakage support fidelity: If the victim’s leakage support differs from the measurement support (e.g., due to unmodeled prefetching or aliasing), P̂(Y|X) may mis-specify the channel. Careful selection of X and validation via independence tests are required.
- Codebook atypicality: Some implementations might map secrets to address sequences with high collision or structure (e.g., masking, shuffling). In such cases, error exponents can differ from random-coding predictions. Our falsification plan explicitly probes this.
- Cross-core and frequency scaling effects: Dynamic frequency changes and thermal throttling alter latency distributions; capacity should be re-measured under the intended operating envelope.
- Multi-modal observations: Combining multiple sensors (e.g., L1 + LLC + branch predictor) yields a composite channel; the law extends, but measurement complexity grows.

Conclusion
We introduce a capacity law for cache timing attacks: with a fixed secret-dependent memory access leakage model, the key-recovery success probability is governed by the measured capacity of the cache-timing channel, a property of noise and eviction dynamics alone. By estimating the channel law and computing capacity, dispersion, and effective independence, we obtain quantitative, attack-agnostic predictions for traces-to-success and success curves that hold across microarchitectures and victim implementations sharing the leakage model. This transforms cache side-channel risk assessment from qualitative judgments to testable, quantitative forecasts and provides a principled basis for evaluating defenses and guiding attack feasibility studies.
