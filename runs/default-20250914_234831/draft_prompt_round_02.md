You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/quantum_ibm_cost.md

- Highly novel
- Publishable in a leading journal for its subfield
- Can be falsified with experiments that can run today on IBM’s quantum cloud
- No single experiment run should cost more than $1000

        Draft:
        Title: A Spectator-Activation Law for Directionality-Asymmetric CX Errors on Fixed-Frequency Superconducting Qubits

Abstract
Two-qubit cross-resonance (CR) gates on fixed-frequency transmons exhibit direction-dependent error rates. We propose and operationalize a simple, falsifiable law: the asymmetry between CX control→target and target→control error per Clifford (EPC) increases approximately linearly with the number of actively driven spectator qubits within two edges of the gate pair. The mechanism combines cross-drive amplitude leakage and residual ZZ interactions that are incompletely refocused by echoed-CR in the presence of concurrent spectator activity. We formalize the asymmetry metric, derive a first-order additive model with geometry-dependent weights, and design an interleaved randomized benchmarking (RB) protocol that can be executed today on IBM heavy-hex devices for under $100–$500 per run. The law predicts a positive, roughly linear trend of EPC asymmetry with spectator count that saturates at large loads and is strongly localized to spectators within two hardware edges. A flat trend or a negligible slope falsifies the law.

Introduction
Two-qubit gates realized via cross-resonance on superconducting transmon platforms are known to be directionally asymmetric: switching which qubit is driven and which is conditioned often changes error rates, even after echo sequences suppress dominant unwanted terms. Concurrency on nearby qubits can further modulate CR Hamiltonian terms via residual ZZ, dynamic Stark shifts, and cross-drive, but a quantitative, geometry-aware rule tying concurrent spectator activity to directional asymmetry has not been established. We introduce a spectator-activation law that predicts how the EPC asymmetry scales with the number and proximity of actively driven spectators. The law is experimentally testable with standard RB on IBM’s cloud backends and offers immediate value for concurrency-aware compilation and scheduling.

Method
Definitions
- CX orientation: CX(c→t) denotes an echoed-CR CNOT compiled with control c and target t. The reverse is CX(t→c).
- Two-edge spectators: The set N2(c,t) comprises all qubits with hardware graph distance ≤ 2 from either c or t on the coupling map.
- Spectator activation: A spectator is “activated” if it receives single-qubit Clifford layers that temporally overlap the CX pulses. Activation is randomized across sequences to avoid coherent bias.
- Asymmetry metric: A(n) = EPC(c→t, n) − EPC(t→c, n), where n is the number of activated spectators selected from N2(c,t). We report both signed and absolute asymmetry |A(n)|.

Mechanism and first-order model
During CR, the effective Hamiltonian includes desired ZX and spurious terms (IX, IZ, ZI, ZZ, XX, XZ, etc.). Echo sequences reduce but do not eliminate the spurious terms, and their residuals depend on:
- Cross-drive: drive on the control at the target’s frequency produces unintended drive on neighbors with coupling-dependent weights.
- Residual ZZ: static couplings with spectators cause state- and activity-dependent Z shifts and conditional phases on c and t.
- Dynamic Stark and detuning shifts: concurrent 1Q drives modify effective detunings, altering CR calibration, especially asymmetrically with respect to which qubit is driven.

For weak, approximately independent spectator couplings, the gate infidelity contribution from spectator s adds to first order:
EPC(c→t, n) ≈ EPC0(c→t) + Σ_s∈S(n) w_s(c→t),
and similarly for EPC(t→c, n). The directional asymmetry then obeys the spectator-activation law:
A(n) = A0 + Σ_s∈S(n) [w_s(c→t) − w_s(t→c)],
with A0 the idle-load asymmetry and S(n) the activated spectator set. In the mean-field limit with random spectator choice (size n) from N2 and weights roughly stationary,
E[A(n)] ≈ A0 + β n, with β = E_s[w_s(c→t) − w_s(t→c)] > 0.
Weights w_s decay rapidly with graph distance and coupling strength; thus restricting to spectators within two edges captures the dominant effect. At large n, higher-order terms (e.g., detuning-induced calibration breakdown) produce saturation or slight curvature.

Predictions
- Positive slope: E[A(n)] increases linearly for small n with slope β > 0.
- Locality: Activating spectators outside N2 yields near-zero slope.
- Orientation sensitivity: β scales with the cross-drive matrix elements and residual ZZ on the driven qubit; pairs with more/larger couplings from the driven node show larger β.
- Concurrency specificity: The effect strengthens when spectator pulses overlap the CR windows and weakens when out of phase.

Experiments (falsification plan)
Devices and pairs
- Target platform: IBM heavy-hex processors (e.g., Heron/Eagle families).
- Select 3–5 qubit pairs (c,t) with at least 3–5 spectators in N2(c,t).

Spectator sets and loads
- Determine N2(c,t). Define loads n ∈ {0, 1, 2, 3} (or up to the available spectators).
- For each sequence seed, sample a random subset S(n) ⊆ N2 of size n. Re-sample across seeds to average over spectator identity.

Activation protocol
- For spectators in S(n), apply random 1Q Cliffords synchronized so their pulses overlap the CX slots (match the CR sub-schedule in the backend’s timing). For spectators not in S(n), idle identity scheduling preserves timing without drives.
- Optional negative control: Activate n spectators at distance ≥ 3 from both c and t with the same timing.

RB protocol
- Use interleaved two-qubit RB on the pair (c,t).
- For each load n, collect:
  - Reference 2Q Clifford RB (shared for both directions).
  - Interleaved RB with CX(c→t).
  - Interleaved RB with CX(t→c).
- Sequence lengths: L ∈ {2, 4, 8, 16, 32, 64}.
- Seeds: 4 per set. Total per set: 24 circuits.
- Shots: 1000 per circuit.

Shot and cost accounting (per pair)
- Loads: 4 (n = 0,1,2,3).
- Sets per load: 3 (reference + two interleaved) = 12 sets.
- Circuits: 12 sets × 24 = 288 circuits.
- Shots: 288,000 per pair.
- On current IBM pricing, this is significantly under $1000, typically $100–$300 depending on queue and device; feasible in a single session.

Data analysis
- Fit survival probabilities vs length to extract decay parameters p_ref(n), p_c→t(n), p_t→c(n).
- Interleaved EPC per orientation: r_dir(n) ≈ [(1 − p_dir(n)/p_ref(n))] × (d − 1)/d for 2Q RB with d = 4, giving r_dir(n) = (3/4)(1 − p_dir/p_ref).
- Asymmetry: A(n) = r_c→t(n) − r_t→c(n). Also report |A(n)|.
- Linear model: Fit A(n) = A0 + β n over n = 0..3. Test H0: β = 0 (two-sided) via ordinary least squares with robust errors. Report slope, 95% CI, and R^2.
- Locality test: Repeat with distance ≥ 3 spectators; expect β ≈ 0.
- Orientation mechanism: Correlate β with degree and calibrated ZZ estimates around the driven qubit; expect larger β with higher local coupling.

Falsification criteria
- The law is supported if β > 0 with p < 0.01 and effect size exceeding 1–2 standard errors across at least two pairs, and the distance ≥ 3 control yields β statistically indistinguishable from zero.
- The law is falsified if β ≈ 0 (or negative) across loads and pairs, or if the effect is non-local (β similar for distance ≥ 3).

Discussion
Implications
- Concurrency-aware compilation: Schedulers should penalize activating spectators in N2 of the driven qubit during CX to limit asymmetry growth. The linear model provides a simple cost term for timing and placement.
- Calibration and echo design: Echo phases and CR amplitude calibration could be optimized under loaded conditions to reduce β, trading off idle-load performance for realistic multi-qubit activity.
- Predictive mapping: Measuring β per pair yields a coarse crosstalk map that can inform layout selection and gate-time staggering.

Mechanistic interpretation
The observed slope β arises from additive first-order contributions of spectators to residual IZ/ZI/ZZ and drive detuning during CR. Because CR direction selects which qubit is driven and which local couplings are most engaged, the imbalance in these additive terms manifests as a direction-dependent increment that grows with the number of active, proximate spectators. The two-edge cutoff reflects the rapid attenuation of both static ZZ and microwave cross-drive in heavy-hex layouts.

Practical considerations
- Using shared references keeps cost low while maintaining valid interleaved RB inference.
- Overlapping spectator pulses with CR windows is essential; asynchronous activation should strongly reduce the effect, serving as an internal check.
- Averaging over spectator identities via random subsets makes the estimated β robust to specific local idiosyncrasies.

Limitations
- RB assumptions: Interleaved RB provides an average EPC; coherent and context-dependent errors may not be fully captured. However, the trend in EPC asymmetry under controlled concurrency is the primary object of interest.
- Load saturation: At high n, nonlinearity can appear from compounded detuning and pulse distortions; we constrain to n ≤ 3–5.
- Device specificity: Results generalize to fixed-frequency transmon CR platforms but quantitative β will vary across generations and calibrations.
- Drift: Time-dependent calibration drift can bias slopes. Mitigate by interleaving load conditions and directions in the submission order and by using shared references per load.

Conclusion
We propose and operationalize a spectator-activation law for directionality-asymmetric CX errors on fixed-frequency superconducting qubits: the EPC asymmetry between CX orientations grows approximately linearly with the number of actively driven spectators within two hardware edges. The law is mechanistically grounded in residual ZZ and cross-drive under echoed-CR, yields concrete, locality-sensitive predictions, and is falsifiable with today’s IBM cloud hardware via a modest-cost interleaved RB campaign. Validating or refuting this law will directly inform concurrency-aware compilation, calibration under load, and crosstalk-aware architecture evaluation.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — OMIT this section entirely if your judgment is Publish
