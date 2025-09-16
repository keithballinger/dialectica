Minor Revisions

Brief critique
- Overall: Strong, novel, falsifiable, and cost-feasible. The physical mechanism and falsification plan are clear and appropriate for IBM heavy-hex devices. The paper likely meets the “leading subfield journal” bar with small improvements.
- Key refinements needed:
  - Concurrency enforcement: Specify exactly how spectator pulses will be synchronized with the CR window in the IBM/Qiskit stack. Without explicit scheduling guidance, overlap may be inconsistent and dilute the effect.
  - Statistical power and drift control: Increase seeds per length and adopt a round-robin interleaving scheme across (reference, c→t, t→c) and loads at the granularity of (length, seed) to suppress drift. Still fits within the <$1000 constraint.
  - Locality/negative control: Define selection to avoid shared control lines or known long-range microwave crosstalk. State that N₂ excludes c and t.
  - Modeling and analysis: Pre-register model comparison (constant vs linear vs quadratic) and adopt mixed-effects across pairs to assess generality. Use binomial-weighted fits or bootstrap to provide robust CIs.
  - Confounders: Note that measurement crosstalk and uncorrelated spectator errors should largely cancel in the asymmetry difference; state this explicitly.
  - Minor: Tighten notation, clarify sampling of S(n) (with/without replacement across sequences), and provide a brief verification step to confirm pulse overlap via schedule inspection.

Revised draft (Markdown)

Title: A Spectator-Activation Law for Directionality-Asymmetric CX Errors on Fixed-Frequency Superconducting Qubits

Abstract
Two-qubit cross-resonance (CR) gates on fixed-frequency transmons exhibit direction-dependent error rates. We posit a simple, falsifiable law: the error-per-Clifford (EPC) asymmetry between CX(control→target) and CX(target→control) increases approximately linearly with the number of actively driven spectator qubits within two hardware-graph edges of the gate pair. The mechanism combines microwave cross-drive and residual ZZ interactions that are imperfectly refocused by echoed-CR under concurrent spectator activity. We formalize a signed asymmetry metric, derive a first-order additive model with geometry-dependent weights, and design an interleaved randomized benchmarking (RB) protocol executable on IBM heavy-hex devices today for <$500 per run. The law predicts a positive, localized slope of EPC asymmetry versus spectator count that saturates at high loads. A flat, negative, or nonlocal trend falsifies the law.

Introduction
Cross-resonance CNOTs on fixed-frequency transmons show orientation-dependent errors even with echo sequences suppressing leading unwanted terms [refs]. Concurrent activity on nearby qubits modifies CR performance via residual ZZ, dynamic Stark shifts, and microwave cross-drive [refs], but a quantitative, geometry-aware rule linking spectator activity to directional asymmetry is lacking. We introduce a spectator-activation law predicting how EPC asymmetry scales with the number and proximity of actively driven spectators, and we provide a cloud-executable RB protocol for falsification.

Method

Definitions
- CX orientation: CX(c→t) denotes an echoed-CR CNOT with control c and target t; CX(t→c) is the reverse.
- Two-edge spectators: N₂(c,t) is the set of physical qubits at graph distance ≤2 from either c or t, excluding c and t.
- Spectator activation: A spectator is activated if it receives single-qubit Clifford layers scheduled to overlap the CX pulse window. Activation is randomized across circuits and sequences; spectators not selected receive identity/delay to maintain timing.
- Asymmetry metric (primary endpoint): A(n) = EPC(c→t, n) − EPC(t→c, n), where n is the number of activated spectators sampled from N₂(c,t).

Mechanism and first-order model
During echoed-CR, the effective Hamiltonian includes the desired ZX and residual terms (IX, IZ, ZI, ZZ) [refs]. Spectator activity modulates these residuals through:
- Cross-drive: drive on c at ωt leaks onto neighbors via control lines.
- Residual ZZ: static couplings to spectators shift c and t in a state-dependent fashion.
- Dynamic Stark shifts: spectator drives alter detunings experienced by c and t.

Assuming weak, approximately independent spectator couplings, the EPC contributions add to first order:
EPC(c→t, n) ≈ EPC₀(c→t) + Σs∈S(n) ws(c→t)
and similarly for EPC(t→c, n). Thus
A(n) = A₀ + Σs∈S(n) [ws(c→t) − ws(t→c)].
If spectators are uniformly sampled from N₂ and effects are stationary, the expectation is linear:
E[A(n)] ≈ A₀ + β n, with β = E_s[ws(c→t) − ws(t→c)]. We predict β > 0 because residuals are typically larger when emanating from the driven control. Weights decay rapidly with graph distance, motivating restriction to N₂; we anticipate saturation at large n.

Predictions
1) Positive slope: E[A(n)] grows linearly for small n with β > 0.
2) Locality: Activating spectators at distance ≥3 yields slope consistent with zero within uncertainty.
3) Orientation sensitivity: |β| correlates with the driven qubit’s connectivity and coupling strengths.
4) Concurrency specificity: Effects maximize when spectator pulses overlap the CX pulse window.

Experiments (falsification plan)

Devices and pairs
- Platform: IBM heavy-hex processors (e.g., Heron/Eagle families).
- Selection: 3–5 pairs (c,t) with ≥4 candidate spectators in N₂(c,t). Avoid qubits sharing known microwave control lines with distant “negative-control” spectators when testing locality.

Spectator loads and sampling
- Loads: n ∈ {0,1,2,3,…} up to |N₂(c,t)|.
- For each RB sequence, choose S(n) uniformly at random from N₂(c,t) without replacement (per sequence). Resample S(n) independently across sequences. Non-selected spectators receive identity/delay to preserve alignment.

Enforcing temporal overlap (IBM/Qiskit)
- Circuit-level option: Place spectator 1Q Clifford layers in the same layer as the target CX using barriers around a CX+spectator block; transpile with scheduling_method='asap' or 'alap' and optimization_level=0–1 to preserve structure.
- Verification: For each configuration, obtain the pulse schedule (qiskit.transpile(..., scheduling_method=...), then qiskit.scheduler or target backend schedule inspection) and confirm that spectator pulses overlap the CX ECR window.
- Pulse-level fallback (if needed): On backends with pulse access, use Qiskit Pulse to build a block that plays the backend-calibrated CX schedule while concurrently playing randomized 1Q calibrated pulses on spectators, tiled to fill the CX duration. Validate amplitudes and timing against backend calibrations.

RB protocol
- For each load n, collect:
  1) Reference 2Q Clifford RB on (c,t) with spectator load n (shared across directions).
  2) Interleaved RB with CX(c→t).
  3) Interleaved RB with CX(t→c).
- Sequence lengths: L ∈ {2,4,8,16,32,64}.
- Seeds: 8–12 random seeds per L (increased from 4–5 for precision; still <$1000).
- Shots: 1024 per circuit.
- Drift mitigation: Use block-randomized, round-robin submission at the granularity of (n, L, seed) cycling through [reference, c→t, t→c], and shuffle across n. Repeat the cycle to spread any slow drift uniformly.

Negative control (locality)
- Repeat the protocol with spectators chosen only at distance ≥3 (disjoint from N₂). Prefer qubits on different control lines or frequency groups when possible.

Cost accounting (per pair)
- Example: 4 loads × 3 sets × 6 lengths × 8 seeds = 576 circuits; total shots ≈ 590k. Current IBM pricing yields ~$200–$700, under the $1000/run cap. Fewer/more seeds can trade cost for precision.

Data analysis
- For each (n, dir), fit survival P(L) = A p^L + B using binomial-weighted nonlinear least squares or maximum likelihood. Obtain p_ref(n) and p_dir(n).
- EPC (2Q IRB): EPC_dir(n) = (3/4) [1 − p_dir(n)/p_ref(n)].
- Asymmetry: A(n) = EPC_c→t(n) − EPC_t→c(n).
- Modeling:
  - Per-pair linear model: A(n) = A₀ + β n; test H0: β = 0 (two-sided).
  - Model comparison: evaluate constant vs linear vs quadratic via AIC/BIC; report if quadratic term is needed (saturation).
  - Across pairs: fit a mixed-effects model with random intercepts and slopes: A(n) = (A₀ + u_pair) + (β + v_pair) n + ε, and test β > 0 (one-sided) and β ≠ 0 (two-sided). Alternatively, meta-analyze pairwise β estimates.
- Uncertainty: report 95% CIs via parametric or bootstrap resampling over seeds; include cluster-robust SEs across seeds within pairs.
- Power check (pre-run): using pilot data (n=0,1), estimate Var[A(n)] to confirm ability to detect |β| ≥ 0.002–0.005 EPC/activated spectator with the chosen seeds.

Falsification criteria (pre-registered)
- Support: β > 0 with p < 0.01 for the fixed-effect across pairs; majority (>70%) of pairs show positive β with CIs excluding zero; negative-control slope consistent with zero within 95% CI.
- Falsify: β ≤ 0 or indistinguishable from zero across pairs; or negative control yields significant slope inconsistent with locality (after accounting for known long-range crosstalk).

Confounders and mitigations
- Gate-dependent noise in IRB: Using a per-load reference partially cancels spectator-induced baseline effects; asymmetry further cancels orientation-independent errors.
- Measurement crosstalk: Expected to be largely orientation-independent and thus cancel in A(n).
- Calibration drift: Controlled by round-robin interleaving and randomization across loads/directions; residual drift assessed by time-stamp covariates in the mixed-effects model.
- Spectator compilation artifacts: Verification of pulse overlap is required; exclude datasets failing overlap checks.

Implications
A validated law yields a simple concurrency-aware cost in compilers: penalize activating spectators within N₂ of the driven control. It also motivates echo and CR calibration strategies robust to spectator-induced detunings and provides a practical diagnostic for crosstalk localization.

Limitations
- Linearity holds for small n; saturation or non-additivity may appear at high loads. Model comparison addresses this.
- β is device- and calibration-specific; the claim is about trend (sign/locality), not a universal coefficient.
- IRB reports average EPC; coherent error structure is not fully resolved, though the asymmetry metric targets the directional component of interest.

Conclusion
The spectator-activation law predicts that CX EPC asymmetry grows linearly with the number of active spectators within two hardware edges, driven by cross-drive and residual ZZ under concurrency. The proposed RB protocol on IBM devices can falsify this today within a modest budget. Confirmation would provide an actionable rule for concurrency-aware scheduling and hardware-aware control; a null or inverted result would refute the law and redirect modeling of CR asymmetries.

Reproducibility
We will release Qiskit scripts to: (i) select N₂(c,t), (ii) construct overlap-verified circuits/schedules, (iii) execute round-robin jobs, and (iv) perform weighted RB fits and mixed-effects analysis. References to standard CR physics and RB methodology will be included in the final version.
