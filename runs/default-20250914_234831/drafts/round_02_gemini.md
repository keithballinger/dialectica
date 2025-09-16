Minor Revisions

The draft presents a novel, important, and well-defined hypothesis—the Spectator-Activation Law—that is directly falsifiable with a rigorous and cost-effective experimental plan. The strengths are its clarity, specificity, and the tight integration of a physical mechanism with a concrete verification protocol. The proposed work is of high quality and suitable for a leading journal.

The recommended revisions are minor and aim to enhance clarity and rigor:
1.  **Asymmetry Metric:** The primary model and prediction concern the *signed* asymmetry `A(n)`. The role of the absolute asymmetry `|A(n)|` is unclear and should be clarified or de-emphasized to focus the narrative.
2.  **Drift Mitigation:** The excellent point about mitigating calibration drift by interleaving experimental conditions is a critical part of the protocol and should be moved from the "Limitations" to the "Experiments" section.
3.  **Protocol Clarity:** Small details in the experimental protocol could be stated more clearly, such as the number of seeds per sequence length and the rationale for pulse synchronization.
4.  **Notation:** Notation for Error Per Clifford should be consistent throughout (e.g., use EPC instead of `r`).

---

### Revised Draft

**Title:** A Spectator-Activation Law for Directionality-Asymmetric CX Errors on Fixed-Frequency Superconducting Qubits

**Abstract**
Two-qubit cross-resonance (CR) gates on fixed-frequency transmons exhibit direction-dependent error rates. We propose and operationalize a simple, falsifiable law: the asymmetry between CX(control→target) and CX(target→control) error per Clifford (EPC) increases approximately linearly with the number of actively driven spectator qubits within two hardware graph edges of the gate pair. The proposed mechanism combines cross-drive amplitude leakage and residual ZZ interactions that are incompletely refocused by echoed-CR in the presence of concurrent spectator activity. We formalize the asymmetry metric, derive a first-order additive model with geometry-dependent weights, and design an interleaved randomized benchmarking (RB) protocol executable on IBM heavy-hex devices for under $500 per run. The law predicts a positive, roughly linear trend of EPC asymmetry with spectator count that saturates at high loads and is strongly localized. A flat or negligible trend falsifies the law.

**Introduction**
Two-qubit gates realized via cross-resonance on superconducting transmon platforms are known to be directionally asymmetric: switching which qubit is the control (driven) and which is the target often changes error rates, even after echo sequences suppress dominant unwanted Hamiltonian terms. Concurrency on nearby qubits can further modulate CR Hamiltonian terms via residual ZZ coupling, dynamic Stark shifts, and microwave cross-drive, but a quantitative, geometry-aware rule tying concurrent spectator activity to directional asymmetry has not been established. We introduce a spectator-activation law that predicts how the EPC asymmetry scales with the number and proximity of actively driven spectators. The law is experimentally testable with standard RB on IBM’s cloud backends and offers immediate value for concurrency-aware compilation and scheduling.

**Method**
**Definitions**
-   **CX orientation:** `CX(c→t)` denotes an echoed-CR CNOT with control `c` and target `t`. The reverse is `CX(t→c)`.
-   **Two-edge spectators:** The set `N₂(c,t)` comprises all qubits with hardware graph distance ≤ 2 from either `c` or `t`.
-   **Spectator activation:** A spectator is “activated” if it receives single-qubit Clifford layers that temporally overlap with the `CX` gate pulses. Activation is randomized across sequences to average over specific gate choices.
-   **Asymmetry metric:** `A(n) = EPC(c→t, n) − EPC(t→c, n)`, where `n` is the number of activated spectators selected from `N₂(c,t)`. This signed metric is the primary object of study.

**Mechanism and First-Order Model**
The effective Hamiltonian during an echoed-CR pulse contains the desired ZX term and spurious residuals (e.g., IX, IZ, ZI, ZZ). These residuals are modulated by:
-   **Cross-drive:** Drive on the control at the target’s frequency produces unintended drives on neighbors.
-   **Residual ZZ:** Static couplings with spectators cause state-dependent frequency shifts on `c` and `t`.
-   **Dynamic Stark shifts:** Concurrent single-qubit drives on spectators modify effective detunings, altering CR performance.

For weak, approximately independent spectator couplings, the EPC contribution from a spectator `s` is additive to first order:
`EPC(c→t, n) ≈ EPC₀(c→t) + Σ_{s∈S(n)} w_s(c→t)`,
and similarly for `EPC(t→c, n)`. The directional asymmetry then follows the spectator-activation law:
`A(n) = A₀ + Σ_{s∈S(n)} [w_s(c→t) − w_s(t→c)]`,
where `A₀` is the idle-load asymmetry and `S(n)` is the set of `n` activated spectators. Assuming spectators are chosen randomly from `N₂` and their mean-field effect is stationary, the expected asymmetry is linear in `n`:
`E[A(n)] ≈ A₀ + βn`, where the slope `β = E_s[w_s(c→t) − w_s(t→c)]` represents the average marginal increase in asymmetry per spectator. We predict `β > 0` because cross-drive and ZZ effects are typically stronger when emanating from the driven control qubit. Weights `w_s` decay rapidly with graph distance, justifying the restriction to `N₂`.

**Predictions**
1.  **Positive slope:** `E[A(n)]` increases linearly for small `n` with slope `β > 0`.
2.  **Locality:** Activating spectators outside `N₂` (e.g., at distance ≥ 3) yields a slope consistent with zero.
3.  **Orientation sensitivity:** The magnitude of `β` should correlate with hardware properties like the number and strength of couplings connected to the driven qubit `c`.
4.  **Concurrency specificity:** The effect is maximized when spectator pulses temporally overlap the CR pulse window.

**Experiments (Falsification Plan)**
**Devices and Qubit Pairs**
-   **Platform:** IBM heavy-hex processors (e.g., Heron/Eagle families).
-   **Selection:** 3–5 qubit pairs `(c,t)` with at least 4–5 spectators in `N₂(c,t)`.

**Protocol**
-   **Spectator loads:** For each pair, define loads `n ∈ {0, 1, 2, 3, ...}` up to the available number of spectators in `N₂`.
-   **Randomized activation:** For each RB sequence, a random subset `S(n) ⊆ N₂` of size `n` is chosen. Spectators in `S(n)` receive random single-qubit Clifford gates, while those not in `S(n)` receive identity gates to maintain circuit timing. Spectator pulses are synchronized to overlap with the two-qubit CX gate pulses, maximizing interaction.
-   **RB protocol:** Use standard interleaved two-qubit RB on the pair `(c,t)`. For each load `n`, collect data for three sets of circuits:
    1.  Reference 2Q Clifford RB (shared for both directions).
    2.  Interleaved RB with `CX(c→t)`.
    3.  Interleaved RB with `CX(t→c)`.
-   **Parameters:** Sequence lengths `L ∈ {2, 4, 8, 16, 32, 64}`; 4-5 random seeds per length; 1024 shots per circuit.
-   **Drift mitigation:** To mitigate calibration drift, the execution order of circuits should be randomized across different loads and gate directions.
-   **Negative control:** To test locality, the protocol is repeated for `n` spectators chosen from outside `N₂` (graph distance ≥ 3).

**Cost Accounting (per pair)**
-   For a typical experiment with 4 loads (`n=0,1,2,3`), 3 circuit sets per load, 6 sequence lengths, and 4 seeds: `4 × 3 × 6 × 4 = 288` circuits.
-   Total shots: `288 × 1024 ≈ 295,000`.
-   Estimated cost on current IBM hardware is $100–$500, well within the feasibility constraint.

**Data Analysis**
-   For each load `n` and direction `dir ∈ {c→t, t→c}`, fit the survival probability vs. sequence length `L` to an exponential `P(L) = A p^L + B` to extract the decay parameters `p_ref(n)` and `p_dir(n)`.
-   Calculate EPC: `EPC_dir(n) = (3/4)(1 − p_dir(n)/p_ref(n))`.
-   Calculate asymmetry: `A(n) = EPC_c→t(n) − EPC_t→c(n)`.
-   Fit the linear model `A(n) = A₀ + βn` using ordinary least squares. Test the null hypothesis `H₀: β = 0` (two-sided test).
-   Report the slope `β`, its 95% confidence interval, and `R²`.

**Falsification Criteria**
-   The law is **supported** if `β` is positive and statistically significant (e.g., p < 0.01) across multiple qubit pairs, and the negative control experiment on distant spectators yields a `β` statistically indistinguishable from zero.
-   The law is **falsified** if `β` is consistently indistinguishable from zero or is negative, or if the effect is non-local (i.e., the negative control yields a significant `β`).

**Discussion**
**Implications**
A validated law provides a quantitative basis for concurrency-aware compilation. Schedulers could use the linear model as a cost function to penalize activating spectators within `N₂` of a CX gate's control qubit. The findings would also inform the design of CR echo sequences, which could be optimized for robustness to spectator activity.

**Limitations**
-   Interleaved RB provides an average EPC. While it may not capture all coherent error dynamics, the *difference* in EPC is a robust first-order measure of the asymmetric error component that the law seeks to explain.
-   The linear model may break down at high spectator loads (`n`) where compounded detuning shifts cause non-additive effects. Our experiment focuses on the low-`n` regime where the linear approximation is most likely to hold.
-   The quantitative value of `β` is device- and calibration-specific, though the qualitative linear, local trend is expected to generalize across fixed-frequency transmon platforms.

**Conclusion**
We propose a spectator-activation law for CX error asymmetry: the EPC asymmetry grows linearly with the number of active spectators within two hardware edges. The law is grounded in the physics of cross-drive and residual ZZ coupling, yields falsifiable predictions, and can be tested today on cloud quantum hardware with a modest experimental budget. Validating this law would provide a simple, actionable rule for mitigating a key source of crosstalk error in superconducting quantum computers.
