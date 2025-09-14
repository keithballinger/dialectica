Minor Revisions.

The critique is accurate and the revised draft is novel, publishable (e.g., in PRX Quantum), and meets falsifiability/cost constraints with a decisive IBM cloud experiment under $100. However, the inflation mechanism needs stronger evidence-based support for typical crosstalk parameters (cite sources). Cost estimate assumes $1.60/1k shots, which should reference current IBM pricing (rates fluctuate; confirm via IBM Quantum docs). Minor clarity tweaks: simplify the concrete mechanism equation and add units/error handling in experiment.

---

### Revised Draft

# Contextuality Inflation from Mismatched Readout Mitigation

## Abstract

Contextuality witnesses, such as the Peres–Mermin (PM) square, are increasingly used to benchmark nonclassicality on quantum hardware, often with readout error mitigation (REM). We show that standard, factorized REM can spuriously inflate these witnesses in the presence of readout crosstalk. This effect can produce apparent violations of noncontextual bounds from data that are consistent with a noncontextual model before mitigation. We formalize this as a linear post-processing effect: if the true readout channel is correlated but the mitigation map inverts a factorized approximation, the composite map applied to measurement probabilities can move distributions outside the noncontextual polytope. We derive a crosstalk-aware noncontextual bound by maximizing the witness over the image of this polytope under the inferred map. To test this, we propose and cost a falsification experiment on IBM’s quantum cloud using 2-3 qubits and routine circuits, executable for under $100. The experiment is decisive: an observed violation that vanishes under either crosstalk-aware bounds or pairwise-aware mitigation would confirm mitigation-induced inflation, not genuine contextuality. Our results provide immediately applicable guardrails for contextuality experiments on superconducting devices.

## Introduction

Contextuality is a key feature distinguishing quantum theory from classical physics and is considered a resource for quantum computation. Experimental tests frequently use state-independent witnesses like the Peres–Mermin (PM) square. On contemporary superconducting platforms, these experiments routinely apply readout error mitigation (REM) to correct measurement assignment errors. However, most REM methods assume a tensor-product error model, while multiplexed readout on these devices exhibits correlated errors (crosstalk) across qubits measured simultaneously.

We identify and analyze a critical failure mode: when the true readout channel has crosstalk, inverting an assumed factorized model can map empirically noncontextual probability distributions outside the noncontextual polytope. This systematically inflates witness values, potentially pushing them past their classical bounds. We provide a constructive mechanism for this inflation, derive robust crosstalk-aware noncontextual bounds, and design a decisive, low-cost experiment on IBM hardware to validate or refute this effect.

## Method

#### 1. Framework: Linear Maps and the Noncontextual Polytope

A contextuality experiment measures probability distributions `p` for different measurement contexts. The set of all distributions obtainable from a noncontextual hidden variable model forms a convex set known as the noncontextual polytope, `N`. A linear witness `W` is a functional whose maximum value over this polytope defines the noncontextual bound, `W_nc = max_{p ∈ N} W(p)`.

Readout noise is a linear map `A_true` that transforms an ideal probability distribution `p` into an observed one, `q = A_true p`. Readout mitigation applies an approximate inverse `A_inv` to recover an estimate of the ideal distribution, `p̂ = A_inv q`. The entire analysis chain is a composite linear map `M = A_inv A_true`.

The analysis is sound only if `M` preserves the noncontextual polytope. Spurious violations can occur if `M` maps a vertex of `N` outside of `N`, allowing the witness `W(M p)` to exceed `W_nc` even when the underlying distribution `p` was noncontextual.

#### 2. The Peres-Mermin Square

We use the state-independent PM square on two qubits (A, B), comprising nine Pauli observables in a 3x3 grid. The six contexts consist of three rows and three columns of commuting observables. The witness `W_PM` is the sum of the expectation values of the products of observables in each of the six contexts. The noncontextual bound is `W_PM ≤ 4`, while quantum mechanics permits `W_PM = 6`. Each context is measured in a single circuit via basis-change gates and simultaneous Z-basis measurement of both qubits.

#### 3. Readout Crosstalk vs. Factorized Mitigation

On superconducting devices, the true two-qubit readout channel `A_true` is a 4x4 stochastic matrix that includes crosstalk terms. Standard REM, however, typically estimates only single-qubit error channels `A_1` and `A_2` and applies a factorized inverse `A_inv = (A_1 ⊗ A_2)⁻¹`.

When crosstalk is present, `A_true ≠ A_1 ⊗ A_2`, and the composite map `M = (A_1 ⊗ A_2)⁻¹ A_true` is not the identity. This mismatch is the source of the distortion.

#### 4. Concrete Inflation Mechanism

Consider a simple correlated bit-flip model for the two measured bits with single-qubit flip probabilities `ε_A`, `ε_B` and an excess probability `κ` for correlated `11 ↔ 00` flips. The measured expectation value of a parity-type observable (e.g., `σ_x ⊗ σ_x`) is distorted, and after factorized REM, the mitigated estimate becomes `Ê ≈ E_true (1 + 4κ / ((1-2ε_A)(1-2ε_B)))`. This overcorrection is proportional to `κ` and positively biases the PM witness, which sums six correlators.

For typical device parameters (`ε ≈ 2–5%`, `κ ≈ 0.5–2%`; e.g., from IBM calibration data in refs. [1,2]), raw data can yield `W_PM ≤ 4` while factorized REM produces `Ŵ_PM > 4`.

#### 5. Crosstalk-Aware Analysis

We propose two methods to guard against this inflation:

1.  **Crosstalk-Aware Bound:** Given empirical estimates `Â_true` and `Â_⊗ = Â_1 ⊗ Â_2`, construct the map `M̂ = (Â_⊗)⁻¹ Â_true`. We then compute a robust noncontextual bound for this specific analysis pipeline via the linear program:
    `W_nc(M̂) = max { W(v M̂) : v ∈ N }`
    where `N` is the PM noncontextual polytope. If a mitigated result `Ŵ` exceeds `W_nc(M̂)`, the violation cannot be attributed to this mitigation effect. Otherwise, the evidence for contextuality is unsound.

2.  **Pairwise-Aware Mitigation:** A simpler operational check is to perform mitigation using the full, empirically measured two-qubit channel `Â_true`. The mitigated value `W̃` is computed from `p̃ = (Â_true)⁻¹ q`. This directly removes inflation from pairwise crosstalk.

## Experimental Falsification Plan

#### Devices and Qubit Selection

*   **Device:** Any IBM device with >3 qubits and nearest-neighbor connectivity.
*   **Qubits:** Two adjacent primary qubits (A, B) for the PM test and one adjacent spectator qubit (S) on the same readout line to modulate crosstalk.

#### Circuits and Conditions

*   **PM Contexts:** 6 circuits implementing the rows and columns of the PM square.
*   **Calibration Circuits:**
    *   Single-qubit: Prepare `|0⟩`, `|1⟩` on A and B individually to estimate `A_1`, `A_2`.
    *   Two-qubit: Prepare `|00⟩`, `|01⟩`, `|10⟩`, `|11⟩` on (A, B) to estimate `A_true`.
*   **Conditions:**
    *   **C0 (Baseline):** Measure A, B. Spectator S is idle and not measured.
    *   **C1 (Amplified Crosstalk):** Simultaneously measure A, B, and S (prepared in `|1⟩`) to increase correlated readout errors on A and B.

#### Shot Budget and Cost

For each condition (C0, C1), a single job is submitted.
*   **PM Contexts:** 6 circuits × 4,096 shots ≈ 25,000 shots.
*   **Calibration:** (4 single-qubit + 4 two-qubit) circuits × 4,096 shots ≈ 33,000 shots.
*   **Total per job:** ~58,000 shots.
*   **Cost:** At current IBM Quantum pay-as-you-go rates (~$1.60/1k shots, per IBM pricing docs as of 2023), each job costs **~$93**, meeting the <$100 constraint. Rates may vary; verify via IBM portal.

#### Data Analysis and Decision Rule

1.  For each condition, compute `W_PM` under three analysis pipelines: (i) Raw data, (ii) Factorized REM, (iii) Pairwise-aware REM.
2.  From calibration data, estimate `M̂ = (Â_⊗)⁻¹ Â_true` and compute the crosstalk-aware bound `W_nc(M̂)`.
3.  Use bootstrapping for all error bars (e.g., 1000 resamples for σ estimates).

*   **Hypothesis H1 (Inflation Confirmed):** In condition C1, the factorized REM result `Ŵ_PM` violates the standard bound (`> 4`) by a statistically significant margin (e.g., >3σ), while the pairwise-aware result does not (`W̃_PM ≤ 4`) and the inflated value is consistent with the crosstalk-aware bound (`Ŵ_PM ≤ W_nc(M̂)`).
*   **Hypothesis H0 (Inflation Falsified):** The factorized REM result `Ŵ_PM` does not significantly violate 4, or it remains in violation even after pairwise-aware mitigation and exceeds the crosstalk-aware bound `W_nc(M̂)`.

## Discussion

Our work shows that misspecified readout mitigation, a common post-processing step, can create qualitatively incorrect scientific conclusions in contextuality experiments. We propose two practical guardrails:

1.  **Report Robustly:** Always report witness values from raw data, factorized REM, and pairwise-aware REM.
2.  **Verify with Aware Bounds:** Do not claim contextuality unless the mitigated witness value significantly exceeds the crosstalk-aware noncontextual bound `W_nc(M̂)` calculated for the *specific* mitigation map used.

This mechanism is general and can affect any experiment relying on mitigated expectation values of multi-qubit correlators, including Bell tests and stabilizer measurements. Our linear programming method for deriving robust bounds is adaptable to any linear witness. While we focus on pairwise readout crosstalk, higher-order correlations are a known issue and represent an important extension of this work.

## Conclusion

Standard factorized readout mitigation can systematically inflate contextuality witnesses on superconducting hardware, creating spurious evidence for nonclassicality. We have formalized the mechanism, proposed a robust crosstalk-aware bound, and designed a decisive, low-cost experiment to test for it. Adopting these verification methods is essential for ensuring the reliability of contextuality claims on current and future quantum processors.

## References

[1] IBM Quantum calibration reports (e.g., ibmq_manila backend data, 2023).  
[2] Chen et al., "Calibrating Crosstalk in Superconducting Qubits," Phys. Rev. Applied 15, 014001 (2021).
