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
        Minor Revisions

Title
Selection-rule–induced Z-parity bias in cross-resonance CNOT noise

Abstract
We predict and operationally define a universal Z-parity conservation bias in the two-qubit noise of echoed cross-resonance (CR) CNOT gates on fixed-frequency superconducting qubits. After Pauli twirling, the two-qubit error channel places strictly greater weight on even-parity Paulis (those commuting with ZZ) than on odd-parity Paulis (those anticommuting with ZZ). This follows from the structure of the driven-CR Hamiltonian, the symmetry of echoed sequences that cancel single-X/Y terms to first order, and the dominance of dephasing-like and ZZ-type correlated errors. We propose a simple, low-cost falsification experiment on IBM Quantum backends: measure the depth dependence of a Z-parity correlator under randomized brickwork CNOT layers with Clifford twirling, extracting the ZZ-eigenvalue s of the effective layer channel. Our theory predicts s > 0.86 on typical edges, implying a parity preservation excess R(d) = 0.5 s^d ≥ 0.02 at depth d ≈ 20. If R(20) < 0.02 on most edges, the hypothesis is falsified. The protocol fits within <$100 per job with 50 circuits × 10k shots. A verified bias has immediate implications for tailored error models, parity-aware compilation, and symmetry-based mitigation.

Introduction
Native two-qubit entangling gates on IBM’s fixed-frequency transmons are realized by the cross-resonance (CR) interaction, typically compiled into echoed CR (ECR) sequences to implement CNOTs. Despite steady fidelity improvements, predictive device-level noise structure remains essential for calibration, compilation, and error mitigation. Here we articulate a concrete, falsifiable hypothesis: two-qubit noise associated with CR-based CNOTs conserves Z-parity more often than it flips it.

We define Z-parity on two qubits as the eigenvalue of ZZ; more generally, on a register S, parity is the product ∏i∈S Zi. For a Pauli channel, error operators that commute with ZZ (even-parity: II, ZI, IZ, ZZ, XX, YY, XY, YX) preserve two-qubit Z-parity; those that anticommute (odd-parity: XI, IX, YI, IY, ZX, XZ, ZY, YZ) flip it. The claim is that, after standard echo/twirling, even-parity errors carry greater total weight.

We motivate this bias physically: (i) ECR sequences cancel single-qubit X/Y drive terms to first order, suppressing odd-parity XI/IX contributions; (ii) residual coherent terms are dominated by ZI/IZ/ZZ and small two-flip XX/YY admixtures—each even; (iii) dominant stochastic noise during the CR pulse is dephasing-like (L[Z]) on one or both qubits and correlated ZZ fluctuations from spectator couplings; both preserve parity; (iv) single-qubit amplitude damping contributes odd components but is typically smaller than dephasing over a 200–600 ns CNOT on current devices.

We provide an operational test that directly estimates the ZZ eigenvalue s of the effective layer channel under Pauli twirling. The proposed threshold is conservative and testable today on IBM Quantum systems.

Method
Theoretical model and prediction
- Effective Hamiltonian and echo symmetry: A driven CR interaction yields an effective Hamiltonian H_eff ≈ Ω_ZX ZX + α_IX IX + α_ZI ZI + α_ZZ ZZ + …, with spectator-induced ZZ and higher-order terms. Echoed CR sequences are designed so that single-X/Y terms (IX, ZI) cancel to first order, leaving the entangling ZX plus predominantly Z-type and ZZ terms. Residual miscalibration leads to small coherent over/under-rotations, while stochastic noise during the pulse is dominated by dephasing channels L[Z_c], L[Z_t] and correlated L[Z_c + η Z_t].
- Pauli twirling: Interleaving random single-qubit Paulis on both qubits before and after each CNOT compiles coherent and non-Pauli errors into an effective Pauli channel Λ with probabilities {w_P}. The action of Λ on ZZ is characterized by the Pauli transfer eigenvalue s ≡ M_ZZ,ZZ = ∑_P w_P χ_P, where χ_P = +1 if P commutes with ZZ and −1 otherwise. Thus s = 1 − 2 p_odd, with p_odd ≡ ∑_{P∈odd} w_P. A parity-preserving bias is precisely s > 1 − 2 p_tot/2, i.e., p_odd < p_tot/2.
- Expected magnitude: For representative IBM devices (CNOT durations 200–600 ns; T1 ≈ 50–200 μs; Tφ ≈ 30–200 μs; EPC 1–4%), we estimate per-CNOT contributions: dephasing/dephasing-like and ZZ-correlated terms at O(0.5–2%), amplitude damping at O(0.2–1%), residual single-X/Y at <0.2–0.5% after echo. This yields p_odd/p_tot ≲ 0.4, i.e., s ≳ 1 − 2(0.4 p_tot) ≈ 1 − 0.8 p_tot. For p_tot ~ 0.02–0.04, s ≈ 0.984–0.968 on a single CNOT. For a brickwork layer comprising disjoint CNOTs, s_layer is the product of pairwise s_edge to a good approximation under locality (or measured directly). Conservatively, we predict s_layer ≥ 0.86 on typical edges and layers.
- Observable: For circuits designed so the ideal output has zero global Z-parity expectation, noise-induced parity bias produces a strictly positive Z-parity correlator whose depth dependence yields s_layer. Quantitatively, for a layer channel with ZZ eigenvalue s_layer, the excess probability that measured parity matches the randomly assigned ideal parity is R(d) = 0.5 s_layer^d after depth d.

Operational definition
- Z-parity correlator: For an N-qubit line, define C_N ≡ ⟨∏_{i=1}^N Z_i⟩ computed from bitstrings as C_N = P_even − P_odd.
- Parity preservation excess: Using randomized circuits whose ideal average has C_N,ideal = 0, define R(d) ≡ (C_N(d) + 1)/2 − 0.5 = 0.5 C_N(d). Under the twirled model, C_N(d) = s_layer^d.
- Falsification threshold: If R(20) < 0.02 (equivalently s_layer < 0.86) on most line edges and devices tested, the hypothesis is falsified.

Experiments (falsification plan)
Devices and cost
- Backends: Any 7–27 qubit IBM Falcon/Eagle with ECR-based CNOTs and access to 10k-shot jobs.
- Cost: 50 circuits × 10k shots per job; 2–3 jobs for different edges or depths. Total <$100 per job under current pricing.

Protocol A: Two-qubit ZZ-eigenvalue (local)
- Goal: Estimate s_edge for a given coupled pair.
- Circuit:
  1) For each m ∈ {1, 2, 5, 10, 15, 20}, generate K = 50 random-twirled sequences:
     - For k in 1..m: sample independent P_pre, P_post ∈ {I, X, Y, Z}⊗2; append P_pre, one compiled CNOT on the edge, then P_post chosen so that the net ideal Heisenberg action maps ZZ to ±ZZ (i.e., track frame to preserve measurement of ZZ). A simple choice is Pauli twirling with compensation: choose P_post = CNOT P_pre CNOT to keep ZZ invariant in the ideal frame.
  2) Prepare |00⟩; execute sequence; measure both qubits in Z; compute z1 z2 ∈ {±1}; average to get ⟨ZZ⟩_m over twirls and shots.
- Analysis: Fit ⟨ZZ⟩_m ≈ s_edge^m (single exponential through origin in log space). If s_edge < 0.86 at m ≈ 20 on most edges, falsify. Else accept local parity bias.

Protocol B: Multiqubit brickwork Z-parity correlator (global)
- Goal: Measure s_layer across a 5–7 qubit line.
- Circuit construction for each depth d ∈ {1, 2, 5, 10, 15, 20}:
  1) Brickwork CNOT layers: alternate parallel CNOTs on edges (1,2), (3,4), … and (2,3), (4,5), … for d layers.
  2) Single-qubit Clifford twirling: before each CNOT layer, apply random Pauli Q_i ∈ {I, X, Y, Z} on each qubit; after the layer, apply the tracked inverse so that the net ideal Heisenberg action maps Z^{⊗N} to ± Z^{⊗N} (Clifford frame tracking).
  3) Random sign balancing: insert a random global Z flip with 50% probability to ensure C_N,ideal = 0 on average across circuits.
  4) Measurement: measure all qubits in Z; compute C_N(d) = average of ∏_i z_i over shots and circuits.
- Analysis: Fit C_N(d) ≈ s_layer^d. Compute R(d) = 0.5 C_N(d). Falsification criterion: R(20) < 0.02 (i.e., s_layer < 0.86) on the majority of tested lines/edge-placements.

Practical notes
- Readout error: Calibrate a 2^N-to-parity confusion matrix via preparing computational basis states of Hamming weight 0–N and measuring the induced parity bias; apply a 2×2 correction to (P_even, P_odd) to first order. Alternatively, use symmetric randomized assignment (random global Z flip) to cancel static readout offsets.
- Compilation: Use Qiskit transpilation with optimization level low to preserve intended parallelism and with pulse-efficient ECR schedules (CXBackendProperties). Enable dynamic decoupling off to avoid masking native noise; run a control set with DD on to probe robustness.
- Shots: 10,000 shots per circuit give SE ≤ 0.01 on C_N(d) down to ~0.02 signals; aggregate over 50 circuits to push SE below 0.005.

Discussion
Interpretation
- s > 0.86 indicates a clear dominance of even-parity errors in the effective two-qubit channel under ECR, consistent with echo symmetry and dephasing/ZZ dominance. Observing s near unity locally but lower globally would implicate spectator-induced ZZ and crosstalk as principal contributors—still even-parity.
- If the falsification criterion is met (R(20) < 0.02), this would imply either unexpectedly strong odd-parity components (e.g., residual XI/IX, ZX miscalibration, leakage-induced channels) or that twirling/compilation assumptions were violated. Either outcome is informative for calibration and modeling.

Implications
- Modeling: Incorporating a Z-parity bias into noise models improves prediction of parity-sensitive observables, randomized benchmarking analyses, and Pauli error mitigation priors.
- Compilation: Parity-aware compilation can align logical parity checks or conserve-parity subroutines with hardware tendencies, potentially reducing logical error rates.
- Mitigation: Symmetry verification and post-selection on Z-parity become more effective when noise preserves the monitored symmetry; quantifying s guides the expected gain.

Limitations
- Twirling fidelity: Imperfect Pauli/Clifford twirling or frame-tracking errors can bias s estimates. We mitigate by using simple Paulis and verifying frame identities on simulators with the device’s basis gates.
- Readout systematics: Parity observables are sensitive to correlated readout errors; include parity-calibrated correction or cross-validate with mid-circuit parity checks on a spare ancilla where available.
- Device variability: Edges differ in coupling, detuning, and spectator layout; claims are statistical across “most edges” rather than universal per-edge guarantees.
- Leakage: Population in |2⟩ is not Pauli and may map partially into effective odd components after twirling. Measuring leakage (via qutrit tomography on a subset) would strengthen attribution but is not required for falsification.

Conclusion
We propose and operationalize a device-level hypothesis: noise in echoed CR-based CNOTs exhibits a Z-parity conservation bias. Under Pauli twirling, the two-qubit error channel’s ZZ eigenvalue s exceeds 0.86 on typical IBM edges, implying a measurable parity preservation excess R(20) > 0.02 under randomized CNOT-layer circuits. A compact experiment using 50 circuits × 10k shots per setting suffices to verify or falsify this claim today on IBM Quantum backends. Confirmation would justify parity-aware noise models and compilation; falsification would pinpoint unexpectedly strong odd-parity mechanisms, guiding improved ECR design and calibration.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
