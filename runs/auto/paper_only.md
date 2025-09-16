# Selection-rule–induced Z-parity bias in cross-resonance CNOT noise

## Abstract
We predict and operationally define a Z-parity conservation bias in the two-qubit noise of echoed cross-resonance (CR) CNOTs on fixed-frequency superconducting qubits. After single-qubit Pauli twirling, the resulting two-qubit error channel places strictly greater probability on even-parity Paulis (commuting with ZZ) than on odd-parity Paulis (anticommuting with ZZ). This follows from the driven-CR Hamiltonian, the symmetry of echoed sequences canceling single-qubit drive terms to first order, and the dominance of dephasing-like and ZZ-type correlated fluctuations during the pulse. We propose low-cost falsification experiments on IBM Quantum backends: estimate the ZZ eigenvalue s of the Pauli transfer matrix for (i) a single edge (local) and (ii) a brickwork layer (global). The hypothesis is falsified if the fitted layer eigenvalue satisfies s_layer < 0.86 (equivalently, the parity-preservation excess at depth 20, R(20) = 0.5 s_layer^20, is < 0.02) on most tested edges/lines. Each job uses O(50) circuits with 10k shots each; at typical per-shot prices this is O($100–$400) per job, well below the $1000/job cap (verify current IBM pricing). Confirmation would motivate parity-aware error models, compilers, and symmetry-based mitigation.

## Introduction
IBM’s fixed-frequency transmons implement CNOTs via the cross-resonance interaction compiled into echoed CR (ECR) sequences. We articulate a falsifiable hypothesis: two-qubit noise associated with CR-based CNOTs conserves Z-parity more often than it flips it.

For two qubits, define Z-parity as the eigenvalue of ZZ. Even-parity Paulis commute with ZZ (II, ZI, IZ, ZZ, XX, YY, XY, YX); odd-parity Paulis anticommute (XI, IX, YI, IY, ZX, XZ, ZY, YZ). Under Pauli twirling, let w_P be the Pauli error weights including the identity. The ZZ eigenvalue of the Pauli transfer matrix (PTM) is
s ≡ M_ZZ,ZZ = ∑_P w_P χ_P = 1 − 2 p_odd,
where χ_P is +1 for even and −1 for odd, and p_odd is the total odd-parity error probability (excluding the identity weight). A parity-preservation bias is equivalent to s > 0.

Physical motivation:
- Echo symmetry suppresses single-qubit X/Y drive terms to first order, reducing XI/IX-type odd-parity errors.
- Residual coherent terms are dominated by ZI/IZ/ZZ over-rotations and small XX/YY admixtures—all even-parity.
- Stochastic noise during the pulse is predominantly dephasing-like (L[Z]) on one/both qubits and correlated ZZ fluctuations, both parity-preserving.
- Amplitude damping (odd-parity contributions) over a 200–600 ns CNOT is typically smaller than dephasing for present T1/Tφ.

We provide two protocols that directly estimate s and s_layer with today’s IBM hardware.

## Theoretical model and prediction
- Effective Hamiltonian and echo symmetry: The driven-CR interaction yields an effective Hamiltonian with leading terms ZX (entangling), plus single-qubit IX/IY on the target, ZI/IZ Stark shifts, and ZZ. Echoed sequences cancel the leading IX/IY to first order, leaving ZX plus predominantly ZI/IZ/ZZ terms.
- Pauli-twirled picture: Interleaving random single-qubit Paulis before/after each CNOT compiles coherent/non-Pauli errors into a Pauli channel. In this representation, s = 1 − 2 p_odd as above.
- Expected magnitude: For CNOT durations 200–600 ns, T1 ~ 50–200 μs, Tφ ~ 30–200 μs, and p_tot ~ 1–4%, we estimate p_odd/p_tot ≲ 0.4, giving s ≳ 1 − 0.8 p_tot ≈ 0.968–0.992 per gate. For a brickwork layer composed of disjoint CNOTs with eigenvalues s_edge, the layer eigenvalue is s_layer = ∏ s_edge over the layer’s edges. Conservatively, we predict s_layer ≥ 0.86 on typical devices.

## Operational definition and observable
- Z-parity correlator on N qubits: C_N ≡ ⟨∏_i Z_i⟩ = P_even − P_odd from measured bitstrings.
- Labeled correlator: To avoid readout offsets and nonzero ideal signals, we assign a random sign σ ∈ {±1} to each circuit (by optionally appending a Z to one qubit) and compute the labeled correlator Ĉ_N ≡ ⟨σ · ∏_i Z_i⟩. Under the twirled model, Ĉ_N(d) = A s_layer^d with A ≈ 1 capturing SPAM.
- Match-probability view: The excess probability that the measured parity matches the assigned σ is R(d) = p_match − 0.5 = 0.5 s_layer^d.

## Falsification criteria
- Local: If most measured edges have s_edge < 0.93 (a typical two-CNOT-per-layer composition gives s_layer ≈ s_edge^2 < 0.86), the hypothesis is challenged locally.
- Global: If s_layer < 0.86 (equivalently R(20) < 0.02) on most tested lines/devices, the hypothesis is falsified.

## Experiments
### Devices and cost
- Backends: 7–27 qubit IBM Falcon/Eagle systems with ECR-based CNOTs.
- Job sizing and cost: Use ~50 circuits × 10k shots = 5×10^5 shots per job. Cost ≈ shots × (price/shot). With typical rates O(10^−4–10^−3 $/shot), this is O($50–$500)/job. The full study (2–3 jobs) totals O($100–$1000). Verify current IBM pricing; all jobs remain below $1000.

### Protocol A: Two-qubit ZZ eigenvalue (local)
Goal: Estimate s_edge for a chosen coupled pair.

For each depth m ∈ {1, 2, 5, 10, 15, 20}, generate K = 50 random circuits:
1) Prepare a ZZ-eigenstate, e.g., |00⟩. Optionally calibrate a single SPAM factor A by also running m = 0.
2) For k = 1..m, apply: random Pauli P_k, then CNOT, then random Pauli Q_k (draw P_k, Q_k ∈ {I, X, Y, Z}⊗2). Insert barriers so twirls are not canceled by the transpiler.
3) Frame-correct: apply a final two-qubit Pauli P_corr that inverts the ideal effect of all twirls so the net ideal unitary is identity.
4) Optional sign assignment: with 50% probability append Z on the first qubit and record σ = −1; otherwise σ = +1.
5) Measure Z⊗Z. Compute the labeled correlator Ĉ_2(m) = ⟨σ · Z⊗Z⟩ by multiplying the measured parity by σ per shot or per circuit.

Analysis: Fit Ĉ_2(m) = A s_edge^m via weighted least squares on log-scale (excluding nonpositive values or using a generalized regression). Report s_edge with confidence intervals. Control: a 1q-only variant (replace CNOT by idles of equal duration) bounds 1q-induced bias; subtract its decay rate if desired.

### Protocol B: Multiqubit brickwork parity (global)
Goal: Estimate s_layer across a 5–7 qubit line.

For each depth d ∈ {1, 2, 5, 10, 15, 20}, generate K = 50 random circuits:
1) Brickwork layers: Alternate parallel CNOTs on (1,2),(3,4),… and (2,3),(4,5),… for d layers.
2) Pauli twirling: Before and after every CNOT, apply independently sampled single-qubit Paulis on its endpoints. Add barriers to prevent cancellation.
3) Global frame correction: compute and apply single-qubit Pauli corrections that make the ideal net unitary the identity.
4) Random sign assignment: with 50% probability append a Z to qubit 1; record σ ∈ {±1}.
5) Measure all qubits in Z. Compute the labeled correlator Ĉ_N(d) = ⟨σ · ∏_i Z_i⟩.

Analysis: Fit Ĉ_N(d) = A s_layer^d and compute R(d) = 0.5 s_layer^d. Falsification: s_layer < 0.86 (or R(20) < 0.02) on most tested lines.

## Practical considerations
- Readout: Labeled correlators cancel constant readout offsets. For precision, measure the readout confusion matrix and apply a linear inversion; check that s estimates are stable with/without correction.
- Transpilation: Use low optimization (e.g., Qiskit O1) to preserve structure; insert barriers around each CNOT and its twirls; disable dynamic decoupling to probe native noise. Run a separate DD-enabled control to assess environmental drift.
- Shots and uncertainties: With 10k shots/circuit, SE ≤ 0.01 for parity observables; averaging over K = 50 circuits yields SE < 0.005, sufficient to resolve R(20) ≈ 0.02. Use bootstrap or clustered-robust errors across circuits.
- Leakage: Population leaving the computational subspace biases s downward and can produce non-Pauli effects. Monitor leakage with standard postselection (e.g., assignment to noncomputational states, when available) or include leakage-aware modeling in the fit.

## Discussion
Interpretation:
- Confirmation (s_layer ≥ 0.86): Supports even-parity dominance in ECR CNOT noise consistent with echo symmetry and dephasing/ZZ fluctuations. A gap between local and global s implicates spectator-induced ZZ/crosstalk.
- Falsification (s_layer < 0.86): Implies unexpectedly strong odd-parity mechanisms (residual XI/IX, ZX miscalibration, amplitude damping, leakage pathways), pointing to calibration and gate-design targets.

Implications:
- Modeling: Incorporate parity-sector asymmetry into device noise models and simulators.
- Compilation: Parity-aware scheduling/placement can exploit naturally suppressed parity flips.
- Mitigation: Symmetry checks/postselection and tailored twirling benefit from quantified s.

Limitations:
- Imperfect 1q gates and frame corrections bias s downward; Protocol A’s 1q-only control bounds this.
- Device variability: Results are edge- and day-dependent; our claim is about typical behavior across edges.
- Assumptions: The Pauli-twirled approximation neglects coherent inter-cycle interference; randomized compiling reduces but does not eliminate it.

## Conclusion
We state a falsifiable, low-cost hypothesis about CR-CNOT noise: after Pauli twirling, even-parity Pauli errors dominate, yielding s_edge close to unity and s_layer ≥ 0.86 on typical IBM devices. Two concise protocols on today’s IBM cloud can confirm or falsify this via parity-decay measurements with jobs well under $1000. Outcomes directly inform noise models, compilation, and mitigation, regardless of confirmation or falsification.
