1) Unitarity–Visibility Bounds on Binary Classical Capacity of a Qubit Channel
Summary: The binary classical capacity of any single-qubit noisy channel is tightly bounded by explicit functions of the channel unitarity u and readout visibility r, collapsing to C2 ≈ 1 − h2((1+ur)/2) for near-unital noise.
For a smart layperson: How much reliable information one qubit can carry depends on how coherent the hardware is and how clearly it can be read out. This theory gives a simple, predictive formula from two standard calibration numbers. If the noise is of a common type, the formula becomes essentially exact.
Falsification: On one IBM backend, estimate u via unitarity randomized benchmarking and r from the measurement confusion matrix; then empirically maximize mutual information over a sweep of state encodings and measurement bases with 50k–60k shots; if the measured capacity lies outside the predicted [g−(u,r), g+(u,r)] interval beyond statistical error, the theory is falsified.
IBM cost plan: 1–2 qubits, ~12 jobs × 5k shots; free with open backends or well under $100 via pay‑as‑you‑go at these shot counts.
Novelty: It links an operational capacity directly to two routine device metrics with tight, experimentally checkable bounds.

2) Adaptive-Measurement Phase Diagram for Accessible Information on Noisy Qubits
Summary: For a 4-state equatorial ensemble, adaptive two-stage measurements with mid-circuit feedforward beat any fixed POVM whenever (u,r) falls in a predicted region determined by device unitarity and readout visibility.
For a smart layperson: Taking a quick first look and then deciding how to measure can reveal more than any one-shot measurement. This theory predicts exactly when that adaptive strategy wins on a given device based on two calibrated numbers. It draws a sharp yes/no boundary for the benefit of adaptivity.
Falsification: Calibrate u and r; implement the ensemble and compare accessible information from (a) an optimized fixed pre-rotation + measure POVM versus (b) an adaptive two-step measurement using mid-circuit measurement and conditional rotation; if the observed advantage disagrees with the predicted region (present vs absent), falsify.
IBM cost plan: 1–2 qubits using dynamic circuits, ~8–10 jobs × 5k shots; free on open dynamic backends or <$50 paid.
Novelty: It provides the first device-calibrated phase diagram predicting adaptive-measurement information gain on present-day hardware.

3) Correlated-Dephasing–Induced Superadditivity of Holevo Information
Summary: Engineered common-mode dephasing on two qubits yields a classical ensemble whose jointly decoded Holevo information exceeds twice the single-qubit value, demonstrating correlation-enabled superadditivity.
For a smart layperson: Two noisy channels can carry more together than separately if their noise is shared; we predict a concrete way to see this with phase noise that hits two qubits in the same way. Simple entangling readout should reveal the “1+1>2” information effect.
Falsification: Use two neighbors plus a toggled spectator to induce correlated Z-noise; verify correlation via simultaneous RB; measure single-qubit χ1 for a 4-state equatorial code; then encode across both qubits and perform CNOT-based joint decoding to estimate χ2; if χ2 ≤ 2χ1 within error bars, falsify.
IBM cost plan: 3 qubits, ~10–15 jobs × 10k shots; free on open backends or within <$100 pay‑as‑you‑go.
Novelty: It predicts a concrete, low-depth manifestation of superadditivity driven by realistic, controllable noise correlations.

4) Finite-Blocklength Strong-Converse Law for Qubit Dephasing Channels
Summary: Above capacity, the average error of classical communication over a dephasing channel obeys a finite-n strong-converse lower bound with exponent E_sc(R,p) that remains accurate for n≈8–12.
For a smart layperson: Pushing information faster than a noisy link allows makes errors blow up exponentially; we give a precise formula for how fast, even for short messages, for the simplest quantum-induced noise.
Falsification: Implement memoryless dephasing via Pauli-twirled gates with calibrated phase-flip p; for n≤12 and rates R>1−h2(p), run random codebooks with ML decoding via basis rotations; if the observed error probabilities fall significantly below exp(−nE_sc) beyond confidence intervals, falsify.
IBM cost plan: 1 qubit, ~150k total shots across configurations; free on open backends or ≲$100 paid.
Novelty: It delivers a testable strong-converse error exponent tailored to near-term devices and short blocks.

5) QRAC Performance Law from Unitarity and Readout Visibility
Summary: The optimal success probability of the 2→1 quantum random access code equals (1+√2 u r)/2 up to second-order corrections in nonunitality, yielding a closed-form hardware-dependent law.
For a smart layperson: A qubit can store two bits so either one can be recovered on demand; noise lowers the success chance. We give a simple formula tying that chance to two standard calibration numbers of the device.
Falsification: Estimate u via unitarity RB and r via readout calibration; implement optimized 2→1 QRAC preparations/measurements and compare observed success to (1+√2 u r)/2 within shot-noise; significant deviation falsifies.
IBM cost plan: 1 qubit, ~6 settings × 10k shots ≈ 60k shots; free on open backends or <$50 paid.
Novelty: It provides the first closed-form, calibration-driven prediction for QRAC success on real hardware.

6) Measurement-Induced Information Balance with Calibrated Nonideality
Summary: For bipartite pure states, a projective measurement on one qubit satisfies I(X:B) = ΔI_coh ± δ, where δ is a computable correction depending only on calibrated assignment error and readout dephasing.
For a smart layperson: Measuring one half of an entangled pair converts some quantum-only information into ordinary classical information; we give a precise accounting identity for that conversion, with a correction term determined by how imperfect the measurement is.
Falsification: Prepare Bell states; calibrate measurement confusion and readout-induced dephasing; measure A and tomograph B pre/post to get ΔI_coh; compute I(X:B) from outcome statistics; if the equality fails beyond the predicted δ band, falsify.
IBM cost plan: 2 qubits, ~120k shots (tomography + outcomes); free on open backends or within <$100 paid.
Novelty: It yields an experimentally calibrated equality linking classical mutual information to lost coherent information under realistic measurements.

7) Partially Entangled States Optimize Entanglement-Assisted Capacity under Amplitude Damping
Summary: For amplitude damping with parameter γ, the entanglement-assisted one-shot classical capacity is maximized by Schmidt weight λ*(γ)=(1−γ)/(2−γ), making maximally entangled assistance suboptimal for 0<γ<1.
For a smart layperson: The best way to use shared entanglement over a lossy quantum link is not “maximally entangled” but “just right,” and we provide the exact recipe as a function of loss.
Falsification: Realize damping via an ancilla-based Kraus circuit; vary λ via controlled rotations and evaluate mutual information with optimized encodings/decodings; if the capacity peak does not occur near λ*(γ) and maximal entanglement isn’t worse over a γ range, falsify.
IBM cost plan: 3 qubits, ~175k shots over γ and λ grids; feasible free or ≲$100 paid.
Novelty: It predicts a simple, testable break from the common maximally-entangled optimum assumption for a key noisy channel.

8) Classical-Shadow Compression Bounds Min-Entropy of Shallow Random Circuits
Summary: The min-entropy of outputs from shallow random Clifford circuits is lower bounded by a linear function of the empirical classical-shadow compression rate with slope set by device unitarity.
For a smart layperson: The harder your measurement data “shadows” are to compress, the more randomness the circuit produced; we give a quantitative rule connecting these two and show how device coherence sets the scale.
Falsification: On 6–10 qubits at depth≈5, estimate unitarity u via RB; collect random-Pauli shadow data, compute a per-qubit compression rate, and independently estimate min-entropy via collision estimators; if H_min falls below the predicted bound, falsify.
IBM cost plan: 6–10 qubits, ~100k shots across 50 circuits; free on open backends or <$100 paid.
Novelty: It provides the first experimentally grounded bridge from shadow-data compressibility to rigorous randomness guarantees.

9) Quantifying Information Leakage from Residual ZZ Coupling
Summary: Residual ZZ coupling J causes a lower-bounded mutual information leak from a spectator’s X-basis bit to a target’s Z readout after time t, with I_leak ≥ 1−h2((1+cos(2Jt)·r)/2).
For a smart layperson: Weak always-on interactions make one qubit’s measurement reveal something about a neighbor; we give a formula predicting how many bits can leak as a function of that coupling and wait time.
Falsification: Calibrate J via echo; prepare the spectator in random X eigenstates, idle for various t, and measure the target in Z; compare measured I(spectator;target) to the predicted bound; systematic violation falsifies.
IBM cost plan: 2 qubits, ~6 time points × 10k shots ≈ 60k shots; free with open backends or <$50 paid.
Novelty: It turns a ubiquitous hardware imperfection into a quantitative, device-calibrated information-leak law.

10) Shadow-Based Unbiased Estimator with Variance Bound for Holevo Information
Summary: A classical-shadow estimator for the Holevo quantity χ of small cq-ensembles is asymptotically unbiased with variance bounded by a closed-form function of sample size and unitarity, enabling sub-tomographic χ estimation.
For a smart layperson: A key measure of how much classical information a quantum signal can carry is hard to get; we propose a lightweight estimator using random simple measurements and prove how accurate it should be given the device’s noise and data volume.
Falsification: Prepare a 4-state single-qubit ensemble; gather random-Pauli shadow data and compute the proposed χ estimator and predicted variance; independently obtain χ via tomography; across repeated runs, if the tomography value lies outside the estimator’s confidence band more often than expected, falsify.
IBM cost plan: 1–2 qubits, ~40k shots (20 random bases × 2k shots); free on open backends or well under $100 paid.
Novelty: It introduces a practical, variance-guaranteed method to estimate Holevo information without full state tomography.
