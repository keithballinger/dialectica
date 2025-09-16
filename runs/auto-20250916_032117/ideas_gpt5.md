1) Bipartite Checkerboard Phase Drift on Heavy-hex during ECR
Summary: Coherent Z-phase drift on idle qubits alternates in sign across the heavy-hex bipartition when neighboring echoed cross-resonance (ECR) gates are driven.
For a smart layperson: IBM chips lay qubits in an interleaved “checkerboard” network. We predict that when a two-qubit gate runs, nearby idle qubits’ phases rotate, but with a checkerboard pattern of positive and negative shifts tied to the lattice coloring. This comes from how microwave leakage and echoing cancel differently on the two sublattices.
Falsification: On a 7–27 qubit patch, run phase-sensitive interleaved randomized benchmarking: drive an ECR on a chosen edge while surrounding idle qubits undergo Ramsey with a phase sweep; fit the sign of phase drift versus graph parity. Lack of a robust alternation pattern (sign linked to bipartition) falsifies the theory.
Novelty: Prior crosstalk work quantified magnitudes but did not predict a bipartite sign structure tied to heavy-hex symmetry under ECR drive.

2) Dynamic-Reset Neighbor Cooling
Summary: Mid-circuit measurement-plus-reset of a qubit transiently improves T2* of adjacent qubits by depleting shared readout photons.
For a smart layperson: Measuring and resetting one qubit briefly “drains” its local microwave environment. We predict that, right after this, a neighboring qubit keeps its quantum phase longer than it otherwise would.
Falsification: On a line of 3–5 qubits with dynamic circuits, compare Ramsey fringes on a target qubit with and without a neighbor’s mid-circuit measure+reset 0.5–2 μs earlier; if the extracted T2* does not increase beyond statistical uncertainty (e.g., >2σ), the hypothesis is falsified.
Novelty: Introduces and tests a concrete, short-timescale, neighbor-coherence benefit from active reset on fixed-frequency IBM devices.

3) One-Factor Model for Simultaneous Readout Crosstalk
Summary: The multi-qubit readout assignment error matrix is approximately low-rank (rank-1 dominant), reflecting a common-mode fluctuation.
For a smart layperson: When many qubits are read out together, most errors come from a shared “global wobble,” not independent mistakes on each qubit. We propose a simple one-factor model captures this.
Falsification: Prepare all 2^n computational basis states for n=6–10 qubits (subsampling if needed), perform simultaneous readout, estimate the confusion matrix, and compute singular values; if more than one singular value carries comparable weight (>20% each), falsify the low-rank model.
Novelty: Moves beyond per-qubit or pairwise crosstalk to a falsifiable low-rank structure claim for simultaneous readout.

4) Graph-Distance-1 Correlated-Error Bound under Simultaneous ECR
Summary: Two-qubit-gate-induced correlated errors are confined to spectators within graph distance 1, with distance-2 correlations below 1e-3 in Pauli error rates.
For a smart layperson: Running several two-qubit gates at once should only disturb immediate neighbors, not qubits two steps away. We predict a sharp boundary for how far correlated errors reach.
Falsification: Execute simultaneous RB on disjoint ECR pairs while probing spectators at distances 1, 2, and 3 using parity checks or cycle benchmarking; if distance-2 spectators repeatedly show correlations >1e-3, the theory is falsified.
Novelty: Provides a quantitative radius and threshold for correlated errors on heavy-hex, not just qualitative “locality.”

5) Quadratic Stretch Nonlinearity in Zero-Noise Extrapolation
Summary: Pulse-stretching used for zero-noise extrapolation introduces a coherent Z error that scales quadratically with stretch factor, bending the extrapolation curve.
For a smart layperson: Lengthening gates to “turn up” noise should change error linearly, but we predict an extra phase mistake that grows with the square of the length, curving the trend.
Falsification: For single- and two-qubit circuits measuring known observables, vary stretch s∈{1,1.25,1.5,1.75,2}, plot error vs s, and fit a + b s + c s^2; if c is statistically indistinguishable from zero across devices and circuits, falsify.
Novelty: Challenges the standard linear-noise scaling assumption for pulse-stretched ZNE with a specific quadratic term.

6) Layout-Induced Heavy-Output Plateau
Summary: Optimizing mapping to maximize parallel disjoint ECRs induces a plateau in heavy-output probability (HOP) versus depth, raising HOP by ≥0.05 compared to default layouts.
For a smart layperson: Random circuits are used to test quantum chips; deeper circuits usually degrade a key score called HOP. We predict that smartly arranging gates to run in parallel creates a flat region where HOP stays higher than usual.
Falsification: On 10–20 qubits, generate matched random circuits; compile with (a) default transpiler and (b) parallel-disjoint-ECR-aware layout; measure HOP vs depth with ≥10k shots/circuit; absence of a sustained ≥0.05 HOP gap falsifies.
Novelty: Predicts a specific, measurable stabilization feature in a standard metric arising purely from layout strategy.

7) Even-Parity Bias from Readout Frequency Crowding
Summary: Simultaneous readout of closely spaced resonator frequencies biases outcomes toward even parity due to amplifier compression.
For a smart layperson: If several qubits are measured at similar microwave tones, the amplifier gets slightly saturated and nudges results toward an even number of “1”s. We predict a detectable tilt toward even-parity outcomes.
Falsification: Select 4–6 qubits with crowded readout frequencies, prepare uniform superpositions, perform simultaneous readout for ≥100k shots, and compare parity distribution to binomial expectation; lack of significant even-parity excess falsifies.
Novelty: Links readout-frequency crowding to a concrete parity-level bias rather than generic error inflation.

8) Active-Reset Reduces SPAM Asymmetry
Summary: Measurement-based active reset reduces the asymmetry between preparing/measuring |0> versus |1> by at least 20% relative to passive thermalization.
For a smart layperson: Preparing a qubit by waiting can favor the “0” state, leading to uneven mistakes between 0 and 1. We predict that actively resetting makes the two states more equally reliable.
Falsification: On several qubits, measure SPAM error for |0> and |1> using (a) wait-based init and (b) measure+X reset; compute asymmetry |e0−e1|; if the reduction under (b) is <20% relative on average, falsify.
Novelty: Proposes a quantitative, device-level asymmetry law for SPAM under active reset on IBM hardware.

9) Post-Calibration Superlinear Drift Window
Summary: After backend recalibration, two-qubit error rates drift superlinearly for ~30 minutes before entering a linear/noise-floor regime.
For a smart layperson: Right after a tune-up, the system settles; we predict a brief period where errors worsen faster than a straight line, then stabilize.
Falsification: Submit short interleaved RB jobs every 3–5 minutes for ~2 hours spanning a known calibration; fit piecewise models and test for an initial convex segment ending within 20–40 minutes; absence of such a breakpoint and convexity falsifies.
Novelty: Introduces a time-localized, testable drift model tied to provider calibration events.

10) Cost-Optimal Repetition Code Length on IBM Heavy-Hex
Summary: For current IBM error rates (e≈0.5–2%), three-qubit repetition yields the lowest dollar cost per achieved logical bit fidelity compared to 1 or 5 qubits with majority vote.
For a smart layperson: Repeating a bit three or five times reduces errors, but costs more runs; we predict three repeats give the best reliability per dollar right now.
Falsification: Implement 1-, 3-, and 5-qubit repetition codes estimating a classical bit; at fixed total budget (convert shots and runtime using IBM pricing), compare cost per (1−logical error); if three-qubit is not optimal, falsify.
Novelty: Makes a concrete, pricing-aware optimality claim for code length that can be tested on today’s cloud hardware.
