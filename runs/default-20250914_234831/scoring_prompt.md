You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Rate each idea on a scale of 1–10 for novelty, falsifiability, and feasibility under the constraints. Provide a one-sentence rationale per idea.

        Constraints of Paper:
        From: constraints/quantum_ibm_cost.md

- Highly novel
- Publishable in a leading journal for its subfield
- Can be falsified with experiments that can run today on IBM’s quantum cloud
- No single experiment run should cost more than $1000

        Ideas:
        1) Dressed-Basis Dephasing in IBM Transmons
Summary: The dominant dephasing axis is rotated from Z by a device-specific angle set by average control, predicting a peak in unitarity when benchmarking in that rotated frame.
For a smart layperson: Qubits can lose their “clock” in a particular direction on the Bloch sphere; this claim says that direction is not the obvious one, but a tilted axis determined by how the qubit is driven. If true, simple pre/post rotations can align circuits to the quietest direction and make them more stable.
Falsification: Perform single-qubit unitarity randomized benchmarking while sweeping a pre/post rotation angle θ around Z (implement via virtual-Z). A distinct maximum of unitarity at θ* ≠ 0 consistently across several qubits supports the theory; a flat curve or a peak at θ=0 falsifies it.
IBM cost plan: Use 1–5 qubits, 12 angles, 32 sequences/angle, 1024 shots/sequence (≈400k shots total); on a small backend this fits comfortably under $100 per run.
Novelty: No published work has mapped the full orientation of dephasing on IBM devices and linked it to control-induced dressed frames.

2) Exponential Distance Law for Readout Back-Action on Heavy-Hex
Summary: Simultaneous readout induces spectator-qubit Z-phase shifts that decay exponentially with heavy-hex graph distance with decay length ~1 edge.
For a smart layperson: Measuring one qubit can nudge its neighbors; this claim predicts that the nudge gets rapidly smaller the farther away you go on the chip, following a simple exponential rule. It ties the strength of this effect to the chip’s wiring layout.
Falsification: Run two-qubit Ramsey experiments on a spectator while toggling simultaneous readout on a target at distances d=1–4; extract induced phase shift Δϕ(d) and fit to A0 exp(−d/ξ). A good fit with ξ≈1 edge supports the theory; lack of exponential decay or long tails falsifies it.
IBM cost plan: Use 5–7 qubits, 5 distances, 20 circuits/distance, 4096 shots/circuit (~400k shots); well under $100 on standard IBM backends.
Novelty: Provides and tests a concrete distance law for measurement back-action across real device topology.

3) Percolation-Critical Measurement-Induced Transition on Heavy-Hex
Summary: The critical measurement probability pc for the entanglement transition in noisy Clifford circuits equals the site-percolation threshold of the heavy-hex graph (pc≈0.50±0.03).
For a smart layperson: Randomly measuring qubits can kill long-range quantum links; this claim predicts the exact tipping point matches a classic math threshold for how things connect on the chip’s lattice. It turns a complex quantum transition into a simple geometry number.
Falsification: On 8–12 qubits in heavy-hex layout with dynamic circuits, apply random Clifford layers with mid-circuit Z-measurements/resets at probability p, then estimate an entanglement proxy (e.g., stabilizer entropy or two-point mutual information). Locate pc from the crossing of finite-size curves; a value far from heavy-hex percolation (~0.5) falsifies.
IBM cost plan: 7 p-values × 20 instances × 1024 shots on 10 qubits (~143k shots); split into a few jobs to keep each run well under $100.
Novelty: Directly links measurement-induced transitions to a percolation threshold dictated by real hardware connectivity.

4) Limit-Cycle Error Drift from Cryocooler Vibration Lines
Summary: Two-qubit gate error rates oscillate at a narrow 1–2 Hz line due to cryocooler-induced frequency modulation of transmons.
For a smart layperson: The refrigerator that keeps qubits cold vibrates at a regular beat; this theory says that beat subtly modulates qubit errors in a periodic way you can see if you watch carefully over time.
Falsification: Repeatedly run short interleaved RB on a fixed pair for ~10 minutes, time-stamping each estimate; compute the periodogram of the EPC time series. A spectral peak near 1–2 Hz supports the theory; a flat spectrum falsifies it.
IBM cost plan: 300 RB mini-runs (each 16 sequences × 256 shots) ≈ 1.2M shots; chunk into ~6 jobs to maintain per-job cost < $100, or reduce shots to keep a single run < $100.
Novelty: Predicts and tests a specific, narrowband environmental imprint on error drift, beyond generic “slow drift.”

5) Universal Linear Law for Z2 Symmetry Verification
Summary: For circuits preserving Z-parity, post-selection on mid-circuit parity checks reduces infidelity linearly with acceptance A, r_post ≈ r0·A, when parity-violating errors dominate.
For a smart layperson: If your circuit should conserve “evenness,” you can check and discard runs that break it; this claim says the remaining error shrinks in a simple straight-line way with how many runs you keep. It offers a predictable payoff for symmetry checks.
Falsification: Implement 6-qubit parity-preserving Clifford circuits with parity checks every k layers (k varied to tune A), using dynamic circuits; measure output error vs A. A clear linear relation with slope ≈ r0 supports; systematic curvature falsifies.
IBM cost plan: 5 k-settings × 20 instances × 2000 shots on 6 qubits ≈ 200k shots; under $100 on current devices.
Novelty: Predicts a simple, universal scaling law for symmetry verification efficacy across circuits on IBM hardware.

6) Drive-Induced Quasiparticle-Like T1 Suppression with ms Recovery
Summary: High-depth two-qubit drive bursts transiently reduce nearby qubits’ T1 with recovery time τ≈1–10 ms, consistent with quasiparticle generation.
For a smart layperson: Hammering a region with many gates briefly “heats” nearby qubits so they relax faster; this effect fades within milliseconds. It’s like a splash that quickly settles.
Falsification: Alternate a “heating” block (≥100 CX layers) on a pair with immediate T1 measurements on a spectator at varying delays; compare T1 with vs without heating. A significant T1 dip that recovers with ms-scale τ supports; no dip falsifies.
IBM cost plan: 20 delay points × 2000 shots with/without heating (~80k shots total); comfortably < $100 per experiment.
Novelty: Proposes and tests a concrete transient, local T1 effect from digital gate bursts on gate-model hardware.

7) Randomized Compiling Isotropizes Coherent Error Orientation
Summary: With randomized compiling, error per Clifford becomes independent of the axis of underlying coherent single-qubit over-rotations, yielding invariant RB EPC across compiled orientation settings.
For a smart layperson: Twirling the gates with random twists should scramble directional control errors so they behave like uniform noise; this claim says the overall error rate then won’t care which way the original bias points.
Falsification: Perform single-qubit RB under three compiled orientation settings (e.g., interleaving virtual-Z to reorient frames) with and without randomized compiling. Invariance of EPC across orientations only when RC is on supports; residual dependence under RC falsifies.
IBM cost plan: 3 orientations × (RC on/off) × 24 sequences × 1000 shots ≈ 144k shots; well under $100 on small backends.
Novelty: Directly probes a core but rarely validated assumption of randomized compiling on real hardware.

8) Heavy-Output Probability Predicted by Unitarity-Depth Law
Summary: Heavy-output probability H for random circuits follows H ≈ 0.5 + α·u_eff^d, where u_eff is two-qubit unitarity and d is entangling depth.
For a smart layperson: How often a random quantum program gives “heavy” answers should drop in a predictable way with circuit depth, governed by how coherent the hardware noise is. This gives a simple formula to forecast performance from a basic noise measurement.
Falsification: Measure single- and two-qubit unitarity (u_eff) on a chosen subgraph; run random heavy-hex circuits at several depths d and compute H; fit H vs d to the proposed exponential form and check α consistency across layouts. Systematic deviation from the u_eff^d law falsifies.
IBM cost plan: Unitarity RB (few settings) plus 5 depths × 20 instances × 1024 shots (~130k shots total); under $100 per run.
Novelty: Provides a predictive, falsifiable link between local unitarity spectroscopy and a global circuit-performance metric.

9) Factorized Parity-Error Model on Heavy-Hex Loops
Summary: Multi-qubit stabilizer (parity) violation rates on 4–6 qubit loops factorize as the product of adjacent two-qubit parity-flip probabilities.
For a smart layperson: Errors on a ring of qubits may behave like independent links; this theory says the chance the whole ring flips parity is just the product of the pairwise error chances. It offers a simple way to predict big errors from small ones.
Falsification: Prepare 4- and 6-qubit ring-cluster stabilizers; measure stabilizer violation rates; independently extract pairwise parity-flip rates via two-qubit RB or parity checks; compare loop violations to the product model. Significant mismatch falsifies.
IBM cost plan: 6-qubit stabilizer circuits: 30 instances × 2000 shots, plus pairwise RB (~160k shots); < $100 per experiment.
Novelty: Introduces and tests a compositional error model tailored to heavy-hex loops on real devices.

10) Directionality-Asymmetric CX Errors Follow Spectator-Activation Law
Summary: CX error asymmetry between control→target directions grows with the number of actively driven spectator qubits within two edges, due to cross-drive and residual ZZ.
For a smart layperson: A two-qubit gate can be worse in one direction, and it gets even worse if neighbors are busy at the same time. This proposes a simple rule: the more nearby activity, the bigger the asymmetry.
Falsification: For a fixed pair, perform interleaved RB of CX in both directions while activating 0, 1, 2, … nearby spectators with random single-qubit Clifford layers; plot EPC asymmetry vs number activated. A positive, roughly linear trend supports; flat trend falsifies.
IBM cost plan: 4 spectator-load settings × 2 directions × 24 sequences × 1000 shots (~192k shots); under $100 on mid-scale backends.
Novelty: Predicts a quantitative, connectivity-dependent law for CX directionality under realistic concurrent activity.


        Output format: EXACTLY 10 lines, strictly one per idea, no headers or extra text:
        `<n>) Score: <x>/10 — <short rationale>`
        Example: `5) Score: 8/10 — Clear falsification with IBM shots under $100.`
