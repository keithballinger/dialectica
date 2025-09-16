1) Z-Parity–Biased Noise in Cross-Resonance CNOTs
Summary: Two-qubit errors on CR-based CNOTs conserve global Z-parity significantly more than they flip it.
For a smart layperson: The hardware’s native two-qubit interaction follows selection rules that tend to keep an “even/odd” property of the qubits’ states unchanged. That means some error types are more likely than others. We predict a measurable bias toward preserving this parity across many gates.
Falsification: On a 5–7 qubit line, run random CNOT-layer circuits designed to maximally scramble parity and measure the Z-parity correlator versus depth. If the parity preservation excess R = P(parity preserved) − 0.5 is <0.02 at depth ~20 across most edges, the theory is falsified.
IBM cost plan: Single job with 50 circuits × 10k shots on a 7–27 qubit backend fits within <$100; repeat on 2–3 edges if desired as separate jobs, each still <$100.
Novelty: No published work posits a quantitative, device-level Z-parity conservation bias as a universal feature of CR noise channels with explicit thresholds.

2) Motional Narrowing from Idle Gaps in Superconducting Circuits
Summary: Introducing specific microsecond-scale idle gaps between gate layers reduces dephasing via motional narrowing of 1/f noise.
For a smart layperson: Some noise is “slow,” like a drifting breeze; pausing briefly can average it out. We predict that adding the right tiny pauses between layers can actually improve accuracy, not worsen it. There should be a sweet spot delay that minimizes error.
Falsification: On 5–10 qubits, run matched random circuits at fixed depth while sweeping an inserted idle delay τ between layers (e.g., 0–1 μs). If circuit error monotonically worsens with τ and no error minimum appears (e.g., at τ* ~ 200–600 ns), the theory is falsified.
IBM cost plan: One job with 8 τ-values × 20 circuits × 5k shots fits under <$100 on mid-scale backends.
Novelty: A concrete, falsifiable prediction of a non-monotonic delay–error curve due to motional narrowing has not been experimentally posed for cloud transmon devices.

3) Readout Multiplexing Creates Positive Error Covariance
Summary: Simultaneous readout on shared multiplexed lines induces positive bit-flip correlations between those qubits.
For a smart layperson: When several qubits are measured through the same “wire,” their errors can become linked. We predict small but definite correlations in their readout mistakes. This is a hardware-level signature of measurement crowding.
Falsification: Prepare |0…0⟩ and perform repeated simultaneous measurements on 10–20 qubits; compute the covariance matrix of observed bit-flips. If qubit pairs on the same readout group show no positive covariance above 3σ from zero, the theory is falsified.
IBM cost plan: Single job with 100k total shots (batched in 10 circuits × 10k shots) on a 15–27 qubit device is within <$100.
Novelty: Prior studies assess readout error rates, but a general, quantitative theory predicting systematic positive covariance from multiplexing on today’s cloud devices is new.

4) Palindromic Circuit Echo Suppresses Coherent Over-rotation Errors
Summary: Time-reversal–mirrored (palindromic) circuits cancel coherent gate errors, yielding >30% relative fidelity gain over randomized compilation at moderate depth.
For a smart layperson: If you play a sequence forward then backward, certain consistent mistakes cancel out. We predict mirrored circuits will beat standard randomization by a big margin, not just a tiny tweak. The improvement should grow with depth until noise becomes fully random.
Falsification: On 5–7 qubits, compare heavy-output probability or XEB fidelity for matched random circuits versus their palindromic mirrors at depths 20–60. If the median relative improvement is ≤5% at depth ~40, the theory is falsified.
IBM cost plan: One job with 40 circuits (20 random, 20 mirrored) × 5k shots fits under <$100.
Novelty: While echoes exist, a strong, depth-dependent, device-level inequality against randomized compilation with explicit performance thresholds is a new theory.

5) Spectator Excitation Amplifies CR Gate Errors
Summary: Setting neighbor “spectator” qubits to |1⟩ increases CR CNOT infidelity by ≥20% due to dispersive shifts.
For a smart layperson: A gate between two qubits can be disrupted if nearby qubits are excited, like a conversation disturbed by a loud neighbor. We predict a quantifiable hit to accuracy when neighboring qubits are in the 1 state.
Falsification: Perform interleaved RB on a chosen edge with adjacent spectators prepared in |0⟩ vs |1⟩. If the error-per-Clifford increase with spectators=|1⟩ is <10% across most tested edges, the theory is falsified.
IBM cost plan: One job with 2 RB experiments × 20 sequences × 5 lengths × 2k shots fits under <$100 on a 5–7 qubit subset.
Novelty: A quantitative, edge-agnostic lower bound on spectator-induced error inflation for CR gates has not been established experimentally.

6) Waveform Resolution Imprints Quantized Angle-Error Beating
Summary: Finite DAC/AWG resolution induces periodic over/under-rotation patterns detectable as beating in long SX sequences.
For a smart layperson: Control electronics can only set pulse amplitudes on a grid, not continuously. Tiny rounding errors add up in patterns that cause the state to wobble in and out periodically. We predict a clear oscillation beyond simple decay.
Falsification: On single qubits, apply n repetitions of SX with n swept 1–300, measure P0; fit to damped cosine(s). If no secondary frequency component beyond simple T1/T2 decay is detected above noise or the fit prefers a single-exponential, the theory is falsified.
IBM cost plan: One job with 300 circuits × 2k shots each (batched) remains within <$100 on single/two-qubit allocations.
Novelty: Linking concrete beating frequencies in repeated gates to quantized control resolution on cloud hardware is a new, testable electronics–to–error hypothesis.

7) Leakage Bursts Cause Overdispersed Heavy-Output Time Series
Summary: Rare leakage to |2⟩ creates time-clustered fidelity drops, yielding a heavy-output Fano factor >1.5 across runs.
For a smart layperson: Occasionally a qubit “leaks” out of the usual 0/1 space and needs time to recover, causing several bad results in a row. This makes the ups and downs over time too large to be mere chance. We predict strongly “clumpy” failure statistics.
Falsification: Repeatedly run an identical random circuit instance (e.g., 2k repeats × 2k shots) and compute time-binned heavy-output rates; estimate Fano factor. If Fano ≈1 (Poisson-like) within CI and never exceeds 1.5, the theory is falsified.
IBM cost plan: One job with 200 identical circuits × 2k shots fits within <$100 on a 5–7 qubit device.
Novelty: A quantitative burstiness law for heavy-output statistics tied to leakage dynamics has not been proposed or tested on public superconducting platforms.

8) Nonlocal Dephasing from Mid-Circuit Measurement Backaction
Summary: Mid-circuit measure+reset on one qubit increases dephasing rates on non-neighbor targets by ≥5%.
For a smart layperson: Measuring one qubit mid-computation can jostle others through shared electronics, even if they’re not directly connected. We predict a measurable extra blur of quantum phases far from the measured qubit.
Falsification: On a dynamic-circuits backend, run Ramsey on a target while toggling mid-circuit measure+reset on a remote qubit at fixed timing; extract T2*. If added dephasing is <5% and not statistically significant across multiple placements, falsify.
IBM cost plan: One job with Ramsey sequences across 6 delay points × on/off backaction × 2k shots per circuit fits within <$100.
Novelty: Prior work studies local crosstalk; a quantified, nonlocal dephasing effect specifically from mid-circuit operations on cloud devices is new.

9) Millisecond-Scale Duty-Cycle Heating Degrades T1 and Recovers Exponentially
Summary: High-duty-cycle gate bursts transiently reduce T1 and gate fidelity, recovering with a 1–10 ms time constant.
For a smart layperson: Rapid, intense gate activity slightly warms or disturbs the chip locally, briefly worsening performance before it cools back. We predict a clear dip-and-recover behavior in energy-relaxation times.
Falsification: Run a “heater” block (dense CR/SX layers for ~1–5 ms), then measure T1 or interleaved RB at variable post-heater waits (0.1–20 ms). If no ≥10% T1 drop and no exponential recovery fit emerges, falsify.
IBM cost plan: One job with heater+measurement sequences across ~10 wait times × 2k shots each stays within <$100 on a 2–5 qubit subset.
Novelty: A specific dynamical law linking duty-cycle to transient T1 suppression with a millisecond recovery constant is untested on public transmon hardware.

10) Orientation Asymmetry Law for Heavy-Hex Logical CNOT Networks
Summary: Logical circuits compiled with CNOTs aligned to hardware-preferred orientations achieve >10% higher HOG than their orientation-reversed equivalents.
For a smart layperson: On these chips, a CNOT from A→B is not the same as B→A. We predict that whole computations run measurably better when you choose the “easy” direction throughout, even though the logic is identical.
Falsification: Generate pairs of logically equivalent random circuits (depth 20–60) differing only by globally reversing CNOT orientations via local conjugations; compare HOG. If the median HOG gain is ≤3% at depth ~40, falsify.
IBM cost plan: One job with 40 paired circuits × 5k shots on a 5–7 qubit subgraph fits within <$100.
Novelty: A device-level, quantitative asymmetry law for orientation-biased compilations on heavy-hex with explicit thresholds is new.
