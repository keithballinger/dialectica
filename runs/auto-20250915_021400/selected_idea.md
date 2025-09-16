Selected idea #1:

1) Quantum Rational Inattention Equilibrium (QRIE) in 2×2 Games
Summary: Players’ mixed strategies equal a logit quantal response whose temperature is an explicit function of qubit dephasing (T2) and circuit depth.
For a smart layperson: People don’t process all information perfectly; they make noisy choices. This theory links that “noise” to a physical noise you can dial on a quantum chip, predicting exactly how often each choice is made. If the chip gets noisier, people should act more randomly in a specific, measurable way.
Falsification: Implement 2-qubit circuits where each player’s action is RY(βΔu) with a tunable idle delay to induce dephasing; predict β ∝ effective coherence (from backend T2 and delay). In Python (Qiskit Aer), fit β from simulated counts; on IBM hardware, estimate β from 10k-shot counts and test βhardware/βsim vs predicted T2 scaling; reject if deviation > 15% across ≥3 delay settings.
IBM cost plan: Use IBM Quantum free tier or ≤5 circuits × 10k shots; under common pay-per-shot rates this is <$50, $0 on free queue. Python simulations (NumPy/Qiskit Aer) are free locally.
Novelty: It derives a closed-form link between physical decoherence and the logit parameter of quantal response, providing a directly testable bridge from information economics to device physics.
