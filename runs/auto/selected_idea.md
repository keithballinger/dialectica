Selected idea #1:

1) Z-Parity–Biased Noise in Cross-Resonance CNOTs
Summary: Two-qubit errors on CR-based CNOTs conserve global Z-parity significantly more than they flip it.
For a smart layperson: The hardware’s native two-qubit interaction follows selection rules that tend to keep an “even/odd” property of the qubits’ states unchanged. That means some error types are more likely than others. We predict a measurable bias toward preserving this parity across many gates.
Falsification: On a 5–7 qubit line, run random CNOT-layer circuits designed to maximally scramble parity and measure the Z-parity correlator versus depth. If the parity preservation excess R = P(parity preserved) − 0.5 is <0.02 at depth ~20 across most edges, the theory is falsified.
IBM cost plan: Single job with 50 circuits × 10k shots on a 7–27 qubit backend fits within <$100; repeat on 2–3 edges if desired as separate jobs, each still <$100.
Novelty: No published work posits a quantitative, device-level Z-parity conservation bias as a universal feature of CR noise channels with explicit thresholds.
