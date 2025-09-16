Selected idea #1:

1) Unitarity–Visibility Bounds on Binary Classical Capacity of a Qubit Channel
Summary: The binary classical capacity of any single-qubit noisy channel is tightly bounded by explicit functions of the channel unitarity u and readout visibility r, collapsing to C2 ≈ 1 − h2((1+ur)/2) for near-unital noise.
For a smart layperson: How much reliable information one qubit can carry depends on how coherent the hardware is and how clearly it can be read out. This theory gives a simple, predictive formula from two standard calibration numbers. If the noise is of a common type, the formula becomes essentially exact.
Falsification: On one IBM backend, estimate u via unitarity randomized benchmarking and r from the measurement confusion matrix; then empirically maximize mutual information over a sweep of state encodings and measurement bases with 50k–60k shots; if the measured capacity lies outside the predicted [g−(u,r), g+(u,r)] interval beyond statistical error, the theory is falsified.
IBM cost plan: 1–2 qubits, ~12 jobs × 5k shots; free with open backends or well under $100 via pay‑as‑you‑go at these shot counts.
Novelty: It links an operational capacity directly to two routine device metrics with tight, experimentally checkable bounds.
