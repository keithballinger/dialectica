You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, or Minor Revisions.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, or Minor Revisions.

        Constraints of Paper:
        From: constraints/quantum_ibm_cost.md

- Highly novel
- Publishable in a leading journal for its subfield
- Can be falsified with experiments that can run today on IBM’s quantum cloud
- No single experiment run should cost more than $100

        Draft:
        Title
Readout-mitigation-induced contextuality inflation on superconducting qubits

Abstract
Contextuality witnesses such as the Peres–Mermin (PM) square are increasingly used to benchmark nonclassicality on today’s quantum hardware, often together with readout error mitigation (REM). We show that standard, factorized REM can spuriously inflate contextuality witnesses in the presence of readout crosstalk, producing apparent violations of noncontextual bounds from data that admit a noncontextual explanation before mitigation. We formalize this as a linear post-processing effect: if the true readout channel is correlated while the mitigator inverts a product channel, the composite map applied to probabilities need not preserve the noncontextual polytope. We derive crosstalk-aware noncontextual bounds by maximizing the witness over the image of the polytope under the inferred map and give a concrete inflation mechanism for the PM square on two qubits. We propose and cost out a falsification experiment on IBM’s quantum cloud using 3–4 transmon qubits, 6 PM contexts, and routine calibration circuits (<$100 per batch), comparing raw, standard REM, and crosstalk-aware analyses. The experiment is decisive: an observed violation that vanishes under crosstalk-aware bounds or under pairwise-aware mitigation indicates mitigation-induced inflation, not genuine contextuality. Our results provide immediately applicable guardrails for contextuality experiments on current superconducting devices.

Introduction
Contextuality is a defining nonclassical feature of quantum theory and a resource for quantum advantage. Experimental tests often employ state-independent witnesses such as the Peres–Mermin (PM) square on two qubits. Contemporary demonstrations on superconducting devices routinely apply readout error mitigation (REM) to counter measurement assignment errors and improve witness values. However, most REM pipelines assume a tensor-product readout confusion model, while actual readout on multiplexed superconducting platforms exhibits correlated errors (crosstalk) across simultaneously measured qubits.

We identify and analyze a failure mode: when the true readout channel is correlated, inverting a factorized model can map empirically noncontextual probability distributions outside the noncontextual polytope, inflating witness values past noncontextual bounds. We provide a constructive mechanism, derive robust crosstalk-aware noncontextual bounds, and design an experiment on IBM transmons that can validate or refute this effect with today’s hardware.

Method
Witness and measurement scheme
- We use the state-independent Peres–Mermin square on two qubits A,B, with nine Pauli observables arranged in three commuting rows and three commuting columns. The noncontextual inequality is WPM ≤ 4, while quantum mechanics achieves WPM = 6. Here WPM is the sum of the six context products (three row products and three column products), each being the product of three commuting observables’ outcomes.
- Each context is measured in a single circuit by applying basis-change gates to map the commuting Pauli set to Z-basis measurements on A and B, followed by simultaneous measurement of both qubits. All nine observables and their triple products can be computed from the two-bit outcomes within each context.

Readout noise and mitigation as linear maps
- Let p denote the ideal distribution over bitstrings per context, q = Atrue p the observed distribution after readout, and Atrue a 4×4 stochastic matrix capturing the two-qubit readout channel. Standard REM estimates a factorized channel A⊗ = A1 ⊗ A2 from single-qubit calibration and outputs p̂ = A⊗−1 q = M p with M = A⊗−1 Atrue.
- If Atrue ≠ A1 ⊗ A2 (correlated crosstalk), M is generally not stochastic and may map valid classical (noncontextual) behaviors outside the noncontextual polytope, allowing linear functionals such as WPM to increase beyond their noncontextual bounds.

Concrete inflation mechanism
- Consider a simple correlated bit-flip model for the two measured bits (post-rotation): with single-qubit flip probabilities εA, εB and an excess probability κ for correlated flips beyond independence. Then, for parity-type observables (e.g., σx⊗σx contexts), the measured correlator scales as
  Emeas ≈ (1 − 2εA)(1 − 2εB) Etrue + 4κ Etrue + O(ε^2, εκ).
- Factorized REM inverts only the diagonal factors, yielding
  Ê ≈ Etrue + [4κ / ((1 − 2εA)(1 − 2εB))] Etrue,
  i.e., a systematic overcorrection of magnitude ≈ 4κ for small ε.
- For the PM witness (a sum of six context products built from single- and two-qubit parities), this produces a positive bias that accumulates across contexts. It is straightforward to construct parameter ranges ε ≈ 2–5%, κ ≈ 0.5–2% where raw data obey WPM ≤ 4 while factorized REM returns WPM > 4.

Crosstalk-aware noncontextual bounds
- Given an empirical estimate Âtrue of the joint readout channel (from two-qubit calibration) and the factorized mitigator Â⊗, define the analysis map M̂ = Â⊗−1 Âtrue. We compute a robust bound
  Wnc(M̂) = max{ W(v M̂) : v ∈ N },
  where N is the noncontextual polytope for the PM scenario (convex hull of deterministic value assignments consistent with compatibility).
- This is a linear program: maximize W over v subject to v ≥ 0, v1 = 1, and linear compatibility constraints, then apply M̂ per context to map to the mitigated space. If a reported Ŵ exceeds Wnc(M̂), the violation cannot be attributed solely to mitigation-induced distortion; if Ŵ ≤ Wnc(M̂), the evidence for contextuality is not sound.
- An alternative operationally simpler guardrail is to apply pairwise-aware mitigation by directly inverting Âtrue (full 4×4 per measured pair) instead of Â⊗. This removes inflation from pairwise crosstalk and provides a necessary check.

Experiments (falsification plan)
Devices and qubit selection
- Use any 27–127Q IBM Falcon/Eagle device with nearest-neighbor connectivity. Select two adjacent qubits for the PM test (A,B) and one or two neighboring spectator qubits (S) on the same readout multiplexing line to modulate crosstalk.

Contexts and circuits
- Six PM contexts (3 rows, 3 columns). Each context: basis-change gates (from the PM construction) followed by simultaneous measurement of A,B (and spectators when used).
- Calibration circuits:
  - Single-qubit assignment: prepare |0>, |1> on A and on B to estimate A1, A2 (four states).
  - Two-qubit joint assignment: prepare |00>, |01>, |10>, |11> on A,B to estimate Atrue (4×4).
  - Optional spectator dependence: repeat the above while preparing spectator S in |0> or |1> (and measuring it) to quantify conditional crosstalk.

Conditions
- C0: Baseline — measure only A,B; spectators idle and not measured.
- C1: Amplified crosstalk — simultaneously measure spectators prepared in |1> (and optionally additional spectators on the same readout line) together with A,B. This typically increases correlated assignment errors on IBM devices.
- For each condition, run three analyses:
  1) Raw (no mitigation).
  2) Factorized REM: invert A1 ⊗ A2 per context.
  3) Pairwise-aware REM: invert Atrue per context.
  4) Crosstalk-aware bound: compute Wnc(M̂) using M̂ = (A1 ⊗ A2)−1 Atrue.

Shot budget and cost
- Per condition:
  - PM contexts: 6 circuits × 20,000 shots ≈ 120,000 shots.
  - Calibrations: (4 single-qubit + 4 two-qubit) × 10,000 shots ≈ 80,000 shots.
  - Optional spectator-dependent repeats double calibration shots.
- Total per condition ≤ 250,000 shots. Across C0 and C1: ≤ 500,000 shots. This is comfortably below $100 per batch on IBM’s public cloud pricing; individual experiments can be split if needed to ensure no single submitted job exceeds $100.

Data analysis and pre-registered decision rule
- Compute WPM for each analysis pipeline with bootstrap error bars (10,000 resamples).
- Estimate Âtrue and Â1, Â2 with Wilson intervals; propagate into M̂ and into Wnc(M̂) via sampling to obtain confidence intervals on the crosstalk-aware bound.
- Falsification criteria:
  - H1 (inflation): In C1, factorized REM yields ŴPM > 4 by ≥ 5σ, while pairwise-aware REM yields W̃PM ≤ 4 within errors and ŴPM ≤ Wnc(M̂) within errors. Interpretation: apparent violation is consistent with mitigation-induced inflation.
  - H0 (no inflation): Either factorized REM does not cross 4 significantly or it remains above both 4 and Wnc(M̂) even after pairwise-aware mitigation; the latter would refute our mechanism and suggest genuine contextuality or another unmodeled effect.
- Control: Repeat with spectators in |0> (C0) vs |1> (C1) to modulate crosstalk. The inflation, if present, should increase with spectator-induced crosstalk.

Predicted outcomes
- On current devices with known readout correlations at the percent level, we predict C1 to show ŴPM − 4 ≈ 0.2–0.8 under factorized REM, while raw and pairwise-aware analyses remain ≤ 4. The crosstalk-aware noncontextual bound Wnc(M̂) will sit above 4 and below the inflated ŴPM, certifying that the “violation” is attributable to mitigation, not physics.

Discussion
- Conceptual: Readout mitigation is a linear post-processing that, when misspecified, can push data outside foundationally meaningful polytopes (here, the noncontextual set). This is not a small technicality: it can qualitatively change the conclusion of contextuality tests.
- Practical guardrails:
  - Always report raw, factorized-mitigated, and pairwise-mitigated witness values.
  - Quantify crosstalk and compute crosstalk-aware noncontextual bounds based on the actually used mitigation map.
  - Avoid claiming contextuality unless the violation exceeds the crosstalk-aware bound and is robust to pairwise-aware mitigation.
- Broader impact: The same mechanism can affect Bell tests, magic-state witnesses, and stabilizer parities inferred under readout mitigation. Our LP-based bound generalizes to any linear witness over commuting contexts.

Limitations
- We treat readout crosstalk; gate-level and SPAM correlations prior to readout are not modeled here and could also bias witnesses.
- Our crosstalk-aware bounds assume time-stationary channels within a job; drift can widen confidence intervals.
- We focus on the PM square; extensions to KCBS on qutrits or to sequential-measurement contextuality require additional control and may face different loopholes.
- Pairwise-aware mitigation removes two-qubit readout correlations but not higher-order correlations across many simultaneously measured qubits; for larger subsystems, scalable correlated mitigation is more challenging.

Conclusion
Standard factorized readout mitigation can inflate contextuality witnesses on superconducting qubits in the presence of readout crosstalk, yielding spurious violations of noncontextuality. We provide a simple theoretical mechanism, a crosstalk-aware noncontextual bound computable by linear programming, and a low-cost, decisive experiment on IBM hardware. Applying these guardrails will make contextuality claims on today’s quantum processors falsifiable and reliable.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown)
