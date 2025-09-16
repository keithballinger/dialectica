You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Write the first draft of a short scientific paper in Markdown based on the selected idea.

        Constraints of Paper:
        From: constraints/cosmo.md

- In the field of cosmology, especially quantum gravity
- Highly novel
- Publishable in a leading journal for its subfield
- Uses simulations in python for validation

        Selected Idea:
        Selected idea #1:

1) Primordial Bell Correlations in the CMB from Entangled Inflationary Initial States
Summary: Quantum-gravity–induced mode entanglement before or during inflation generates a phase-locked trispectrum that couples large- and small-scale E/B modes in a distinctive kernel.
For a smart layperson: If the universe’s earliest ripples were quantum-entangled, their fingerprints would show up as subtle, coordinated patterns in the polarization of the cosmic microwave background. These patterns link big and small swirls in a way ordinary physics cannot mimic. We can look for that very specific “dance step” in the CMB data.
Falsification: Build a Python pipeline (healpy, numpy, jax/camb) to simulate CMB maps with the predicted trispectrum kernel, derive an optimal estimator, and apply it to Planck/ACT/Simons Observatory maps; a null result with sensitivity below the predicted amplitude rules it out. Cross-check that lensing and foregrounds cannot fake the kernel by end-to-end simulations.
Novelty: Unlike previous non-Gaussian feature searches, this targets a concrete entanglement-induced trispectrum with fixed angle and scale dependence that cannot be produced by lensing or standard inflation.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
