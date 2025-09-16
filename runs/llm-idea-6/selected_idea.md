Selected idea #6:

6) Annealed Logit Noise for Controlled Exploration
Summary: Add calibrated Gaussian noise to logits early and anneal to zero, improving robustness on hard steps without harming easy ones.
For a smart layperson: A tiny nudge of randomness early helps escape bad first guesses, then we quiet it down to finish carefully. It’s like brainstorming first, polishing later.
Falsification: Implement per-step noise σ_t = σ0/(1+αt) applied only when entropy>τ; evaluate on code (HumanEval, MBPP) and long-form QA (LongForm, NarrativeQA) for pass@k, hallucination rate, and length; compare to fixed temperature and top-p schedules at matched tokens.
Novelty: Proposes an annealed noise schedule targeted by per-step uncertainty, decoupled from temperature, as an inference-only robustness mechanism.
