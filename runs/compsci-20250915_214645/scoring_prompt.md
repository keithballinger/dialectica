You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Rate each idea on a scale of 1–10 for novelty, falsifiability, and feasibility under the constraints. Provide a one-sentence rationale per idea.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on AI (such as but not limited to LLMs), avoid  anything blockchain related ideas
- Highly novel
- Publishable in a leading journal for its subfield

        Ideas:
        1) Gradient Noise Governs In-Context Learning
Summary: The magnitude of stochastic gradient noise during pretraining causally determines an LLM’s in-context learning ability, independent of final training loss.
For a smart layperson: When training is “noisy,” models may learn to rapidly adapt from prompts at test time. This proposes that how noisy the updates are during training—not just how accurate the model is—controls how well it learns from context.
Falsification: Pretrain matched models to the same perplexity while systematically varying effective batch size or adding Langevin noise; then evaluate standardized in-context learning suites. If in-context learning scales monotonically with measured gradient noise (estimated via gradient variance) holding loss fixed, the theory is supported; otherwise it is false.
Novelty: Prior work correlates ICL with scale and data, but no causal theory ties it directly to the controllable statistic of gradient noise.

2) Attention-Head Capacity Sets Algorithmic Stack Depth
Summary: The maximum stack-like algorithm an LLM can emulate scales with the number of attention heads per layer rather than with depth or total parameters.
For a smart layperson: Some problems require tracking nested brackets or counting, like using a mental “stack.” This theory says the count of attention heads is the real limiter on such mental stacks, more than how many layers the model has.
Falsification: Train models with matched parameters but vary heads-per-layer; test on synthetic Dyck-k and controlled stack-depth tasks with length extrapolation; perform head ablations. If maximal solvable depth aligns with head count (and not layers/params), the theory holds; otherwise it fails.
Novelty: Shifts the hypothesized locus of algorithmic working memory from layers/width to attention head multiplicity with testable predictions.

3) A Unified Law Trading Training Compute for Test-Time Deliberation
Summary: On reasoning tasks, increasing inference-time tokens (deliberation/self-consistency) compensates for fewer training tokens via a predictable power-law equivalence.
For a smart layperson: You can train longer or think longer at test time; this posits a quantitative exchange rate between the two. It predicts how much extra “thinking” can make up for less training.
Falsification: Train model families with varied training token budgets; evaluate with increasing CoT tokens and self-consistency samples; fit accuracy to a joint power law in train and inference tokens. If a stable exponent and exchange rate generalize across tasks/models, it’s supported; if not, it’s rejected.
Novelty: Provides a precise, falsifiable scaling relation linking training compute and inference-time compute for reasoning accuracy.

4) Retrieval-Induced Parametric Knowledge Atrophy
Summary: Reliance on retrieval during training/inference causes a systematic decline in parametric (internal) knowledge, reducing closed-book performance over time.
For a smart layperson: If a model learns to “look things up,” it may stop remembering them internally. This theory predicts that using retrieval makes the model worse at answering without retrieval.
Falsification: Finetune identical models with and without RAG exposure while matching loss; periodically test closed-book QA and factual probing with retrieval disabled; measure forgetting after continued RAG-heavy training. If RAG models lose more internal knowledge than controls, the theory holds; otherwise it’s false.
Novelty: Proposes and tests a concrete negative training externality of RAG on internal knowledge retention.

5) Compression Phase Transitions Predict OOD Generalization
Summary: Abrupt representation compression events during training predict subsequent gains in out-of-distribution generalization.
For a smart layperson: Models sometimes suddenly simplify their internal representations. This claims those “crunch points” foreshadow when the model will do better on unfamiliar data.
Falsification: Track layer-wise rank/intrinsic dimension and mutual information during training; detect discrete drops; measure OOD performance before/after these events across seeds and hyperparameters. If OOD gains reliably follow compression transitions (and not matched loss decreases), the theory is supported; otherwise not.
Novelty: Connects measurable, discrete representational shifts to OOD generalization with timing-based predictions.

6) RoPE Base Determines Extrapolation Horizon
Summary: The rotary position embedding base sets an exponential horizon for length generalization that is largely scale-invariant.
For a smart layperson: A single knob in how models track word positions dictates how far they can handle longer inputs than seen in training. Changing that knob should predictably move the limit.
Falsification: Train matched models differing only in RoPE base; evaluate perplexity and task accuracy far beyond training lengths; estimate horizon from error growth. If horizons shift as predicted by the base and remain stable across model sizes, the theory is supported; otherwise it’s refuted.
Novelty: Offers a concrete, architecture-level control for length extrapolation with specific, testable scaling.

7) Path-Degeneracy Regularization Yields Monosemantic Features
Summary: Penalizing redundant gradient pathways during training induces sparser, more monosemantic internal features without sacrificing accuracy.
For a smart layperson: Models often mix many concepts in single units; reducing redundant routes for information flow should force cleaner, more interpretable features. This predicts better interpretability at similar performance.
Falsification: Train models with a path diversity regularizer (e.g., Jacobian orthogonality/trace penalties) vs. controls; evaluate feature sparsity/monosemanticity via dictionary learning and concept activation vectors; compare task performance. If interpretability improves substantially at equal accuracy, support; otherwise reject.
Novelty: Introduces a specific, testable training principle linking gradient-path geometry to semantic disentanglement.

8) LayerNorm Gain Orientation Predicts Instruction Following
Summary: The principal orientation of mid-layer LayerNorm gain vectors is a stronger predictor of instruction-following ability than RLHF reward metrics.
For a smart layperson: How normalization knobs point inside the model may matter more for following instructions than how much it was rewarded during fine-tuning. This claims a simple internal measurement can forecast instruction skill.
Falsification: Compute principal angles of LayerNorm gain vectors across models; correlate with instruction benchmarks controlling for size and reward signal; perturb gains (re-initialize/rescale) to test causal impact while keeping perplexity stable. If orientation predicts and causally affects performance beyond reward metrics, the theory stands; else it fails.
Novelty: Identifies a concrete, measurable internal statistic as primary for instruction following, surpassing external reward magnitude.

9) CoT Dispersion Is a Calibrated Epistemic Uncertainty Estimator
Summary: The variability of chain-of-thought samples provides a calibrated estimate of a model’s epistemic uncertainty about final answers.
For a smart layperson: When the model’s “thoughts” disagree more, it’s less sure; when they agree, it’s more confident—and this should be reliably tied to correctness. This proposes a way to trust the model’s confidence without extra training.
Falsification: Compute dispersion metrics (entropy/variance) over multiple CoT samples; assess calibration (ECE/Brier) of induced probabilities vs. accuracy across tasks; compare to deep ensembles and MC dropout baselines. If CoT dispersion yields better or comparable calibration consistently, support; otherwise reject.
Novelty: Formalizes and tests CoT variability as a principled uncertainty estimator across domains.

10) Zero-Shot Cross-Modal Alignment Emerges at a Text Isometry Threshold
Summary: A text model attains zero-shot vision-language alignment once its embedding space reaches a specific isometry threshold detectable by linear probes.
For a smart layperson: Without joint training, a good language space can be “close enough” to image spaces that a simple mapping ties them together. This predicts a measurable tipping point where this starts to work.
Falsification: Train text-only models of varying scales; freeze them and learn a small linear map from fixed vision embeddings; evaluate zero-shot V+L tasks; compute an isometry score (distortion under linear probes) and test for a threshold predicting alignment success. If performance jumps coincide with the predicted threshold across seeds/datasets, it’s supported; else refuted.
Novelty: Posits and tests a precise geometric criterion in text space that enables cross-modal alignment without joint pretraining.


        Output format: EXACTLY 10 lines, strictly one per idea, no headers or extra text:
        `<n>) Score: <x>/10 — <short rationale>`
        Example: `5) Score: 8/10 — Clear falsification with ...`
