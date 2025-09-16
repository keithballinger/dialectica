You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Rate each idea on a scale of 1–10 for novelty, falsifiability, and feasibility under the constraints. Provide a one-sentence rationale per idea.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Ideas:
        1) Entropy-Proportional Compute Allocation for LLM Decoding
Summary: The compute needed per token to maintain accuracy is proportional to the token’s predictive entropy, so dynamically allocating more compute to high-entropy tokens preserves accuracy at lower average cost.
For a smart layperson: Some words are easy to guess, others are hard; the model “knows” this via its uncertainty. This idea says we should spend more effort on the hard words and less on the easy ones, without losing overall quality. That would make LLMs faster on average with the same answers.
Falsification: Implement an inference policy that adjusts compute (e.g., number of active layers, KV-cache precision, or draft/verify steps) as a function of token entropy from a small LLM (e.g., 1–7B). Measure task accuracy versus average FLOPs and show that proportional-to-entropy allocation yields the same accuracy as fixed compute at lower cost; failure to do so falsifies the theory.
Novelty: It posits a quantitative law linking token-level entropy to optimal compute allocation at inference time, not previously established for LLMs.

2) Logarithmic Salience Sparsity in the KV Cache
Summary: Only O(log L) past tokens (out of context length L) materially influence next-token prediction, and they can be identified by a simple salience score learned from hidden states.
For a smart layperson: Although models “remember” many prior words, only a few truly matter for predicting the next one. This theory claims the number that matter grows very slowly with context size and can be picked out cheaply. That means we can drop most of the memory without changing answers.
Falsification: Train a lightweight salience predictor (e.g., linear probe on hidden states) and keep only the top-s tokens per step; evaluate accuracy vs. context length while sweeping s ≈ c·log L. If accuracy collapses compared to full cache for in-distribution tasks, the theory is false.
Novelty: It proposes a specific logarithmic scaling law for influential tokens and an operational salience criterion for safe KV pruning.

3) Order-Preserving Quantization Invariance
Summary: If quantization preserves the ordering of pairwise cosine similarities among attention keys/queries and the top-k logit ordering, sequence outputs remain invariant up to paraphrase-level metrics.
For a smart layperson: Shrinking numbers to fewer bits usually hurts quality. This idea says if we keep the “rank order” of what’s most similar and most likely, the model will write essentially the same text. It gives a precise target for safe compression.
Falsification: Implement per-channel quantizers that either preserve or deliberately violate similarity and top-k logit orderings; compare outputs on QA and summarization by exact match and semantic similarity. If order-preserving schemes do not maintain output invariance, the theory is falsified.
Novelty: It identifies order preservation—not absolute error—as the sufficient condition for quantization-safe LLM inference.

4) Universal Margin Threshold for Early-Exit Decoding
Summary: A fixed logit margin (top-1 minus top-2) serves as a model- and task-agnostic stopping rule that preserves accuracy while enabling early exit or truncated computation.
For a smart layperson: When the model is very sure about the next word, we can stop doing extra work. This theory claims there’s a single confidence gap that works across models and tasks to safely stop early.
Falsification: Implement early-exit when margin > τ and reduce computation (e.g., skip remaining layers or accept speculative drafts) across multiple small models and datasets; if no single τ keeps accuracy within a small bound (e.g., <1% drop) while saving compute, the theory fails.
Novelty: It posits a cross-model invariant threshold on logit gaps for safe early-exit, rather than per-model tuning.

5) Retrieval Phase-Transition Threshold in RAG Inference
Summary: Answer accuracy in RAG exhibits a sharp phase transition as a function of retrieval relevance; above a critical relevance, retrieval helps, below it harms compared to no retrieval.
For a smart layperson: Adding documents can help or hurt depending on how relevant they are. This idea predicts a tipping point: once documents are “relevant enough,” they help a lot; below that, they confuse the model.
Falsification: Sweep retriever relevance via controlled noise injection or score thresholds and plot accuracy; demonstrate or refute a statistically significant kink/phase transition compared to a smooth curve using change-point tests.
Novelty: It frames RAG efficacy as a critical phenomenon with a measurable threshold, not a gradual trade-off.

6) KL-Governed Acceptance in Speculative Decoding
Summary: The expected token acceptance rate in speculative decoding is determined by the KL divergence between drafter and verifier distributions, independent of prompt content.
For a smart layperson: Fast “draft and check” decoding accepts drafts when the small model agrees with the big one. This theory says acceptance depends only on how similar their probability patterns are overall, not on the particular text.
Falsification: Use small open models as drafter/verifier pairs with varying KL divergences (measured on a held-out set); measure acceptance rates across diverse prompts. If acceptance does not collapse to a function of KL (controlling for temperature), the theory is false.
Novelty: It provides a content-agnostic predictive law linking a statistical distance (KL) to acceptance behavior in speculative decoding.

7) Surprisal-Rank Skipping in Multi-Head Attention
Summary: When token surprisal is below a threshold, the effective attention matrix rank falls, allowing safe skipping or low-rank approximation of attention in selected layers without output change.
For a smart layperson: If the next word is easy to predict, the model’s attention becomes simpler. This theory says we can safely skip part of the attention computation for easy tokens without changing the answer.
Falsification: Estimate token surprisal online; apply rank-k attention approximation or gate off attention in layers when surprisal < s*. If output accuracy degrades beyond a set margin relative to full attention, the hypothesis is rejected.
Novelty: It links an observable (token surprisal) to a structural property (effective rank) to guide dynamic attention skipping.

8) Top-k Order Invariance under Logit Steering
Summary: Steering with additive logit biases that preserve the relative ordering among the top-k candidates leaves downstream task accuracy unchanged within a bounded margin.
For a smart layperson: You can nudge a model’s style by biasing word probabilities. This idea says as long as you don’t reshuffle the top few choices, the model’s task performance won’t suffer noticeably.
Falsification: Apply steering vectors that either preserve or perturb top-k rankings; evaluate task accuracy and semantic fidelity. If preserving top-k order still causes significant degradation versus control, the theory is falsified.
Novelty: It formulates a precise invariance claim for safe steering based on top-k rank preservation.

9) Self-Consistency Sample-Count Calibration Law
Summary: The Brier score of a self-consistency confidence estimator is a deterministic function of the number of samples and average token entropy, largely independent of model size beyond a small threshold.
For a smart layperson: Asking the model the same question several times lets you estimate how confident it should be. This theory predicts exactly how confidence improves as you add more tries, regardless of the specific small model used.
Falsification: Measure calibration (Brier score) versus sample count across several small LLMs and tasks while controlling entropy via temperature; if curves don’t collapse to a shared function, the law is false.
Novelty: It asserts a model-size-invariant scaling law for calibration via self-consistency at inference.

10) Branching-Factor Invariance in Tree-Based Speculative Decoding
Summary: For a fixed drafter–verifier KL, the expected verifier steps per accepted token are invariant to the branching factor of the speculative tree.
For a smart layperson: You can try multiple draft continuations in parallel before checking them. This theory says trying more branches doesn’t change how much checking you need per correct token if the small and big models agree to the same degree.
Falsification: Implement tree-based speculative decoding with varying branching factors on small models; measure verification steps per accepted token while holding drafter–verifier KL fixed. Deviation from invariance falsifies the claim.
Novelty: It predicts a new invariance property in parallel speculative decoding that depends only on distributional similarity, not search breadth.


        Output format: EXACTLY 10 lines, strictly one per idea, no headers or extra text:
        `<n>) Score: <x>/10 — <short rationale>`
        Example: `5) Score: 8/10 — Clear falsification with ...`
