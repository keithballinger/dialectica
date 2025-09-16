You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Write the first draft of a short scientific paper in Markdown based on the selected idea.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on AI (such as but not limited to LLMs), avoid  anything blockchain related ideas
- Highly novel
- Publishable in a leading journal for its subfield

        Selected Idea:
        Selected idea #1:

1) Gradient Noise Governs In-Context Learning
Summary: The magnitude of stochastic gradient noise during pretraining causally determines an LLM’s in-context learning ability, independent of final training loss.
For a smart layperson: When training is “noisy,” models may learn to rapidly adapt from prompts at test time. This proposes that how noisy the updates are during training—not just how accurate the model is—controls how well it learns from context.
Falsification: Pretrain matched models to the same perplexity while systematically varying effective batch size or adding Langevin noise; then evaluate standardized in-context learning suites. If in-context learning scales monotonically with measured gradient noise (estimated via gradient variance) holding loss fixed, the theory is supported; otherwise it is false.
Novelty: Prior work correlates ICL with scale and data, but no causal theory ties it directly to the controllable statistic of gradient noise.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
