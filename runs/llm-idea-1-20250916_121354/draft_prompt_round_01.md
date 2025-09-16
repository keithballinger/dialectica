You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Write the first draft of a short scientific paper in Markdown based on the selected idea.
        Field: general

        Constraints of Paper:
        From: constraints/llm.md

- Research focused on Large Language Model inference
- Very impactful on quality, performance, or agentic workflows
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models


        Selected Idea:
        Selected idea #1:

1) Entropy-Gated Reflective Decoding
Summary: Trigger brief “think” segments only when next-token entropy exceeds a threshold to improve accuracy per compute.
For a smart layperson: The model pauses to think only when it’s unsure about the next word, and proceeds normally when it’s confident. This uses extra effort precisely where it matters, saving time while keeping answers accurate.
Falsification: Implement per-step entropy from logits; if entropy>τ, insert a bounded self-reflection snippet before resuming normal decoding. Compare accuracy vs tokens/latency on GSM8K and BBH against fixed CoT and no-CoT; if no Pareto improvement, the theory is false.
Novelty: Prior work uses fixed or global thinking; token-level uncertainty-gated thinking without training is new.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
