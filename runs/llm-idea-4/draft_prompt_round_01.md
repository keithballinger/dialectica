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
        Selected idea #4:

4) Lattice-of-Thought Decoding with Online Pruning
Summary: Expand multiple short reasoning branches in parallel and prune them with a lightweight scorer to outperform linear CoT at the same token budget.
For a smart layperson: Instead of thinking in one straight line, the model sketches a few short possibilities, then keeps only the promising ones. This keeps options open early and wastes fewer words overall.
Falsification: Implement a fixed-budget lattice (e.g., width 3, depth 2, iterative) with a scorer using sum logprob + answer consistency; evaluate on GSM8K, SVAMP, and ARC-C; compare accuracy at equal total tokens to linear CoT and self-consistency.
Novelty: Brings beam-like branching to reasoning traces with an explicit online pruning policy optimized for token budgets, not sequence likelihood alone.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
