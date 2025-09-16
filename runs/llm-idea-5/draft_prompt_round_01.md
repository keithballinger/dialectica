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
        Selected idea #5:

5) Lookahead-Consistency Token Filtering
Summary: Before committing a token, run cheap k-step micro-rollouts and favor tokens whose continuations agree semantically, reducing hallucinations.
For a smart layperson: If a next word leads down paths that disagree with each other, it’s risky; if short peeks ahead mostly agree, it’s safer. We only choose next words that look stable under quick peeks.
Falsification: For each step, shortlist top-p tokens, roll out k=3 short continuations with a 1–2B “draft” model, score agreement via NLI or string entailment, reweight logits accordingly; test on TruthfulQA, BioASQ-lite, and WikiSQL text-to-SQL for factuality/exact-match versus normal and speculative decoding.
Novelty: Introduces agreement-based micro-lookaheads as a token-level filter distinct from draft acceptance checks in speculative decoding.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
