You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Write the first draft of a short scientific paper in Markdown based on the selected idea.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Selected Idea:
        Selected idea #6:

6) Error-Correcting Speculative Decoding via Branch Voting
Summary: View speculative branches as redundant codes and fuse them with an error-correcting vote to improve accuracy at the same verifier budget.
For a smart layperson: Make several quick guesses and then combine them in a way that corrects occasional bad guesses, similar to how CDs fix scratches using redundancy. You keep the speed from quick guesses and gain reliability from smart fusion.
Falsification: Implement speculative decoding with a small drafter; add fusion rules (e.g., weighted majority by branch KL confidence, tie-breaking by verifier logits) before acceptance. Match verifier compute across methods and compare perplexity, acceptance rate, and downstream task accuracy versus standard accept/reject schemes. Stress-test under heavier quantization or noisy drafters.
Novelty: Coding-theoretic fusion of speculative branches goes beyond existing accept/reject or simple voting strategies.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
