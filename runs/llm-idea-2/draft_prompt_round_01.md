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
        Selected idea #2:

2) Uncertainty-Gated Decoding Mode Switching
Summary: Switch between greedy decoding and stochastic sampling per step using a calibrated uncertainty score derived from MC-dropout variance of logits.
For a smart layperson: When the model is sure, pick the top choice; when it’s shaky, explore a bit. We measure shakiness by making the model “blink” slightly (dropout) and seeing how much its opinions change, then decide to explore or not.
Falsification: Add 2–3 MC-dropout passes per step only when margin/entropy exceeds a tuned threshold; test on HumanEval/MBPP (code), TriviaQA/TruthfulQA (factuality), and GSM8K; compare accuracy and tokens to pure greedy, temperature sampling, and nucleus baselines at matched latency.
Novelty: First inference-time scheduler that uses MC-dropout disagreement to gate per-step decoding mode without model retraining.


        Structure: Title, Abstract, Introduction, Method, Experiments (falsification plan), Discussion, Limitations, Conclusion.
