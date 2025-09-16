Selected idea #1:

1) Entropy-Guided Chain-of-Thought Budgeting
Summary: Allocate reasoning token budget per query using a running entropy integral of next-token distributions to cut tokens without hurting accuracy.
For a smart layperson: Some questions need lots of thinking tokens, others need very few. By watching how unsure the model is as it writes, we let it think longer only when uncertainty is high. This should save words and time while keeping answers just as good.
Falsification: Implement an entropy controller on open 7B models (e.g., Mistral-7B, Llama-3-8B) for GSM8K, StrategyQA, and BBH; compare fixed-length CoT and self-consistency baselines to a target 30–50% token reduction with ≤1% absolute accuracy drop; ablate thresholds and uncertainty windows.
Novelty: Uses an on-the-fly entropy integral as a principled budget controller for reasoning length, rather than fixed prompts or heuristic stopping.
