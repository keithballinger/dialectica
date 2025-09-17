Selected idea #2:

2) Token Influence Sparsity Law in Attention
Summary: For most tokens, next-token prediction depends on only O(√L) past positions (L=context length), identifiable on-the-fly by cumulative attention mass without retraining.
For a smart layperson: Even though the model “looks” at the whole history, only a small, predictable slice really matters for the next word. This suggests you can keep just the few most influential past words and throw away the rest. That would make long-context inference cheaper without hurting accuracy.
Falsification: For small models (TinyLlama, OPT-1.3B), prune per head the KV cache to the smallest set of past positions whose attention mass reaches a target (e.g., 95%), with a cap scaling as c·√L; measure perplexity and QA accuracy vs full cache as L grows. If performance degrades or required set scales faster than √L, the theory is falsified.
Novelty: Introduces a concrete √L scaling law for online KV sparsification discoverable via native attention, not offline training or fixed windows.
