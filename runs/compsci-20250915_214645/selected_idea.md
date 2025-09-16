Selected idea #1:

1) Gradient Noise Governs In-Context Learning
Summary: The magnitude of stochastic gradient noise during pretraining causally determines an LLM’s in-context learning ability, independent of final training loss.
For a smart layperson: When training is “noisy,” models may learn to rapidly adapt from prompts at test time. This proposes that how noisy the updates are during training—not just how accurate the model is—controls how well it learns from context.
Falsification: Pretrain matched models to the same perplexity while systematically varying effective batch size or adding Langevin noise; then evaluate standardized in-context learning suites. If in-context learning scales monotonically with measured gradient noise (estimated via gradient variance) holding loss fixed, the theory is supported; otherwise it is false.
Novelty: Prior work correlates ICL with scale and data, but no causal theory ties it directly to the controllable statistic of gradient noise.
