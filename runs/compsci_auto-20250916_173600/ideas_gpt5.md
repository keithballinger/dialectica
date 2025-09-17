1) Entropy-Guided KV Cache Bitwidth Law
Summary: Next-token predictive entropy determines the minimum per-token KV cache precision needed to preserve accuracy.
For a smart layperson: When the model is very sure about the next word, its internal “memory” for that step can be stored more coarsely; when it’s unsure, it needs more detail. This proposes a rule that links how uncertain the model is to how many bits we must allocate to its memory at that moment.
Falsification: Compute token-wise entropy during inference, quantize the KV cache per token to bitwidths given by a simple entropy-to-bits mapping, and measure perplexity/accuracy vs a uniform-precision baseline. If the predicted mapping fails to preserve accuracy at the same memory budget or shows no correlation between entropy and required precision, the theory is false.
Novelty: It posits a content-adaptive, per-token precision law grounded in uncertainty, rather than uniform or heuristic per-layer quantization.

2) Exponential Attention Influence under RoPE
Summary: Under rotary positional embeddings, a past token’s causal influence on the current prediction decays exponentially with attention distance at a rate set by RoPE frequencies.
For a smart layperson: Words far back in the context matter less, and this theory predicts exactly how fast that importance fades based on the model’s positional encoding math. It claims a specific exponential decay curve rather than a vague “it fades.”
Falsification: Measure token influence via occlusion or influence functions across distances and fit an exponential curve whose rate is derived from the RoPE frequency scale; repeat across layers and heads on small models. If the decay is not exponential or the rate systematically deviates from RoPE-derived predictions, the theory is false.
Novelty: It links a precise, measurable decay law to RoPE parameters, moving beyond qualitative locality claims.

3) Margin–Beam Equivalence Criterion
Summary: When the top-1 logit margin exceeds an adaptive threshold, greedy decoding provably matches beam search output for that step.
For a smart layperson: If the model is much more confident in one next word than the rest, more complicated search won’t change the choice. This provides a concrete confidence threshold to safely skip expensive search.
Falsification: Empirically estimate a margin threshold from calibration data, then compare greedy vs beam outputs conditioned on margin bins; if substantial disagreements occur above threshold across datasets and models, the theory is false.
Novelty: It formulates a testable, quantitative condition under which beam search is redundant, not just an intuition.

4) Low-Rank Long-Range Memory Conjecture
Summary: Tokens beyond a context horizon influence predictions only through a low-rank summary of their KV states.
For a smart layperson: Very old parts of the conversation can be compressed into a tiny summary without changing what the model will say next. The claim is that a small number of “summary numbers” capture everything needed from the distant past.
Falsification: Replace far-past KVs with a learned low-rank projection (rank r) and test perplexity/accuracy as a function of r and distance; if high ranks are required or performance drops sharply even for moderate r, the conjecture is false.
Novelty: It posits a structural low-rank bottleneck for long-range effects, enabling principled cache compression beyond heuristics.

5) Predictability Plateau Layer Hypothesis
Summary: There exists a layer index k* after which a linear readout attains near-maximal next-token predictability, implying later layers mainly calibrate rather than add content.
For a smart layperson: Midway through the network, the essential answer is already present; later layers mostly fine-tune the confidence. This predicts a plateau where simple readouts are almost as good as the full model.
Falsification: Train linear probes on each layer’s hidden states to predict logits and chart accuracy vs layer; if no plateau emerges or large gaps persist to the final layer, the hypothesis is false; also test early-exit with a linear head to verify negligible quality loss.
Novelty: It connects probing plateaus to actionable early-exit criteria, not just descriptive layer analyses.

6) Static Head Inactivity Invariance
Summary: A stable subset of attention heads with low activation-saliency can be zeroed across inputs with negligible loss, enabling static inference-time pruning.
For a smart layperson: Some attention “units” almost never matter; you can turn them off all the time and hardly notice. This claims the same units stay unimportant across many prompts.
Falsification: Rank heads by gradient×activation on a small corpus, prune the lowest x% statically, and measure performance across diverse datasets; if the “inactive set” varies wildly by input or pruning causes significant loss, the theory is false.
Novelty: It asserts an input-agnostic invariance set of dispensable heads, not just per-input dynamic sparsity.

7) Temperature–Perplexity Local Linearity
Summary: Around temperature T=1, expected negative log-likelihood varies linearly with T under nucleus sampling for typical LLMs.
For a smart layperson: Small tweaks to how “random” the model is produce proportional changes in how surprised it is by the data. This gives a simple rule to set temperature precisely.
Falsification: Sweep small temperature intervals around 1, measure NLL, and fit linear vs quadratic models; consistent curvature or model-specific nonlinearity falsifies the theory.
Novelty: It offers a practical calibration law linking sampling temperature to measurable loss without retraining.

8) Proposal-Length Law for Speculative Decoding
Summary: The latency-optimal draft length L* scales with the draft model acceptance rate a as L* ≈ c/(1−a), for a hardware-dependent constant c.
For a smart layperson: If the small helper model is often right, it should propose longer chunks; there’s a simple formula for how long. This removes guesswork in tuning chunk sizes.
Falsification: Implement speculative decoding with varied draft lengths, measure end-to-end latency, estimate a and c, and test whether predicted L* matches the empirical optimum across models and hardware; systematic mismatch falsifies the law.
Novelty: It introduces a quantitative scaling rule connecting acceptance rate to optimal chunk length rather than heuristic tuning.

9) Suffix KV Reusability via Phase-Aligned RoPE
Summary: KV states for a shared suffix can be reused across different prefixes by applying a deterministic rotary phase correction.
For a smart layperson: If two prompts end the same way, you can recycle the model’s internal memory for that ending after a simple alignment step, even if their beginnings differ. This could save time when many prompts share endings.
Falsification: Precompute suffix KVs, derive a phase shift from prefix lengths, and compare reused vs freshly computed KVs and perplexity; if alignment fails to match outputs or degrades accuracy, the theory is false.
Novelty: It proposes a concrete, testable reuse mechanism exploiting RoPE structure to enable cross-prompt KV sharing.

10) Logit Variance–Guided Adaptive Compute Allocation
Summary: Per-token logit variance from lightweight stochastic forward passes predicts when extra inference compute will most improve quality.
For a smart layperson: If the model’s answers wobble a lot for a word, spending extra effort right then pays off; if they’re steady, you can save time. This turns uncertainty into a real-time budget controller.
Falsification: Enable inference-time dropout to estimate logit variance, allocate beams/experts or longer context only on high-variance steps, and compare quality–latency trade-offs to uniform baselines; lack of gains or weak correlation falsifies the claim.
Novelty: It grounds dynamic compute in a measurable uncertainty signal at token granularity, not heuristic triggers.
