Selected idea #2:

2) Low-Rank KV Cache Subspace Hypothesis
Summary: The keys/values accumulated during decoding lie predominantly in a low-rank subspace per head, enabling strong KV compression with negligible loss.
For a smart layperson: The model stores memories (KV cache) as it writes a sentence; this memory may be mostly redundant. If most of the useful information lives in a small number of directions, we can compress it without changing the output much.
Falsification: Apply per-head SVD or randomized low-rank projection to the KV cache during generation at varying ranks and measure perplexity/accuracy vs. baseline on small models; if low-rank projections cause large performance drops at modest ranks, the hypothesis is false.
Novelty: Prior KV compression focuses on quantization or eviction; this posits and tests an intrinsic low-rank structure per head at inference time.
