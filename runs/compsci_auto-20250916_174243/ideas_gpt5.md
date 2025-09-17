1) Entropy-Adaptive KV Cache Pruning
Summary: The informativeness of cached tokens, measured by attention entropy, predicts when past keys/values can be pruned with negligible loss.
For a smart layperson: LLMs keep a memory of every word seen so far to help choose the next word. This idea says we can measure which past words the model is confused about using an entropy score, and safely forget many of them without changing the answer much. It should make inference faster with long prompts.
Falsification: Implement a per-token entropy metric from attention distributions and prune tokens below a threshold during decoding on TinyLlama/Mistral-7B-instruct. Measure speedup and perplexity/QA accuracy versus no pruning across narrative, QA, and code datasets. The theory is false if entropy fails to predict safe pruning (i.e., same pruning rate causes large accuracy drops across tasks).
Novelty: Prior pruning uses fixed windows or attention mass thresholds; using attention entropy as a principled, task-adaptive predictor of safe KV deletion is new.

2) Logit-Margin Controlled Cheap-Path Decoding
Summary: The margin between the top-1 and top-2 logits at each step predicts when reduced precision or a smaller “shadow” model will produce an identical next token.
For a smart layperson: If the model is very sure about the next word, you can compute it more cheaply without changing the result. The confidence can be read directly from how much the model prefers the top choice over the runner-up. Switching to a cheaper path when confidence is high should save time without hurting quality.
Falsification: During decoding on LLaMA-7B/Pythia-1.4B, switch to 4-bit matmuls or a small calibrated student for steps with margin above τ. Record token agreement, task accuracy, and latency versus full-precision baseline; vary τ to trace trade-offs. The theory is false if large margins do not reliably predict agreement or if accuracy degrades disproportionately.
Novelty: Confidence-driven early exit is known in classifiers, but coupling logit margin to dynamic precision/model switching at the token level in LLM decoding is new.

3) Cross-Step Lower-Layer Reuse via Linear Extrapolation
Summary: Consecutive decoding steps yield slowly varying lower-layer activations that can be updated by a learned linear extrapolator instead of recomputation.
For a smart layperson: The early layers of the model change little from one word to the next. If we can predict these small changes with a cheap linear rule, we can skip a lot of work. That would make generation faster while keeping answers the same.
Falsification: Cache lower-layer activations at t−1 and predict those at t using a small per-layer linear map fit on a held-out set, then freeze and evaluate on WikiText-103 and HellaSwag with LLaMA-7B/TinyLlama. Compare perplexity/accuracy and speed with full recomputation and naive reuse. The theory fails if linear updates cannot approximate activations without accuracy loss.
Novelty: While KV caching is standard, explicit linear extrapolation of intermediate activations across steps for decoder inference has not been validated.

4) Prefix-KL Predicts Speculative Acceptance and Optimal Draft Width
Summary: The KL divergence between draft and target models on the recent prefix predicts acceptance rate in speculative decoding and thus the optimal draft tree width.
For a smart layperson: When using a small model to guess a few words ahead for a big model, many guesses get rejected. This theory says you can estimate how many guesses will stick by measuring how similarly the two models think about the recent text. That tells you how far to guess to save the most time.
Falsification: Compute rolling prefix KL between a draft (e.g., TinyLlama) and target (e.g., Mistral-7B) and correlate with acceptance rates across domains. Implement a controller that sets tree width from KL and compare throughput and accuracy to fixed-width baselines. The theory is false if prefix KL poorly predicts acceptance or harms throughput/accuracy.
Novelty: Using an online, prefix-conditioned divergence as a control signal for speculative decoding policy is new.

5) Low-Rank Manifold of Contextual Token Importance
Summary: The distribution of past-token importance for next-token prediction lies on a low-rank manifold that can be spanned by a small basis of global “context vectors.”
For a smart layperson: The model doesn’t need to consider every past word independently; their importance patterns often repeat. If these patterns live in a small set of shapes, we can summarize many past words compactly. That could shrink the memory needed during inference.
Falsification: Learn a small basis to project attention scores or gradients w.r.t. KV onto k components using Pythia-1.4B on diverse prompts, then perform attention using only basis-weighted summaries and evaluate perplexity/accuracy. Vary k and compare against full attention and random projections. The theory fails if no small k preserves accuracy.
Novelty: Unlike prior low-rank attention approximations, this posits and tests a shared, task-agnostic low-rank structure of token importance discovered post hoc in pretrained models.

6) Nullspace-Aligned Quantization Minimizes Error Propagation
Summary: Aligning per-channel quantization noise with the downstream layer’s approximate nullspace preserves output accuracy even at very low bitwidths.
For a smart layperson: Rounding numbers can push errors into directions the next layer can’t “see.” If we rotate or rescale before rounding so the error lands in those blind spots, the model should still work well. This could allow more aggressive 2–3 bit inference without losing quality.
Falsification: Estimate each layer’s dominant sensitivity subspace (e.g., via Jacobian SVD on calibration data), learn lightweight rotations/scales to steer quantization noise away from it, then quantize LLaMA-7B/TinyLlama to 2–3 bits. Compare perplexity to SmoothQuant/OmniQuant at equal bitwidth. The theory is false if alignment provides no consistent advantage.
Novelty: Steering quantization noise into learned nullspaces of downstream mappings during inference-time calibration is new.

7) Depth-Progressive Attention Concentration Enables Fixed Top-k Pruning
Summary: In decoder LLMs, attention mass concentrates monotonically with depth onto a small set of past tokens, enabling a fixed top-k key/value per head after a depth threshold.
For a smart layperson: Deeper parts of the model focus more narrowly on a few important past words. If true, we can keep only those few for later layers, making the model faster without hurting results. It’s like skimming the memory more aggressively the deeper you go.
Falsification: Measure cumulative top-k attention mass by depth on Mistral-7B across tasks; identify a depth D where top-k≥τ consistently holds. At inference, enforce per-head top-k for layers ≥D and evaluate accuracy/speed; disprove if concentration is not monotonic or pruning harms accuracy at predicted k.
Novelty: This proposes and tests a depth-wise monotonic concentration property that supports a static, architecture-level pruning rule.

8) Next-Token-Conditioned Late-Layer Subnet Routing
Summary: The identity of the imminent next token largely determines late-layer activation patterns, allowing routing to token-class-specific subnets for the final layers.
For a smart layperson: Right before choosing the next word, the model’s later computations depend mostly on which word it’s about to pick. If we guess that word class (like punctuation, number, common word), we can run a smaller specialized part of the model. This could speed up generation while matching the full model’s choices.
Falsification: Train a lightweight classifier on mid-layer features to predict coarse next-token classes, route final L layers through class-specific low-rank adapters or pruned subnets on Pythia-1.4B/TinyLlama, and compare token agreement/accuracy/latency to full model. The theory fails if activation patterns are not class-predictive or routing degrades quality.
Novelty: Conditioning late-layer computation on predicted next-token classes for deterministic speedups is new distinct from MoE or speculative decoding.

9) Low-Rank Beam State Sharing in Autoregressive Search
Summary: Hidden states of different beams at the same time step lie in an affine subspace that permits shared computation via a base state plus low-rank updates.
For a smart layperson: When exploring multiple candidate sentences, their internal representations are very similar. If we represent them as a common part plus a few small differences, we can process many beams almost as cheaply as one. That could make beam search practical for LLMs.
Falsification: During beam search on LLaMA-7B for summarization/translation, factorize beam hidden states via rank-r updates and run matmuls using shared bases with low-rank corrections; measure accuracy (BLEU/ROUGE), token agreement with standard beam search, and speed. The theory is false if states do not compress well or quality drops at low r.
Novelty: Affine subspace sharing for beam-level hidden states to amortize per-layer computation is new.

10) Exponential Aging of KV Contribution in Pretrained LLMs
Summary: The contribution of each cached token to next-token prediction decays approximately exponentially with its distance in the sequence under fixed prompt distribution.
For a smart layperson: Older words matter less in a predictable way, like fading memories. If the fade follows a simple exponential rule, we can summarize far-back tokens compactly instead of storing everything. This would cut memory without changing outputs much.
Falsification: Fit decay kernels from measured gradients or attention attribution versus distance on Mistral-7B/TinyLlama across domains; replace long-range KV with exponentially weighted summaries and evaluate perplexity/QA accuracy. The theory is false if decay is not well approximated by exponentials or summaries hurt performance at predicted compression rates.
Novelty: It posits and tests a simple, universal exponential law for token influence that enables principled KV compression without retraining.
