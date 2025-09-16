1) Entropy-Gated Reflective Decoding
Summary: Trigger brief “think” segments only when next-token entropy exceeds a threshold to improve accuracy per compute.
For a smart layperson: The model pauses to think only when it’s unsure about the next word, and proceeds normally when it’s confident. This uses extra effort precisely where it matters, saving time while keeping answers accurate.
Falsification: Implement per-step entropy from logits; if entropy>τ, insert a bounded self-reflection snippet before resuming normal decoding. Compare accuracy vs tokens/latency on GSM8K and BBH against fixed CoT and no-CoT; if no Pareto improvement, the theory is false.
Novelty: Prior work uses fixed or global thinking; token-level uncertainty-gated thinking without training is new.

2) Multi-Layer Logit Fusion Decoding
Summary: Combine next-token distributions from multiple intermediate layers to reduce hallucination and improve calibration without training.
For a smart layperson: Instead of trusting only the model’s final guess, we average the opinions of several “earlier minds” inside it. If intermediate layers disagree, we temper the output toward safer tokens.
Falsification: Compute softmax(logits) from several layers and fuse with simple weights (e.g., decay by depth) before sampling; evaluate TruthfulQA and factual QA for hallucination rate and Brier score vs baseline. If no significant improvement at similar latency, the theory is false.
Novelty: Using logit-lens–style multi-layer ensembling online for decoding decisions is unexplored.

3) Attention-Mass–Adaptive KV Precision
Summary: Quantize each token’s KV cache precision online in proportion to how much attention it actually receives.
For a smart layperson: Keep detailed memories only for past words the model looks at a lot, and save space on the rest. This cuts memory and speeds up generation with minimal quality loss.
Falsification: Track per-token cumulative attention; assign 8/4/2-bit KV quantization tiers accordingly; test on LongBench and long-context summarization for latency, memory, and quality vs uniform 8-bit. If quality drops more than uniform baselines at equal or higher compute, the theory is false.
Novelty: Dynamic per-token KV precision controlled by observed attention usage has not been tested.

4) LSH-Based KV Cache Deduplication
Summary: Remove near-duplicate KV entries by hashing key vectors and retaining only the most recent representative to shrink cache size.
For a smart layperson: When the model has many very similar memories from repeating phrases, keep just one to save space without losing meaning.
Falsification: Apply rolling LSH on key vectors; on collisions above a similarity threshold, discard older KV entries; measure perplexity and exact-match on summarization/code completion and wall-clock speed. If speed gains don’t materialize or quality degrades beyond 1–2% vs baseline, the theory is false.
Novelty: Online KV cache deduplication via similarity hashing during decoding is new.

5) Self-Consensus Early Stopping for Parallel Decoding
Summary: Run 2–3 low-temperature decoders in parallel and terminate generation early when their outputs agree within a small edit window.
For a smart layperson: If multiple copies of the model start saying the same thing, we stop early to save time and tokens. This avoids needless extra words while keeping answers intact.
Falsification: Launch N parallel decoders; compute rolling n-gram overlap/Levenshtein over the last k tokens; trigger EOS when agreement>θ; test on long-form QA/instructions for token savings and quality parity. If tokens saved <10% at iso-quality, the theory is false.
Novelty: Turning self-consistency into an online early-stopping criterion is novel.

6) Quantized Lookahead Re-Ranking
Summary: Use a low-precision shadow model to simulate a few steps for top candidate tokens and pick the token with the best projected future likelihood.
For a smart layperson: Before choosing a word, the system quickly “peeks ahead” using a cheaper approximation to see which choice leads to a better future sentence.
Falsification: Build a 4-bit quantized copy; at each step, roll out 3–5 tokens for top-m candidates to estimate cumulative log-likelihood; select the best; evaluate on code and math tasks vs greedy/nucleus at matched latency. If quality doesn’t improve, the theory is false.
Novelty: Per-token lookahead using a quantized copy of the same model for on-the-fly re-ranking is new.

7) Self-Contrastive Decoding Against Format Bias
Summary: Subtract logits from a minimally perturbed prompt (e.g., neutralized system/preamble tokens) to suppress format-driven spurious tokens.
For a smart layperson: Ask the model the same question with and without formatting hints and cancel out the parts influenced only by formatting, keeping content-based guesses.
Falsification: Do two forward passes per step (original vs perturbed prompt), subtract scaled differences before sampling; test instruction traps and TruthfulQA for reduced false compliance and hallucinations at similar fluency. If no gains appear, the theory is false.
Novelty: Using paired prompt perturbations as a control variate on logits during decoding is novel.

8) Bandit-Controlled Retrieval Depth in RAG
Summary: An online bandit tunes how many documents to retrieve per query using token-level uncertainty as contextual feedback to minimize latency at fixed accuracy.
For a smart layperson: The system learns when to read more or less background material, balancing speed with getting the right answer.
Falsification: Implement UCB/Thompson over k∈{0,2,5,10} with reward = −latency + α·correctness; uncertainty features from early token entropy; evaluate on HotpotQA/NQ with a 7B model. If the latency–accuracy Pareto doesn’t beat fixed-k baselines, the theory is false.
Novelty: Retrieval depth controlled by uncertainty-driven bandits at inference time is new.

9) Speculative Tool Prefetching from Partial Hidden States
Summary: Predict imminent tool calls from early hidden states and prelaunch tools asynchronously to hide I/O latency in agents.
For a smart layperson: As soon as the model hints it will need a calculator or web search, we start that tool in the background so results are ready when requested.
Falsification: Train a small linear/MLP probe on hidden states to predict tool ID and coarse args; prefetch top-1 tool during generation on a simple multi-tool benchmark; measure wall-clock and error rate vs baseline. If latency doesn’t drop without more wrong tool calls, the theory is false.
Novelty: Using partial hidden states to trigger asynchronous tool prefetching is unexplored.

10) Token-Class–Aware Speculative Drafting
Summary: Let a cheap drafter propose only low-information tokens (function words/punctuation) while the main model supplies content words, reducing risk while speeding decoding.
For a smart layperson: A fast helper fills in “the,” “and,” and commas; the main model focuses on the important words; suggestions are verified before acceptance.
Falsification: Build a POS/probability-based classifier over next-token logits to flag function tokens; apply speculative drafting only on flagged tokens with verifier checks; benchmark speed and quality vs standard speculative decoding on diverse corpora. If it doesn’t outperform standard speculative decoding at iso-quality, the theory is false.
Novelty: Class-conditional, token-type–restricted speculative decoding is new.
