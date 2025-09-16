1) Kalman-Logit Smoothing for Quantized LLM Inference
Summary: Treat successive token logits as a noisy time series and apply a Kalman filter to counteract quantization-induced noise, improving accuracy at fixed compute.
For a smart layperson: Low-precision arithmetic adds jitter to the model’s next-word scores. A Kalman filter is a simple tracker that smooths noisy signals over time, like stabilizing GPS readings. Smoothing logits step-by-step can correct quantization errors without changing the model.
Falsification: Quantize a small open model (e.g., LLaMA-7B or Pythia-2.8B) to 4–8 bits; apply an online Kalman filter over logits with parameters fit on a validation set. Compare perplexity and task accuracy (e.g., WikiText, GSM8K few-shot) versus unfiltered quantized and full-precision baselines under equal latency. Ablate filter gains and show degradation if disabled or mis-specified.
Novelty: Time-series state-space filtering of token logits for quantized LLM inference has not been systematically proposed or tested.

2) Entropy-Guided Adaptive Precision
Summary: Dynamically choose arithmetic precision per token based on predictive entropy, allocating higher precision only when the model is uncertain.
For a smart layperson: Use more careful math when the model is unsure and cheaper math when it’s confident. This should save time without hurting quality by spending compute only where it matters. The controller is simple: read uncertainty and switch precision modes.
Falsification: Implement per-token mixed-precision (e.g., 8/16-bit) for matmuls conditioned on next-token entropy from an auxiliary head or previous step logits. Measure throughput and accuracy on standard corpora versus fixed-precision baselines, and test correlation between entropy and benefit of higher precision. Show control curves of quality vs speed as entropy thresholds vary.
Novelty: Per-token precision control driven by uncertainty for autoregressive decoding has not been explored in LLM inference.

3) Predictive Low-Rank KV Forecasting
Summary: Replace aged KV cache entries with a learned low-rank predictor that forecasts their future attention contribution, reducing memory bandwidth with minimal loss.
For a smart layperson: The model remembers lots of past words, which is expensive. Instead of keeping every detail, keep a compact summary that predicts how old words will still matter later. This keeps speed high while preserving important context.
Falsification: Implement an online low-rank linear dynamical model over KV states (e.g., rank-r projection + AR predictor) for tokens older than a horizon; plug into attention in LLaMA/Mistral small models. Compare long-context perplexity and task scores versus full cache, eviction, and naive low-rank compression at equal latency/memory. Sweep rank and forecasting horizon to identify breakpoints where accuracy falls.
Novelty: Forecasting future KV influence via learned low-rank dynamical models goes beyond static KV compression/eviction methods.

4) Soft-Prefix KV Reuse Across Prompts
Summary: Reuse KV caches from similar but non-identical prefixes by soft-aligning cached states to the new prompt and correcting with a light adapter, accelerating early tokens.
For a smart layperson: If your new prompt starts like something seen before, reuse that prior computation even if it’s not an exact match. A quick alignment step adjusts the old memory to fit the new text. This should speed up the first words without hurting quality.
Falsification: Build a KV cache index over many prefixes; retrieve matches by embedding similarity for new prompts. Apply linear alignment (e.g., Procrustes or learned adapter) to transform cached KVs before continuing inference; compare first-N token latency and accuracy versus no cache and exact-prefix caching. Measure drift and failure cases with increasing prefix mismatch.
Novelty: Extends prompt caching from exact string reuse to approximate KV state reuse via learned soft alignment and correction.

5) Target-Entropy Decoding
Summary: Adjust temperature online to follow a specified entropy schedule, aiming to reduce repetition and hallucination while maintaining utility.
For a smart layperson: Keep the model’s uncertainty at a healthy level—neither too random nor too sure—by changing the temperature on the fly. This should avoid boring repetition and overconfident mistakes. The target uncertainty can vary across the response.
Falsification: Implement feedback control to match a per-step entropy target by tuning temperature; evaluate on open-ended writing and factual QA (e.g., TruthfulQA, HaluEval) for repetition rate, factuality, and human preference at equal token budgets. Compare against fixed temperature and nucleus sampling; ablate different entropy schedules.
Novelty: Treating entropy tracking as a primary decoding objective with closed-loop control is new for LLM inference.

6) Error-Correcting Speculative Decoding via Branch Voting
Summary: View speculative branches as redundant codes and fuse them with an error-correcting vote to improve accuracy at the same verifier budget.
For a smart layperson: Make several quick guesses and then combine them in a way that corrects occasional bad guesses, similar to how CDs fix scratches using redundancy. You keep the speed from quick guesses and gain reliability from smart fusion.
Falsification: Implement speculative decoding with a small drafter; add fusion rules (e.g., weighted majority by branch KL confidence, tie-breaking by verifier logits) before acceptance. Match verifier compute across methods and compare perplexity, acceptance rate, and downstream task accuracy versus standard accept/reject schemes. Stress-test under heavier quantization or noisy drafters.
Novelty: Coding-theoretic fusion of speculative branches goes beyond existing accept/reject or simple voting strategies.

7) Evidence-Aligned Logit Projection
Summary: Project next-token logits onto a subspace derived from retrieved evidence, softly constraining generation to be evidence-consistent.
For a smart layperson: When answering questions, nudge the model toward words supported by documents it just looked up. Instead of hard rules, it’s a gentle steer that keeps fluency while reducing made-up facts.
Falsification: Retrieve relevant passages; build a projection basis from their token embeddings or TF-IDF vectors; at each step project logits toward this subspace with a tunable strength. Evaluate factual QA and citation faithfulness versus baseline and grammar-constrained decoding at matched latency; sweep projection strength and report fluency/perplexity trade-offs.
Novelty: Logit subspace projection as a soft evidence constraint is distinct from prior hard grammar or token masking approaches.

8) Semantic Centroid Attention
Summary: Cluster past tokens online into semantic groups and attend to cluster centroids with optional local refinement, reducing attention cost.
For a smart layperson: Instead of attending to every past word, group similar words and attend to their summaries, diving into details only when needed. This makes thinking cheaper without forgetting important ideas.
Falsification: Implement online clustering (e.g., streaming k-means) over key vectors; compute attention over centroids plus nearest neighbors; compare speed and accuracy on long-context benchmarks versus full attention, sliding windows, and ring/landmark methods. Vary cluster counts and neighbor budgets to find break-even points.
Novelty: On-the-fly semantic clustering to create centroid-based attention targets is a new inference-time reduction method.

9) Meta-Decoding Controller
Summary: A lightweight policy selects the decoding method and hyperparameters per step using features of the current state to optimize quality under a compute budget.
For a smart layperson: A small “decider” watches the model’s signals and switches between greedy, sampling, contrastive, or different temperatures as needed. It spends randomness and compute only where they help.
Falsification: Train a tiny controller (e.g., linear or small MLP) on step-wise features (entropy, logit gaps, repetition signals) via bandit/imitation to maximize task scores at fixed latency; compare to any single decoding strategy on diverse tasks. Test generalization across prompts and to another small model.
Novelty: Per-step adaptive selection among multiple decoding algorithms with a learned policy is not standard practice.

10) Attention Pattern Replay for Repetitive Structures
Summary: Detect repeated or templated segments and reuse earlier attention routing via alignment, skipping redundant computation.
For a smart layperson: When generating lists, tables, or boilerplate, the model often follows the same pattern. Reusing that pattern avoids rethinking the same structure, saving time.
Falsification: Add a repeat detector (e.g., LSH over recent hidden states) to identify segment recurrences; copy or shift attention/KV routing from the prior segment with a small alignment layer. Measure speedups and accuracy on structured generation (lists, JSON, code) versus baseline attention; ablate detection thresholds and alignment strength.
Novelty: Explicit replay of prior attention patterns keyed by detected repetition is a new inference-time acceleration tactic.
