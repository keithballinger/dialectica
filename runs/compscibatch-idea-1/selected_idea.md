Selected idea #1:

1) Entropy-Matched Adaptive Depth Decoding
Summary: Per-token decoding can stop at a variable layer when the model’s predicted entropy matches the teacher-forcing entropy profile, preserving accuracy while reducing compute.
For a smart layperson: Not every word needs the full “thinking” depth of the model. If the model is already confident enough—matching how confident it was during training—you can stop early for that word. This can make responses faster without hurting quality.
Falsification: Measure per-token entropy at intermediate layers and stop when it reaches the training-time entropy curve (estimated offline on a corpus). Compare accuracy and latency against full-depth decoding on standard benchmarks with small open-source LLMs; if accuracy drops significantly or latency doesn’t improve, the theory is false.
Novelty: Unlike prior early-exit methods, this uses a training-entropy matching criterion to govern per-token depth without extra training.
