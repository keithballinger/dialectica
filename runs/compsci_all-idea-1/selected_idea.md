Selected idea #1:

1) Kalman-Logit Smoothing for Quantized LLM Inference
Summary: Treat successive token logits as a noisy time series and apply a Kalman filter to counteract quantization-induced noise, improving accuracy at fixed compute.
For a smart layperson: Low-precision arithmetic adds jitter to the model’s next-word scores. A Kalman filter is a simple tracker that smooths noisy signals over time, like stabilizing GPS readings. Smoothing logits step-by-step can correct quantization errors without changing the model.
Falsification: Quantize a small open model (e.g., LLaMA-7B or Pythia-2.8B) to 4–8 bits; apply an online Kalman filter over logits with parameters fit on a validation set. Compare perplexity and task accuracy (e.g., WikiText, GSM8K few-shot) versus unfiltered quantized and full-precision baselines under equal latency. Ablate filter gains and show degradation if disabled or mis-specified.
Novelty: Time-series state-space filtering of token logits for quantized LLM inference has not been systematically proposed or tested.
