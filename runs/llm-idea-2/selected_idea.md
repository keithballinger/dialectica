Selected idea #2:

2) Uncertainty-Gated Decoding Mode Switching
Summary: Switch between greedy decoding and stochastic sampling per step using a calibrated uncertainty score derived from MC-dropout variance of logits.
For a smart layperson: When the model is sure, pick the top choice; when it’s shaky, explore a bit. We measure shakiness by making the model “blink” slightly (dropout) and seeing how much its opinions change, then decide to explore or not.
Falsification: Add 2–3 MC-dropout passes per step only when margin/entropy exceeds a tuned threshold; test on HumanEval/MBPP (code), TriviaQA/TruthfulQA (factuality), and GSM8K; compare accuracy and tokens to pure greedy, temperature sampling, and nucleus baselines at matched latency.
Novelty: First inference-time scheduler that uses MC-dropout disagreement to gate per-step decoding mode without model retraining.
