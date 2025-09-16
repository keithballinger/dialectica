Title: Uncertainty-Gated Decoding: Per-Token Mode Switching via MC-Dropout for Large Language Models

Abstract
Balancing exploration and reliability during decoding remains a central challenge for large language model (LLM) inference. Greedy decoding is fast and precise but brittle; stochastic sampling improves robustness and creativity but can degrade factuality and determinism. We propose Uncertainty-Gated Decoding (UGD), a per-token scheduler that switches between greedy and sampling modes based on calibrated epistemic uncertainty estimated with Monte Carlo (MC) dropout. At each step, UGD first computes a cheap confidence proxy (margin/entropy); only when uncertain does it invoke 2–3 additional dropout-enabled forward passes to measure disagreement. If epistemic uncertainty is high, UGD samples from the MC-averaged predictive distribution; otherwise it decodes greedily. This yields a tunable speed–quality trade-off without model retraining. We outline a falsification plan across code generation (HumanEval, MBPP), factual QA (TriviaQA, TruthfulQA), and arithmetic reasoning (GSM8K), comparing to greedy, temperature, nucleus, and entropy-gated baselines at matched latency. UGD is simple to implement with small open-source models and KV caching, adds negligible overhead on confident tokens, and targets exploration exactly where it helps.

Introduction
Decoding strategies strongly influence LLM behavior. Deterministic (greedy/beam) decoding often maximizes single-step likelihood but can cascade early errors. Stochastic methods (temperature, top-k/p, typical) increase diversity and robustness, yet can amplify hallucinations and reduce repeatability. Current schedulers are largely heuristic (entropy thresholds, length/position schedules) and agnostic to epistemic uncertainty—whether the model itself is unsure due to limited knowledge rather than local ambiguity.

MC dropout approximates Bayesian inference by treating dropout at test time as sampling from a posterior over weights (Gal & Ghahramani, 2016). The variance across stochastic forward passes estimates epistemic uncertainty. We leverage this idea for token-level scheduling: measure uncertainty only when cheap proxies indicate risk, and then decide whether to exploit (greedy) or explore (sample). To our knowledge, UGD is the first inference-time per-token mode switcher using MC-dropout disagreement to gate decoding without retraining the model.

Method
Problem setting
Given a model pθ(y | x) and partial output y1:t−1, we aim to choose the next token yt by either:
- Greedy: yt = argmaxk pθ(k | x, y1:t−1);
- Sampling: yt ~ q(· | x, y1:t−1), where q can be the model’s distribution or a tempered/top-p variant.

Uncertainty signals
We distinguish:
- Aleatoric proxies (cheap): base-step entropy Hbase = H[pθ(· | ·)], and top-two margin m = p(1) − p(2).
- Epistemic estimates (costly): MC-dropout with K samples. For i ∈ {1..K}, sample p(i) = softmax(z(i)) with dropout enabled; define the predictive mean p̄ = (1/K)∑i p(i). We compute:
  - Predictive entropy H[p̄];
  - Expected entropy E[H] = (1/K)∑i H[p(i)];
  - BALD-style mutual information MI = H[p̄] − E[H] (epistemic component).

Gating rule
1) Base pass (dropout off):
- Compute logits zbase and pbase = softmax(zbase).
- If m ≥ mthresh and Hbase ≤ Hthresh, output greedy argmax; skip MC.

2) MC pass (invoked only if base is uncertain):
- Run K dropout-enabled passes for current step (KV cache reused).
- Compute MI and optionally variance of top-logit Var[z(i)top].
- If MI ≤ MIthresh, decode greedy from pbase.
- Else decode by sampling from p̄ with temperature T(MI) and/or nucleus p(MI).

Temperature schedule
Use a monotone mapping T(MI) = Tmin + (Tmax − Tmin) · clip(MI / MI95, 0, 1), where MI95 is the 95th percentile MI on a dev set; similarly for top-p. This adapts exploration strength to uncertainty magnitude.

Sampling distribution
When sampling, draw from p̄ (MC-averaged) rather than pbase. This integrates model uncertainty and reduces sensitivity to a single dropout mask.

Calibration
- Dev-set calibration chooses (mthresh, Hthresh) to target a sampling invocation rate s (e.g., 10–30% of steps) under latency constraints.
- Choose K ∈ {2, 3, 4} by trading off MI stability vs cost.
- Choose MIthresh to maximize task metric at matched latency vs baselines.

Compute and latency
Let s be the fraction of steps triggering MC. Per-step forward passes ≈ 1 + s · K. Because MC is limited to the current token and reuses KV cache, overhead is small when s is small. UGD can be tuned to match or undercut the wall-clock of nucleus/temperature sampling while improving accuracy.

Pseudocode
- model.eval() for base pass; model.train() only for MC passes on current token. Keep dropout layers active; do not alter weights, grads off.

Algorithm UGD-step(x, y1:t−1, cache):
  zbase, cache = model.forward_step(x, y1:t−1, cache, dropout=False)
  pbase = softmax(zbase)
  m, Hbase = top2_margin(pbase), entropy(pbase)
  if m ≥ mthresh and Hbase ≤ Hthresh:
      return argmax(pbase), cache
  P = []
  for i in 1..K:
      zi, _ = model.forward_step(x, y1:t−1, cache, dropout=True)  # KV reused
      P.append(softmax(zi))
  pbar = mean(P)
  MI = entropy(pbar) - mean(entropy(p) for p in P)
  if MI ≤ MIthresh:
      return argmax(pbase), cache
  T = schedule_T(MI); p_nuc = nucleus_filter(pbar, p= schedule_p(MI))
  return sample(p_nuc, T=T), cache

Experiments (falsification plan)
Goals
- Test if uncertainty-gated switching improves task accuracy and robustness at matched or lower latency and similar token usage.
- Verify that MC uncertainty provides complementary signal beyond entropy/margin gating.

Models
- Small open-source: TinyLlama-1.1B, Phi-2 (2.7B), Llama-2-7B, Mistral-7B.
- Implementation: HuggingFace Transformers with incremental decoding and dropout layers intact.

Datasets and metrics
- Code: HumanEval (pass@1), MBPP (exact match).
- Factual QA: TriviaQA (EM/F1), TruthfulQA (truthfulness).
- Reasoning: GSM8K (accuracy).
- Efficiency: wall-clock latency per sample, tokens/sec, and fraction of steps invoking MC (s).

Baselines
- Greedy decoding.
- Temperature sampling (T tuned per task).
- Nucleus sampling (p tuned).
- Entropy-only gating: switch to sampling when Hbase exceeds threshold; no MC.
- Margin-only gating: switch when m below threshold; no MC.
- Ablations: UGD with K ∈ {2, 3, 4}; UGD sampling from pbase vs p̄; fixed T vs T(MI).

Protocol
- Matched latency: For each baseline, tune its parameters (T, p) to a latency bucket; tune UGD’s (mthresh, Hthresh, MIthresh, K) to match that bucket within ±5%.
- Dev/test split: Use standard validation for calibration; report test results once.
- Seeds: 5 seeds for sampling methods; report mean and 95% CI.
- Caching: KV reused across steps; MC passes recompute only the current step with dropout masks.
- Implementation details:
  - Enable dropout only during MC passes: model.train(); ensure no gradients, torch.no_grad().
  - Use torch.use_deterministic_algorithms(False) for dropout variability.
  - p̄ computed in float32 for stability.
  - Temperature schedule capped at [Tmin=0.7, Tmax=1.2] unless ablated.

Hypotheses
- H1: UGD outperforms greedy and standard sampling on HumanEval/MBPP pass@1 at matched latency by selectively exploring near-ambiguities.
- H2: UGD improves TruthfulQA truthfulness over temperature/nucleus at matched latency by avoiding unnecessary sampling on confident steps.
- H3: UGD matches or exceeds GSM8K accuracy vs baselines with similar or fewer tokens, due to targeted exploration at brittle steps.
- H4: MC uncertainty provides additive benefit over entropy-only gating (significant gains for same s, K small).

Sanity checks
- MI correlates with error probability: bin p(error) vs MI on dev.
- Vary s by sweeping (mthresh, Hthresh); plot accuracy vs latency.
- K-sensitivity: diminishing returns beyond K=3.

Reproducible code sketch (PyTorch)
- Provide reference implementation toggling dropout only for MC passes and reusing caches; publish as minimal open-source repo with evaluation scripts for each dataset.

Discussion
Why it works
- Aleatoric uncertainty (entropy) captures local ambiguity; epistemic uncertainty (MI) detects model indecision stemming from knowledge gaps. Sampling helps primarily in the latter, where exploring alternatives increases chance of hitting a correct program line or fact. UGD targets exploration to those steps, maintaining determinism elsewhere.

Impact on agentic workflows
- Tool-use planners and multi-step agents benefit from reliable backbones with selective exploration at decision bottlenecks (e.g., function signatures, API selection). UGD can also gate when to branch or request external verification, based on MI.

Systems considerations
- Overhead is bounded: only current-step recomputation with dropout, caches reused. The expected compute multiplier is ≈ 1 + s · K. With s≈0.15 and K=3, cost ≈ 1.45×, typically comparable to nucleus sampling overhead while yielding better accuracy.
- Compatible with speculative decoding and KV offloading; gating applies after draft verification.

Relation to prior work
- Typical/top-p sampling adjusts exploration by local probability shape, not epistemic uncertainty. Entropy gating is heuristic. MC dropout has been used for classification uncertainty, but, to our knowledge, not to gate per-token decoding mode in LLMs without retraining. UGD operationalizes Bayesian uncertainty for practical inference-time scheduling.

Limitations
- Requires models with active dropout layers; some architectures omit dropout or fuse it. Workarounds (stochastic layernorm/dropconnect) are future work.
- Calibration may be task- and model-dependent; thresholds may need per-domain tuning.
- MC estimates with small K are noisy; very tight latency budgets may limit gains.
- For safety-critical or compliance settings requiring strict determinism, any sampling may be undesirable; UGD can be configured to disable sampling, but then reduces to an uncertainty monitor.
- Very long contexts may shift MI distribution; periodic recalibration or adaptive MIthresh may be necessary.

Conclusion
We propose Uncertainty-Gated Decoding, a simple, retraining-free scheduler that switches between greedy decoding and stochastic sampling based on per-token epistemic uncertainty estimated via MC dropout. By invoking a few dropout passes only when cheap proxies flag uncertainty, UGD concentrates exploration where it matters, improving quality at comparable latency to standard sampling. The method is easy to implement with small open-source models, compatible with KV caching and existing decoders, and provides a tunable accuracy–latency trade-off. Our falsification plan evaluates the approach across code, factuality, and reasoning tasks against strong baselines and ablations. If borne out, UGD offers a principled, practical path to more reliable and efficient LLM inference.
