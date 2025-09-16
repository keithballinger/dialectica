Title: Entropy-Guided Chain-of-Thought Budgeting

Abstract
We propose a training-free controller that dynamically allocates chain-of-thought (CoT) tokens per query using an on-the-fly integral of next-token entropy. The controller continues reasoning when uncertainty remains high and halts early when uncertainty collapses, yielding large compute savings without degrading accuracy. On open 7–8B models (Mistral-7B, Llama-3-8B), our falsification plan targets 30–50% CoT-token reduction with ≤1% absolute accuracy loss on GSM8K, StrategyQA, and BIG-Bench Hard (BBH). We formalize a simple integral controller with anti-windup and windowed smoothing, show how to adapt self-consistency budgets using the same signal, and outline ablations against fixed-length CoT and probability-threshold halting. The approach is novel in using the cumulative “uncertainty mass” over a trajectory, rather than token-local confidence or fixed-length prompts, to govern reasoning length. It is easy to implement with standard decoding APIs and adds negligible overhead.

Introduction
Large Language Models (LLMs) often benefit from explicit reasoning transcripts (“chain-of-thought”), but such tokens are expensive. Prior work typically fixes CoT length, uses heuristic stop phrases, or trains task-specific halting heads. These strategies either waste compute on easy instances or truncate hard ones prematurely. A key observation is that during reasoning, the model’s uncertainty (as reflected in its next-token distribution) falls sharply once it has consolidated a plan or reached an answer. This suggests an adaptive rule: allocate more reasoning tokens precisely when uncertainty is high, and stop when it has reliably collapsed.

We introduce Entropy-Guided CoT Budgeting (EG-CB), a decoding-time controller that integrates next-token entropy over the unfolding CoT to decide, per instance, how long to “think.” Unlike confidence-threshold halting that triggers on instantaneous peaks, our integral smooths transient spikes and measures the total “uncertainty mass” consumed, reducing premature stops and chattering. EG-CB is training-free, model-agnostic, and compatible with standard prompts, temperatures, and nucleus sampling. We also show how the same uncertainty signal governs self-consistency sampling budgets adaptively.

Method
Setup
- Model: Any autoregressive LLM that exposes logits at each decoding step.
- Prompting: Standard CoT prompts (e.g., “Let’s think step by step.”) followed by an answer cue (e.g., “Therefore, the answer is:”).
- Decoding: Temperature T_r during CoT, temperature T_a for final answer emission (often T_a=0 or low).

Next-token entropy
At step t of reasoning, given logits z_t, define the temperature-adjusted distribution p_t = softmax(z_t / T_r). The token entropy is H_t = -Σ_v p_t(v) log p_t(v). We optionally normalize by log|V|: Ĥ_t = H_t / log|V|, producing values in [0,1]. In practice, the full vocabulary entropy adds negligible cost because logits are already computed for sampling; top-k approximations can be used but we default to exact entropy.

Integral controller
We maintain an integral of deviation from a reference entropy h_ref (normalized units), with anti-windup and smoothing:

- Rolling mean over the last W tokens: m_t = (1/W) Σ_{i=t-W+1..t} Ĥ_i.
- Integral state: U_t = clip(U_{t-1} + (m_t - h_ref), U_min, U_max), with U_0 = B0.
- Halting condition: stop reasoning when both:
  1) U_t ≤ 0 (the cumulative “uncertainty debt” is paid), and
  2) m_t ≤ τ_low for K consecutive steps (stability check).

Intuition:
- When entropy is high (m_t > h_ref), U_t increases, granting more budget (more tokens).
- When entropy collapses (m_t < h_ref), U_t decreases; once it crosses 0 and remains low, we halt.
- The window W and stability K suppress transient spikes; clip bounds prevent windup on outliers.
- A global hard cap L_max guarantees termination.

Answer emission
Upon halt, we switch to the answer phase and decode the short answer with T_a and (optionally) greedy decoding. For multiple-choice tasks we ask for the option; for open-ended math we parse the final number.

Adaptive self-consistency (optional)
For tasks benefiting from self-consistency (SC), we adapt the number of samples k using the integral:
- After the first CoT+answer, compute a scalar difficulty proxy D = max(0, U_t_end)/U_max + m_t_end.
- If D ≥ δ_high, draw additional samples up to k_max; if D ≤ δ_low, keep k=1; otherwise interpolate k.
- Aggregate answers by majority vote (MCQ) or by numeric mode (math).

Hyperparameters
- h_ref ∈ [0.1, 0.4] (normalized), τ_low ∈ [0.05, 0.2], W ∈ {5, 10, 20}, K ∈ {2, 3}, B0 tuned to target average token reduction, L_max set conservatively (e.g., 256 CoT tokens), U_min=0, U_max set to allow long trails (e.g., 10).
- Calibrate on a small dev split (e.g., 200 examples/task) to achieve a desired compute–accuracy Pareto point.

Pseudocode (reasoning phase)
- Inputs: model M, prompt x, params (T_r, h_ref, τ_low, W, K, B0, L_max)
- Initialize: t=0; U=B0; window=[]; stable=0; cot=[]
- while t < L_max:
  - logits = M.forward(x + cot)
  - p = softmax(logits / T_r)
  - H = entropy(p); Ĥ = H / log|V|
  - sample y_t ~ p (or nucleus sampling)
  - append y_t to cot; update window with Ĥ; m = mean(window[-W:])
  - U = min(U_max, max(0, U + (m - h_ref)))
  - if m ≤ τ_low: stable += 1 else stable = 0
  - if U ≤ 0 and stable ≥ K: break
  - t += 1
- Emit answer with answer cue and T_a.

Experiments (falsification plan)
Models
- Mistral-7B-Instruct, Llama-3-8B-Instruct (HF Transformers, float16/bfloat16).
- Single-GPU inference (e.g., A100-40GB or 3090-24GB).

Datasets and prompts
- GSM8K: standard CoT math prompt; parse final numeric answer.
- StrategyQA: short yes/no; CoT optional; extract “Yes/No.”
- BBH: a representative subset (e.g., Date Understanding, Tracking Shuffled Objects, Hyperbaton).

Baselines
- No-CoT direct answer.
- Fixed-length CoT: truncate reasoning at L ∈ {32, 64, 128}.
- Heuristic halting: stop when max token prob ≥ θ for K steps (θ ∈ {0.5, 0.7, 0.9}).
- Self-consistency (SC-k): k ∈ {3, 5, 10}, majority vote.

Our methods
- EG-CB: entropy-integral halting (single sample).
- EG-CB+ASC: EG-CB with adaptive self-consistency (k ∈ [1, k_max], k_max=5).

Metrics
- Accuracy (task-appropriate).
- Mean generated tokens per instance:
  - CoT tokens (reasoning phase).
  - Answer tokens.
  - Total generated tokens (excludes prompt).
- Throughput proxy: wall-clock time and estimated FLOPs (∝ tokens × hidden dim × layers).
- Token reduction vs. accuracy trade-off curves.

Targets for falsification
- Achieve 30–50% reduction in CoT tokens vs. fixed-length CoT-64 with ≤1% absolute accuracy drop on GSM8K; similar trends on StrategyQA and BBH.
- If not achieved across models and tasks, the hypothesis is falsified.

Ablations
- Thresholds: sweep h_ref, τ_low; show sensitivity and Pareto.
- Window/stability: W ∈ {5,10,20}, K ∈ {1,2,3}.
- Temperature: T_r ∈ {0.3, 0.7, 1.0}; show robustness.
- Normalization: Ĥ vs. unnormalized H; top-k entropy approximation (k ∈ {20, 50, 100}).
- Controller form: integral vs. instantaneous threshold; integral without anti-windup; exponential moving average instead of fixed W.
- EG-CB vs. EG-CB+ASC: impact on accuracy and tokens.
- Transfer: tune on a small dev subset of one task, test on others (out-of-domain robustness).

Implementation details
- HuggingFace Transformers generate with return_dict_in_generate=True and output_scores=True to access logits at each token; compute entropy on GPU.
- Negligible overhead: one softmax+entropy per step (already required for sampling); controller ops are O(1).
- Deterministic runs use fixed seeds; report mean±95% CI over 3 runs for stochastic settings.

Discussion
Why the integral?
- Instantaneous confidence thresholds are brittle: short bursts of high confidence can prematurely stop reasoning, while transient ambiguity can trigger unnecessary long thoughts.
- The cumulative deviation from a reference entropy captures how much uncertainty the model needed to resolve. Easy instances quickly drive the integral down; hard ones accumulate positive deviation and receive more tokens.
- The windowed integral with a stability check approximates a sequential test that is less sensitive to local noise, akin to control-theoretic integral action and sequential analysis.

Relation to prior work
- Fixed CoT length and SC-k spend compute uniformly across instances.
- Confidence-based halting and CALM-like approaches use local token probabilities or auxiliary heads; we instead integrate the uncertainty trajectory without training.
- Early-exit Transformers and PonderNet adapt depth; EG-CB adapts generation length. These are complementary and can be combined.

Practical implications
- Drop-in savings for agentic systems: planners, tool-use loops, and scratchpads can spend fewer tokens on straightforward steps and more on ambiguous ones.
- Budget control: operators can tune B0 and h_ref to meet compute SLAs while bounding accuracy loss.

Limitations
- Entropy calibration depends on decoding temperature and model scale; small calibration is needed when these change.
- Entropy may be high for stylistic verbosity rather than epistemic uncertainty; the stability gate mitigates but does not eliminate this.
- Some tasks benefit from deliberate long-form explanations regardless of local uncertainty; EG-CB may truncate helpful pedagogical content unless instructed otherwise.
- The method does not guarantee optimal compute allocation; it approximates difficulty via token-level uncertainty, which can be misaligned with final-answer uncertainty in rare cases.

Conclusion
We present a simple, training-free entropy-integral controller that adaptively budgets chain-of-thought tokens per instance. By integrating next-token uncertainty and halting when it collapses, EG-CB aims to reduce reasoning tokens by 30–50% with negligible accuracy loss on small open models across math, commonsense, and reasoning benchmarks. The method is easy to implement, adds negligible overhead, and provides a principled knob for compute–quality trade-offs in LLM reasoning and agentic workflows.
