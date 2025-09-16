Major Revisions

Brief critique:
- Core idea is promising and relevant to LLM inference, but a central methodological assumption is under-specified: reusing a KV cache computed with dropout off during MC passes does not sample from the same stochastic model as full MC-dropout and can bias MI estimates. This must be clearly framed as an approximation, with ablations quantifying its impact versus (i) full recompute and (ii) partial recompute of a suffix of layers.
- Dropout control is fragile as written. Many Transformer implementations use functional dropout (F.dropout) gated by module.training, not solely nn.Dropout modules. Toggling only Dropout layers may silently leave other dropout paths disabled. Using model.train() during MC passes (under no_grad) is typically safe for modern LLMs (no BatchNorm); document this and test.
- Novelty needs tighter positioning against prior art using disagreement for uncertainty/hallucination detection (e.g., SelfCheckGPT), BALD-style active learning, and uncertainty-aware decoding. Add citations and clarify how UGD differs (per-token gating, retraining-free, MI-driven sampling intensity).
- Evaluation should include uncertainty quality metrics (token-level error detection AUROC, ECE), an agentic benchmark if claiming workflow impact (e.g., function calling/tool-use), and fairness via matched-latency curves. Report MI vs error correlation and gating curves.
- Implementation and reproducibility details are thin: RNG control across MC passes, numerical stability (float32 for entropy/MI), memory footprint (avoid storing full vocab K times), effects of models with dropout=0, and how to get independent masks when batching K passes.
- Policy choices need explicit rationale or ablation: fallback to pbase under low MI even when entropy is high; probability vs logit averaging; schedules T(MI)/p(MI).
- Limitations should note calibration drift across domains and that MI with small K is noisy; propose cheap alternatives (e.g., micro-ensembles via LoRA, SWAG) for models without dropout.

Revised Draft

# Uncertainty-Gated Decoding: Per-Token Mode Switching via MC-Dropout for Large Language Models

Abstract
We propose Uncertainty-Gated Decoding (UGD), a per-token scheduler that switches between greedy and sampling modes based on epistemic uncertainty estimated with Monte Carlo (MC) dropout. A cheap proxy (e.g., entropy or top-2 margin) decides when to invoke K additional dropout-enabled passes to estimate mutual information (MI) between the predictive and posterior distributions. If MI is high, UGD samples from an MC-averaged distribution; otherwise, it decodes greedily. UGD is retraining-free, tunable for latency, and compatible with small open-source models. We present a falsification plan across code generation, factual QA, and arithmetic reasoning, with matched-latency comparisons to strong decoding baselines and uncertainty-only gating. We quantify the accuracy–latency trade-off, the predictive power of MI for error detection, and the effect of KV-cache approximations.

Introduction
Decoding controls LLM behavior: greedy is fast and brittle; sampling boosts diversity at the cost of factuality and determinism. Heuristic adaptivity (e.g., entropy thresholds) conflates aleatoric and epistemic uncertainty. MC dropout approximates Bayesian inference by sampling stochastic forward passes; disagreement (BALD) isolates epistemic uncertainty. We operationalize this at the token level to gate exploration. Unlike prior approaches relying on multiple diverse samples or ensembles for detection (e.g., hallucinations), UGD uses a small number of MC passes to decide when and how intensely to sample, without retraining.

Method

Problem setting
Given pθ(y | x) and prefix y1:t−1, choose yt by either greedy argmax or sampling from a derived distribution q(· | x, y1:t−1).

Uncertainty signals
- Cheap proxies (single pass, dropout off): entropy Hbase and top-2 margin m = p(1) − p(2).
- Epistemic estimate (MC-dropout): run K stochastic passes with dropout on, obtain p(i), compute p̄ = (1/K)∑i p(i) and MI = H[p̄] − (1/K)∑i H[p(i)].

Gating rule
- Base pass: if m ≥ mthresh and Hbase ≤ Hthresh, output argmax(pbase).
- MC pass (when uncertain): compute MI from K passes. If MI ≤ MIthresh, output argmax(pbase). Else sample from p̄ with adaptive temperature T(MI) and/or nucleus p(MI). We use probability averaging by default; we ablate logit averaging.

Adaptive sampling schedules
- T(MI) = Tmin + (Tmax − Tmin) · clip(MI / MI95, 0, 1), where MI95 is a dev-set percentile.
- Analogous schedule for top-p. Schedules are monotone and calibrated on dev sets.

Compute and latency
- Target MC invocation rate s via quantile thresholds on (m, Hbase).
- Small K (2–4) balances stability and cost.
- Expected per-token passes ≈ 1 + s·K. We match latency to baselines within ±5% by tuning s, K, MIthresh.

KV cache and partial MC-dropout
UGD reuses the KV cache from the base pass for efficiency. This yields a partial MC-dropout: only computations for the current token are stochastic; cached keys/values from prior tokens were produced without dropout. This is an approximation that can bias MI downwards. We will:
- Quantify the approximation with ablations: (i) full recompute from scratch with dropout for MC passes, (ii) recompute a suffix of L top layers (mc_suffix_depth), (iii) reuse cache (default).
- Report accuracy, MI–error AUROC, and latency for each option to justify the default.

Implementation details
- Enabling dropout: Many Transformer blocks use functional dropout (F.dropout) gated by module.training. We therefore set model.train(True) during MC passes and model.eval() otherwise, under torch.no_grad(). Modern LLMs use LayerNorm/RMSNorm (no train/eval differences), so this is safe; we document any exceptions.
- RNG: Ensure independent masks across MC passes by advancing RNG state; for batched MC (size K), replicate the step inputs along batch dimension to get independent elementwise dropout.
- Numerics: Compute entropies and MI in float32 with log_softmax for stability. To save memory, accumulate ∑ H[p(i)] online without storing full p(i).
- Reproducibility: Fix seeds, set deterministic CUDA where feasible, record library/hardware versions.
- Models lacking dropout: When dropout=0, we evaluate alternatives (attention dropout enablement if available; micro-ensembles via LoRA; SWAG). We report their cost/benefit relative to MC-dropout.

Pseudocode
```python
@torch.no_grad()
def ugd_step(model, x, y_prefix, cache):
    # 1) Base pass (dropout off)
    model.eval()
    z_base, cache = model.forward_step(x, y_prefix, cache)
    p_base = softmax(z_base.float())
    m, H_base = top2_margin(p_base), entropy(p_base)

    if m >= m_thresh and H_base <= H_thresh:
        return argmax(p_base), cache

    # 2) MC passes (approximate; reuse KV cache by default)
    model.train()  # enable dropout across functional and module paths
    ent_sum = 0.0
    p_bar = 0.0
    for _ in range(K):
        torch.seed()  # or advance a Generator to vary masks
        z_i, _ = model.forward_step(x, y_prefix, cache)  # reuse cache
        p_i = softmax(z_i.float())
        p_bar += p_i
        ent_sum += entropy(p_i)
    model.eval()

    p_bar /= K
    MI = entropy(p_bar) - ent_sum / K

    if MI <= MI_thresh:
        return argmax(p_base), cache

    T = schedule_T(MI)
    p_hat = apply_nucleus(p_bar, p=schedule_p(MI))
    return sample(p_hat, T=T), cache
```

Evaluation (falsification plan)

Goals
1) Test if UGD improves task accuracy over baselines at matched or lower latency. 2) Verify MI adds value beyond entropy/margin. 3) Validate the KV-cache approximation. 4) Assess uncertainty quality.

Models
TinyLlama-1.1B, Phi-2 (2.7B), Llama-2-7B, Mistral-7B.

Datasets and metrics
- Code: HumanEval (pass@1), MBPP (EM).
- Factual QA: TriviaQA (EM/F1), TruthfulQA (truthfulness).
- Reasoning: GSM8K (accuracy).
- Agentic/function calling: a tool-use dataset (e.g., ToolBench or Gorilla) with exactness and API selection accuracy.
- Efficiency: wall-clock latency, tokens/sec, MC invocation rate s.
- Uncertainty quality: token-level error detection AUROC/AUPRC (using negative log-prob of ground-truth token as error), ECE for token probabilities, correlation of MI/entropy with errors, gating curves (performance vs s, K).

Baselines
- Greedy; temperature sampling; nucleus sampling.
- Entropy-only gating (no MC).
- Self-consistency (N samples, vote/majority) at matched latency.
- Ablations: UGD with varying K; sampling from pbase vs p̄; fixed vs adaptive T; logit vs probability averaging; KV-cache reuse vs suffix vs full recompute.

Protocol
- Matched latency: tune each baseline to target latency; tune UGD (s, K, MIthresh) to within ±5% of that latency.
- Calibration: set thresholds by dev-set quantiles to hit a target s (e.g., 0.1–0.3).
- Seeds: 5 runs for stochastic methods; report mean±95% CI; paired bootstrap for significance.

Hypotheses
- H1: UGD > greedy and standard sampling on code tasks at matched latency by targeted exploration.
- H2: UGD improves TruthfulQA truthfulness vs sampling baselines by avoiding unnecessary exploration on confident statements.
- H3: UGD matches/exceeds GSM8K accuracy of baselines at similar latency.
- H4: Full UGD > entropy-only gating at same s, demonstrating added value of epistemic MI.
- H5: KV-cache reuse yields most of the gains at a fraction of the cost; suffix recompute narrows any residual gap.

Relation to prior work
- MC dropout for uncertainty (Gal & Ghahramani, 2016); BALD for epistemic uncertainty.
- Uncertainty for text generation and hallucination detection via sample disagreement (e.g., SelfCheckGPT) and ensembles; calibration for LLMs.
- Adaptive decoding via entropy/top-p/temperature schedules. Distinctively, UGD uses MI-driven, per-token gating with a small number of MC passes and no retraining.

Discussion and impact
UGD separates aleatoric (entropy) from epistemic (MI) uncertainty and targets exploration where the model is likely wrong, preserving determinism elsewhere. In agentic workflows, MI can trigger sampling only at decision points (e.g., tool selection) or escalate to retrieval/verification when MI is high.

Limitations
- Partial MC-dropout with cache reuse is an approximation; we quantify its bias and cost trade-offs.
- Small K yields noisy MI; tight latency budgets may limit benefits.
- Hyperparameters are task/model dependent; calibration drift across domains is possible.
- Models without dropout require alternatives (e.g., attention dropout, LoRA micro-ensembles, SWAG), which may alter latency/benefit profiles.
- Safety-critical contexts may prohibit any sampling; MI could instead trigger external verification or abstention.

Conclusion
UGD is a simple, retraining-free scheduler that uses MC-dropout to gate per-token decoding between greedy and sampling modes. By invoking a few stochastic passes only when cheap proxies indicate uncertainty, UGD concentrates exploration where it matters. The proposed experiments will rigorously test performance, latency, and uncertainty quality, and will quantify the effects of KV-cache approximations to establish practical viability.
