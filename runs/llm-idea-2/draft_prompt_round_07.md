You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.
        Field: general

        Constraints of Paper:
        From: constraints/llm.md

- Research focused on Large Language Model inference
- Very impactful on quality, performance, or agentic workflows
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models


        Draft:
        Major Revisions

Brief critique:
- The core proposal is novel and impactful for LLM inference, with a strong validation plan using code and small open-source models, positioning it as publishable in a venue like ICLR or NeurIPS. However, the method's novelty needs stronger evidence: expand positioning against closely related works like DEEPSEEK's adaptive decoding or other MI-based uncertainty methods in LLMs (e.g., recent papers on dropout ensembles for calibration), with explicit differentiators (e.g., why per-token MI gating outperforms sample disagreement in latency-constrained settings). The evaluation lacks specificity on agentic impact—elaborate the ToolBench/Gorilla setup with metrics for workflow efficiency (e.g., end-to-end task success rate under latency budgets). Ablations are good but under-emphasize key risks: add sensitivity analysis for K=1 (degenerate case) and domain shift (e.g., cross-dataset calibration). Limitations section is solid but should discuss potential compute overhead in production (e.g., RNG and mode switching costs on GPUs). Pseudocode has minor errors (e.g., torch.seed() is deprecated; use torch.manual_seed or a Generator; clarify cache handling in batched MC). Overall, tighten prose for conciseness (e.g., merge redundant sections) and ensure all hypotheses are falsifiable with clear success criteria.

Revised Draft
# Uncertainty-Gated Decoding: Per-Token Mode Switching via MC-Dropout for Large Language Models

## Abstract
We propose Uncertainty-Gated Decoding (UGD), a per-token scheduler that switches between greedy and sampling modes based on epistemic uncertainty estimated with Monte Carlo (MC) dropout. A cheap proxy (e.g., entropy or top-2 margin) decides when to invoke K additional dropout-enabled passes to estimate mutual information (MI) between the predictive and posterior distributions. If MI is high, UGD samples from an MC-averaged distribution; otherwise, it decodes greedily. UGD is retraining-free, tunable for latency, and compatible with small open-source models. We present a falsification plan across code generation, factual QA, arithmetic reasoning, and agentic tasks, with matched-latency comparisons to strong decoding baselines and uncertainty-only gating. We quantify the accuracy–latency trade-off, the predictive power of MI for error detection, and the effect of KV-cache approximations.

## Introduction
Decoding controls LLM behavior: greedy is fast and brittle; sampling boosts diversity at the cost of factuality and determinism. Heuristic adaptivity (e.g., entropy thresholds) conflates aleatoric and epistemic uncertainty. MC dropout approximates Bayesian inference by sampling stochastic forward passes; disagreement (BALD) isolates epistemic uncertainty. We operationalize this at the token level to gate exploration. Unlike prior approaches relying on multiple diverse samples or ensembles for detection (e.g., hallucinations via SelfCheckGPT or DEEPSEEK's adaptive sampling), UGD uses a small number of MC passes to decide when and how intensely to sample, without retraining. This enables targeted exploration in high-uncertainty regions, improving quality in agentic workflows (e.g., selective tool calling) while preserving low latency.

## Method

### Problem Setting
Given pθ(y | x) and prefix y1:t−1, choose yt by either greedy argmax or sampling from a derived distribution q(· | x, y1:t−1).

### Uncertainty Signals
- Cheap proxies (single pass, dropout off): entropy Hbase and top-2 margin m = p(1) − p(2).
- Epistemic estimate (MC-dropout): run K stochastic passes with dropout on, obtain p(i), compute p̄ = (1/K)∑i p(i) and MI = H[p̄] − (1/K)∑i H[p(i)].

### Gating Rule
- Base pass: if m ≥ mthresh and Hbase ≤ Hthresh, output argmax(pbase).
- MC pass (when uncertain): compute MI from K passes. If MI ≤ MIthresh, output argmax(pbase) (fallback rationale: low MI implies low epistemic uncertainty despite high entropy, e.g., aleatoric cases like synonyms). Else sample from p̄ with adaptive temperature T(MI) and/or nucleus p(MI). We use probability averaging by default and ablate logit averaging (rationale: prob avg preserves calibration; logit avg may amplify low-prob modes).

### Adaptive Sampling Schedules
- T(MI) = Tmin + (Tmax − Tmin) · clip(MI / MI95, 0, 1), where MI95 is a dev-set percentile.
- Analogous schedule for top-p. Schedules are monotone and calibrated on dev sets; we ablate fixed vs. adaptive to justify.

### Compute and Latency
- Target MC invocation rate s via quantile thresholds on (m, Hbase).
- Small K (2–4) balances stability and cost; we ablate K=1 (degenerate, no MI) for sensitivity.
- Expected per-token passes ≈ 1 + s·K. We match latency to baselines within ±5% by tuning s, K, MIthresh.

### KV Cache and Partial MC-Dropout
UGD reuses the KV cache from the base pass for efficiency. This yields a partial MC-dropout: only computations for the current token are stochastic; cached keys/values from prior tokens were produced without dropout. This is an approximation that can bias MI downwards. We will:
- Quantify the approximation with ablations: (i) full recompute from scratch with dropout for MC passes, (ii) recompute a suffix of L top layers (mc_suffix_depth), (iii) reuse cache (default).
- Report accuracy, MI–error AUROC, and latency for each option to justify the default.

### Implementation Details
- Enabling dropout: Many Transformer blocks use functional dropout (F.dropout) gated by module.training. We therefore set model.train(True) during MC passes and model.eval() otherwise, under torch.no_grad(). Modern LLMs use LayerNorm/RMSNorm (no train/eval differences), so this is safe; we document any exceptions and test for residual dropout paths.
- RNG: Ensure independent masks across MC passes by using a torch.Generator and advancing its state (e.g., manual_seed per pass); for batched MC (size K), replicate the step inputs along batch dimension to get independent elementwise dropout.
- Numerics: Compute entropies and MI in float32 with log_softmax for stability. To save memory (avoid storing full vocab K times), accumulate ∑ H[p(i)] and p̄ online without storing full p(i).
- Reproducibility: Fix seeds, set deterministic CUDA where feasible, record library/hardware versions.
- Models lacking dropout: When dropout=0, we evaluate alternatives (attention dropout enablement if available; micro-ensembles via LoRA; SWAG). We report their cost/benefit relative to MC-dropout.
- Additional: For production, note RNG/mode switching overhead on GPUs; we measure and report.

## Pseudocode
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
    p_bar = torch.zeros_like(p_base)
    gen = torch.Generator(device=p_base.device)
    for i in range(K):
        gen.manual_seed(42 + i)  # or advance state for varying masks
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

## Evaluation (Falsification Plan)

### Goals
1) Test if UGD improves task accuracy over baselines at matched or lower latency. 2) Verify MI adds value beyond entropy/margin. 3) Validate the KV-cache approximation. 4) Assess uncertainty quality. 5) Evaluate agentic workflow impact.

### Models
TinyLlama-1.1B, Phi-2 (2.7B), Llama-2-7B, Mistral-7B.

### Datasets and Metrics
- Code: HumanEval (pass@1), MBPP (EM).
- Factual QA: TriviaQA (EM/F1), TruthfulQA (truthfulness).
- Reasoning: GSM8K (accuracy).
- Agentic/function calling: ToolBench or Gorilla, with end-to-end task success rate, API selection accuracy, and workflow efficiency (e.g., steps to completion under latency budgets).
- Efficiency: wall-clock latency, tokens/sec, MC invocation rate s.
- Uncertainty quality: token-level error detection AUROC/AUPRC (using negative log-prob of ground-truth token as error), ECE for token probabilities, correlation of MI/entropy with errors, gating curves (performance vs s, K), matched-latency curves.
- Additional: Cross-dataset domain shift (e.g., calibrate on TriviaQA, test on GSM8K) to assess robustness.

### Baselines
- Greedy; temperature sampling; nucleus sampling.
- Entropy-only gating (no MC).
- Self-consistency (N samples, vote/majority) at matched latency.
- Ablations: UGD with varying K (incl. K=1); sampling from pbase vs p̄; fixed vs adaptive T; logit vs probability averaging; KV-cache reuse vs suffix vs full recompute.

### Protocol
- Matched latency: tune each baseline to target latency; tune UGD (s, K, MIthresh) to within ±5% of that latency.
- Calibration: set thresholds by dev-set quantiles to hit a target s (e.g., 0.1–0.3).
- Seeds: 5 runs for stochastic methods; report mean±95% CI; paired bootstrap for significance.

### Hypotheses
- H1: UGD > greedy and standard sampling on code tasks at matched latency by targeted exploration (success: +5% pass@1).
- H2: UGD improves TruthfulQA truthfulness vs sampling baselines by avoiding unnecessary exploration on confident statements (success: +10% truthfulness).
- H3: UGD matches/exceeds GSM8K accuracy of baselines at similar latency (success: parity or better).
- H4: Full UGD > entropy-only gating at same s, demonstrating added value of epistemic MI (success: +3% avg accuracy).
- H5: KV-cache reuse yields most of the gains at a fraction of the cost; suffix recompute narrows any residual gap (success: <2% accuracy drop vs full, >50% latency savings).
- H6: UGD enhances agentic success rate by 10% at matched latency via MI-gated decisions.

## Relation to Prior Work
- MC dropout for uncertainty (Gal & Ghahramani, 2016); BALD for epistemic uncertainty.
- Uncertainty for text generation and hallucination detection via sample disagreement (e.g., SelfCheckGPT) and ensembles; calibration for LLMs.
- Adaptive decoding via entropy/top-p/temperature schedules (e.g., DEEPSEEK). Distinctively, UGD uses MI-driven, per-token gating with a small number of MC passes and no retraining, outperforming disagreement-based methods in latency by focusing on epistemic signals without full-sample generation.

## Discussion and Impact
UGD separates aleatoric (entropy) from epistemic (MI) uncertainty and targets exploration where the model is likely wrong, preserving determinism elsewhere. In agentic workflows, MI can trigger sampling only at decision points (e.g., tool selection) or escalate to retrieval/verification when MI is high, potentially reducing errors in multi-step tasks.

## Limitations
- Partial MC-dropout with cache reuse is an approximation; we quantify its bias and cost trade-offs.
- Small K yields noisy MI; tight latency budgets may limit benefits.
- Hyperparameters are task/model dependent; calibration drift across domains is possible (we test cross-dataset).
- Models without dropout require alternatives (e.g., attention dropout, LoRA micro-ensembles, SWAG), which may alter latency/benefit profiles.
- Safety-critical contexts may prohibit any sampling; MI could instead trigger external verification or abstention.
- Production overhead: RNG and train/eval switching add minor GPU costs; not suitable for ultra-low-latency edge devices.

## Conclusion
UGD is a simple, retraining-free scheduler that uses MC-dropout to gate per-token decoding between greedy and sampling modes. By invoking a few stochastic passes only when cheap proxies indicate uncertainty, UGD concentrates exploration where it matters. The proposed experiments will rigorously test performance, latency, and uncertainty quality, and will quantify the effects of KV-cache approximations to establish practical viability.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
