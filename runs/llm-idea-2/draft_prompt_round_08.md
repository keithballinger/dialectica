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
- Strengths: Clear problem framing, plausible impact on LLM inference and agentic workflows, and a concrete falsification plan using small open-source models. The per-token MI gating idea, adaptive schedules, and KV-cache approximations are practical and code-validatable.
- Novelty/positioning: Needs stronger differentiation from prior adaptive decoding and uncertainty methods (e.g., BALD at sequence/token level, SelfCheckGPT, DEEPSEEK adaptive decoding, calibration/gating by entropy/margin, lightweight ensembles). Explicitly claim what is new (per-token MI gating with partial MC under KV cache, top-M MI, adaptive-K) and why it is preferable under tight latency budgets.
- Practicality risks: Many popular LLMs disable dropout at inference and/or lack dropout modules; relying on model.train() may have no effect. Provide a concrete, safe recipe for enabling stochasticity (attention dropout injection or lightweight micro-ensembles) and quantify overhead. Clarify that PyTorch RNG control for dropout masks is nontrivial and fix pseudocode accordingly.
- Methodological clarity: Formalize the MI estimator, top-M truncation to control cost, adaptive-K policy, and complexity under paged KV caches. Justify probability vs logit averaging more rigorously (calibration evidence).
- Baselines/metrics: Add strong inference baselines (speculative decoding, Rejection/Medusa, draft-verify/self-check, self-consistency at matched latency). Specify agentic setups and success metrics more concretely (tool-call token accuracy, tool latency budgets, repair rates). Provide numeric success criteria per hypothesis.
- Pseudocode issues: Seeding and dropout independence, batched MC passes, no per-pass generator needed, top-M masking, and explicit cache handling. Train/eval toggling may incur kernel recompilations in some backends; measure this.
- Analysis gaps: Add adaptive-K ablation, top-M sensitivity, domain-shift robustness, and scenarios where MI harms performance (aleatoric cases). Discuss synergy/conflict with speculative decoding and KV-quantization. 

Revised Draft
# Uncertainty-Gated Decoding: Per-Token MI-Guided Mode Switching for Efficient LLM Inference

## Abstract
We introduce Uncertainty-Gated Decoding (UGD), a per-token scheduler that switches between deterministic and stochastic decoding based on epistemic uncertainty estimated via lightweight Monte Carlo (MC) perturbations. UGD uses cheap single-pass proxies to identify uncertain prefixes and, only then, runs K additional stochastic forward passes to estimate token-level mutual information (MI) between the MC-averaged predictive and individual predictive distributions. High-MI tokens trigger controlled sampling from the MC-averaged distribution; low-MI tokens decode greedily. UGD is retraining-free, tunable to tight latency budgets, and compatible with small open-source models via dropout or micro-ensembles. We detail a code-validatable plan across code generation, factual QA, arithmetic reasoning, and agentic function-calling, with matched-latency comparisons against strong inference baselines (entropy-gating, self-consistency, speculative decoding). We evaluate accuracy–latency trade-offs, uncertainty quality, and the impact of KV-cache approximations and top-M truncation.

## Contributions
- Per-token MI gating under latency budgets: a practical, retraining-free scheduler that invokes a small number of stochastic passes only where needed.
- KV-cache-aware epistemic estimation: quantify the bias/benefit of partial MC with cache reuse vs suffix/full recompute.
- Cost control mechanisms: top-M MI computation and adaptive-K to bound overhead while preserving gains.
- Agentic integration: MI signals gated at decision tokens (e.g., tool selection) to improve end-to-end task success under latency constraints.
- Strong baselines and falsifiable criteria with small open models; open-source implementation suitable for replication.

## Method

### Problem
Given an LLM pθ(y | x) and prefix y1:t−1, decide yt via either greedy decoding or sampling. We aim to sample only when epistemic uncertainty is high.

### Uncertainty signals
- Cheap single-pass proxies (dropout off): entropy Hbase and top-2 margin m = p(1) − p(2).
- Epistemic estimate (token MI via MC): run K stochastic passes with perturbations that approximate Bayesian model uncertainty (e.g., dropout or micro-ensembles). Let p(i) be the per-pass token distribution. Define p̄ = (1/K)∑i p(i). Token-level MI (BALD) is MI = H[p̄] − (1/K)∑i H[p(i)], computed on an active set of top-M base logits to control cost.

### Gating and sampling
- Stage 1 (cheap gate): If margin is high and entropy low (m ≥ mthresh and Hbase ≤ Hthresh), output argmax(pbase).
- Stage 2 (MI gate): Otherwise, compute MI on the active set. If MI ≤ MIthresh, output argmax(pbase). If MI > MIthresh, sample from p̄ using an adaptive schedule T(MI) and/or p(MI). We default to probability averaging (p̄) and ablate logit averaging.

### Adaptive-K and top-M
- Adaptive-K: Start with Kmin (e.g., 2). If the MI estimate is near the threshold (e.g., within δ of MIthresh) or unstable (high variance across passes), increase K up to Kmax. This reduces unnecessary passes on clearly low- or high-uncertainty tokens.
- Top-M: Compute MI over the top-M tokens from the base pass (e.g., M ∈ {50, 200}), with mass renormalization. We ablate M for accuracy/latency.

### Complexity and latency
- Expected per-token forward passes ≈ 1 + s·E[K], where s is the fraction of tokens triggering MI. Tune (mthresh, Hthresh, MIthresh, Kmin/Kmax, M) to meet a target latency within ±5% of baselines.
- We report tokens/sec and wall-clock time including mode-switch overhead.

### KV-cache approximations
We study three variants:
1) Full recompute: recompute from start with stochasticity enabled (upper bound on quality; highest cost).
2) Suffix recompute: recompute only last D layers (mc_suffix_depth) with stochasticity; reuse lower-layer cache.
3) Cache reuse (default): reuse full cache; only current token path is stochastic.
We compare accuracy, AUROC for error detection, MI calibration, and latency to justify defaults.

## Practical stochasticity sources
- MC-dropout: Activate dropout modules at inference. Many small models (e.g., GPT-2 variants) have residual/attn dropout; others (e.g., LLaMA/Mistral) set p=0. For the latter, we provide:
  - Attention-dropout injection: enable attention probability dropout during MC passes if kernels support it.
  - Micro-ensembles: 2–4 lightweight LoRA heads or SWAG samples; average probabilities to approximate p̄. We report cost/benefit vs dropout.
- Safety: LayerNorm/RMSNorm behave identically in train/eval. Ensure no optimizer/buffer updates; use no_grad.

## Implementation notes
- RNG: Do not reseed per pass. Independence arises by consuming RNG state. For batched MC (size K), replicate the input across batch dimension; dropout/micro-ensemble paths yield independent masks/weights per element.
- Numerics: Compute entropies in float32 with log_softmax. Accumulate ∑ H[p(i)] online; avoid materializing full vocab for K passes via top-M masking.
- Backends: For paged KV caches (e.g., vLLM), measure train/eval toggling overhead. Prefer persistent “MC mode” for runs of tokens to reduce kernel recompilation.
- Reproducibility: Fixed seeds, five-run repeats, hardware/library versions, deterministic flags where feasible.

## Pseudocode (per-step, batched MC with top-M and adaptive-K)
```python
@torch.no_grad()
def ugd_step(model, x, y_prefix, cache, cfg):
    # Base pass (deterministic)
    model.set_stochastic(False)              # impl: dropout off, ensemble=None
    z_base, cache = model.forward_step(x, y_prefix, cache)
    p_base = softmax(z_base.float())
    m, Hb = top2_margin(p_base), entropy(p_base)

    if m >= cfg.m_thresh and Hb <= cfg.H_thresh:
        return argmax(p_base), cache

    # Prepare active set
    topv, topi = torch.topk(p_base, cfg.top_M)   # shapes: [M]
    active_mask = torch.zeros_like(p_base, dtype=torch.bool)
    active_mask[topi] = True

    # Adaptive-K loop
    K = cfg.K_min
    MI, last_MI = None, None
    while True:
        # Batched MC passes (size K)
        model.set_stochastic(True)           # dropout on or ensemble enable
        z_mc, _ = model.forward_step_batched(x, y_prefix, cache, batch_size=K)
        p_mc = softmax(z_mc.float())         # [K, V]
        if cfg.top_M is not None:
            p_mc = renorm(p_mc[:, active_mask])  # [K, M]

        p_bar = p_mc.mean(dim=0)             # [M]
        ent_bar = entropy(p_bar)
        ent_mean = entropy(p_mc).mean()
        MI = ent_bar - ent_mean

        # Stopping rule
        if MI > cfg.MI_thresh or K >= cfg.K_max:
            break
        if last_MI is not None and abs(MI - last_MI) < cfg.delta_MI:
            break
        last_MI = MI
        K = min(cfg.K_max, max(K + cfg.K_step, 2*K))

    # Decision
    if MI <= cfg.MI_thresh:
        return argmax(p_base), cache

    T = schedule_T(MI, cfg)
    p_hat = adapt_nucleus(p_bar, p=schedule_p(MI, cfg))
    token = sample(p_hat, T=T)               # sample within active set
    return map_back(token, topi), cache
```

## Evaluation

### Models
TinyLlama-1.1B, Phi-2 (2.7B), LLaMA-2-7B, Mistral-7B. For models lacking dropout, use injected attention dropout or 2–4 micro-ensemble heads.

### Tasks and metrics
- Code: HumanEval (pass@1), MBPP (EM).
- Factual QA: TriviaQA (EM/F1), TruthfulQA (truthfulness).
- Reasoning: GSM8K (accuracy).
- Agentic/function calling: ToolBench/Gorilla-style tasks. Metrics: end-to-end success, tool-call token accuracy, tool/API selection accuracy, mean steps to completion, and success at fixed latency budgets.
- Efficiency: wall-clock latency, tokens/sec, s (MI invocation rate), E[K], and total extra passes per generated token.
- Uncertainty quality: token-level error detection AUROC/AUPRC (using gold next-token where available), ECE, correlation of MI/entropy with errors, calibration under domain shift.
- Robustness: cross-dataset calibration (calibrate on TriviaQA, test on GSM8K) and prompt/domain shifts.

### Baselines
- Greedy; temperature and nucleus sampling.
- Entropy-only gating (no MC).
- Self-consistency (N samples, vote/majority) at matched latency.
- Speculative decoding (e.g., draft-and-verify) tuned to matched latency.
- Medusa/Rejection sampling variants when available.
- Ablations: UGD with K ∈ {1,2,4,8}; top-M ∈ {50, 200, full}; prob vs logit averaging; fixed vs adaptive schedules; KV reuse vs suffix vs full recompute; dropout vs micro-ensembles.

### Protocol and hypotheses (falsifiable, matched latency ±5%)
- H1 (code): +5% absolute pass@1 vs greedy and standard sampling at matched latency.
- H2 (TruthfulQA): +10% truthfulness vs temperature/nucleus by avoiding unnecessary exploration.
- H3 (GSM8K): parity or better vs baselines at similar latency.
- H4 (value of MI): MI-gated > entropy-gated by +3% average accuracy across tasks at same s.
- H5 (KV approximations): Cache reuse within 2% of full recompute accuracy with ≥50% latency savings; suffix recompute closes any residual gap.
- H6 (agentic): +10% end-to-end success at matched latency via MI-gated tool decisions.
All stochastic methods: 5 seeds, mean±95% CI, paired bootstrap for significance.

## Relation to prior work
- MC-dropout and BALD (Gal & Ghahramani, 2016) and uncertainty in text generation/hallucination detection (e.g., SelfCheckGPT, ensembles).
- Adaptive decoding via entropy/temperature/top-p (e.g., DEEPSEEK-style schedules).
- Efficient inference: speculative decoding, self-consistency, Medusa/rejection sampling.
Distinctiveness: UGD provides token-level MI gating under strict latency via top-M MI, adaptive-K, and KV-cache-aware MC. Unlike disagreement methods requiring full alternative generations, UGD focuses compute on local uncertainty and maintains determinism elsewhere.

## Discussion and impact
UGD separates aleatoric (high-entropy, low-MI) from epistemic (high-MI) regions, concentrating exploration where the model is likely to be wrong. In agentic pipelines, MI can trigger sampling or verification precisely at decision tokens, improving robustness without large latency penalties. UGD is complementary to speculative decoding and can gate when to speculate or verify.

## Limitations
- Stochasticity availability: some LLMs lack dropout at inference; injected attention dropout or micro-ensembles add overhead.
- MI noise at small K; aggressive latency targets may limit gains.
- Hyperparameter sensitivity and calibration drift across domains.
- Partial MC with cache reuse biases MI; we quantify empirically but lack closed-form correction.
- Backend constraints: train/eval toggling and RNG control may incur overhead in certain runtimes.

## Ethics and safety
In safety-critical settings, prefer MI to trigger abstention or external verification rather than sampling. We will release code, configs, and evaluation scripts to promote reproducibility and scrutiny.

## Conclusion
UGD is a simple, practical scheduler that uses lightweight MC estimates of epistemic uncertainty to gate per-token decoding. With top-M MI, adaptive-K, and KV-cache-aware computation, UGD improves accuracy and agentic success at matched latency and is readily validated on small open models. We outline a rigorous experimental plan to test gains, quantify costs, and bound approximations, enabling adoption in real-world LLM inference.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
