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
        Minor Revisions

Brief critique:
The manuscript is exceptionally strong, presenting a clear, well-motivated, and methodologically sound research plan. It has thoroughly addressed all points from the previous major revision request. The proposed method (UGD) is novel, practical, and the evaluation plan is rigorous, featuring strong baselines, falsifiable hypotheses, and a clear-eyed view of implementation challenges (e.g., stochasticity sources, KV cache). The remaining revisions are minor and focus on refining methodological clarity and tightening the experimental plan.

- Methodological clarity: The adaptive sampling step (nucleus/temperature schedules) is mentioned but not fully defined. Clarify how `adapt_nucleus` and `schedule_p` are implemented. The adaptive-K update rule in the pseudocode is slightly complex; a simpler, more standard exponential increase could be easier to analyze and implement.
- Pseudocode: The handling of the KV cache in the batched Monte Carlo step is abstracted away. A brief note in the "Implementation notes" section on how the cache is replicated or shared across the K parallel passes would enhance reproducibility.
- Hypotheses: Hypothesis H3 ("parity or better") is less precise than the others. Specify a concrete margin for "parity" (e.g., within a 2% margin of error of the baseline) to maintain the high standard of falsifiability set by the other hypotheses.

This is a well-designed study poised for high impact. The proposed revisions will further sharpen its already impressive clarity and rigor.

***

### Revised Draft
# Uncertainty-Gated Decoding: Per-Token MI-Guided Mode Switching for Efficient LLM Inference

## Abstract
We introduce Uncertainty-Gated Decoding (UGD), a per-token scheduler that adaptively switches between efficient deterministic decoding and controlled stochastic sampling. UGD uses cheap single-pass proxies to identify low-uncertainty tokens, which are decoded greedily. For remaining tokens, it runs a small number of stochastic forward passes (via dropout or micro-ensembles) to estimate token-level mutual information (MI), a proxy for epistemic uncertainty. High-MI tokens trigger sampling from the Monte Carlo-averaged distribution; low-MI tokens default to greedy decoding. UGD is retraining-free, tunable to tight latency budgets, and compatible with small open-source models. We detail a code-validatable plan across code generation, QA, reasoning, and agentic function-calling, with matched-latency comparisons against strong baselines like self-consistency and speculative decoding. We evaluate accuracy–latency trade-offs, uncertainty quality, and the impact of practical KV-cache approximations and cost-control mechanisms.

## Contributions
- Per-token MI gating under latency budgets: a practical, retraining-free scheduler that invokes a small number of stochastic passes only where needed.
- KV-cache-aware epistemic estimation: quantify the bias/benefit of partial Monte Carlo with cache reuse vs suffix/full recompute.
- Cost control mechanisms: top-M MI computation and adaptive-K passes to bound overhead while preserving gains.
- Agentic integration: MI signals gated at decision tokens (e.g., tool selection) to improve end-to-end task success under latency constraints.
- Strong baselines and falsifiable criteria with small open models; open-source implementation suitable for replication.

## Method

### Problem
Given an LLM pθ(y | x) and prefix y1:t−1, decide the next token yt via either greedy decoding or controlled sampling. We aim to sample only when epistemic uncertainty is high, otherwise defaulting to the efficient greedy path.

### Uncertainty signals
- Cheap single-pass proxies (dropout off): token entropy Hbase and top-2 probability margin m = p(1) − p(2).
- Epistemic estimate (token MI via MC): run K stochastic passes with perturbations that approximate Bayesian model uncertainty (e.g., dropout or micro-ensembles). Let p(i) be the per-pass token distribution. Define the MC-averaged distribution p̄ = (1/K)∑i p(i). Token-level MI (BALD) is MI = H[p̄] − (1/K)∑i H[p(i)], computed on an active set of top-M base logits to control cost.

### Gating and sampling
- Stage 1 (cheap gate): If margin is high and entropy low (m ≥ mthresh and Hbase ≤ Hthresh), output argmax(pbase).
- Stage 2 (MI gate): Otherwise, compute MI on the active set. If MI ≤ MIthresh, output argmax(pbase). If MI > MIthresh, sample from p̄. We use standard nucleus sampling, where the nucleus mass `p` and/or temperature `T` are adaptive functions of the MI value (e.g., `p(MI) = min(0.95, p_base + c*MI)`). We default to probability averaging (p̄) and ablate logit averaging.

### Adaptive-K and top-M
- Adaptive-K: Start with Kmin (e.g., 2). If the MI estimate is near the threshold (e.g., within δ of MIthresh) or unstable (high variance across passes), increase K up to Kmax. This reduces unnecessary passes on clearly low- or high-uncertainty tokens.
- Top-M: Compute MI over the top-M tokens from the base pass (e.g., M ∈ {50, 200}), with probability mass renormalization. The final sample is drawn from this M-dimensional distribution and mapped back to the original vocabulary index. We ablate M for accuracy/latency.

### Complexity and latency
- Expected per-token forward passes ≈ 1 + s·E[K], where s is the fraction of tokens triggering MI. We tune (mthresh, Hthresh, MIthresh, Kmin/Kmax, M) to meet a target latency within ±5% of baselines.
- We report tokens/sec and wall-clock time including mode-switch overhead.

### KV-cache approximations
We study three variants:
1) Full recompute: recompute from start with stochasticity enabled (upper bound on quality; highest cost).
2) Suffix recompute: recompute only last D layers (mc_suffix_depth) with stochasticity; reuse lower-layer cache.
3) Cache reuse (default): reuse full prefix cache; only the current token's forward pass is stochastic.
We compare accuracy, AUROC for error detection, MI calibration, and latency to justify defaults.

## Practical stochasticity sources
- MC-dropout: Activate dropout modules at inference. Many small models (e.g., GPT-2 variants) have residual/attn dropout; others (e.g., LLaMA/Mistral) set p=0. For the latter, we provide:
  - Attention-dropout injection: enable attention probability dropout during MC passes.
  - Micro-ensembles: 2–4 lightweight LoRA heads or SWA-like checkpoints; average probabilities to approximate p̄. We report cost/benefit vs dropout.
- Safety: LayerNorm/RMSNorm behave identically in train/eval. We ensure no optimizer/buffer updates via `torch.no_grad`.

## Implementation notes
- RNG: Do not reseed per pass. Independence arises by consuming RNG state.
- Batched MC: To perform K passes efficiently, replicate the input state K times across the batch dimension. The shared prefix KV cache is likewise replicated. Dropout or micro-ensemble variations then yield independent stochastic computation paths for each item in the batch.
- Numerics: Compute entropies in float32 with log_softmax. Accumulate ∑ H[p(i)] online; avoid materializing the full vocabulary for K passes by using the top-M mask.
- Backends: For paged KV caches (e.g., vLLM), we measure and report any overhead from toggling model state (e.g., dropout on/off), which can trigger kernel recompilations.

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

    # Prepare active set for efficient MI computation
    topv, topi = torch.topk(p_base, cfg.top_M)   # shapes: [M]
    active_mask = torch.zeros_like(p_base, dtype=torch.bool)
    active_mask.scatter_(0, topi, True)

    # Adaptive-K loop
    K = cfg.K_min
    MI, last_MI = None, None
    while True:
        # Batched MC passes (size K)
        model.set_stochastic(True)           # dropout on or ensemble enable
        # Note: forward_step_batched replicates cache K times internally
        z_mc, _ = model.forward_step_batched(x, y_prefix, cache, batch_size=K)
        p_mc = softmax(z_mc.float())         # [K, V]

        p_mc_active = renorm(p_mc[:, active_mask])  # [K, M]
        p_bar = p_mc_active.mean(dim=0)             # [M]
        ent_bar = entropy(p_bar)
        ent_mean = entropy(p_mc_active).mean()
        MI = ent_bar - ent_mean

        # Stopping rule
        if MI > cfg.MI_thresh or K >= cfg.K_max:
            break
        if last_MI is not None and abs(MI - last_MI) < cfg.delta_MI: # converged
            break
        last_MI = MI
        K = min(K * 2, cfg.K_max)

    # Decision
    if MI <= cfg.MI_thresh:
        return argmax(p_base), cache

    # Sample from the MC-averaged distribution over the active set
    p_final = schedule_nucleus(p_bar, p=schedule_p(MI, cfg))
    T = schedule_T(MI, cfg)
    token_idx = sample(p_final, T=T)         # sample from M-dim distribution
    return topi[token_idx], cache           # map back to original vocab
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
- Uncertainty quality: token-level error detection AUROC/AUPRC (using gold next-token), ECE, correlation of MI with errors, calibration under domain shift.
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
- H3 (GSM8K): Not significantly worse than baselines (within a 2% margin) at matched latency, demonstrating no regression on complex reasoning.
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
- Partial MC with cache reuse biases MI; we quantify this empirically but lack a closed-form correction.
- Backend constraints: toggling model state and controlling RNG may incur overhead in certain runtimes.

## Ethics and safety
In safety-critical settings, prefer MI to trigger abstention or external verification rather than sampling. We will release code, configs, and evaluation scripts to promote reproducibility and scrutiny.

## Conclusion
UGD is a simple, practical scheduler that uses lightweight Monte Carlo estimates of epistemic uncertainty to gate per-token decoding. With top-M MI, adaptive-K, and KV-cache-aware computation, UGD can improve accuracy and agentic success at matched latency and is readily validated on small open models. We outline a rigorous experimental plan to test gains, quantify costs, and bound approximations, enabling adoption in real-world LLM inference.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
