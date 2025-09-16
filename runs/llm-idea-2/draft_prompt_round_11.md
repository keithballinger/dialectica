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
The paper is strong, novel, and clearly positioned for high impact in LLM inference under realistic latency budgets. The MI-gated decoding idea is well-motivated, the baselines are appropriate, and the evaluation plan is falsifiable and replicable on small open models. Remaining issues are clarifications and small methodological refinements:
- Define the adaptive sampling schedules precisely (functional forms, defaults, and normalization for MI) and specify the stopping rule for Adaptive-K using a confidence-interval or variance check rather than ad-hoc deltas.
- Make top-M MI truncation explicit, including how tail mass is handled and its effect on MI bias, with a simple bound or approximation.
- Expand KV-cache handling with concrete replication/sharing strategies, memory considerations, and vLLM/paged-cache caveats; provide a memory-aware fallback path.
- Tighten the matched-latency protocol and RNG independence across MC passes to ensure fair and reproducible comparisons.
- Slightly broaden related work (e.g., Confident Adaptive Language Modeling) to situate MI gating among other per-token adaptive inference methods.

Revised Draft
# Uncertainty-Gated Decoding: Per-Token MI-Guided Mode Switching for Efficient LLM Inference

## Abstract
We introduce Uncertainty-Gated Decoding (UGD), a per-token scheduler that adaptively switches between greedy decoding and controlled stochastic sampling. UGD uses cheap one-pass signals to skip sampling on low-uncertainty tokens and invokes a small number of Monte Carlo (MC) forward passes to estimate token-level mutual information (MI, BALD) where epistemic uncertainty is high. High-MI tokens trigger sampling from the MC-averaged distribution; low-MI tokens remain greedy. UGD is retraining-free, latency-tunable, and compatible with small open-source models via MC-dropout or micro-ensembles. We provide a code-validatable plan across code generation, QA, reasoning, and agentic function-calling, with matched-latency comparisons against strong baselines (self-consistency, speculative decoding). We analyze accuracy–latency trade-offs, uncertainty quality, and practical KV-cache approximations, and we release implementation details to ensure reproducibility.

## Contributions
- Per-token MI gating under strict latency: a retraining-free scheduler that concentrates stochastic passes where MI is high.
- KV-cache-aware epistemic estimation: quantify the bias/benefit of cache reuse vs suffix/full recompute and provide memory-aware batched MC.
- Cost-control mechanisms: (i) top-M MI with tail-mass handling and a simple bias bound; (ii) CI-based adaptive-K stopping; (iii) MI-driven sampling schedules with closed-form defaults.
- Agentic integration: MI-gated decisions at tool-selection tokens to improve end-to-end success at matched latency.
- Strong baselines and falsifiable criteria on small open-source models; implementation suitable for replication.

## Method

### Setup
Given an LLM pθ(y | x) and prefix y1:t−1, decide yt via greedy decoding or sampling. We want to sample only when epistemic uncertainty is high.

### Uncertainty signals
- Cheap proxies (single pass, dropout off): token entropy Hbase and top-2 margin m = p(1) − p(2).
- Epistemic estimate (token-level BALD MI): run K stochastic passes (dropout or micro-ensembles). Let pi be the per-pass distribution and p̄ = (1/K)∑i pi. Define MI = H[p̄] − (1/K)∑i H[pi], computed on an active top-M vocabulary subset with explicit tail handling (below).

### Top-M truncation and tail handling
- Active set: take top-M tokens under pbase, indices S. Let mass(S) = ∑v∈S pbase(v); tail mass τ = 1 − mass(S).
- For each pass, renormalize ps = pi restricted to S. For MI, we include a single “tail bucket” with probability τ per pass (constant across passes), yielding M+1 categories. This preserves total mass and reduces MI underestimation.
- Bias note: If per-pass rank changes move mass between S and tail, MI is lower-bounded by the truncated estimate. With one tail bucket, MI error ≤ H(τ) − τ log τ when per-pass variation lies entirely in tail; we report τ and ablate M to bound this effect empirically.

### Gating policy
- Stage 1 (cheap gate): If m ≥ m_thresh and Hbase ≤ H_thresh, output argmax(pbase).
- Stage 2 (MI gate): Otherwise compute MI on S∪{tail}. If MI ≤ MI_thresh, output argmax(pbase); else sample from p̄ over S∪{tail} and, if the tail is drawn, resample from the base distribution restricted to V \ S.

### Sampling schedules (deterministic defaults)
Let MI be normalized to [0,1] via MI_norm = clip(MI / MI_max, 0, 1), where MI_max = log(M+1) (natural log) under the truncated support.
- Top-p: p(MI) = clip(p0 + αp · MI_norm, p_min, p_max). Defaults: p0=0.85, αp=0.10, p_min=0.80, p_max=0.95.
- Temperature: T(MI) = clip(T0 + αT · MI_norm, T_min, T_max). Defaults: T0=0.8, αT=0.4, T_min=0.7, T_max=1.2.
We ablate linear vs convex schedules; linear is default for simplicity and stability.

### Adaptive-K with CI-based stopping
We estimate MI with K stochastic passes, starting from Kmin.
- Compute MIK on S∪{tail} and record per-pass entropies H[pi]. Let varH = Var(H[pi]) and varHbar be the sample variance of H[p̄] via a delta-method approximation or a simple bootstrap over passes (B=100 on small M).
- Stop when one of:
  1) The 95% CI around MIK (bootstrap or normal approx) lies entirely above or below MI_thresh.
  2) K ≥ Kmax.
  3) MIK is farther than ε from MI_thresh (|MIK − MI_thresh| ≥ ε) and the CI half-width ≤ ε/2.
- K schedule: K ← min(2K, Kmax) on each iteration. Defaults: Kmin=2, Kmax∈{4,8}, ε=0.02 nats.

### Complexity and latency
- Expected per-token passes ≈ 1 + s · E[K], s = fraction of tokens invoking MI. We tune (m_thresh, H_thresh, MI_thresh, Kmin/Kmax, M) to meet a target latency within ±5% of baselines. We report s, E[K], and total extra passes/token.

### KV-cache approximations and memory
We study:
1) Full recompute (upper-bound quality): recompute from start with stochasticity.
2) Suffix recompute: recompute last D layers with stochasticity; reuse lower-layer cache.
3) Cache reuse (default): reuse full prefix cache; only current-step forward is stochastic.
- Batched MC implementation: replicate the batch dimension K times; replicate the prefix KV pages lazily with view-based references where supported (paged KV) or via explicit copy otherwise.
- Memory: Additional KV memory ≈ (K−1) × cache_prefix if copies are needed. We provide a memory-aware fallback: reduce K or M when memory headroom < η (default η=10%).
- vLLM/paged cache: toggling dropout modifies kernel selection in some backends; we amortize by grouping tokens with identical stochastic state and measure any overhead.

### Stochasticity sources
- MC-dropout: enable dropout modules at inference. For models with dropout=0 (e.g., LLaMA/Mistral), we use:
  - Attention-prob dropout injection during MC passes only.
  - Micro-ensembles: 2–4 LoRA heads or SWA-style checkpoints; we average probabilities. We report wall-clock and accuracy deltas vs dropout.
- Safety: No parameter updates; `torch.no_grad()` and eval-mode for norms. We ensure independence across MC samples by using Philox RNG with per-sample counter offsets (no reseeding per pass).

### Agentic integration
We detect “decision tokens” via either tokenizer-level markers (e.g., function call JSON fields) or a small classifier over hidden states. We gate more aggressively (lower MI_thresh) on these positions and allow a short MI window across t..t+w to stabilize decisions (default w=2).

### Pseudocode (per-step, batched MC with top-M, tail, adaptive-K)
```python
@torch.no_grad()
def ugd_step(model, x, y_prefix, cache, cfg):
    # Base pass (deterministic)
    model.set_stochastic(False)  # dropout off / ensemble=None
    z_base, cache = model.forward_step(x, y_prefix, cache)
    p_base = softmax(z_base.float())
    m, Hb = top2_margin(p_base), entropy(p_base)

    if m >= cfg.m_thresh and Hb <= cfg.H_thresh:
        return argmax(p_base), cache

    # Active set S and tail mass
    topv, topi = torch.topk(p_base, cfg.top_M)
    S_mask = torch.zeros_like(p_base, dtype=torch.bool)
    S_mask.scatter_(0, topi, True)
    tau = 1.0 - topv.sum()

    # Adaptive-K loop with CI-based stopping
    K = cfg.K_min
    MI_est, CI = None, None
    while True:
        model.set_stochastic(True)
        # forward_step_batched replicates inputs/cache across K; memory-aware internally
        z_mc, _ = model.forward_step_batched(x, y_prefix, cache, batch_size=K)
        p_mc = softmax(z_mc.float())  # [K, V]

        # Restrict to S and tail bucket
        pS = renorm(p_mc[:, S_mask])          # [K, M]
        p_tail = tau.expand(K, 1)             # [K, 1], shared tail mass
        p_aug = torch.cat([pS, p_tail], dim=1)  # [K, M+1]

        p_bar = p_aug.mean(dim=0)             # [M+1]
        H_bar = entropy(p_bar)
        H_i = entropy(p_aug)                  # [K]
        MI_samples = H_bar - H_i              # per-pass plug-in
        MI_est = MI_samples.mean()
        CI = bootstrap_CI(MI_samples, conf=0.95)  # fast bootstrap

        # Stopping rules
        if (CI[0] > cfg.MI_thresh) or (CI[1] < cfg.MI_thresh) or (K >= cfg.K_max):
            break
        if (abs(MI_est - cfg.MI_thresh) >= cfg.eps) and ((CI[1]-CI[0])/2 <= cfg.eps/2):
            break
        K = min(K * 2, cfg.K_max)

    # Decision
    if MI_est <= cfg.MI_thresh:
        return argmax(p_base), cache

    # Adaptive sampling from p_bar over S∪{tail}
    MI_max = math.log(cfg.top_M + 1.0)
    MI_norm = max(0.0, min(1.0, float(MI_est / MI_max)))
    p_top = clip(cfg.p0 + cfg.alpha_p * MI_norm, cfg.p_min, cfg.p_max)
    T = clip(cfg.T0 + cfg.alpha_T * MI_norm, cfg.T_min, cfg.T_max)

    p_final = nucleus(p_bar, p=p_top)
    idx_aug = sample(p_final, T=T)  # index in [0..M] (tail at M)
    if idx_aug == cfg.top_M:
        # Tail selected: resample from base distribution over V \ S
        p_tail_full = renorm(p_base[~S_mask])
        token = index_to_vocab(~S_mask, sample(p_tail_full, T=T))
    else:
        token = topi[idx_aug]
    return token, cache
```

## Evaluation

### Models
TinyLlama-1.1B, Phi-2 (2.7B), LLaMA-2-7B, Mistral-7B. For models lacking dropout, use attention-dropout injection or 2–4 micro-ensemble heads.

### Tasks and metrics
- Code: HumanEval (pass@1), MBPP (EM).
- Factual QA: TriviaQA (EM/F1), TruthfulQA (truthfulness).
- Reasoning: GSM8K (accuracy).
- Agentic/function calling: ToolBench/Gorilla-style tasks; metrics: end-to-end success, tool-call token accuracy, API selection accuracy, steps to completion; success at fixed latency budgets.
- Efficiency: wall-clock latency, tokens/sec, s (MI invocation rate), E[K], extra passes/token, GPU memory overhead.
- Uncertainty quality: token-level error detection AUROC/AUPRC (using gold next-token), ECE (20 bins), MI–error Spearman correlation; cross-domain calibration.

### Matched-latency protocol
- Target tokens/sec from each baseline is measured on a dev split (N≥200 prompts) including KV/cache and mode-switch overhead.
- UGD hyperparameters (m_thresh, H_thresh, MI_thresh, Kmin/Kmax, M) are tuned via binary search on MI_thresh to match baseline tokens/sec within ±5% on the same split.
- All metrics reported on a held-out test split. Each stochastic method uses 5 seeds; we report mean ±95% CI and paired bootstrap significance.

### Baselines
- Greedy; temperature and nucleus sampling.
- Entropy-only gating (no MC).
- Self-consistency at matched latency (N samples).
- Speculative decoding (draft-and-verify) tuned to matched latency.
- Medusa/rejection sampling when available.
- Ablations: UGD with K ∈ {1,2,4,8}; M ∈ {50,200,full}; probability vs logit averaging; fixed vs adaptive schedules; cache reuse vs suffix vs full recompute; dropout vs micro-ensembles; with/without tail bucket.

### Hypotheses (falsifiable, matched latency ±5%)
- H1 (code): +5% absolute pass@1 vs greedy and standard sampling.
- H2 (TruthfulQA): +10% truthfulness vs temperature/nucleus by avoiding unnecessary exploration.
- H3 (GSM8K): Not worse than baselines within a 2% absolute margin.
- H4 (value of MI): MI-gated > entropy-gated by +3% average accuracy at the same s.
- H5 (KV approximations): Cache reuse within 2% of full recompute accuracy with ≥50% latency savings; suffix recompute closes residual gap.
- H6 (agentic): +10% end-to-end success at matched latency via MI-gated tool decisions.

## Relation to prior work
- Epistemic uncertainty via MC-dropout and BALD (Gal & Ghahramani, 2016); uncertainty for hallucination detection/self-checking.
- Adaptive decoding/scheduling: entropy or top-p/temperature schedules; Confident Adaptive Language Modeling (CALM) for early exiting and adaptive computation; speculative decoding; self-consistency; Medusa/rejection sampling.
- Distinctiveness: UGD targets token-level epistemic uncertainty with MI under tight latency via top-M+tail MI, CI-based adaptive-K, and KV-aware MC. It avoids full alternative generations, focusing compute on locally uncertain tokens while preserving determinism elsewhere.

## Discussion and impact
UGD separates aleatoric (high-entropy, low-MI) from epistemic (high-MI) regions, allocating exploration where the model is likely wrong. In agentic pipelines, MI can trigger sampling or verification precisely at decision tokens, improving robustness without large latency penalties. UGD complements speculative decoding and can gate when to speculate or verify.

## Limitations
- Some LLMs lack inference-time dropout; injected attention dropout or micro-ensembles add overhead and may shift calibration.
- MI noise at small K; aggressive latency budgets may limit gains.
- Hyperparameter sensitivity and calibration drift; we provide matched-latency tuning but domain shifts remain.
- Partial MC with cache reuse biases MI; we bound/measure but lack a closed-form correction.
- Backend constraints: toggling stochastic state and managing RNG/caches can incur runtime overhead and memory pressure.

## Ethics and safety
Prefer MI-triggered abstention or external verification in safety-critical contexts. We will release code, configs, and scripts for reproducibility and scrutiny.

## Conclusion
UGD is a practical scheduler that uses lightweight MC estimates of epistemic uncertainty to gate per-token decoding. With top-M+tail MI, CI-based adaptive-K, and KV-cache-aware computation, UGD can improve accuracy and agentic success at matched latency and is readily validated on small open-source models. We provide a rigorous experimental plan to test gains, quantify costs, and bound approximations for real-world LLM inference.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
