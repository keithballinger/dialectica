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

Brief critique
- Missing evidence: The draft presents no empirical results, no ablations, and no statistical analysis. Claims about Pareto improvements are unsubstantiated. Leading venues will not accept without demonstrated, reproducible gains.
- Novelty risk: Adaptive/test-time compute gating and selective Chain-of-Thought have close prior art (e.g., ACT; selective/dynamic CoT; verifier- or margin-gated reasoning; Quiet-STaR; token-level confidence calibration). The draft must delineate concrete novelty beyond “entropy-gated reflections” and “hidden scratchpads,” which have been explored.
- Algorithmic clarity: The pseudocode injects both opening and closing REFLECT tags before generation, making the stop condition ambiguous. Gating within reflection mode, cooldown/hysteresis semantics, and how entropy is computed under top-k/nucleus sampling need precise specification. The visible vs hidden context handling must be robust to avoid leakage.
- Evaluation design: Baselines and compute accounting need rigor (e.g., match total tokens including hidden tokens; report p50/p95 latency; include SC/k>1 as an efficiency-accuracy reference; add confidence/margin-gated baselines). Add falsification criteria and ablations: gating signal (entropy vs margin), dynamic vs fixed reflection length, cooldown/hysteresis, reflection prompts, and effect on different task families (math/commonsense/symbolic).
- Practicality and deployment: Discuss API costs, streaming behavior, cache effects, memory pressure from hidden tokens, and compatibility with speculative decoding or KV-cache reuse. Provide code-level hooks for HF generation.
- Related work: Needs specific citations and positioning relative to dynamic/selective CoT, verifier-guided decoding, token-level uncertainty calibration, internal monologue methods, and adaptive RAG.
- Claims scope: Avoid asserting Pareto expansion without results; present a falsifiable hypothesis and a pre-registered analysis plan.

Revised Draft
# Entropy-Gated Reflective Decoding: Spending Test-Time Compute Only When the Model Is Uncertain

## Abstract
We propose Entropy-Gated Reflective Decoding (EGRD), a training-free inference procedure that injects short, bounded “reflection” segments only when the model’s next-token predictive uncertainty exceeds a threshold. Reflections are appended to the model’s hidden context (scratchpad) but excluded from user-visible output. EGRD thus concentrates extra compute at high-uncertainty steps, aiming to improve the accuracy–compute trade-off relative to standard decoding and always-on chain-of-thought (CoT). The approach is lightweight and implementable with small open-source models. We specify a preregistered evaluation on GSM8K and BBH with Mistral-7B and Llama-3-8B, including matched-compute baselines, latency reporting, and ablations over gating signals and budgets. Our primary falsifiable claim is that EGRD expands the accuracy–compute Pareto frontier relative to standard and fixed-budget CoT; we release code to enable replication.

## 1. Introduction
Explicit test-time reasoning (e.g., CoT) improves accuracy but increases latency and cost by generating long reasoning chains even when the model is already confident. We study whether dynamic, token-level allocation of reasoning can yield better accuracy per unit compute.

EGRD monitors next-token uncertainty during decoding and triggers brief reflection segments only when uncertainty is high. Reflections are hidden from the user but condition subsequent tokens. The design goal is a simple, training-free controller that (i) reduces unnecessary reasoning when the model is confident and (ii) deploys extra compute when it matters.

Contributions:
- A training-free, token-level controller that gates short hidden reflections by uncertainty, with guardrails (global budget, cooldown, hysteresis).
- A practical implementation compatible with off-the-shelf HF models and sampling strategies.
- A preregistered evaluation plan with matched-compute baselines, latency distributions, and ablations to isolate the value of uncertainty gating versus fixed reasoning budgets.
- A falsification protocol: if no configuration improves the accuracy–compute frontier on GSM8K/BBH for Mistral-7B/Llama-3-8B, the hypothesis is rejected.

## 2. Related Work
- Adaptive computation and halting: ACT (Graves, 2016); early-exit/anytime inference in transformers. We adapt the principle to autoregressive decoding via uncertainty-triggered reflections without training.
- Selective/dynamic CoT: Prior work decides when to use CoT or how long to think based on confidence signals or verifiers, and studies when CoT helps or hurts. EGRD differs in token-level gating during generation and hidden, bounded reflections.
- Verifier- and confidence-guided decoding: Token-level confidence calibration, margin/entropy-based stopping, token-critic/verifier reranking inform our choice of gating signals and baselines.
- Internal monologue: Quiet-STaR and related approaches encourage latent reasoning during training. EGRD achieves hidden reasoning purely at inference, requiring no parameter updates.
- Efficiency methods: Speculative decoding accelerates sampling but keeps the same distribution; EGRD changes the computation schedule to improve reasoning.

(Full citations in final version.)

## 3. Method

### 3.1 Overview
At decoding step t, compute a scalar uncertainty u_t from the next-token distribution p_t (entropy or margin). If u_t exceeds threshold τ and budgets allow, insert a reflection segment:
- Open a special tag <REFLECT> that prompts structured reasoning.
- Generate up to L_reflect tokens (hidden), stopping on </REFLECT> or the length cap, optionally with higher temperature to diversify hypotheses.
- Append a control token <CONTINUE> and resume normal decoding of visible text.

Reflections never appear in the user-visible string; they modify the hidden context only.

### 3.2 Gating signals
- Entropy H_t = −Σ_i p_t(i) log p_t(i), computed over top-k or nucleus tokens to reduce cost and noise.
- Logit margin M_t = log p_t(y1) − log p_t(y2), where y1,y2 are top-1/2 tokens; lower margins indicate uncertainty.
- Token-type priors (optional): upweight gating around numerals/equations or before EOS.

We report ablations across H_t and M_t and calibrate per model/task.

### 3.3 Budgets and stability
- Global token budget B_reflect per example.
- Cooldown c: disable triggers for the next c visible tokens after a reflection.
- Hysteresis: arm trigger when u_t > τ_high; allow re-trigger only after u_t < τ_low (τ_low < τ_high) to avoid oscillations.
- Disable gating inside <REFLECT>…</REFLECT> segments.

### 3.4 Dynamic stopping for reflections
Rather than a fixed L_reflect, we permit early stopping when uncertainty drops:
- After each reflection token, recompute u_t’ on a one-token lookahead context; if u_t’ ≤ τ_stop, emit </REFLECT> and resume answer decoding.
- This approximates a myopic value-of-compute rule: continue reflecting while expected uncertainty remains high.

### 3.5 Pseudocode
```
function EGRD(model, prompt, params):
  # params: τ_high, τ_low, τ_stop, L_reflect_max, B_reflect, cooldown
  ctx_hidden = prompt
  out_visible = ""
  used_reflect = 0
  cd = 0
  armed = True

  while not stop_visible(out_visible):
    logits = model(ctx_hidden).last_token_logits
    probs = softmax(logits, sampling_head=params.answer_head)
    u = uncertainty(probs, params.gating_type)

    if armed and cd == 0 and used_reflect < B_reflect and u > params.τ_high:
      # Open reflection
      ctx_hidden += "<REFLECT>\nThink step-by-step. Briefly outline subproblems and key facts.\n"
      tokens_this_reflect = 0
      while tokens_this_reflect < params.L_reflect_max:
        tok = sample_reflection(model, ctx_hidden, params.reflect_head)
        if tok == "</REFLECT>": break
        ctx_hidden += tok
        tokens_this_reflect += 1
        # Optional dynamic stop if uncertainty has dropped
        u2 = peek_uncertainty(model, ctx_hidden, params)
        if u2 <= params.τ_stop: break
      ctx_hidden += "</REFLECT>\n<CONTINUE>\n"
      used_reflect += tokens_this_reflect
      cd = params.cooldown
      armed = False
    else:
      tok = sample_answer(probs, params.answer_head)
      ctx_hidden += tok
      out_visible += tok
      cd = max(0, cd - 1)
      if u < params.τ_low: armed = True

  return out_visible
```

Implementation notes:
- Compute uncertainty from the already-available logits at each visible token; overhead is negligible relative to generation.
- Ensure tokenizer reserves special tokens; filter any accidental REFLECT/CONTINUE emissions from the visible channel.
- Freeze temperature/top-p for answer tokens; allow distinct reflect_head settings.

### 3.6 Calibration of thresholds
We set τ values on a small calibration set (≤500 examples):
- Fixed reflection rate: choose τ_high to target a reflection trigger rate r ∈ {5%, 10%, 20%}.
- Quantile-based per-position: compute u distributions by position bucket (early/mid/late) to mitigate nonstationarity.
- We grid-search τ_stop and hysteresis gap (τ_high − τ_low) and report all settings.

## 4. Evaluation

### 4.1 Hypotheses and falsification
Primary: For at least one configuration per model/task, EGRD achieves strictly higher accuracy at equal or lower total tokens than both Standard and Budgeted CoT baselines (Pareto expansion). Falsification: If no configuration expands the frontier under matched compute and latency budgets, we reject the hypothesis.

Secondary: Entropy (or margin) gating outperforms random or periodic reflections at matched reflection budgets.

### 4.2 Datasets and models
- GSM8K (exact match).
- BBH (selected reasoning sub-tasks; task-specific accuracy).
- Optional: TruthfulQA and StrategyQA for robustness.
- Models: Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct (HF).

### 4.3 Baselines
- Standard decoding: greedy or temperature 0.2, no CoT.
- Fixed CoT: “Let’s think step by step” + final answer, no gating.
- Budgeted CoT: Fixed CoT truncated to a token budget matched to EGRD’s total hidden tokens.
- Self-consistency (k=5): reference upper-bound on accuracy vs compute.
- Dynamic/margin-gated CoT (prior-art style): trigger CoT once per example when uncertainty is high; shows value of token-level vs example-level gating.
- Random reflections: same budget as EGRD to isolate gating value.

All baselines are matched on total generated tokens (visible + hidden) and we also report wall-clock latency.

### 4.4 Metrics and reporting
- Primary: accuracy; total tokens; wall-clock latency (median and p95) with KV-cache enabled.
- Secondary: number and length of reflections; trigger rate; uncertainty trajectories pre/post reflection.
- Statistical tests: bootstrap 95% CIs; McNemar test vs. strongest baseline under matched compute.
- Plots: accuracy vs total tokens; accuracy vs latency; cumulative gain vs reflection budget.

### 4.5 Ablations
- Gating: entropy vs margin; hysteresis on/off; cooldown sweep.
- Reflection: fixed vs dynamic stopping; prompt variants; reflection temperature.
- Sampling: greedy vs low-temperature answer decoding.
- Task-type: math vs commonsense vs symbolic (subset of BBH).

### 4.6 Implementation and reproducibility
- ~300 LOC HF-based controller; supports fp16 with KV-cache.
- Seeded runs; commit reproducible configs; log per-example traces (triggers, lengths, entropies).
- Open-source under Apache-2.0.

## 5. Discussion
- Why it can help: Localized uncertainty often precedes errors; brief targeted reasoning can restructure context and reduce ambiguity, improving next-token fidelity without committing to full CoT.
- When it may not: Poor calibration → mistimed triggers; tasks needing long-horizon derivations; reflection tokens can anchor on wrong paths.
- Interactions: Compatible with speculative decoding (controller runs on accepted tokens); can complement retrieval by triggering light-weight lookups when uncertainty spikes.
- Deployment: Hidden tokens increase API cost; bursty reflections impact p95 latency; streaming UIs should buffer or display typing indicators during hidden compute.

## 6. Limitations and risks
- Entropy/margin are imperfect uncertainty proxies; per-task calibration is required.
- Hidden reflections add context length; long conversations may hit context limits.
- Potential leakage of control tokens; requires tokenizer hygiene and filtering.
- We do not alter model parameters; gains may saturate on stronger models with better calibration.

## 7. Conclusion
EGRD is a simple, training-free method that allocates test-time reasoning where uncertainty is high, aiming to improve accuracy per unit compute. It is easy to implement on small open-source models and yields clear, falsifiable predictions. We release a reference implementation and preregistered evaluation to facilitate independent validation.

## Acknowledgments
To be added.

## References
To be completed with specific citations (ACT; selective/dynamic CoT; verifier/Token-Critic; Quiet-STaR; self-consistency; calibration/confidence; speculative decoding).


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
