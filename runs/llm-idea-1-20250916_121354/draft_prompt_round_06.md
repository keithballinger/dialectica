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

Brief critique
- **Computational cost accounting**: The `peek_uncertainty` function in the pseudocode implies an extra forward pass per reflection token for dynamic stopping. This significant cost is not explicitly mentioned or accounted for in the evaluation metrics, which currently focus on "total tokens." Metrics must include total forward passes or estimated FLOPs for a fair comparison.
- **Methodological precision**: The pseudocode is strong, but several details could be tightened. The nature of the "hidden context" (`ctx_hidden`) should be explicitly defined as the full sequence available to the model (e.g., in its KV cache). The computation of entropy over a top-k/nucleus set should be noted as an approximation of true entropy. The mechanism for "position buckets" in threshold calibration needs brief clarification.
- **Baseline specification**: The "Budgeted CoT" baseline should specify the truncation strategy (e.g., simple token-level cutoff after the CoT prompt) to ensure reproducibility and fair comparison.
- **Practical implications**: The discussion of deployment could more explicitly state that bursty compute creates a non-uniform latency profile (i.e., high variance in inter-token latency), which is a specific challenge for real-time interactive applications beyond simply elevating p95 latency.

Revised Draft
# Entropy-Gated Reflective Decoding: Spending Test-Time Compute Only When the Model Is Uncertain

## Abstract
We propose Entropy-Gated Reflective Decoding (EGRD), a training-free inference procedure that injects short, bounded “reflection” segments only when the model’s next-token predictive uncertainty exceeds a threshold. Reflections are appended to the model’s internal context (i.e., scratchpad) but are excluded from the user-visible output. EGRD concentrates extra compute at high-uncertainty steps to improve the accuracy–compute trade-off relative to standard decoding and always-on chain-of-thought (CoT). We specify a preregistered evaluation on GSM8K and BBH with Mistral-7B and Llama-3-8B, including matched-compute baselines and ablations. Our primary falsifiable claim is that EGRD expands the accuracy–compute Pareto frontier relative to standard and fixed-budget CoT; we release code to enable replication.

## 1. Introduction
Explicit test-time reasoning via chain-of-thought (CoT) improves accuracy but increases latency and cost by generating long reasoning chains, even when the model is already confident. We study whether dynamic, token-level allocation of reasoning can yield better accuracy per unit of compute.

EGRD monitors next-token uncertainty during decoding and triggers brief reflection segments only when uncertainty is high. Reflections are hidden from the user but condition subsequent token generation. The design goal is a simple, training-free controller that (i) reduces unnecessary reasoning when the model is confident and (ii) deploys extra compute when it matters most.

Contributions:
- A training-free, token-level controller that gates short hidden reflections by uncertainty, with guardrails (global budget, cooldown, hysteresis).
- A practical implementation compatible with off-the-shelf Hugging Face models and sampling strategies.
- A preregistered evaluation plan with matched-compute baselines, latency distributions, and ablations to isolate the value of uncertainty gating.
- A falsification protocol: if no configuration improves the accuracy–compute frontier on GSM8K/BBH for Mistral-7B/Llama-3-8B, the hypothesis is rejected.

## 2. Related Work
- **Adaptive computation and halting**: ACT (Graves, 2016) and early-exit transformers adapt computation during a single forward pass. We adapt the principle to autoregressive decoding via uncertainty-triggered reflections without training.
- **Selective/dynamic CoT**: Prior work decides *when* to use CoT or how long to think based on confidence signals or verifiers. EGRD differs in its token-level gating *during* generation and its use of hidden, bounded reflections.
- **Verifier- and confidence-guided decoding**: Token-level confidence calibration, margin/entropy-based stopping, and token-critic/verifier reranking inform our choice of gating signals and baselines.
- **Internal monologue**: Quiet-STaR and related approaches encourage latent reasoning during training. EGRD achieves hidden reasoning purely at inference time.
- **Efficiency methods**: Speculative decoding accelerates sampling but preserves the output distribution; EGRD alters the computation schedule to improve reasoning quality.

(Full citations in final version.)

## 3. Method

### 3.1 Overview
At each visible decoding step *t*, we compute a scalar uncertainty *u_t* from the next-token distribution *p_t*. If *u_t* exceeds a threshold *τ* and budget constraints are met, we insert a reflection segment:
- Open a special tag `<REFLECT>` that prompts structured reasoning.
- Generate up to *L_reflect* hidden tokens, stopping on `</REFLECT>` or the length cap.
- Append a control token `<CONTINUE>` and resume normal decoding of visible text.

Reflections modify the model's internal state (i.e., the KV cache) but never appear in the final output string.

### 3.2 Gating signals
- **Entropy H_t** = −Σ_i *p_t*(i) log *p_t*(i). We use this as our primary signal, approximating it over the top-k or nucleus token set to reduce cost.
- **Logit margin M_t** = log *p_t*(y₁) − log *p_t*(y₂), where y₁, y₂ are the top-1/2 tokens; lower margins indicate uncertainty.
- **Token-type priors** (optional): upweight gating probability around numerals, equations, or before the EOS token.

We will report ablations across H_t and M_t and calibrate thresholds per model/task.

### 3.3 Budgets and stability
- **Global token budget B_reflect**: A maximum number of hidden tokens per example.
- **Cooldown c**: Disable triggers for the next *c* visible tokens after a reflection.
- **Hysteresis**: Arm the trigger when *u_t* > *τ_high*; allow re-trigger only after *u_t* < *τ_low* (*τ_low* < *τ_high*) to prevent oscillations.
- Disable gating inside `<REFLECT>…</REFLECT>` segments.

### 3.4 Dynamic stopping for reflections
Instead of a fixed length *L_reflect*, we can permit early stopping when uncertainty drops:
- After each reflection token, recompute uncertainty *u_t’* on a one-token lookahead context. If *u_t’* ≤ *τ_stop*, emit `</REFLECT>` and resume answer decoding.
- This approximates a myopic value-of-compute rule: continue reflecting only while expected uncertainty remains high. The cost of this lookahead (one forward pass per reflection token) must be included in total compute accounting.

### 3.5 Pseudocode
```python
def EGRD(model, prompt, params):
  # params: τ_high, τ_low, τ_stop, L_reflect_max, B_reflect, cooldown
  ctx_hidden = prompt      # Full context for the model's KV cache
  out_visible = ""         # User-visible output string
  used_reflect, cd = 0, 0
  armed = True

  while not stop_visible(out_visible):
    logits = model(ctx_hidden).last_token_logits
    probs = softmax(logits, sampling_head=params.answer_head)
    u = uncertainty(probs, params.gating_type)

    if armed and cd == 0 and used_reflect < B_reflect and u > params.τ_high:
      # Open reflection
      ctx_hidden += "<REFLECT>\nThink step-by-step. Briefly outline subproblems.\n"
      tokens_this_reflect = 0
      while tokens_this_reflect < params.L_reflect_max:
        tok = sample(model, ctx_hidden, params.reflect_head)
        if tok == "</REFLECT>": break
        ctx_hidden += tok
        tokens_this_reflect += 1
        # Optional dynamic stop if uncertainty has dropped
        if params.dynamic_stop:
          u2 = peek_uncertainty(model, ctx_hidden, params)
          if u2 <= params.τ_stop: break
      ctx_hidden += "</REFLECT>\n<CONTINUE>\n"
      used_reflect += tokens_this_reflect
      cd = params.cooldown
      armed = False
    else:
      tok = sample(probs, params.answer_head)
      ctx_hidden += tok
      out_visible += tok
      cd = max(0, cd - 1)
      if u < params.τ_low: armed = True

  return out_visible
```
**Implementation notes**:
- `sample` can use different configurations (temp, top-p) for reflection vs. answer heads.
- `peek_uncertainty` requires a forward pass; its cost must be tracked.
- Ensure tokenizer reserves special tokens; filter any accidental control token emissions from `out_visible`.

### 3.6 Calibration of thresholds
We set *τ* values on a small calibration set (≤500 examples) by targeting a specific reflection trigger rate *r* ∈ {5%, 10%, 20%}. We also explore quantile-based thresholds derived from uncertainty distributions over sequence position buckets (e.g., first 10%, middle 80%, last 10%) to mitigate nonstationarity.

## 4. Evaluation

### 4.1 Hypotheses and falsification
**Primary**: For at least one configuration, EGRD achieves strictly higher accuracy at equal or lower total compute than both Standard and Budgeted CoT baselines (Pareto expansion). Falsification: If no configuration expands the frontier, we reject the hypothesis.
**Secondary**: Uncertainty gating outperforms random or periodic reflections at a matched reflection budget.

### 4.2 Datasets and models
- **Datasets**: GSM8K (exact match), BBH (subset of reasoning tasks; task-specific accuracy).
- **Models**: Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct (via Hugging Face).

### 4.3 Baselines
- **Standard decoding**: Greedy or low-temperature sampling, no CoT.
- **Fixed CoT**: "Let’s think step by step" prompt, no gating.
- **Budgeted CoT**: Fixed CoT with reasoning truncated at the token-level to a budget matched to EGRD’s total hidden tokens.
- **Self-consistency (k=5)**: Reference upper-bound on accuracy vs compute.
- **Example-level gated CoT**: A simplified baseline that triggers CoT once per example if initial uncertainty is high, to isolate the value of token-level gating.
- **Random reflections**: Same budget as EGRD but triggered randomly, to isolate the value of the gating signal.

### 4.4 Metrics and reporting
- **Primary**: Accuracy, total forward passes (as a proxy for FLOPs), and wall-clock latency (median and p95) with KV-cache enabled.
- **Secondary**: Total tokens generated (visible + hidden), number and length of reflections, trigger rate.
- **Analysis**: Bootstrap 95% CIs for accuracy; McNemar's test vs. strongest baseline under matched compute.
- **Plots**: Accuracy vs. Forward Passes; Accuracy vs. Latency.

### 4.5 Ablations
- **Gating**: Entropy vs. margin; hysteresis on/off; cooldown duration.
- **Reflection**: Fixed vs. dynamic stopping; prompt variants; reflection temperature.
- **Task-type**: Math vs. commonsense vs. symbolic reasoning.

### 4.6 Implementation and reproducibility
Our implementation is a ~300 LOC controller for Hugging Face models. We will release all code, configs, and per-example logs under an Apache-2.0 license to ensure reproducibility.

## 5. Discussion
- **Why it can help**: Localized uncertainty often precedes errors. Brief, targeted reasoning can restructure context and reduce ambiguity, improving next-token fidelity without the cost of full CoT.
- **When it may not**: Poorly calibrated models may trigger reflections ineffectively. Tasks needing long-horizon derivations may not benefit from short, bounded reasoning.
- **Interactions**: EGRD is compatible with speculative decoding (the controller runs on accepted tokens) and can complement RAG by triggering retrieval when uncertainty spikes.
- **Deployment**: Hidden tokens increase API cost. The bursty nature of reflections creates a non-uniform latency profile (high variance in inter-token time), challenging for streaming UIs which may need to buffer or display typing indicators.

## 6. Limitations and risks
- Entropy and margin are imperfect uncertainty proxies; per-task calibration is required.
- Hidden reflections add to the sequence length, potentially exhausting the context window in long conversations, making the method better suited for few-shot tasks.
- Leakage of control tokens into the visible output requires careful tokenizer configuration and filtering.
- As a training-free method, gains may saturate on larger, better-calibrated models.

## 7. Conclusion
EGRD is a simple, training-free inference method that allocates test-time reasoning where uncertainty is high, aiming to improve accuracy per unit of compute. It is easy to implement on open-source models and yields clear, falsifiable predictions. We release a reference implementation and preregistered evaluation to facilitate independent validation.

## References
To be completed with specific citations (ACT; selective/dynamic CoT; verifier/Token-Critic; Quiet-STaR; self-consistency; calibration/confidence; speculative decoding).


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
