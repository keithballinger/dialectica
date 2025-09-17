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
- The proposed method is highly novel, focusing on LLM inference with token-level uncertainty gating and hidden reflections, potentially impactful for improving accuracy-compute trade-offs in agentic workflows. It aligns with validation via code and small OSS models (e.g., Mistral-7B, Llama-3-8B). The preregistered evaluation plan is rigorous and reproducible, enhancing scientific value. However, the draft is incomplete for publication: the Results section is a placeholder, with no empirical data, logs, or validation of hypotheses. Leading journals (e.g., NeurIPS, ICLR) require demonstrated evidence of impact through experiments; without this, claims remain speculative. Minor issues include inconsistent pseudocode (e.g., sampling and stopping logic could be clarified) and abstract phrasing that overemphasizes future work over current contributions.

Revised Draft
# Entropy-Gated Reflective Decoding: Spending Test-Time Compute Only When the Model Is Uncertain

## Abstract
We introduce Entropy-Gated Reflective Decoding (EGRD), a training-free inference method that dynamically inserts short, hidden "reflection" segments during LLM decoding only when next-token uncertainty exceeds a calibrated threshold. These reflections update the model's internal KV cache to refine reasoning but are excluded from user-visible output, optimizing the accuracy-compute trade-off compared to standard decoding or always-on chain-of-thought (CoT). EGRD incorporates precise compute accounting, guardrails (e.g., budgets, cooldowns), and dynamic stopping via KV forking. We detail a preregistered evaluation on GSM8K and BBH using small open-source models (Mistral-7B, Llama-3-8B), with planned release of code and logs for reproducibility.

## 1. Introduction
Chain-of-thought (CoT) prompting enhances LLM reasoning but incurs high latency and cost by generating lengthy chains indiscriminately, even when the model is confident. EGRD addresses this by adaptively allocating test-time compute at the token level, triggering hidden reflections only during uncertainty spikes to improve efficiency without sacrificing quality.

The method is model-agnostic, requiring no training, and uses entropy or logit margins as gating signals. Key features include hysteresis for stability, bounded reflection lengths, and myopic dynamic stopping to minimize unnecessary compute.

Contributions:
- A novel token-level gating mechanism for hidden reflections, with guardrails to ensure controlled compute usage.
- Detailed algorithm with integrated compute metrics, compatible with standard LLM frameworks like Hugging Face.
- Preregistered hypotheses and evaluation protocol, including matched-compute baselines and ablations.
- Open-source implementation for validation on accessible models.

## 2. Related Work
- Adaptive computation: Works like ACT (Graves, 2016) and early-exit Transformers adapt intra-pass compute; EGRD extends this to autoregressive decoding without retraining.
- Dynamic CoT: Approaches such as Dynamic Chain-of-Thought (Diao et al., 2023) or Self-Refine (Madaan et al., 2023) adjust reasoning at example or fixed-budget levels; EGRD enables finer-grained, token-level control with hidden outputs.
- Confidence-guided methods: Token-level entropy/margin thresholds (Kadavath et al., 2022; Ren et al., 2023) and verifiers (Dhuliawala et al., 2023) inspire our gating; we apply them for on-the-fly reflection triggering.
- Latent reasoning: Quiet-STaR (Zelikman et al., 2024) trains internal monologues; EGRD induces them inference-only.
- Efficiency techniques: Speculative decoding (Chen et al., 2023) speeds up sampling; EGRD complements it by reallocating compute for quality gains.

## 3. Method

### 3.1 Overview
During decoding, at each step t, EGRD computes uncertainty u_t from the next-token distribution. If u_t > threshold and constraints allow, it inserts a hidden reflection: append an opening tag (e.g., "<REFLECT>"), generate up to L_reflect tokens with a reasoning prompt, close with "</REFLECT>", and resume via "<CONTINUE>". Hidden tokens update KV cache but are filtered from output.

### 3.2 Gating Signals
- Entropy: H_t = −Σ p_t(i) log p_t(i) over nucleus/top-k for consistency with sampling.
- Logit margin: M_t = log p_t(top1) - log p_t(top2).
- Optional priors: Reweight u_t near task-specific tokens (e.g., numerals), ablated for impact.

Thresholds are calibrated on ≤500 examples to target trigger rates of 5-20%.

### 3.3 Guardrails
- Per-example budget B_reflect for hidden tokens.
- Cooldown c after reflections to prevent chaining.
- Hysteresis: Trigger at τ_high, re-arm at τ_low < τ_high.
- No gating inside reflections.

### 3.4 Dynamic Stopping
To halt reflections efficiently:
- After each reflection token, fork KV cache (efficient reference, no full copy).
- Append closing tags to fork and compute peeked uncertainty u_vis for next visible token.
- Stop if u_vis ≤ τ_stop; otherwise continue up to L_reflect or B_reflect.
- Forked passes count toward total compute.

This approximates value-of-information: reflect only if it reduces imminent uncertainty.

### 3.5 Compute Accounting
Metrics include:
- Forward-pass equivalents (FPE): All Transformer calls with KV cache.
- Token counts: Visible, hidden, total.
- Latency: Median/p95 on fixed setup, including prompts.

Baselines use identical accounting.

### 3.6 Pseudocode
```python
def egrd_decode(model, prompt, params):
    # params: τ_high, τ_low, τ_stop, L_reflect_max, B_reflect, cooldown, gating_type, answer_head, reflect_head, dynamic_stop
    ctx = prompt  # KV-backed context
    out = ""      # Visible output
    used_reflect, cd, armed = 0, 0, True

    while not stop_visible(out):
        logits = model.forward_last(ctx)
        probs = softmax(logits, params.answer_head)  # Apply temp/top-p
        u = uncertainty(probs, params.gating_type)

        if armed and cd == 0 and used_reflect < B_reflect and u > params.τ_high:
            ctx += "<REFLECT>\nThink step-by-step. Be concise.\n"
            tok_count = 0
            while tok_count < params.L_reflect_max and used_reflect < B_reflect:
                tok_r = sample_next(model, ctx, params.reflect_head)
                ctx += tok_r
                tok_count += 1
                used_reflect += 1

                if params.dynamic_stop:
                    ctx_peek = ctx + "</REFLECT>\n<CONTINUE>\n"
                    logits_peek = model.forward_last(ctx_peek)  # Forked pass, counts as FPE
                    probs_peek = softmax(logits_peek, params.answer_head)
                    u_vis = uncertainty(probs_peek, params.gating_type)
                    if u_vis <= params.τ_stop or tok_r in ["</REFLECT>"]:
                        break

            ctx += "</REFLECT>\n<CONTINUE>\n"
            cd = params.cooldown
            armed = False
        else:
            tok_v = sample_next(model, ctx, params.answer_head)
            ctx += tok_v
            out += tok_v
            cd = max(0, cd - 1)
            if u < params.τ_low:
                armed = True

    return out
```
Notes: Tags are compact strings; filter from out; entropy uses sampling-consistent subset.

### 3.7 Calibration
Sweep thresholds on calibration split to hit target rates; optimize τ_stop for efficiency vs. accuracy.

## 4. Evaluation

### 4.1 Hypotheses (Preregistered)
- Primary: EGRD Pareto-dominates standard and budgeted CoT in accuracy vs. FPE.
- Secondary: Uncertainty gating > random/periodic under matched budgets.
- Tertiary: Dynamic stopping > fixed-length at equal accuracy.

### 4.2 Setup
Datasets: GSM8K, BBH (arithmetic/symbolic/commonsense subsets). Models: Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct (bf16, single GPU).

### 4.3 Baselines
- Standard decoding.
- Fixed/Budgeted CoT (hidden to match formatting).
- Example-level gated CoT.
- Random/periodic reflections.
- Self-consistency (k=5) as upper bound.

### 4.4 Metrics
Accuracy with CIs/tests; FPE/tokens/latency; reflection stats; Pareto plots.

### 4.5 Environment
A100 GPU, PyTorch/Transformers/FlashAttention; full specs and seeds for repro.

### 4.6 Ablations
Gating variants, reflection params, task types.

### 4.7 Reproducibility
Release code (~300 LOC), configs, logs.

## 5. Results
[Populate with experimental data, including tables/plots of accuracy vs. compute, hypothesis tests, and ablations. Ensure all claims are evidence-based from released logs.]

## 6. Discussion
EGRD targets uncertainty hotspots to boost fidelity efficiently. Limitations: Calibration overhead; potential for tag leakage or bias amplification. Composable with accelerators; suitable for streamed inference with buffering.

## 7. Limitations and Risks
Imperfect proxies; context pressure; model-dependent gains; requires filtering to prevent leaks.

## 8. Conclusion
EGRD advances adaptive LLM inference. With validated results, it demonstrates potential to expand efficiency frontiers.

## References
[As in original, with any additions for completeness.]


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
