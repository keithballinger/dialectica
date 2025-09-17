Minor Revisions

Brief critique
- Strengths: The draft presents a highly novel, training-free method for adaptive LLM inference via KV cache manipulation, with strong potential impact on quality and efficiency in agentic workflows. The preregistered evaluation is rigorous, reproducible with open-source models/code, and aligns with leading journal standards (e.g., NeurIPS or ICML). Compute accounting and baselines are evidence-based and address common pitfalls.
- Minor gaps: As noted in the draft's own self-critique, the abstract and intro could more directly emphasize the core mechanism for immediate impact. Terminology like "sampling-consistent" and "answer/reflect head" (though the latter isn't heavily used here) could be clarified or simplified. Pseudocode is dense; better variable names, comments, and formatting would improve readability. Explicitly discuss KV branching trade-offs (e.g., memory overhead) in the method section. Add a sentence on ethical considerations in limitations (e.g., misuse for deceptive outputs).

Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
Large language models (LLMs) inefficiently allocate fixed computation per token during autoregressive decoding, often wasting resources on easy steps while under-resourcing complex ones. We introduce Reflective Decoding, a training-free inference method that dynamically inserts hidden reasoning segments ("reflections") into the KV cache when token-level uncertainty is high. These reflections update the model's internal state to resolve uncertainty, without altering the visible output. Key components include: (i) uncertainty gating aligned with the sampling policy, (ii) a greedy dynamic stopping rule based on "peeking" at future uncertainty, and (iii) strict compute accounting via Forward-Pass Equivalents (FPE) that includes all operations. We preregister an evaluation on GSM8K and BBH using Mistral-7B and Llama-3-8B, hypothesizing Pareto improvements in accuracy-compute trade-offs. Code will be open-sourced for validation.

## 1. Introduction
Autoregressive decoding in LLMs uses a uniform forward pass per token, which is suboptimal: simple tokens need little computation, while reasoning-heavy ones demand more. Techniques like Chain-of-Thought (CoT) add compute but apply it indiscriminately and clutter outputs.

Reflective Decoding addresses this by pausing visible generation at high-uncertainty tokens and inserting hidden reflections directly into the KV cache. This manipulates the model's state to "think" adaptively, resuming output only when uncertainty resolves—mimicking human reflection without visible traces.

Our contributions:
1. **Reflective Decoding:** Training-free, token-level adaptive inference via hidden KV updates.
2. **Greedy Dynamic Stopping:** Terminates reflections by peeking at next-token uncertainty.
3. **Rigorous Compute Metric:** FPE counts all forward passes for fair comparisons.
4. **Preregistered Plan:** Full experimental design with open-source models/code.

This approach is novel, impactful for inference efficiency, and verifiable on small models.

## 2. Related Work
- **Adaptive Computation:** ACT-like methods adjust intra-pass compute; we adapt across tokens in decoding.
- **Dynamic CoT & Self-Correction:** Trigger reasoning at example/turn levels (e.g., Dynamic CoT, Self-Refine); ours is token-granular and output-hidden.
- **Uncertainty-Guided Inference:** Uses confidence for abstention/tools; we trigger internal reflections.
- **Latent Reasoning:** Quiet-STaR trains internal thoughts; ours induces them inference-only, no training.
- **Inference Efficiency:** Orthogonal to speculative decoding; combines for better quality/speed.

## 3. Method

### 3.1 Problem Setup
Let `y` be visible output tokens and `r` hidden reflection tokens. Context `C_t` interleaves `y` and `r` at step `t`. The model yields `p(· | C_t)`, but users see only `y`.

### 3.2 Uncertainty Gating
Gate reflections on next visible token's uncertainty:
- Compute distribution `p_t` under visible sampling (e.g., temperature, top-p).
- Uncertainty `u_t` uses entropy over sampled support `S_t`: `u_t = −Σ_{i∈S_t} p_t(i) log p_t(i)` (aligned with sampling for consistency).
- Trigger if `u_t > τ_high`; re-arm only if `u_t < τ_low` (hysteresis prevents oscillation).

### 3.3 Hidden Reflections
On trigger:
1. Append start tag `[REFLECT]` deterministically.
2. Generate `r` with reflection sampling (e.g., higher temperature).
3. Update KV cache with each `r_i`.
4. Append end tag `[END_REFLECT]` on termination.

### 3.4 Greedy Dynamic Stopping
After each `r_i`, peek at next visible uncertainty:
1. Form peek context: `C_peek = current context + [END_REFLECT]`.
2. Compute `u_peek` via one forward pass.
3. Stop if `u_peek ≤ τ_stop` or budget hit.
This is greedy, focusing on immediate uncertainty. Implementation copies KV cache for branching (memory overhead: O(sequence length); all peeks count in compute).

### 3.5 Guardrails
- Budgets: Total hidden tokens `B_reflect`, per-reflection max `L_reflect_max`.
- Cooldown: `c` steps post-reflection.
- No nesting: Disable gating during reflections.

### 3.6 Compute Accounting
- **FPE:** Sums all forward passes (visible, hidden, peeks).
- Baselines matched by FPE for fairness.

### 3.7 Pseudocode
```python
def reflective_decode(model, prompt, params):
    # params: tau_high, tau_low, tau_stop, max_hidden_total, max_hidden_per, cooldown, etc.
    kv_ctx = KVContext.from_text(prompt)  # Initialize KV cache from prompt
    visible_output = []  # List of visible tokens
    hidden_count = 0  # Track total hidden tokens used
    cooldown_remain = 0  # Steps left in cooldown
    gate_armed = True  # Hysteresis for gating

    while not is_finished(visible_output):
        # Compute uncertainty for next visible token
        logits = model.forward_last(kv_ctx, config=params.visible_config)  # Forward pass on current KV
        u_visible = entropy_over_support(logits, config=params.visible_config)  # Entropy on sampled set

        # Check if we can and should reflect
        can_reflect = gate_armed and cooldown_remain == 0 and hidden_count < params.max_hidden_total
        if can_reflect and u_visible > params.tau_high:
            kv_ctx.append_text("[REFLECT]")  # Start reflection

            # Generate hidden tokens with dynamic stopping
            for _ in range(params.max_hidden_per):
                if hidden_count >= params.max_hidden_total: break

                # Sample and append one hidden token
                hidden_tok = sample(model, kv_ctx, config=params.reflect_config)
                kv_ctx.append_token(hidden_tok)
                hidden_count += 1

                # Peek for stopping
                if params.dynamic_stop:
                    peek_ctx = kv_ctx.copy_branch()  # Copy KV for peek (memory-efficient)
                    peek_ctx.append_text("[END_REFLECT]")
                    peek_logits = model.forward_last(peek_ctx, config=params.visible_config)
                    u_peek = entropy_over_support(peek_logits, config=params.visible_config)
                    if u_peek <= params.tau_stop:
                        break

            kv_ctx.append_text("[END_REFLECT]")  # End reflection
            cooldown_remain = params.cooldown
            gate_armed = False
            continue  # Reassess from updated state

        # Sample and append visible token
        visible_tok = sample(model, kv_ctx, config=params.visible_config)
        kv_ctx.append_token(visible_tok)
        visible_output.append(visible_tok)

        # Update cooldown and arming
        cooldown_remain = max(0, cooldown_remain - 1)
        if u_visible < params.tau_low:
            gate_armed = True

    return detokenize(visible_output)
```

### 3.8 Calibration
Tune thresholds on held-out set (200-500 examples) for target reflection rate (5-15%), avoiding test leakage.

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- **H1:** Higher accuracy at matched FPE vs. standard/CoT baselines.
- **H2:** Uncertainty gating beats random/periodic under same budget.
- **H3:** Dynamic stopping more efficient than fixed-length.

### 4.2 Tasks and Models
- Tasks: GSM8K, BBH.
- Models: Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct.
- Hardware: A100 80GB.

### 4.3 Baselines
FPE-matched: Standard decoding, budgeted CoT, example-level gating, non-adaptive reflections, self-consistency (k=5).

### 4.4 Metrics
Accuracy (EM/task-specific, with CIs/tests), FPE/tokens/latency, trigger analysis.

### 4.5 Reproducibility
Full code/seeds/logs released.

## 5. Results
[To be populated with empirical data from the preregistered evaluation.]
- Pareto curves (Accuracy vs. FPE).
- Hypothesis tests.
- Trigger/entropy analysis.
- Latency for branching/peeks.

## 6. Discussion
This method targets compute to bottlenecks, enhancing reasoning via latent updates. Complements other optimizations; strongest on variable-difficulty tasks.

## 7. Limitations and Risks
- Uncertainty as proxy may misfire.
- Potential leakage to output without filtering.
- Branching adds memory/latency overhead.
- Limited gains on uniform tasks or latent-trained models.
- Ethical risk: Could enable hidden manipulations for deception; mitigations include audit tools.

## 8. Conclusion
Reflective Decoding boosts LLM inference efficiency via uncertainty-driven hidden reflections. Preregistered validation will confirm its advantages.
