Minor Revisions

Brief critique
- Strengths: This is a well-structured and rigorous research proposal. The core idea—token-level uncertainty gating to insert hidden reflections that manipulate the KV cache—is novel, plausible, and impactful. The method is training-free and designed for validation on accessible open-source models. The preregistered evaluation plan is comprehensive and anticipates key criticisms, particularly through its strict compute accounting (including peeks), matched-compute baselines, and leakage-prevention protocols.
- Minor gaps: The abstract and introduction could be more impactful by leading with the core mechanism more directly. Some terminology ("sampling-consistent," "answer/reflect head") is slightly opaque and could be simplified. The pseudocode is functional but dense; minor improvements to variable names and comments would enhance readability. The trade-offs in implementing the KV branching for dynamic stopping could be stated more explicitly.

Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
We introduce Reflective Decoding, a training-free inference algorithm that adaptively allocates computation by dynamically inserting hidden reasoning segments (“reflections”) into a large language model’s KV cache. Gated by token-level uncertainty, these reflections steer subsequent generation while remaining invisible to the user. This allows the model to “think” on-demand at difficult reasoning steps without altering the final output’s structure. The method includes (i) an uncertainty-gating mechanism consistent with the sampling policy, (ii) a myopic dynamic stopping criterion that terminates reflections once uncertainty is resolved, and (iii) a strict compute accounting framework that includes all forward passes. We preregister hypotheses and a validation plan on GSM8K and BBH using Mistral-7B and Llama-3-8B. The goal is to demonstrate Pareto improvements in the accuracy-compute trade-off. Our implementation will be open-sourced.

## 1. Introduction
Standard autoregressive decoding allocates a fixed amount of computation—one forward pass—to generate each token. This is inefficient: it is excessive for simple tokens but insufficient for complex reasoning steps. While methods like Chain-of-Thought (CoT) provide more compute, they do so uniformly, wasting resources on easy problems and altering the output format.

We propose Reflective Decoding, an inference-time algorithm that allocates compute at the token level. When the model exhibits high uncertainty about the next token, it pauses visible output generation. Instead, it generates a short, hidden reflection that modifies its internal state (the KV cache). Once the reflection is complete, the model resumes generating the visible output, now conditioned on the updated state. This process is analogous to a human pausing to think when confused.

Our contributions are:
1.  **Reflective Decoding:** A training-free, token-level algorithm for adaptive inference using hidden KV cache manipulation.
2.  **Myopic Dynamic Stopping:** A mechanism to terminate reflections efficiently by "peeking" at the next visible token's predicted uncertainty.
3.  **Rigorous Accounting:** A precise compute metric, Forward-Pass Equivalents (FPE), that counts all model calls, ensuring fair baseline comparisons.
4.  **Preregistered Evaluation:** A complete, preregistered experimental design with open-source models and code for full reproducibility.

## 2. Related Work
- **Adaptive Computation:** Methods like ACT adjust computation within a single forward pass. We adapt computation *across* tokens during autoregressive decoding.
- **Dynamic CoT & Self-Correction:** Approaches like Dynamic CoT or Self-Refine trigger reasoning or correction at the example or turn level. Reflective Decoding operates at a finer, token-level granularity and keeps the reasoning process hidden from the output.
- **Confidence-Guided Inference:** Prior work uses uncertainty scores to trigger abstention or request external tools. We use these signals to trigger internal, self-contained computation.
- **Latent Reasoning:** Methods like Quiet-STaR learn to produce internal monologues during training. Our approach induces similar behavior at inference time only, requiring no model modification or training.
- **Inference Efficiency:** Speculative decoding and other techniques accelerate generation. Reflective Decoding is orthogonal and can be composed with these methods to improve the quality of the generated tokens, not just their speed.

## 3. Method

### 3.1 Problem Setup
Let `y` be the sequence of visible output tokens and `r` be hidden reflection tokens. At any step `t`, the context `C_t` is an interleaved sequence of `y` and `r`. The model produces a next-token distribution `p(· | C_t)`. The user only ever sees `y`.

### 3.2 Uncertainty Gating
The decision to reflect is based on the uncertainty of the next visible token.
- At step `t`, we compute the next-token probability distribution `p_t` under the primary sampling configuration (e.g., temperature, top-p).
- Uncertainty `u_t` is calculated over the token probabilities within the sampled support `S_t`, making the signal consistent with the sampling decision. We primarily use entropy: `u_t = −Σ_{i∈S_t} p_t(i) log p_t(i)`.
- A reflection is triggered if `u_t > τ_high`. To prevent rapid re-triggering, a hysteresis mechanism is used: the gate is re-armed only after uncertainty drops below a separate threshold, `u_t < τ_low`.

### 3.3 Hidden Reflections
When a reflection is triggered:
1.  **Tag Insertion:** The controller deterministically appends a start tag (e.g., `[REFLECT]`) to the context. We do not rely on the model to generate structural tokens.
2.  **Reflection Generation:** The model generates a sequence of hidden tokens `r` using a separate reflection-specific sampling configuration (e.g., higher temperature to encourage exploration).
3.  **State Update:** Each generated `r_i` updates the KV cache, steering subsequent computation.
4.  **Tag Closure:** The reflection is terminated by the controller, which appends an end tag (e.g., `[END_REFLECT]`).

### 3.4 Myopic Dynamic Stopping
To avoid fixed-length, potentially wasteful reflections, we use a dynamic stopping criterion. After each reflection token `r_i` is generated, the controller "peeks" to estimate the uncertainty of the *next visible token*.
1.  Let `C_ref` be the current context ending in `r_i`.
2.  A temporary context is formed: `C_peek = C_ref + [END_REFLECT]`.
3.  A single forward pass calculates the peeked uncertainty `u_peek` over `p(· | C_peek)`.
4.  If `u_peek ≤ τ_stop` or a budget is exceeded, the reflection is terminated.
This is "myopic" as it optimizes greedily for the next visible step's uncertainty.

**Implementation:** The peek step requires a temporary KV cache branch. This is implemented by copying the current KV state and running a forward pass on the short suffix. All peek passes are counted in our compute budget.

### 3.5 Guardrails
- **Budgets:** Maximum hidden tokens per example (`B_reflect`) and per reflection (`L_reflect_max`).
- **Cooldown:** A mandatory refractory period of `c` steps after a reflection ends.
- **No Re-entrancy:** Gating is disabled during a reflection.

### 3.6 Compute Accounting
- **Forward-Pass Equivalent (FPE):** Our primary metric. One FPE is one forward pass over the sequence. Total FPE is the sum of passes for visible tokens, hidden tokens, and peek steps.
- **Baselines:** Are strictly matched by total FPE, not output token count, for fair comparison.

### 3.7 Pseudocode
```python
def reflective_decode(model, prompt, params):
    # params: tau_high, tau_low, tau_stop, L_reflect_max, B_reflect, etc.
    ctx = KVContext.from_text(prompt)
    visible_tokens = []
    hidden_tokens_used = 0
    cooldown_counter = 0
    is_armed = True  # Hysteresis state

    while not is_finished(visible_tokens):
        # 1. Get uncertainty for the next visible token
        logits = model.forward_last(ctx, config=params.visible_sampling)
        uncertainty = calculate_uncertainty(logits, config=params.visible_sampling)

        # 2. Decide whether to trigger a reflection
        can_reflect = (is_armed and cooldown_counter == 0 and
                       hidden_tokens_used < params.B_reflect)
        if can_reflect and uncertainty > params.tau_high:
            ctx.append_text("[REFLECT]")
            
            # 3. Generate the hidden reflection
            for i in range(params.L_reflect_max):
                if hidden_tokens_used >= params.B_reflect: break
                
                # Sample one hidden token and update state
                tok_h = sample(model, ctx, config=params.reflect_sampling)
                ctx.append_token(tok_h)
                hidden_tokens_used += 1

                # 4. Check dynamic stopping criterion
                if params.use_dynamic_stopping:
                    ctx_peek = ctx.branch()  # Lightweight copy of KV state
                    ctx_peek.append_text("[END_REFLECT]")
                    logits_peek = model.forward_last(ctx_peek, config=params.visible_sampling)
                    u_peek = calculate_uncertainty(logits_peek, config=params.visible_sampling)
                    if u_peek <= params.tau_stop:
                        break
            
            ctx.append_text("[END_REFLECT]")
            cooldown_counter = params.cooldown
            is_armed = False
            continue  # Re-evaluate next token from the new state

        # 5. Generate one visible token
        tok_v = sample(model, ctx, config=params.visible_sampling)
        ctx.append_token(tok_v)
        visible_tokens.append(tok_v)

        # 6. Update state
        cooldown_counter = max(0, cooldown_counter - 1)
        if uncertainty < params.tau_low:
            is_armed = True

    return detokenize(visible_tokens)
```

### 3.8 Calibration
Hyperparameters (`τ_high`, `τ_low`, `τ_stop`) are calibrated on a small, held-out set (e.g., 200-500 examples) disjoint from the test set to target a specific reflection rate (e.g., 5-15% of steps) and budget utilization. This prevents any leakage from the test set.

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- **H1 (Pareto Improvement):** At matched FPE, Reflective Decoding achieves higher accuracy than standard decoding and budgeted CoT baselines.
- **H2 (Gating Efficacy):** Uncertainty-gating outperforms non-adaptive gating (random or periodic reflections) under the same compute budget.
- **H3 (Stopping Efficacy):** Myopic dynamic stopping is more compute-efficient (lower FPE for the same accuracy) than fixed-length reflections.

### 4.2 Tasks and Models
- **Tasks:** GSM8K (math reasoning), BBH (diverse reasoning).
- **Models:** Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct.
- **Hardware:** Single A100 80GB to ensure consistent latency measurement.

### 4.3 Baselines
All baselines are matched by total FPE.
- **Standard Decoding:** Greedy and nucleus sampling.
- **Budgeted CoT:** Always-on CoT, truncated to match FPE.
- **Example-Level Gating:** CoT triggered based on prompt-level uncertainty.
- **Non-Adaptive Gating:** Reflections triggered randomly or periodically.
- **Self-Consistency:** `k=5` as a high-compute quality ceiling.

### 4.4 Metrics
- **Accuracy:** EM for GSM8K, task-specific for BBH. Reported with 95% bootstrap CIs. Paired significance tests (e.g., McNemar's) will be used.
- **Compute:** FPE, token counts (visible, hidden, peek), and wall-clock latency.
- **Analysis:** Trigger rates, entropy reduction pre/post reflection, failure modes.

### 4.5 Reproducibility
We will release all code, configuration files, seeds, and execution logs to enable full audit and replication of our findings.

## 5. Results
[To be populated with empirical data from the preregistered evaluation.]
- Pareto curves (Accuracy vs. FPE) for all methods.
- Statistical outcomes for hypotheses H1-H3.
- Analysis of reflection triggers and their effect on uncertainty.
- Latency overhead measurements for KV branching and peeking.

## 6. Discussion
Reflective Decoding offers a principled way to allocate inference compute where it is most needed. By manipulating the KV cache directly, it induces a form of inference-time latent reasoning without requiring specialized training. The performance gains are expected to be most pronounced on tasks with heterogeneous difficulty, where a few hard steps determine overall success. The framework is general and complementary to other inference optimizations.

## 7. Limitations and Risks
- **Proxy Signal:** Uncertainty is a proxy for difficulty and can be miscalibrated.
- **Leakage:** Reflections could leak into the visible output if not handled carefully with system prompts and strict output filtering.
- **Overhead:** The KV branching and peek passes introduce latency overhead that must be carefully measured.
- **Generalization:** Benefits may be smaller on models already trained for latent reasoning or on tasks that are uniformly easy/hard.

## 8. Conclusion
Reflective Decoding is a simple, training-free algorithm for improving the accuracy-compute trade-off in LLM inference. By enabling the model to "think" via hidden state manipulation only when it is uncertain, it provides a more efficient and targeted approach to autoregressive reasoning. Our preregistered study on open-source models aims to validate this claim rigorously.
