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

The proposal for Reflective Decoding is novel and addresses a core inefficiency in autoregressive models. However, the draft's central claim of improved compute-accuracy trade-offs is unsubstantiated due to flawed and incomplete compute accounting. The cost of the "peek" mechanism and marker tokens is non-trivial but is either understated or omitted from the proposed FPE formula. This invalidates any comparison against baselines.

Furthermore, critical methodological details are underspecified. The paper lacks concrete guidance on marker token selection, practical implementation of the costly peek operation, and robust uncertainty calibration. The evaluation plan, while strong in its preregistration commitment, requires more precise definitions of compute-matched baselines to ensure fair comparisons. Finally, the paper needs to include diagnostic metrics (e.g., change in uncertainty post-reflection) to validate that the proposed mechanism functions as hypothesized. These revisions are necessary to establish the technical soundness and reproducibility of the work.

### Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
Autoregressive language models typically allocate uniform computation per output token, regardless of difficulty. We propose Reflective Decoding, a training-free inference method that adaptively allocates computation by inserting hidden “reflection” tokens directly into the Key-Value (KV) cache when next-token uncertainty is high. These reflections update the model's internal state to refine its predictions without altering the visible output sequence. The method comprises: (i) an uncertainty gate aligned with the active sampling policy to trigger reflections; (ii) a greedy stopping rule that efficiently "peeks" at post-reflection uncertainty to terminate reflections dynamically; and (iii) strict compute accounting via Forward-Pass Equivalents (FPE), which correctly incorporates the overhead of marker tokens and the peek mechanism. Reflections are masked from the final output via an emission disallow list. We preregister an evaluation on GSM8K and BBH with Mistral-7B and Llama-3-8B, hypothesizing Pareto improvements in accuracy versus compute over standard decoding and various compute-matched Chain-of-Thought (CoT) baselines. The approach is model-agnostic and reproducible with open-source models and code.

## 1. Introduction
Large Language Models (LLMs) often expend unnecessary computation on simple tokens while under-allocating resources for complex reasoning steps. Methods like Chain-of-Thought (CoT) increase computation globally by generating visible reasoning traces, which can be verbose and are applied indiscriminately.

We introduce Reflective Decoding, a method that allocates computation at the token level, guided by model uncertainty. When the model is uncertain, it pauses visible generation, emits a sequence of hidden reflection tokens that update its internal KV cache state, and resumes visible decoding once uncertainty subsides. This allows for targeted, internal "thought" without fine-tuning or generating an explicit, visible scratchpad.

Our contributions are:
- A training-free method for token-level adaptive computation via hidden KV cache updates and emission masking.
- A greedy dynamic stopping rule that uses a low-overhead "peek" at future uncertainty to determine reflection length.
- A correct and transparent compute accounting framework (FPE), explicitly including the costs of marker tokens and the peek operation.
- A preregistered, reproducible evaluation plan with rigorously compute-matched baselines and mechanistic diagnostics on open models.

## 2. Related Work
- **Selective CoT and Self-Consistency:** These methods trigger reasoning or generate multiple trajectories at the example level, typically with visible traces. We operate at the token level and hide our computational traces.
- **Internal/Latent Thoughts (e.g., Quiet-STaR):** Such methods use training-time objectives to encourage internal reasoning. Our approach is a training-free inference technique.
- **Uncertainty-Guided Inference:** Prior work uses uncertainty to trigger actions like tool use or abstention. We use it to inject targeted, hidden computation to reduce uncertainty in situ.
- **Inference Efficiency:** Our method is orthogonal and complementary to optimizations like speculative decoding, paged KV caches, and FlashAttention.

## 3. Method

### 3.1 Setup and Notation
The generated sequence consists of visible tokens `y` and hidden reflection tokens `r`. At any step `t`, the context `C_t` is an interleaved sequence of prior visible and hidden tokens. The model defines the probability distribution `p(· | C_t)`. The end-user only observes the sequence of visible tokens `y`; `r` tokens exist only within the KV cache.

### 3.2 Sampling-Consistent Uncertainty Gating
We estimate uncertainty under the active sampling policy to decide whether to reflect.
1.  Apply temperature `T` and nucleus/top-p sampling truncation to the logits, then renormalize the probabilities over the truncated support set `S_t`.
2.  Define uncertainty `u_t` as either:
    *   **Renormalized Entropy (Default):** `u_t = −Σ_{i∈S_t} p_t(i) log p_t(i)`. This can be optionally normalized by `log |S_t|` to scale it to `[0,1]`.
    *   **Alternatives:** `1 − p_max`, logit margin, or top-k entropy (evaluated in ablations).
3.  A reflection is triggered when `u_t > τ_high`. We use hysteresis, re-arming the trigger only when `u_t < τ_low`, to prevent rapid oscillations.

**Calibration:**
We support two modes for setting thresholds: (a) using normalized entropy with portable default thresholds (e.g., `τ_high=0.65`, `τ_low=0.45`); or (b) calibrating thresholds on a small development set (200-500 examples) to achieve a target reflection rate (e.g., reflections on 5–15% of generation steps). We will report sensitivity and cross-task stability for our chosen thresholds.

### 3.3 Hidden Reflections and Emission Masking
When a reflection is triggered:
1.  Append a start marker `R_START` to the KV cache.
2.  Generate hidden tokens `r` using a distinct "reflection" sampling policy (e.g., higher temperature/top-p) to encourage exploration.
3.  Upon stopping (Section 3.4), append an end marker `R_END` to the KV cache.

**Emission Control:** We maintain an emission disallow list containing the token IDs for `R_START`, `R_END`, and any other reflection-only control tokens. During visible decoding steps, the logits for these tokens are set to `-∞`, ensuring they are never sampled. This preserves sampling consistency over the remaining vocabulary.

**Marker Selection:** The choice of markers impacts cost and robustness.
- **Preferred:** Single-ID special tokens reserved in the model's vocabulary (e.g., unused tokens in the Llama family). This minimizes marker length and peek cost.
- **Alternative:** Short, statistically rare multi-token sequences verified to be (a) infrequent in representative text corpora, (b) stable under BPE tokenization across different contexts, and (c) not semantically misleading.
We verify that markers are unreachable during visible decoding and are filtered from all internal logs and user interfaces.

### 3.4 Greedy Dynamic Stopping via Peek
After generating each hidden reflection token `r_i`, we decide whether to continue reflecting.
1.  Create a branched KV cache state (e.g., using copy-on-write).
2.  In this branched state, append the end marker `R_END`.
3.  Perform a forward pass to get the next-token logits under the visible sampling policy and compute the "peek" uncertainty, `u_peek`.
4.  Stop reflecting if `u_peek ≤ τ_stop`, or if a per-reflection or global length budget is exceeded.

**Cost of a Peek:** This operation is not free. Appending an `R_END` marker of length `m_e` tokens requires `m_e` forward passes on the branched path. This cost is incurred for *every* hidden token generated.

**Optimization:** We propose **Dual-Path Batching** to mitigate this cost. At each reflection step, we can evaluate two paths in a single micro-batch: the next hidden token `r_{i+1}` and the first visible token after appending `R_END`. This leverages shared prefix computation in the KV cache and amortizes scheduling overhead.

### 3.5 Budgets and Guardrails
- **Global Hidden Budget (`B_total`):** The maximum total number of hidden tokens per generated sequence.
- **Per-Reflection Cap (`L_reflect`):** The maximum length of a single reflection sequence.
- **Cooldown (`c`):** The minimum number of visible steps after a reflection before the trigger is re-armed.
- **No Nesting:** The uncertainty gate is disabled during a reflection.

### 3.6 Compute Accounting
We use Forward-Pass Equivalents (FPE) to provide a transparent measure of compute. Let:
- `V`: number of visible tokens generated.
- `H`: total hidden tokens generated.
- `N_ref`: number of reflections triggered.
- `m_s`, `m_e`: token lengths of `R_START` and `R_END` markers.

The total FPE is the sum of passes for visible generation, hidden generation, appending markers, and performing peeks:
- **Visible Generation:** `V`
- **Hidden Generation:** `H`
- **Markers (Main Path):** `N_ref * (m_s + m_e)`
- **Peeks (Branched Path):** `H * m_e` (one peek per hidden token)

**Total FPE = V + H + N_ref * (m_s + m_e) + H * m_e = V + H(1 + m_e) + N_ref(m_s + m_e)**

We also report:
- **Latency:** Wall-clock time and tokens/second, with and without dual-path batching.
- **Memory:** Peak KV cache allocation, number of branches created, and copy-on-write overhead.

### 3.7 Pseudocode
```python
def reflective_decode(model, prompt, cfg):
    # cfg contains thresholds, budgets, sampling policies, and markers
    kv = KV.from_text(prompt)
    y, hidden_used, n_reflections = [], 0, 0
    cooldown_counter, gate_armed = 0, True

    while not stop_condition(y):
        logits = model.forward_last(kv, cfg.visible_policy)
        logits = apply_banlist(logits, cfg.emission_banlist)
        u_vis = entropy_over_truncated(logits, cfg.visible_policy)

        can_reflect = (gate_armed and cooldown_counter == 0 and
                       hidden_used < cfg.B_total)

        if can_reflect and u_vis > cfg.tau_high:
            # --- Start Reflection ---
            append_tokens(model, kv, cfg.R_START)  # Cost: m_s FPE
            n_reflections += 1
            gate_armed = False

            for _ in range(cfg.L_reflect):
                if hidden_used >= cfg.B_total: break

                # Generate one hidden token
                h_logits = model.forward_next(kv, cfg.reflect_policy)
                h_tok = sample_from_logits(h_logits)
                append_token(kv, h_tok)
                hidden_used += 1  # Cost: 1 FPE

                # Peek to check for stop condition
                kv_peek = kv.branch_copy()
                append_tokens(model, kv_peek, cfg.R_END)  # Cost: m_e FPE
                peek_logits = model.forward_last(kv_peek, cfg.visible_policy)
                peek_logits = apply_banlist(peek_logits, cfg.emission_banlist)
                u_peek = entropy_over_truncated(peek_logits, cfg.visible_policy)

                if u_peek <= cfg.tau_stop: break

            append_tokens(model, kv, cfg.R_END)  # Cost: m_e FPE
            cooldown_counter = cfg.cooldown
            continue

        # --- Normal Visible Generation ---
        v_tok = sample_from_logits(logits)
        append_token(kv, v_tok)
        y.append(v_tok)  # Cost: 1 FPE

        cooldown_counter = max(0, cooldown_counter - 1)
        if u_vis < cfg.tau_low:
            gate_armed = True

    return detokenize(y)
```

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- **H1:** At matched FPE, Reflective Decoding achieves higher accuracy than standard decoding and visible CoT baselines.
- **H2:** Uncertainty-gated reflections outperform non-adaptive strategies (e.g., periodic triggers) with the same hidden-token budget.
- **H3:** Greedy dynamic stopping ("peek") reduces FPE for comparable accuracy versus fixed-length reflections.

### 4.2 Tasks, Models, Hardware
- **Tasks:** GSM8K, BBH (subset), plus a non-math benchmark for generality.
- **Models:** Mistral-7B-Instruct, Llama-3-8B-Instruct.
- **Hardware:** A100 80GB GPUs, using an inference stack with FlashAttention and paged KV caches.

### 4.3 Baselines (Compute-Matched)
All baselines will be configured to match the average FPE of our method on a per-task basis.
- **Standard Decoding:** Greedy or temperature sampling with no reflections.
- **Prefix CoT:** A fixed budget of visible CoT tokens prepended to the response. The budget is set to match our FPE.
- **Example-Level Selective CoT:** A visible CoT prefix is added only if initial uncertainty is high. FPE is matched.
- **Self-Consistency:** `k` samples are generated (with or without CoT), where `k` is chosen to match FPE.
- **ToT/ReAct-style Visible Reasoning:** A fixed-size beam search or tree exploration with visible reasoning steps, where the total FPE is matched.
- **Uniform Reflections (Ablation):** Hidden reflections are triggered periodically (e.g., every N steps), with the total hidden token budget matched to our method.
- **Fixed-Length Reflections (Ablation):** Reflections are uncertainty-gated but have a fixed length, without dynamic stopping.

### 4.4 Metrics and Analyses
- **Primary:** Accuracy (task-specific, with 95% CIs) vs. Compute (FPE, latency, tokens/s).
- **Secondary:** Peak memory usage, branch counts/sizes, copy-on-write overhead.
- **Mechanistic Diagnostics:**
    - Distributions of Δ-uncertainty and Δ-NLL (pre- vs. post-reflection) to validate the mechanism.
    - Analysis of trigger precision versus problem difficulty proxies.
- **Error Analysis:** Characterization of problem types where reflections help or hurt; ablations on the choice of uncertainty metric.

### 4.5 Reproducibility
We will release all code, configuration files, random seeds, and environment specifications. Our scripts will cover calibration, evaluation, and analysis to ensure full reproducibility.

## 5. Limitations and Risks
- **Imperfect Gating:** Uncertainty is a proxy for difficulty; misfired triggers waste compute. Budgets and hysteresis are mitigations.
- **Overhead:** Marker and peek costs are non-trivial. The method's efficiency depends on single-ID markers and optimized implementations like dual-path batching.
- **Context Length:** Hidden tokens consume context window. The approach requires careful budget management and may need integration with techniques like sliding-window attention for very long sequences.
- **Safety and Leakage:** Hidden content is never emitted but must be strictly filtered from user-facing logs and telemetry. We will provide audit controls and verification tests to ensure no leakage.

## 6. Conclusion
Reflective Decoding offers a training-free, model-agnostic method for adaptive computation in LLMs. By manipulating the KV cache with hidden reflections triggered by uncertainty, it directs compute to where it is most needed. With a correct compute accounting framework and a rigorous, preregistered evaluation, we aim to demonstrate a reproducible path toward Pareto improvements in the accuracy-compute trade-off.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
