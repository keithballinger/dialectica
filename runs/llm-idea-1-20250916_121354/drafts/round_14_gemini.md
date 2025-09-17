Minor Revisions

Brief critique
- Strengths: The proposed method is highly novel, training-free, and addresses a fundamental limitation of autoregressive models. The combination of sampling-consistent uncertainty gating, a greedy peek-based stopping rule, and principled compute accounting (FPE) is technically sound and well-motivated. The preregistered evaluation plan is exceptionally rigorous, featuring strong baselines, ablations, and a clear focus on reproducibility.
- Gaps to address:
  - **Baseline Clarity:** The distinction between "Budgeted CoT" and "Non-adaptive reflections" in the evaluation section (Sec 4.3) is ambiguous. Clarify if CoT is visible vs. hidden and if "non-adaptive" refers to the trigger (e.g., periodic) or length (fixed).
  - **Cost Immediacy:** The computational cost of the "peek" operation should be mentioned directly in the method description (Sec 3.4) for immediate clarity, rather than being deferred until the accounting section (Sec 3.6).
  - **Uncertainty Scale:** The definition of uncertainty (Sec 3.2) could benefit from a brief comment on whether thresholds (τ) require task-specific calibration or if the entropy measure is normalized to be more stable across contexts.
  - **Marker Token Selection:** The practical concerns section (Sec 3.7) should offer slightly more concrete guidance on selecting robust `R_START`/`R_END` markers (e.g., using statistically rare tokens or reserved tokens from the model's vocabulary).

Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
Autoregressive language models typically allocate a fixed forward pass per generated token, irrespective of difficulty. We propose Reflective Decoding, a training-free inference method that adaptively allocates computation by inserting hidden “reflection” tokens directly into the Key-Value (KV) cache when model uncertainty is high. These reflections update the model's internal state to facilitate reasoning without altering the visible output. The method is composed of: (i) an uncertainty gate aligned with the active sampling policy (temperature and nucleus truncation); (ii) a greedy dynamic stopping rule that terminates reflection by "peeking" at post-reflection uncertainty; and (iii) strict compute accounting via Forward-Pass Equivalents (FPE), reported alongside latency and memory. Reflections are masked from the final output using a logit-based disallow list. We preregister an evaluation on GSM8K and BBH with Mistral-7B and Llama-3-8B, hypothesizing Pareto improvements in accuracy versus compute over standard decoding and budgeted Chain-of-Thought. The approach is model-agnostic, training-free, and reproducible with open-source code.

## 1. Introduction
Large Language Models (LLMs) often waste computation on simple generation steps while under-allocating it for complex reasoning. Methods like Chain-of-Thought (CoT) add computation indiscriminately and expose reasoning traces to the user. We introduce Reflective Decoding: when the model’s next-token uncertainty is high, we temporarily pause visible generation, insert hidden reflection tokens that only update the KV cache, and resume visible decoding once uncertainty subsides. This enables token-level adaptive computation without fine-tuning or a visible scratchpad.

Our contributions are:
- A training-free method for token-level adaptive reasoning via hidden KV cache updates and emission masking.
- A greedy dynamic stopping criterion using a single-step uncertainty peek after a hypothetical reflection end.
- Formal compute accounting (FPE) with explicit latency and memory analysis, including KV cache branching overhead.
- A preregistered evaluation with comprehensive ablations, reproducible on small open-source models.

## 2. Related Work
- **Selective/Dynamic CoT:** Triggers reasoning on a per-example or per-turn basis. Our method operates at token granularity and hides reflections from the output.
- **Internal/Latent Thoughts:** Methods that encourage internal reasoning traces during training. Our approach is training-free and applied purely at inference.
- **Uncertainty-Guided Decoding:** Uses confidence scores for abstention or tool use. We leverage uncertainty to trigger internal state updates for improved generation.
- **Inference Efficiency:** Complementary to techniques like speculative decoding and advanced caching, and can be combined with them.

We will provide precise citations in the full paper.

## 3. Method

### 3.1 Setup and Notation
Let visible tokens be denoted by `y` and hidden reflection tokens by `r`. At step `t`, the context `C_t` interleaves previously emitted visible tokens and inserted hidden reflections. The model defines the probability distribution `p(· | C_t)`. The end-user observes only the sequence of `y` tokens; `r` tokens update the KV cache but are never emitted as output.

### 3.2 Sampling-Consistent Uncertainty Gating
We estimate next-token uncertainty under the exact sampling policy being used for generation:
- Apply temperature `T` and nucleus (top-p) truncation to the logits, then renormalize the probabilities over the retained support set `S_t`.
- Define uncertainty as the entropy of this renormalized distribution: `u_t = −Σ_{i∈S_t} p_t(i) log p_t(i)`. Thresholds are determined via calibration on a dev set, as the scale of `u_t` can vary.
- Trigger a reflection when `u_t > τ_high`. To prevent rapid re-triggering, the gate is only re-armed once `u_t < τ_low` (hysteresis).
Alternatives (e.g., logit margin, top-k entropy) are included in our ablations.

### 3.3 Hidden Reflections and Emission Masking
Upon triggering a reflection:
- Append a start marker token sequence `R_START` to the KV cache.
- Generate hidden tokens `r` using a dedicated reflection sampling policy (e.g., higher temperature or broader top-p) to encourage exploration.
- Upon stopping (Sec. 3.4), append an end marker `R_END` to the KV cache.

**Emission Control:** To prevent markers or reflection content from appearing in the output, we maintain a disallow list for visible decoding that masks `R_START`, `R_END`, and any other reflection-only control tokens by setting their logits to `-∞`.

### 3.4 Greedy Dynamic Stopping via Peek
After generating each hidden token `r_i`, we decide whether to continue the reflection:
- Create a branched "peek" KV state by efficiently copying the current KV cache (e.g., copy-on-write) and appending the `R_END` marker.
- Compute the next-token uncertainty `u_peek` on this peek state using the visible sampling policy. This "peek" costs one additional forward pass.
- Stop the reflection if `u_peek ≤ τ_stop`, a per-reflection length cap is reached, or the global hidden-token budget is exhausted.
This greedy rule uses a single-step peek per hidden token; multi-step peeks are evaluated as an ablation.

### 3.5 Budgets and Guardrails
- **Global hidden budget `B_total`**: Maximum number of hidden tokens per sequence.
- **Per-reflection cap `L_reflect`**: Maximum length of any single reflection.
- **Cooldown `c`**: A minimum number of visible steps after a reflection ends before another can begin.
- **No Nesting**: The uncertainty gate is disabled during a reflection.

### 3.6 Compute Accounting and Complexity
We report performance against three metrics:
- **FPE (Forward-Pass Equivalents):** The total number of model forward passes. For a sequence with `V` visible tokens and `R` hidden tokens, with one peek per hidden token, `FPE = V + R (generation) + R (peeking) = V + 2R`.
- **Latency:** Wall-clock time per sequence and tokens/second.
- **Memory:** Peak KV cache memory, accounting for branching.

**Implementation:** Naive KV copying for each peek is prohibitively expensive. We implement copy-on-write branching over KV blocks, such that peeks only allocate new memory for the appended `R_END` tokens while referencing the shared prefix. Microbenchmarks will quantify this overhead.

### 3.7 Practical Concerns
- **Vocabulary and Tags:** We select multi-token markers (`R_START`/`R_END`, e.g., `<reflect>`, `</reflect>`) by identifying token sequences that are statistically rare in natural text, or by using reserved special tokens if available. These markers are always masked from visible decoding (Sec. 3.3).
- **Context Limits:** Reflections consume KV memory and context length. We enforce strict budgets and can optionally integrate with context management techniques like sliding windows.
- **Safety:** Hidden reflection tokens are scanned by a lightweight content filter. We provide hooks to log and audit all hidden content during evaluation, with an option to disable reflections in sensitive applications.

### 3.8 Pseudocode
```python
def reflective_decode(model, prompt, cfg):
    # cfg: thresholds (tau_high, tau_low, tau_stop),
    #      budgets (B_total, L_reflect, cooldown),
    #      sampling (visible, reflect),
    #      emission_banlist (containing R_START, R_END tags)
    kv = KV.from_text(prompt)
    y = []
    hidden_used = 0
    cooldown_counter = 0
    gate_armed = True

    while not stop_condition(y, kv):
        # Calculate uncertainty for the next visible token
        logits = model.forward_last(kv, cfg.visible)
        logits = apply_banlist(logits, cfg.emission_banlist)
        u_vis = entropy_over_truncated(logits, cfg.visible)

        can_reflect = (gate_armed and cooldown_counter == 0 and
                       hidden_used < cfg.B_total)

        if can_reflect and u_vis > cfg.tau_high:
            append_tokens(kv, cfg.tags.R_START)
            gate_armed = False  # Disarm gate during reflection

            for r_len in range(cfg.L_reflect):
                if hidden_used >= cfg.B_total: break

                # Generate one hidden token
                h_tok, kv = sample_and_append(model, kv, cfg.reflect)
                hidden_used += 1

                # Peek ahead by appending R_END to a branched KV cache
                kv_peek = kv.branch_copy()  # Uses copy-on-write
                append_tokens(kv_peek, cfg.tags.R_END)
                peek_logits = model.forward_last(kv_peek, cfg.visible)
                peek_logits = apply_banlist(peek_logits, cfg.emission_banlist)
                u_peek = entropy_over_truncated(peek_logits, cfg.visible)

                if u_peek <= cfg.tau_stop:
                    break

            append_tokens(kv, cfg.tags.R_END)
            cooldown_counter = cfg.cooldown
            continue

        # Normal visible generation step
        # Re-use logits from uncertainty calculation
        v_tok, kv = sample_and_append_from_logits(model, kv, logits, cfg.visible)
        y.append(v_tok)

        # Update gating state
        cooldown_counter = max(0, cooldown_counter - 1)
        if u_vis < cfg.tau_low:
            gate_armed = True

    return detokenize(y)
```

### 3.9 Calibration
Hyperparameters (thresholds, budgets) are tuned on a held-out development split (200–500 examples) to target a desired reflection rate (e.g., 5–15% of steps). All hyperparameters are frozen before evaluation on the test set. We will report sensitivity curves over `τ_high` and `B_total`.

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- **H1:** At matched FPE, Reflective Decoding improves accuracy over standard decoding and baselines with visible CoT.
- **H2:** Uncertainty-gated reflections outperform non-adaptive reflection strategies (e.g., periodic or random triggers) at an equivalent hidden-token budget.
- **H3:** Greedy dynamic stopping reduces FPE for comparable accuracy versus fixed-length reflections.

### 4.2 Tasks and Models
- **Tasks:** GSM8K (math word problems), BBH (diverse reasoning). One additional non-math benchmark (e.g., short-form QA) to test generality.
- **Models:** Mistral-7B-Instruct, Llama-3-8B-Instruct.
- **Hardware:** A100 80GB; inference implemented with FlashAttention and paged KV cache.

### 4.3 Baselines
- **Standard Decoding:** Greedy or sampling with no reflections.
- **Prefix CoT:** A fixed budget of *visible* CoT tokens generated at the beginning of each problem, matched by FPE.
- **Example-Level Selective CoT:** The entire problem receives a CoT prefix if its initial uncertainty is high.
- **Self-Consistency:** `k` standard decoding samples, with `k` chosen to match the FPE of our method.
- **Uniform Reflections (Ablation):** Same hidden-token budget as our method but triggered periodically (e.g., every N steps) instead of via uncertainty gating.
- **Fixed-Length Reflections (Ablation):** Gated by uncertainty, but reflections have a fixed length without dynamic stopping.

### 4.4 Metrics and Analysis
- **Accuracy:** Task-specific (e.g., exact match), reported with 95% CIs.
- **Compute:** FPE, wall-clock latency, and tokens/second.
- **Memory:** Peak KV cache allocation and number/size of branches.
- **Trigger Diagnostics:** Correlation of reflection triggers with problem difficulty; effect of reflections on subsequent token uncertainty.
- **Error Analysis:** Qualitative analysis of problem categories helped or hurt by reflections.

### 4.5 Reproducibility
We will release open-source code with deterministic decoding seeds, configuration files, and scripts for calibration, evaluation, and figure generation. The repository will include detailed environment specifications and model identifiers.

## 5. Results
This section will be populated after the preregistered experimental runs are complete. We expect to show:
- Accuracy–FPE Pareto curves demonstrating the superiority of Reflective Decoding over baselines.
- Ablation results supporting hypotheses H2 and H3.
- Microbenchmarks quantifying the latency and memory overhead of KV branching.

## 6. Discussion
Reflective Decoding efficiently targets computation to moments of high uncertainty, improving reasoning without exposing scratchpads. The method is orthogonal to other inference optimizations like speculative decoding and can be combined with them for further gains. Its primary benefit lies in tasks with variable difficulty, where a small number of targeted hidden steps can significantly reduce uncertainty and prevent errors.

## 7. Limitations and Risks
- **Imperfect Proxy:** Uncertainty is a useful but imperfect proxy for reasoning difficulty. Misfired reflections can waste compute. Budgets and cooldowns mitigate this risk.
- **Overhead:** KV branching increases memory usage and latency. Our copy-on-write implementation reduces but does not eliminate this overhead.
- **Context Length:** Reflections consume context window length. The method is most effective when budgets are managed to avoid premature truncation.
- **Safety:** Hidden content is not exposed to the end-user but could contain undesirable text. We include audit hooks and content filters, and recommend that developers log or disable reflections in high-stakes, unmoderated settings.

## 8. Conclusion
Reflective Decoding is a training-free, uncertainty-guided method for adaptive computation in LLMs. By manipulating the KV cache to induce hidden reasoning without visible traces, it offers a new way to balance accuracy and computational cost. Our preregistered, reproducible evaluation on open models is designed to rigorously test its potential to achieve Pareto improvements on reasoning tasks.
