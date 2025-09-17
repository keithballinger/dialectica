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
- Strengths: Clear, training-free mechanism; well-motivated uncertainty gating; explicit compute accounting; preregistered evaluation on open-source models; plausible impact on agentic/interactive workflows; practical pseudocode.
- Gaps to address:
  - Novelty positioning: explicitly distinguish from internal/hidden CoT, deliberate decoding, and entropy-gated CoT to avoid overstated claims.
  - Emission control: specify a robust mechanism to prevent [REFLECT]/[END_REFLECT] tokens from leaking into visible output (e.g., a disallow list and logit masking).
  - Memory/latency analysis: quantify KV branching overhead and propose an implementation strategy (copy-on-write, block sharing) with bounds and microbenchmarks; clarify how this interacts with long-context limits.
  - Uncertainty definition: precisely define “sampling-consistent” entropy (temperature- and top-p–adjusted, renormalized) and justify over alternatives (margin, top-k entropy).
  - Compute metric: formalize FPE and report both FPE and latency; note that FPE can understate overhead from KV copies and cache management.
  - Evaluation: add ablations (no-tags, fixed-length reflections, no dynamic stopping, random gating), trigger quality diagnostics, and results on at least one non-math task to test generality; ensure strict dev/test separation for threshold tuning.
  - Safety/ethics: strengthen the discussion of hidden content risks and auditing hooks.

Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
Autoregressive LLMs allocate a fixed forward pass per token, regardless of difficulty. We propose Reflective Decoding, a training-free inference method that adaptively inserts hidden “reflection” tokens into the KV cache when uncertainty spikes, updating the internal state without changing the visible output. The method comprises: (i) uncertainty gating aligned with the active sampling policy (temperature and nucleus truncation), (ii) a greedy dynamic stopping rule that “peeks” at post-reflection uncertainty, and (iii) strict compute accounting via Forward-Pass Equivalents (FPE), reported alongside latency/memory. Reflections are masked from the visible output via an emission disallow list. We preregister a study on GSM8K and BBH with Mistral-7B and Llama-3-8B, hypothesizing Pareto improvements in accuracy vs. compute over standard decoding and budgeted CoT. The approach is model-agnostic, training-free, and reproducible with open-source code.

## 1. Introduction
LLMs often waste compute on easy steps and under-allocate it on hard ones. CoT adds computation indiscriminately and exposes traces to the user. We introduce Reflective Decoding: when the model’s next-token uncertainty is high, we temporarily pause visible generation, insert hidden reflection tokens that only update the KV cache, and resume visible decoding once uncertainty drops. This enables token-level adaptive computation without fine-tuning or visible scratchpads.

Contributions:
- Training-free token-level adaptive reasoning via hidden KV updates and emission masking.
- Greedy dynamic stopping using a single-step uncertainty peek after a hypothetical reflection end.
- Formal compute accounting (FPE) with explicit latency and memory analysis, including KV branching overhead.
- Preregistered evaluation and ablations reproducible on small open-source models.

## 2. Related Work
- Selective/dynamic CoT: triggers reasoning on some examples or turns; our method operates at token granularity and hides reflections from output.
- Internal/latent thoughts: methods that encourage internal reasoning traces during training; our approach is training-free and purely at inference.
- Uncertainty-guided decoding: uses confidence for abstention or tool use; we leverage uncertainty to trigger internal state updates.
- Inference efficiency: complementary to speculative decoding and caching; can be combined.

We will provide precise citations in the full paper.

## 3. Method

### 3.1 Setup and notation
Let visible tokens be y and hidden tokens be r. At step t, the context C_t interleaves previously emitted visible tokens and inserted hidden reflections. The model defines p(· | C_t). Users see only y; r updates the KV cache but is never emitted.

### 3.2 Sampling-consistent uncertainty gating
We estimate next-token uncertainty under the actual sampling policy:
- Apply temperature T and nucleus (top-p) truncation to logits, renormalize over the retained support S_t.
- Define u_t = −Σ_{i∈S_t} p_t(i) log p_t(i).
- Trigger a reflection when u_t > τ_high, and re-arm the gate only when u_t < τ_low (hysteresis).
Alternatives (logit margin, top-k entropy) are included in ablations.

### 3.3 Hidden reflections and emission masking
On trigger:
- Append a start marker token sequence R_START (e.g., a rare multi-token string) to the KV cache.
- Generate hidden tokens r using a reflection policy (typically higher temperature or broader top-p) to encourage exploration.
- Upon stopping (Sec. 3.4), append R_END to the KV cache.

Emission control:
- Maintain a disallow list for visible decoding that masks R_START, R_END, and any reflection-only control tokens with −∞ logits.
- Optionally add a repetition penalty or banlist on substrings indicative of reflection tags.

### 3.4 Greedy dynamic stopping via peek
After each hidden token r_i:
- Create a branched KV “peek” state by shallow-copying the KV cache and appending R_END.
- Compute u_peek under the visible sampling policy.
- Stop the reflection if u_peek ≤ τ_stop, a per-reflection length cap is reached, or the global hidden-token budget is exhausted.
This greedy rule uses a single-step peek per hidden token; multi-step peeks are an ablation.

### 3.5 Budgets and guardrails
- Global hidden budget B_total, per-reflection cap L_reflect.
- Cooldown c visible steps after a reflection ends.
- No nesting: gating disabled during a reflection.
- Optional max reflection rate per sequence.

### 3.6 Compute accounting and complexity
We report:
- FPE: the total number of model forward passes, including visible steps, hidden steps, and peeks. For a sequence with V visible tokens and R hidden tokens, with one peek per hidden token:
  FPE = V + R + R = V + 2R.
- Latency: wall-clock per sequence and tokens/s.
- Memory: peak KV memory with branching.

Implementation note: naive KV copying per peek is memory-heavy. We implement copy-on-write branching over KV blocks (layer, head, time) so that peeks only allocate new pages for appended tokens; the shared prefix is referenced. Microbenchmarks quantify the overhead.

### 3.7 Practical concerns
- Vocabulary and tags: we use multi-token markers (e.g., “<reflect> … </reflect>”) chosen to minimize collisions in normal generation; markers are masked from visible decoding (Sec. 3.3).
- Context limits: reflections lengthen sequences and consume KV memory. We enforce budgets and optionally truncate far-history caches (e.g., sliding window) when safe for the task.
- Safety: reflection tokens are scanned by a lightweight filter; hooks allow logging and auditing hidden content during evaluation.

### 3.8 Pseudocode
```python
def reflective_decode(model, prompt, cfg):
    # cfg: thresholds (tau_high, tau_low, tau_stop),
    #      budgets (B_total, L_reflect, cooldown),
    #      sampling (visible, reflect),
    #      banlist for visible emission (tags)
    kv = KV.from_text(prompt)
    y = []
    hidden_used = 0
    cooldown = 0
    gate_armed = True

    while not stop_condition(y, kv):
        # Visible uncertainty under actual sampling policy
        logits = model.forward_last(kv, cfg.visible)
        logits = apply_banlist(logits, cfg.visible.banlist)
        u_vis = entropy_over_truncated(logits, cfg.visible)

        can_reflect = (gate_armed and cooldown == 0 and
                       hidden_used < cfg.B_total)

        if can_reflect and u_vis > cfg.tau_high:
            append_tokens(kv, cfg.tags.R_START)

            per_len = 0
            while per_len < cfg.L_reflect and hidden_used < cfg.B_total:
                # Generate one hidden token
                h_tok, kv = sample_and_append(model, kv, cfg.reflect)
                hidden_used += 1
                per_len += 1

                # Peek with R_END appended on a branched KV
                kv_peek = kv.branch_copy()      # copy-on-write
                append_tokens(kv_peek, cfg.tags.R_END)
                peek_logits = model.forward_last(kv_peek, cfg.visible)
                peek_logits = apply_banlist(peek_logits, cfg.visible.banlist)
                u_peek = entropy_over_truncated(peek_logits, cfg.visible)

                if u_peek <= cfg.tau_stop:
                    break

            append_tokens(kv, cfg.tags.R_END)
            cooldown = cfg.cooldown
            gate_armed = False
            continue

        # Visible generation step
        logits = model.forward_last(kv, cfg.visible)
        logits = apply_banlist(logits, cfg.visible.banlist)
        v_tok, kv = sample_and_append_from_logits(model, kv, logits, cfg.visible)
        y.append(v_tok)

        # Update gating state
        cooldown = max(0, cooldown - 1)
        if u_vis < cfg.tau_low:
            gate_armed = True

    return detokenize(y)
```

### 3.9 Calibration
Thresholds and budgets are tuned on a held-out development split (200–500 items) to target a reflection rate of 5–15%. We freeze hyperparameters before test evaluation. We report sensitivity curves over τ_high and B_total.

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- H1: At matched FPE, Reflective Decoding improves accuracy over standard decoding and budgeted CoT.
- H2: Uncertainty gating outperforms random or periodic reflection triggers at matched hidden-token budgets.
- H3: Greedy dynamic stopping reduces FPE for comparable accuracy vs. fixed-length reflections.

### 4.2 Tasks and models
- Tasks: GSM8K (math word problems), BBH (diverse reasoning). One additional non-math benchmark (e.g., short-form QA) to test generality.
- Models: Mistral-7B-Instruct, Llama-3-8B-Instruct.
- Hardware: A100 80GB; inference with FlashAttention and paged KV.

### 4.3 Baselines
- Standard decoding (no reflections).
- Budgeted CoT: fixed number of hidden tokens per problem.
- Selective CoT at example level (uncertainty-gated).
- Self-consistency CoT (k samples) matched by FPE.
- Non-adaptive reflections: same hidden-token budget without gating or stopping.
- Ablations: no tags (no R_START/R_END), no dynamic stopping, alternative uncertainty metrics.

### 4.4 Metrics and analysis
- Accuracy (exact match or task-specific), with 95% CIs and paired tests where applicable.
- Compute: FPE, wall-clock latency, tokens/s.
- Memory: peak KV allocation, number/size of branches.
- Trigger diagnostics: reflection frequency vs. u_t, effect of reflections on subsequent uncertainty.
- Error analysis: categories helped/hurt by reflections.

### 4.5 Reproducibility
- Open-source code (deterministic decoding seeds, configs, logging of hidden/visible tokens, FPE counters).
- Scripts for calibration, evaluation, and figure generation.
- Detailed environment (CUDA, drivers) and model hashes.

## 5. Results
To be populated after preregistered runs:
- Accuracy–FPE Pareto curves with baselines.
- Reflection rate vs. accuracy trade-offs.
- Latency and memory overhead of branching (microbenchmarks).
- Ablation outcomes supporting H2–H3.

## 6. Discussion
Reflective Decoding targets compute to moments of high uncertainty, improving reasoning without exposing scratchpads. It is orthogonal to speculative decoding and caching; combining them may yield additional speedups. The approach is most beneficial on variable-difficulty instances where a small number of hidden steps reduces uncertainty.

## 7. Limitations and risks
- Uncertainty is an imperfect proxy; misfires can waste compute or derail. We mitigate with budgets and cooldowns.
- KV branching increases memory and latency; copy-on-write mitigates but does not eliminate overhead.
- Long reflections consume context length; budgets and sliding windows are necessary.
- Safety: hidden content could include undesirable text. We include audit hooks, filters, and an option to log or disable reflections in sensitive settings.

## 8. Conclusion
Reflective Decoding is a training-free, uncertainty-guided method for adaptive computation that manipulates the KV cache to induce internal reasoning without visible traces. We provide a preregistered, reproducible evaluation plan on open models and expect Pareto improvements in accuracy vs. compute on reasoning tasks.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
