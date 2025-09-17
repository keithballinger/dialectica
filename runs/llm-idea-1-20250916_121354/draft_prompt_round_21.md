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
- Overall: Strong, novel inference-time method with clear compute accounting and a plausible path to reproducible evaluation on open models. The token-level gating and peek-based stopping are interesting and likely impactful for accuracy–compute trade-offs.
- Needed clarifications (minor but important for reproducibility and validation):
  1) Dual-Path Batching: Provide a concrete, step-by-step description (and a small ASCII diagram) of how branching and micro-batching share KV prefixes, plus memory implications. Clarify the expected wall-clock savings versus unchanged FPE.
  2) Error analysis: Add quantitative diagnostics (e.g., trigger-rate vs. failure correlations, logistic regression controlling for difficulty proxies, and banlist-only controls) to establish mechanism validity and rule out artifacts.
  3) Pseudocode and controls: Ensure all variables are defined, use consistent forward API, and explicitly ban marker tokens during reflection sampling to avoid degenerate loops. Add a “banlist-only” control (no reflections) to show masking doesn’t itself alter behavior.
  4) Markers and tokenization: Detail practical marker choices for open models without adding new special tokens (e.g., safe rare multi-token markers), and include a small preflight test to verify stability and low prior probability.
  5) Context/memory: Specify branching memory costs, copy-on-write behavior, and guardrails for long contexts.

Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
Autoregressive language models typically allocate uniform computation per output token, regardless of difficulty. We propose Reflective Decoding, a training-free inference method that adaptively allocates computation by inserting hidden “reflection” tokens into the Key-Value (KV) cache when next-token uncertainty is high. These reflections update the model’s internal state to refine predictions without altering the visible output. The method comprises: (i) a sampling-consistent uncertainty gate to trigger reflections; (ii) a greedy stopping rule that “peeks” at post-reflection uncertainty to terminate reflections dynamically; and (iii) strict compute accounting via Forward-Pass Equivalents (FPE), incorporating marker and peek overhead. Reflections are masked from the final output via an emission disallow list. We preregister evaluation on GSM8K and BBH with Mistral-7B and Llama-3-8B, hypothesizing Pareto improvements in accuracy versus compute over standard decoding and compute-matched Chain-of-Thought (CoT). The approach is model-agnostic and reproducible with open models and code.

## 1. Introduction
Large Language Models (LLMs) often overspend computation on easy tokens and underspend on difficult steps. Visible CoT increases compute globally and can be verbose. We introduce Reflective Decoding, which allocates compute at the token level, guided by model uncertainty. When uncertainty is high, decoding pauses, a hidden reflection subsequence is generated (conditioned on the same context), and visible decoding resumes after uncertainty subsides. This yields targeted internal “thought” without fine-tuning or visible scratchpads.

Contributions:
- A training-free, token-level adaptive computation method via hidden KV updates and emission masking.
- A greedy dynamic stopping rule using a low-overhead “peek” to decide reflection length.
- Transparent compute accounting (FPE) including marker and peek costs.
- A preregistered, reproducible evaluation with compute-matched baselines and mechanistic diagnostics on open models.

## 2. Related Work
- Selective CoT and self-consistency: Typically example-level and visible; we operate at token-level and keep reflections hidden.
- Latent/internal thoughts (e.g., Quiet-STaR): Training-time; ours is purely inference-time.
- Uncertainty-guided inference: Prior work triggers tools/abstention; we inject internal computation to reduce uncertainty in situ.
- Inference efficiency: Orthogonal to speculative decoding, paged KV caches, and attention kernels; complementary in practice.

## 3. Method

### 3.1 Setup and Notation
We interleave visible tokens y with hidden reflection tokens r. At step t, context C_t includes both y and r. The model distribution is p(· | C_t). Users see only y; r exists in the KV cache and logs are filtered.

### 3.2 Sampling-Consistent Uncertainty Gating
Uncertainty is computed under the active sampling policy (temperature and nucleus truncation applied before measurement).
- Default: Renormalized entropy u_t = −Σ_{i∈S_t} p_t(i) log p_t(i), optionally normalized by log|S_t|.
- Alternatives: 1 − p_max, margin, top-k entropy (ablations).
- Trigger: Reflect when u_t > τ_high; hysteresis: re-arm when u_t < τ_low; optional cooldown c visible steps.

Calibration:
- Portable defaults for normalized entropy (e.g., τ_high=0.65, τ_low=0.45).
- Or calibrate on a small dev set (200–500) to target a reflection rate (e.g., 5–15% steps). We report sensitivity and cross-task stability.

### 3.3 Hidden Reflections and Emission Masking
When triggered:
1) Append start marker R_START to KV.
2) Generate hidden tokens r using a reflection sampling policy (e.g., higher T/top-p).
3) Upon stopping (3.4), append end marker R_END.

Controls:
- Emission mask: At visible decoding steps, set logits for {R_START, R_END} (and any control tokens) to −∞.
- During reflection, also ban {R_START, R_END} to prevent degenerate self-marking.
- Reflection tokens r are standard tokens (e.g., natural-language scratchpad). Markers are short and rare.

Marker selection:
- Preferred: Pre-existing sentinel tokens in the tokenizer (if available and empirically safe).
- Practical fallback for open models: Short, rare multi-token sequences meeting: (a) low prior probability under base decoding; (b) stable tokenization; (c) neutral semantics. We provide scripts to auto-search candidates and verify “unreachability” under visible policy.

### 3.4 Greedy Dynamic Stopping via Peek
After each hidden token r_i:
1) Branch KV (copy-on-write).
2) Append R_END in the branch.
3) One forward pass under the visible policy to measure peek uncertainty u_peek.
4) Stop reflecting if u_peek ≤ τ_stop, or if per-reflection/global budgets are hit.

Peek cost:
- Appending R_END of length m_e costs m_e forwards per hidden token. We account for this explicitly (3.6).

Dual-Path Batching (DPB) optimization:
- Idea: Evaluate the next hidden continuation and the peeked visible continuation in one micro-batch from a shared prefix.
- Sketch:
  - Maintain two continuations from the same prefix KV:
    - Path A: reflection policy → token r_{i+1}
    - Path B: visible policy after branch+R_END → peek logits
  - Compute both in a single engine step using prefix-duplication and copy-on-write.

ASCII schematic:
```
... prefix ── R_START ── r_1 ... r_i  (shared KV)
                 ├─ Path A: + r_{i+1} (reflect policy)
                 └─ Path B: + R_END → peek (visible policy)
```
- Expected effect: Same FPE; reduced wall-clock due to shared memory locality and scheduler amortization (empirically up to ~30–50% depending on engine). We report both latency and FPE.

Implementation notes:
- vLLM/paged KV: use block-level prefix duplication; branch creates a lightweight view; appending R_END in branch touches m_e tokens only.
- Memory: Track number of branches, live branch length, and copy-on-write bytes written per step.

### 3.5 Budgets and Guardrails
- Global hidden budget B_total (tokens).
- Per-reflection cap L_reflect.
- Cooldown c visible steps after each reflection.
- No nesting: gating disabled during reflection.
- Context safety: enforce max context usage; optionally integrate sliding-window attention for long generations.

### 3.6 Compute Accounting (FPE)
Let V = visible tokens, H = hidden tokens, N_ref = number of reflections, and m_s, m_e the token lengths of R_START/R_END. Then:
- Visible: V
- Hidden: H
- Markers (main path): N_ref (m_s + m_e)
- Peeks (branched): H m_e
Total FPE = V + H + N_ref(m_s + m_e) + H m_e = V + H(1 + m_e) + N_ref(m_s + m_e)
We also report latency (with/without DPB), tokens/s, and peak KV memory.

### 3.7 Pseudocode
```python
def reflective_decode(model, prompt, cfg):
    """
    cfg: thresholds (tau_high, tau_low, tau_stop), budgets (B_total, L_reflect),
         policies (visible_policy, reflect_policy), markers (R_START, R_END),
         emission_banlist (includes markers), cooldown, stop_condition(y, kv).
    """
    kv = KV.from_text(prompt)
    y, hidden_used, n_reflections = [], 0, 0
    cooldown_counter, gate_armed = 0, True

    while not cfg.stop_condition(y, kv):
        logits = model.forward_last(kv, cfg.visible_policy)
        logits = apply_banlist(logits, cfg.emission_banlist)
        u_vis = entropy_over_truncated(logits, cfg.visible_policy)

        can_reflect = (gate_armed and cooldown_counter == 0 and
                       hidden_used < cfg.B_total)

        if can_reflect and u_vis > cfg.tau_high:
            # Start reflection
            append_tokens(model, kv, cfg.R_START)       # FPE += m_s
            n_reflections += 1
            gate_armed = False

            for _ in range(cfg.L_reflect):
                if hidden_used >= cfg.B_total:
                    break

                # Next hidden token
                h_logits = model.forward_last(kv, cfg.reflect_policy)
                h_logits = apply_banlist(h_logits, cfg.marker_banlist)  # ban {R_START,R_END} here too
                h_tok = sample_from_logits(h_logits, cfg.reflect_policy)
                append_token(kv, h_tok)                  # FPE += 1
                hidden_used += 1

                # Peek via branched path
                kv_peek = kv.branch_copy()
                append_tokens(model, kv_peek, cfg.R_END) # FPE += m_e
                peek_logits = model.forward_last(kv_peek, cfg.visible_policy)
                peek_logits = apply_banlist(peek_logits, cfg.emission_banlist)
                u_peek = entropy_over_truncated(peek_logits, cfg.visible_policy)

                if u_peek <= cfg.tau_stop:
                    break

            append_tokens(model, kv, cfg.R_END)         # FPE += m_e
            cooldown_counter = cfg.cooldown
            continue

        # Visible token
        v_tok = sample_from_logits(logits, cfg.visible_policy)
        append_token(kv, v_tok)                          # FPE += 1
        y.append(v_tok)

        cooldown_counter = max(0, cooldown_counter - 1)
        if u_vis < cfg.tau_low:
            gate_armed = True

    return detokenize(y)
```

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- H1: At matched FPE, Reflective Decoding improves accuracy over standard decoding and visible CoT baselines.
- H2: Uncertainty-gated reflections outperform non-adaptive (periodic) triggers at the same hidden-token budget.
- H3: Greedy peek-based stopping reduces FPE for comparable accuracy versus fixed-length reflections.

### 4.2 Tasks, Models, Hardware
- Tasks: GSM8K, BBH (subset), plus a non-math benchmark for generality.
- Models: Mistral-7B-Instruct, Llama-3-8B-Instruct.
- Hardware: A100 80GB; inference stack with FlashAttention and paged KV caches.

### 4.3 Baselines (Compute-Matched; per-task average FPE)
- Standard decoding (greedy/temperature).
- Prefix CoT: fixed visible scratchpad budget matched to our FPE.
- Example-level selective CoT (triggered by initial uncertainty).
- Self-consistency: k samples (with/without CoT), k chosen to match FPE.
- ToT/ReAct-style visible reasoning with matched total FPE.
- Uniform reflections (ablation): periodic triggers with matched hidden budget.
- Fixed-length reflections (ablation): uncertainty-gated, no peek.
- Banlist-only control: emission banlist active, reflections disabled (to isolate masking effects).

### 4.4 Metrics and Analyses
Primary:
- Accuracy with 95% CIs vs compute (FPE), latency, tokens/s.

Secondary:
- Peak KV memory, branch counts/sizes, copy-on-write bytes, DPB speedup.

Mechanistic diagnostics and error analysis:
- Δ-uncertainty and Δ-NLL distributions (pre- vs post-reflection); area under “uncertainty-reduction” curve versus reflection length.
- Trigger precision/recall vs difficulty proxies (problem length, numerical breadth, operator depth).
- Correlations: per-example trigger rate vs failure (Pearson/Spearman), and logistic regression predicting success from (FPE, triggers, Δ-uncertainty), controlling for difficulty proxies.
- Banlist-only control vs baseline to confirm emission masking alone does not change accuracy or entropy.
- Ablations over uncertainty metrics and threshold calibration; sensitivity of performance to τ_high/τ_low/τ_stop and reflection policy temperature.

### 4.5 Reproducibility
- Release code, configs, seeds, environment specs, and scripts for calibration/evaluation/analysis.
- Preflight tests: marker prior probability (<1e−4), tokenization stability checks, and “unreachability” under visible policy.
- Deterministic runs via seed control; report each source of randomness.

## 5. Limitations and Risks
- Gating is imperfect; misfires waste compute. Hysteresis, cooldown, and budgets mitigate.
- Overhead: markers and peeks are non-trivial; efficiency benefits depend on short markers and DPB.
- Context length: reflections consume context; integrate with sliding-window attention for long sequences.
- Marker OOD effects: if special tokens are unused in pretraining, they may induce distribution shift; we default to rare multi-token markers validated by preflight tests.
- Safety/leakage: hidden tokens are filtered from UIs/logs; provide audit utilities and tests to verify no leakage.

## 6. Conclusion
Reflective Decoding is a training-free, model-agnostic method for token-level adaptive computation using hidden reflections triggered by uncertainty. With principled compute accounting, a peek-based stopping rule, and a preregistered open-model evaluation, we aim to deliver reproducible Pareto improvements in the accuracy–compute trade-off and a practical path to higher-quality, lower-latency inference.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
