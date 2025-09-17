Minor Revisions

Brief critique
- Novelty and potential impact: Strong. Training-free, token-level adaptive computation via hidden KV updates is a fresh angle on internal reasoning and could be impactful for quality/compute trade-offs.
- Technical soundness: Generally solid, but compute accounting undercounts key costs (marker tokens during both reflections and peeks). Clarifying this affects fairness versus baselines and overall conclusions.
- Clarity: Good overall; needs sharper definitions for baselines, uncertainty calibration, and marker-token design to avoid leakage and reduce overhead.
- Practicality: KV branching, batching peeks, and marker selection are implementable on open models, but the draft should explicitly state constraints and optimizations (e.g., single-ID markers, dual-path batching).
- Evaluation: Preregistered plan is strong. Add clearer compute-matched baselines (including ToT/ReAct-style visible reasoning), and report Δ-uncertainty and Δ-loss after reflections to validate mechanism.
- Risks/limitations: Expand on logging/leakage risks, effect of disallow lists on sampling consistency, and robustness across tasks/models.

Actionable changes required
- Correct compute accounting to include start/end marker tokens for both actual reflections and per-step peeks; define marker lengths and revisit FPE formulas and fairness.
- Make the cost of the peek explicit in-method (not just in accounting) and propose practical optimizations (e.g., single-token markers, dual-path batched peeks).
- Clarify baseline taxonomy and FPE matching, including ToT/ReAct and self-consistency with visible CoT.
- Provide uncertainty calibration guidance (normalized entropy vs. dev-set calibration) and defaults; discuss cross-task stability.
- Strengthen marker-token guidance: prefer single-ID reserved tokens when available; otherwise, verify rarity and BPE stability; document leak-proofing in logs/interfaces.
- Add diagnostics: Δ-uncertainty and Δ-NLL after reflections; trigger-quality vs. difficulty; failure modes where reflection fails to reduce uncertainty.

Revised Draft
# Reflective Decoding: Adaptive Computation via Hidden KV Cache Manipulation

## Abstract
Autoregressive language models allocate similar compute per output token regardless of difficulty. We propose Reflective Decoding, a training-free inference method that adaptively allocates computation by inserting hidden “reflection” tokens directly into the Key-Value (KV) cache when next-token uncertainty is high. These reflections update the model's internal state without altering the visible output. The method comprises: (i) an uncertainty gate aligned with the active sampling policy; (ii) a greedy stopping rule that “peeks” at post-reflection uncertainty; and (iii) strict compute accounting via Forward-Pass Equivalents (FPE), latency, and memory, including marker-token overhead and branching costs. Reflections are masked from emission via a disallow list. We preregister evaluation on GSM8K and BBH with Mistral-7B and Llama-3-8B, hypothesizing Pareto gains in accuracy vs. compute over standard decoding and budgeted Chain-of-Thought (CoT). The approach is model-agnostic, training-free, and reproducible with open-source code.

## 1. Introduction
LLMs often under-allocate compute on hard steps and over-allocate on easy ones. Visible CoT increases compute indiscriminately and exposes traces. We introduce Reflective Decoding: when next-token uncertainty is high, we pause visible generation, emit hidden reflection tokens that update the KV cache, and resume visible decoding once uncertainty subsides. This enables token-level adaptive computation without fine-tuning or visible scratchpads.

Contributions:
- A training-free method for token-level adaptive reasoning via hidden KV updates and emission masking.
- A greedy dynamic stopping rule that peeks at uncertainty after a hypothetical reflection end.
- Correct and transparent compute accounting (FPE), including marker-token and peek costs; latency and memory measurements with KV branching.
- A preregistered, reproducible evaluation with strong baselines and ablations on open models.

## 2. Related Work
- Selective/Dynamic CoT and Self-Consistency: example-level triggering and majority voting with visible traces; we operate at token granularity and hide reflections.
- Internal/Latent Thoughts (e.g., Quiet-STaR): train-time encouragement of internal reasoning; our approach is training-free at inference.
- Uncertainty-guided behaviors: abstention/tool use triggered by uncertainty; we instead inject hidden computation to reduce uncertainty in situ.
- Inference efficiency: complementary to speculative decoding, paged KV caches, and flash attention; can be combined.

We will provide full citations in the paper.

## 3. Method

### 3.1 Setup and Notation
Visible tokens are `y`, hidden reflection tokens are `r`. At step `t`, the context `C_t` interleaves prior visible tokens and hidden reflections. The model defines `p(· | C_t)`. The user only sees `y`; `r` updates the KV cache but is never emitted.

### 3.2 Sampling-Consistent Uncertainty Gating
We estimate uncertainty under the active sampling policy:
- Apply temperature T and nucleus/top-p truncation, then renormalize over support `S_t`.
- Define uncertainty u_t as either:
  - Renormalized entropy: `u_t = −Σ_{i∈S_t} p_t(i) log p_t(i)`, optionally normalized by `log |S_t|` to yield [0,1]; or
  - Alternatives: 1 − p_max, logit margin, or top-k entropy (reported in ablations).
- Trigger reflection when `u_t > τ_high`. Use hysteresis (`re-arm` only when `u_t < τ_low`) to avoid thrashing.

Calibration:
- We provide two modes: (a) normalized-entropy with default thresholds (e.g., τ_high=0.65, τ_low=0.45) for portability; and (b) dev-set calibration of τ on 200–500 examples to target a reflection rate (e.g., 5–15% of steps).
- We report sensitivity and cross-task transfer for chosen thresholds.

### 3.3 Hidden Reflections and Emission Masking
On trigger:
- Append a start marker `R_START` to the KV cache.
- Generate hidden tokens `r` with a reflection policy (e.g., higher temperature/top-p) to encourage exploration.
- Upon stopping (Sec. 3.4), append an end marker `R_END`.

Emission control:
- Maintain an emission disallow list that masks `R_START`, `R_END`, and any reflection-only control tokens by setting their logits to −∞ during visible decoding. This preserves sampling consistency over the remaining support (we re-normalize after masking).

Marker selection (robustness and cost):
- Prefer single-ID reserved special tokens present in the model’s vocabulary (e.g., reserved tokens in Llama-family), minimizing marker length and peek cost.
- If unavailable, use short, statistically rare multi-token sequences verified to be:
  - Rare in natural text (low frequency in pretraining-like corpora approximations),
  - Stable under BPE segmentation across contexts,
  - Not semantically misleading.
- Always verify that markers are unreachable in visible decoding (banlist) and appropriately logged/filtered in internal telemetry.

### 3.4 Greedy Dynamic Stopping via Peek
After generating each hidden token `r_i`:
- Create a branched KV state (copy-on-write) and append `R_END` to the branch.
- Compute next-token uncertainty `u_peek` on this branched state under the visible sampling policy.
- Stop reflection if `u_peek ≤ τ_stop`, or on hitting per-reflection length cap or the global hidden-token budget.

Cost of a peek:
- Appending `R_END` in the branch requires forward passes equal to the number of tokens in `R_END` (m_e). We highlight this cost here and in Sec. 3.6.
- Optimization: Dual-path batching—evaluate the continuation with and without `R_END` in a single micro-batch using shared prefix KV to amortize overhead.

### 3.5 Budgets and Guardrails
- Global hidden budget `B_total`: max hidden tokens per sequence.
- Per-reflection cap `L_reflect`: max hidden tokens per reflection.
- Cooldown `c`: minimum visible steps after a reflection before re-arming.
- No nesting: the gate is disabled during a reflection.

### 3.6 Compute Accounting, Latency, and Memory
Let:
- V: number of visible tokens generated,
- H: total hidden tokens generated across all reflections,
- N_ref: number of reflections,
- m_s, m_e: token lengths of `R_START`, `R_END`.

Forward-Pass Equivalents (FPE):
- Visible generation: V
- Hidden generation: H
- Markers on the main path: N_ref(m_s + m_e)
- Peek cost per hidden token: appending `R_END` on the branched path costs m_e per hidden token (assuming logits for the post-`R_END` position are available from that last forward).
- Total: FPE = V + H + N_ref(m_s + m_e) + H·m_e
  = V + H(1 + m_e) + N_ref(m_s + m_e)

We report FPE alongside:
- Latency: wall-clock time and tokens/s, with and without dual-path batching.
- Memory: peak KV allocation, number and size of branches, and copy-on-write granularity.

Implementation notes:
- Branching uses copy-on-write over KV blocks/pages to avoid O(sequence) copies.
- We micro-batch peek and non-peek paths to leverage tensor cores and reduce scheduling overhead.
- We reuse precomputed logits for visible steps whenever possible.

### 3.7 Practical Concerns
- Context limits: Hidden tokens and markers consume context. We enforce strict budgets and can integrate sliding-window or chunked attention if needed.
- Safety and leakage: Hidden reflections are never emitted and are masked from UI logs by default. We provide hooks to audit hidden content and configurable filters. We document end-to-end tests verifying that markers never surface in user-visible channels.
- Failure modes: Misfired reflections may not reduce uncertainty; budgets, hysteresis, and cooldown mitigate oscillations. We report Δ-uncertainty distributions to characterize efficacy.

### 3.8 Pseudocode
```python
def reflective_decode(model, prompt, cfg):
    # cfg: thresholds (tau_high, tau_low, tau_stop),
    #      budgets (B_total, L_reflect, cooldown),
    #      sampling (visible, reflect),
    #      tags (R_START, R_END), emission_banlist
    kv = KV.from_text(prompt)
    y = []
    hidden_used = 0
    cooldown_counter = 0
    gate_armed = True
    n_reflections = 0

    while not stop_condition(y, kv):
        logits_vis = model.forward_last(kv, cfg.visible)
        logits_vis = apply_banlist(logits_vis, cfg.emission_banlist)
        u_vis = entropy_over_truncated(logits_vis, cfg.visible)

        can_reflect = (gate_armed and cooldown_counter == 0 and
                       hidden_used < cfg.B_total)

        if can_reflect and u_vis > cfg.tau_high:
            # Start reflection
            append_tokens(model, kv, cfg.tags.R_START)  # costs m_s FP
            n_reflections += 1
            gate_armed = False

            for _ in range(cfg.L_reflect):
                if hidden_used >= cfg.B_total:
                    break

                # Generate one hidden token
                h_tok = sample_from_logits(model.forward_next(kv, cfg.reflect))
                append_token(kv, h_tok)
                hidden_used += 1

                # Peek: branch, append R_END, and compute u_peek
                kv_peek = kv.branch_copy()
                append_tokens(model, kv_peek, cfg.tags.R_END)  # costs m_e FP
                peek_logits = model.forward_last(kv_peek, cfg.visible)
                peek_logits = apply_banlist(peek_logits, cfg.emission_banlist)
                u_peek = entropy_over_truncated(peek_logits, cfg.visible)

                if u_peek <= cfg.tau_stop:
                    break

            # End reflection
            append_tokens(model, kv, cfg.tags.R_END)  # costs m_e FP
            cooldown_counter = cfg.cooldown
            continue

        # Normal visible generation
        v_tok = sample_from_logits(logits_vis)
        append_token(kv, v_tok)
        y.append(v_tok)

        # Update gating state
        cooldown_counter = max(0, cooldown_counter - 1)
        if u_vis < cfg.tau_low:
            gate_armed = True

    return detokenize(y)
```

### 3.9 Calibration and Defaults
- Tune τ and budgets on a small dev split to target a reflection rate (5–15%) and hidden-token cap (e.g., ≤10% of visible tokens).
- Report sensitivity to τ_high, τ_stop, and B_total.
- Provide portable defaults with normalized entropy when calibration is unavailable.

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- H1: At matched FPE, Reflective Decoding improves accuracy over standard decoding and visible CoT baselines.
- H2: Uncertainty-gated reflections outperform non-adaptive reflection strategies (periodic/random triggers) at matched hidden-token budgets.
- H3: Greedy dynamic stopping reduces FPE for comparable accuracy versus fixed-length reflections.

### 4.2 Tasks, Models, Hardware
- Tasks: GSM8K, BBH, plus one non-math benchmark (e.g., short-form QA) for generality.
- Models: Mistral-7B-Instruct, Llama-3-8B-Instruct.
- Hardware: A100 80GB; inference with FlashAttention and paged KV caches.

### 4.3 Baselines (Compute-Matched)
- Standard decoding: greedy or temperature sampling, no reflections.
- Prefix CoT: fixed budget of visible CoT tokens at the beginning; budget chosen to match FPE with our method (includes CoT tokens).
- Example-level Selective CoT: visible CoT prefix if initial uncertainty is high; FPE-matched.
- Self-Consistency: k samples (with or without visible CoT), k chosen to match FPE.
- ToT/ReAct-style visible reasoning: fixed small beam/tree with FPE-matched compute.
- Uniform Reflections (Ablation): hidden reflections triggered periodically (every N steps), same H as ours.
- Fixed-Length Reflections (Ablation): uncertainty-gated but without dynamic stopping.
Clarifications: “Budgeted CoT” refers to visible tokens counted in FPE; “Non-adaptive reflections” refers to periodic/random triggers with fixed cadence; lengths may be fixed or capped as specified.

### 4.4 Metrics and Analyses
- Accuracy: task-specific metrics with 95% CIs.
- Compute: FPE (with marker and peek costs), latency, tokens/s.
- Memory: peak KV, branch counts/sizes, copy-on-write overhead.
- Mechanistic diagnostics: distributions of Δ-uncertainty and Δ-NLL from pre- to post-reflection; trigger precision vs. difficulty proxies.
- Error analysis: categories helped/hurt; ablations for uncertainty metric choice.

### 4.5 Reproducibility
We release code, configs, seeds, environment specs, and model hashes. Scripts cover calibration, evaluation, and figure generation. We include tests verifying marker non-emission and logging hygiene.

## 5. Results
To be populated post-preregistration. We expect:
- Accuracy–FPE Pareto curves showing improvements over compute-matched baselines.
- Ablations supporting H2 and H3.
- Microbenchmarks quantifying peek/branching overhead and speedups from dual-path batching.

## 6. Discussion
Reflective Decoding targets computation to uncertain steps, improving reasoning without visible traces. It is orthogonal to speculative decoding and caching optimizations. Effectiveness hinges on reliable uncertainty gating, low-cost markers, and efficient branching; our ablations and diagnostics analyze these factors.

## 7. Limitations and Risks
- Uncertainty is an imperfect proxy; misfires waste compute. Budgets and hysteresis mitigate.
- Overhead: marker and peek costs can be non-trivial; single-ID markers and batched peeks reduce impact.
- Context length pressure: hidden tokens consume window; budgets and sliding windows help.
- Safety and leakage: hidden content must be filtered and excluded from user-visible logs; we provide audit controls.

## 8. Conclusion
Reflective Decoding provides a training-free, uncertainty-guided mechanism for adaptive computation in LLMs by manipulating the KV cache with hidden reflections. With corrected compute accounting and rigorous, preregistered evaluation on open models, it offers a reproducible path to Pareto improvements in accuracy vs. compute.
