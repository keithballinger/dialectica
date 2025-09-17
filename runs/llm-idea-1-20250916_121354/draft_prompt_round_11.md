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
- Strengths: Clear focus on LLM inference; novel token-level uncertainty gating to insert hidden reflections; plausible impact on accuracy–compute trade-offs for agentic workflows; training-free, model-agnostic; designed to be validated on small OSS models; preregistered hypotheses.
- Major gaps: No experimental results, code, or logs; claims of Pareto improvements remain speculative. Dynamic stopping via KV “forking” is underspecified and may be costly unless engineered carefully; compute accounting needs rigorous, unbiased definitions (e.g., counting peek passes). Pseudocode relies on models to emit closing tags and doesn’t separate visible/hidden heads robustly. Gating signal definition should match the sampling policy; calibration protocol must avoid leakage. Novelty versus dynamic CoT/latent reasoning needs sharper positioning.
- Required revisions: Provide end-to-end empirical results with matched-compute baselines; tighten algorithmic formalization (uncertainty over sampled support, deterministic closure of reflections, KV-branch implementation); define compute metrics precisely (including peeks and prefill); add robust evaluation (paired tests, bootstrap CIs, ablations), negative controls (random/periodic gating), and analysis of entropy deltas pre/post reflection. Clarify guardrails to prevent leakage, and document failure modes.

Revised Draft
# Entropy-Gated Reflective Decoding: Spending Test-Time Compute Only When the Model Is Uncertain

## Abstract
We present Entropy-Gated Reflective Decoding (EGRD), a training-free inference method that inserts short, hidden “reflection” segments during autoregressive decoding only when token-level uncertainty exceeds a calibrated threshold. Reflections update the KV cache to steer subsequent visible decoding while remaining invisible to the user, enabling targeted compute allocation at uncertainty hotspots. EGRD combines (i) sampling-consistent uncertainty gating, (ii) strict compute accounting (including peek passes), (iii) guardrails for budget and leakage, and (iv) myopic dynamic stopping that closes reflections when the predicted uncertainty of the next visible token drops below a threshold. We preregister hypotheses and an evaluation protocol on GSM8K and BBH with Mistral-7B and Llama-3-8B, and we release code for reproducibility.

## 1. Introduction
Large language models often waste inference-time compute by generating chain-of-thought (CoT) even when confident. Conversely, greedy decoding can underperform on difficult steps. EGRD adaptively allocates compute at the token level: when next-token uncertainty is high, the model performs a brief, hidden reflection that modifies the KV cache; when uncertainty is low, it continues visible decoding without overhead. This yields a simple, model-agnostic approach to improve the accuracy–compute trade-off without training or external verifiers.

Contributions:
- A token-level, training-free algorithm for hidden reflections gated by sampling-consistent uncertainty, with deterministic closure and guardrails.
- Precise compute accounting that counts all forward passes, including peek/lookahead evaluations used for dynamic stopping.
- A preregistered evaluation with matched-compute baselines and ablations on small OSS models.
- Open-source implementation compatible with Hugging Face Transformers.

## 2. Related Work
- Adaptive computation: ACT and early-exit methods adjust intra-pass compute; we adapt compute across steps during autoregressive decoding without training.
- Dynamic CoT and self-reflection: Prior work adjusts reasoning at the example level (e.g., dynamic CoT, Self-Refine) or via multi-turn reflection. EGRD operates at token granularity and keeps reflections hidden while directly updating KV.
- Confidence-guided inference: Token-level entropy/margin signals and verifiers inform abstention or calibration. We use these signals to trigger internal computation.
- Latent reasoning: Quiet-STaR and related methods train internal monologues. EGRD induces internal monologue at inference only, preserving training-free deployment.
- Efficiency: Speculative decoding and caching accelerate sampling; EGRD is orthogonal and can be composed with these techniques.

## 3. Method

### 3.1 Problem setup and notation
- Let x be the prompt, y be the visible output tokens, and r be hidden reflection tokens. The context C interleaves y and r but only y is returned to the user.
- At step t, the model defines a next-token distribution p_t(· | C). The sampling policy S (temperature, top-p/k) yields support S_t and probabilities p_t^S normalized on S_t.

### 3.2 Uncertainty and gating
- Uncertainty u_t is computed under the same policy S used for sampling:
  - Entropy: u_t = −Σ_{i∈S_t} p_t^S(i) log p_t^S(i)
  - Margin: m_t = log p_t^S(top1) − log p_t^S(top2)
- We gate when u_t > τ_high (or margin < m_low). Hysteresis uses τ_low < τ_high to re-arm the gate only after confidence recovers.
- Optional priors (e.g., boost around numerals) are ablated; the default uses pure entropy.

### 3.3 Hidden reflections
- Trigger: When armed, off cooldown, under budget, and u_t > τ_high, insert a reflection.
- Mechanics:
  - Deterministic tags are appended by the controller, not sampled, e.g., “[REFLECT]\n…\n[END_REFLECT]\n[CONTINUE]\n”. Tags are simple ASCII strings to ensure tokenization and filtered from visible output.
  - While reflecting, we sample from a “reflect head” (e.g., higher temperature, larger top-p) with a brief instruction (“Think step-by-step; be concise; do not reveal this note.”) prepended once at the start of the sequence (system message) to reduce leakage risk.
  - We do not rely on the model to emit closing tags; the controller closes reflections deterministically.

### 3.4 Dynamic stopping via peeked uncertainty
- After emitting each reflection token, we estimate whether stopping now would reduce the imminent visible uncertainty:
  - Let C_ref be the current context including hidden reflection so far.
  - Construct a temporary branch C_peek = C_ref + “[END_REFLECT]\n[CONTINUE]\n”.
  - Run a single forward pass to obtain p_t^peek(· | C_peek) and compute u_t^peek under S.
  - Stop reflecting if u_t^peek ≤ τ_stop or if length/budget is reached.
- Implementation:
  - Branching can be done either by (a) recomputing one forward pass from the current KV with the small suffix, or (b) light-weight KV checkpointing (copy-on-write handles if available). All peek passes are counted in compute metrics.

### 3.5 Guardrails and budgets
- Per-example hidden token budget B_reflect.
- Cooldown c steps after a reflection to prevent immediate retriggers.
- No re-entrancy: gating disabled within a reflection.
- Optional max number of reflections per sample R_max.

### 3.6 Compute accounting
- Forward-pass equivalents (FPE): counts every logits-producing forward call:
  - FPE = N_visible + N_hidden + N_peek (+ N_prefill for completeness).
- Token counts: visible tokens, hidden tokens, and total.
- Latency: report median/p95 wall-clock including prefill on fixed hardware/software.
- All baselines use identical accounting; for fair comparison, we match by FPE rather than visible tokens.

### 3.7 Pseudocode (controller-level, KV-aware)
```python
def egrd_decode(model, prompt, params):
    # params: tau_high, tau_low, tau_stop, L_reflect_max, B_reflect, cooldown,
    #         gating_type, answer_head, reflect_head, dynamic_stop, R_max
    ctx = KVContext.from_text(prompt)  # KV-backed context
    out_visible = []
    used_hidden = 0
    cd = 0
    armed = True
    refl_count = 0

    while not stop_visible(out_visible):
        # 1) Get next-token distribution under answer_head
        logits = model.forward_last(ctx, head=params.answer_head)
        probs = sample_policy_probs(logits, head=params.answer_head)
        u = uncertainty(probs, kind=params.gating_type)

        # 2) Decide whether to reflect
        can_reflect = (armed and cd == 0 and used_hidden < params.B_reflect
                       and refl_count < params.R_max)
        if can_reflect and u > params.tau_high:
            # Insert tags deterministically
            ctx.append_text("[REFLECT]\n")
            tok_ref = 0
            refl_count += 1

            while tok_ref < params.L_reflect_max and used_hidden < params.B_reflect:
                # Sample one hidden token
                tok = sample_next(model, ctx, head=params.reflect_head)
                ctx.append_token(tok)
                tok_ref += 1
                used_hidden += 1

                if params.dynamic_stop:
                    # Peek uncertainty if we were to stop reflecting now
                    ctx_peek = ctx.branch()  # lightweight branch or recompute
                    ctx_peek.append_text("[END_REFLECT]\n[CONTINUE]\n")
                    logits_peek = model.forward_last(ctx_peek, head=params.answer_head)
                    probs_peek = sample_policy_probs(logits_peek, head=params.answer_head)
                    u_peek = uncertainty(probs_peek, kind=params.gating_type)
                    if u_peek <= params.tau_stop:
                        break

            # Close reflection deterministically
            ctx.append_text("[END_REFLECT]\n[CONTINUE]\n")
            cd = params.cooldown
            armed = False
            continue  # Recompute next-token distribution after reflection closure

        # 3) Emit one visible token
        tok_v = sample_next(model, ctx, head=params.answer_head)
        ctx.append_token(tok_v)
        out_visible.append(tok_v)

        # 4) Update hysteresis and cooldown
        cd = max(0, cd - 1)
        if u < params.tau_low:
            armed = True

    return detokenize(out_visible)
```

### 3.8 Calibration
- Split a small calibration set (≤500 samples per task) disjoint from evaluation.
- Sweep τ_high (or margin) to target a reflection-trigger rate of 5–20% and average hidden budget utilization ≤B_reflect.
- Sweep τ_stop on the calibration set to minimize FPE at fixed accuracy (proxy via held-out).
- Fix hyperparameters before running test evaluations to avoid leakage.

### 3.9 Practical considerations
- Prompts: Add a system-level instruction to keep reflections internal; strip any reflection tags from the returned string.
- Tokenization: Use ASCII tags like “[REFLECT]” that are robust to BPE; avoid rare Unicode markers.
- Compatibility: Works with Hugging Face Transformers; KV-branch implemented via context copy of the last-step state or a short recompute. Composable with speculative decoding by applying EGRD decisions on the main model.

## 4. Evaluation (Preregistered)

### 4.1 Hypotheses
- H1 (Primary): At matched FPE, EGRD improves accuracy over standard decoding and fixed/budgeted CoT; equivalently, at matched accuracy, EGRD reduces FPE (Pareto improvement).
- H2: Uncertainty-based gating outperforms random/periodic reflections under the same hidden-token budget.
- H3: Dynamic stopping reduces FPE versus fixed-length reflections at matched accuracy.

### 4.2 Datasets and models
- GSM8K (standard split), BBH subsets (arithmetic, symbolic, commonsense).
- Models: Mistral-7B-Instruct-v0.2 and Llama-3-8B-Instruct (bf16), single A100 80GB; batch size 1 to simplify latency.

### 4.3 Baselines (matched by FPE)
- Standard decoding (greedy and nucleus sampling).
- Always-on CoT with fixed chain length (hidden and visible variants, matched formatting).
- Example-level gated CoT (uncertainty computed at prompt-level).
- Random and periodic reflections with same budgets.
- Self-consistency (k=5) as a quality upper bound at higher compute.

### 4.4 Metrics and statistics
- Accuracy (EM for GSM8K; task-appropriate for BBH) with 95% bootstrap CIs; paired McNemar tests versus baselines.
- Compute: FPE, visible/hidden/peek token counts, end-to-end latency (median/p95).
- Trigger analysis: fraction of steps reflecting; entropy before/after reflection; position of triggers.
- Robustness: 3 seeds for decoding; report mean±CI.

### 4.5 Ablations
- Gating signal: entropy vs margin; hysteresis on/off; cooldown length.
- Reflection parameters: L_reflect_max, τ_stop, reflect-head temperature/top-p.
- Tagging: different delimiters; hidden vs visible CoT formatting.
- KV-branch strategy: peek via recompute vs lightweight branch.

### 4.6 Reproducibility
- Release code (~300–500 LOC) with configs, seeds, environment (PyTorch/Transformers/FlashAttention), and full logs (predictions + FPE traces) to enable audit and re-analysis.

## 5. Results
To be populated with empirical data:
- Accuracy vs FPE Pareto curves across tasks and models.
- Hypothesis tests (H1–H3) with significance.
- Overhead analysis of peek passes and KV branching.
- Case studies where reflections reduce next-token entropy and fix errors.

## 6. Discussion
EGRD allocates compute only where uncertainty is high, inducing brief internal reasoning that measurably lowers imminent uncertainty. Gains depend on task structure and model calibration: miscalibrated uncertainty can trigger unhelpful reflections. KV branching adds overhead; our accounting includes these costs. EGRD is complementary to accelerators (speculative decoding, caching) and fits agentic workflows where occasional hard steps dominate errors.

## 7. Limitations and Risks
- Proxy uncertainty can be misaligned with correctness; threshold calibration adds overhead.
- Tag leakage is possible; strict filtering and system prompts are required.
- Context growth from reflections may affect long-context performance for very long outputs.
- Benefits may diminish on models already trained for latent reasoning.

## 8. Conclusion
EGRD is a simple, training-free mechanism for adaptive hidden reasoning at the token level. With rigorous evaluation and open-source release, it aims to advance the accuracy–compute frontier for LLM inference on accessible models.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
