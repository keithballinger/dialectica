Major Revisions

Brief critique
- Evidence gap and internal inconsistency: The abstract and Section 5 imply positive results, while a bracketed note says results are not yet available. Leading venues will not accept without completed, reproducible experiments; remove provisional claims or add full results with code, tables, and plots.
- Novelty positioning: Token-level uncertainty-gated hidden reflections are promising, but you must delineate from prior selective/dynamic CoT, verifier-guided decoding, and internal monologue work. Add missing citations and an ablation isolating “token-level gating + hidden scratchpad” from example-level gating and always-on short CoT.
- Algorithmic clarity: Dynamic stopping during reflection is underspecified. Precisely define how you “peek” the post-reflection uncertainty (KV forking, close-tag insertion, and compute accounting). Clarify entropy approximation costs under top-k/top-p sampling.
- Compute accounting and fairness: Define forward-pass counting identically across methods (visible, hidden, and lookahead passes), include prompt-processing cost, and report accuracy vs. matched compute and vs. latency (median/p95). Ensure Budgeted CoT and other baselines use identical answer formatting and “hidden scratchpad” visibility rules.
- Implementation details: Specify tokenizer/tag choices that won’t explode token counts; detail filtering to prevent control-token leakage; report hardware (e.g., A100 80GB), software stack (CUDA, PyTorch, Transformers, FlashAttention), dtype/quantization, batch size, and KV-cache settings.
- Reproducibility and calibration: Separate calibration and test splits; document threshold selection, seeds, and hyperparameter grids. Release code, configs, and per-example logs.
- Risk analysis: Expand on bias amplification and long-context pressure from hidden tokens; discuss fallback behavior and monitoring in deployment.

Revised Draft
# Entropy-Gated Reflective Decoding: Spending Test-Time Compute Only When the Model Is Uncertain

## Abstract
We propose Entropy-Gated Reflective Decoding (EGRD), a training-free inference procedure that injects short, bounded “reflection” segments only when the model’s next-token uncertainty exceeds a threshold. Reflections are appended to the model’s internal context (scratchpad) but are excluded from the user-visible output. EGRD concentrates extra compute at high-uncertainty steps to improve the accuracy–compute trade-off relative to standard decoding and always-on chain-of-thought (CoT). We present a complete algorithm and a preregistered evaluation plan on GSM8K and BBH with Mistral-7B and Llama-3-8B, including matched-compute baselines and ablations. Code, configs, and logs will be released to enable validation with small open-source models.

## 1. Introduction
Chain-of-thought (CoT) improves reasoning but increases latency and cost by generating long reasoning chains even when the model is already confident. We ask whether token-level allocation of test-time reasoning can yield better accuracy per unit of compute.

EGRD monitors next-token uncertainty during decoding and triggers short hidden “reflection” segments when uncertainty is high. Reflections influence the model’s internal state (KV cache) but are omitted from the user-visible output. The controller is training-free, model-agnostic, and designed to:
- avoid unnecessary reasoning when the model is confident, and
- spend extra compute only when uncertainty spikes.

Contributions:
- A training-free token-level controller that gates hidden reflections using uncertainty with guardrails (budget, cooldown, hysteresis).
- A precise, reproducible algorithm with compute accounting that integrates with Hugging Face generation loops.
- A preregistered evaluation plan with matched-compute baselines, latency distributions, and ablations to isolate the value of uncertainty gating.
- An open-source implementation for small models (Mistral-7B, Llama-3-8B) to enable independent validation.

## 2. Related Work
- Adaptive computation and halting: ACT and early-exit Transformers adapt per-input compute within a pass; we adapt this idea to autoregressive decoding without training.
- Selective/dynamic CoT: Prior work decides when/how much to think using confidence signals or verifiers at the example level or with fixed budgets. We perform token-level gating with hidden, bounded reflections during generation.
- Verifier- and confidence-guided decoding: Token-level confidence, margin/entropy thresholds, and token-critic/verifier reranking inform our gating signals and baselines.
- Internal monologue: Methods like Quiet-STaR train latent reasoning; EGRD induces hidden reasoning purely at inference time.
- Efficiency methods: Speculative decoding accelerates sampling without altering distributions; EGRD reallocates computation for quality improvements and can compose with accelerators.

## 3. Method

### 3.1 Overview
At each visible decoding step t, the controller computes uncertainty u_t from the next-token distribution p_t. If u_t exceeds a threshold and budget constraints are met, it opens a hidden reflection segment:
- Insert an opening tag (e.g., “<REFLECT>”) and a brief instruction for structured reasoning.
- Generate up to L_reflect hidden tokens, optionally with dynamic stopping.
- Close the reflection (“</REFLECT>”), insert a resume tag (“<CONTINUE>”), and continue visible decoding.

Hidden tokens update the model’s KV cache but are not emitted to the user. Tags are literal strings (no special-token IDs required) chosen to avoid token explosion and unintended semantics.

### 3.2 Gating signals
- Entropy H_t = −Σ_i p_t(i) log p_t(i), computed over the nucleus or top-k set consistent with the sampling head. We validate that approximations preserve the trigger ranking.
- Logit margin M_t = log p_t(y₁) − log p_t(y₂), with y₁/y₂ as top-1/2 tokens.
- Optional token-type priors: modest reweighting near numerals/equations/EOS (reported as an ablation).

We calibrate thresholds on a small split (≤500 examples), targeting trigger rates r ∈ {5%, 10%, 20%}.

### 3.3 Budgets and stability
- Global hidden-token budget B_reflect per example.
- Cooldown c visible tokens after a reflection to avoid back-to-back triggers.
- Hysteresis with τ_high and τ_low (τ_low < τ_high) to reduce oscillation.
- Gating disabled within reflections.

### 3.4 Dynamic stopping with KV forking
To stop reflections when added compute no longer reduces uncertainty about the next visible token:
- After each reflection token, fork the current KV state (no-copy reference and one appended token; frameworks like PyTorch + FlashAttention permit efficient caching).
- In the forked state, append the close+resume tags (“</REFLECT>\n<CONTINUE>\n”) and compute u_vis, the uncertainty for the next visible token distribution with the “answer head.”
- If u_vis ≤ τ_stop, end the reflection; otherwise continue reflecting up to L_reflect or budget B_reflect.
- The forked forward pass is counted in total compute.

This procedure approximates a myopic value-of-compute rule: reflect while it reduces uncertainty about the imminent visible token.

### 3.5 Compute accounting
We report:
- Forward-pass equivalents (FPE): number of Transformer steps with KV cache enabled, including visible tokens, hidden reflection tokens, and forked “peek” passes.
- Output tokens (visible only) and total tokens (visible + hidden).
- Wall-clock latency (median, p95) with fixed hardware/software.
All baselines are measured with identical counting conventions and include prompt and system tokens.

### 3.6 Pseudocode
```python
def egrd_decode(model, prompt, params):
    # params: τ_high, τ_low, τ_stop, L_reflect_max, B_reflect, cooldown,
    #         gating_type ∈ {entropy, margin}, answer_head, reflect_head, dynamic_stop
    ctx = prompt               # text buffer mirrored in the model's KV
    out = ""                   # user-visible output
    used_reflect, cd = 0, 0
    armed = True

    while not stop_visible(out):
        logits = model.forward_last(ctx)              # KV cached
        probs = sample_head_softmax(logits, params.answer_head)
        u = uncertainty(probs, params.gating_type)

        if armed and cd == 0 and used_reflect < B_reflect and u > params.τ_high:
            # Open reflection
            ctx += "<REFLECT>\nThink step-by-step. Be concise.\n"
            tok_count = 0
            while tok_count < params.L_reflect_max and used_reflect < B_reflect:
                tok_r = sample_next(model, ctx, params.reflect_head)
                if tok_r == "</REFLECT>":
                    break
                ctx += tok_r
                tok_count += 1
                used_reflect += 1

                if params.dynamic_stop:
                    # KV fork: evaluate next visible uncertainty if we were to close now
                    u_vis = peek_next_visible_uncertainty(model, ctx, params)
                    if u_vis <= params.τ_stop:
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
Implementation notes:
- Heads: reflection vs. answer may use different temperatures/top-p.
- Tags are plain strings (e.g., XML-like) known to tokenize compactly for Llama/Mistral; we filter any occurrence of these tags from the visible output.
- Entropy over the nucleus/top-k is consistent with the active sampling head; we confirm that rankings of u_t are stable vs. full-vocab entropy.
- KV forking can be implemented by cloning the attention cache metadata and appending short tag sequences; we count the extra forward pass.

### 3.7 Threshold calibration
- Split each dataset into calibration (≤500 examples) and test.
- For each model/task, sweep τ_high over the empirical u_t distribution to achieve target trigger rates r. Set τ_low = α · τ_high (α ∈ [0.7, 0.9]).
- Set τ_stop by minimizing reflection length subject to non-inferior validation accuracy vs. fixed L_reflect on the calibration split.
- Fix all hyperparameters prior to test-set evaluation.

## 4. Evaluation

### 4.1 Hypotheses (preregistered)
Primary: EGRD yields strictly higher accuracy at equal or lower compute (FPE) than Standard decoding and Budgeted CoT for at least one configuration, expanding the accuracy–compute Pareto frontier.
Secondary: Uncertainty gating outperforms random or periodic reflections under matched hidden-token budgets.
Tertiary: Dynamic stopping reduces compute at equal accuracy vs. fixed L_reflect.

### 4.2 Datasets and models
- GSM8K (exact match) and BBH (subset: arithmetic, symbolic, and commonsense tasks).
- Mistral-7B-Instruct-v0.2 and Llama-3-8B-Instruct (Hugging Face, bf16), single-GPU.

### 4.3 Baselines
- Standard: greedy or low-temperature decoding, no CoT.
- Fixed CoT: “Let’s think step by step” with visible chain-of-thought.
- Budgeted CoT: same prompt but truncate reasoning tokens to match EGRD’s hidden-token budget; chain remains hidden to equalize answer formatting.
- Example-level gated CoT: trigger a single CoT segment if initial uncertainty exceeds a threshold.
- Random/periodic reflections: same hidden-token budget as EGRD, no uncertainty signal.
- Self-consistency (k=5): reference upper bound for accuracy vs. compute.
- Optional verifier reranking baseline when available.

All methods use identical prompts and answer-format instructions; only the reasoning schedule differs.

### 4.4 Metrics and reporting
- Accuracy (per task), with bootstrap 95% CIs and McNemar’s test vs. the best baseline at matched compute.
- Compute: FPE, total tokens (visible + hidden).
- Latency: median and p95 on fixed hardware.
- Reflection stats: trigger rate, reflections per example, reflection length.
- Plots: Accuracy vs. FPE; Accuracy vs. median/p95 latency; CDFs of reflection length.

### 4.5 Hardware and software
- GPU: NVIDIA A100 80GB, CUDA 12.x; CPU details; batch size; tokenization backend.
- Software: PyTorch 2.x, Transformers x.y, FlashAttention v2, inference dtype (bf16), KV cache enabled, no quantization unless stated.
- Seeds and deterministic flags specified; per-example logs released.

### 4.6 Ablations
- Gating signal: entropy vs. margin; with/without hysteresis; cooldown c ∈ {0,1,3}.
- Reflection: fixed L_reflect vs. dynamic stop; reflection temperature; instruction variants.
- Trigger policy: token-type priors on/off.
- Task type: arithmetic vs. symbolic vs. commonsense.

### 4.7 Reproducibility
We release:
- ~300 LOC controller and wrappers for Hugging Face models.
- Configs, calibration splits, thresholds, and hyperparameter grids.
- Per-example logs with u_t traces, triggers, and compute accounting.

## 5. Results
[To be populated with completed experiments following the preregistered plan. No claims are made here until results, code, and logs are available.]

## 6. Discussion
Why it may help: Local spikes in uncertainty often precede errors. Short, targeted reflections can restructure context and reduce ambiguity, improving next-token fidelity without paying the full cost of always-on CoT.

When it may not: Poor calibration can cause misfires; tasks requiring very long derivations may not benefit from brief reflections; reflection tags/instructions can interact with model priors.

Composability: EGRD is orthogonal to speculative decoding (the controller operates on accepted tokens) and can trigger retrieval or verifier checks on uncertainty spikes.

Deployment: Hidden tokens increase API cost; bursts increase latency variance (p95). Streamed UIs may need buffering or adaptive chunking.

## 7. Limitations and risks
- Entropy/margin are imperfect uncertainty proxies; per-task calibration is required.
- Hidden reflections lengthen the internal sequence and can pressure context windows in long dialogues.
- Control-token leakage must be filtered from visible output.
- Gains may shrink with larger, better-calibrated models.
- Hidden reflections can amplify biases if they entrench incorrect reasoning; monitor with targeted evals.

## 8. Conclusion
EGRD is a simple, training-free approach to allocate test-time reasoning where uncertainty is high. We specify the algorithm, compute accounting, and a preregistered evaluation plan suitable for validation on small open-source models. Upon releasing code and results, we will assess whether EGRD expands the accuracy–compute Pareto frontier relative to standard and fixed-budget CoT decoding.

## References
- Chen, C., et al. (2023). Accelerating large language model decoding with speculative sampling. arXiv:2302.01318.
- Graves, A. (2016). Adaptive computation time for recurrent neural networks. arXiv:1603.08983.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
- Madaan, A., et al. (2023). Self-Refine: Iterative refinement with self-feedback. NeurIPS.
- Ren, L., et al. (2023). Confidence estimation and OOD for LLMs. ICLR.
- Wang, X., et al. (2023). Self-consistency improves chain-of-thought reasoning in LLMs. arXiv:2203.11171.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS.
- Zelikman, E., et al. (2024). Quiet-STaR: Language models can teach themselves to think before speaking. arXiv:2403.09629.

(Additional citations to dynamic/selective CoT and verifier-guided decoding to be completed in the camera-ready.)
