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
- Novelty and focus: The entropy-integral halting (EIH) idea is clear, training-free, and squarely in LLM inference. It plausibly impacts efficiency–quality trade-offs for agentic workflows and is testable on small open models. Structure and evaluation plan are appropriate for a leading venue.
- Claims vs mechanics: The text over-attributes termination to U_max. U_max bounds the integral state but does not ensure termination; L_max (and/or answer-cue injection) guarantees it. Clarify this to avoid misleading implications.
- Code issues (must-fix):
  - Temperature zero: entropy computation divides by Tr; Tr=0 causes NaNs. Clamp or special-case Tr for entropy.
  - Phase switch bug: when the answer cue is detected, the code breaks before advancing the model with the cue token, so answer-phase logits are stale. Update past_kv/logits before switching phases.
  - Consistent answer-phase entry: when EIH halts without an emitted cue, inject the cue tokens (with cache) before answer decoding to exclude answer tokens from the integral and standardize formatting.
  - Minor: make top-k/top-p filtering operate on temperature-scaled logits (already correct); keep that explicit. Add a hard stop L_max in the loop condition and clarify its role.
- Textual refinements:
  - Replace redundancy between Algorithm 1 and appendix with a high-level overview in main text (kept).
  - Explicitly motivate the windowed mean and dual-condition halting, and correct the role of U_max.
  - Add a tiny, self-contained test for the controller math (no model required) to validate fixes.
- Evaluation: The preregistration is solid. Add a brief termination/stability check, and report failures where entropy stays high until L_max.

Revised Draft
# Entropy-Integral Halting for Chain-of-Thought Decoding

## Abstract
We introduce a training-free decoding controller that adaptively halts chain-of-thought (CoT) generation by integrating next-token entropy over time. Treating entropy as a proxy for unresolved branching, the controller stops when (i) a bounded-memory integral of smoothed entropy has decayed to its budget and (ii) recent entropy is stably low. This integral rule is robust to transient spikes and verbosity, is model-agnostic, and adds negligible overhead with cached decoding. We release reference code and a preregistered protocol on small open models to enable community validation. Contributions: (1) a windowed entropy-integral halting rule with a stability condition; (2) a minimal implementation using past-key caching; (3) an evaluation suite with baselines and ablations; and (4) analyses of calibration, overhead, and generalization.

## 1. Introduction
Reasoning traces often improve reliability but increase latency and token cost. Fixed CoT lengths waste compute on easy inputs and truncate hard ones. Adaptive schemes that rely on instantaneous signals (e.g., max-prob updates, local entropy dips, stop phrases) can be brittle to transient confidence spikes and formatting variability.

We propose an entropy-integral controller that accumulates a smoothed next-token entropy signal and halts when both a budgeted integral is depleted and the recent entropy is consistently low. Intuitively, the controller allocates more tokens while uncertainty persists, and stops once uncertainty collapses, providing difficulty-adaptive length without additional training.

## 2. Related Work
- Fixed-length/stop-phrase CoT: simple but inefficient; sensitive to prompt style.
- Local confidence/entropy thresholds and patience decoding: vulnerable to transient spikes and verbosity.
- Adaptive computation time (ACT, PonderNet): adapt compute via training; not a training-free, token-level controller.
- Confident adaptive language modeling (CALM): layer/token exits; complementary to our trajectory-level signal.
- Uncertainty-aware decoding: prior local entropy/logit-margin criteria; we formalize a bounded-memory integral with an explicit stability condition.

## 3. Method

### 3.1 Setup
- Any autoregressive LM exposing logits and cached decoding.
- Standard CoT prompting plus an answer cue (e.g., “Therefore, the answer is:”).
- Temperatures: Tr for CoT, Ta for answer (often Ta=0).

### 3.2 Entropy signal
At step t, with logits z_t, define p_t = softmax(z_t/Tr). Entropy H_t = −Σ_v p_t(v) log p_t(v), normalized Ĥ_t = H_t / log |V|. We compute H_t from pre-sampling logits each step for stability.

### 3.3 Entropy-Integral Halting (EIH)
Maintain:
- Windowed mean m_t over the last W normalized entropies (causal padding early).
- Integral state U_t updated as U_t = clip(U_{t−1} + (m_t − h_ref), 0, U_max) with U_0 as the initial budget.

Halt CoT when both:
- U_t ≤ 0 (budget depleted), and
- m_t ≤ τ_low (recent entropy stably low),
with an unconditional hard cap L_max on CoT tokens.

Interpretation and design choices:
- Windowing reduces single-token noise; the dual condition prevents stopping on transient dips.
- U_max bounds the memory of past uncertainty so brief periods of high entropy cannot indefinitely delay halting once m_t falls; L_max guarantees termination even if uncertainty never collapses.
- We exclude the answer segment from the integral by switching phase at the cue or by injecting the cue when halting due to EIH.

### 3.4 Optional: Adaptive self-consistency
Define a difficulty proxy D = (U_end / U_max) + m_end and map D to k consistency samples via a linear schedule [k_min, k_max]. Ablate the components.

### 3.5 Algorithm overview
- Initialize window, U=U_0, and decode with cache.
- At each CoT step: compute Ĥ_t from logits, update m_t and U_t, check halting (U_t ≤ 0 and m_t ≤ τ_low) or answer-cue detection.
- If halting due to EIH and the cue has not been emitted, inject the cue tokens (cached) and switch to answer decoding at Ta.
- Answer phase uses temperature Ta and ignores the controller.

Default hyperparameters (small dev-sweep): W=5, h_ref=0.2, τ_low=0.1, U_0=10, U_max=20, L_max=128; modest task-specific tuning via a 200-example dev set.

## 4. Evaluation Protocol (Preregistered)

### 4.1 Models
- Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct; optional: Qwen2-7B, Phi-3-medium.

### 4.2 Tasks
- GSM8K, StrategyQA, BIG-Bench Hard subsets; stress tests: verbose prompts, multilingual paraphrases, distractors.

### 4.3 Baselines
- No-CoT; Fixed-L CoT (L ∈ {32, 64, 128}); stop-phrase; local entropy/confidence thresholds; patience decoding; CALM-style proxy; oracle Fixed-L with small-k self-consistency.

### 4.4 Metrics and reporting
- Primary: accuracy and average CoT tokens.
- Secondary: latency, tokens/s, GPU util; energy (NVML/CodeCarbon) as Joules/token.
- Curves: accuracy vs average CoT tokens; AUP.
- Robustness: prompt/temperature/style variants.
- Statistics: paired bootstrap 95% CIs; Holm–Bonferroni across baselines.
- Tuning guardrails: tune on 200-example dev; report only held-out.

### 4.5 Ablations
- Remove integral; vary W ∈ {1,3,5,9}; Tr ∈ {0.2,0.7,1.0}.
- Heatmaps over h_ref × τ_low; cross-task transfer.
- Drop stability condition; adaptive self-consistency variants.
- Answer-cue handling (explicit injection vs detection).

### 4.6 Reproducibility artifacts
- MIT-licensed code, configs, seeds, environment files; prompts and extraction rules; dataset versions/splits; CPU verification on 10–100 samples; commit hashes and manifests.

## 5. Analysis Plan (Confirmatory)
- H1 (Pareto): EIH ≥5% higher AUP vs patience decoding at matched accuracy on GSM8K and ≥1 BBH subset (paired bootstrap; Holm–Bonferroni).
- H2 (Overhead): controller overhead <2% of decode time with cache (microbenchmarks).
- H3 (Stability): when EIH halts, appending 5 extra reasoning tokens leaves the final answer unchanged in ≥90% of easy cases (short-CoT dev-labeled).
- Failure analysis: verbosity-inflated entropy, distribution shifts, truncation of pedagogical but unnecessary steps.
- Safety: refusal prompts; confirm no elevated truncation of safety rationales vs patience at matched accuracy.

## 6. Discussion
Why an entropy integral? Local signals are noisy and style-sensitive. A bounded-memory integral over a smoothed uncertainty signal:
- Allocates compute proportional to unresolved branching.
- Filters transient dips (stability condition).
- Composes with architectural early-exit methods.

Limitations:
- Entropy conflates epistemic and stylistic uncertainty; windowing and dual-condition halting mitigate but do not eliminate this.
- Light per-task tuning is needed; we report transfer and provide defaults.
- Requires logits access; pure black-box APIs may not support EIH.

## 7. Broader Impact
Reducing unnecessary CoT tokens lowers cost and energy. Miscalibrated halting could shorten safety rationales; we include diagnostics and recommend conservative defaults in high-stakes settings.

## 8. Conclusion
EIH is a simple, training-free controller for adaptive CoT halting that is robust to local noise and integrates easily with standard decoding. With open code and a preregistered protocol on small open models, we enable independent validation and expect improved efficiency–quality trade-offs in agentic workflows.

## References
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks.
- Banino, A., et al. (2021). PonderNet: Learning to Ponder.
- Schuster, T., et al. (2022). Confident Adaptive Language Modeling.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models.

## Appendix A: Minimal HF Integration (PyTorch, corrected)
```python
import torch
import torch.nn.functional as F
import math
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import top_k_top_p_filtering

ANSWER_CUE = "Therefore, the answer is:"

@torch.no_grad()
def eih_generate(
    model, tok, prompt,
    Tr=0.7, Ta=0.0,
    W=5, h_ref=0.2, tau_low=0.1,
    U0=10.0, U_max=20.0, L_max=128,
    top_k=None, top_p=None,
    max_new_tokens=256,
):
    """
    Entropy-Integral Halting (EIH) with answer-phase separation.
    Returns (cot_text, answer_text).
    """
    device = model.device
    eos_id = tok.eos_token_id
    cue_ids = tok.encode(ANSWER_CUE, add_special_tokens=False)
    input_ids = tok(prompt, return_tensors="pt").to(device).input_ids

    # Prime model and cache
    out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_logits = out.logits[:, -1, :]

    # Controller state
    window = deque(maxlen=W)
    U = U0
    cot_ids = []
    emitted_cue = False
    steps = 0

    # Helper: compute normalized entropy safely
    def normalized_entropy(logits, temp):
        t = max(temp, 1e-5)
        log_probs = F.log_softmax(logits / t, dim=-1)
        probs = log_probs.exp()
        H = -(probs * log_probs).sum(dim=-1)  # nats
        return (H / math.log(probs.size(-1))).item()

    # CoT loop (bounded by L_max)
    while steps < L_max:
        # 1) Controller update from pre-sampling logits
        H_norm = normalized_entropy(next_logits, Tr)
        window.append(H_norm)
        m = sum(window) / len(window)
        U = min(U_max, max(0.0, U + (m - h_ref)))

        # 2) Halting due to EIH? If yes, inject cue (once) and switch to answer phase
        if (U <= 0.0) and (m <= tau_low):
            if not emitted_cue:
                # Inject cue tokens (advance cache/logits; do not count as CoT)
                cue_tensor = torch.tensor(cue_ids, device=device).unsqueeze(0)
                out = model(input_ids=cue_tensor, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]
                emitted_cue = True
            break

        # 3) Sample next CoT token (apply temperature before filtering)
        if Tr == 0.0:
            # Greedy on raw logits (no filtering)
            next_token = torch.argmax(next_logits, dim=-1)
        else:
            softened = next_logits / Tr
            filtered = top_k_top_p_filtering(softened, top_k=top_k, top_p=top_p)
            next_token = torch.distributions.Categorical(logits=filtered).sample()

        token_id = next_token.item()
        cot_ids.append(token_id)
        steps += 1

        # 4) Advance model with the sampled token to keep cache/logits consistent
        out = model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

        # 5) Check for cue or EOS after advancing
        if len(cot_ids) >= len(cue_ids) and cot_ids[-len(cue_ids):] == cue_ids:
            emitted_cue = True
            break
        if eos_id is not None and token_id == eos_id:
            # End entirely; no answer phase expected
            emitted_cue = True  # prevents cue injection
            break

    # Answer phase (start from current cache/logits)
    answer_ids = []
    remaining = max(0, max_new_tokens - len(cot_ids))
    for _ in range(remaining):
        if Ta == 0.0:
            next_token = torch.argmax(next_logits, dim=-1)
        else:
            probs = torch.softmax(next_logits / max(Ta, 1e-5), dim=-1)
            next_token = torch.distributions.Categorical(probs=probs).sample()

        token_id = next_token.item()
        answer_ids.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break

        out = model(input_ids=next_token.unsqueeze(0), past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    cot_text = tok.decode(cot_ids, skip_special_tokens=True)
    answer_text = tok.decode(answer_ids, skip_special_tokens=True)
    return cot_text, answer_text
```

### Appendix A.1: Tiny controller sanity test (no model required)
```python
def test_eih_math():
    # Simulate a decreasing entropy window; expect halting once budget decays
    W, h_ref, tau_low = 3, 0.2, 0.1
    U0, U_max = 2.0, 5.0
    entropies = [0.5, 0.4, 0.3, 0.2, 0.08, 0.08, 0.08]  # normalized
    from collections import deque
    window, U, halts = deque(maxlen=W), U0, []
    def mavg():
        return sum(window)/len(window)
    for t, H in enumerate(entropies, 1):
        window.append(H)
        m = mavg()
        U = min(U_max, max(0.0, U + (m - h_ref)))
        halts.append((t, U, m, (U <= 0.0) and (m <= tau_low)))
    # Halt should occur after the running mean drops well below thresholds
    assert any(flag for (_, _, _, flag) in halts), "EIH did not halt in synthetic test."
    return halts

print(test_eih_math())
```

## Appendix B: Preregistration Summary
- Primary outcomes: accuracy and average CoT tokens on held-out sets with fixed prompts and seeds.
- Confirmatory comparisons: EIH vs fixed-length and patience baselines at matched accuracy; paired bootstrap; Holm–Bonferroni.
- Overhead: wall-clock microbenchmarks with/without controller (cache on); energy via NVML/CodeCarbon.
- Deviations: any changes to prompts, temperatures, or models will be documented and re-run across all methods.

## Artifact Checklist
- Source code, configs, seeds; exact prompts and answer-cue rules; dataset versions/splits; environment spec; commands and manifests for replication; license and model card references.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
