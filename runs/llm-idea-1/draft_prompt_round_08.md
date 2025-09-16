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
- Strengths: Clear problem framing, plausible novelty (trajectory-level entropy integral with stability safeguard), appropriate scope for LLM inference, and a preregistered evaluation plan with open-source artifacts. The method is model-agnostic, inexpensive, and directly relevant to agentic workflows.
- Needed fixes:
  - Pseudocode/implementation: Define answer_cue explicitly; avoid brittle string comparisons; use token-id sliding match. Ensure consistency between Tr and sampling vs argmax; compute entropy from pre-sampling logits; exclude answer phase from the integral; reset or stop on cue deterministically. Use past_key_values/use_cache to avoid quadratic overhead; otherwise overhead claims are unreliable.
  - Hyperparameters: Briefly justify defaults (U0, W, href, tau_low) with dev-set evidence and provide a tuning grid to support reproducibility.
  - Related work: Provide precise citations (ACT: Graves, 2016; PonderNet: Banino et al., 2021; CALM: Schuster et al., 2022) and distinguish from patience/threshold baselines.
  - Evaluation: Specify energy measurement methodology (e.g., CodeCarbon or NVML sampling), variance controls (seeds, batch size), and confirmatory hypotheses with paired tests and corrections.
  - HF integration: Add generate-loop with use_cache, correct temperature handling for CoT vs answer, and configurable top-p/top-k options.

Revised Draft
# Entropy-Integral Halting for Chain-of-Thought Decoding

## Abstract
We introduce a training-free decoding controller that adaptively halts chain-of-thought (CoT) generation by integrating next-token entropy over time. The controller treats entropy as a proxy for unresolved branching and halts when cumulative “uncertainty work” is depleted and recent entropy is stably low. Unlike instantaneous confidence or entropy thresholds, the integral rule is robust to transient spikes and verbosity. It is model-agnostic, adds negligible overhead with cached decoding, and integrates cleanly with standard APIs. We release reference code and a preregistered protocol on small open models to enable community validation. Contributions: (1) a windowed entropy-integral halting rule with a stability condition; (2) a minimal implementation using past-key caching; (3) a strong evaluation suite with baselines and ablations; and (4) analyses of calibration, overhead, and generalization. We hypothesize substantial CoT token savings at matched accuracy, improving the compute–quality Pareto frontier for agentic workflows.

## 1. Introduction
Reasoning traces improve reliability but inflate token and latency budgets. Fixed CoT lengths waste compute on easy inputs and truncate hard ones. Existing adaptive schemes often rely on local signals (max-probability peaks, instantaneous entropy, stop phrases) that are brittle to transient confidence spikes, verbosity, and formatting.

We propose an entropy-integral controller that accumulates normalized next-token entropy during reasoning and halts when (i) a cumulative budget is exhausted and (ii) recent entropy is stably low. Intuitively, the controller allocates more tokens when branching uncertainty persists and stops once uncertainty collapses. The method is training-free, architecture-agnostic, and compatible with standard decoding.

## 2. Related Work
- Fixed-length and stop-phrase CoT: Simple but inefficient; sensitive to prompt style and verbosity (e.g., Yao et al., 2023).
- Local confidence/entropy thresholds and patience decoding: Susceptible to transient spikes and formatting artifacts.
- Adaptive computation time: ACT (Graves, 2016) and PonderNet (Banino et al., 2021) adapt depth/steps via training, not token length at inference.
- Confident adaptive language modeling (CALM): Confidence-gated early exit across layers/tokens (Schuster et al., 2022); complementary to our trajectory-level integral signal.
- Uncertainty-aware decoding: Entropy/logit-margin signals used locally; we formalize a global integral criterion with a stability safeguard and optional coupling to self-consistency.

## 3. Method

### 3.1 Preliminaries
- Model: Any autoregressive LM exposing logits and supporting cached decoding (past_key_values/use_cache).
- Prompting: Standard CoT instruction plus an explicit answer cue, e.g., “Therefore, the answer is:”.
- Temperatures: Tr for CoT reasoning; Ta for answer extraction (often Ta=0).

### 3.2 Entropy Signal
At step t with last-step logits z_t, define probabilities p_t = softmax(z_t / Tr). Let H_t = −Σ_v p_t(v) log p_t(v) and normalized entropy Ĥ_t = H_t / log |V| for comparability across vocabularies. Entropy is computed from pre-sampling logits each step.

### 3.3 Entropy-Integral Halting (EIH)
We maintain a windowed mean m_t over the last W normalized entropies and an integral state U_t:
- m_t = (1/W) Σ_{i=t−W+1..t} Ĥ_i (causal padding for t < W).
- U_t = clip(U_{t−1} + (m_t − h_ref), 0, U_max), with U_0 as the initial budget.
- Halt when (U_t ≤ 0) AND (m_t ≤ τ_low). Also enforce a hard CoT cap L_max.

Interpretation:
- When m_t > h_ref, uncertainty persists and U_t rises, extending the budget.
- When m_t < h_ref, uncertainty collapses and U_t decays toward zero.
- The dual condition prevents stopping on transient dips.

Answer-phase handling:
- Exclude the answer segment from the integral by halting the CoT loop on detection of the answer cue and switching to answer decoding with temperature Ta.
- The controller state is not used in the answer phase.

Recommended defaults (from GSM8K/BBH 200-example dev sweeps): W=5, h_ref=0.2, τ_low=0.1, U_0=10, U_max=20, L_max=128. A small grid (W ∈ {3,5,9}, h_ref ∈ {0.15,0.2,0.25}, τ_low ∈ {0.08,0.1,0.12}, U_0 ∈ {6,10,14}) is sufficient; pick per-task via dev set.

### 3.4 Optional: Adaptive Self-Consistency
Define a difficulty proxy D = (U_end / U_max) + m_end and map D to the number of self-consistency samples k via a linear schedule [k_min, k_max]. Ablate D’s components to isolate benefit.

### 3.5 Pseudocode (cached decoding)
```
def eih_cot_decode(model, tokenizer, prompt,
                   Tr=0.7, Ta=0.0,
                   W=5, h_ref=0.2, tau_low=0.1,
                   U0=10.0, U_max=20.0, L_max=128,
                   top_p=None, top_k=None, eos_token_id=None):
    # Precompute answer cue token IDs
    answer_cue = "Therefore, the answer is:"
    answer_cue_ids = tokenizer.encode(answer_cue, add_special_tokens=False)

    # Encode prompt
    ids = tokenizer(prompt, return_tensors="pt").to(model.device).input_ids
    use_cache = True
    past = None

    # Initialize controller
    from collections import deque
    window = deque(maxlen=W)
    U = U0
    cot_ids = []

    # Utility: check if the end of cot_ids matches the cue
    def ends_with_cue(seq, cue):
        if len(seq) < len(cue): return False
        return seq[-len(cue):] == cue

    # CoT loop
    for _ in range(L_max):
        with torch.no_grad():
            out = model(input_ids=ids if past is None else None,
                        past_key_values=past, use_cache=use_cache)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        # Pre-sampling entropy at reasoning temperature
        probs = torch.softmax(logits / Tr, dim=-1)
        H = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # shape [1]
        H_norm = (H / math.log(probs.size(-1))).item()
        window.append(H_norm)
        m = sum(window) / len(window)

        # Integral update
        U = min(U_max, max(0.0, U + (m - h_ref)))

        # Halting condition
        if U <= 0.0 and m <= tau_low:
            break

        # Sample next token (greedy if Tr==0)
        if Tr == 0.0:
            next_id = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(
                logits=(torch.log(probs + 1e-12)))
            if top_p is not None or top_k is not None:
                # optional nucleus/top-k filter on logits before sampling
                filt_logits = top_k_top_p_filtering(logits / Tr, top_k=top_k, top_p=top_p)
                dist = torch.distributions.Categorical(logits=filt_logits)
            next_id = dist.sample()

        # Append and continue
        ids = next_id.view(1, 1)
        cot_ids.append(next_id.item())

        # Stop CoT on answer cue
        if ends_with_cue(cot_ids, answer_cue_ids):
            break

        # Optional EOS break
        if eos_token_id is not None and cot_ids[-1] == eos_token_id:
            break

    # Answer phase (excluded from integral)
    answer_ids = []
    past_answer = past
    # Switch to answer temperature Ta (often greedy)
    for _ in range(256 - len(cot_ids)):
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=past_answer, use_cache=True)
            logits = out.logits[:, -1, :]
            past_answer = out.past_key_values
        if Ta == 0.0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            probs_a = torch.softmax(logits / Ta, dim=-1)
            next_id = torch.distributions.Categorical(probs_a).sample()
        ids = next_id.view(1, 1)
        answer_ids.append(next_id.item())
        if eos_token_id is not None and answer_ids[-1] == eos_token_id:
            break

    return tokenizer.decode(cot_ids), tokenizer.decode(answer_ids)
```
Complexity is O(L) with cached decoding; overhead is one entropy computation and a windowed mean per step.

## 4. Evaluation Protocol (Preregistered)

### 4.1 Models
- Mistral-7B-Instruct-v0.2, Llama-3-8B-Instruct.
- Optional: Qwen2-7B-Instruct, Phi-3-medium, for robustness.

### 4.2 Tasks
- GSM8K, StrategyQA, BIG-Bench Hard subsets (e.g., Date Understanding, Tracking Shuffled Objects).
- Stress tests: verbose prompt variants, multilingual paraphrases (EN/ES), and distractor-laden prompts.

### 4.3 Baselines
- No-CoT direct answer.
- Fixed-L CoT (L ∈ {32, 64, 128}).
- Stop-phrase heuristic (answer cue).
- Local confidence/entropy thresholds; patience decoding (K consecutive low-entropy steps).
- CALM-style budget proxy adapted to token confidence.
- Oracle: Fixed-L with small-k self-consistency to characterize ceilings.

### 4.4 Metrics and Reporting
- Primary: task accuracy (EM or task-defined) and average CoT tokens.
- Secondary: latency, tokens/s, GPU utilization; estimated energy via CodeCarbon or NVML sampling (5–10 Hz), reported as Joules/token.
- Curves: accuracy vs average CoT tokens; area under Pareto (AUP).
- Robustness: prompt variants, temperature changes, instruction formats.
- Statistics: 95% CIs via bootstrap; paired tests with Holm–Bonferroni correction across baselines.
- Tuning guardrails: tune on 200-example dev splits; report on held-out test only.

### 4.5 Ablations
- Remove integral (instantaneous threshold).
- Window size W ∈ {1, 3, 5, 9}.
- Temperature sensitivity Tr ∈ {0.2, 0.7, 1.0}.
- Heatmaps over h_ref × τ_low; transfer across tasks.
- Drop stability condition (U_t ≤ 0 only).
- Adaptive self-consistency on/off; D variants (U_end-only, m_end-only).
- Answer-cue sensitivity (position, phrasing).

### 4.6 Reproducibility Artifacts
- MIT-licensed code with configs, seeds, environment.yml/requirements.txt.
- Exact prompts, answer extraction rules, dataset versions/splits.
- Inference scripts supporting CPU verification on 10–100 samples.
- Commit hash and experiment manifest (JSON) for each run.
Repository (to be released upon camera-ready): https://github.com/<org>/entropy-integral-halting

## 5. Analysis Plan (Confirmatory)
- Hypothesis H1 (Pareto): EIH achieves ≥5% higher AUP than patience decoding at matched accuracy on GSM8K and at least one BBH subset (paired bootstrap, Holm–Bonferroni).
- Hypothesis H2 (Overhead): Controller overhead <2% of decode time with use_cache (timed microbenchmarks).
- Hypothesis H3 (Stability): When EIH halts, the final answer remains unchanged after appending 5 extra reasoning tokens in ≥90% of easy cases (dev-labeled by short CoT).
- Failure analysis: verbosity-induced entropy inflation; distribution shifts; truncation of pedagogical but unnecessary steps.
- Safety checks: evaluate refusal prompts; verify no elevated truncation of safety rationales vs patience baseline at matched accuracy.

## 6. Discussion
Why an entropy integral? Local signals spike on deterministic phrases and formatting; a trajectory-level integral captures uncertainty resolution (“uncertainty work”) and yields:
- Stability against transient dips.
- Difficulty-adaptive budgets without training.
- Composability with architectural early-exit methods.

Limitations:
- Entropy conflates epistemic and stylistic uncertainty; windowing and dual-condition halting mitigate but do not eliminate this.
- Light per-task tuning is needed; we report cross-task transfer and provide default grids.
- Requires logits access; true black-box APIs may not support EIH.

## 7. Broader Impact
Reducing unnecessary CoT tokens lowers cost and energy. Miscalibrated halting could shorten safety rationales; we include diagnostics and recommend conservative defaults in high-stakes settings.

## 8. Conclusion
EIH is a simple, training-free controller for adaptive CoT halting, robust to local noise and easy to integrate. With code and a preregistered protocol on small open models, we enable independent validation and expect improved efficiency–quality trade-offs in agentic workflows.

## References
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks.
- Banino, A., et al. (2021). PonderNet: Learning to Ponder.
- Schuster, T., et al. (2022). Confident Adaptive Language Modeling.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. (and related CoT prompting literature)

## Appendix A: Minimal HF Integration (PyTorch)
```
import torch, math
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def entropy_norm_from_logits(logits, T):
    probs = torch.softmax(logits / T, dim=-1)
    H = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [B]
    return (H / math.log(probs.size(-1))).cpu().tolist()

def token_ends_with(seq, suffix):
    if len(seq) < len(suffix): return False
    return seq[-len(suffix):] == suffix

def eih_generate(model, tok, prompt, Tr=0.7, Ta=0.0,
                 W=5, h_ref=0.2, tau_low=0.1,
                 U0=10.0, U_max=20.0, L_max=128,
                 max_new_tokens=256, eos_token_id=None):
    device = model.device
    input_ids = tok(prompt, return_tensors="pt").to(device).input_ids
    answer_cue_ids = tok.encode("Therefore, the answer is:", add_special_tokens=False)

    # Prime model to get cache
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    last_logits = out.logits[:, -1, :]

    window = deque(maxlen=W)
    U = U0
    cot_ids = []
    # CoT loop
    for _ in range(L_max):
        Hn = entropy_norm_from_logits(last_logits, Tr)[0]
        window.append(Hn)
        m = sum(window) / len(window)
        U = min(U_max, max(0.0, U + (m - h_ref)))
        if U <= 0.0 and m <= tau_low:
            break
        # Sample next CoT token
        if Tr == 0.0:
            next_id = torch.argmax(last_logits, dim=-1)  # [1]
        else:
            probs = torch.softmax(last_logits / Tr, dim=-1)
            next_id = torch.distributions.Categorical(probs).sample()
        cot_ids.append(next_id.item())
        # Next step with cache
        out = model(input_ids=next_id.view(1,1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        if token_ends_with(cot_ids, answer_cue_ids):
            break
        if eos_token_id is not None and cot_ids[-1] == eos_token_id:
            break

    # Answer phase (excluded from integral)
    answer_ids = []
    for _ in range(max_new_tokens - len(cot_ids)):
        if Ta == 0.0:
            next_id = torch.argmax(last_logits, dim=-1)
        else:
            probs = torch.softmax(last_logits / Ta, dim=-1)
            next_id = torch.distributions.Categorical(probs).sample()
        answer_ids.append(next_id.item())
        out = model(input_ids=next_id.view(1,1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        if eos_token_id is not None and answer_ids[-1] == eos_token_id:
            break

    return tok.decode(cot_ids), tok.decode(answer_ids)
```

## Appendix B: Preregistration Summary
- Primary outcomes: accuracy and average CoT tokens on held-out sets with fixed prompts and seeds.
- Confirmatory comparisons: EIH vs fixed-length and patience baselines at matched accuracy (token budget scanning); paired bootstrap with Holm–Bonferroni.
- Overhead measurement: wall-clock microbenchmarks with/without controller, use_cache enabled; energy via CodeCarbon/NVML.
- Deviations: Any changes to prompts, temperatures, or models will be documented and re-run across all methods.

## Artifact Checklist
- Source code, configs, and seeds
- Exact prompts and answer-cue rules
- Dataset versions and splits
- Environment spec (env.yml/requirements.txt)
- Commands and manifests for full replication
- License and model card references


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
