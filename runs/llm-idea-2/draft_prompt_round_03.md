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
        Minor Revisions.

The proposal is strong, with a novel, well-motivated method and a rigorous falsification plan. The core idea of using MC-dropout to gate decoding modes is a significant contribution to inference-time scheduling. The writing is clear and direct. Revisions are minor and focus on improving precision for reproducibility.

1.  **Methodological Precision:** The use of `model.train()` to enable dropout is imprecise. While `torch.no_grad()` prevents weight updates, `model.train()` can affect the behavior of other layers (e.g., LayerNorm). The text and pseudocode should specify that *only* dropout modules are switched to training mode for MC passes.
2.  **Gating Justification:** The gating rule defaults to the `pbase` argmax even after a costly MC pass reveals low mutual information. This is a reasonable choice for efficiency and determinism, but the rationale should be stated explicitly.
3.  **Clarity:** The abstract and introduction can be slightly sharpened for directness.

The proposed research is impactful and the experimental design is sound. The suggested revisions will strengthen the paper's technical correctness and reproducibility.

### Revised Draft

**Title:** Uncertainty-Gated Decoding: Per-Token Mode Switching via MC-Dropout for Large Language Models

**Abstract**
Large language model (LLM) inference must balance exploration with reliability. Greedy decoding is fast but brittle, while stochastic sampling enhances creativity at the cost of factuality. We propose Uncertainty-Gated Decoding (UGD), a per-token scheduler that switches between greedy and sampling modes based on epistemic uncertainty estimated with Monte Carlo (MC) dropout. At each step, UGD computes a cheap confidence proxy (e.g., token probability margin); only when the model is not confident does it invoke 2–3 additional dropout-enabled forward passes to measure uncertainty. If epistemic uncertainty is high, UGD samples from the MC-averaged predictive distribution; otherwise, it decodes greedily. This provides a tunable speed–quality trade-off without model retraining. We outline a falsification plan across code generation (HumanEval, MBPP), factual QA (TriviaQA, TruthfulQA), and arithmetic reasoning (GSM8K), comparing UGD to standard decoding methods at matched latency. UGD is simple to implement, adds negligible overhead on confident tokens by reusing the KV cache, and surgically targets exploration to high-uncertainty steps.

**Introduction**
Decoding strategies critically influence LLM behavior. Deterministic methods like greedy search maximize single-step likelihood but can propagate early errors. Stochastic methods (e.g., temperature or nucleus sampling) increase diversity and robustness but can introduce hallucinations and degrade repeatability. Current adaptive strategies are often heuristic (e.g., entropy thresholds) and agnostic to epistemic uncertainty—the model's own lack of knowledge, as distinct from ambiguity inherent in the data.

MC dropout approximates Bayesian inference by treating dropout at test time as sampling from a posterior distribution over model weights (Gal & Ghahramani, 2016). The variance across these stochastic forward passes is a proxy for epistemic uncertainty. We leverage this to gate decoding at the token level: first, use a cheap proxy to detect potential uncertainty; second, if needed, run MC passes to confirm it; finally, decide whether to exploit (greedy) or explore (sample). To our knowledge, UGD is the first inference-time, per-token mode switcher that uses MC-dropout disagreement to gate decoding without requiring any model retraining.

**Method**

**Problem Setting**
Given a model pθ(y | x) and a partial output y1:t−1, we choose the next token yt by either:
-   **Greedy:** yt = argmaxk pθ(k | x, y1:t−1)
-   **Sampling:** yt ~ q(· | x, y1:t−1), where q is a distribution derived from the model, possibly modified by temperature or nucleus filtering.

**Uncertainty Signals**
We use a two-stage process to manage computational cost:
-   **Aleatoric Proxies (Cheap):** Computed from a single forward pass with dropout disabled. We use the entropy of the predictive distribution, Hbase = H[pθ(· | ·)], and the margin between the top two token probabilities, m = p(1) − p(2).
-   **Epistemic Estimate (Costly):** Computed using MC-dropout with K forward passes. For each pass i ∈ {1..K}, we sample a distribution p(i) with dropout enabled. We define the predictive mean p̄ = (1/K)∑i p(i) and compute the mutual information (MI) between the model's posterior and predictive distributions: MI = H[p̄] − (1/K)∑i H[p(i)]. This term isolates the epistemic component of the total uncertainty.

**Gating Rule**
1.  **Base Pass (Dropout Off):**
    -   Compute logits zbase and probability distribution pbase.
    -   If m ≥ mthresh and Hbase ≤ Hthresh, the model is confident. Output the greedy token argmax(pbase) and terminate the step.

2.  **MC Pass (Invoked if uncertain):**
    -   Run K dropout-enabled forward passes for the current token, reusing the KV cache from the base pass.
    -   Compute the mutual information MI.
    -   If MI ≤ MIthresh, epistemic uncertainty is low. Decode greedily from `pbase` to maintain determinism on confident predictions.
    -   Else, epistemic uncertainty is high. Decode by sampling from the MC-averaged distribution `p̄`, potentially modulated by an uncertainty-aware temperature T(MI) and/or nucleus p(MI).

**Adaptive Sampling**
When sampling is triggered, its intensity is scaled by the magnitude of the uncertainty. We use a monotone mapping T(MI) = Tmin + (Tmax − Tmin) · clip(MI / MI95, 0, 1), where MI95 is the 95th percentile MI value observed on a calibration set. A similar schedule can be applied to the nucleus probability p. Sampling from the MC-averaged distribution p̄, rather than pbase, integrates information from multiple dropout masks and is a more robust estimate of the predictive distribution.

**Calibration and Latency**
The thresholds (mthresh, Hthresh, MIthresh) and number of passes K are hyperparameters.
-   (mthresh, Hthresh) are set on a development set to achieve a target MC invocation rate s (e.g., s ∈ [0.1, 0.3]), balancing performance and latency.
-   K is typically small (2–4) to trade off estimate stability with cost.
-   The expected number of forward pass computations per token is 1 + s · K. Since MC passes only recompute the current step while reusing the KV cache, the overhead is minimal when s is small. UGD can be tuned to match or reduce the latency of standard sampling methods.

**Pseudocode**
```python
# Helper to toggle dropout layers' mode
def set_dropout_mode(model, enabled: bool):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train(enabled)

# Single decoding step
def UGD_step(model, x, y1:t-1, cache):
    # 1. Base pass (dropout off)
    set_dropout_mode(model, False)
    z_base, cache = model.forward_step(x, y1:t-1, cache)
    p_base = softmax(z_base)
    m, H_base = top2_margin(p_base), entropy(p_base)

    if m >= m_thresh and H_base <= H_thresh:
        return argmax(p_base), cache

    # 2. MC pass (invoked if uncertain)
    set_dropout_mode(model, True)
    P = []
    for _ in range(K):
        # KV cache is reused for efficiency
        z_i, _ = model.forward_step(x, y1:t-1, cache)
        P.append(softmax(z_i))
    set_dropout_mode(model, False) # Revert state

    p_bar = mean(P)
    MI = entropy(p_bar) - mean([entropy(p) for p in P])

    if MI <= MI_thresh:
        return argmax(p_base), cache # Low epistemic uncertainty
    else:
        T = schedule_T(MI)
        p_nuc = nucleus_filter(p_bar, p=schedule_p(MI))
        return sample(p_nuc, T=T), cache
```

**Experiments (Falsification Plan)**

**Goals:**
1.  Test if uncertainty-gated switching improves task accuracy over baselines at matched or lower latency.
2.  Verify that the MC-based epistemic uncertainty signal provides benefits beyond simpler proxies like entropy or margin.

**Models:** TinyLlama-1.1B, Phi-2 (2.7B), Llama-2-7B, Mistral-7B.

**Datasets and Metrics:**
-   **Code Generation:** HumanEval (pass@1), MBPP (exact match).
-   **Factual QA:** TriviaQA (EM/F1), TruthfulQA (truthfulness).
-   **Reasoning:** GSM8K (accuracy).
-   **Efficiency:** Wall-clock latency per sample, tokens/sec, and MC invocation rate (s).

**Baselines:**
-   Greedy decoding.
-   Temperature sampling (T tuned per task).
-   Nucleus sampling (p tuned).
-   Entropy-only gating: switch to sampling when Hbase > Hthresh (no MC pass).
-   Ablations: UGD with varying K; UGD sampling from pbase vs p̄; fixed T vs adaptive T(MI).

**Protocol:**
-   **Matched Latency:** For each baseline, we tune its parameters (T, p) to a target latency. We then tune UGD's parameters (s, K, MIthresh) to match that latency within ±5% and compare task metrics.
-   **Calibration:** All thresholds are calibrated on standard development splits.
-   **Reproducibility:** Experiments will be run with 5 random seeds for sampling methods.

**Hypotheses:**
-   H1: UGD outperforms greedy and standard sampling methods on HumanEval/MBPP at matched latency by selectively exploring during syntactically or semantically ambiguous code generation steps.
-   H2: UGD improves truthfulness on TruthfulQA over sampling baselines by avoiding unnecessary exploration on confident, factual statements.
-   H3: UGD matches or exceeds GSM8K accuracy of baselines with similar latency, by applying exploration at critical reasoning steps.
-   H4: The full UGD model significantly outperforms entropy-only gating at the same MC invocation rate s, demonstrating the value of the epistemic signal.

**Discussion**

**Why It Works:** UGD decouples two forms of uncertainty. Aleatoric uncertainty (high entropy) reflects local ambiguity where several tokens are plausible. Epistemic uncertainty (high MI) reflects a model's lack of knowledge. Stochastic sampling is most beneficial in the latter case, where exploring alternatives can overcome a model's flawed primary guess. UGD targets this exploration, while preserving determinism elsewhere.

**Impact:** In agentic workflows, UGD can enable more reliable tool use by restricting exploration to decision points (e.g., API selection) while executing known procedures greedily. High MI could also serve as a signal to seek external verification.

**Systems Considerations:** The computational overhead is bounded and predictable (≈ 1 + s · K). For s=0.15 and K=3, the cost multiplier is ~1.45x, often comparable to the overhead of standard sampling implementations, but with more targeted benefits. The method is compatible with speculative decoding and other modern inference optimizations.

**Relation to Prior Work:** While methods like nucleus sampling adapt to the local probability distribution, they do not explicitly model epistemic uncertainty. Entropy-gating is a heuristic special case of UGD's first stage. Our work is the first, to our knowledge, to operationalize MC-dropout as a practical, per-token, retraining-free gating mechanism for LLM decoding.

**Limitations:**
-   The method requires models with active dropout layers. Architectures without dropout would require alternatives like stochastic LayerNorm.
-   Hyperparameter calibration may be task- and model-dependent.
-   The MI estimate from a small K is noisy, which may limit gains under very tight latency constraints.
-   For safety-critical applications requiring full determinism, any sampling is prohibited.

**Conclusion**
We propose Uncertainty-Gated Decoding, a scheduler that dynamically switches between greedy and stochastic decoding based on MC-dropout estimates of epistemic uncertainty. By using a cheap proxy to trigger a more expensive uncertainty check, UGD focuses exploration where the model is most likely to be wrong. This retraining-free method offers a principled way to improve the trade-off between quality and latency in LLM inference. Our proposed experiments are designed to rigorously test this hypothesis across a diverse set of tasks and models.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
