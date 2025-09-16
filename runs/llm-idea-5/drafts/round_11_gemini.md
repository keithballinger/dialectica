Minor Revisions

Brief critique
- Specificity prior definition: The text mentions a "specificity prior" and the pseudocode refers to `idf_stats`, but the mechanism is not explicitly defined in this draft. Briefly define how specificity is calculated (e.g., using mean IDF of rollout tokens).
- NLI score derivation: The NLI score `s_pair` is based on `P(entail)` and `P(contradict)`. For full clarity, explicitly state that these are the softmax probabilities from the NLI classifier for the respective labels.
- Pseudocode dependencies: The pseudocode is excellent but could be slightly tightened by making all external dependencies, such as `idf_stats` and the `calibrator` model, explicit arguments in the function signature for maximum clarity and reproducibility.

Revised Draft
# Lookahead-Consistency Decoding: Preventive Decoding via Micro-Rollout Agreement

## Abstract
We propose Lookahead-Consistency Decoding (LCD), a decoding-time algorithm that reduces unforced errors by preferring next tokens whose short-horizon continuations agree semantically. For each deterministic high-probability token, LCD uses a small draft model to generate short “micro-rollouts,” then scores their agreement with a lightweight hierarchy: fast rollout-state similarity (reusing draft hidden states) followed by selective, high-precision NLI or structural checks. The resulting agreement score modulates the target model’s logits, discouraging tokens that lead to divergent or unstable futures. Unlike speculative decoding (speed) or sequence-level reranking (post hoc), LCD adds a token-local, preventive risk signal during decoding. We detail an efficiency-focused design—uncertainty gating, KV reuse, anti-collapse regularization, and explicit score calibration—and an evaluation plan on factuality, structured generation, and tool-use with small open models.

## 1. Introduction
Autoregressive decoding commits to locally likely tokens without assessing short-term semantic stability, causing factual errors, contradictions, and brittle structured outputs.

Prior methods misalign in granularity or cost: self-consistency and verifier reranking are post hoc and expensive; contrastive approaches adjust logits from internal states without probing multi-token consequences; SelfCheckGPT evaluates contradictions after generation; speculative decoding targets speed. We introduce a lightweight, preventive mechanism that evaluates short-horizon risk before committing to a token.

Lookahead-Consistency Decoding (LCD) probes candidate tokens with cheap micro-rollouts and favors tokens whose futures agree semantically, supplying a real-time risk signal that guides decoding.

Contributions:
- Algorithm: Token-level lookahead that modulates logits via agreement among draft-model micro-rollouts.
- Efficiency: Uncertainty gating; hierarchical scoring that reuses draft hidden states, with selective NLI/structural checks; batched rollouts with KV reuse; explicit calibration and anti-collapse diversity checks.
- Evaluation: Compute-normalized tests on factuality, structured outputs, and tool-use with strong baselines and ablations.
- Open validation: End-to-end code with small, open-source models (e.g., Pythia-1.4B, TinyLlama-1.1B, DeBERTa-MNLI).

## 2. Method

### 2.1 Notation
- Target model T: primary generator (e.g., Pythia-1.4B).
- Draft model D: small model for rollouts (e.g., TinyLlama-1.1B).
- Optional embedding encoder E: e.g., all-MiniLM-L6-v2 (used in ablations).
- NLI scorer S: e.g., deberta-v3-small-mnli.
At step t with context C, T emits next-token distribution P_T. We deterministically select m candidate tokens A.

### 2.2 Uncertainty-gated activation
LCD activates only on “risky” steps to bound overhead:
- Entropy gate: H(P_T) ≥ h.
- Margin gate: p1 − p2 ≤ δ.
- Representation-drift gate: Let g(C) be a cached prefix representation computed as the mean-pooled last-layer hidden states of T over all tokens in C. Maintain a moving average ḡ_{t−1}. Define drift = cosine_distance(g(C), ḡ_{t−1}). Trigger if drift ≥ ρ. Update ḡ_t ← (1−γ)·ḡ_{t−1} + γ·g(C) (γ≈0.2).
Defaults target 20–30% activation: h=3.0 bits, δ=0.10, ρ=0.05. All gates measured from T in a single forward pass; g(C) is available from T’s hidden states already computed for next-token logits.

### 2.3 Candidate selection and micro-rollouts
- Candidate set A: Deterministically choose the top m tokens by probability after applying standard top-p or top-k filtering (temperature τ_T). No stochasticity is used in A.
- Rollouts: For each a∈A, sample k continuations of length L with D at temperature τ_D. This is the only source of stochasticity.
- KV reuse and memory: Share the KV cache for C across all m×k rollouts. For each candidate a, append its single-token prefix once (creating a small per-candidate KV block) and extend L tokens for each of the k samples. Implement with batched “multi-prefix” decoding and views/indices to avoid physically duplicating the base KV. We report VRAM overhead from per-candidate KV blocks and enforce batch-size caps accordingly.

### 2.4 Agreement scoring with anti-collapse and context-aware NLI
We compute a per-candidate agreement score s(a)∈(0,1).

Step 1: Fast rollout-state consensus (default) with anti-collapse
- For each rollout r, use D’s last-layer token hidden states over the L generated tokens. Obtain a fixed-length vector via mean-pooling and L2-normalize to e_r. This avoids an extra encoder pass. As an ablation, E can encode r instead.
- Compute pairwise cosine similarities among k rollouts: cos∈[−1,1]; map to [0,1] via cos01=(cos+1)/2. Let s_emb=mean(cos01).
- Specificity prior: To discourage generic continuations, we define `spec` as the mean inverse-document-frequency (IDF) of tokens in a rollout, pre-calculated on a reference corpus and normalized to [0,1].
- Collapse penalty: detect trivial agreement by measuring diversity across rollouts, e.g., distinct-2 ratio or average token-level Jaccard distance d∈[0,1]. Define s_div=clip(d−d_min,0,1)/(1−d_min) with d_min≈0.1.
- Penalize low specificity and diversity: s0_raw = s_emb − λ_spec·(1−spec) − λ_col·(1−s_div).
- Clip and map: s0(a)=clip(s0_raw,0,1). Hyperparameters: λ_spec=0.3, λ_col=0.3.

Step 2: Selective NLI or structural checks (precise)
- Select top u candidates by s0 for refinement (u≤2 by default).
- Text tasks (NLI):
  - Inputs are full pairs with context: (C ⊕ r_i, C ⊕ r_j). Truncate via left-truncation of C to meet S’s max length; never truncate rollouts.
  - For each ordered pair (i≠j), use the softmax probabilities for the 'entailment' (`P_ent`) and 'contradiction' (`P_con`) labels from S. Define s_pair = clip(P_ent − P_con, −1, 1); map to [0,1]: s_pair01=(s_pair+1)/2.
  - s_nli(a)=mean s_pair01 over all k·(k−1) directional pairs.
- Structured tasks (e.g., SQL/code):
  - Validity: fraction parsable by a task-appropriate parser; treat unparsable as 0 similarity.
  - Similarity: 1 − normalized edit or AST distance (in [0,1]) over all unordered pairs; map invalid pairs to 0.
  - s_struct(a)=mean over pairs.
Final agreement:
- Text: s(a)=α·s0(a) + (1−α)·s_nli(a).
- Structured: s(a)=α·s0(a) + (1−α)·s_struct(a).
For non-refined candidates, s(a)=s0(a).

Calibration:
- Learn a 1D monotonic calibrator f (isotonic or Platt) mapping raw s to ŝ∈(0,1) on a dedicated validation split disjoint from test prompts. We use ŝ in logit modulation.

### 2.5 Logit modulation
- For each a∈A, compute bonus b(a)=β·log(ŝ(a)/(1−ŝ(a))), with ŝ clipped to [ε,1−ε], ε=1e−4.
- Adjust logits: z′(a)=z(a)+b(a); tokens not in A retain z.
- Optional hard filter (off by default): drop a with ŝ(a)<τ.
- Sample next token from softmax(z′) at the original temperature.

### 2.6 Compute and memory model (per activated step)
- Draft tokens: ≈ m·k·L tokens through D (batched).
- Similarity stage:
  - Default (reuse D states): no extra forward passes; cost is pooling and pairwise cosines O(k^2) per a.
  - Ablation with E: add m·k encodes.
- NLI pairs: u·k·(k−1) directional forward passes (batched).
- KV/memory: one base KV for C, plus m small per-candidate prefix KVs and m·k rollout KVs of length L. We report VRAM headroom and enforce a dynamic cap on m·k·L to avoid OOM.
- Defaults: m=4, k=2, L=6, u=2, τ_T=0.7, τ_D=0.8, λ_spec=0.3, λ_col=0.3, α=0.5; β tuned per task; gating rate≈25%.
- Target overhead: ≤2.0× median latency vs. standard decoding for T on an A100 (bf16). We report FLOPs, VRAM, and wall-clock latency.

### 2.7 Practical rollout controls for structured tasks
- Early stop criteria: stop rollouts at delimiters (e.g., semicolon, closing bracket), on balanced brackets, or at max L. Use partial parsing to identify stable cut points.
- AST distance: tree edit distance with size-normalized scaling; cache parser states to amortize cost.

### 2.8 Pseudocode (single step)
```python
def lcd_step(C, T, D, S, params, idf_stats=None, calibrator=None, E=None, state=None):
    # state carries gbar (moving-average prefix rep)
    z, hidden = T.forward_with_hidden(C)  # next-token logits and last-layer hiddens
    if state is None: state = {}
    gC = hidden.mean(dim=seq_dim)  # mean-pooled prefix rep
    drift = cosine_distance(gC, state.get("gbar", gC))
    state["gbar"] = (1 - params.gamma) * state.get("gbar", gC) + params.gamma * gC

    if not is_risky_step(z, drift, params):  # entropy, margin, drift
        return sample_from_logits(z, params.temperature), state

    A = top_m_after_filter(z, params.m, params.top_p, params.top_k)
    rollouts, d_hidden = sample_rollouts_with_states(D, C, A, params.k, params.L, batch=True)

    scores0 = {}
    for a in A:
        # Fast similarity (default): reuse D states; optional E for ablation
        if E is None:
            e = [h.mean(dim=seq_dim).normalize() for h in d_hidden[a]]
        else:
            e = encode_batch(E, rollouts[a], normalize=True)
        s_emb = mean_pairwise_cosine01(e)
        spec = mean_specificity(rollouts[a], idf_stats) if idf_stats else 1.0
        diversity = distinct_n_ratio(rollouts[a], n=2)
        s_div = clip((diversity - params.d_min) / (1 - params.d_min), 0.0, 1.0)
        s0 = s_emb - params.lambda_spec * (1 - spec) - params.lambda_col * (1 - s_div)
        scores0[a] = clip(s0, 0.0, 1.0)

    A_sel = top_u_by_score(A, scores0, params.u)
    scores = {}
    for a in A:
        if a in A_sel:
            if params.task_type == "text":
                s_ref = nli_pairwise_score01(S, C, rollouts[a], truncate="left")
            else:
                s_ref = structural_pairwise_score01(rollouts[a])
            s = params.alpha * scores0[a] + (1 - params.alpha) * s_ref
        else:
            s = scores0[a]
        scores[a] = calibrator.predict(s) if calibrator else s  # maps to (0,1)

    z_prime = z.clone()
    for a in A:
        sa = clip(scores[a], params.eps, 1 - params.eps)
        z_prime[a] += params.beta * log(sa / (1 - sa))
        if params.tau is not None and scores[a] < params.tau:
            z_prime[a] = -float("inf")
    return sample_from_logits(z_prime, params.temperature), state
```

## 3. Experimental validation

### 3.1 Hypotheses and falsification
- H1 (Quality): LCD improves primary metrics by ≥3% absolute over tuned baselines under matched or lower end-to-end compute.
- H2 (Efficiency): With gating and hierarchical scoring, median latency overhead ≤2.0× vs. standard decoding.
- H3 (Mechanism): Agreement ŝ(a) of chosen tokens positively correlates with final output correctness; gains concentrate on gated steps.

Falsification: Reject claims if (i) no statistically significant improvement vs. compute-matched baselines, (ii) overhead >2.5× without compensating quality gains, or (iii) no positive correlation between ŝ and correctness.

### 3.2 Tasks and metrics
- TruthfulQA (gen): %True; %Truthful+Informative.
- BioASQ factoid QA: EM, F1.
- WikiSQL: EM; execution accuracy.
- Function calling (ToolBench-style subset): exact tool and argument match.

### 3.3 Models
- T: Pythia-1.4B; Mistral-7B-Instruct (for scale transfer).
- D: TinyLlama-1.1B; Pythia-410M; ablations with self-draft (T in 8-bit) and early-exit T for short rollouts.
- S: deberta-v3-small/base-mnli.
- E (ablation): all-MiniLM-L6-v2.

### 3.4 Baselines (compute-matched)
- Standard decoding: greedy; tuned top-p; beam search (width=4).
- Length-L token lookahead (T-only): rerank next-token candidates by expected logprob over L-step continuations from T (beam/Monte Carlo), matched compute.
- Verifier-guided next-token reranking: rank A using NLI/structural verifier on (C ⊕ single-step draft), no micro-rollouts.
- Self-consistency: best-of-k=5 majority vote.
- Verifier-reranked beam: NLI/structural verifier on beams.
- Best-of-N reranking: verifier on N full outputs, budget-matched.
- SelfCheckGPT-style post hoc scoring.
- Contrastive decoding (e.g., DoLa/CD).
- Speculative decoding (for composability and speed).

### 3.5 Protocol
- Compute normalization: match total FLOPs, VRAM headroom, and wall-clock per example, counting T tokens, D rollout tokens, NLI forwards, and (if used) E encodes.
- Hyperparameter tuning: β, gating thresholds (h, δ, ρ), α, λ_spec, λ_col, τ_D tuned on dedicated validation sets per task; calibrator trained on a disjoint split.
- Statistical reporting: mean±95% CI via stratified bootstrap (≥1k replicates) with Benjamini–Hochberg FDR control across tasks.
- Controls: length-controlled decoding; ≥5 seeds; temperature sweeps for all methods; report perplexity deltas on held-out corpora to detect distribution shifts.
- Ablations: remove gating; similarity-only; NLI-only; no specificity prior; no collapse penalty; vary m,k,L; weaker/stronger D; self-draft vs. external D; with vs. without calibrator; swap E on/off.
- Mechanism checks: correlation between ŝ and correctness; gains concentrated on gated steps; synthetic structured tasks (e.g., bracket matching) to probe L-range effects.

### 3.6 Efficiency engineering details
- Batching: single batched forward for all m×k rollouts with shared context KV; per-candidate prefix KV materialized once.
- Mixed precision: bf16; int8 quantization for D and S in ablations.
- Throughput: tokens/sec and latency distributions (median, p90); VRAM profiling.
- Cache locality: contiguous candidate grouping to maximize KV reuse; dynamic backoff of m·k·L if nearing VRAM limits.

## 4. Relation to prior work
LCD differs from:
- Self-consistency/best-of-N/verifier reranking: post hoc and sequence-level vs. LCD’s token-local, preventive signal.
- Speculative decoding: targets speed; LCD targets quality and is composable with speculative acceptance.
- SelfCheckGPT: post hoc hallucination detection vs. LCD’s preventive agreement prior.
- Contrastive methods (e.g., DoLa): internal-state adjustments vs. LCD’s external multi-token probing.
- Chain/Tree-of-Thought: sample multi-path reasoning but do not feed a calibrated, per-token risk signal into the decoding distribution.
- Token lookahead (new baseline above): uses T-only logprob projections; LCD adds semantic agreement via a separate draft and verifiers.

## 5. Limitations
- Overhead and VRAM: Practicality depends on gating, batching, and m·k·L; high-entropy prompts can trigger frequent activation. We cap batch sizes and report VRAM headroom.
- Scorer fragility: NLI on short texts can be unreliable; mitigated by similarity prefilter, contradiction-aware aggregation, specificity prior, diversity checks, and structural parsers for code/SQL.
- Degeneration risk: Penalizing divergence may suppress minority-correct or creative answers. Restrict β in creative domains or disable LCD.
- Myopia: Short rollouts (L) can miss long-range issues; we study L-quality trade-offs.
- Draft-model bias: Mismatch between D and T can misguide; we evaluate self-draft and quantized T rollouts.
- Distribution shift: Logit modulation can alter token distributions; we report perplexity change to monitor harm.

## 6. Conclusion
LCD injects a token-local stability prior by scoring agreement among cheap micro-rollouts and modulating logits accordingly. This preventive signal aims to steer models away from unstable trajectories that produce factual and structural errors. Our falsification-oriented experiments will test whether this lookahead signal yields reliable quality gains at acceptable cost. If validated, LCD is a composable building block for more reliable inference, especially in agentic workflows where single-token errors are costly.

## 7. Reproducibility
We will release:
- Code with deterministic seeds, exact HF model IDs, and scripts for KV batching, multi-prefix decoding, and gating.
- IDF corpus and statistics; calibrator checkpoints; all hyperparameters.
- Evaluation prompts, compute-accounting scripts, and analysis notebooks in a container with VRAM/latency profilers.
