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
- **Candidate Selection Ambiguity:** The process for selecting the candidate set `A` is ambiguous. The text mentions "top-p/top-k sampling," but logit modification methods typically select the deterministic top-`m` tokens. Clarify whether selection is stochastic or deterministic.
- **NLI Cost Inconsistency:** The description of pairwise NLI checks (Sec 2.4) implies `k(k-1)` inferences per candidate (checking `r_i→r_j` and `r_j→r_i`), but the compute model (Sec 2.6) estimates `u·k·(k−1)/2` passes. These figures must be reconciled. The cost is likely `u·k·(k-1)` for standard unidirectional NLI models.
- **Hyperparameter Tuning Protocol:** The logit modulation strength `β` is a critical hyperparameter noted as "tuned per task." The experimental plan should explicitly state that `β` and other key parameters (e.g., gating thresholds) will be tuned on a dedicated validation set for each task to ensure fair evaluation.
- **Title Conciseness:** The subtitle is descriptive but long. A more concise version would improve impact.

Revised Draft
# Lookahead-Consistency Decoding: Preventive Decoding via Micro-Rollout Agreement

## Abstract
We propose Lookahead-Consistency Decoding (LCD), a decoding-time algorithm that reduces unforced errors by preferring next tokens whose short-term futures exhibit high semantic agreement. For each high-potential token, LCD uses a small draft model to generate brief “micro-rollouts,” then scores their agreement with a lightweight hierarchy: fast embedding consensus followed by selective, high-precision NLI or structural checks. The resulting agreement score modulates the target model’s logits, discouraging tokens that lead to divergent or unstable futures. Unlike speculative decoding (speed) or verifier reranking (sequence-level, post hoc), LCD adds a token-local, preventive risk signal during decoding. We detail an efficiency-focused design with uncertainty gating, batched KV reuse, and explicit score calibration. We outline a falsification-oriented evaluation on factuality, structured generation, and tool-use with small open models.

## 1. Introduction
Standard autoregressive decoding (e.g., top-p sampling) commits to locally probable tokens without assessing downstream semantic stability, yielding factual errors, contradictions, or brittle structured outputs.

Prior reliability methods operate at mismatched granularity. Self-consistency and verifier reranking are post hoc and expensive (multiple full generations). Contrastive methods (e.g., DoLa) adjust logits from internal states without probing multi-token consequences. Consistency methods like SelfCheckGPT evaluate contradictions after generation. Speculative decoding improves speed, not quality. What is missing is a lightweight, preventive mechanism that assesses a token’s short-horizon semantic risk before committing.

Lookahead-Consistency Decoding (LCD) estimates local semantic stability by probing candidate tokens with cheap micro-rollouts. It favors tokens whose futures agree semantically, supplying a real-time risk signal that guides decoding.

Contributions:
- Algorithm: Token-level lookahead that modulates logits by agreement among draft-model micro-rollouts.
- Efficiency: Uncertainty gating, hierarchical scoring (embedding prefilter → selective NLI/structural), batched rollouts with KV reuse, explicit score calibration.
- Evaluation: Compute-normalized tests on factuality and structured outputs with strong baselines and ablations.
- Open validation: End-to-end code with small, open-source models (e.g., Pythia-1.4B, TinyLlama-1.1B, DeBERTa-MNLI).

## 2. Method

### 2.1 Notation
- Target model T: primary generator (e.g., Pythia-1.4B).
- Draft model D: small model for rollouts (e.g., TinyLlama-1.1B).
- Embedding encoder E: e.g., all-MiniLM-L6-v2.
- NLI scorer S: e.g., deberta-v3-small-mnli.
At step t with context C, T emits next-token distribution P_T. We select `m` candidate tokens A.

### 2.2 Uncertainty-gated activation
LCD activates only on “risky” steps to bound overhead:
- Entropy gate: H(P_T) ≥ h.
- Margin gate: p1 − p2 ≤ δ.
- Representation drift: cosine distance between last-layer [EOS] (or pooled) hidden states at t and t−1 ≥ ρ.
Default: use last-layer mean-pooled hidden, h=3.0 bits, δ=0.10, ρ=0.05, targeting 20–30% activation. Gates are measured from T; computing drift adds a single cosine per step.

### 2.3 Candidate selection and micro-rollouts
- Candidate set A: Select top `m` candidates from the distribution P_T after applying standard top-p or top-k filtering (temperature τ_T).
- Rollouts: for each a∈A, sample k continuations of length L with D at temperature τ_D to capture plausible diversity.
- KV reuse: share the KV cache for C across all m×k rollouts; each candidate adds a single-token prefix KV, then extends L tokens. All rollouts are batched.

### 2.4 Agreement scoring with explicit normalization
We output a per-candidate agreement score s(a)∈(0,1).

Step 1: Embedding consensus (fast)
- Encode each rollout r with E → vector e_r (L2-normalized).
- Pairwise cosine similarities among k rollouts: cos∈[−1,1]. Convert to [0,1]: cos01=(cos+1)/2. Define s_emb as the mean of cos01 over all pairs.
- Specificity prior: discourage generic rollouts.
  - Let IDF(w) be from a fixed corpus (e.g., Wikipedia dump used to train E). For rollout r with tokens w_r: spec(r)=clip(Σ_w∈r IDF(w)/|r|, 0, s_max)/s_max to map to [0,1], where s_max is the 95th percentile of this statistic on a held-out corpus.
  - Candidate specificity: spec(a)=mean_r spec(r).
- Preliminary score: s0(a)=clip(s_emb(a) − λ·(1−spec(a)), 0, 1), λ∈[0,1].

Step 2: Selective NLI or structural checks (precise)
- Select top u candidates by s0 for refinement (u≤2 by default).
- For text tasks: pairwise NLI over rollout pairs (both directions). For pair (r_i,r_j), define:
  - p_ent = 0.5·(P(r_i→r_j, entail)+P(r_j→r_i, entail))
  - p_con = 0.5·(P(r_i→r_j, contra)+P(r_j→r_i, contra))
  - s_pair = clip(p_ent − p_con, −1, 1); map to [0,1]: s_pair01=(s_pair+1)/2.
- s_nli(a) = mean s_pair01 across pairs.
- For structured tasks (SQL/code): replace NLI with structural checks:
  - Validity: fraction parsable.
  - Similarity: 1 − normalized edit/AST distance (in [0,1]).
  - s_struct(a): mean over pairs; already in [0,1].
Final agreement:
- Text: s(a)=α·s0(a) + (1−α)·s_nli(a).
- Structured: s(a)=α·s0(a) + (1−α)·s_struct(a).
α∈[0,1]. For non-refined candidates, s(a)=s0(a).

Calibration:
- Optionally learn a 1D monotonic calibrator f (isotonic or Platt scaling) mapping raw s to ŝ in (0,1) on a small held-out set to improve stability across domains. We use ŝ in logit modulation.

### 2.5 Logit modulation
- For each a∈A, compute bonus b(a)=β·logit(ŝ(a))=β·log(ŝ(a)/(1−ŝ(a))). Since ŝ∈(0,1), b can be positive or negative; to ensure numeric stability, clip ŝ∈[ε,1−ε], ε=1e−4.
- Adjust logits: z′(a)=z(a)+b(a) for a∈A. Tokens not in A retain z.
- Optional hard filter (off by default): drop a with ŝ(a)<τ to enforce minimum agreement.
- Sample next token from softmax(z′) at original temperature.

Notes:
- This implements both reward (ŝ>0.5) and penalty (ŝ<0.5), aligning with the preventive goal.
- Because only A is modulated, consider m expansion in high-entropy steps to avoid missing viable tokens.

### 2.6 Compute model and defaults
- Cost per activated step: draft tokens ≈ m·k·L; embeddings: m·k encodes; NLI pairs: `u·k·(k-1)` forward passes (batched), as each pair is checked bidirectionally.
- Defaults: m=4, k=2, L=6, u=2, τ_T=0.7, τ_D=0.8, λ=0.3, α=0.5, β tuned per task, gating rate≈25%.
- Target overhead: ≤2.0× median latency vs. standard decoding for T on an A100 with bf16; we report FLOPs and wall-clock latency.

### 2.7 Pseudocode (single step)
```python
def lcd_step(C, T, D, E, S, params):
    if not is_risky_step(T, C, params):  # entropy, margin, drift
        return sample_from_T(T, C, params)

    A, z = get_candidate_set(T, C, params)  # candidates and logits
    rollouts = sample_rollouts(D, C, A, params.k, params.L, batch=True)

    scores0 = {}
    for a in A:
        s_emb = mean_pairwise_cosine01(encode_batch(E, rollouts[a]))
        spec = mean_specificity(rollouts[a], params.idf_stats, params.s_max)
        s0 = clip(s_emb - params.lambda_ * (1 - spec), 0.0, 1.0)
        scores0[a] = s0

    A_sel = top_u_by_score(A, scores0, params.u)
    scores = {}
    for a in A:
        if a in A_sel:
            if params.task_type == "text":
                s_ref = nli_pairwise_score01(S, rollouts[a], batch=True)
            else:
                s_ref = structural_pairwise_score01(rollouts[a])
            s = params.alpha * scores0[a] + (1 - params.alpha) * s_ref
        else:
            s = scores0[a]
        scores[a] = calibrate_if_enabled(s, params.calibrator)  # maps to (0,1)

    z_prime = z.clone()
    for a in A:
        sa = clip(scores[a], params.eps, 1 - params.eps)  # numeric stability
        bonus = params.beta * log(sa / (1 - sa))  # logit
        z_prime[a] += bonus
    # Optional hard filter (off by default)
    if params.tau is not None:
        for a in A:
            if scores[a] < params.tau:
                z_prime[a] = -float("inf")
    return sample_from_logits(z_prime, params.temperature)
```

## 3. Experimental validation

### 3.1 Hypotheses and falsification
- H1 (Quality): LCD improves primary metrics by ≥3% absolute over tuned baselines under matched or lower end-to-end compute.
- H2 (Efficiency): With gating and hierarchical scoring, median latency overhead ≤2.0× vs. standard decoding.
- H3 (Mechanism): Agreement ŝ(a) of chosen tokens positively correlates with final output correctness; gains concentrate on gated steps.

Falsification: Reject claims if (i) no statistically significant improvement vs. compute-matched baselines, (ii) overhead >2.5× without compensating quality gains, or (iii) no positive correlation between ŝ and correctness.

### 3.2 Tasks and metrics
- TruthfulQA (gen): %True, %Truthful+Informative.
- BioASQ factoid QA: EM, F1.
- WikiSQL: EM, execution accuracy.
- Function calling (ToolBench-style subset): exact tool and argument match.

### 3.3 Models
- T: Pythia-1.4B; Mistral-7B-Instruct (for scale transfer).
- D: TinyLlama-1.1B; Pythia-410M; ablation with T early-exit or 8-bit T short-rollouts.
- E: all-MiniLM-L6-v2.
- S: deberta-v3-small-mnli; deberta-v3-base-mnli.

### 3.4 Baselines (compute-matched)
- Standard decoding: greedy; tuned top-p; beam search (width=4).
- Self-consistency: best-of-k=5 majority vote.
- Verifier-reranked beam: NLI/structural verifier on beams.
- Best-of-N reranking: verifier on N full outputs, budget-matched to LCD.
- SelfCheckGPT-style post hoc scoring.
- Contrastive decoding (e.g., DoLa/CD).
- Speculative decoding (for composability and speed).

### 3.5 Protocol
- Compute normalization: match total FLOPs/wall-clock per example, counting T tokens, D rollout tokens, E encodes, and S inferences.
- Hyperparameter tuning: Key hyperparameters (β, gating thresholds) will be tuned on a dedicated validation set for each task.
- Significance: report mean±95% CI via stratified bootstrap (≥1k replicates).
- Controls: length-controlled decoding; ≥5 seeds; temperature sweep for all methods.
- Ablations: remove gating; embedding-only; NLI-only; no specificity prior; vary m,k,L; weaker/stronger D; self-draft vs. external D; with vs. without calibrator f.
- Mechanism checks: correlation between ŝ and correctness; performance gains concentrated on gated steps; synthetic structured tasks (e.g., bracket matching) to test long-range myopia vs. L.

### 3.6 Efficiency engineering details
- Batching: single batched forward for all m×k rollouts with shared context KV; per-candidate prefix KV duplicated once.
- Mixed precision: bf16; int8 quantization for D and S in ablations.
- Throughput: report tokens/sec and latency distributions (median, p90).
- Cache locality: ensure contiguous candidate grouping to maximize KV reuse.

## 4. Relation to prior work
LCD differs from:
- Self-consistency/best-of-N/verifier reranking: post hoc and sequence-level vs. LCD’s token-local, preventive signal.
- Speculative decoding: focuses on speed; LCD targets quality, but is composable with speculative acceptance.
- SelfCheckGPT: post hoc hallucination detection vs. LCD’s preventive agreement prior.
- Contrastive methods (e.g., DoLa): internal-state adjustments vs. LCD’s external multi-token probing.
- Chain/Tree-of-Thought: sample multi-path reasoning but do not feed a calibrated, per-token risk signal into the decoding distribution.

## 5. Limitations
- Overhead: Practicality depends on gating and batching; pathological high-entropy prompts may trigger frequent activation.
- Scorer fragility: NLI on short texts can be unreliable; mitigated by embedding prefilter, contradiction-aware aggregation, specificity prior, and structural checks for code/SQL.
- Degeneration risk: Penalizing divergence may suppress creative/minority-correct answers. Scope to reliability-critical domains or reduce β; disable in open-ended creative tasks.
- Myopia: Short rollouts (L) may miss long-range issues; probe L-quality trade-offs.
- Draft-model bias: Mismatch between D and T could misguide; evaluate self-draft and quantized T-rollouts.

## 6. Conclusion
LCD adds a token-local stability prior to decoding by scoring agreement among cheap micro-rollouts. This preventive signal aims to steer models away from unstable trajectories that produce factual and structural errors. Our falsification-oriented experiments will test whether this lookahead signal yields reliable quality gains at acceptable cost. If validated, LCD is a composable building block for more reliable inference, especially in agentic workflows where single-token errors are costly.

## 7. Reproducibility
We will release:
- Code with deterministic seeds, exact HF model IDs, and scripts for KV batching and gating.
- IDF corpus and s_max statistics, calibrator checkpoints, and all hyperparameters.
- Evaluation prompts, compute accounting scripts, and analysis notebooks in a containerized setup.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
