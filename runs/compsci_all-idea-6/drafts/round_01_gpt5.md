Major Revisions

Title
Error-Correcting Speculative Decoding via Branch Voting

Abstract
Speculative decoding accelerates large language model (LLM) inference by proposing multi-token drafts from a small “drafter” model and verifying them with the target “verifier” model. Existing schemes typically accept or reject a single drafted branch based on verifier probabilities, discarding the redundancy present in multiple speculative branches. We propose Error-Correcting Speculative Decoding via Branch Voting (ECSV): treat speculative branches as noisy redundant encodings of the next tokens and fuse them with a reliability-weighted vote before verification. Under a fixed verifier compute budget, ECSV increases the probability that the fused proposal matches the verifier, improving acceptance rate and perplexity without increasing the number of verifier forward passes. We provide: (i) a coding-theoretic formulation where branches are repetition codes passed through branch-specific noise; (ii) a practical on-the-fly reliability estimator using drafter uncertainty, inter-branch agreement, and a small number of verifier spot checks; (iii) a fusion rule that reduces to MAP decoding under an independence model; and (iv) a falsifiable experimental plan using open models (e.g., TinyLlama 1.1B → Mistral 7B) that matches verifier budget while measuring perplexity, acceptance length, and downstream task accuracy. We hypothesize that ECSV outperforms standard accept/reject and unweighted voting baselines, especially with noisier or quantized drafters.

Introduction
Speculative decoding amortizes verifier compute by drafting several future tokens with a small model and verifying them with the large target model. Speedups depend critically on how often the verifier accepts a long drafted prefix. Many methods invest verifier compute to accept or reject a single drafted branch per step; multiple branches (via diverse sampling or tree search) are used mainly for fallback, not fusion.

Observation: a set of branches constitutes redundant information about the next k tokens. Branches are correlated but not identical; each has tokenwise error relative to the verifier. From a coding-theoretic perspective, these branches are noisy encodings produced by a shared source (the verifier’s next-token distribution) transmitted through branch-specific noise (drafter bias and sampling). Error-correcting codes improve reliability by redundancy and weighted fusion based on channel reliabilities.

Hypothesis: If we (1) generate several low-cost branches, (2) estimate per-branch reliabilities with cheap signals and a few verifier spot checks, and (3) fuse the branches into a single k-token proposal with a reliability-weighted vote, then for the same verifier budget we increase the expected accepted prefix length and reduce perplexity.

Contributions:
- Coding-theoretic framing of speculative branches as repetition codes with branch-specific noise.
- ECSV: a reliability-weighted branch fusion algorithm that preserves verifier budget.
- Practical reliability estimation with minimal verifier overhead.
- Falsifiable plan and open-source implementation path on small models.

Method
Setup
- Verifier model V (e.g., 7B) produces next-token distribution pV(. | x).
- Drafter model D (e.g., 1B) proposes B branches of depth k: for branch b in {1..B}, tokens tb,1:k with drafter distributions q_b,j(.).
- Standard speculative decoding verifies a single branch, accepting the longest prefix consistent with V under rejection sampling. We instead fuse branches into a single proposal t*,1:k, then verify that fused sequence.

Coding-theoretic view
- For position j, the unknown correct symbol y_j is drawn from pV(. | x, t*,1:j-1).
- Each branch b provides a noisy observation t̂_b,j with error rate ε_b,j = P(t̂_b,j ≠ y_j).
- Under independent branch noise with known ε_b,j, MAP decoding reduces to a reliability-weighted majority: choose y_j maximizing sum_b log( (1−ε_b,j) if t̂_b,j = y_j else ε_b,j/(V−1) ), where V is vocab size.
- In practice, ε_b,j is unknown and branches are not independent; we estimate reliabilities and use them as voting weights.

Reliability signals
We estimate a per-branch reliability r_b and, optionally, per-position r_b,j, using:
- Drafter uncertainty: lower entropy H(q_b,j) implies higher confidence. Define u_b,j = −H(q_b,j).
- Inter-branch agreement: agreement_b,j = fraction of other branches matching t̂_b,j.
- Local self-consistency: q_b,j(t̂_b,j) (drafter’s probability mass on its own sampled token).
- Sparse verifier spot checks: at a small set S of positions (selected for high disagreement or uncertainty), run V to obtain pV(.). Compute consistency features c_b,j = log pV(t̂_b,j) − log pV(second-best). These require verifier passes but we constrain |S| to match baseline verifier budget (see Budget control).

Reliability model
We combine signals into weights via a monotone mapping:
- r_b,j = α1·u_b,j + α2·agreement_b,j + α3·log q_b,j(t̂_b,j) + α4·I[j∈S]·c_b,j
- r_b = mean_j r_b,j over a sliding window or over 1..k.
We then transform to positive weights w_b,j = exp(γ·r_b,j) (positionwise) or w_b = exp(γ·r_b) (branchwise). Hyperparameters α, γ chosen on a small held-out set; γ implicitly maps reliability to an effective error rate.

Fusion rule
For each position j in 1..k:
1) Collect candidate tokens C_j = union_b {t̂_b,j} ∪ top-M tokens under average q_b,j for tie-breaking.
2) Score each token t in C_j by S(t,j) = sum_b w_b,j·1[t̂_b,j = t] + λ·soft_vote(t), where soft_vote(t) = sum_b w_b,j·q_b,j(t).
3) Select t*_j = argmax_t S(t,j). If ties remain, break by a small verifier logit check at j if available; else by average q_b,j.

The fused draft t*,1:k is then passed to the verifier for acceptance, identical to standard speculative decoding’s final check, except using the fused tokens.

Budget control
We enforce an equal verifier budget to standard speculative decoding:
- Baseline: verify up to L tokens per step (expected accepted length) with at most one verifier forward per token position.
- ECSV: split the same expected number of verifier forwards into (i) sparse spot checks at uncertain positions before fusion (|S_pre|), and (ii) acceptance verification of the fused tokens (|S_post|). Constrain |S_pre| + |S_post| ≤ L.
- If fusion increases expected accept length beyond L, we cap |S_post| at L − |S_pre| and bank the surplus (i.e., defer savings to later steps), ensuring matched average verifier tokens per output token.

Practical algorithm (per decoding step)
1) Drafting: Sample B branches of depth k from D with moderate temperature and nucleus p.
2) Scoring: Compute q_b,j distributions and features u, agreement, self-consistency.
3) Select spot-check positions S_pre: choose up to R positions with highest entropy or disagreement; run V on these positions to get logits and compute c_b,j. This uses R verifier forwards.
4) Fusion: Compute weights w_b,j and t*,1:k by the fusion rule above.
5) Acceptance: As in standard speculative decoding, run V forward along t*,1:k until a mismatch; accept the matched prefix. Ensure verifier forwards used here are ≤ L − R.
6) If verifier budget unused (due to early mismatch), optionally allocate one additional spot check for the next step.

Complexity
- Drafter cost: O(B·k) small-model forwards; kept cheap by using a 1B-class model and sampling in batch.
- Verifier cost: same as baseline by construction; extra spot checks reduce acceptance verification passes accordingly.
- Memory: stores B·k tokens and light features; minimal overhead.

Why it should work
- Majority-decoding bound: If more than half of the total reliability weight favors the true token at j, the fused token matches V with probability greater than any single branch. With even weakly informative, diverse branches (ε_b,j < 0.5 on average), reliability-weighted votes strictly improve correctness probabilities.
- Acceptance multiplier: Acceptance length is the length of the longest prefix where fused tokens match V. Increasing per-position match probability multiplicatively increases expected prefix length.

Experiments (falsification plan)
Goals
- Test whether ECSV improves verifier-aligned accuracy (perplexity vs V, acceptance length, downstream task accuracy) for the same verifier budget.
- Stress robustness with noisy/quantized drafters and diverse branch counts.

Models
- Verifiers: Mistral-7B-v0.3 (Apache 2.0), Pythia-6.9B.
- Drafters: TinyLlama-1.1B, Pythia-1.4B, Qwen1.5-0.5B.
- Quantization: 4-bit NF4 for drafter; optional 4-bit for verifier in stress tests.

Data
- Language modeling: WikiText-103 test and The Pile validation slices.
- Short-form QA: TriviaQA (unfiltered) short answers.
- Reasoning: GSM8K (concise final answer), with CoT disabled to keep k small.

Baselines
- SD-1: Standard speculative decoding (single branch), tuned k and temperature.
- SD-multi: Multi-branch accept-first-that-passes (no fusion).
- Vote-unif: Unweighted majority vote across branches, then standard verification.
- Oracle-Top1: Best single branch chosen with full verifier lookahead (upper bound; not budget-matched; for reference only).

ECSV variants
- ECSV-lite: weights from drafter entropy and inter-branch agreement only (no spot checks).
- ECSV-spot: adds R=1..4 verifier spot checks per step focused on top-disagreement positions.
- ECSV-soft: includes soft_vote (λ>0).

Verifier budget matching
- Define L_base as average verifier tokens per output token for SD-1 at its optimal hyperparameters.
- For each method, tune hyperparameters under the constraint that average verifier tokens per output token ≤ L_base ± 1%.
- Report both matched-budget metrics and wall-clock throughput on a single A100.

Metrics
- Perplexity with respect to verifier V on held-out text (teacher-forced).
- Acceptance length: expected number of tokens accepted per verify step; acceptance rate distribution.
- Effective speed: tokens/sec and verifier tokens per output token.
- Downstream accuracy: exact match for QA and GSM8K final answers.
- Robustness: performance under drafter quantization and increased sampling temperature.

Ablations
- Branch count B ∈ {2, 4, 8}; depth k ∈ {2, 4, 8}.
- Weighting signals: entropy-only, agreement-only, self-consistency-only, +spot checks.
- Spot-check budget R ∈ {0,1,2,4} and selection policy (entropy vs disagreement).
- Tie-breaking by verifier logits vs average q.

Falsification criteria
- If, at matched verifier budget, ECSV fails to improve both (i) perplexity and (ii) acceptance length over SD-1 on ≥2 datasets, reject the hypothesis.
- If Vote-unif matches ECSV, the reliability modeling is not contributing; revise method.
- If improvements vanish under low B (e.g., B=2) and realistic k (e.g., k=4), the approach may be impractical.

Expected outcomes
- ECSV-lite > SD-1 in acceptance length by 5–15% with equal budget; small perplexity gains.
- ECSV-spot adds further 2–5% acceptance gains with R=1–2.
- Gains increase with noisier drafters (quantized or higher temperature).
- Downstream accuracy improves modestly (1–2 pp) at the same or better throughput.

Discussion
Relation to prior work
- Standard speculative decoding accepts or rejects a single branch; multi-branch variants select among branches but do not fuse them.
- Approaches like Medusa add network heads to predict multiple tokens; our approach is model-agnostic and uses branch redundancy available in any drafter.
- Self-consistency and voting have been used across full generations; ECSV operates at token-level inside a single decode step with strict verifier budget accounting.

Why reliability weighting matters
- Branches differ in quality: some are confidently wrong, others uncertain but closer to V. Entropy, inter-branch agreement, and sparse verifier cues provide calibrated signals of branch noise. Weighted fusion approximates MAP decoding under a simple noise model, while unweighted votes can be misled by correlated errors.

Failure modes and mitigations
- Correlated errors: If branches coalesce on the same wrong token due to shared drafter bias, voting will not help. Spot checks targeted at high-agreement/high-entropy positions can detect and flip such cases.
- Overfitting weights: Over-tuned α, γ may not generalize. Use small held-out calibration and prefer simple monotone mappings.
- Budget drift: Strictly track and cap verifier forwards per output token.

Limitations
- Independence assumption is violated in practice; branches share model and context, limiting the theoretical guarantees.
- The fused sequence is not an exact sample from the verifier distribution; like standard speculative decoding, we target correctness and efficiency, not sampling fidelity.
- Extra drafter compute and branch storage increase memory and latency on small GPUs; benefits may shrink when drafter is not much cheaper than verifier.
- Hyperparameter sensitivity: B, k, and weighting coefficients require tuning per model pair.

Conclusion
We propose ECSV, a coding-theoretic approach to speculative decoding that fuses multiple drafted branches via reliability-weighted voting before verification. By converting redundancy into error correction, ECSV increases the probability that the fused draft matches the verifier, enabling longer accepted prefixes and lower perplexity under the same verifier budget. The method is simple to implement with open models, budget-aware, and testable with clear falsification criteria. If validated, ECSV provides a principled and practical upgrade to speculative decoding wherever multiple branches can be generated cheaply.
