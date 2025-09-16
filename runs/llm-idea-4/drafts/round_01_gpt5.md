Major Revisions

Title: Lattice-of-Thought Decoding: Budget-Aware Online Pruning for Efficient Reasoning in LLMs

Abstract:
We introduce Lattice-of-Thought (LoT) decoding, a budget-aware inference procedure that expands multiple short reasoning branches in parallel and prunes them online with a lightweight scorer. Unlike linear Chain-of-Thought (CoT) or sampling-based self-consistency, LoT explicitly allocates a fixed token budget across a shallow lattice of partial proofs, using in-model signals (token log-probabilities) and answer-consistency cues to concentrate tokens on promising branches. LoT requires no external verifiers, runs with small open-source models, and can be implemented with standard logprob access. We provide an experimental falsification plan on GSM8K, SVAMP, and ARC-Challenge, comparing accuracy at matched total-token budgets against linear CoT and self-consistency. We detail the algorithm, budget accounting, hyperparameter schedules, and ablations to isolate the contribution of online pruning versus branching alone. The central hypothesis is that early, shallow exploration plus online pruning improves accuracy per token by reducing wasted reasoning on dead-end traces.

Introduction:
Reasoning-oriented decoding for large language models typically follows either:
- Linear Chain-of-Thought (CoT): one long trace that risks committing early to suboptimal trajectories and wasting tokens.
- Self-consistency: many independent full traces with majority voting, which is robust but token-inefficient due to late discovery of dead ends.
- Tree-of-Thought-style search: branching with external verifiers, often incurring heavy evaluation overhead that is impractical under tight inference budgets.

In many reasoning tasks, uncertainty is highest early and collapses as partial computations converge. This suggests an inference principle: allocate width early to keep options open, cheaply probe which answers are emerging, and then focus on a small number of branches. We formalize this as Lattice-of-Thought (LoT) decoding: a shallow, budgeted lattice of short branches with online pruning governed by a scorer that combines normalized in-model likelihood with cross-branch answer consistency. LoT differs from beam search by optimizing for accuracy per token under a fixed budget rather than sequence likelihood, and from previous ToT methods by relying on cheap, in-model signals instead of trained external verifiers.

Method:
Problem setting:
- Input: question x.
- Output: final answer y with optional reasoning trace r.
- Constraint: total token budget B_total across all model calls, including reasoning, scoring “guess” probes, and final answer tokens.

High-level procedure:
- Expand K short branches in parallel for small chunks of tokens.
- After each chunk, obtain a cheap answer guess from each branch.
- Score each branch using a combination of normalized log-likelihood and cross-branch consistency of guessed answers.
- Prune to a fixed width and repeat until budget is exhausted or a branch finalizes.

Branch representation:
- A branch b holds: prompt prefix P (task instruction), question x, partial reasoning tokens r_b, cumulative token count t_b, cumulative logprob sum L_b, and the latest short “guess” y_hat_b.

Scoring:
- Likelihood score s_lm(b): average per-token logprob of r_b to reduce length bias. Use either running average or a recency-weighted average over the last M tokens.
- Consistency score s_cons(b): cluster branches by their guessed answer y_hat. Let c(y) be the number of branches predicting y. A simple, calibration-free form is s_cons(b) = log(1 + c(y_hat_b)).
- Total score s(b) = z_lm(b) + α * z_cons(b), where z_· are per-step z-normalized scores across active branches, α ∈ [0.5, 2] controls the weight on consistency.

Guess extraction (cheap consistency):
- After generating a chunk of reasoning tokens for a branch, append a minimal, restricted guess prompt:
  - Arithmetic/short-form answers: “Therefore, the answer is:” and decode at most 3–5 tokens greedily; extract numeric string.
  - Multiple choice: “Answer:” with constrained decoding to options {A, B, C, D}.
- Do not commit guess tokens to the branch reasoning; count them toward the budget.
- Normalize guesses (strip punctuation, normalize numerals, simple rounding for floats).

Online pruning and lattice schedule:
- Initialize width W0 (e.g., 6) and chunk length C0 (e.g., 8 tokens).
- At iteration i:
  1) From each active branch, sample S short continuations (e.g., S=1–2) with temperature τ and nucleus p, each up to C_i tokens; update L_b by summing logprobs.
  2) For each resulting branch, run a cheap guess probe (≤5 tokens).
  3) Score all branches via s(b); prune to width W_i+1 (e.g., decay to 3 over time).
  4) Increase chunk length modestly (e.g., C_{i+1} = C_i + ΔC) and decrease width (W schedule) to shift from breadth to depth.
  5) Stop if any branch emits an explicit finalization trigger (e.g., “Final Answer:” or meets task-specific end condition) or if budget is exhausted.
- Output: if multiple finalized branches remain, choose the most common answer; tie-break by highest s(b) or total logprob.

Budgeting and accounting:
- Total B_total includes:
  - All reasoning tokens across branches.
  - All guess tokens for consistency probes.
  - Final answer tokens.
- A typical configuration at B_total ≈ 300 tokens might be:
  - Iteration 1: W0=6, chunk 8, guess 3 → ~66 reasoning + 18 guess
  - Iteration 2: W1=4, chunk 12, guess 3 → ~48 reasoning + 12 guess
  - Iteration 3: W2=3, chunk 20, guess 3 → ~60 reasoning + 9 guess
  - Finalization and short explanation: remainder
- Overhead of guesses is ≤15% of the budget and is offset by pruning low-promise branches.

Differences from beam search and ToT:
- Beam search maximizes likelihood of a single sequence; LoT targets accuracy per token by mixing likelihood with emerging answer agreement.
- ToT methods often require heavy external verifiers or planning steps; LoT uses the model’s own probabilities and minimal probes.
- Self-consistency explores late and discards entire samples; LoT discards early, preserving budget for promising reasoning lines.

Algorithm (pseudocode-style steps):
1) Input x, budget B_total, width schedule {W_i}, chunk schedule {C_i}, α, τ, p.
2) Initialize one root branch b0 with prompt P and question x, L_b0=0, t_b0=0.
3) Active set A ← {b0}.
4) For i = 1.. while tokens used < B_total:
   a) Expand: For each b in A, sample up to S continuations of length ≤ C_i with temperature τ and nucleus p; update L and t.
   b) Guess: For each new branch, append a minimal guess prompt and decode ≤ G tokens; extract y_hat; update t.
   c) Score: Compute s_lm as mean logprob; compute s_cons from cluster counts; z-normalize; s = z_lm + α*z_cons.
   d) Prune: Keep top W_i branches by s; deduplicate near-identical suffixes (e.g., identical last 12 tokens).
   e) Terminate early if any branch emits “Final Answer:” and has a valid y_hat.
5) Output majority answer among finalized branches; tie-break by s or L.

Experiments (falsification plan):
Goals:
- Test whether LoT improves accuracy at equal total token budgets relative to linear CoT and self-consistency.
- Quantify the trade-off between guess overhead and pruning gains.
- Validate that improvements hold across small open-source models.

Models:
- Llama 3 8B Instruct (or 8B base with task prompt), Mistral 7B Instruct, Qwen2 7B Instruct. Use FP16 or 4-bit quantization; require logprob access.

Datasets and setups:
- GSM8K (grade-school math, free-form numeric answers).
- SVAMP (math word problems, numeric answers).
- ARC-Challenge (multiple-choice science; answers A–D).
- Few-shot prompting: 6–8 CoT exemplars per dataset, held fixed across methods.

Baselines:
- Linear CoT: greedy decoding of one reasoning trace with “Let’s think step by step,” then final answer.
- Self-consistency: N samples (e.g., N=5, 10), majority vote over answers; no online pruning.
- Optional: beam search over tokens (width 3–5) targeting likelihood, to contrast with budget-aware scoring.

Budget-matched evaluation:
- Budgets B_total ∈ {150, 300, 600} tokens per question, inclusive of all model tokens (reasoning, guesses, answers).
- For self-consistency, determine N and max length so total tokens match B_total on average; enforce with truncation and early stopping.
- For CoT, cap length to match B_total; if shorter, allow a brief verification prompt to use remaining budget (parity with LoT finalization).

Metrics:
- Accuracy (exact match for numeric, multiple-choice letter accuracy for ARC).
- Total tokens used per sample (mean, std).
- Overhead fraction spent on guesses (LoT).
- Stability: variance across 3 random seeds (sampling noise).

Implementation details:
- HuggingFace Transformers with logprob capture.
- Constrained decoding for ARC options via logit masking.
- Numeric normalization: strip commas, spaces; accept within 1e-3 tolerance for floats.
- Deduplication: compare last-k tokens; k=12.
- Scoring weights: α tuned on a small development split (5% of training set or held-out subset of test if no train labels), reported separately; also report α=1.0 default.
- Temperature τ ∈ {0.5, 0.7}; nucleus p ∈ {0.9, 0.95}.

Ablations:
- No-consistency (α=0): likelihood-only pruning.
- No-likelihood: consistency-only pruning.
- No-guess: replace s_cons with 0; observe drop.
- Width/depth schedule: fixed width vs decaying width; chunk size fixed vs increasing.
- Guess length: 1, 3, 5 tokens.
- S=1 vs S=2 continuations per branch per step.

Hypotheses and falsifiable outcomes:
- H1: At fixed B_total, LoT > CoT accuracy on GSM8K, SVAMP, and ARC-C.
  - Falsified if no significant improvement at any budget on ≥2 models.
- H2: At fixed B_total, LoT achieves accuracy comparable to self-consistency while using fewer tokens for incorrect branches (lower wasted-token fraction).
  - Falsified if self-consistency dominates LoT at matched budgets.
- H3: Consistency component materially contributes to gains.
  - Falsified if α=0 matches or exceeds full LoT.
- H4: Benefits persist under small models (≤8B).
  - Falsified if gains vanish on 7–8B models.

Expected failure modes and diagnostics:
- If guesses are noisy early, s_cons may be misleading; measure early-step guess accuracy vs final answers to calibrate α or delay consistency until later iterations.
- If likelihood favors superficial verbosity, normalize by per-token logprob and clip repeated boilerplate.

Discussion:
Why it can help:
- Early-stage uncertainty: Many problems have multiple plausible solution schemas; short probes can identify which schemas converge to consistent answers, allowing budget reallocation.
- Cheap signals: In-model logprobs and tiny answer probes are far cheaper than full verifier calls or independent full-trace sampling.
- Budget-awareness: By pruning early and lengthening chunks later, LoT avoids spending most of the budget discovering that a path is a dead end.

Relationship to prior work:
- Chain-of-Thought improves reasoning but is linear and commit-heavy.
- Self-consistency reduces variance but is token-costly.
- Tree-/Graph-of-Thought frameworks introduce structured exploration but often rely on external verifiers or agentic tool use, increasing overhead.
- LoT occupies a middle ground: shallow structure, in-model scoring, explicit token budgeting, and online pruning.

Practical considerations:
- API support: Requires logprob access and short constrained completions; widely available in open-source stacks.
- Parallelization: Branch expansions can be batched; guess probes can be concatenated in a single batched pass to amortize overhead.
- Determinism vs diversity: Temperature controls exploration; small τ early with wider width can balance diversity and stability.

Limitations:
- Consistency ≠ correctness: Clustering around an incorrect early guess can prune the correct branch; mitigation via delayed or tempered α in early steps.
- Domain specificity: Numeric tasks benefit from compact guesses; open-ended generation may need task-specific guess extractors or validators.
- Overhead sensitivity: If model logprob calibration is poor, likelihood scores can be unreliable; consider rank-based or z-normalized scoring, as proposed.
- Implementation complexity: Managing multiple branches and budget accounting adds engineering overhead relative to linear decoding.

Conclusion:
We propose Lattice-of-Thought decoding, a budget-aware, online-pruned branching strategy for LLM reasoning that uses only in-model signals and minimal guess probes. The method is simple to implement with open-source models, directly optimizes accuracy per token, and provides clear falsification routes against strong baselines at matched budgets. If validated, LoT offers a practical alternative to both linear CoT and large-sample self-consistency, improving the efficiency of reasoning inference and enabling better performance under tight token constraints.
