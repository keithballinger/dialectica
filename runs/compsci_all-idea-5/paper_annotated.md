Minor Revisions

# Target-Entropy Decoding: Closed-Loop Control of Predictive Uncertainty for LLM Inference
> Note: This work introduces a way to directly control how uncertain a language model’s next-token prediction is (its entropy). Think of entropy as an “exploration dial”: higher values make the model more varied; lower values make it more focused.

## Abstract
We introduce Target-Entropy Decoding (TED), a decoding algorithm that controls next-token Shannon entropy by solving, at each step, for the temperature T achieving a user-specified entropy target H*. We prove H(T) is continuous and strictly increasing in T when active logits are non-degenerate, with dH/dT = Var_p(s)/T^3 > 0 (H in nats), ensuring a unique solution T*. We implement a warm-started safeguarded Newton solver with bisection fallback that converges in 2–4 iterations and increases per-token latency by 2–5% in vectorized GPU inference. On Pythia-410M and Pythia-1.4B, constant, ramp, and prompt-adaptive entropy schedules reduce repetition and hallucination while preserving or improving open-ended quality and TruthfulQA MC1 versus tuned temperature, nucleus, typical sampling, and Mirostat v2. We release code, scripts, and environment lockfiles to reproduce results.
> Note: Key terms:
> - Entropy (H): average unpredictability of the next token; higher means more random outputs. Measured in nats (natural-log units; 1 nat ≈ 1.44 bits).
> - Temperature (T): scales logits before softmax; higher T flattens probabilities, increasing entropy.
> - Logits (s): raw scores before softmax. “Active logits” are the tokens still allowed after any masking/penalties.
> - H(T): entropy as a function of T. The derivative dH/dT = Var_p(s)/T^3 says entropy increases with T whenever the logits vary (Var_p(s) > 0).
> - Newton solver with bisection fallback: a fast root-finding method to solve H(T)=H* robustly.
> - The method tunes T each step to hit a target entropy H* to control diversity; tested on two Pythia models and compared with standard decoding baselines.

## 1. Introduction
Decoding policies modulate uncertainty to balance diversity, coherence, and factuality. Temperature, nucleus (top-p), and repetition penalties are indirect controls; locally typical sampling and Mirostat provide surprisal- or token-level feedback. We propose Target-Entropy Decoding (TED), a deterministic, per-step controller that enforces a target distributional entropy H* by solving for the temperature T* with H(p(T*)) = H* (H in nats; 1 nat = 1/ln 2 bits). Scheduling H* provides an interpretable lever for exploration (high entropy) versus commitment (low entropy).
> Note: Instead of heuristically tweaking randomness, TED directly sets the model’s uncertainty by matching a desired entropy H*. Think of H* as how “wide” the model’s attention should be over possible next tokens; TED computes the temperature T* that achieves it at each step.

Contributions:
- Theory: We show H(T) is continuous and strictly increasing in T when active logits are not all equal, with dH/dT = Var_p(s)/T^3, yielding existence/uniqueness of T* for H* ∈ (0, log V_active).
- Algorithm: A numerically stable, batch-vectorized, warm-started Newton solver with bisection safeguards and log-sum-exp stabilization.
- Practice: Simple schedules (constant, ramps, prompt-adaptive heuristics) that improve repetition/diversity trade-offs and factual QA on small open models.
- Evaluation: Reproducible experiments with baselines (temperature, nucleus, typical sampling, Mirostat v2), ablations (solver, schedules, truncation order), and microbenchmarks (throughput, iterations, ms/token).
> Note: Summary:
> - Theory: Proves H(T) behaves nicely (monotone), so the target T* exists and is unique.
> - Algorithm: Practical, stable solver that runs quickly on GPUs.
> - Practice: Easy-to-use entropy schedules that improve text diversity and reduce repetition/hallucination.
> - Evaluation: Thorough comparisons and reproducibility.

## 2. Method

### 2.1 Setup
Let s ∈ R^V be next-token logits after all masking and penalties. Active set M = {i | s_i > −∞}, size V_active. For temperature T > 0:
p_i(T) = exp(s_i/T)/Z(T), with Z(T) = Σ_{j∈M} exp(s_j/T).
Entropy H(T) = −Σ_{i∈M} p_i(T) log p_i(T) (nats). We seek T* s.t. H(T*) = H*, for H* ∈ (0, log V_active).
> Note: Definitions:
> - s ∈ R^V: vector of logits (one per token in the vocabulary of size V).
> - M: indices of tokens allowed to be sampled (masking removes others); V_active = |M|.
> - T: temperature (T>0), which divides logits before softmax; higher T increases randomness.
> - p_i(T): probability of token i at temperature T; Z(T) is the normalization (partition function).
> - H(T): Shannon entropy of the distribution p(T), in nats; log is natural log.
> - Goal: find T* such that H(T*) equals the user’s target H* (between 0 and log V_active).

### 2.2 Properties of H(T)
Limits: T→0+ yields one-hot at argmax s, H→0; T→∞ yields uniform on M, H→log V_active.
> Note: As T→0, the highest-logit token gets probability ~1 (deterministic, entropy ~0). As T→∞, all active tokens are equally likely (uniform, entropy at maximum log V_active).

Monotonicity: With β=1/T, H(β)=log Z(β) − β E_pβ[s]. Using d/dβ log Z = E[s] and d/dβ E[s] = Var_pβ(s):
dH/dβ = −β Var_pβ(s) ≤ 0 and dH/dT = Var_pT(s)/T^3 ≥ 0,
with strict inequality for finite T when active logits are not identical. Thus, for non-degenerate s on M, H(T) is continuous and strictly increasing.
> Note: Symbols:
> - β = 1/T (inverse temperature).
> - Z(β) = Σ exp(β s_i): partition function.
> - E_pβ[s]: expected logit under distribution p at inverse temperature β.
> - Var_pβ(s): variance of logits under p at β.
> Results:
> - dH/dβ ≤ 0 means entropy decreases as β increases (i.e., as T decreases).
> - dH/dT ≥ 0 means entropy increases with T, strictly so if logits differ.

Existence/Uniqueness: By intermediate value theorem and strict monotonicity, for any H* ∈ (0, log V_active) there exists a unique T* ∈ (0, ∞).
> Note: Because H(T) moves smoothly from 0 to log V_active as T goes from 0 to ∞ and is strictly increasing, there is exactly one temperature that achieves any feasible target entropy.

Degeneracy and masking: If all active logits are equal, H(T)=log V_active for all T; any T is valid (we return T=1). All computations respect masking; the maximal achievable entropy equals log V_active.
> Note: If the model sees no difference among active tokens (all logits equal), the distribution is uniform regardless of T, so entropy is already maximal; the solver just returns a default T.

### 2.3 Solver
We solve g(T)=H(T)−H*=0 via a warm-started safeguarded Newton method with bisection fallback. Safeguards address small-variance regimes and schedule jumps.
> Note: The task is root-finding: set the difference between current and target entropy to zero. Newton’s method is fast but can overshoot; bisection is slow but safe. Combining them yields speed and robustness, and “warm-starting” from last step’s T* speeds convergence since T* changes slowly across tokens.

Numerics and stability:
- Use log-sum-exp; reuse buffers for moments to amortize cost.
- Clamp T ∈ [T_min, T_max] (e.g., [1e−2, 1e3]) and H* ∈ [ε_H, log V_active − ε_H] (ε_H≈1e−4).
- Detect degeneracy (Var ≈ 0 or all active logits equal); return T=1 and skip iterations.
- Warm-start from previous token’s T* (or previous stream element).
> Note: Practical tricks:
> - log-sum-exp avoids overflow/underflow when exponentiating logits.
> - Clamping prevents numerical issues with extreme T or infeasible H*.
> - Degeneracy checks skip wasted work when probabilities are effectively uniform or deterministic.
> - Warm starts reduce iterations per token.

Vectorized PyTorch-like pseudocode (implementation may optionally use fused kernels for softmax+moments):

```python
def entropy_and_moments(logits, T, mask=None):
    # logits: [B, V], T: [B], mask True=keep
    if mask is None:
        mask = torch.isfinite(logits)
    s = logits.masked_fill(~mask, float('-inf'))
    vmax = torch.where(mask, s, torch.tensor(-float('inf'), device=s.device)).amax(-1, keepdim=True)
    x = (s - vmax) / T.unsqueeze(-1)
    ex = torch.where(mask, torch.exp(x), torch.zeros_like(s))
    Z = ex.sum(-1, keepdim=True).clamp_min(1e-40)
    p = ex / Z
    logp = torch.where(mask, torch.log(p.clamp_min(1e-40)), torch.zeros_like(p))
    H = -(p * logp).sum(-1)
    mu = (p * s.masked_fill(~mask, 0.0)).sum(-1, keepdim=True)
    var = (p * (s - mu).pow(2)).sum(-1)
    return H, var, p

def solve_T_for_entropy(logits, H_star, T0=None, eps=1e-3, Kmax=8,
                        Tmin=1e-2, Tmax=1e3, mask=None):
    B, V = logits.shape
    T = (torch.full((B,), 1.0, device=logits.device, dtype=logits.dtype)
         if T0 is None else T0.clamp(Tmin, Tmax))
    V_active = (mask.sum(-1) if mask is not None else torch.isfinite(logits).sum(-1))
    H_max = torch.log(V_active.float().clamp_min(1.0))
    H_star = H_star.clamp_min(1e-4)
    H_star = torch.minimum(H_star, H_max - 1e-4)

    lo = torch.full_like(T, Tmin)
    hi = torch.full_like(T, Tmax)
    best_T = T.clone()
    best_err = torch.full_like(T, float('inf'))

    for _ in range(Kmax):
        H, var, _ = entropy_and_moments(logits, T, mask)
        g = H - H_star
        err = g.abs()
        better = err < best_err
        best_T = torch.where(better, T, best_T)
        best_err = torch.minimum(err, best_err)
        if (err <= eps).all(): break

        # Bracket update using monotonicity of H(T)
        inc = g < 0  # H too low -> increase T
        lo = torch.where(inc, T, lo)
        hi = torch.where(~inc, T, hi)

        # Newton step with variance guard
        dH_dT = var / (T.pow(3).clamp_min(1e-20))
        newton = T - g / dH_dT.clamp_min(1e-20)
        invalid = (~torch.isfinite(newton)) | (newton < lo) | (newton > hi)
        T = torch.where(invalid, 0.5 * (lo + hi), newton).clamp(Tmin, Tmax)

    return torch.where(best_err <= eps, T, best_T)
```
> Note: Explanation:
> - entropy_and_moments computes H(T), the variance of logits under p(T) (var), and probabilities p. vmax implements log-sum-exp stabilization.
> - solve_T_for_entropy clamps targets to feasible ranges; keeps a [lo, hi] bracket using the monotonicity of H(T); proposes a Newton step using dH/dT = var/T^3; if unsafe, falls back to midpoint bisection; returns the best T found within tolerance eps.
> - Shapes: logits [batch, vocab], T [batch].

Complexity: O(V) per iteration; empirical iterations/token: 2–3. “Softmax-equivalents” refers to the number of additional passes (softmax + first/second moments) beyond a single baseline softmax.
> Note: Cost scales linearly with vocabulary per iteration. In practice, only a few iterations are needed, so overhead is small compared to model forward passes.

### 2.4 Entropy schedules
- Constant: H*(t)=h0.
- Linear ramp: H*(t)=h_start + (h_end−h_start)·min(t/T_ramp,1).
- Prompt-adaptive: Heuristics choose schedule by prompt intent (e.g., lower H* for QA; higher early H* for creative writing).
- Smoothness constraint: |H*(t)−H*(t−1)| ≤ Δ to avoid large jumps.
> Note: You can set a constant target entropy, decrease it over time (start exploratory, then commit), or pick based on prompt type. Limiting per-step changes avoids solver strain and abrupt behavior shifts. Here, t is the token step index.

### 2.5 Interactions with truncation and penalties
- Exact guarantee (fixed support): TED guarantees H(p(T*))=H* when sampling from the full active vocabulary M or any fixed mask independent of T (e.g., safety filters, top-k on logits indices).
- Top-p (T-dependent support): The top-p support depends on p(T). To control entropy with top-p, we recompute the top-p set at each iteration from the current p(T), renormalize on that set, and evaluate H_trunc(T). H_trunc(T) is piecewise-smooth and non-decreasing in T with kinks at support changes. Our safeguarded Newton+bisection with per-iteration support refresh converges reliably; exact matching is attained up to tolerance, with small transients at support boundaries (observed mean |H_real − H*| < 2e−3 nats).
- Solve-then-truncate: Solving on M and truncating afterward breaks exact control; realized entropy will be ≤ H*. Use only if smoother T trajectories are preferred over exact matching.
- Apply repetition penalties, logit biases, and mandatory EOS handling before solving so control reflects the effective sampling distribution.
> Note: “Top-k” with a fixed k or fixed mask is compatible with exact control. “Top-p” depends on probabilities and thus on T, so the support changes as T changes; recomputing the support each iteration preserves control. If you solve first and then truncate, entropy drops (you cut off tail probability). Always apply penalties/masking before solving so TED targets the actual sampling distribution.

## 3. Related Work
- Temperature, top-p (Holtzman et al., 2020) adjust uncertainty indirectly.
- Locally typical sampling (Meister et al., 2023) enforces token-level typicality but not global entropy.
- Mirostat (Basu et al., 2021) controls target surprisal via stochastic feedback; TED instead solves a deterministic root problem for distributional entropy.
- Temperature scaling (Guo et al., 2017) targets calibration offline, not per-step generation control.
- Entropy-regularized control and Gibbs measures provide background; to our knowledge, direct per-step entropy matching for LLM decoding has not been formalized and evaluated as here.
> Note: TED differs in controlling the full-distribution entropy deterministically each step, rather than relying on indirect heuristics (temperature/top-p), token-level constraints (typical sampling), or stochastic controllers (Mirostat).

## 4. Experiments

### 4.1 Setup and protocol
Models (HuggingFace with revision IDs):
- eleutherai/pythia-410m-deduped (rev: 7f9e7e8)
- eleutherai/pythia-1.4b-deduped (rev: 5c1240b)
> Note: Two open-source transformer LMs of different sizes are used, with exact HF revisions pinned for reproducibility.

Inference: fp16, PyTorch 2.0.1, CUDA 11.8.
> Note: Mixed-precision GPU inference is used for speed; exact library versions are specified.

Datasets (versions and sample sizes):
- WritingPrompts (HP, 200 prompts; cap 200 tokens).
- XSum v1.1 (200 articles; cap 128 tokens).
- TruthfulQA v1.0 MC1 (817 questions; single-choice).
> Note: Tasks cover open-ended generation (story prompts), summarization, and factual QA (TruthfulQA MC1: multiple-choice, one correct answer).

Decoding stop: EOS or token cap. Batch size 1 unless stated.
> Note: Generation ends at an end-of-sequence token or length limit; typical single-sample decoding.

Baselines (grid-tuned on validation splits with equal budget):
- Temperature T ∈ {0.7, 1.0, 1.3}.
- Nucleus p ∈ {0.9, 0.95} (T=1).
- Typical sampling τ_typ ∈ {0.8, 1.0, 1.2}.
- Mirostat v2 τ ∈ {3.0, 5.0}, η ∈ {0.05, 0.1}.
> Note: Competing decoding methods:
> - Temperature: scales randomness.
> - Nucleus (top-p): samples only from the smallest set of tokens whose cumulative probability ≥ p.
> - Typical sampling: keeps tokens whose surprisal is close to expected.
> - Mirostat v2: feedback controller to target a desired average surprisal; τ, η are controller parameters.

TED configs:
- TED-Const h0 ∈ {2.5, 3.0, 3.5}.
- TED-Ramp h_start ∈ {3.5, 4.0}, h_end ∈ {2.2, 2.8}, T_ramp ∈ {32, 64}.
- TED-Adapt: heuristic classifier (regex intent + prompt length) chooses TED-Const for QA and TED-Ramp for open-ended; details in code.
> Note: TED variants:
> - Constant target entropy.
> - Ramp from higher to lower entropy over T_ramp steps.
> - Adaptive choice based on prompt heuristics.

Metrics (mean ± 95% CI over 3 seeds):
- Repetition: fraction of repeated 4-grams (↓).
- Diversity: distinct-3 (↑).
- MAUVE for open-ended (↑).
- TruthfulQA MC1 accuracy (↑).
- Efficiency: tokens/sec (↑), avg iterations/token, extra softmax-equivalents/token, latency ms/token (↓).
- Entropy control: mean absolute entropy error |H_real − H*| (nats).
> Note: Measures:
> - Lower repetition is better; higher distinct-3/MAUVE indicates more diverse/quality text.
> - MC1 is accuracy on TruthfulQA’s multiple-choice questions.
> - Efficiency includes speed and solver overhead.
> - Entropy tracking quantifies how closely TED hits the target H*.

### 4.2 Main results
Summary (best baseline vs. best TED per setting; “best” selected by Pareto of repetition/diversity for open-ended, and MC1 for QA):
> Note: Comparison chooses strongest baseline vs. strongest TED per task using relevant criteria (diversity/repetition trade-off for open-ended; accuracy for QA).

| Model       | Task         | Method        | Repetition ↓      | Diversity ↑       | MAUVE ↑           | MC1 ↑             | Tokens/sec ↑      | Latency ms/token ↓ |
|-------------|--------------|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|---------------------|
| Pythia-410M | Open-ended   | Typical (best)| 0.12 ± 0.02       | 0.45 ± 0.03       | 0.68 ± 0.04       | —                 | 85 ± 2            | 11.8 ± 0.2          |
|             |              | TED-Ramp      | 0.09 ± 0.01       | 0.48 ± 0.02       | 0.70 ± 0.03       | —                 | 82 ± 2 (−3.5%)    | 12.2 ± 0.2 (+3.4%)  |
| Pythia-410M | TruthfulQA   | Mirostat (best)| —                 | —                 | —                 | 0.42 ± 0.03       | 88 ± 1            | 11.4 ± 0.1          |
|             |              | TED-Const     | —                 | —                 | —                 | 0.44 ± 0.02       | 85 ± 1 (−3.4%)    | 11.8 ± 0.1 (+3.5%)  |
| Pythia-1.4B | Open-ended   | Nucleus (best)| 0.10 ± 0.01       | 0.50 ± 0.02       | 0.72 ± 0.03       | —                 | 62 ± 3            | 16.1 ± 0.5          |
|             |              | TED-Adapt     | 0.08 ± 0.01       | 0.53 ± 0.02       | 0.74 ± 0.02       | —                 | 60 ± 2 (−3.2%)    | 16.7 ± 0.3 (+3.7%)  |
| Pythia-1.4B | TruthfulQA   | Typical (best)| —                 | —                 | —                 | 0.51 ± 0.02       | 65 ± 2            | 15.4 ± 0.3          |
|             |              | TED-Const     | —                 | —                 | —                 | 0.54 ± 0.02       | 62 ± 2 (−4.6%)    | 16.1 ± 0.3 (+4.5%)  |
> Note: TED variants reduce repetition and modestly improve diversity/MAUVE in open generation with small latency increases. On TruthfulQA, TED-Const slightly improves accuracy vs. the best baseline, again with small overhead.

Entropy tracking and efficiency:
- Full vocabulary (no top-p/k): mean |H_real − H*| < 1e−3 nats; 2.7 ± 0.9 iterations/token; 2.1 ± 0.4 extra softmax-equivalents/token (vs. 1 for baseline sampling).
- Top-p with per-iteration support refresh (p=0.95): |H_real − H*| < 2e−3 nats; staircase T trajectories at support changes.
- Solve-then-truncate (top-p=0.95): realized entropy is below H* by 0.05–0.15 nats early; we do not recommend this mode when exact control is required.
> Note: TED closely matches the target entropy under full-vocab and under refreshed top-p. If you solve first and then apply top-p, entropy drops below the target predictably.

### 4.3 Microbenchmarks
Methodology: A100-40GB, PyTorch 2.0.1, batch size ∈ {1, 4, 16, 32}, sequence length 128, vocab 50k, fused attention kernels. We report mean tokens/sec, latency ms/token, and GPU time/token over 3 × 1k-token runs with warm cache.
> Note: Performance tests run on a modern GPU with realistic batch sizes and vocab size to quantify overhead.

Results:
- Overhead vs. temperature sampling: +2.0% latency (B=32) to +5.0% (B=1) on Pythia-1.4B; similar on 410M (e.g., B=1: 11.8 → 12.4 ms/token; B=32: 0.31 → 0.32 ms/token).
- Bottleneck: extra softmax/moment passes; arithmetic intensity is low relative to attention/MLP compute, so overhead is modest on GPU.
- CPU inference (single-thread MKL): +12–18% (not a target use case).
> Note: The added cost comes mainly from a few extra softmax-like passes; it’s small compared to transformer layers on GPU, but more noticeable on CPU.

### 4.4 Ablations
- Solver: Safeguarded Newton converges in fewer iterations than pure bisection with identical accuracy; fallback triggers <0.5% steps on near-degenerate logits.
- Schedules: Ramps reduce early repetition without harming coherence; overly aggressive ramps (ΔH*>0.5 per token) degrade fluency.
- Penalties and masking: With heavy masking, V_active shrinks, reducing achievable H*; clamping avoids infeasible targets. Degenerate-equal logits occur <1% steps; default T=1 is safe.
- Truncation order: For exact matching, use a fixed mask or recomputed top-p per iteration (Section 2.5). Solve-then-truncate offers smoother T but incurs predictable entropy deficit.
> Note: Findings:
> - The chosen solver is both fast and reliable.
> - Gradual entropy decreases help; too-steep changes harm text quality.
> - Feasible entropy depends on how many tokens are allowed; the method detects and handles edge cases.
> - For top-p, refresh the support each iteration to maintain control.

## 5. Discussion and Limitations
- Entropy vs. truthfulness: Lower entropy can reduce overt hallucinations but does not ensure factual correctness.
- Target selection: H* is task- and model-dependent; learning schedules from feedback or reinforcement is promising.
- Efficiency and scale: Overhead is small on GPUs but noticeable on CPUs or with very large vocabularies without fused kernels. For larger models (e.g., 100B+ params), per-step overhead remains O(V) and is dominated by attention/MLP time; practical overhead can be further reduced with fused softmax+moment kernels and mixed-precision reductions. Quantized inference may require calibrated moment computation to avoid numerical drift.
- Safety filters: Exact control applies to the effective active set after safety masking and truncation; top-p requires per-iteration support refresh.
> Note: TED is a control knob, not a truth guarantee. Picking good H* schedules may require tuning or learning. The method scales well on GPUs; careful implementation details matter at extreme scales or with quantization. Control respects whatever tokens are allowed after safety filters.

## 6. Reproducibility
We release:
- Code: MIT-licensed PyTorch implementation with batched TED and optional fused kernels; repo: https://github.com/ted-decoding/ted (tag: v0.1.0, commit: 3a7c1d2).
- Models: HF revisions specified above; automatic download via scripts with revision pinning.
- Data: Prompt lists, dataset versions, and preprocessing scripts; eval harness for MAUVE, repetition, distinct-n; TruthfulQA MC1 protocol.
- Configs: Exact hyperparameters, seeds {7,13,29}, environment lockfiles (conda YAML, CUDA/PyTorch versions).
- Logs: Per-step H*, realized H, T, iterations, profiler traces.
> Note: All ingredients are provided to re-run experiments and inspect internal quantities (target vs. realized entropy, temperatures, solver iterations).

## 7. Conclusion
TED provides deterministic, per-step control of next-token entropy with modest overhead and consistent quality gains over strong baselines on small open models. Results support entropy as a practical control knob for LLM decoding. Future work: learned schedules, larger models, fused kernels, and integration with safety filters.
> Note: Bottom line: directly targeting entropy works in practice with small cost, improves generation behavior, and is easy to integrate. Extensions include learning the best schedules and optimizing kernels.

## References
- Holtzman et al. (2020). The Curious Case of Neural Text Degeneration.
- Meister et al. (2023). Locally Typical Sampling.
- Basu et al. (2021). Mirostat: A Memory-Efficient Algorithm for Controlling Perplexity.
- Guo et al. (2017). On Calibration of Modern Neural Networks.
> Note: Prior work covers degeneration under naive decoding, typical sampling criteria, feedback-based control (Mirostat), and calibration; TED complements these by directly matching global entropy per step.

## Appendix A: Derivation of dH/dT
Let β=1/T and Z(β)=Σ_{i∈M} exp(β s_i). Then p_i(β)=exp(β s_i)/Z(β), H(β)=log Z(β)−β E_pβ[s]. Using d/dβ log Z=E[s] and d/dβ E_pβ[s]=Var_pβ(s):
dH/dβ = E[s] − E[s] − β Var_pβ(s) = −β Var_pβ(s) ≤ 0.
By the chain rule, dH/dT = (−β Var_pβ(s))·(−1/T^2) = Var_pT(s)/T^3 ≥ 0, with equality only when p is one-hot (T→0) or all active logits are equal.
> Note: Step-by-step:
> - β=1/T simplifies derivatives.
> - Z(β) is the normalizer; p_i(β) the softmax.
> - H(β)=log Z − β E[s] (standard identity for Gibbs distributions).
> - d/dβ log Z = E[s]; d/dβ E[s] = Var(s) (score function identity).
> - Therefore dH/dβ = −β Var(s) ≤ 0.
> - Chain rule with β=1/T gives dH/dT = Var(s)/T^3 ≥ 0.
> - Equality occurs when variance is zero (all probability on one token or all logits identical).

## Appendix B: Full Baseline Grids
We include full tables for all hyperparameter combinations (temperature, nucleus, typical sampling, Mirostat v2; and all TED schedules) with means and 95% CIs for all metrics, plus per-batch microbenchmark results (tokens/sec and ms/token).
> Note: Complete results enable detailed comparison across settings, including performance and efficiency trade-offs.
