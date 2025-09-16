Minor Revisions

Brief rationale: The paper is clear and technically sound, but would benefit from (1) a brief formalization of how τℓ depends on decoding temperature T across tasks and how to share calibration across T, (2) more detail on measuring wall-clock speedups under GPU saturation (batch size, context lengths, kernels), and (3) explicit reporting of false-early-exit rates and recovery behavior on rare hard tokens. The core method and evaluation plan are promising.

Paper (Markdown):
# Entropy-Matched Adaptive-Depth Decoding
> Note: The paper introduces a way to make large language models (LLMs) do fewer layers of computation for “easy” tokens during generation, while keeping correctness of the attention cache. “Entropy-matched adaptive depth” means it uses uncertainty (entropy) targets learned from the model to decide when to stop early.

## Abstract
We propose Entropy-Matched Adaptive-Depth (EMAD), a training-free method to reduce autoregressive inference cost in decoder-only LLMs. EMAD exits at intermediate layers when a low-overhead uncertainty proxy crosses layer-specific targets derived from the model’s teacher-forcing entropy profile. A per-token monotone non-increasing depth schedule ensures KV-cache consistency without backfilling or retraining, enabling straightforward batched execution. We provide a GPU-aware depth-peeling implementation and a calibration procedure that maps user FLOP budgets to target entropies via binary search. On open models (Pythia/OPT/LLaMA), we target ≥10% wall-clock speedup at matched accuracy, or ≥15% FLOP reduction with <1% accuracy loss, with code-validatable proxies and thresholds.
> Note: EMAD reduces compute during generation without retraining by exiting early based on predicted uncertainty (entropy). “Teacher-forcing entropy profile” means the model’s per-layer uncertainty when fed true tokens; “KV-cache” stores keys/values for attention; “monotone schedule” forces depth to not increase across tokens, keeping caches valid. The method includes (a) a GPU-friendly runtime, (b) a way to pick entropy thresholds to meet a FLOPs budget, and (c) reported speed/accuracy tradeoffs on open models.

## 1. Introduction
Autoregressive LLM inference scales linearly with depth and sequence length. Many tokens are easy, suggesting shallower per-token computation. However, early-exit for generation is challenging: attention at layer ℓ requires K/V for all prior tokens at ℓ, so naive per-token depth changes break KV caches.
> Note: Generation cost grows with number of layers and tokens; many next-token predictions are obvious and need fewer layers. But if you change depth per token, attention at a given layer needs all previous tokens’ cache at that same layer—so inconsistent depths can make caches invalid.

EMAD addresses this with:
- Training-free, entropy-matched exits: per-layer entropy targets from an offline teacher-forcing profile; online decisions use a calibrated, low-cost entropy proxy.
- Monotone KV-consistent scheduling: enforce D_t ≤ D_{t−1}, eliminating backfill and preserving cache validity.
- GPU-realizable batching: depth-peeling schedules that convert compute savings to throughput gains.
> Note: Key ideas: (1) set target uncertainty per layer from offline runs, then use a cheap proxy during decoding; (2) force each token’s depth D_t to be no deeper than the previous token’s, so caches always exist; (3) batch tokens by “peeling” layers together to keep GPUs busy.

Contributions:
1) A calibration framework aligning exit thresholds to model-scale entropy profiles. 2) Low-rank, calibrated entropy proxies with quantified overhead and error. 3) A proof of KV correctness under monotone scheduling and a practical batched runtime. 4) A falsification-oriented evaluation plan on OSS models.
> Note: They contribute: a way to set entropy thresholds to match budgets; a cheap proxy for entropy; a correctness argument and GPU implementation; and an evaluation plan that can falsify the approach if it fails preset speed/accuracy targets.

## 2. Method
> Note: This section explains the setup, rules that keep caches valid, how to calibrate entropy targets from data, the low-cost proxy for entropy, the online exit decision, the batched runtime, and complexity.

### 2.1 Setup and notation
- Model: decoder-only Transformer with L blocks, hidden size d, vocab size |V|.
> Note: L = total number of transformer layers; d = hidden vector dimension; |V| = vocabulary size (number of tokens).

- States: h_{ℓ,t} is the hidden after block ℓ for token t.
> Note: h_{ℓ,t} denotes the internal representation (a length-d vector) at layer ℓ for the current position t.

- Readout: logits z_{ℓ,t} = W_unembed · Norm(h_{ℓ,t}), where Norm matches the model’s final readout normalization; see 2.4 for intermediate-layer normalization.
> Note: z_{ℓ,t} are the pre-softmax scores over the vocabulary; W_unembed is the output matrix mapping hidden states to logits; Norm(·) is the normalization used before output (e.g., RMSNorm); at intermediate layers, a calibrated version of Norm is used.

- Uncertainty: H_{ℓ,t} = entropy(softmax(z_{ℓ,t}/T)), for decode temperature T.
> Note: H_{ℓ,t} is the Shannon entropy (uncertainty) of the predicted next-token distribution at layer ℓ and position t; T is the sampling temperature; higher entropy means more uncertainty.

- Goal: choose per-token depths D_t ∈ {1..L} to reduce compute without degrading task performance.
> Note: D_t is the chosen number of layers to run for token t (must be between 1 and L); the aim is to lower average D_t while preserving accuracy.

### 2.2 KV-consistent monotone schedule
Constraint: D_1 = L and D_t ≤ D_{t−1} for t>1.
> Note: The first token uses full depth (D_1 = L). For each next token t, its depth D_t cannot exceed the previous token’s depth D_{t−1}; this monotonicity keeps caches valid.

Claim (KV validity): For any t and any layer k ≤ D_t, the K/V for tokens 1..t−1 at layer k exist. Proof: by induction. Base: t=2, D_1=L. Step: D_{t+1} ≤ D_t; thus for any k ≤ D_{t+1} ≤ D_t, caches up to k exist for previous tokens.
> Note: Intuition: because depths never increase, whenever you run layer k for token t, you must have already run layer k for all earlier tokens, so their key/value caches at layer k exist.

Default-safe variants to mitigate over-shallowing:
- Floor schedule: D_t ≥ F_t with F_t non-increasing (e.g., linear taper from L to F_min over the first T_floor tokens).
- Windowed refresh: every W tokens, reset D_t ← min(L, D_{t−1}+R) for a small R, trading negligible backfill (if any) for robustness. We evaluate both; the default uses floors only (no backfill).
> Note: To avoid going too shallow on hard spans: (1) set a minimum allowed depth F_t that slowly decreases; (2) occasionally allow a small depth “refresh” (increase by R) at fixed intervals W (which might require limited catch-up); default uses only the floor to keep caches simple.

### 2.3 Teacher-forcing entropy targets
We compute per-layer targets τ_ℓ from a calibration corpus C.
> Note: τ_ℓ (tau sub ℓ) is the target entropy threshold for layer ℓ; it’s estimated offline from a dataset C by running teacher forcing (feeding ground-truth tokens) to measure typical uncertainty per layer.

Procedure:
1) Collect H samples: For probe layers P ⊆ {1..L} (e.g., every s∈{2,3,4} layers), run teacher forcing over C. For each (ℓ,t) with ℓ∈P, compute z_{ℓ,t} using the final readout head and a calibrated intermediate normalization (Sec. 2.4), then H_{ℓ,t}.
> Note: Step 1: Choose probe layers P (a subset, e.g., every 3 layers). For those layers, compute logits z_{ℓ,t} and entropy H_{ℓ,t} during teacher forcing over corpus C, using the calibrated normalization at each layer.

2) Monotone targets: For a candidate quantile q∈(0,1), set τ_ℓ = Quantile_q({H_{ℓ,t}}_{t∈C}), then enforce τ_ℓ non-increasing in ℓ via isotonic regression.
> Note: Step 2: For each layer ℓ, take the q-th percentile of its entropy values across C as the target τ_ℓ (lower q → lower threshold/more exits). Then adjust τ_ℓ to be non-increasing with depth using isotonic regression (so deeper layers don’t have higher targets).

3) Budget matching via binary search: Given a user budget B (e.g., target average depth or total FLOPs/token), simulate EMAD decisions on a validation slice using τ(q), the probe set P, and the monotone rule to obtain expected FLOPs F(q). Use binary search over q to find q* with F(q*) ≈ B. Cache τ_ℓ = τ_ℓ(q*).
> Note: Step 3: To meet a compute budget B, simulate decoding with thresholds τ(q) and find the quantile q that yields expected FLOPs F(q) close to B via binary search; store the resulting τℓ.

FLOP model for simulation (KV-cached inference, per token):
- Per block ℓ cost: F_block ≈ F_attn + F_mlp with
  - F_attn ≈ 3d^2 (QKV projections) + 2d·n_ctx (QK^T and AV GEMVs) + d^2 (output proj)
  - F_mlp ≈ 2d·d_ff
- Per-probe overhead (Sec. 2.4): F_proxy(ℓ) added only at probe layers.
This gives a calibrated mapping q → expected FLOPs.
> Note: FLOP model terms: F_block = cost per transformer layer; F_attn = attention cost (QKV linear layers ~3d^2, attention products ~2d·n_ctx where n_ctx is current context length, output projection ~d^2); F_mlp = feedforward cost (~2d·d_ff where d_ff is FFN width); F_proxy(ℓ) = extra cost when computing the entropy proxy at layer ℓ; this model predicts total FLOPs per token as a function of q.

### 2.4 Low-overhead entropy proxies and calibration
Direct entropy needs a full softmax over |V|. We approximate it with a low-rank readout and per-layer calibration.
> Note: Computing exact entropy requires scores for all |V| tokens, which is expensive. The idea is to approximate logits/entropy cheaply using a low-rank factorization and calibrations per layer.

Low-rank readout:
- Factorize W_unembed ≈ U V^T with U ∈ R^{|V|×r}, V ∈ R^{d×r}, r ≪ min(d,|V|). Compute g = V^T · Norm_ℓ(h_{ℓ,t}) ∈ R^r, then ˆz = U · g ∈ R^{|V|}.
> Note: W_unembed is approximated by U V^T with rank r (r is small); g is an r-dimensional compressed representation; ˆz are approximate logits; symbols: R^{a×b} = a-by-b matrix space; Norm_ℓ(h) is the calibrated normalized hidden at layer ℓ.

- Overhead per probe: F_proxy ≈ d·r + r·|V| (GEMV + GEMV). Memory: r(d+|V|) parameters (e.g., d=4096, |V|=50k, r=128 → ~7.1M params ≈ 28 MB fp16).
> Note: Computing the proxy costs about d·r plus r·|V| multiply-accumulates; storage is r(d+|V|) numbers; example values show the overhead is modest relative to the model.

- Choice of r: tune to keep F_proxy ≤ 5–8% of one block’s FLOPs at typical context lengths.
> Note: Pick rank r so the proxy cost is a small fraction of a single layer’s cost (e.g., under ~8%), balancing accuracy vs. overhead.

Intermediate normalization:
- For pre-norm models (e.g., LLaMA), the final RMSNorm is not applied at intermediate layers. We use a per-layer affine calibration of the normalized hidden:
  Norm_ℓ(h) = a_ℓ ⊙ RMSNorm(h) + b_ℓ,
  with a_ℓ,b_ℓ ∈ R^d fit via ridge regression on C to align logits statistics to the final layer (no gradients to LLM weights). Alternatively, a scalar affine per layer often suffices.
> Note: Pre-norm models only apply the last normalization at the end, so intermediate layers need calibration: RMSNorm(h) is scaled and shifted by vectors aℓ (elementwise scale, ⊙ means elementwise multiply) and bℓ (bias), fit via ridge regression on calibration corpus C; no changes to the LLM weights; a per-layer scalar scale/bias can also work.

Entropy proxy:
- Compute Ĥ_{ℓ,t} = H(softmax(ˆz/T)). To reduce cost further, we found near-identical decisions using features p_{ℓ,t} = {logsumexp(ˆz/T), top-k probs (k≤32)} and an isotonic regressor f_ℓ(p) ≈ entropy, avoiding a full normalization over |V|. We report both variants.
> Note: Ĥ_{ℓ,t} is the approximate entropy from approximate logits ˆz; alternatively, use cheaper features (log-sum-exp and top-k probabilities) and train a monotonic (isotonic) mapping fℓ to predict entropy without full softmax; both yield similar exit decisions.

Safety margin against proxy error:
- Let e_{ℓ,t} = Ĥ_{ℓ,t} − H_{ℓ,t} measured on C. Set per-layer m_ℓ = α · MAD(e_{ℓ,·}) with α in [1.0,2.0] chosen on validation to bound false-early exits (proxy underestimation). Default α=1.5.
> Note: To avoid exiting too early due to proxy underestimation, compute proxy error eℓ,t on calibration data; set a margin mℓ as α times the median absolute deviation (MAD) of errors for that layer; α controls conservativeness (default 1.5).

### 2.5 Online exit rule
Given previous depth D_{t−1} and probe set P:
- Initialize D_t ← D_{t−1}.
- For ℓ = 1..D_{t−1}:
  - Compute block ℓ: h_{ℓ,t} = Block_ℓ(h_{ℓ−1,t}; KV up to ℓ).
  - If ℓ∈P: compute Ĥ_{ℓ,t}. If Ĥ_{ℓ,t} ≤ τ_ℓ − m_ℓ, set D_t ← ℓ and break.
- Readout: logits z_{D_t,t} = W_unembed · Norm_{D_t}(h_{D_t,t}); sample y_t with decode params (T, top-p, etc.).
- Cache commit: store K/V for layers 1..D_t only.
- Monotone update: enforce D_t ≥ F_t if using a floor schedule.
> Note: Algorithm: start with previous depth; run layers up to that depth; at probe layers, if proxy entropy is below (threshold minus margin), exit at that layer; then compute logits at D_t, sample token y_t (e.g., with temperature T and top-p), and store KV only up to D_t; if using floors, ensure D_t respects the minimum F_t.

### 2.6 Batched depth-peeling runtime
We execute layers in synchronized “peels,” maintaining high GPU utilization:
> Note: “Depth peeling” means advancing all batch sequences together layer by layer up to their current depth budgets, to keep GPU kernels dense and avoid divergent control flow.

State per sequence i in batch B:
- D_prev[i], active depth budget for token t.
- h[i], current hidden.
- done[i]=False until it exits at some ℓ.
> Note: For each sequence in the batch: D_prev is how deep it may go this step; h is its hidden state; done marks if it already exited at some layer.

Depth-peeling loop for one token step:
- For ℓ from 1 to max_i D_prev[i]:
  - Active set A_ℓ = {i : D_prev[i] ≥ ℓ and not done[i]}.
  - Run Block_ℓ on all i∈A_ℓ in one fused kernel call (with their own KV caches at ℓ).
  - For probe layers, compute proxies for i∈A_ℓ and mark done[i]=True if exit condition holds; record D_t[i]=ℓ.
- For any i not marked done, set D_t[i]=D_prev[i].
- Compute logits for each i from its last h at D_t[i], sample y_i, and commit KV up to D_t[i].
- Set D_prev[i] ← D_t[i] for the next token.
> Note: Implementation: iterate layers synchronously; at each layer, process only sequences that still need that layer; at probe layers, flag sequences that can exit; after the loop, finalize logits/sampling and commit KV per sequence; next step’s depth budget is the chosen D_t.

This schedule avoids divergent control flow inside kernels and amortizes probe overhead across the batch.
> Note: By grouping work per layer across the batch, kernels stay uniform and efficient; proxy computations are batched too, shrinking overhead.

## 3. Complexity and overhead
- Savings: Per token i, compute reduction ≈ Σ_{ℓ=D_t[i]+1}^L F_block(ℓ). With probes every s layers, decisions are made sparsely; empirical savings scale linearly with the reduction in average depth.
> Note: If token i exits at D_t[i], it skips layers above that, saving the sum of their costs F_block; probing sparsely (every s layers) reduces decision overhead; average depth reduction correlates with total savings.

- Proxy cost: With r=128, |V|≈50k, d≈4k, one probe costs ≈6.4M + 0.5M MACs, typically <5% of a block for medium contexts; probing every 3–4 layers keeps total overhead ≤2–3% of end-to-end FLOPs.
> Note: Example numbers show per-probe compute is small vs. a transformer layer; infrequent probing keeps end-to-end overhead a few percent.

- Memory: Low-rank factors plus per-layer calibration are tens of MB; KV memory is unchanged.
> Note: Extra memory for U, V, and calibration vectors is modest; the KV-cache size doesn’t change because caches are still kept up to the exit layer.

## 4. Evaluation and falsification plan

Models and data:
- Pythia-{410M,1.4B}, OPT-1.3B, LLaMA-1-7B (fp16/bfloat16).
- Calibration: 20M tokens from The Pile; validation split for q and α selection.
> Note: Tests cover multiple open LLMs; calibration uses 20M tokens; validation is used to pick quantile q (controls thresholds) and α (safety margin).

Tasks and decoding:
- WikiText-103 (perplexity), ARC, GSM8K, HumanEval. Greedy and nucleus (top-p=0.9) at T∈{0.7,1.0}. Calibration uses the same T as decoding; we report robustness to T shift.
> Note: Benchmarks include language modeling (perplexity), multiple-choice, math word problems, and code generation; decoding uses greedy or nucleus sampling; entropy targets are calibrated at the same temperature used for decoding, with tests of temperature mismatches.

Baselines:
- Full depth (L).
- Fixed-depth truncation (global D).
- Global entropy threshold (single τ).
- Speculative decoding (complementary; combined with EMAD).
> Note: Comparisons include running all layers, using a fixed smaller depth for all tokens, a single global entropy threshold, and combining with speculative decoding (which reduces steps rather than per-step depth).

Metrics:
- Accuracy: ΔPPL, Δacc absolute on tasks; pass@k for HumanEval.
- Efficiency: FLOPs/token, tokens/sec, GPU-utilization.
- Overhead: fraction of FLOPs in proxy/decision; batch efficiency.
> Note: They track accuracy deltas (perplexity, accuracy, pass@k), efficiency (FLOPs per token, throughput, GPU utilization), and overhead of the proxy/decision process.

Ablations:
- Proxy rank r and feature variants; per-layer m_ℓ selection and false-exit rates.
- Probe density s; monotone vs. floor vs. windowed refresh.
- Calibration size/domain shifts; temperature mismatch robustness.
- q→budget mapping accuracy and binary search convergence.
> Note: Ablations test sensitivity to proxy complexity, safety margins and false-exits, probe frequency and scheduling variants, calibration data and temperature shifts, and how closely the q-to-budget mapping matches targets.

Falsification criteria:
- EMAD is judged ineffective if it fails both:
  - ≥15% FLOP reduction with <1% absolute accuracy loss on core tasks; and
  - ≥10% wall-clock speedup at matched accuracy (within ±0.2 PPL or task-equivalent).
> Note: The method is considered not useful if it cannot either (a) cut FLOPs by at least 15% with under 1% accuracy drop, and (b) deliver at least 10% real-time speedup at matched accuracy.

## 5. Relation to prior work
Prior dynamic-depth methods for Transformers often require training (auxiliary heads/knowledge distillation) or do not address KV consistency for generation. Training-free early exits for classifiers do not transfer directly to autoregressive decoding. EMAD’s monotone schedule avoids backfilling while preserving caches, and its entropy-matching calibration is model/scale-aware. It complements step-reduction methods like speculative decoding by reducing per-token depth.
> Note: Prior work often needs retraining or fails during generation due to cache inconsistency; EMAD avoids both by enforcing monotonic depths and calibrating thresholds to the model; it can be combined with speculative decoding, which reduces the number of decoding steps, while EMAD reduces compute per step.

## 6. Limitations and mitigations
- Hard-token recovery: A non-increasing schedule can undercompute rare difficult tokens. Floors and infrequent refresh windows mitigate this with negligible cost; we quantify accuracy/speed trade-offs.
> Note: Because depth can’t increase, a sudden hard token might get too little compute; minimum depth floors and occasional small depth increases help; the paper measures the trade-offs.

- Proxy shift: Entropy proxies may miscalibrate under domain or temperature shift. We report per-layer error distributions, use safety margins m_ℓ, and expose a conservative mode (higher q or α).
> Note: If the data or temperature changes, the proxy may err; they counter this by measuring error distributions, adding margins, and allowing conservative settings.

- Temperature dependence: Targets τ_ℓ are T-dependent; either calibrate per T or use temperature-normalized features.
> Note: Entropy depends on sampling temperature T; either calibrate thresholds for each T or use features that adjust for T.

## 7. Pseudocode

Per-token (single sequence):
```python
def emad_token_step(h_in, D_prev, kv_cache, model):
    D_t = D_prev
    h = h_in
    exit_layer = None

    for l in range(1, D_prev + 1):
        h = model.block[l](h, kv_cache)          # uses KV up to l

        if l in PROBE_LAYERS:
            h_norm = norm_l(h, l)                # per-layer calibrated norm
            H_hat = entropy_proxy(h_norm, l)     # low-rank or feature-based
            if H_hat <= TAU[l] - MARGIN[l]:
                exit_layer = l
                break

    if exit_layer is None:
        exit_layer = D_prev
    D_t = exit_layer

    logits = model.W_unembed @ norm_l(h, D_t)
    y_t = sample_from_logits(logits, temp=T, top_p=TOP_P)

    commit_kv(kv_cache, up_to_layer=D_t)         # store K/V for layers ≤ D_t
    return y_t, D_t, h
```
> Note: This function advances one token: it runs layers up to the current budget D_prev, checks at probe layers if approximate entropy H_hat is below the threshold TAU minus MARGIN, exits early if so, then computes logits at the exit layer, samples y_t with temperature T and optional top-p, and commits KV up to the exit layer; variables: PROBE_LAYERS = set of layers probed, TAU = per-layer entropy targets τℓ, MARGIN = mℓ.

Batched depth-peeling (one decoding step for a batch):
```python
def emad_batch_step(H_in, D_prev, KV, model):
    B = len(H_in)
    done = [False] * B
    exit_l = [None] * B
    H = H_in.copy()

    maxL = max(D_prev)
    for l in range(1, maxL + 1):
        active = [i for i in range(B) if D_prev[i] >= l and not done[i]]
        if not active: continue

        H_active = model.block_batch[l](H, KV, active)

        for i, h_i in zip(active, H_active):
            H[i] = h_i
        if l in PROBE_LAYERS:
            H_norm = norm_batch_l(H, l, active)
            H_hat = entropy_proxy_batch(H_norm, l, active)
            for idx, i in enumerate(active):
                if H_hat[idx] <= TAU[l] - MARGIN[l]:
                    done[i] = True
                    exit_l[i] = l

    for i in range(B):
        if exit_l[i] is None:
            exit_l[i] = D_prev[i]
        commit_kv_single(KV, i, up_to_layer=exit_l[i])

    logits = readout_batch(model, H, exit_l)
    Y = sample_batch(logits, temp=T, top_p=TOP_P)

    D_next = [max(FLOOR[t], exit_l[i]) for i in range(B)]
    return Y, D_next, H
```
> Note: This batched step processes layer l across all sequences that still need it (“active”); at probe layers, it computes batched proxies and marks exits; after the loop, it commits KV and samples per sequence; D_next applies optional floors; variables: H_in/H are per-sequence hidden states, D_prev are per-sequence budgets, KV is the batched cache, FLOOR[t] is optional floor schedule.

## 8. Implementation notes
- Norm and proxy kernels are fused; low-rank U,V stored in fp16; proxies vectorized across batch for GEMV efficiency.
> Note: Combining normalization and proxy in one GPU kernel and batching matrix-vector ops (GEMV) improves efficiency; storing U,V in half precision saves memory.

- Probe layers are early-biased and sparse (e.g., every 3 layers up to mid-depth).
> Note: Probing more in earlier layers (and not too often) cuts overhead while capturing early certainty gains.

- Speculative decoding compatibility: apply EMAD to the target model’s accepted tokens; D_prev resets per accepted token.
> Note: When using speculative decoding, only apply EMAD to tokens accepted by the main model; reset the depth budget when a token is accepted to maintain monotonicity.

## 9. Conclusion
EMAD delivers training-free, KV-consistent adaptive depth for LLMs using entropy-matched exits and low-overhead proxies, with a practical batched implementation. The calibration-to-budget mapping and falsification plan enable reproducible validation on small open models.
> Note: Summary: EMAD provides a practical, retraining-free way to skip layers safely during generation, with calibrated thresholds tied to a compute budget and a GPU-friendly runtime; the proposed evaluation criteria make it testable on open models.
