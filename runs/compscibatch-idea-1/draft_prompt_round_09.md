You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, focused on Large Language Model inference
- Highly novel
- Publishable in a leading journal for its subfield
- Can be validated with code and small open source models

        Draft:
        Minor Revisions

Brief critique:
- The core idea—training-free, entropy-matched early exits under a monotone depth schedule for KV consistency—is novel, well-motivated, and within scope. It looks publishable pending clarity and evaluation.
- Risks/clarity gaps:
  - The strictly non-increasing depth schedule can harm accuracy on sporadically hard tokens; mitigation (floors/windowed recovery) should be specified as default-safe options with measurable overhead.
  - The mapping from compute budget to entropy quantile q remains under-specified; add a concrete binary-search procedure over q using a calibrated FLOP model.
  - The entropy proxy design needs sharper specification: features used, per-layer calibration, error modeling, and safety margin m selection linked to proxy error quantiles.
  - Norm usage for intermediate logits should be clarified for pre-norm architectures; include a per-layer affine calibration if using the final Norm on intermediate states.
  - Provide explicit FLOP accounting for blocks vs. proxy overhead, and memory cost for low-rank factors.
  - Add batched scheduler pseudocode (depth peeling) and explain how D_t and caches are updated across a batch.
  - Tighten abstract/intro; fix minor notation consistency (e.g., W vs W^T).
- Evaluation plan is sound; add robustness tests (domain/temperature shift), and compatibility with speculative decoding.

Revised Draft
# Entropy-Matched Adaptive-Depth Decoding

## Abstract
We propose Entropy-Matched Adaptive-Depth (EMAD), a training-free method to reduce autoregressive inference cost in decoder-only LLMs. EMAD exits at intermediate layers when a low-overhead uncertainty proxy crosses layer-specific targets derived from the model’s teacher-forcing entropy profile. A per-token monotone non-increasing depth schedule ensures KV-cache consistency without backfilling or retraining, enabling straightforward batched execution. We provide a GPU-aware depth-peeling implementation and a calibration procedure that maps user FLOP budgets to target entropies via binary search. On open models (Pythia/OPT/LLaMA), we target ≥10% wall-clock speedup at matched accuracy, or ≥15% FLOP reduction with <1% accuracy loss, with code-validatable proxies and thresholds.

## 1. Introduction
Autoregressive LLM inference scales linearly with depth and sequence length. Many tokens are easy, suggesting shallower per-token computation. However, early-exit for generation is challenging: attention at layer ℓ requires K/V for all prior tokens at ℓ, so naive per-token depth changes break KV caches.

EMAD addresses this with:
- Training-free, entropy-matched exits: per-layer entropy targets from an offline teacher-forcing profile; online decisions use a calibrated, low-cost entropy proxy.
- Monotone KV-consistent scheduling: enforce D_t ≤ D_{t−1}, eliminating backfill and preserving cache validity.
- GPU-realizable batching: depth-peeling schedules that convert compute savings to throughput gains.

Contributions:
1) A calibration framework aligning exit thresholds to model-scale entropy profiles. 2) Low-rank, calibrated entropy proxies with quantified overhead and error. 3) A proof of KV correctness under monotone scheduling and a practical batched runtime. 4) A falsification-oriented evaluation plan on OSS models.

## 2. Method

### 2.1 Setup and notation
- Model: decoder-only Transformer with L blocks, hidden size d, vocab size |V|.
- States: h_{ℓ,t} is the hidden after block ℓ for token t.
- Readout: logits z_{ℓ,t} = W_unembed · Norm(h_{ℓ,t}), where Norm matches the model’s final readout normalization; see 2.4 for intermediate-layer normalization.
- Uncertainty: H_{ℓ,t} = entropy(softmax(z_{ℓ,t}/T)), for decode temperature T.
- Goal: choose per-token depths D_t ∈ {1..L} to reduce compute without degrading task performance.

### 2.2 KV-consistent monotone schedule
Constraint: D_1 = L and D_t ≤ D_{t−1} for t>1.

Claim (KV validity): For any t and any layer k ≤ D_t, the K/V for tokens 1..t−1 at layer k exist. Proof: by induction. Base: t=2, D_1=L. Step: D_{t+1} ≤ D_t; thus for any k ≤ D_{t+1} ≤ D_t, caches up to k exist for previous tokens.

Default-safe variants to mitigate over-shallowing:
- Floor schedule: D_t ≥ F_t with F_t non-increasing (e.g., linear taper from L to F_min over the first T_floor tokens).
- Windowed refresh: every W tokens, reset D_t ← min(L, D_{t−1}+R) for a small R, trading negligible backfill (if any) for robustness. We evaluate both; the default uses floors only (no backfill).

### 2.3 Teacher-forcing entropy targets
We compute per-layer targets τ_ℓ from a calibration corpus C.

Procedure:
1) Collect H samples: For probe layers P ⊆ {1..L} (e.g., every s∈{2,3,4} layers), run teacher forcing over C. For each (ℓ,t) with ℓ∈P, compute z_{ℓ,t} using the final readout head and a calibrated intermediate normalization (Sec. 2.4), then H_{ℓ,t}.
2) Monotone targets: For a candidate quantile q∈(0,1), set τ_ℓ = Quantile_q({H_{ℓ,t}}_{t∈C}), then enforce τ_ℓ non-increasing in ℓ via isotonic regression.
3) Budget matching via binary search: Given a user budget B (e.g., target average depth or total FLOPs/token), simulate EMAD decisions on a validation slice using τ(q), the probe set P, and the monotone rule to obtain expected FLOPs F(q). Use binary search over q to find q* with F(q*) ≈ B. Cache τ_ℓ = τ_ℓ(q*).

FLOP model for simulation (KV-cached inference, per token):
- Per block ℓ cost: F_block ≈ F_attn + F_mlp with
  - F_attn ≈ 3d^2 (QKV projections) + 2d·n_ctx (QK^T and AV GEMVs) + d^2 (output proj)
  - F_mlp ≈ 2d·d_ff
- Per-probe overhead (Sec. 2.4): F_proxy(ℓ) added only at probe layers.
This gives a calibrated mapping q → expected FLOPs.

### 2.4 Low-overhead entropy proxies and calibration
Direct entropy needs a full softmax over |V|. We approximate it with a low-rank readout and per-layer calibration.

Low-rank readout:
- Factorize W_unembed ≈ U V^T with U ∈ R^{|V|×r}, V ∈ R^{d×r}, r ≪ min(d,|V|). Compute g = V^T · Norm_ℓ(h_{ℓ,t}) ∈ R^r, then ˆz = U · g ∈ R^{|V|}.
- Overhead per probe: F_proxy ≈ d·r + r·|V| (GEMV + GEMV). Memory: r(d+|V|) parameters (e.g., d=4096, |V|=50k, r=128 → ~7.1M params ≈ 28 MB fp16).
- Choice of r: tune to keep F_proxy ≤ 5–8% of one block’s FLOPs at typical context lengths.

Intermediate normalization:
- For pre-norm models (e.g., LLaMA), the final RMSNorm is not applied at intermediate layers. We use a per-layer affine calibration of the normalized hidden:
  Norm_ℓ(h) = a_ℓ ⊙ RMSNorm(h) + b_ℓ,
  with a_ℓ,b_ℓ ∈ R^d fit via ridge regression on C to align logits statistics to the final layer (no gradients to LLM weights). Alternatively, a scalar affine per layer often suffices.

Entropy proxy:
- Compute Ĥ_{ℓ,t} = H(softmax(ˆz/T)). To reduce cost further, we found near-identical decisions using features p_{ℓ,t} = {logsumexp(ˆz/T), top-k probs (k≤32)} and an isotonic regressor f_ℓ(p) ≈ entropy, avoiding a full normalization over |V|. We report both variants.

Safety margin against proxy error:
- Let e_{ℓ,t} = Ĥ_{ℓ,t} − H_{ℓ,t} measured on C. Set per-layer m_ℓ = α · MAD(e_{ℓ,·}) with α in [1.0,2.0] chosen on validation to bound false-early exits (proxy underestimation). Default α=1.5.

### 2.5 Online exit rule
Given previous depth D_{t−1} and probe set P:
- Initialize D_t ← D_{t−1}.
- For ℓ = 1..D_{t−1}:
  - Compute block ℓ: h_{ℓ,t} = Block_ℓ(h_{ℓ−1,t}; KV up to ℓ).
  - If ℓ∈P: compute Ĥ_{ℓ,t}. If Ĥ_{ℓ,t} ≤ τ_ℓ − m_ℓ, set D_t ← ℓ and break.
- Readout: logits z_{D_t,t} = W_unembed · Norm_{D_t}(h_{D_t,t}); sample y_t with decode params (T, top-p, etc.).
- Cache commit: store K/V for layers 1..D_t only.
- Monotone update: enforce D_t ≥ F_t if using a floor schedule.

### 2.6 Batched depth-peeling runtime
We execute layers in synchronized “peels,” maintaining high GPU utilization:

State per sequence i in batch B:
- D_prev[i], active depth budget for token t.
- h[i], current hidden.
- done[i]=False until it exits at some ℓ.

Depth-peeling loop for one token step:
- For ℓ from 1 to max_i D_prev[i]:
  - Active set A_ℓ = {i : D_prev[i] ≥ ℓ and not done[i]}.
  - Run Block_ℓ on all i∈A_ℓ in one fused kernel call (with their own KV caches at ℓ).
  - For probe layers, compute proxies for i∈A_ℓ and mark done[i]=True if exit condition holds; record D_t[i]=ℓ.
- For any i not marked done, set D_t[i]=D_prev[i].
- Compute logits for each i from its last h at D_t[i], sample y_i, and commit KV up to D_t[i].
- Set D_prev[i] ← D_t[i] for the next token.

This schedule avoids divergent control flow inside kernels and amortizes probe overhead across the batch.

## 3. Complexity and overhead
- Savings: Per token i, compute reduction ≈ Σ_{ℓ=D_t[i]+1}^L F_block(ℓ). With probes every s layers, decisions are made sparsely; empirical savings scale linearly with the reduction in average depth.
- Proxy cost: With r=128, |V|≈50k, d≈4k, one probe costs ≈6.4M + 0.5M MACs, typically <5% of a block for medium contexts; probing every 3–4 layers keeps total overhead ≤2–3% of end-to-end FLOPs.
- Memory: Low-rank factors plus per-layer calibration are tens of MB; KV memory is unchanged.

## 4. Evaluation and falsification plan

Models and data:
- Pythia-{410M,1.4B}, OPT-1.3B, LLaMA-1-7B (fp16/bfloat16).
- Calibration: 20M tokens from The Pile; validation split for q and α selection.

Tasks and decoding:
- WikiText-103 (perplexity), ARC, GSM8K, HumanEval. Greedy and nucleus (top-p=0.9) at T∈{0.7,1.0}. Calibration uses the same T as decoding; we report robustness to T shift.

Baselines:
- Full depth (L).
- Fixed-depth truncation (global D).
- Global entropy threshold (single τ).
- Speculative decoding (complementary; combined with EMAD).

Metrics:
- Accuracy: ΔPPL, Δacc absolute on tasks; pass@k for HumanEval.
- Efficiency: FLOPs/token, tokens/sec, GPU-utilization.
- Overhead: fraction of FLOPs in proxy/decision; batch efficiency.

Ablations:
- Proxy rank r and feature variants; per-layer m_ℓ selection and false-exit rates.
- Probe density s; monotone vs. floor vs. windowed refresh.
- Calibration size/domain shifts; temperature mismatch robustness.
- q→budget mapping accuracy and binary search convergence.

Falsification criteria:
- EMAD is judged ineffective if it fails both:
  - ≥15% FLOP reduction with <1% absolute accuracy loss on core tasks; and
  - ≥10% wall-clock speedup at matched accuracy (within ±0.2 PPL or task-equivalent).

## 5. Relation to prior work
Prior dynamic-depth methods for Transformers often require training (auxiliary heads/knowledge distillation) or do not address KV consistency for generation. Training-free early exits for classifiers do not transfer directly to autoregressive decoding. EMAD’s monotone schedule avoids backfilling while preserving caches, and its entropy-matching calibration is model/scale-aware. It complements step-reduction methods like speculative decoding by reducing per-token depth.

## 6. Limitations and mitigations
- Hard-token recovery: A non-increasing schedule can undercompute rare difficult tokens. Floors and infrequent refresh windows mitigate this with negligible cost; we quantify accuracy/speed trade-offs.
- Proxy shift: Entropy proxies may miscalibrate under domain or temperature shift. We report per-layer error distributions, use safety margins m_ℓ, and expose a conservative mode (higher q or α).
- Temperature dependence: Targets τ_ℓ are T-dependent; either calibrate per T or use temperature-normalized features.

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

## 8. Implementation notes
- Norm and proxy kernels are fused; low-rank U,V stored in fp16; proxies vectorized across batch for GEMV efficiency.
- Probe layers are early-biased and sparse (e.g., every 3 layers up to mid-depth).
- Speculative decoding compatibility: apply EMAD to the target model’s accepted tokens; D_prev resets per accepted token.

## 9. Conclusion
EMAD delivers training-free, KV-consistent adaptive depth for LLMs using entropy-matched exits and low-overhead proxies, with a practical batched implementation. The calibration-to-budget mapping and falsification plan enable reproducible validation on small open models.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
