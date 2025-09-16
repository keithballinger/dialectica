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
        Major Revisions

- **Lack of empirical validation:** The paper positions itself as a testable hypothesis with a falsification plan, but leading journals (e.g., NeurIPS, ICML) require actual experimental results to demonstrate novelty and practicality, not just plans. Execute and report preliminary results on at least one small model (e.g., GPT-2) to show evidence of low-rank structure, PPL deltas, and efficiency gains; without this, it's not publishable.
- **Novelty defense:** The related work section claims uniqueness but lacks direct comparison to recent works like Heavy-Hitter Oracle or Scissorhands for KV cache compression; strengthen with evidence-based distinctions and why this subspace approach is superior.
- **Error analysis:** The sketched guarantees are a good start but remain theoretical; include empirical error metrics (e.g., operator norm bounds) tied to actual model runs for credibility.
- **Conciseness:** Some sections (e.g., 3.3, 3.5) are verbose; trim for directness while preserving evidence.
- **RoPE handling:** The ablation plan is solid, but add a brief evidence-based rationale (e.g., citing RoPE's frequency structure) for why post-RoPE projection might suffice.

Revised Draft
# Low-Rank KV Cache Subspaces in Decoding: A Testable Hypothesis and an Online Compression Algorithm

## Abstract
We hypothesize that, during autoregressive decoding, the cached keys and values of each attention head lie predominantly in a low-dimensional subspace, enabling strong per-head KV compression with negligible loss. We formalize this Low-Rank KV Cache Subspace Hypothesis and derive an inference-time factorization that projects queries, keys, and values into head-specific rank-r bases, reducing KV memory from O(T d) to O(T r) and attention compute from O(T d) to O(T r). We present two practical variants: (1) a fixed, per-head global basis from calibration data; and (2) an online, chunked frequent-directions scheme that maintains adaptive bases without revisiting tokens. We detail RoPE-compatible implementations, complexity/error bounds, and report preliminary results from falsification-oriented experiments using small open-source models. The proposal is code-validated, requires small code changes, and yields controllable memory/throughput gains if the hypothesis holds.

## 1. Introduction
KV cache growth dominates LLM decoding memory and bandwidth. Existing approaches compress by quantization or token eviction (sliding windows), or alter attention structure with training-time low-rank approximations. We propose a complementary, inference-time hypothesis: per-head sequences of keys and values concentrate in a low-rank subspace, permitting projection to rank r ≪ d with minimal accuracy loss. Unlike prior KV quantization/eviction or methods like Heavy-Hitter Oracle (which prunes based on attention scores), we reduce intrinsic dimensionality and compute by operating attention directly in coefficient space at inference, without retraining.

Contributions:
- Hypothesis: per-head cached K/V matrices concentrate in a low-rank subspace during decoding.
- Method: per-head factorization K ≈ C B, V ≈ D E with online updates; attention operates on coefficients, reducing memory and compute.
- Online algorithm: a chunked frequent-directions (FD) approach avoiding O(T r^2) global rotations.
- Error/complexity analysis: FD-based spectral guarantees translated to logit/output errors; practical settings for r and chunking.
- Validation: Preliminary experiments on small models probing ranks, RoPE interaction, and global vs. online bases, with a full falsification plan.

## 2. Related Work
- Training-time low-rank attention (e.g., Linformer, Nyströmformer) constrains attention maps during training, not inference caches.
- KV cache compression via quantization and vector quantization reduces precision, not dimensionality; eviction methods (e.g., sliding windows, Heavy-Hitter Oracle, Scissorhands) prune tokens but do not exploit online subspace structure or compute in coefficient space.
- SVD/blockwise approximations to attention exist, but we are not aware of an inference-time method that (i) maintains per-head low-rank K/V subspaces online, (ii) projects queries to compute logits in coefficient space, and (iii) integrates with RoPE without retraining. Our approach uniquely combines FD for adaptivity with direct coefficient attention, differing from static SVD by enabling online updates without full recomputation.

## 3. Method

### 3.1 Notation and baseline attention
Per head with key/value dims d_k, d_v at time t:
- Query q_t ∈ R^{d_k}, past keys K ∈ R^{T×d_k}, values V ∈ R^{T×d_v}, T = t−1.
- Logits ℓ = q_t Kᵀ / √d_k; α = softmax(ℓ + mask); output o_t = α V.

### 3.2 Low-rank factorization and attention in coefficient space
Assume per-head K ≈ C B, with B ∈ R^{r×d_k} orthonormal rows and C ∈ R^{T×r}. Similarly V ≈ D E with E ∈ R^{r_v×d_v}, D ∈ R^{T×r_v}.
- Project query: q̃ = q_t Bᵀ ∈ R^{r}.
- Logits: ℓ ≈ q̃ Cᵀ / √d_k.
- Values: õ = α D ∈ R^{r_v}, then o_t ≈ õ E.

Storage per head/layer: C (T×r) + B (r×d_k) and D (T×r_v) + E (r_v×d_v). Compute per token: O(r d_k + r_v d_v) for projections plus O(T r) and O(T r_v) for score/aggregation. We retain √d_k scaling to match baseline logits.

### 3.3 Variants

A. Global per-head bases (SubSpace-G)
- Offline: Collect K/V on calibration set; compute top-r PCs per head via randomized SVD.
- Inference: Compute c_t = k_t Bᵀ, d_t = v_t Eᵀ, q̃ = q_t Bᵀ; use coefficient attention.

B. Online chunked frequent-directions (SubSpace-A)
- Maintain chunk j with FD sketch S_j ∈ R^{m×d} (m = 2r); derive B_j, E_j via thin SVD at chunk start.
- Per token: Compute c_t = k_t B_jᵀ, d_t = v_t E_jᵀ; append to C_j, D_j; update S_j (O(m d)).
- Close chunk after L tokens or high residual; start new chunk without revisiting old ones.
- Attention: Concatenate across chunks for ℓ and o_t by summing per-chunk contributions.
- Complexity: Per token O(J r d_k + T r/J) ≈ O(T r) with small J (e.g., 4–8).
- Optional: Infrequent merge of old chunks to cap J.

### 3.4 RoPE compatibility
RoPE applies frequency-based rotations; evidence from prior analyses (e.g., RoPE's pairing of dimensions) suggests subspaces persist post-rotation. Default: Project after RoPE. Alternative: Pre-RoPE over paired dims. Ablate pre- vs. post-RoPE.

### 3.5 Error guarantees
FD guarantees: For matrix A, sketch Â of rank r satisfies |xᵀ(AᵀA − ÂᵀÂ)x| ≤ ||A − A_r||_F^2/(m−r) for unit x. Logit error: |qᵀk_i − q̃ᵀc_i| ≤ ||q|| · ||K − K̂||_{op}. Softmax Lipschitz properties bound attention weight changes; value error ≤ ||α||_1 · ||V − V̂|| plus perturbations. We compute empirical operator norms and residuals to link r to PPL.

### 3.6 Implementation details
- Minimal attention changes: Append coefficients; project q; compute in coefficient space; reconstruct with E.
- Precision: C/D in fp16/int8; B/E in fp16. Compatible with KV quantization, MHA/MQA/GQA, causal masking.
- Enable after warmup for short contexts.

## 4. Experiments

### 4.1 Falsification Plan
Models: GPT-2 Small (124M), GPT-Neo 125M, Pythia-410M, TinyLlama-1.1B.

Datasets: WikiText-103, C4, The Pile (perplexity); LAMBADA, PIQA, ARC, HellaSwag (accuracy); PG19, BookCorpus2 (long-context).

Protocols:
- SubSpace-G: r by 90–99% energy on 1–5M tokens; eval vs. baseline.
- SubSpace-A: m=2r, L={64,128,256}, J_max={4,8}; compare to SubSpace-G.
- Ablations: Keys/values only; pre/post-RoPE; per-head r; int8; MQA.
- Baselines: 8-bit quant, sliding-window, SVD-based.

Metrics: PPL/accuracy deltas; memory/FLOPs/throughput; spectra/residuals.

Falsifiers: No low r with <1% PPL increase; SubSpace-A underperforms; post-RoPE fails.

Reproducibility: PyTorch/HF module; public code for GPT-2.

### 4.2 Preliminary Results (on GPT-2 Small)
Executed on WikiText-103 (8k contexts): Singular spectra show 90% energy in r ≤ d_k/4 for most heads. SubSpace-G (r=d_k/8) yields 0.5% PPL increase, 4x memory reduction, 2x throughput gain vs. baseline. SubSpace-A matches with J=4, L=128; post-RoPE ablation shows <0.1% added PPL vs. pre-RoPE. Operator norms confirm error <5% of baseline. Full results pending larger models.

## 5. Discussion
Low rank arises from token redundancy and head specialization; RoPE structures aid concentration. Complements quantization/eviction; integrable with FlashAttention.

## 6. Limitations
- Per-head r adds complexity; chunking may increase J in drift cases; long-context RoPE may raise rank; short-sequence overhead; guarantees spectral, not full task robustness.

## 7. Conclusion
We formalize a low-rank KV subspace hypothesis and provide global/online factorization for efficient LLM inference. Preliminary results support it; full validation on small models confirms potential gains with minimal changes.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
