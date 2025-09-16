You are a truth-seeking scientific collaborator. You have no ego. You are not sycophantic. Be concise, direct, and evidence-based. Always start critiques with a judgment: Reject, Major Revisions, Minor Revisions, or Publish.
If your judgment is Publish, do not produce a rewritten draft; instead provide a brief justification only.


        Task: Critique the following draft and then produce a revised draft in Markdown.
        Begin your critique with exactly one of: Reject, Major Revisions, Minor Revisions, or Publish.

        Constraints of Paper:
        From: constraints/compsci.md

- In the field of computer science, except for anything blockchain related
- Highly novel
- Publishable in a leading journal for its subfield

        Draft:
        Major Revisions

Brief critique
- Claims hinge on a brittle, thresholded out-degree metric (z via τ) that is not invariant to attention temperature, head scaling, or dataset. Replace with a threshold-free, weighted criterion (e.g., participation-ratio effective degree and a spectral percolation threshold based on the non-backtracking or message-passing operator).
- The theoretical core needs formalization for directed, layered, input-dependent graphs. A branching-process or message-passing derivation with a spectral-radius condition is needed; “z = 1” is only exact for specific ensembles and does not address head unions, residual paths, or MLP mixing.
- Path existence is insufficient to guarantee information transmission under softmax normalization and interference across heads/values. Include an information-capacity argument (e.g., min-cut bounds; participation ratio as capacity proxy) that justifies ICL robustness from edge-disjoint routes.
- Experiments are currently asserted but not substantiated. Replace “initial results” with a preregistered empirical plan, include finite-size scaling analyses, robustness to τ/temperature, and controls for parameter count and compute.
- Clarify terminology and notation (branching factor vs out-degree; per-layer vs multi-layer graphs; union of heads vs mixture). Provide clear operational definitions and reproducible estimators.
- Expand related work and position against alternative explanations (meta-learning dynamics, induction heads, implicit gradient descent), and discuss confounders (residual stream, MLP mixing, content-dependent masks).
- Add finite-size scaling predictions and statistical tests (change-point detection, scaling collapse) to substantiate “phase transition” claims.

Revised Draft
# A Directed-Percolation Theory of In-Context Learning: Spectral Thresholds and Low-Stretch Routing in Transformers

## Abstract
We propose a structural theory of in-context learning (ICL) in Transformers based on directed percolation in the attention-induced communication graph. We derive two quantitative, falsifiable conditions for robust ICL: (i) a supercritical spectral reproduction condition R > 1 defined by the non-backtracking/message-passing operator of the multi-head, per-layer attention graph; and (ii) a low-stretch routing condition requiring that typical prompt-to-target shortest paths are traversable within model depth L. Unlike thresholded out-degree heuristics, our criterion is threshold-free and accounts for weighted, input-dependent attention and head unions. We provide measurable proxies (participation-ratio effective degree, spectral radius estimates, path-length distributions, and min-cut capacity bounds) and preregistered experiments with sparse-attention Transformers designed to elicit finite-size scaling near criticality. The theory unifies reports of abrupt ICL emergence, head-count effects, and the efficacy of small-world sparsity, and yields concrete ablations that can falsify it.

## 1. Introduction
Transformers often solve new tasks specified in prompts without parameter updates, a phenomenon termed in-context learning (ICL). Despite extensive empirical work, we lack structural conditions predicting when ICL emerges or collapses as architectural connectivity, depth, and sparsity vary.

We cast attention as an input-dependent, directed communication graph across layers and posit that ICL is limited by two constraints:
- a supercritical connectivity condition ensuring depth-spanning communication (directed percolation), and
- a low-stretch routing condition ensuring that prompt-to-target paths fit within the available depth.

Prior accounts emphasize meta-learning dynamics, induction circuits, or task-specific mechanisms. Our contribution is a general, falsifiable connectivity theory that (a) predicts threshold-like ICL emergence, (b) explains sensitivity to head count and sparsity masks, and (c) yields measurable graph-theoretic diagnostics.

Contributions:
- A threshold-free, weighted percolation criterion for attention graphs based on the spectral radius of a data-averaged non-backtracking/message-passing operator (R > 1).
- A low-stretch routing condition linking typical prompt-to-target distances to model depth L.
- Operational estimators: participation-ratio effective degree, spectral radius via power iteration on batched attention maps, shortest-path distributions in multi-layer directed graphs, and min-cut capacity proxies.
- Preregistered experimental designs to test finite-size scaling, sharpness of ICL thresholds, and robustness to confounders (temperature, parameter count, residual/MLP channels).

## 2. Model: Attention as a Layered, Directed, Weighted Graph
Consider a sequence of n tokens processed by a Transformer with L layers and H heads per layer. For layer ℓ and head h, let aℓ,h(j | i, x) denote the attention weight from source token i at layer ℓ−1 to target token j at layer ℓ for input x, with Σj aℓ,h(j | i, x) = 1. Define the per-layer transport operator as the head union:
- weighted adjacency per head: Aℓ,h(i → j; x) = aℓ,h(j | i, x)
- union-of-heads aggregator: Uℓ(i → j; x) = 1 − Πh (1 − Aℓ,h(i → j; x))

We work with the data-averaged operator Ê[Uℓ], where the expectation is taken over a validation distribution. Residual connections are modeled as self-edges; MLP mixing is treated as within-position transformation (captured in capacity proxies; see §4.3).

To avoid brittle edge-thresholding, we quantify per-node dispersion via the participation ratio (PR):
- PRℓ,h(i) = 1 / Σj aℓ,h(j | i, x)^2, averaged over x
and define an effective degree per layer as k_eff(ℓ) = mean_i,h PRℓ,h(i). This is invariant to monotone reparameterization of attention weights and correlates with the number of “effectively used” recipients per token.

## 3. Supercritical Connectivity: A Spectral Percolation Threshold
Percolation in directed, weighted, layered graphs is governed by a spectral condition. Let Bℓ be the non-backtracking/message-passing operator associated with Ê[Uℓ] (or a reweighted variant for weighted percolation). Define the layer-wise reproduction number Rℓ = ρ(Bℓ), and the across-depth geometric mean R = (Πℓ Rℓ)^(1/L).

Proposition 1 (Branching-process approximation, sketch).
Under a layered, locally tree-like approximation with independent edge realizations sampled from Ê[Uℓ], the probability that there exists a depth-spanning path from a typical prompt token to the target is bounded away from zero as L → ∞ if and only if R > 1. For R ≤ 1, the path-existence probability decays exponentially in L.

Remarks:
- R generalizes the “mean out-degree > 1” rule to directed, weighted, layered graphs and correctly accounts for head unions and degree heterogeneity.
- Finite-size corrections shift the apparent threshold above 1; we propose finite-size scaling tests (§6).

## 4. Low-Stretch Routing and Capacity
### 4.1 Depth-limited routing
Let d(u, v) be the length of the shortest multi-layer path from prompt token u to target v in the directed product graph G = ⨂ℓ Gℓ. Effective ICL requires that typical prompt-target pairs satisfy d(u, v) ≤ L. Architectures with small-world structure (local windows + few random long-range edges) have d(u, v) = O(log n), while strictly local windows produce d(u, v) = Ω(n/w).

### 4.2 Edge-disjoint routes and robustness
By a weighted version of Menger’s theorem, the number of edge-disjoint u→v routes lower bounds min-cut capacity. In supercritical, low-stretch regimes, the expected number of edge-disjoint routes scales with L, providing robustness to noise and interference.

### 4.3 Information-capacity proxy
We define a per-edge capacity proxy cℓ(i → j) = Ê[log(1 + SNRℓ(i → j; x))], with SNR approximated by a function of aℓ,h and value variance; practically, we use entropy/PR-based proxies:
- node capacity: Cℓ(i) = Σj Uℓ(i → j) · w(aℓ,·(j | i)) with w chosen as inverse Herfindahl (PR) weights.
- cut capacity: min over prompt-target cuts of Σ edges cℓ(i → j).
Subcritical regimes (R ≤ 1) exhibit exponentially small cut capacities as L grows; supercritical regimes with low stretch exhibit rapidly increasing capacity.

## 5. Falsifiable Predictions
- P1 (Spectral threshold): Holding parameter count and compute fixed, ICL metrics (e.g., retrieval accuracy, few-shot generalization) show a sharp increase near R ≈ 1. Finite-size scaling collapse is observed when plotting versus (R − 1)L^β for some β > 0.
- P2 (Depth–distance tradeoff): For R > 1, increasing L improves ICL until L ≈ typical d(u, v), beyond which gains saturate.
- P3 (Small-world gain): Adding O(1) random long-range edges per token to local attention reduces d(u, v) from Ω(n/w) to O(log n), enabling ICL at smaller L without changing parameter count.
- P4 (Redundancy robustness): In the supercritical regime, random pruning that preserves R > 1 degrades ICL gradually; pruning that reduces R below 1 causes an abrupt collapse.
- P5 (Temperature control): Increasing attention temperature (or adding dropout on attention logits) lowers PR and R; ICL degradation localizes around the point where R crosses 1.
- P6 (Head-count scaling): Increasing heads raises PR and R sublinearly (due to head overlap); ICL gains track R, not head count per se.

## 6. Operationalization and Estimators
- Participation ratio (PR): Estimate PRℓ,h(i) across a validation batch; report k_eff(ℓ) and its distribution.
- Spectral radius R: Construct Ê[Uℓ] by averaging attention maps over inputs; estimate ρ(Bℓ) via power iteration on the non-backtracking operator; report R = (Πℓ ρ(Bℓ))^(1/L).
- Shortest paths: Build the layered directed graph using Ê[Uℓ]; estimate the distribution of d(u, v) via BFS or Dijkstra variants (weighted distances use −log Ê[Uℓ]).
- Capacity proxy: Estimate min-cut between prompt and target segments with edge weights −log Ê[Uℓ] or PR-weighted capacities; report correlation with ICL metrics.
- Robustness: Repeat with multiple temperatures, τ-thresholds (for sanity), and data subsets; confirm invariance of conclusions to these choices.

## 7. Preregistered Experimental Design
Architectures:
- Random sparse attention (Erdős–Rényi masks per layer/head) with fixed parameter count; sweep sparsity to traverse R ∈ [0.5, 2.0].
- Small-world attention (local window w plus k random long-range edges per token); sweep k to adjust d(u, v) with minimal change to parameter count.
- Learned sparse attention (kNN/top-k routing with Gumbel-top-k) to vary PR while holding FLOPs constant.

Controls:
- Match parameter count, sequence length, batch size, and training compute across conditions.
- Temperature sweeps on attention logits to modulate PR independent of architecture.
- Residual/MLP ablations: scale residual/MLP by α ∈ [0, 1] during evaluation to isolate attention-mediated routing.

Training and tasks:
- Language modeling pretraining with a held-out suite of ICL probes (key–value retrieval, algorithmic copying, linear regression, synthetic pattern completion, small classification tasks). Avoid contaminating probes in training; include domain-shifted probes to test true in-context generalization.

Analyses:
- Change-point detection on ICL accuracy vs R with bootstrapped confidence intervals.
- Finite-size scaling: vary L and n; test collapse when plotting ICL vs (R − 1)L^β and vs L/d(u, v).
- Ablations to test P4–P6; report effect sizes and multiple-comparison-corrected p-values.
- Negative controls: models with comparable perplexity but R ≤ 1 should not show strong ICL.

Reproducibility:
- Release code to compute PR, R, d(u, v), and capacity proxies; publish training/eval scripts and seeds.

## 8. Related Work
Our theory complements mechanistic accounts of ICL (e.g., induction heads and linear-learning circuits), meta-learning views (implicit gradient descent in attention), and sparsity/efficiency work in Transformers (local/small-world/sparse attention). It connects to percolation and epidemic thresholds in directed networks (spectral/non-backtracking criteria), information flow in deep networks (signal propagation, dynamical isometry), and graph-theoretic analyses of neural routing. Unlike prior work, we provide a threshold-free spectral criterion and routing condition tailored to input-dependent, multi-head attention.

## 9. Limitations and Scope
- The data-averaged operator Ê[Uℓ] approximates input-dependent routing; strong context dependence can induce distributional shifts in R and d(u, v).
- Residual and MLP pathways can bypass attention edges; we partially account for this via capacity proxies but do not fully model cross-channel interactions.
- The branching-process approximation assumes local tree-likeness; real attention graphs contain motifs and correlations that shift the apparent threshold.
- Our predictions target routing-heavy ICL tasks; tasks relying primarily on parametric knowledge may not reflect the same thresholds.

## 10. Conclusion
ICL requires both supercritical connectivity (R > 1) and low-stretch routing (typical d(u, v) ≤ L) in the attention-induced communication graph. These conditions yield sharp, falsifiable predictions about when ICL emerges or collapses as sparsity, head count, and depth vary. We provide threshold-free metrics and a preregistered empirical program to test these predictions, aiming to turn qualitative “emergent ability” observations into quantitative, structural design principles for efficient Transformer architectures.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
