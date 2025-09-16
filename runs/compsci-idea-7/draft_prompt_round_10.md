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
        Minor Revisions

Brief critique
- The revised draft effectively addresses most prior concerns by formalizing the non-backtracking operator, clarifying attention normalization and multi-head aggregation with bounds, distinguishing annealed vs. quenched criteria, and incorporating capacity models with min-cut bounds. It provides rigorous estimators, falsifiable predictions, and a preregistered plan, enhancing novelty and testability for a leading ML journal like NeurIPS or JMLR. However, minor issues remain: (i) the proposition in Section 3 is still informal—add a formal statement with references to supporting theorems in network science; (ii) Section 5's capacity proxies lack empirical justification or citations for their validity (e.g., link to mutual information bounds in deep nets); (iii) predictions in Section 7 could specify quantitative thresholds or effect sizes for falsifiability; (iv) add a brief discussion of computational cost for estimators (e.g., power iteration on large NB matrices). These are polish items, not core flaws.

Revised Draft
# A Directed-Percolation Theory of In-Context Learning: Spectral Thresholds and Depth-Constrained Routing in Transformers

## Abstract
We propose a structural theory of in-context learning (ICL) in Transformers, framing it as directed percolation on the attention-induced communication graph. The theory predicts that ICL capability emerges when two conditions hold over a task/data distribution: (i) a supercritical path-proliferation condition, R > 1, defined via the spectral radius of a weighted non-backtracking operator on the (data-averaged) multi-head attention graph, and (ii) a depth-constrained routing condition requiring that prompt-to-target information paths fit within the model depth L. We operationalize these conditions with estimators for spectral radius, path-length distributions, and min-cut capacity. We present falsifiable predictions (including finite-size scaling) and a preregistered experimental plan. The framework unifies observations of abrupt ICL emergence, head-count effects, sparsity, and small-world attention into a single, quantitative model.

## 1. Introduction
Transformers often acquire the ability to solve new tasks specified in-context, without parameter updates. We lack quantitative criteria predicting when this capability emerges as a function of depth, head count, sparsity, and training dynamics.

We model attention as an input-dependent, layered, directed, weighted graph and hypothesize two structural constraints governing ICL capability (over a data distribution):
- Supercritical connectivity: long-range path proliferation quantified by a spectral threshold R > 1.
- Depth-constrained routing: typical prompt-to-target path length must be less than L.

Our goal is a falsifiable theory with measurable diagnostics that predict the onset of ICL and guide architectural design, especially for sparse/small-world attention.

## 2. Communication graph construction
Setup. Consider a decoder-only Transformer with sequence length n, depth L, and H heads per layer. For input x, the attention weight from source token i at layer l−1 to target token j at layer l and head h is a_l,h(j|i, x) ∈ [0,1]. In standard attention, for each target j, weights are normalized over sources i: sum_i a_l,h(j|i, x) = 1. Causal masking restricts i ≤ j (or a relative variant), yielding a layered DAG over positions.

Directed edges. We define a per-head weighted adjacency for layer l as W_l,h(x) with entries W_l,h[i→j] = a_l,h(j|i, x). This encodes value flow from i to j. Note that this orientation implies outgoing weights from i are not necessarily normalized (sum_j W_l,h[i→j] may ≠ 1), contrasting with standard target-normalized attention; we justify this as modeling information outflow.

Multi-head aggregation. We require an effective edge weight W_l[i→j] ∈ [0,1]. We adopt a “soft-OR” union under an independence approximation across heads:
- U_l(i→j | x) = 1 − Π_h (1 − W_l,h[i→j]).
This is tractable and models redundant pathways. It is a first-order approximation; we provide bounds that hold without independence:
- Lower bound: U_l ≥ max_h W_l,h.
- Upper bound: U_l ≤ min(1, sum_h W_l,h).
All subsequent spectral criteria can be computed with either U_l or these bounds to assess robustness to head dependence.

Data averaging. ICL capability is a property over a task/data distribution. We define the annealed operator W̄_l = E_x[U_l(·|x)]. Note E[1 − Π(1 − ·)] ≠ 1 − Π(1 − E[·]); we fix the order: aggregate per input, then average. We also analyze quenched quantities that explicitly keep input dependence (Section 6).

Residual/MLP. Residual connections are modeled as self-edges (i→i). Value and MLP transformations modify content on this backbone; we integrate them via capacity proxies (Section 5), not in the reachability threshold.

## 3. Spectral percolation via non-backtracking operators
Rationale. Global connectivity in heterogeneous directed graphs is governed by path proliferation rather than raw degree. Non-backtracking (NB) operators downweight immediate edge reversals and degree-localization, yielding accurate percolation thresholds in sparse, heterogeneous networks and layered DAGs. In causal DAGs, NB is preferable to adjacency as it mitigates heterogeneous degree effects on path growth, even without cycles.

Weighted NB operator. For a layer l with effective adjacency W̄_l, define the directed edge set E_l = {(i→j): W̄_l[i→j] > 0}. The weighted non-backtracking matrix B_l ∈ R^{|E_l|×|E_l|} has entries:
- B_l[(i→j),(j→k)] = W̄_l[j→k] if k ≠ i; else 0.
This encodes one-step growth of non-immediately-reversing paths and generalizes to weighted, directed graphs. For causal attention, “backtracking” excludes the trivial 2-hop i→j→i through self/residual edges and mitigates degree inflation.

Layer aggregation. The number/weight of length-L paths across the layered graph scales like the product of per-layer NB growth factors. We define:
- R_l = spectral_radius(B_l).
- R = exp( (1/L) * sum_{l=1..L} log R_l ), the geometric mean reproduction factor. (We justify geometric mean over product for its stability in heterogeneous layers and alignment with Lyapunov-like exponents.)

Proposition 1 (Formal, annealed). Under (i) local tree-likeness of W̄_l (sparse, weak short cycles), (ii) bounded second moments of weights, and (iii) stationarity/ergodicity across layers, the expected weight of long-range prompt-to-target paths remains bounded away from zero as L→∞ if R > 1, and decays exponentially if R ≤ 1 (cf. Krzakala et al., 2007; Bordenave et al., 2015 for NB thresholds in random graphs). Thus, R=1 marks a percolation threshold for communication reachability.

This threshold is necessary for robust ICL capability; it does not guarantee sufficient computational fidelity (Section 5).

## 4. Depth-constrained routing
Path length. Let d(u,v) be the shortest number of layers required to route from prompt token u (early positions) to target token v (later positions) on the layered graph defined by {W̄_l}. ICL requires that a typical prompt-to-target path fits within L:
- Depth-constrained routing: d_char < L,
where d_char is a characteristic path length (e.g., median of d(u,v) over prompts/targets). Local attention yields d_char = Ω(n); small-world augmentations (local windows + sparse long-range edges) yield d_char = O(log n), dramatically reducing required depth.

## 5. Information capacity beyond reachability
R > 1 certifies path existence but not information sufficiency. We complement reachability with a capacity model.

Edge capacities. Let S_l(i→j) denote an estimate of channel strength from token i to j at layer l that incorporates both routing and value transformation. Practical proxies include:
- Attention-weighted value bandwidth: S_l(i→j) = Ū_l(i→j) * ||W_V,l||_op (or a per-head variant), approximating operator-norm bounds on signal propagation (e.g., as in dynamical isometry literature).
- Information proxy: S_l(i→j) = Ū_l(i→j) * sqrt(Var[ value_j | mask i-only ]), motivated by mutual information lower bounds (e.g., Tishby et al., 2000).
- Entropy-based proxy: S_l(i→j) = Ū_l(i→j) * H(attention over i for target j), capturing uncertainty reduction.

Min-cut bound. Define a layered DAG with capacities c_l(i→j) = S_l(i→j). The minimum s–t cut capacity from prompt set to target tokens upper-bounds end-to-end information flow. In the supercritical regime, we predict growth of this min-cut with L (until saturation), while it vanishes as L increases in the subcritical regime.

## 6. Input dependence: annealed vs quenched criteria
Define per-input operators U_l(x) and NB matrices B_l(x). Two complementary thresholds:
- Annealed: R_annealed via W̄_l = E_x[U_l(x)], then R from {B_l(W̄_l)}.
- Quenched: R_quenched via the top Lyapunov exponent of random NB products:
  R_quenched = exp( lim_{L→∞} (1/L) E_x[ log ||B_L(x_L) ... B_1(x_1)|| ] ).
We expect R_annealed ≥ R_quenched in many settings; large Var[R_l(x)] near criticality creates broad transitions. We therefore:
- Report both R_annealed and R_quenched estimates.
- Analyze the distribution of R_l(x) and path lengths d_x(u,v) across inputs, especially near the threshold. Do not commute E and nonlinearity; quenched handles input variance directly.

## 7. Falsifiable predictions
- P1 (Spectral threshold): As a control parameter (e.g., sparsity, head count, temperature) is varied, ICL accuracy averaged over tasks shows a sharp increase when R crosses 1 (e.g., >20% accuracy jump). Finite-size scaling collapses performance when plotted against (R − 1) * L^beta for some beta > 0; beta is estimated empirically across n and L.
- P2 (Depth-distance tradeoff): For supercritical models (R > 1), increasing L improves ICL until L ≈ d_char, after which gains saturate (plateau within 5% accuracy).
- P3 (Small-world advantage): Adding k random long-range edges per layer to a local pattern sharply reduces d_char (≈ log n/k), enabling ICL at markedly smaller L (fixed parameters/FLOPs), with accuracy gains >10% for k>1.
- P4 (Pruning phase change): Random edge pruning that maintains R > 1 degrades ICL smoothly; pushing R below 1 causes an abrupt collapse (>50% accuracy drop).
- P5 (Temperature): Increasing attention temperature (or sharpening softmax) modulates PR and R; tuning through R ≈ 1 induces a sharp ICL transition (change-point detectable with p<0.01).
- P6 (Heads vs overlap): ICL gains track increases in R (sublinear with head count due to overlap), not raw H (e.g., diminishing returns after H=4).
- P7 (Annealed–quenched gap): Near criticality, the annealed threshold predicts earlier apparent emergence than the quenched threshold; input-to-input variance spikes near the transition (Var[R_l(x)] >1).

## 8. Estimation and diagnostics
- Participation ratio (PR): For each head, estimate effective out-degree PR_l,h(i) = 1 / sum_j a_l,h(j|i)^2 over validation batches. Summarize its distribution and relate to R.
- Spectral radius:
  - Annealed: Compute W̄_l = E_x[U_l(x)], build B_l(W̄_l), estimate R_l via power iteration on B_l (converges in O(|E_l| log |E_l|) time); set R = exp(mean_l log R_l).
  - Quenched: Sample batches of B_l(x), run power iteration on random products to estimate the Lyapunov exponent (with bootstrapped CIs); computational cost O(L * |E_l|^2) per sample, mitigated by subsampling.
  - Robustness: Repeat with max-over-heads and capped-sum aggregators to bound the effect of head dependence.
- Shortest paths: Build the layered DAG from {W̄_l}; compute d(u,v) distributions (unweighted or with costs proportional to 1/Ū_l) using BFS (O(L n^2)). Summarize d_char.
- Capacity proxies and min-cut: Define c_l(i→j); compute s–t min-cut capacities between prompt and target layers using max-flow algorithms (O(L n^3)); correlate with ICL accuracy.
- Change-point/scaling: Use validated change-point detectors (e.g., PELT) on accuracy vs control parameter; fit finite-size scaling exponents via cross-validated regression; test collapse across n and L. Control for confounders like LayerNorm gain and residual scaling.

## 9. Preregistered experimental design
Architectures.
- Random sparse attention (Erdos–Renyi masks) sweeping sparsity to modulate R.
- Small-world attention (local windows + k random long-range edges per layer) sweeping k to modulate d_char.
- Head-count sweeps with fixed per-head dimension; temperature sweeps at inference to modulate R without retraining.

Controls.
- Fix total parameters, training FLOPs, sequence length, and dataset across conditions.
- Standardize optimization (optimizer, LR schedule), LayerNorm and residual scales.
- At evaluation, isolate pathways via multiplicative gates on attention/residual/MLP to quantify backbone vs content effects.

Tasks.
- ICL probes: key–value retrieval, copy/induction, algorithmic tasks, few-shot classification across heterogeneous input distributions.

Analyses.
- Estimate R_annealed, R_quenched, d_char, and min-cut per trained model.
- Change-point detection for ICL onset; finite-size scaling across {n, L}.
- Variance analysis of R_l(x) near criticality; ablate heads/edges to test P4–P6.

## 10. Related work
Our framework connects mechanistic accounts of ICL (e.g., induction circuits, Olsson et al., 2022) with network-science percolation using spectral/non-backtracking criteria (Newman, 2018), and with small-world theories of efficient routing (Watts & Strogatz, 1998). It complements work on sparse Transformers (efficiency, Child et al., 2019) by providing predictive structural thresholds, and relates to information-flow analyses in deep networks (e.g., dynamical isometry, Pennington et al., 2017; min-cut bounds, Raghu et al., 2017).

## 11. Limitations and scope
- Annealed vs quenched: We predict capability over distributions, not per-instance behavior; averaging can misestimate thresholds when input variance is high.
- Necessary vs sufficient: R > 1 ensures reachability; computation requires sufficient capacity and low interference. Our capacity proxies are pragmatic and imperfect.
- Assumptions: Independence across heads and local tree-likeness are approximations. We provide bounds and robustness checks but do not prove tightness for real attention graphs. Discuss failures: head dependence reduces U_l below soft-OR; cycles (if any) inflate R.
- Estimation error: Spectral estimates on large NB matrices and random products require careful numerical control (e.g., sparse implementations); finite-size and training nonstationarity can blur thresholds.

## 12. Conclusion
We advance a quantitative, falsifiable theory of ICL as directed percolation with depth-constrained routing. A supercritical NB spectral threshold (R > 1) is necessary for long-range communication, and characteristic path length must fit within model depth. Coupled with capacity diagnostics, these criteria yield sharp, testable predictions about ICL emergence and its dependence on architecture. This framework aims to shift emergent-ability studies from descriptive to predictive, enabling principled design of sparse and small-world attention mechanisms.


        Output sections:
        1) Judgment line
        2) Brief critique
        3) Revised draft (Markdown) — start this section with a single line heading exactly: "Revised Draft". OMIT this section entirely if your judgment is Publish.
