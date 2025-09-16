Major Revisions

Brief critique
- Core claim needs formalization. The non-backtracking threshold is stated informally for a data-averaged, layered, directed graph, but the operator is not precisely defined for weighted, multi-head, causal attention. Specify the edge set, the weighted Hashimoto/non-backtracking operator B_l on directed edges, and justify the geometric mean aggregation R = exp((1/L) sum_l log rho(B_l)) versus a product operator across layers.
- Attention orientation and normalization are ambiguous. Standard attention is normalized over sources for each target (sum_i a(j|i)=1). Your definition a_l,h(j|i) as “from i to j” implies outgoing weights are not normalized. Clarify direction, normalization, and how multi-head aggregation maps to an effective edge weight in [0,1]. Provide upper/lower bounds (max-over-heads and capped-sum) and state the independence assumption underlying the “soft-OR” union U_l.
- Averaging across inputs is nontrivial. Distinguish “annealed” (rho of E[U_l(x)]) from “quenched” (Lyapunov exponent of random products) criteria. Report both R_annealed and R_quenched; do not commute E and nonlinearity. Analyze variance of R(x) and its role near the critical region.
- Reachability vs computation needs a principled bridge. Treat R>1 as necessary, not sufficient. Introduce a concrete capacity model: define per-edge capacities c_l(i->j) that incorporate attention weight and value/MLP bandwidth (e.g., spectral norm- or mutual-information proxies), and use min-cut bounds. Specify estimable proxies and their limitations.
- Causality and layering must be explicit. The causal mask yields a DAG over positions; non-backtracking still helps with heterogeneous degree but is not “about cycles” here. State why NB is preferable to adjacency for predicting path proliferation in layered DAGs.
- Predictions and estimators require more rigor. Define how R is estimated from batches (annealed vs quenched), how path-length distributions are computed in a layered, weighted DAG, and how change-point detection and finite-size scaling (choice of exponent) will be performed and validated. Control for confounders (LayerNorm gain, residual scaling, training dynamics near criticality).
- Scope and assumptions need sharpening. Make the independence across heads and the local tree-like assumption explicit, provide bounds when they fail, and discuss how residual/MLP pathways are integrated (or isolated) in the structural model.

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

Directed edges. We define a per-head weighted adjacency for layer l as W_l,h(x) with entries W_l,h[i→j] = a_l,h(j|i, x). This encodes value flow from i to j.

Multi-head aggregation. We require an effective edge weight W_l[i→j] ∈ [0,1]. We adopt a “soft-OR” union under an independence approximation across heads:
- U_l(i→j | x) = 1 − Π_h (1 − W_l,h[i→j]).
This is tractable and models redundant pathways. It is a first-order approximation; we provide bounds that hold without independence:
- Lower bound: U_l ≥ max_h W_l,h.
- Upper bound: U_l ≤ min(1, sum_h W_l,h).
All subsequent spectral criteria can be computed with either U_l or these bounds to assess robustness to head dependence.

Data averaging. ICL capability is a property over a task/data distribution. We define the annealed operator W̄_l = E_x[U_l(·|x)]. Note E[1 − Π(1 − ·)] ≠ 1 − Π(1 − E[·]); we fix the order: aggregate per input, then average. We also analyze quenched quantities that explicitly keep input dependence (Section 6).

Residual/MLP. Residual connections are modeled as self-edges (i→i). Value and MLP transformations modify content on this backbone; we integrate them via capacity proxies (Section 5), not in the reachability threshold.

## 3. Spectral percolation via non-backtracking operators
Rationale. Global connectivity in heterogeneous directed graphs is governed by path proliferation rather than raw degree. Non-backtracking (NB) operators downweight immediate edge reversals and degree-localization, yielding accurate percolation thresholds in sparse, heterogeneous networks and layered DAGs.

Weighted NB operator. For a layer l with effective adjacency W̄_l, define the directed edge set E_l = {(i→j): W̄_l[i→j] > 0}. The weighted non-backtracking matrix B_l ∈ R^{|E_l|×|E_l|} has entries:
- B_l[(i→j),(j→k)] = W̄_l[j→k] if k ≠ i; else 0.
This encodes one-step growth of non-immediately-reversing paths and generalizes to weighted, directed graphs. For causal attention, “backtracking” excludes the trivial 2-hop i→j→i through self/residual edges and mitigates degree inflation.

Layer aggregation. The number/weight of length-L paths across the layered graph scales like the product of per-layer NB growth factors. We define:
- R_l = spectral_radius(B_l).
- R = exp( (1/L) * sum_{l=1..L} log R_l ), the geometric mean reproduction factor.

Proposition (informal, annealed). Under (i) local tree-likeness of W̄_l (sparse, weak short cycles), (ii) bounded second moments of weights, and (iii) stationarity/ergodicity across layers, the expected weight of long-range prompt-to-target paths remains bounded away from zero as L→∞ if R > 1, and decays exponentially if R ≤ 1. Thus, R=1 marks a percolation threshold for communication reachability.

This threshold is necessary for robust ICL capability; it does not guarantee sufficient computational fidelity (Section 5).

## 4. Depth-constrained routing
Path length. Let d(u,v) be the shortest number of layers required to route from prompt token u (early positions) to target token v (later positions) on the layered graph defined by {W̄_l}. ICL requires that a typical prompt-to-target path fits within L:
- Depth-constrained routing: d_char < L,
where d_char is a characteristic path length (e.g., median of d(u,v) over prompts/targets). Local attention yields d_char = Omega(n); small-world augmentations (local windows + sparse long-range edges) yield d_char = O(log n), dramatically reducing required depth.

## 5. Information capacity beyond reachability
R > 1 certifies path existence but not information sufficiency. We complement reachability with a capacity model.

Edge capacities. Let S_l(i→j) denote an estimate of channel strength from token i to j at layer l that incorporates both routing and value transformation. Practical proxies include:
- Attention-weighted value bandwidth: S_l(i→j) = Ū_l(i→j) * ||W_V,l||_op (or a per-head variant).
- Information proxy: S_l(i→j) = Ū_l(i→j) * sqrt(Var[ value_j | mask i-only ]).
- Entropy-based proxy: S_l(i→j) = Ū_l(i→j) * H(attention over i for target j).

Min-cut bound. Define a layered DAG with capacities c_l(i→j) = S_l(i→j). The minimum s–t cut capacity from prompt set to target tokens upper-bounds end-to-end information flow. In the supercritical regime, we predict growth of this min-cut with L (until saturation), while it vanishes as L increases in the subcritical regime.

## 6. Input dependence: annealed vs quenched criteria
Define per-input operators U_l(x) and NB matrices B_l(x). Two complementary thresholds:
- Annealed: R_annealed via W̄_l = E_x[U_l(x)], then R from {B_l(W̄_l)}.
- Quenched: R_quenched via the top Lyapunov exponent of random NB products:
  R_quenched = exp( lim_{L→∞} (1/L) E_x[ log ||B_L(x_L) ... B_1(x_1)|| ] ).
We expect R_annealed ≥ R_quenched in many settings; large Var[R_l(x)] near criticality creates broad transitions. We therefore:
- Report both R_annealed and R_quenched estimates.
- Analyze the distribution of R_l(x) and path lengths d_x(u,v) across inputs, especially near the threshold.

## 7. Falsifiable predictions
- P1 (Spectral threshold): As a control parameter (e.g., sparsity, head count, temperature) is varied, ICL accuracy averaged over tasks shows a sharp increase when R crosses 1. Finite-size scaling collapses performance when plotted against (R − 1) * L^beta for some beta > 0; beta is estimated empirically across n and L.
- P2 (Depth-distance tradeoff): For supercritical models (R > 1), increasing L improves ICL until L ≈ d_char, after which gains saturate.
- P3 (Small-world advantage): Adding k random long-range edges per layer to a local pattern sharply reduces d_char (≈ log n/k), enabling ICL at markedly smaller L (fixed parameters/FLOPs).
- P4 (Pruning phase change): Random edge pruning that maintains R > 1 degrades ICL smoothly; pushing R below 1 causes an abrupt collapse.
- P5 (Temperature): Increasing attention temperature (or sharpening softmax) modulates PR and R; tuning through R ≈ 1 induces a sharp ICL transition.
- P6 (Heads vs overlap): ICL gains track increases in R (sublinear with head count due to overlap), not raw H.
- P7 (Annealed–quenched gap): Near criticality, the annealed threshold predicts earlier apparent emergence than the quenched threshold; input-to-input variance spikes near the transition.

## 8. Estimation and diagnostics
- Participation ratio (PR): For each head, estimate effective out-degree PR_l,h(i) = 1 / sum_j a_l,h(j|i)^2 over validation batches. Summarize its distribution and relate to R.
- Spectral radius:
  - Annealed: Compute W̄_l = E_x[U_l(x)], build B_l(W̄_l), estimate R_l via power iteration on B_l; set R = exp(mean_l log R_l).
  - Quenched: Sample batches of B_l(x), run power iteration on random products to estimate the Lyapunov exponent (with bootstrapped CIs).
  - Robustness: Repeat with max-over-heads and capped-sum aggregators to bound the effect of head dependence.
- Shortest paths: Build the layered DAG from {W̄_l}; compute d(u,v) distributions (unweighted or with costs proportional to 1/Ū_l). Summarize d_char.
- Capacity proxies and min-cut: Define c_l(i→j); compute s–t min-cut capacities between prompt and target layers; correlate with ICL accuracy.
- Change-point/scaling: Use validated change-point detectors on accuracy vs control parameter; fit finite-size scaling exponents via cross-validated regression; test collapse across n and L.

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
Our framework connects mechanistic accounts of ICL (e.g., induction circuits) with network-science percolation using spectral/non-backtracking criteria, and with small-world theories of efficient routing. It complements work on sparse Transformers (efficiency) by providing predictive structural thresholds, and relates to information-flow analyses in deep networks (e.g., dynamical isometry, min-cut bounds).

## 11. Limitations and scope
- Annealed vs quenched: We predict capability over distributions, not per-instance behavior; averaging can misestimate thresholds when input variance is high.
- Necessary vs sufficient: R > 1 ensures reachability; computation requires sufficient capacity and low interference. Our capacity proxies are pragmatic and imperfect.
- Assumptions: Independence across heads and local tree-likeness are approximations. We provide bounds and robustness checks but do not prove tightness for real attention graphs.
- Estimation error: Spectral estimates on large NB matrices and random products require careful numerical control; finite-size and training nonstationarity can blur thresholds.

## 12. Conclusion
We advance a quantitative, falsifiable theory of ICL as directed percolation with depth-constrained routing. A supercritical NB spectral threshold (R > 1) is necessary for long-range communication, and characteristic path length must fit within model depth. Coupled with capacity diagnostics, these criteria yield sharp, testable predictions about ICL emergence and its dependence on architecture. This framework aims to shift emergent-ability studies from descriptive to predictive, enabling principled design of sparse and small-world attention mechanisms.
