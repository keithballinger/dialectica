# A Directed-Percolation Theory of In-Context Learning: Spectral Thresholds and Depth-Constrained Routing in Transformers

## Abstract
We propose a structural theory of in-context learning (ICL) in Transformers that treats information propagation as directed percolation on the attention-induced communication graph. The theory posits two necessary conditions for robust ICL over a task distribution: (i) a supercritical path-proliferation condition R > 1, defined via a backtracking-suppressed line-graph operator on the (data-averaged) multi-head attention graph, and (ii) a depth-constrained routing condition requiring that prompt-to-target paths fit within model depth L. We operationalize these criteria with scalable estimators for spectral growth, path-length distributions, and min-cut capacity using Jacobian-based edge capacities. We present falsifiable predictions (including finite-size scaling) and a preregistered plan. The framework unifies observations of abrupt ICL emergence, head-count effects, sparsity, and small-world attention under a single quantitative model.

## 1. Introduction
Transformers often acquire ICL: solving tasks specified in the prompt without parameter updates. We lack predictive criteria that specify when ICL emerges as a function of attention sparsity, head count, and depth.

We model attention as an input-dependent, layered, directed, weighted graph. Two structural constraints govern ICL capability over a data distribution:
- Supercritical connectivity (R > 1): long-range proliferation of non-degenerate information paths.
- Depth-constrained routing: prompt-to-target paths fit within L layers.

We derive estimators and falsifiable predictions to assess these constraints and guide design of sparse/small-world attention.

## 2. Communication graph and aggregation
Setup. Consider a decoder-only Transformer with sequence length n, depth L, and H heads per layer. For input x, attention from source position i at layer l−1 to target position j at layer l and head h is a_l,h(j|i,x) ∈ [0,1], normalized per target j. Causal masking enforces i ≤ j.

Edges. Define per-head weighted adjacency W_l,h(x) with entries W_l,h[i→j] = a_l,h(j|i,x). Residual connections are modeled as self-edges i→i.

Aggregating heads. We require an effective per-layer weight W_l. We aggregate per input x using:
- U_l(i→j|x) = 1 − Π_h (1 − W_l,h[i→j]) (soft-OR).
Bounds independent of head independence:
- max_h W_l,h[i→j] ≤ U_l(i→j|x) ≤ min(1, Σ_h W_l,h[i→j]).
We carry both the soft-OR value and these bracketing bounds through diagnostics to assess sensitivity to head dependence.

Data averaging. Define the annealed operator W̄_l = E_x[U_l(·|x)] (aggregate within input, then average). We also analyze quenched quantities that keep input dependence (Section 6).

## 3. Backtracking-suppressed spectral growth in causal, layered graphs
Why suppress backtracking? Spectral thresholds for percolation in heterogeneous graphs are governed by growth of non-degenerate walks (walks that do not immediately reverse). In causal, layered DAGs with i ≤ j, immediate reversals do not exist except for stacking self-loops (i→i→i). Suppressing those trivial walks avoids spuriously inflating growth via residual accumulation.

Backtracking-suppressed line-graph operator. For layer l with effective adjacency W̄_l, let E_l = {(i→j): W̄_l[i→j] > 0}. Define L_l ∈ R^{|E_l|×|E_l|} by
- L_l[(i→j),(j→k)] = W̄_l[j→k] if k ≠ i; 0 otherwise.
With causal masking, k < j is disallowed, so k = i can only occur when i = j = k; thus L_l removes stacked self-loop transitions (i→i)→(i→i). For general directed graphs (e.g., with bidirectional edges), L_l reduces to the weighted Hashimoto non-backtracking operator.

Layer aggregation and reproduction factor. Let R_l = ρ(L_l) be the spectral radius. Define the geometric-mean reproduction factor
- R = exp((1/L) Σ_{l=1}^L log R_l),
which matches the top Lyapunov exponent for i.i.d. (or stationary ergodic) layers and stabilizes heterogeneous products.

Assumptions. We consider a random layered graph model induced by data averaging with:
- A1 (Sparsity/local tree-likeness across layers): bounded average out-degree, vanishing short non-self cycles in the large-n limit (within layers).
- A2 (Bounded weight moments): E[W̄_l[i→j]^2] < ∞.
- A3 (Stationarity/ergodicity): {W̄_l} is stationary across layers (or admits an ergodic decomposition).

Proposition 1 (Annealed threshold; formal statement). Under A1–A3, there exists a critical threshold R⋆ = 1 such that:
- If R ≤ 1, the expected total weight of non-degenerate prompt-to-target walks of length L decays at least exponentially with L.
- If R > 1, the expected weight remains bounded away from zero as L increases (and grows for sufficiently large L), yielding a positive probability of long-range communication via non-degenerate paths.
Moreover, for strictly causal layered DAGs the operator L_l equals the non-backtracking operator restricted to forward edges and with self-loop stacking suppressed.

Proof sketch. Map edge-weighted paths to walks on the line graph of the layered DAG; immediate backtracking corresponds to (i→j) followed by (j→i). Under causality, only stacked self-loops remain; removing them yields L_l. Standard results for the growth of non-backtracking walks and percolation thresholds in sparse random graphs (e.g., Hashimoto; Krzakala et al., 2013; Bordenave, Lelarge, and Massoulié, 2015) extend to weighted settings with bounded moments and local tree-likeness by replacing counts with weighted branching factors. Stationarity across layers yields a multiplicative ergodic theorem argument giving the geometric mean criterion R > 1 for supercritical growth.

Remark. In strictly causal settings without residual self-loops, L_l and adjacency have the same spectral threshold; suppressing self-loop stacking avoids trivial inflation from residual accumulation.

## 4. Depth-constrained routing
Define d(u,v) as the minimal number of layers needed to route from prompt token u to target v on the layered graph defined by {W̄_l}, with edge cost 1 (or more generally cost proportional to −log W̄_l). Define a characteristic path length d_char (e.g., median over prompt-target pairs).

Depth-constrained routing condition:
- d_char < L.
Local attention implies d_char = Ω(n). Adding sparse long-range edges per layer yields small-world behavior with d_char = O(log n) under standard conditions (Watts–Strogatz-type augmentations), allowing ICL at significantly smaller L.

## 5. Information capacity via Jacobian-based edge capacities
Reachability (R > 1) is necessary but not sufficient. We define edge capacities using pathwise sensitivity to quantify usable information flow.

Edge capacities from Jacobians. Let h_{l,j} be the post-attention hidden state at layer l, position j. Define
- c_l(i→j) = E_x[|∂h_{l,j}(x)/∂h_{l-1,i}(x)|],
estimated via directional derivatives and Hutchinson’s trace estimators. This captures the effective channel strength, incorporating attention and value mixing. Alternatives include squared sensitivities or task-conditioned Jacobians (e.g., ∂ℓ/∂h).

Layered min-cut bound. Form a layered DAG with capacities c_l(i→j). For source set S (prompt tokens) and sink set T (targets), the maximum s–t flow upper-bounds the total Jacobian-mediated information transfer from S to T (by submultiplicativity/data processing). We predict:
- In the supercritical regime, s–t min-cut scales up with L until saturation; in subcritical regimes, it vanishes with L.

Empirical practicality. Compute c_l via mini-batch estimates with low-overhead Jacobian-vector products; run max-flow (Dinic or push-relabel) per model on sparsified graphs.

## 6. Annealed versus quenched thresholds
Define per-input operators U_l(x), L_l(x). Two thresholds:
- Annealed: build W̄_l = E_x[U_l(x)], then L̄_l = L_l(W̄_l), and R_annealed from {L̄_l}.
- Quenched: top Lyapunov exponent of random products:
  R_quenched = exp(lim_{L→∞} (1/L) E_x[log ||L_L(x_L)···L_1(x_1)||]).
Typically R_annealed ≥ R_quenched; increased input variance near criticality widens the annealed–quenched gap. We estimate both, report their difference, and analyze the distribution of per-layer growth factors across inputs.

## 7. Falsifiable predictions
- P1 (Spectral threshold): As a control parameter (e.g., sparsity, head count, attention temperature) varies, ICL accuracy exhibits a change-point at R ≈ 1. A logistic model for accuracy vs. R yields a threshold parameter near 1 with tight CIs; finite-size scaling collapses performance when plotted against (R − 1)L^β for some β > 0.
- P2 (Depth-distance tradeoff): For supercritical models (R > 1), increasing L improves ICL until L ≈ d_char, after which gains saturate (plateau detected by a slope change indistinguishable from zero within CI).
- P3 (Small-world benefit): Adding k random long-range edges per layer (fixed FLOPs) reduces d_char ≈ O(log n/k), shifting the ICL threshold to smaller L and larger accuracy at fixed L. The shift magnitude scales with measured change in d_char.
- P4 (Pruning phase change): Random edge pruning that keeps R > 1 degrades ICL smoothly; dropping R below 1 produces a sharp collapse identified by change-point detection.
- P5 (Temperature control): Modulating attention temperature shifts R across 1 and induces a reproducible ICL transition without retraining.
- P6 (Heads vs. overlap): Gains from more heads track increases in R (diminishing returns with overlapping edges); bracketing aggregators (max vs. capped-sum) bound the effect size.
- P7 (Annealed–quenched gap): Near criticality, annealed thresholds anticipate emergence relative to quenched; input-to-input variance of per-layer growth spikes near the transition.

## 8. Estimation and computational costs
- Participation ratio (PR): For each head h, PR_l,h(i) = 1 / Σ_j a_l,h(j|i)^2; relate distributional summaries to R.
- Spectral growth:
  - Annealed: build W̄_l and L̄_l; estimate R_l via power iteration. Cost O(k|E_l|) for k iterations with sparse edge-level multiplication.
  - Quenched: sample sequences {L_l(x)} and estimate the Lyapunov exponent via products with periodic QR reorthogonalization; cost O(kL|E|) per sequence; reduce variance via bootstrapping and subgraph sampling.
  - Scalability: use sparsification, graph coarsening, or Nyström/sketching for approximate spectra with error bounds.
- Shortest paths: Compute d(u,v) on the layered DAG using multi-source BFS/Dijkstra. Cost O(|E| + |V|) (unweighted) or O(|E| log |V|) (weighted).
- Capacities and min-cut: Estimate c_l(i→j) via Jacobian-vector products; run max-flow using Dinic (O(|E|√|V|)) or push-relabel; exploit layer structure for faster blocking flows.
- Robustness to head aggregation: Repeat all diagnostics with soft-OR, max-over-heads, and capped-sum to bound uncertainty from head dependence.

## 9. Preregistered experimental design
Architectures.
- Random sparse attention (Erdős–Rényi masks) sweeping sparsity.
- Small-world attention: local windows + k random long-range edges per layer.
- Head count sweeps with fixed per-head dimension; inference-time temperature sweeps.

Controls.
- Fix total parameters, training FLOPs, sequence length, datasets, and optimization hyperparameters across conditions.
- Standardize LayerNorm/residual scales; log activations to detect confounds.

Tasks.
- ICL probes: key–value retrieval, copy/induction, algorithmic tasks (addition, parentheses), and few-shot classification across heterogeneous distributions.

Analyses.
- Estimate R_annealed, R_quenched, d_char, min-cut per model; report CIs.
- Change-point detection (e.g., PELT) for ICL onset; finite-size scaling across {n, L}.
- Variance of per-layer growth across inputs; controlled ablations to test P3–P6.

## 10. Related work
- Non-backtracking spectra and percolation thresholds: Hashimoto; Krzakala et al., 2013; Bordenave, Lelarge, Massoulié, 2015; Newman, 2018.
- Small-world routing and diameter: Watts & Strogatz, 1998; per-layer small-world constructions.
- Sparse Transformers and efficiency: Child et al., 2019; routing in sparse attention.
- Attention flow and Jacobians: Abnar & Zuidema, 2020; gradient-based attribution and path sensitivity; dynamical isometry and signal propagation (Pennington et al., 2017).
- Mechanistic ICL: induction circuits and attention patterns (e.g., Olsson et al., 2022).

## 11. Limitations
- Applicability of spectral thresholds depends on sparsity and local tree-likeness; dense attention may require alternative random-matrix analyses.
- R > 1 is necessary but not sufficient; interference and value transformations can block computation despite reachability. Jacobian capacities mitigate but do not eliminate this gap.
- Head independence is an approximation; bracketing aggregators bound but do not remove dependence.
- Estimation error for large models requires careful numerical control; finite-size and training nonstationarity can blur transitions.

## 12. Conclusion
ICL emergence can be predicted by two structural conditions: supercritical backtracking-suppressed spectral growth (R > 1) and depth-constrained routing (d_char < L). Coupled with Jacobian-based capacity diagnostics, these criteria yield sharp, testable predictions about when ICL appears and how it depends on architecture. This framework aims to shift emergent-ability studies from descriptive to predictive, informing principled design of sparse and small-world attention.
